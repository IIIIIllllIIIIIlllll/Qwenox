// 33_cvec.inc — 控制向量（refusal projection）：加载 llama.cpp 控制向量 GGUF，
// 在每层 HC 残差上做投影消融 h -= s·(h·v)·v（project）或加法引导 h += s·v（add）。
//
// 语义对齐 Cudila/llama.cpp 的 cvec-hooks 分支（HF:
// Cudecnik/Qwen3.8-Flash-Next-refusal-projection）：
//   * 张量 direction.K 用于第 K 层（0 基）的输出残差（参考实现里 layer il 取
//     direction.il），作用区间是闭区间 [lo, hi]，默认 4..44；
//   * project 模式加载时把向量单位化，s = scale × |原始向量|；add 模式 s = scale、
//     向量保持原样（数学上等价于参考的 h += strength·raw）；
//   * 作用点是每层 mlp HC 写入之后（4 条 HC 分支各自处理，逐 token），即参考实现
//     build_hc_combine 之后调 build_cvec 的位置；
//   * 只作用于 trunk；MTP / DFlash2 草稿头不另加向量（参考：draft head unsteered）。
//
// 配置（启动期固定，见 docs/CVEC.md）：
//   QWENOX_CVEC=<file.gguf>     空/未设 = 关闭
//   QWENOX_CVEC_MODE=project|add（默认 project）
//   QWENOX_CVEC_SCALE=<float>   （默认 1.0，必须 >0 且有限）
//   QWENOX_CVEC_LAYERS="4 44"   （闭区间，1<=lo<=hi<=47；空格或逗号分隔）
//   QWENOX_CVEC_DIR=per-layer|single:<L>（默认 per-layer；只走环境变量）
//   QWENOX_CVEC_PROBE=<layer>   （诊断：打印该层投影前后 cos；会禁用 graph）
//
// 关闭时零开销：不分配设备缓冲、不发射 kernel、不改变快照指纹。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// splitmix64 finalizer（51_host_cfg.inc 的 mix64 在 40_model.inc 之后才定义）
static inline uint64_t cvec_mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// 残差 [row][branch*n + c]：一个 block 处理 (row, branch)，dot 走两级归约，
// 同步后统一回写。PROJECT 时 out = x - s·dot·v，否则 out = x + s·v。
template <bool BF16, bool PROJECT>
__global__ void k_cvec_apply(void* __restrict__ resid, const float* __restrict__ v,
                             float s, int row_stride, int n) {
  const int row = blockIdx.x;
  const int br = blockIdx.y;
  const int tid = threadIdx.x;
  const size_t base = (size_t)row * row_stride + (size_t)br * n;
  uint16_t* hbase = (uint16_t*)resid + base;
  float* fbase = (float*)resid + base;
  auto load1 = [&](int c) { return BF16 ? bf2f(hbase[c]) : fbase[c]; };
  float dot = 0.f;
  for (int c = tid; c < n; c += blockDim.x) dot += load1(c) * v[c];
#pragma unroll
  for (int o = 16; o; o >>= 1) dot += __shfl_xor(dot, o, 32);
  __shared__ float red[32];
  if ((tid & 31) == 0) red[tid >> 5] = dot;
  __syncthreads();
  if (tid < 32) {
    float x = tid < (blockDim.x >> 5) ? red[tid] : 0.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) x += __shfl_xor(x, o, 32);
    if (tid == 0) red[0] = x;
  }
  __syncthreads();
  const float b = PROJECT ? -s * red[0] : s;
  for (int c = tid; c < n; c += blockDim.x) {
    const float y = load1(c) + b * v[c];
    if (BF16) hbase[c] = f2bf(y); else fbase[c] = y;
  }
}

// 控制向量状态 + 加载器 + 应用入口。g_str（当前流）由调用方传入。
struct Cvec {
  bool on_ = false;
  int mode_ = 1;        // 0 add, 1 project
  float scale_ = 1.0f;  // 用户 scale（project 时乘原始范数后落 per-layer s）
  int lo_ = 4, hi_ = 44;
  int single_ = -1;     // >=0：整个区间都用 direction.<single_>
  std::string path_;
  std::vector<float> host_;   // [n_layer][n_embd]：实际施加的向量（project 为单位向量）
  std::vector<float> host_u_; // [n_layer][n_embd]：单位向量（probe 的 cos 用）
  std::vector<float> host_s_; // [n_layer]
  float* d_v_ = nullptr;      // [n_layer][n_embd]
  float* d_s_ = nullptr;      // [n_layer]
  int probe_ = -1;            // QWENOX_CVEC_PROBE 层号
  int probe_shown_ = 0;       // 已报告的批次数（最多 2）
  float* d_probe_ = nullptr;  // [2][branches][n_embd] 前/后残差（f32）
  float probe_cos_[2][4] = {};  // [pre/post][branch]
  int probe_n_ = 0;

  bool on() const { return on_; }
  bool project() const { return mode_ == 1; }
  bool in_range(int l) const { return on_ && l >= lo_ && l <= hi_; }
  const float* dir(int l) const { return d_v_ + (size_t)l * g_cfg.d; }
  float s(int l) const { return d_s_[l]; }
  bool probe_on() const { return probe_ >= 0; }

  static const char* layer_suffix() { return ".mlp_hyper_connection"; }

  // "layers.<l>.mlp_hyper_connection" → l，其余（含 mtp.*）→ -1。
  static int mlp_prefix_layer(const std::string& p) {
    const char* suf = layer_suffix();
    const size_t sl = strlen(suf);
    if (p.size() <= sl || p.compare(p.size() - sl, sl, suf) != 0) return -1;
    if (p.compare(0, 7, "layers.") != 0) return -1;
    const char* c = p.c_str() + 7;
    if (*c < '0' || *c > '9') return -1;
    int l = 0;
    while (*c >= '0' && *c <= '9') { l = l * 10 + (*c - '0'); c++; }
    return *c == '.' && c == p.c_str() + p.size() - sl ? l : -1;
  }

  // 环境变量解析与参数校验（不碰设备）。precheck() 在加载权重前调用一次，
  // 避免配置写错要等 60 GiB 权重加载完才报错。
  void parse_env() {
    const char* p = getenv("QWENOX_CVEC");
    if (!p || !*p) return;
    path_ = p;
    if (const char* m = getenv("QWENOX_CVEC_MODE"); m && *m) {
      if (!strcmp(m, "project")) mode_ = 1;
      else if (!strcmp(m, "add")) mode_ = 0;
      else throw std::runtime_error(
          std::string("QWENOX_CVEC_MODE must be project or add (got \"") + m + "\")");
    }
    if (const char* sc = getenv("QWENOX_CVEC_SCALE"); sc && *sc) {
      char* end = nullptr;
      scale_ = strtof(sc, &end);
      if (*end || !(scale_ > 0.f) || !std::isfinite(scale_))
        throw std::runtime_error(
            std::string("QWENOX_CVEC_SCALE must be a finite number > 0 (got \"") + sc + "\")");
    }
    if (const char* ly = getenv("QWENOX_CVEC_LAYERS"); ly && *ly) {
      int lo = -1, hi = -1;
      if (sscanf(ly, "%d %d", &lo, &hi) != 2 && sscanf(ly, "%d,%d", &lo, &hi) != 2)
        throw std::runtime_error(
            std::string("QWENOX_CVEC_LAYERS must be \"<lo> <hi>\" (got \"") + ly + "\")");
      lo_ = lo;
      hi_ = hi;
    }
    if (const char* dr = getenv("QWENOX_CVEC_DIR"); dr && *dr) {
      if (strcmp(dr, "per-layer") != 0) {
        int l = -1;
        if (sscanf(dr, "single:%d", &l) != 1)
          throw std::runtime_error(std::string(
              "QWENOX_CVEC_DIR must be per-layer or single:<layer> (got \"") + dr + "\")");
        single_ = l;
      }
    }
    const int n_layer = g_cfg.layers;
    if (lo_ < 1 || hi_ < lo_ || hi_ > n_layer - 1)
      throw std::runtime_error(
          "QWENOX_CVEC_LAYERS must be 1.." + std::to_string(n_layer - 1) +
          " with lo<=hi (got " + std::to_string(lo_) + ".." + std::to_string(hi_) + ")");
    if (single_ != -1 && (single_ < 1 || single_ > n_layer - 1))
      throw std::runtime_error("QWENOX_CVEC_DIR single:<layer> out of range: " +
                               std::to_string(single_));
    on_ = true;
  }

  // 权重加载前的预检：配置合法 + 文件可打开。失败抛 runtime_error。
  static void precheck() {
    Cvec c;
    c.parse_env();
    if (!c.on_) return;
    FILE* f = fopen(c.path_.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + c.path_);
    fclose(f);
  }

  void load() {
    parse_env();
    if (!on_) return;
    if (g_cfg.d != 2560 || g_cfg.branches != 4)
      throw std::runtime_error(
          "QWENOX_CVEC: control vectors are only supported on the 2560-d / "
          "4-branch model");
    const int n_layer = g_cfg.layers;
    const int n = g_cfg.d;

    gguf::File f(path_);
    const std::string arch = f.kv_s("general.architecture");
    if (arch != "controlvector")
      fprintf(stderr, "cvec: warning: %s architecture is \"%s\", expected controlvector\n",
              path_.c_str(), arch.c_str());
    const std::string hint = f.kv_s("controlvector.model_hint");
    if (!hint.empty() && hint != "qwen4exp")
      fprintf(stderr, "cvec: warning: model_hint is \"%s\", expected qwen4exp\n", hint.c_str());

    host_.assign((size_t)n_layer * n, 0.f);
    host_u_.assign((size_t)n_layer * n, 0.f);
    host_s_.assign(n_layer, 0.f);
    float nmin = 0.f, nmax = 0.f;
    int loaded = 0, expected = 0;
    std::vector<float> raw(n);
    std::vector<int> missing;
    for (int il = lo_; il <= hi_; il++) {
      const int src = single_ >= 0 ? single_ : il;
      const std::string name = "direction." + std::to_string(src);
      const gguf::Tensor* t = f.find(name);
      if (!t) {
        missing.push_back(src);
        if (single_ < 0) continue;
        throw std::runtime_error("QWENOX_CVEC: " + path_ + " has no tensor " + name);
      }
      if (t->nd != 1 || t->ne[0] != (uint64_t)n)
        throw std::runtime_error("QWENOX_CVEC: " + name + " must be 1-D with " +
                                 std::to_string(n) + " elements");
      if (t->type != gguf::F32 && t->type != gguf::F16 && t->type != gguf::BF16)
        throw std::runtime_error("QWENOX_CVEC: " + name +
                                 " must be F32/F16/BF16 (got type " +
                                 std::to_string(t->type) + ")");
      gguf::dequant_row(t->type, t->data, raw.data(), n);
      double nrm = 0.0;
      for (int j = 0; j < n; j++) nrm += (double)raw[j] * raw[j];
      nrm = sqrt(nrm);
      if (!std::isfinite(nrm) || nrm <= 0.0)
        throw std::runtime_error("QWENOX_CVEC: " + name + " has zero/invalid norm");
      float* dst = &host_[(size_t)il * n];
      float* dtu = &host_u_[(size_t)il * n];
      for (int j = 0; j < n; j++) {
        const float u = (float)(raw[j] / nrm);
        dtu[j] = u;
        dst[j] = project() ? u : raw[j];
      }
      host_s_[il] = project() ? scale_ * (float)nrm : scale_;
      if (!loaded) { nmin = nmax = (float)nrm; }
      nmin = std::min(nmin, (float)nrm);
      nmax = std::max(nmax, (float)nrm);
      loaded++;
    }
    expected = hi_ - lo_ + 1;
    if (loaded == 0)
      throw std::runtime_error("QWENOX_CVEC: " + path_ + " has no direction tensors in layers " +
                               std::to_string(lo_) + ".." + std::to_string(hi_));
    if (!missing.empty()) {
      std::string list;
      for (size_t i = 0; i < missing.size() && i < 8; i++)
        list += (i ? "," : "") + std::to_string(missing[i]);
      if (missing.size() > 8) list += ",...";
      fprintf(stderr, "cvec: warning: %zu direction(s) missing in range, skipped: %s\n",
              missing.size(), list.c_str());
    }
    on_ = true;
    CK(hipMalloc((void**)&d_v_, (size_t)n_layer * n * sizeof(float)));
    CK(hipMalloc((void**)&d_s_, (size_t)n_layer * sizeof(float)));
    CK(hipMemcpy(d_v_, host_.data(), (size_t)n_layer * n * sizeof(float),
                 hipMemcpyHostToDevice));
    CK(hipMemcpy(d_s_, host_s_.data(), (size_t)n_layer * sizeof(float),
                 hipMemcpyHostToDevice));
    fprintf(stderr,
            "qwenox: cvec %s: mode=%s dir=%s layers=%d..%d scale=%g, %d/%d directions "
            "(norm %.6f..%.6f)\n",
            path_.c_str(), project() ? "project" : "add",
            single_ < 0 ? "per-layer" : ("single:" + std::to_string(single_)).c_str(), lo_,
            hi_, (double)scale_, loaded, expected, (double)nmin, (double)nmax);
    if (!project())
      fprintf(stderr,
              "cvec: warning: add mode with scale %g — the additive vector file is "
              "meant for scale 0.05-0.15 on layers 16..32\n", (double)scale_);

    if (const char* pr = getenv("QWENOX_CVEC_PROBE"); pr && *pr) {
      int l = atoi(pr);
      if (l < 1 || l > n_layer - 1)
        throw std::runtime_error("QWENOX_CVEC_PROBE: layer out of range");
      if (!in_range(l))
        fprintf(stderr, "cvec: warning: probe layer %d is outside %d..%d, ignored\n", l, lo_,
                hi_);
      else {
        probe_ = l;
        CK(hipMalloc((void**)&d_probe_, (size_t)2 * g_cfg.branches * n * sizeof(float)));
        fprintf(stderr, "cvec: probe on layer %d (decode graphs disabled)\n", probe_);
      }
    }
  }

  // 前/后残差抓取（仅 probe；graphs 已被禁用）。
  void probe_capture(int l, void* buf, int P, bool bf16, hipStream_t st, bool post) {
    const int n = g_cfg.d;
    const size_t row = (size_t)g_cfg.branches * n;
    float* dst = d_probe_ + (post ? (size_t)g_cfg.branches * n : 0);
    const size_t off = (size_t)(P - 1) * row;
    if (bf16)
      k_bf16_to_f32<<<(unsigned)((row + 255) / 256), 256, 0, st>>>(
          (const uint16_t*)buf + off, dst, row);
    else
      CK(hipMemcpyAsync(dst, (float*)buf + off, row * sizeof(float),
                        hipMemcpyDeviceToDevice, st));
  }

  void apply(int l, void* buf, int P, bool bf16, hipStream_t st) {
    if (!in_range(l) || P < 1) return;
    const int n = g_cfg.d;
    const int row_stride = g_cfg.branches * n;
    dim3 grid((unsigned)P, (unsigned)g_cfg.branches);
    const bool pr = probe_ == l;
    if (pr) probe_capture(l, buf, P, bf16, st, false);
    if (bf16) {
      if (project())
        k_cvec_apply<true, true><<<grid, 256, 0, st>>>(buf, dir(l), s(l), row_stride, n);
      else
        k_cvec_apply<true, false><<<grid, 256, 0, st>>>(buf, dir(l), s(l), row_stride, n);
    } else {
      if (project())
        k_cvec_apply<false, true><<<grid, 256, 0, st>>>(buf, dir(l), s(l), row_stride, n);
      else
        k_cvec_apply<false, false><<<grid, 256, 0, st>>>(buf, dir(l), s(l), row_stride, n);
    }
    if (pr) probe_capture(l, buf, P, bf16, st, true);
  }

  // 在 host 同步之后调用；只打印前 2 个批次。
  void probe_report() {
    if (probe_ < 0 || probe_shown_ >= 2) return;
    const int n = g_cfg.d;
    std::vector<float> pre((size_t)g_cfg.branches * n), post(pre.size());
    CK(hipMemcpy(pre.data(), d_probe_, pre.size() * sizeof(float),
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(post.data(), d_probe_ + pre.size(), post.size() * sizeof(float),
                 hipMemcpyDeviceToHost));
    const float* u = &host_u_[(size_t)probe_ * n];
    auto cos1 = [&](const float* h) {
      double d = 0, q = 0;
      for (int j = 0; j < n; j++) { d += (double)h[j] * u[j]; q += (double)h[j] * h[j]; }
      return q > 0 ? (float)(d / sqrt(q)) : 0.f;
    };
    float mx = 0.f, sum = 0.f;
    fprintf(stderr, "cvec probe L%d (%s):", probe_, project() ? "project" : "add");
    for (int b = 0; b < g_cfg.branches; b++) {
      probe_cos_[0][b] = cos1(&pre[(size_t)b * n]);
      probe_cos_[1][b] = cos1(&post[(size_t)b * n]);
      fprintf(stderr, " br%d pre=%.5f post=%.5f", b, (double)probe_cos_[0][b],
              (double)probe_cos_[1][b]);
      const float a = fabsf(probe_cos_[1][b]);
      mx = std::max(mx, a);
      sum += a;
    }
    if (project()) {
      const float lim = 2e-2f;
      fprintf(stderr, " mean|post|=%.5f max|post|=%.5f %s\n", (double)(sum / g_cfg.branches),
              (double)mx, mx < lim ? "PASS" : "FAIL");
    } else {
      fprintf(stderr, "\n");
    }
    probe_shown_++;
  }

  // 快照指纹：关闭时 0（不改变既有指纹）；开启时随参数和向量内容变化。
  uint64_t fingerprint() const {
    if (!on_) return 0;
    uint64_t h = 0x4345564350524f4aULL;  // "CVECPROJ"
    auto add = [&](uint64_t x) { h = cvec_mix64(h ^ x); };
    auto bits = [](float f) {
      uint32_t u;
      memcpy(&u, &f, 4);
      return (uint64_t)u;
    };
    add((uint64_t)mode_);
    add(bits(scale_));
    add((uint64_t)lo_);
    add((uint64_t)hi_);
    add((uint64_t)(single_ + 1));
    uint64_t hv = 1469598103934665603ULL;
    for (float f : host_) hv = (hv ^ bits(f)) * 1099511628211ULL;
    add(hv);
    return h;
  }

  void release() {
    if (d_v_) { (void)hipFree(d_v_); d_v_ = nullptr; }
    if (d_s_) { (void)hipFree(d_s_); d_s_ = nullptr; }
    if (d_probe_) { (void)hipFree(d_probe_); d_probe_ = nullptr; }
  }
};
