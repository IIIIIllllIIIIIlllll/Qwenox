// ---- DFlash2 draft head: kernels + GpuModel methods ----------------------
// Sidecar loading, target-tap capture (HC mean), context injection, the
// 5-block draft forward, GPU top-k, the selector chain walk and the greedy
// spec loop. Included after 40_model.inc, so GpuModel is complete here.
//
// Reference: DFlash2/upstream/model.py (fixed revision 07ebd93d) and
// DFlash2/upstream/qwen38-dflash2-capture.patch. See DFLASH2-HANDOFF.md.
// df2bench/up_ref.py runs the upstream DFlash2DraftModel on the engine's
// captured features (QWENOX_DF2_RDUMP) and matches the drafts token for token.

// ---------------------------------------------------------------------------
// Sidecar
// ---------------------------------------------------------------------------
static const uint8_t* g_df2_pred_host = nullptr;  // [vocab][rank] BF16 rows
static const uint8_t* g_df2_succ_host = nullptr;
static int g_df2_vocab = 0;
// Sidecars quantized to Q8_0 are dequantized to BF16 for the host selector
// walk; BF16 sidecars keep pointing straight into the mmap.
static std::vector<uint16_t> g_df2_pred_storage;
static std::vector<uint16_t> g_df2_succ_storage;

static uint16_t df2_f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  const uint32_t r = u + 0x7fffu + ((u >> 16) & 1u);  // RNE
  return (uint16_t)(r >> 16);
}

static const uint8_t* df2_selector_host(const gguf::Tensor& t, const char* what,
                                        std::vector<uint16_t>& out) {
  if (t.nd != 2 || t.ne[0] != kDf2Rank)
    throw std::runtime_error(std::string("dflash2: selector ") + what +
                             " has unexpected shape");
  if (t.type == gguf::BF16) return t.data;
  if (t.type != gguf::Q8_0 && t.type != gguf::F32 && t.type != gguf::F16)
    throw std::runtime_error(std::string("dflash2: selector ") + what +
                             " has unsupported type " + gguf::type_name(t.type));
  const uint64_t cols = t.ne[0], rows = t.ne[1], rb = t.row_bytes();
  out.resize((size_t)rows * cols);
  std::vector<float> f(cols);
  for (uint64_t r = 0; r < rows; r++) {
    gguf::dequant_row(t.type, t.data + r * rb, f.data(), cols);
    for (uint64_t c = 0; c < cols; c++)
      out[(size_t)r * cols + c] = df2_f32_to_bf16(f[c]);
  }
  return (const uint8_t*)out.data();
}

// Opens the sidecar (mmap stays alive). Nothing is added to the Checkpoint:
// df2_init builds its own device copies (fused, q8g32 by default) straight
// from the mmap, and the selector codebooks stay host-only (CPU chain walk,
// 2 x 127 MiB). Q8_0 sidecars are accepted; the codebooks are expanded to
// BF16 once here.
static void df2_load_sidecar(Checkpoint& ck) {
  (void)ck;
  const char* path = getenv("QWENOX_DFLASH2");
  if (!path || !*path) return;
  if (!g_gguf_df2) g_gguf_df2 = new gguf::File(path);
  const gguf::Tensor& pred = g_gguf_df2->at("selector_predecessor.weight");
  const gguf::Tensor& succ = g_gguf_df2->at("selector_successor.weight");
  g_df2_pred_host = df2_selector_host(pred, "predecessor", g_df2_pred_storage);
  g_df2_succ_host = df2_selector_host(succ, "successor", g_df2_succ_storage);
  if (pred.ne[1] != succ.ne[1])
    throw std::runtime_error("dflash2: selector codebooks: vocab mismatch");
  g_df2_vocab = (int)pred.ne[1];
  const std::string cb = pred.type == gguf::BF16
                             ? std::string("BF16 codebooks")
                             : std::string(gguf::type_name(pred.type)) +
                                   " codebooks -> BF16";
  fprintf(stderr,
          "dflash2: sidecar %s (5 blocks, mask=%lld, selector vocab=%d, %s)\n", path,
          (long long)g_gguf_df2->kv_i("tokenizer.ggml.mask_token_id", 248077),
          g_df2_vocab, cb.c_str());
}

// Host f32 <-> f16 (RNE) for the q8g32 scale plane.
static uint16_t df2_f2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t ex = (x >> 23) & 0xffu;
  uint32_t m = x & 0x7fffffu;
  if (ex == 0xff) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
  const int e = (int)ex - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000u;
    const int shift = 14 - e;
    uint32_t h = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) h++;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
  return (uint16_t)(sign | h);
}
static float df2_h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
  if (e == 0) {
    if (!m) {
      x = sign;
    } else {
      e = 127 - 15 + 1;
      while (!(m & 0x400u)) {
        m <<= 1;
        e--;
      }
      x = sign | (e << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (e == 31) {
    x = sign | 0x7f800000u | (m << 13);
  } else {
    x = sign | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}

// Draft weight storage (QWENOX_DF2_WQ): q8 (q8g32, dtype 8, default) or
// bf16. A 4-bit IQ4_NL variant measured slower at P = 8 (dequant-bound) and
// lower acceptance, so it is not offered.
enum Df2WQ { kDf2WBF16 = 0, kDf2WQ8 = 1 };

// Row-concatenates GGUF matrices (ne[0] = cols, contiguous) and uploads them
// as q8g32 (dtype 8: int8 [rows][cols] then f16 scales [rows][cols/32], Q8_0
// rounding: d = amax/127, q = round(x/d)) or BF16 (dtype 0). BF16 sidecars
// keep the bit-exact fast path; Q8_0/F32/F16 rows are dequantized first.
static WView df2_upload(const std::vector<std::string>& names, int wq,
                        size_t* bytes) {
  uint64_t cols = 0, rows = 0;
  std::vector<const gguf::Tensor*> ts;
  bool all_bf16 = true;
  for (const std::string& n : names) {
    const gguf::Tensor& t = g_gguf_df2->at(n);
    if (t.nd != 2 || (t.type != gguf::BF16 && t.type != gguf::Q8_0 &&
                      t.type != gguf::F32 && t.type != gguf::F16))
      throw std::runtime_error("dflash2: " + n + " is not a supported matrix (" +
                               gguf::type_name(t.type) + ")");
    if (!cols) cols = t.ne[0];
    if (cols != t.ne[0]) throw std::runtime_error("dflash2: fused cols mismatch at " + n);
    rows += t.ne[1];
    ts.push_back(&t);
    all_bf16 = all_bf16 && t.type == gguf::BF16;
  }
  WView w{};
  w.rows = rows;
  w.cols = cols;
  const uint64_t ng = cols / 32;
  if (all_bf16) {
    std::vector<const uint16_t*> rowp;
    rowp.reserve(rows);
    for (const gguf::Tensor* t : ts)
      for (uint64_t i = 0; i < t->ne[1]; i++)
        rowp.push_back((const uint16_t*)t->data + i * cols);
    if (wq == kDf2WBF16 || cols % 32) {
      std::vector<uint16_t> h(rows * cols);
      for (uint64_t r = 0; r < rows; r++) memcpy(&h[r * cols], rowp[r], cols * 2);
      uint8_t* d = nullptr;
      DALLOC(&d, h.size() * 2);
      CK(hipMemcpy(d, h.data(), h.size() * 2, hipMemcpyHostToDevice));
      w.dtype = 0;
      w.data = d;
      *bytes += h.size() * 2;
      return w;
    }
    const uint64_t nq = rows * cols;
    std::vector<uint8_t> h(nq + rows * ng * 2);
    int8_t* q = (int8_t*)h.data();
    uint16_t* s = (uint16_t*)(h.data() + nq);
    auto work = [&](uint64_t r0, uint64_t r1) {
      for (uint64_t r = r0; r < r1; r++) {
        const uint16_t* src = rowp[r];
        for (uint64_t gi = 0; gi < ng; gi++) {
          float v[32], amax = 0.f;
          for (int i = 0; i < 32; i++) {
            v[i] = hgn::bf16_to_f32(src[gi * 32 + i]);
            amax = std::max(amax, std::fabs(v[i]));
          }
          const uint16_t dh = df2_f2h(amax / 127.f);
          const float dq = df2_h2f(dh), inv = dq > 0.f ? 1.f / dq : 0.f;
          for (int i = 0; i < 32; i++) {
            const float x = std::nearbyint(v[i] * inv);
            q[r * cols + gi * 32 + i] = (int8_t)std::max(-127.f, std::min(127.f, x));
          }
          s[r * ng + gi] = dh;
        }
      }
    };
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nt; i++)
      th.emplace_back(work, rows * i / nt, rows * (i + 1) / nt);
    for (std::thread& t : th) t.join();
    uint8_t* d = nullptr;
    DALLOC(&d, h.size());
    CK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice));
    w.dtype = 8;
    w.data = d;
    *bytes += h.size();
    return w;
  }

  struct Src {
    const gguf::Tensor* t;
    uint64_t row;
  };
  std::vector<Src> src;
  src.reserve(rows);
  for (const gguf::Tensor* t : ts)
    for (uint64_t i = 0; i < t->ne[1]; i++) src.push_back({t, i});

  if (wq == kDf2WBF16 || cols % 32) {
    std::vector<uint16_t> h(rows * cols);
    auto work = [&](uint64_t r0, uint64_t r1) {
      std::vector<float> f(cols);
      for (uint64_t r = r0; r < r1; r++) {
        const Src& s = src[r];
        gguf::dequant_row(s.t->type, s.t->data + s.row * s.t->row_bytes(), f.data(), cols);
        for (uint64_t c = 0; c < cols; c++)
          h[r * cols + c] = df2_f32_to_bf16(f[c]);
      }
    };
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned i = 0; i < nt; i++)
      th.emplace_back(work, rows * i / nt, rows * (i + 1) / nt);
    for (std::thread& t : th) t.join();
    uint8_t* d = nullptr;
    DALLOC(&d, h.size() * 2);
    CK(hipMemcpy(d, h.data(), h.size() * 2, hipMemcpyHostToDevice));
    w.dtype = 0;
    w.data = d;
    *bytes += h.size() * 2;
    return w;
  }

  const uint64_t nq = rows * cols;
  std::vector<uint8_t> h(nq + rows * ng * 2);
  int8_t* q = (int8_t*)h.data();
  uint16_t* s = (uint16_t*)(h.data() + nq);
  auto work = [&](uint64_t r0, uint64_t r1) {
    std::vector<float> f(cols);
    for (uint64_t r = r0; r < r1; r++) {
      const Src& sr = src[r];
      gguf::dequant_row(sr.t->type, sr.t->data + sr.row * sr.t->row_bytes(),
                        f.data(), cols);
      for (uint64_t gi = 0; gi < ng; gi++) {
        float amax = 0.f;
        for (int i = 0; i < 32; i++)
          amax = std::max(amax, std::fabs(f[gi * 32 + i]));
        const uint16_t dh = df2_f2h(amax / 127.f);
        const float dq = df2_h2f(dh), inv = dq > 0.f ? 1.f / dq : 0.f;
        for (int i = 0; i < 32; i++) {
          const float x = std::nearbyint(f[gi * 32 + i] * inv);
          q[r * cols + gi * 32 + i] = (int8_t)std::max(-127.f, std::min(127.f, x));
        }
        s[r * ng + gi] = dh;
      }
    }
  };
  const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nt; i++)
    th.emplace_back(work, rows * i / nt, rows * (i + 1) / nt);
  for (std::thread& t : th) t.join();
  uint8_t* d = nullptr;
  DALLOC(&d, h.size());
  CK(hipMemcpy(d, h.data(), h.size(), hipMemcpyHostToDevice));
  w.dtype = 8;
  w.data = d;
  *bytes += h.size();
  return w;
}

static float* df2_up_f32(const std::string& name, uint64_t want) {
  const gguf::Tensor& t = g_gguf_df2->at(name);
  if (t.numel() != want ||
      (t.type != gguf::BF16 && t.type != gguf::F32 && t.type != gguf::F16))
    throw std::runtime_error("dflash2: bad tensor " + name);
  std::vector<float> h((size_t)want);
  if (t.type == gguf::F32) {
    memcpy(h.data(), t.data, (size_t)want * 4);
  } else if (t.type == gguf::F16) {
    for (uint64_t i = 0; i < want; i++) {
      uint16_t v;
      memcpy(&v, t.data + 2 * i, 2);
      h[(size_t)i] = gguf::h2f(v);
    }
  } else {
    const uint16_t* s = (const uint16_t*)t.data;
    for (uint64_t i = 0; i < want; i++) h[(size_t)i] = hgn::bf16_to_f32(s[i]);
  }
  float* d = nullptr;
  DALLOC(&d, want * 4);
  CK(hipMemcpy(d, h.data(), want * 4, hipMemcpyHostToDevice));
  return d;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
__global__ void k_df2_rope_tab_init(double theta, int rotary, float mscale) {
  int i = threadIdx.x;
  if (i < rotary / 2) g_df2_rope_tab[i] = pow(theta, -2.0 * i / rotary);
  if (i == 0) g_df2_rope_mscale = mscale;
}

// HC mean of one completed trunk layer: out = mean over the 4 x 2560 streams,
// for chunk rows r0 + blockIdx.y. ring_cap > 0: prompt tap ring (slot =
// (base+r) % ring_cap); 0: verify vtap (row = local r).
template <bool BF16>
__global__ void k_df2_tap_capture(const void* __restrict__ Rb,
                                  float* __restrict__ out, int P, int r0,
                                  int base, int t, int ring_cap) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = r0 + blockIdx.y;
  if (c >= kDf2H || r >= P) return;
  float v;
  if (BF16) {
    const uint16_t* p = (const uint16_t*)Rb + (size_t)r * 4 * kDf2H;
    v = (bf2f(p[c]) + bf2f(p[kDf2H + c]) + bf2f(p[2 * kDf2H + c]) +
         bf2f(p[3 * kDf2H + c])) *
        0.25f;
  } else {
    const float* p = (const float*)Rb + (size_t)r * 4 * kDf2H;
    v = (p[c] + p[kDf2H + c] + p[2 * kDf2H + c] + p[3 * kDf2H + c]) * 0.25f;
  }
  const int row = ring_cap ? ((base + r) % ring_cap) : r;
  out[((size_t)row * kDf2Layers + t) * kDf2H + c] = v;
}

// Gather P tap rows ([layer][H] each) into [row][layer][H] (fc input layout).
__global__ void k_df2_gather_tap(const float* __restrict__ tap,
                                 float* __restrict__ feat, int base, int P,
                                 int is_vtap) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int t = blockIdx.y;
  int r = blockIdx.z;
  if (c >= kDf2H || r >= P) return;
  const size_t row = is_vtap ? (size_t)r : (size_t)((base + r) % kDf2TapRows);
  feat[((size_t)r * kDf2Layers + t) * kDf2H + c] =
      tap[(row * kDf2Layers + t) * kDf2H + c];
}

// Standard RMSNorm (Qwen3): y = x * rsqrt(mean(x^2)+eps) * w.
__global__ void k_df2_rmsnorm(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ y, int H, float eps) {
  const int r = blockIdx.x;
  const float* xr = x + (size_t)r * H;
  __shared__ float sm[256];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < H; i += blockDim.x) ss += xr[i] * xr[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  const float inv = rsqrtf(sm[0] / H + eps);
  for (int i = t; i < H; i += blockDim.x)
    y[(size_t)r * H + i] = xr[i] * inv * w[i];
}

// Grouped dynamic causal conv, block-local (left zero pad), side 0 or 1.
// side 1 reuses the dynamic coefficients projected from the pre-conv input.
__global__ void k_df2_conv(const float* __restrict__ in,
                           const float* __restrict__ dyn,
                           const float* __restrict__ base,
                           float* __restrict__ out, int P, int side) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y;
  if (c >= kDf2H || r >= P) return;
  const int G = kDf2ConvG;
  float acc = 0.f;
  for (int tap = 0; tap < kDf2ConvK; tap++) {
    if (r - tap < 0) break;
    const float b = base[(side * kDf2ConvK + tap) * kDf2H + c];
    const float d =
        dyn[(size_t)r * (2 * kDf2ConvK * kDf2Groups) +
            (side * kDf2ConvK + tap) * kDf2Groups + c / G];
    acc += (b + d) * in[(size_t)(r - tap) * kDf2H + c];
  }
  out[(size_t)r * kDf2H + c] = acc;
}

// conv side 1 + residual add: x += conv(in, side 1).
__global__ void k_df2_conv_add(const float* __restrict__ in,
                               const float* __restrict__ dyn,
                               const float* __restrict__ base,
                               float* __restrict__ x, int P) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y;
  if (c >= kDf2H || r >= P) return;
  float acc = 0.f;
  for (int tap = 0; tap < kDf2ConvK; tap++) {
    if (r - tap < 0) break;
    const float b = base[(kDf2ConvK + tap) * kDf2H + c];
    const float d = dyn[(size_t)r * (2 * kDf2ConvK * kDf2Groups) +
                        (kDf2ConvK + tap) * kDf2Groups + c / kDf2ConvG];
    acc += (b + d) * in[(size_t)(r - tap) * kDf2H + c];
  }
  x[(size_t)r * kDf2H + c] += acc;
}

// Per-head RMSNorm + NeoX RoPE (half split) for q (NH heads) and k (NKV),
// in the fused qkv rows (row stride rs). One block per (head, row).
__global__ void k_df2_qk_norm_rope(float* __restrict__ q, float* __restrict__ k,
                                   const float* __restrict__ qw,
                                   const float* __restrict__ kw, int P, int base,
                                   float eps, int rs) {
  const int h = blockIdx.x;
  const int r = blockIdx.y;
  if (r >= P) return;
  const bool isq = h < kDf2NH;
  const int hh = isq ? h : h - kDf2NH;
  float* x = (isq ? q : k) + (size_t)r * rs + (size_t)hh * kDf2HD;
  const float* w = isq ? qw : kw;
  __shared__ float sm[128], yn[128];
  const int t = threadIdx.x;
  const float v = x[t];
  sm[t] = v * v;
  __syncthreads();
  for (int off = 64; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  yn[t] = v * rsqrtf(sm[0] / kDf2HD + eps) * w[t];
  __syncthreads();
  if (t < 64) {
    const double ang = (double)(base + r) * g_df2_rope_tab[t];
    float cs = (float)cos(ang), sn = (float)sin(ang);
    if (g_df2_rope_mscale != 1.f) cs *= g_df2_rope_mscale, sn *= g_df2_rope_mscale;
    const float x0 = yn[t], x1 = yn[t + 64];
    yn[t] = x0 * cs - x1 * sn;
    yn[t + 64] = x0 * sn + x1 * cs;
  }
  __syncthreads();
  x[t] = yn[t];
}

// Non-causal sliding-window GQA for one (kv head, block row): the 4 q heads
// sharing the kv head run together so every K/V row is read once. Keys are
// the context ring (positions [lo, hi)) then the block's own noise K/V (all
// visible). 128 threads = 4 wave32s: the K pass is wave-cooperative (each
// lane owns 4 of the 128 dims, coalesced 512 B rows), the V pass is
// thread-per-dim. q/kb/vb live in the fused qkv rows (row stride rs).
__global__ void __launch_bounds__(128)
k_df2_attn(const float* __restrict__ q, const float* __restrict__ kb,
           const float* __restrict__ vb, int rs, const float* __restrict__ kctx,
           const float* __restrict__ vctx, float* __restrict__ out, int P,
           int base, int tap_hi, int window) {
  constexpr int G = kDf2NH / kDf2NKV;  // 4
  constexpr int NMAX = kDf2Window + kDf2Block + 8;
  const int kvh = blockIdx.x;
  const int r = blockIdx.y;
  const int t = threadIdx.x, lane = t & 31, wv = t >> 5;
  const int qpos = base + r;
  int lo = qpos - (window - 1);
  if (lo < 0) lo = 0;
  const int hi = base < tap_hi ? base : tap_hi;
  const int nctx = hi > lo ? hi - lo : 0;
  const int n = nctx + P;
  __shared__ float sc[G][NMAX];
  __shared__ float red[G][4];
  const float scale = rsqrtf((float)kDf2HD);
  float qv[G][4];
#pragma unroll
  for (int g = 0; g < G; g++) {
    const float4 q4 = *(const float4*)(q + (size_t)r * rs +
                                       (size_t)(kvh * G + g) * kDf2HD + lane * 4);
    qv[g][0] = q4.x * scale;
    qv[g][1] = q4.y * scale;
    qv[g][2] = q4.z * scale;
    qv[g][3] = q4.w * scale;
  }
  for (int idx = wv; idx < n; idx += 4) {
    const float* kp =
        idx < nctx ? kctx + ((size_t)((lo + idx) % kDf2Window) * kDf2NKV + kvh) * kDf2HD
                   : kb + (size_t)(idx - nctx) * rs + (size_t)kvh * kDf2HD;
    const float4 k4 = *(const float4*)(kp + lane * 4);
    float d[G];
#pragma unroll
    for (int g = 0; g < G; g++)
      d[g] = qv[g][0] * k4.x + qv[g][1] * k4.y + qv[g][2] * k4.z + qv[g][3] * k4.w;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
#pragma unroll
      for (int g = 0; g < G; g++) d[g] += __shfl_xor(d[g], off, 32);
    if (lane == 0)
#pragma unroll
      for (int g = 0; g < G; g++) sc[g][idx] = d[g];
  }
  __syncthreads();
  float mx[G];
#pragma unroll
  for (int g = 0; g < G; g++) mx[g] = -1e30f;
  for (int idx = t; idx < n; idx += 128)
#pragma unroll
    for (int g = 0; g < G; g++) mx[g] = fmaxf(mx[g], sc[g][idx]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
#pragma unroll
    for (int g = 0; g < G; g++) mx[g] = fmaxf(mx[g], __shfl_xor(mx[g], off, 32));
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < G; g++) red[g][wv] = mx[g];
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; g++)
    mx[g] = fmaxf(fmaxf(red[g][0], red[g][1]), fmaxf(red[g][2], red[g][3]));
  __syncthreads();
  float se[G];
#pragma unroll
  for (int g = 0; g < G; g++) se[g] = 0.f;
  for (int idx = t; idx < n; idx += 128)
#pragma unroll
    for (int g = 0; g < G; g++) {
      const float e = expf(sc[g][idx] - mx[g]);
      sc[g][idx] = e;
      se[g] += e;
    }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
#pragma unroll
    for (int g = 0; g < G; g++) se[g] += __shfl_xor(se[g], off, 32);
  if (lane == 0)
#pragma unroll
    for (int g = 0; g < G; g++) red[g][wv] = se[g];
  __syncthreads();
  float inv[G];
#pragma unroll
  for (int g = 0; g < G; g++)
    inv[g] = 1.f / (red[g][0] + red[g][1] + red[g][2] + red[g][3]);
  float acc[G];
#pragma unroll
  for (int g = 0; g < G; g++) acc[g] = 0.f;
  for (int idx = 0; idx < n; idx++) {
    const float* vp =
        idx < nctx ? vctx + ((size_t)((lo + idx) % kDf2Window) * kDf2NKV + kvh) * kDf2HD
                   : vb + (size_t)(idx - nctx) * rs + (size_t)kvh * kDf2HD;
    const float vv = vp[t];
#pragma unroll
    for (int g = 0; g < G; g++) acc[g] += sc[g][idx] * vv;
  }
#pragma unroll
  for (int g = 0; g < G; g++)
    out[((size_t)r * kDf2NH + kvh * G + g) * kDf2HD + t] = acc[g] * inv[g];
}

// Split-key variant for long contexts: every block takes one kv head and a
// 128-key slice of [context ring | block keys] for ALL B rows x 4 q heads
// (32 queries), so each K/V row is read once per round instead of once per
// block row. Partial (max, sum, acc) per (split, query) go to part; the
// combine kernel merges them. Key p of the context is visible to row r iff
// p >= base + r - (window - 1) (row 0's window start is the common lo).
constexpr int kDf2AttnS = 128;  // keys per split
constexpr int kDf2AttnQ = kDf2Block * (kDf2NH / kDf2NKV);  // 32 queries
constexpr int kDf2AttnMaxSplit = (kDf2Window + kDf2Block + kDf2AttnS - 1) / kDf2AttnS;
__global__ void __launch_bounds__(256)
k_df2_attn_part(const float* __restrict__ q, const float* __restrict__ kb,
                const float* __restrict__ vb, int rs,
                const float* __restrict__ kctx, const float* __restrict__ vctx,
                float* __restrict__ part, int P, int base, int tap_hi, int window) {
  constexpr int G = kDf2NH / kDf2NKV;
  constexpr int QN = kDf2AttnQ, S = kDf2AttnS, T = 32;  // key tile
  const int kvh = blockIdx.x, sp = blockIdx.y, t = threadIdx.x;
  int lo = base - (window - 1);
  if (lo < 0) lo = 0;
  const int hi = base < tap_hi ? base : tap_hi;
  const int nctx = hi > lo ? hi - lo : 0;
  const int n = nctx + P;
  const int k0 = sp * S;
  const int nq = P * G;
  float* pm = part + ((size_t)(kvh * kDf2AttnMaxSplit + sp) * QN) * (2 + kDf2HD);
  if (k0 >= n) {
    if (t < QN) pm[(size_t)t * (2 + kDf2HD)] = -INFINITY, pm[(size_t)t * (2 + kDf2HD) + 1] = 0.f;
    return;
  }
  __shared__ float qs[QN][kDf2HD + 1];
  __shared__ float ks[T][kDf2HD + 1];
  __shared__ float sc[QN][S];
  const float scale = rsqrtf((float)kDf2HD);
  for (int i = t; i < QN * kDf2HD; i += 256) {
    const int qi = i / kDf2HD, d = i - qi * kDf2HD;
    const int r = qi / G, g = qi - r * G;
    qs[qi][d] = qi < nq ? q[(size_t)r * rs + (size_t)(kvh * G + g) * kDf2HD + d] * scale : 0.f;
  }
  auto kptr = [&](int idx, const float* ctx, const float* blk) -> const float* {
    return idx < nctx ? ctx + ((size_t)((lo + idx) % kDf2Window) * kDf2NKV + kvh) * kDf2HD
                      : blk + (size_t)(idx - nctx) * rs + (size_t)kvh * kDf2HD;
  };
  const int kn = min(S, n - k0);
  const int qi = t >> 3, kk = t & 7;
  for (int tb = 0; tb < kn; tb += T) {
    __syncthreads();
    for (int i = t; i < T * kDf2HD; i += 256) {
      const int j = i / kDf2HD, d = i - j * kDf2HD;
      ks[j][d] = tb + j < kn ? kptr(k0 + tb + j, kctx, kb)[d] : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < T / 8; u++) {
      const int j = kk + u * 8;
      float acc = 0.f;
#pragma unroll 16
      for (int d = 0; d < kDf2HD; d++) acc += qs[qi][d] * ks[j][d];
      const int idx = k0 + tb + j;
      const int r = qi / G;
      bool vis = tb + j < kn && qi < nq;
      if (vis && idx < nctx) vis = lo + idx >= base + r - (window - 1);
      if (tb + j < S) sc[qi][tb + j] = vis ? acc : -INFINITY;
    }
  }
  __syncthreads();
  // Per query: max / exp / sum over the slice (8 threads per query).
  {
    float m = -INFINITY;
    for (int j = kk; j < kn; j += 8) m = fmaxf(m, sc[qi][j]);
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor(m, off, 8));
    float l = 0.f;
    for (int j = kk; j < kn; j += 8) {
      const float e = m == -INFINITY ? 0.f : expf(sc[qi][j] - m);
      sc[qi][j] = e;
      l += e;
    }
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) l += __shfl_xor(l, off, 8);
    if (kk == 0) {
      pm[(size_t)qi * (2 + kDf2HD)] = m;
      pm[(size_t)qi * (2 + kDf2HD) + 1] = l;
    }
  }
  // acc[qi][d0..d0+15] = sum_j p[qi][j] * V[j][d]
  const int d0 = kk * 16;
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) acc[i] = 0.f;
  for (int tb = 0; tb < kn; tb += T) {
    __syncthreads();
    for (int i = t; i < T * kDf2HD; i += 256) {
      const int j = i / kDf2HD, d = i - j * kDf2HD;
      ks[j][d] = tb + j < kn ? kptr(k0 + tb + j, vctx, vb)[d] : 0.f;
    }
    __syncthreads();
    const int jn = min(T, kn - tb);
    for (int j = 0; j < jn; j++) {
      const float p = sc[qi][tb + j];
#pragma unroll
      for (int i = 0; i < 16; i++) acc[i] += p * ks[j][d0 + i];
    }
  }
#pragma unroll
  for (int i = 0; i < 16; i++) pm[(size_t)qi * (2 + kDf2HD) + 2 + d0 + i] = acc[i];
}

// Merge split partials -> out[(r*NH + head)*HD + d]. grid (NKV, P*G), 128 thr.
__global__ void k_df2_attn_combine(const float* __restrict__ part,
                                   float* __restrict__ out, int nsplit) {
  constexpr int G = kDf2NH / kDf2NKV;
  const int kvh = blockIdx.x, qi = blockIdx.y, d = threadIdx.x;
  const int r = qi / G, g = qi - r * G;
  float M = -INFINITY;
  for (int s = 0; s < nsplit; s++)
    M = fmaxf(M, part[((size_t)(kvh * kDf2AttnMaxSplit + s) * kDf2AttnQ + qi) * (2 + kDf2HD)]);
  float L = 0.f, O = 0.f;
  for (int s = 0; s < nsplit; s++) {
    const float* p = part + ((size_t)(kvh * kDf2AttnMaxSplit + s) * kDf2AttnQ + qi) * (2 + kDf2HD);
    if (p[0] == -INFINITY) continue;
    const float w = expf(p[0] - M);
    L += p[1] * w;
    O += p[2 + d] * w;
  }
  out[((size_t)r * kDf2NH + kvh * G + g) * kDf2HD + d] = O / L;
}

// out[r][c] = silu(gu[r][c]) * gu[r][FF + c]  (fused gate|up rows).
__global__ void k_df2_silu_mul(const float* __restrict__ gu,
                               float* __restrict__ out, int P) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * kDf2FF) return;
  const int r = i / kDf2FF, c = i - r * kDf2FF;
  const float g = gu[(size_t)r * 2 * kDf2FF + c];
  out[i] = (g / (1.f + expf(-g))) * gu[(size_t)r * 2 * kDf2FF + kDf2FF + c];
}

// Context injection for one draft layer: per-head k RMSNorm + RoPE at the
// absolute position, then K/V into the ring. kn/vn: rows of the fused
// all-layer K|V projection (row stride rs).
__global__ void k_df2_inject_kv(const float* __restrict__ kn,
                                const float* __restrict__ vn,
                                const float* __restrict__ kw,
                                float* __restrict__ kctx,
                                float* __restrict__ vctx, int P, int base,
                                float eps, int rs) {
  const int h = blockIdx.x;
  const int r = blockIdx.y;
  if (r >= P) return;
  const int t = threadIdx.x;
  const float* kp = kn + (size_t)r * rs + (size_t)h * kDf2HD;
  __shared__ float sm[128], yn[128];
  const float v = kp[t];
  sm[t] = v * v;
  __syncthreads();
  for (int off = 64; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  yn[t] = v * rsqrtf(sm[0] / kDf2HD + eps) * kw[t];
  __syncthreads();
  if (t < 64) {
    const double ang = (double)(base + r) * g_df2_rope_tab[t];
    float cs = (float)cos(ang), sn = (float)sin(ang);
    if (g_df2_rope_mscale != 1.f) cs *= g_df2_rope_mscale, sn *= g_df2_rope_mscale;
    const float x0 = yn[t], x1 = yn[t + 64];
    yn[t] = x0 * cs - x1 * sn;
    yn[t + 64] = x0 * sn + x1 * cs;
  }
  __syncthreads();
  const size_t slot = (size_t)((base + r) % kDf2Window);
  kctx[(slot * kDf2NKV + h) * kDf2HD + t] = yn[t];
  vctx[(slot * kDf2NKV + h) * kDf2HD + t] = vn[(size_t)r * rs + (size_t)h * kDf2HD + t];
}

// BF16 embedding gather (dtype 0 trunks / QWENOX_DF2_EMB_BF16); dtype 5/8
// use embed_rows.
__global__ void k_df2_embed_bf16(const uint16_t* __restrict__ et,
                                 const int* __restrict__ tok,
                                 float* __restrict__ out, int P) {
  const int r = blockIdx.x;
  if (r >= P) return;
  const uint16_t* row = et + (size_t)tok[r] * kDf2H;
  for (int c = threadIdx.x; c < kDf2H; c += blockDim.x)
    out[(size_t)r * kDf2H + c] = bf2f(row[c]);
}

// Two-stage top-16 + log-sum-exp per logits row (ties -> lower id, same as a
// sequential insertion scan). Stage 1 (grid npos x kDf2TopkChunks, 256 thr):
// each thread keeps a sorted top-16 + online (max, sum) over a strided slice
// of its chunk; 16 rounds of block argmax over the 256 list heads give the
// chunk's top-16. Partial row: [16 v][16 id][max][sum]. Stage 2 (npos blocks):
// the same merge over the chunks' 16 x kDf2TopkChunks candidates.
// Output row layout: [16 values][16 ids (int bits)][lse].
constexpr int kDf2TopkChunks = 32;
constexpr int kDf2TopkPart = 2 * kDf2TopK + 2;

// Block-wide pop of the best head among per-thread sorted lists, kDf2TopK
// times. Each thread's list (tv/ti, descending) is consumed via ptr.
template <int NT>
__device__ void df2_topk_merge_lists(const float* tv, const int* ti, float* ov, int* oi) {
  __shared__ float hv[NT];
  __shared__ int hi_[NT], ht[NT];
  const int t = threadIdx.x;
  int ptr = 0;
  for (int k = 0; k < kDf2TopK; k++) {
    float myv = -INFINITY;
    int myi = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < kDf2TopK; j++)
      if (j == ptr) myv = tv[j], myi = ti[j];
    hv[t] = myv;
    hi_[t] = myi;
    ht[t] = t;
    __syncthreads();
    for (int off = NT / 2; off > 0; off >>= 1) {
      if (t < off) {
        const float v2 = hv[t + off];
        const int i2 = hi_[t + off];
        if (v2 > hv[t] || (v2 == hv[t] && i2 < hi_[t])) {
          hv[t] = v2;
          hi_[t] = i2;
          ht[t] = ht[t + off];
        }
      }
      __syncthreads();
    }
    if (t == 0) {
      ov[k] = hv[0];
      oi[k] = hi_[0];
    }
    if (ht[0] == t) ptr++;
    __syncthreads();
  }
}

__device__ __forceinline__ void df2_ms_merge(float& m, float& s, float m2, float s2) {
  const float mm = fmaxf(m, m2);
  const float a = m == -INFINITY ? 0.f : s * expf(m - mm);
  const float b = m2 == -INFINITY ? 0.f : s2 * expf(m2 - mm);
  m = mm;
  s = a + b;
}

__device__ __forceinline__ void df2_list_insert(float* tv, int* ti, float v, int i) {
  if (!(v > tv[kDf2TopK - 1] || (v == tv[kDf2TopK - 1] && i < ti[kDf2TopK - 1]))) return;
  int p = kDf2TopK - 1;
  while (p > 0 && (tv[p - 1] < v || (tv[p - 1] == v && ti[p - 1] > i))) {
    tv[p] = tv[p - 1];
    ti[p] = ti[p - 1];
    p--;
  }
  tv[p] = v;
  ti[p] = i;
}

__global__ void __launch_bounds__(256)
k_df2_topk_part(const float* __restrict__ lg, int V, float* __restrict__ part) {
  const int row = blockIdx.x, ch = blockIdx.y, t = threadIdx.x;
  const int cs = (V + kDf2TopkChunks - 1) / kDf2TopkChunks;
  const int i0 = ch * cs, i1 = min(V, i0 + cs);
  const float* x = lg + (size_t)row * V;
  float tv[kDf2TopK];
  int ti[kDf2TopK];
#pragma unroll
  for (int k = 0; k < kDf2TopK; k++) tv[k] = -INFINITY, ti[k] = 0x7fffffff;
  float m = -INFINITY, s = 0.f;
  for (int i = i0 + t; i < i1; i += 256) {
    const float v = x[i];
    df2_ms_merge(m, s, v, 1.f);
    df2_list_insert(tv, ti, v, i);
  }
  __shared__ float rm[256], rs_[256];
  rm[t] = m;
  rs_[t] = s;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (t < off) df2_ms_merge(rm[t], rs_[t], rm[t + off], rs_[t + off]);
    __syncthreads();
  }
  float* o = part + ((size_t)row * kDf2TopkChunks + ch) * kDf2TopkPart;
  if (t == 0) {
    o[2 * kDf2TopK] = rm[0];
    o[2 * kDf2TopK + 1] = rs_[0];
  }
  __shared__ float ov[kDf2TopK];
  __shared__ int oi[kDf2TopK];
  df2_topk_merge_lists<256>(tv, ti, ov, oi);
  if (t < kDf2TopK) {
    o[t] = ov[t];
    ((int*)o)[kDf2TopK + t] = oi[t];
  }
}

// One block per row, kDf2TopkChunks threads: thread c owns chunk c's list.
__global__ void __launch_bounds__(kDf2TopkChunks)
k_df2_topk_merge(const float* __restrict__ part, float* __restrict__ out) {
  const int row = blockIdx.x, t = threadIdx.x;
  const float* p = part + ((size_t)row * kDf2TopkChunks + t) * kDf2TopkPart;
  float tv[kDf2TopK];
  int ti[kDf2TopK];
#pragma unroll
  for (int k = 0; k < kDf2TopK; k++) tv[k] = p[k], ti[k] = ((const int*)p)[kDf2TopK + k];
  __shared__ float rm[kDf2TopkChunks], rs_[kDf2TopkChunks];
  rm[t] = p[2 * kDf2TopK];
  rs_[t] = p[2 * kDf2TopK + 1];
  __syncthreads();
  for (int off = kDf2TopkChunks / 2; off > 0; off >>= 1) {
    if (t < off) df2_ms_merge(rm[t], rs_[t], rm[t + off], rs_[t + off]);
    __syncthreads();
  }
  float* o = out + (size_t)row * kDf2TopkRow;
  if (t == 0) o[2 * kDf2TopK] = rm[0] + logf(rs_[0]);
  __shared__ float ov[kDf2TopK];
  __shared__ int oi[kDf2TopK];
  df2_topk_merge_lists<kDf2TopkChunks>(tv, ti, ov, oi);
  if (t < kDf2TopK) {
    o[t] = ov[t];
    ((int*)o)[kDf2TopK + t] = oi[t];
  }
}

// ---------------------------------------------------------------------------
// GpuModel methods
// ---------------------------------------------------------------------------
bool GpuModel::df2_init() {
  const char* path = getenv("QWENOX_DFLASH2");
  if (!path || !*path) return false;
  if (!g_gguf_df2) {
    fprintf(stderr, "dflash2: sidecar %s was not loaded (df2_load_sidecar)\n", path);
    return false;
  }
  auto t_init = std::chrono::steady_clock::now();
  df2.mask_id = (int)g_gguf_df2->kv_i("tokenizer.ggml.mask_token_id", 248077);
  df2.block_size = (int)g_gguf_df2->kv_i("dflash.block_size", kDf2Block);
  df2.window = (int)g_gguf_df2->kv_i("dflash.attention.sliding_window", kDf2Window);
  df2.norm_eps = (float)g_gguf_df2->kv_f("dflash.attention.layer_norm_rms_epsilon",
                                         g_cfg.norm_eps);
  if (df2.block_size > kDf2Block) df2.block_size = kDf2Block;
  if (df2.window > kDf2Window) df2.window = kDf2Window;
  {
    const gguf::Value* tv = g_gguf_df2->kv("dflash.target_layers");
    if (tv && tv->is_array && tv->ai.size() >= kDf2Layers) {
      for (int i = 0; i < kDf2Layers; i++) {
        // GGUF stores layer INPUT indices (HF completed layer + 1).
        const int l = (int)tv->ai[(size_t)i] - 1;
        df2.trunk_layer[i] = l >= 0 ? l : 0;
      }
    }
  }
  if (g_df2_vocab && g_df2_vocab != g_cfg.vocab)
    throw std::runtime_error("dflash2: selector vocab " + std::to_string(g_df2_vocab) +
                             " != trunk vocab " + std::to_string(g_cfg.vocab));
  df2.sel_pred_host = g_df2_pred_host;
  df2.sel_succ_host = g_df2_succ_host;
  if (!df2.sel_pred_host || !df2.sel_succ_host)
    throw std::runtime_error("dflash2: selector codebooks not loaded");

  // QWENOX_DF2_WQ=bf16|q8 (default q8): draft matrix storage.
  const int q8 = [] {
    const char* e = getenv("QWENOX_DF2_WQ");
    if (!e || !*e) return (int)kDf2WQ8;
    return !strcmp(e, "bf16") ? (int)kDf2WBF16 : (int)kDf2WQ8;
  }();
  size_t wbytes = 0;
  df2.fc = df2_upload({"fc.weight"}, q8, &wbytes);
  std::vector<std::string> kvn;
  for (int l = 0; l < kDf2Layers; l++) {
    kvn.push_back("blk." + std::to_string(l) + ".attn_k.weight");
    kvn.push_back("blk." + std::to_string(l) + ".attn_v.weight");
  }
  df2.kv_all = df2_upload(kvn, q8, &wbytes);
  df2.sel_hidden = df2_upload({"selector_hidden.weight"}, kDf2WBF16, &wbytes);
  if (df2.fc.cols != kDf2Feat || df2.fc.rows != kDf2H ||
      df2.kv_all.rows != (uint64_t)kDf2Layers * 2 * kDf2NKV * kDf2HD ||
      df2.kv_all.cols != kDf2H || df2.sel_hidden.rows != kDf2Rank ||
      df2.sel_hidden.cols != kDf2H)
    throw std::runtime_error("dflash2: fc/kv/selector_hidden shape mismatch");
  df2.d_enc_norm = df2_up_f32("enc.output_norm.weight", kDf2H);
  df2.d_out_norm = df2_up_f32("output_norm.weight", kDf2H);

  for (int l = 0; l < kDf2Layers; l++) {
    auto nm = [&](const char* n) { return "blk." + std::to_string(l) + "." + n; };
    Df2LayerW& w = df2.lw[l];
    w.qkv = df2_upload({nm("attn_q.weight"), nm("attn_k.weight"), nm("attn_v.weight")},
                       q8, &wbytes);
    w.o = df2_upload({nm("attn_output.weight")}, q8, &wbytes);
    w.gu = df2_upload({nm("ffn_gate.weight"), nm("ffn_up.weight")}, q8, &wbytes);
    w.down = df2_upload({nm("ffn_down.weight")}, q8, &wbytes);
    w.attn_conv_proj = df2_upload({nm("attn_conv_proj.weight")}, q8, &wbytes);
    w.ffn_conv_proj = df2_upload({nm("ffn_conv_proj.weight")}, q8, &wbytes);
    if (w.qkv.rows != kDf2QKV || w.qkv.cols != kDf2H || w.o.rows != kDf2H ||
        w.o.cols != kDf2NH * kDf2HD || w.gu.rows != 2 * kDf2FF || w.gu.cols != kDf2H ||
        w.down.rows != kDf2H || w.down.cols != kDf2FF ||
        w.attn_conv_proj.rows != 2 * kDf2ConvK * kDf2Groups ||
        w.attn_conv_proj.cols != kDf2H ||
        w.ffn_conv_proj.rows != 2 * kDf2ConvK * kDf2Groups ||
        w.ffn_conv_proj.cols != kDf2H)
      throw std::runtime_error("dflash2: block tensor shape mismatch at layer " +
                               std::to_string(l));
    w.d_attn_norm = df2_up_f32(nm("attn_norm.weight"), kDf2H);
    w.d_ffn_norm = df2_up_f32(nm("ffn_norm.weight"), kDf2H);
    w.d_q_norm = df2_up_f32(nm("attn_q_norm.weight"), kDf2HD);
    w.d_k_norm = df2_up_f32(nm("attn_k_norm.weight"), kDf2HD);
    w.d_attn_base = df2_up_f32(nm("attn_conv_base"), 2 * kDf2ConvK * kDf2H);
    w.d_ffn_base = df2_up_f32(nm("ffn_conv_base"), 2 * kDf2ConvK * kDf2H);
  }

  const size_t tap_bytes = (size_t)kDf2TapRows * kDf2Layers * kDf2H * 4;
  const size_t vtap_bytes = (size_t)kDf2VerifyRows * kDf2Layers * kDf2H * 4;
  const size_t kv_bytes = (size_t)kDf2Layers * kDf2Window * kDf2NKV * kDf2HD * 4;
  const size_t inj = kDf2InjRows;
  const size_t B = kDf2Block;
  DALLOC(&df2.d_tap, tap_bytes);
  DALLOC(&df2.d_vtap, vtap_bytes);
  DALLOC(&df2.d_kctx, kv_bytes);
  DALLOC(&df2.d_vctx, kv_bytes);
  DALLOC(&df2.d_feat, inj * kDf2Feat * 4);
  DALLOC(&df2.d_th, inj * kDf2H * 4);
  DALLOC(&df2.d_thn, inj * kDf2H * 4);
  DALLOC(&df2.d_kvn, inj * df2.kv_all.rows * 4);
  DALLOC(&df2.d_x, B * kDf2H * 4);
  DALLOC(&df2.d_norm, B * kDf2H * 4);
  DALLOC(&df2.d_conv, B * kDf2H * 4);
  DALLOC(&df2.d_qkv, B * kDf2QKV * 4);
  DALLOC(&df2.d_attn, B * kDf2NH * kDf2HD * 4);
  DALLOC(&df2.d_o, B * kDf2H * 4);
  DALLOC(&df2.d_dyn, B * 2 * kDf2ConvK * kDf2Groups * 4);
  DALLOC(&df2.d_gu, B * 2 * kDf2FF * 4);
  DALLOC(&df2.d_ff, B * kDf2FF * 4);
  DALLOC(&df2.d_out, B * kDf2H * 4);
  DALLOC(&df2.d_selh, B * kDf2Rank * 4);
  DALLOC(&df2.d_topk, B * kDf2TopkRow * 4);
  DALLOC(&df2.d_topk_part, B * kDf2TopkChunks * kDf2TopkPart * 4);
  DALLOC(&df2.d_tok, B * 4);
  DALLOC(&df2.d_apart,
         (size_t)kDf2NKV * kDf2AttnMaxSplit * kDf2AttnQ * (2 + kDf2HD) * 4);
  {
    const char* e = getenv("QWENOX_DF2_ATTN_SPLIT_MIN");
    df2.attn_split_min = e && *e ? atoi(e) : 256;
  }
  CK(hipMemset(df2.d_kctx, 0, kv_bytes));
  CK(hipMemset(df2.d_vctx, 0, kv_bytes));

  // Experiments: raw BF16 [vocab][H] embedding / lm_head files (HF exact)
  // for the draft only, and a trunk tap layer shift.
  auto load_raw = [&](const char* env, const char* what) -> const uint16_t* {
    const char* p = getenv(env);
    if (!p || !*p) return nullptr;
    const size_t bytes = (size_t)g_cfg.vocab * kDf2H * 2;
    FILE* f = fopen(p, "rb");
    if (!f) throw std::runtime_error(std::string("dflash2: cannot open ") + p);
    std::vector<uint16_t> h(bytes / 2);
    if (fread(h.data(), 1, bytes, f) != bytes)
      throw std::runtime_error(std::string("dflash2: short read ") + p);
    fclose(f);
    uint16_t* d = nullptr;
    DALLOC(&d, bytes);
    CK(hipMemcpy(d, h.data(), bytes, hipMemcpyHostToDevice));
    fprintf(stderr, "dflash2: %s from %s (BF16 exact)\n", what, p);
    return d;
  };
  df2.d_emb_bf16 = load_raw("QWENOX_DF2_EMB_BF16", "embedding");
  if (const uint16_t* lm = load_raw("QWENOX_DF2_LM_BF16", "lm_head")) {
    df2.lm_bf16.dtype = 0;
    df2.lm_bf16.data = (const uint8_t*)lm;
    df2.lm_bf16.rows = (uint64_t)g_cfg.vocab;
    df2.lm_bf16.cols = kDf2H;
    df2.lm_bf16_on = true;
  }
  if (const char* s = getenv("QWENOX_DF2_LSHIFT")) {
    const int sh = atoi(s);
    for (int i = 0; i < kDf2Layers; i++)
      df2.trunk_layer[i] = std::max(0, std::min(g_cfg.layers - 1, df2.trunk_layer[i] + sh));
  }
  {
    const char* c = getenv("QWENOX_DF2_CONF");
    df2.conf_th = c && *c ? (float)atof(c) : 0.5f;
  }

  df2.tap_hi = 0;
  df2.cap_mode = 0;
  k_df2_rope_tab_init<<<1, 64, 0, g_str>>>(10000.0, kDf2HD, 1.0f);
  CK(hipStreamSynchronize(g_str));
  df2.on = true;
  if (const char* gb = getenv("QWENOX_DF2_GBENCH")) {
    // Debug: GEMV time / effective bandwidth of each draft matrix at P rows.
    const int P = atoi(gb) > 0 ? atoi(gb) : 8;
    float *x = nullptr, *y = nullptr;
    DALLOC(&x, (size_t)P * kDf2Feat * 4);
    DALLOC(&y, (size_t)P * g_cfg.vocab * 4);
    CK(hipMemset(x, 0, (size_t)P * kDf2Feat * 4));
    auto bench = [&](const char* nm, const WView& w) {
      for (int i = 0; i < 3; i++) gemm(w, x, y, P, w.cols);
      CK(hipStreamSynchronize(g_str));
      const int it = 50;
      auto a = std::chrono::steady_clock::now();
      for (int i = 0; i < it; i++) gemm(w, x, y, P, w.cols);
      CK(hipStreamSynchronize(g_str));
      const double us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - a).count() / it;
      const double bytes = (double)w.rows * w.cols * (w.dtype == 8 ? 1.0625 : w.dtype == 0 ? 2.0 : 0.5);
      fprintf(stderr, "df2 gbench P=%d %-10s dtype %2d %6llux%-6llu %8.1f us %6.1f GB/s\n", P,
              nm, w.dtype, (unsigned long long)w.rows, (unsigned long long)w.cols, us,
              bytes / us / 1e3);
    };
    bench("qkv", df2.lw[0].qkv);
    bench("o", df2.lw[0].o);
    bench("gate|up", df2.lw[0].gu);
    bench("down", df2.lw[0].down);
    bench("conv_proj", df2.lw[0].attn_conv_proj);
    bench("fc", df2.fc);
    bench("kv_all", df2.kv_all);
    bench("sel_hidden", df2.sel_hidden);
    WView lm = wview(ckpt.at("lm_head.weight"));
    bench("lm_head", lm);
    CK(hipFree(x));
    CK(hipFree(y));
  }
  fprintf(stderr,
          "dflash2: draft head enabled (layers [%d,%d,%d,%d,%d], %s weights %.0f MiB, "
          "tap %.0f MiB, kv %.0f MiB, conf %.2f, %.1f s)\n",
          df2.trunk_layer[0], df2.trunk_layer[1], df2.trunk_layer[2],
          df2.trunk_layer[3], df2.trunk_layer[4], q8 == kDf2WQ8 ? "q8g32" : "BF16",
          wbytes / 1048576.0, tap_bytes / 1048576.0, kv_bytes * 2 / 1048576.0,
          df2.conf_th,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t_init).count());
  return true;
}

void GpuModel::df2_capture_layer(int l, int P, int base, int mode) {
  int t = -1;
  for (int i = 0; i < kDf2Layers; i++)
    if (df2.trunk_layer[i] == l) t = i;
  if (t < 0) return;
  const bool ring = mode == 1;
  // Verify passes wider than a DFlash2 block (ngram chunks) carry no
  // DFlash2 rows.
  if (!ring && P > kDf2VerifyRows) return;
  // Prompt chunks: only the last 2048 rows can ever be inside the draft's
  // attention window (older rows are >= 2048 positions behind this chunk's
  // end), and they fill the ring without slot collisions.
  const int r0 = ring && P > kDf2TapRows ? P - kDf2TapRows : 0;
  float* dst = ring ? df2.d_tap : df2.d_vtap;
  const dim3 grid((kDf2H + 255) / 256, P - r0);
  if (gr_bf16)
    k_df2_tap_capture<true><<<grid, 256, 0, g_str>>>(
        (const void*)d_Rb, dst, P, r0, base, t, ring ? kDf2TapRows : 0);
  else
    k_df2_tap_capture<false><<<grid, 256, 0, g_str>>>(
        (const void*)d_Rb, dst, P, r0, base, t, ring ? kDf2TapRows : 0);
}

// Called after an HC MLP write-back completes (prefix = "layers.<l>.mlp_...").
// Prompt chunks capture into the tap ring (cap_mode 1). Every speculative
// verify (spec_snap, any drafter, eager or graph-captured) captures its rows
// into vtap, so verify graphs recorded by one drafter serve the other and the
// DFlash2 context can follow MTP rounds.
void GpuModel::df2_hc_capture(const std::string& prefix, int P) {
  if (!df2.on || vz_rowtab_ != nullptr) return;
  const int mode = df2.cap_mode ? df2.cap_mode : (spec_snap ? 2 : 0);
  if (!mode) return;
  // Only the MLP site is a completed layer; the attention site writes the
  // mid-layer stream (same layer id in the prefix).
  if (prefix.find(".mlp_hyper_connection") == std::string::npos) return;
  if (prefix.compare(0, 7, "layers.") != 0) return;
  const int l = atoi(prefix.c_str() + 7);
  df2_capture_layer(l, P, vz_base_, mode);
}

// Prompt chunk [base, base+P): inject its last <= 2048 rows (see capture).
void GpuModel::df2_inject_chunk(int base, int P) {
  if (!df2.on || P <= 0) return;
  const int n0 = P > kDf2TapRows ? P - kDf2TapRows : 0;
  df2_inject(base + n0, P - n0, false);
}

void GpuModel::df2_inject(int base, int P, bool verify_rows) {
  if (!df2.on || P <= 0) return;
  // Feed prompt rows through the <= kDf2InjRows workspace in blocks.
  if (!verify_rows && P > kDf2InjRows) {
    for (int off = 0; off < P; off += kDf2InjRows)
      df2_inject(base + off, std::min(kDf2InjRows, P - off), false);
    return;
  }
  if (P > kDf2InjRows) {
    fprintf(stderr, "dflash2: inject P=%d exceeds %d\n", P, kDf2InjRows);
    return;
  }
  if (verify_rows) {
    // Keep the tap ring complete for committed generated rows too (same
    // [row][layer][H] layout, one contiguous row each).
    const size_t rb = (size_t)kDf2Layers * kDf2H * 4;
    for (int r = 0; r < P; r++)
      CK(hipMemcpyAsync(df2.d_tap + (size_t)((base + r) % kDf2TapRows) * kDf2Layers * kDf2H,
                        df2.d_vtap + (size_t)r * kDf2Layers * kDf2H, rb,
                        hipMemcpyDeviceToDevice, g_str));
  }
  const dim3 grid((kDf2H + 255) / 256, kDf2Layers, P);
  k_df2_gather_tap<<<grid, 256, 0, g_str>>>(
      verify_rows ? df2.d_vtap : df2.d_tap, df2.d_feat, base, P,
      verify_rows ? 1 : 0);
  gemm(df2.fc, df2.d_feat, df2.d_th, P, kDf2Feat);
  k_df2_rmsnorm<<<P, 256, 0, g_str>>>(df2.d_th, df2.d_enc_norm, df2.d_thn,
                                      kDf2H, df2.norm_eps);
  // One projection for all 5 layers' K|V: row = [l0 K | l0 V | l1 K | ...].
  const int rs = (int)df2.kv_all.rows;
  gemm(df2.kv_all, df2.d_thn, df2.d_kvn, P, kDf2H);
  for (int l = 0; l < kDf2Layers; l++) {
    const float* kn = df2.d_kvn + (size_t)l * 2 * kDf2NKV * kDf2HD;
    k_df2_inject_kv<<<dim3(kDf2NKV, P), kDf2HD, 0, g_str>>>(
        kn, kn + kDf2NKV * kDf2HD, df2.lw[l].d_k_norm,
        df2.d_kctx + (size_t)l * kDf2Window * kDf2NKV * kDf2HD,
        df2.d_vctx + (size_t)l * kDf2Window * kDf2NKV * kDf2HD, P, base,
        df2.norm_eps, rs);
  }
  if (base + P > df2.tap_hi) df2.tap_hi = base + P;
}

// Greedy selector chain walk over the GPU top-16 rows: anchor -> position 1
// predecessor; the chosen candidate becomes the next predecessor.
// conf ([npos][3]): full-vocab softmax prob of the chosen draft, selector
// softmax prob over the 16 candidate scores, unary top-1 prob.
static void df2_select_host(const float* topk, const float* selh,
                            const uint8_t* pred_cb, const uint8_t* succ_cb,
                            int npos, int rank, int anchor, int* drafts,
                            float* conf) {
  int pred = anchor;
  for (int j = 0; j < npos; j++) {
    const float* row = topk + (size_t)j * kDf2TopkRow;
    const int* ids = (const int*)row + kDf2TopK;
    const float lse = row[2 * kDf2TopK];
    const float* gate = selh + (size_t)j * rank;
    const uint16_t* pr = (const uint16_t*)pred_cb + (size_t)pred * rank;
    float pg[kDf2Rank];
    for (int r = 0; r < rank; r++) pg[r] = hgn::bf16_to_f32(pr[r]) * gate[r];
    float best = -INFINITY, sc[kDf2TopK];
    int best_k = 0;
    for (int k = 0; k < kDf2TopK; k++) {
      sc[k] = -INFINITY;
      const int cand = ids[k];
      if (cand < 0 || cand >= g_df2_vocab || !(row[k] > -INFINITY)) continue;
      const uint16_t* sr = (const uint16_t*)succ_cb + (size_t)cand * rank;
      float dot = 0.f;
      for (int r = 0; r < rank; r++) dot += pg[r] * hgn::bf16_to_f32(sr[r]);
      sc[k] = row[k] + dot;
      if (sc[k] > best) {
        best = sc[k];
        best_k = k;
      }
    }
    double se = 0;
    for (int k = 0; k < kDf2TopK; k++)
      if (sc[k] > -INFINITY) se += std::exp((double)sc[k] - best);
    conf[j * 3 + 0] = (float)std::exp((double)row[best_k] - lse);
    conf[j * 3 + 1] = (float)(1.0 / se);
    conf[j * 3 + 2] = (float)std::exp((double)row[0] - lse);
    drafts[j] = ids[best_k];
    pred = ids[best_k];
  }
}

void GpuModel::df2_propose(int anchor, int g, int* drafts, bool select) {
  // The draft was trained on full blocks: [anchor, mask x (block_size-1)].
  // Attention inside the block is bidirectional, so trailing mask rows change
  // the earlier positions' predictions; always run the full block (the
  // weight reads dominate, rows are nearly free) and select only g drafts.
  static const int blk_env = [] {
    const char* e = getenv("QWENOX_DF2_BLOCK");
    return e ? atoi(e) : 0;
  }();
  int B = blk_env > 0 ? std::min(blk_env, df2.block_size) : df2.block_size;
  if (B < g + 1) B = g + 1;
  df2.last_B = B;
  {
    int toks[kDf2Block];
    toks[0] = anchor;
    for (int i = 1; i < B; i++) toks[i] = df2.mask_id;
    CK(hipMemcpyAsync(df2.d_tok, toks, B * sizeof(int), hipMemcpyHostToDevice,
                      g_str));
  }
  static const bool prof = getenv("QWENOX_DF2_PROF") != nullptr;
  auto tnow = [] { return std::chrono::steady_clock::now(); };
  auto tms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto pt0 = tnow();
  if (prof) {
    CK(hipStreamSynchronize(g_str));  // previous round's injection
    auto t = tnow();
    df2.prof_ms[0] += tms(pt0, t);
    pt0 = t;
  }
  WView et = wview(ckpt.at("embed_tokens.weight"));
  if (df2.d_emb_bf16) {
    k_df2_embed_bf16<<<B, 256, 0, g_str>>>(df2.d_emb_bf16, df2.d_tok, df2.d_x, B);
  } else if (et.dtype == 5 || et.dtype == 8) {
    embed_rows(et, df2.d_tok, B, df2.d_x, kDf2H);
  } else if (et.dtype == 0) {
    k_df2_embed_bf16<<<B, 256, 0, g_str>>>((const uint16_t*)et.data, df2.d_tok,
                                          df2.d_x, B);
  } else {
    throw std::runtime_error("dflash2: unsupported embed dtype " +
                             std::to_string(et.dtype));
  }

  const int base = pos;
  const int nkeys = [&] {
    const int lo = std::max(0, base - (df2.window - 1));
    const int hi = std::min(base, df2.tap_hi);
    return std::max(0, hi - lo) + B;
  }();
  const dim3 cgrid((kDf2H + 255) / 256, B);
  float* const dq = df2.d_qkv;
  float* const dk = df2.d_qkv + kDf2NH * kDf2HD;
  float* const dv = dk + kDf2NKV * kDf2HD;
  // QWENOX_DF2_SKIP=gemm|other|attn (timing experiments only: output invalid).
  static const int skip = [] {
    const char* e = getenv("QWENOX_DF2_SKIP");
    if (!e) return 0;
    return !strcmp(e, "gemm") ? 1 : !strcmp(e, "other") ? 2 : !strcmp(e, "attn") ? 3 : 0;
  }();
  auto gemm = [&](const WView& w, const float* x, float* y, int P, uint64_t xs) {
    if (skip != 1) this->gemm(w, x, y, P, xs);
  };
  for (int l = 0; l < kDf2Layers && skip != 4; l++) {
    const Df2LayerW& w = df2.lw[l];
    if (skip == 2) {  // only the GEMVs
      gemm(w.attn_conv_proj, df2.d_norm, df2.d_dyn, B, kDf2H);
      gemm(w.qkv, df2.d_conv, df2.d_qkv, B, kDf2H);
      gemm(w.o, df2.d_attn, df2.d_o, B, kDf2NH * kDf2HD);
      gemm(w.ffn_conv_proj, df2.d_norm, df2.d_dyn, B, kDf2H);
      gemm(w.gu, df2.d_conv, df2.d_gu, B, kDf2H);
      gemm(w.down, df2.d_ff, df2.d_o, B, kDf2FF);
      continue;
    }
    // attention sublayer: norm -> conv(side 0) -> attn -> conv(side 1) + x
    k_df2_rmsnorm<<<B, 256, 0, g_str>>>(df2.d_x, w.d_attn_norm, df2.d_norm,
                                        kDf2H, df2.norm_eps);
    gemm(w.attn_conv_proj, df2.d_norm, df2.d_dyn, B, kDf2H);
    k_df2_conv<<<cgrid, 256, 0, g_str>>>(df2.d_norm, df2.d_dyn, w.d_attn_base,
                                         df2.d_conv, B, 0);
    gemm(w.qkv, df2.d_conv, df2.d_qkv, B, kDf2H);
    k_df2_qk_norm_rope<<<dim3(kDf2NH + kDf2NKV, B), kDf2HD, 0, g_str>>>(
        dq, dk, w.d_q_norm, w.d_k_norm, B, base, df2.norm_eps, kDf2QKV);
    if (skip != 3) {
      const float* kc = df2.d_kctx + (size_t)l * kDf2Window * kDf2NKV * kDf2HD;
      const float* vc = df2.d_vctx + (size_t)l * kDf2Window * kDf2NKV * kDf2HD;
      if (nkeys > df2.attn_split_min) {
        const int ns = (nkeys + kDf2AttnS - 1) / kDf2AttnS;
        k_df2_attn_part<<<dim3(kDf2NKV, ns), 256, 0, g_str>>>(
            dq, dk, dv, kDf2QKV, kc, vc, df2.d_apart, B, base, df2.tap_hi,
            df2.window);
        k_df2_attn_combine<<<dim3(kDf2NKV, B * (kDf2NH / kDf2NKV)), kDf2HD, 0, g_str>>>(
            df2.d_apart, df2.d_attn, ns);
      } else {
        k_df2_attn<<<dim3(kDf2NKV, B), 128, 0, g_str>>>(
            dq, dk, dv, kDf2QKV, kc, vc, df2.d_attn, B, base, df2.tap_hi,
            df2.window);
      }
    }
    gemm(w.o, df2.d_attn, df2.d_o, B, kDf2NH * kDf2HD);
    k_df2_conv_add<<<cgrid, 256, 0, g_str>>>(df2.d_o, df2.d_dyn, w.d_attn_base,
                                             df2.d_x, B);
    // FFN sublayer
    k_df2_rmsnorm<<<B, 256, 0, g_str>>>(df2.d_x, w.d_ffn_norm, df2.d_norm,
                                        kDf2H, df2.norm_eps);
    gemm(w.ffn_conv_proj, df2.d_norm, df2.d_dyn, B, kDf2H);
    k_df2_conv<<<cgrid, 256, 0, g_str>>>(df2.d_norm, df2.d_dyn, w.d_ffn_base,
                                         df2.d_conv, B, 0);
    gemm(w.gu, df2.d_conv, df2.d_gu, B, kDf2H);
    k_df2_silu_mul<<<(B * kDf2FF + 255) / 256, 256, 0, g_str>>>(df2.d_gu, df2.d_ff, B);
    gemm(w.down, df2.d_ff, df2.d_o, B, kDf2FF);
    k_df2_conv_add<<<cgrid, 256, 0, g_str>>>(df2.d_o, df2.d_dyn, w.d_ffn_base,
                                             df2.d_x, B);
  }
  k_df2_rmsnorm<<<B, 256, 0, g_str>>>(df2.d_x, df2.d_out_norm, df2.d_out,
                                      kDf2H, df2.norm_eps);
  if (prof) {
    CK(hipStreamSynchronize(g_str));
    auto t = tnow();
    df2.prof_ms[1] += tms(pt0, t);
    pt0 = t;
  }
  const int npos = g;
  gemm(df2.sel_hidden, df2.d_out + kDf2H, df2.d_selh, npos, kDf2H);
  const WView lm = df2.lm_bf16_on ? df2.lm_bf16
                   : dlm_on       ? w_dlm
                                  : wview(ckpt.at("lm_head.weight"));
  gemm(lm, df2.d_out + kDf2H, d_logits_b, npos, kDf2H);
  k_df2_topk_part<<<dim3(npos, kDf2TopkChunks), 256, 0, g_str>>>(d_logits_b, g_cfg.vocab,
                                                                 df2.d_topk_part);
  k_df2_topk_merge<<<npos, kDf2TopkChunks, 0, g_str>>>(df2.d_topk_part, df2.d_topk);
  CK(hipMemcpyAsync(df2.h_topk, df2.d_topk, (size_t)npos * kDf2TopkRow * 4,
                    hipMemcpyDeviceToHost, g_str));
  CK(hipMemcpyAsync(df2.h_selh, df2.d_selh, (size_t)npos * kDf2Rank * 4,
                    hipMemcpyDeviceToHost, g_str));
  CK(hipStreamSynchronize(g_str));
  if (prof) {
    auto t = tnow();
    df2.prof_ms[2] += tms(pt0, t);
    pt0 = t;
    static bool said = false;
    if (!said) {
      said = true;
      fprintf(stderr, "dflash2: lm_head dtype %d %llux%llu\n", lm.dtype,
              (unsigned long long)lm.rows, (unsigned long long)lm.cols);
    }
  }
  if (select)
    df2_select_host(df2.h_topk, df2.h_selh, df2.sel_pred_host, df2.sel_succ_host,
                    npos, kDf2Rank, anchor, drafts, df2.conf);
  if (prof) {
    df2.prof_ms[3] += tms(pt0, tnow());
    df2.prof_n++;
  }
}

// QWENOX_DF2_RDUMP=<dir> + QWENOX_DF2_RDUMP_ROUNDS=1,5,...: at those rounds
// write r<N>.meta (base anchor g tap_hi B), r<N>.tap (whole tap ring f32),
// r<N>.out (post-norm block hidden [B][H]) and r<N>.drafts (int32 [g]) for the
// upstream-model reference (df2bench/up_ref.py). Debug only.
void GpuModel::df2_round_dump(int round, int anchor, int g, const int* drafts) {
  static const char* dir = getenv("QWENOX_DF2_RDUMP");
  if (!dir || !*dir) return;
  static const std::vector<int> want = [] {
    std::vector<int> v;
    const char* s = getenv("QWENOX_DF2_RDUMP_ROUNDS");
    std::string t = s ? s : "1";
    size_t i = 0;
    while (i < t.size()) {
      size_t j = t.find(',', i);
      if (j == std::string::npos) j = t.size();
      v.push_back(atoi(t.substr(i, j - i).c_str()));
      i = j + 1;
    }
    return v;
  }();
  if (std::find(want.begin(), want.end(), round) == want.end()) return;
  char p[512];
  snprintf(p, sizeof p, "%s/r%d.meta", dir, round);
  if (FILE* f = fopen(p, "w")) {
    fprintf(f, "%d %d %d %d %d\n", pos, anchor, g, df2.tap_hi, df2.last_B);
    fclose(f);
  }
  auto bin = [&](const char* ext, const void* d, size_t bytes, bool dev) {
    snprintf(p, sizeof p, "%s/r%d.%s", dir, round, ext);
    std::vector<uint8_t> h(bytes);
    if (dev) {
      CK(hipStreamSynchronize(g_str));
      CK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost));
    } else {
      memcpy(h.data(), d, bytes);
    }
    if (FILE* f = fopen(p, "wb")) {
      fwrite(h.data(), 1, bytes, f);
      fclose(f);
    }
  };
  bin("tap", df2.d_tap, (size_t)kDf2TapRows * kDf2Layers * kDf2H * 4, true);
  bin("out", df2.d_out, (size_t)df2.last_B * kDf2H * 4, true);
  bin("drafts", drafts, (size_t)g * 4, false);
  fprintf(stderr, "dflash2: round dump r%d (base %d anchor %d g %d)\n", round, pos,
          anchor, g);
}

// QWENOX_DF2_RINGDUMP=<dir>: write the draft K/V rings to <dir>/<tag>_<pos>.ring
// (int32 pos, int32 tap_hi, then kctx and vctx f32). Debug only: compares
// verify-injected context against a fresh prefill of the same tokens.
static void df2_ring_dump(const Df2State& s, int pos, const char* tag) {
  const char* dir = getenv("QWENOX_DF2_RINGDUMP");
  if (!dir || !*dir) return;
  const size_t n = (size_t)kDf2Layers * kDf2Window * kDf2NKV * kDf2HD;
  std::vector<float> h(n);
  char p[512];
  snprintf(p, sizeof p, "%s/%s_%d.ring", dir, tag, pos);
  FILE* f = fopen(p, "wb");
  if (!f) return;
  CK(hipStreamSynchronize(g_str));
  int hdr[2] = {pos, s.tap_hi};
  fwrite(hdr, sizeof hdr, 1, f);
  CK(hipMemcpy(h.data(), s.d_kctx, n * 4, hipMemcpyDeviceToHost));
  fwrite(h.data(), 4, n, f);
  CK(hipMemcpy(h.data(), s.d_vctx, n * 4, hipMemcpyDeviceToHost));
  fwrite(h.data(), 4, n, f);
  fclose(f);
  fprintf(stderr, "dflash2: ring dump %s\n", p);
}

// Greedy DFlash2 spec loop. Every round drafts a full block (up to 7 drafts),
// then verifies only the confident prefix: the chain stops at the first
// position where the running product of selector probabilities (softmax over
// the 16 candidate scores) drops below conf_th (QWENOX_DF2_CONF, default
// 0.5; 0 = always verify gamma). The draft costs the same for any gamma but
// each verified row costs ~4.5 ms, so low-confidence tails are not worth
// verifying (df2bench/sim.py replay of logged rounds).
//
// hybrid (MTP state live on entry: mtp_pos == pos-1, d_R = h_{pos-1}): each
// round drafts with DFlash2 or MTP, whichever has the higher recent
// committed-tokens-per-ms (EWMA over its own rounds, the other arm re-tried
// when it has not run for kDf2HybExplore rounds). Both drafters stay live:
// DFlash2 rounds patch the MTP KV over the committed positions (as the chain
// loop does for ngram rounds), MTP rounds feed the DFlash2 context from the
// verify taps. On return the MTP invariant still holds.
std::vector<int> GpuModel::df2_spec_loop(
    int cur, int gen, int gamma, const std::function<bool(int)>& emit,
    bool hybrid, int mtp_gamma) {
  df2_ring_dump(df2, pos, "start");
  struct EndDump {
    GpuModel& m;
    ~EndDump() { df2_ring_dump(m.df2, m.pos, "end"); }
  } end_dump{*this};
  last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
  live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
  std::vector<int> out;
  out.reserve((size_t)std::max(0, gen) + 1);
  // QWENOX_DF2_GAMMA overrides; otherwise confidence truncation picks the
  // verified length out of a full block (gamma = block_size - 1).
  static const int g_env = [] {
    const char* e = getenv("QWENOX_DF2_GAMMA");
    return e ? atoi(e) : 0;
  }();
  gamma = g_env > 0 ? g_env : (df2.conf_th > 0.f ? df2.block_size - 1 : gamma);
  gamma = std::max(1, std::min(gamma, df2.block_size - 1));
  hybrid = hybrid && mtp_avail;
  GammaCtl gm(mtp_gamma, false);  // MTP rounds (hybrid)
  struct Arm {
    double tok = 0, ms = 0;
    int n = 0, last = -1000000;
  } arm[2];  // 0 = DFlash2, 1 = MTP
  int arm_rounds[2] = {0, 0}, arm_tok[2] = {0, 0};
  double arm_ms[2] = {0, 0};
  static const double hyb_alpha = [] {
    const char* e = getenv("QWENOX_DF2_HYB_ALPHA");
    return e ? atof(e) : 0.2;
  }();
  static const int hyb_explore = [] {
    const char* e = getenv("QWENOX_DF2_HYB_EXPLORE");
    return e ? atoi(e) : 16;
  }();
  static const int hyb_force = [] {  // debug: -1 adaptive, 0 DF2, 1 MTP, 2 alternate
    const char* e = getenv("QWENOX_DF2_HYB_FORCE");
    return e ? atoi(e) : -1;
  }();
  std::vector<int> drafts(kDf2Block), v(kDf2Block + 1);
  int depth_hits[kDf2Block] = {0}, depth_tot[kDf2Block] = {0};
  int mdepth_hits[kDf2Block] = {0}, mdepth_tot[kDf2Block] = {0};
  int glen_hist[kDf2Block + 1] = {0};
  int rounds = 0, rollbacks = 0;
  double t_draft = 0, t_verify = 0, t_roll = 0;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto T0 = now();
  // QWENOX_DF2_DENSE=<file> (debug, hybrid only): commit exactly one token per
  // round while recording BOTH drafters' full proposals at every position:
  // "pos cur | 7 DF2 drafts | 7 x (p_full q_sel p_top1) | 7 MTP drafts |
  // 7 MTP top-1 probs". df2bench/dense_sim.py scores them against the final
  // greedy output and replays drafter-selection policies.
  static FILE* dense_f = [] {
    const char* p = getenv("QWENOX_DF2_DENSE");
    return p && *p ? fopen(p, "a") : nullptr;
  }();
  if (dense_f && hybrid) {
    std::vector<float> L;
    while ((int)out.size() < gen) {
      const int q = pos - 1;
      mtp_ckpt_save();
      int md[kDf2Block - 1];
      float mp[kDf2Block - 1];
      for (int j = 0; j < kDf2Block - 1; j++) {
        mtp_step(j ? d_mR : d_R, j ? md[j - 1] : cur, &L);
        if (!j)
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        int am = 0;
        float mx = L[0];
        for (int i = 1; i < g_cfg.vocab; i++)
          if (L[i] > mx) mx = L[i], am = i;
        double se = 0;
        for (int i = 0; i < g_cfg.vocab; i++) se += std::exp((double)L[i] - mx);
        md[j] = am;
        mp[j] = (float)(1.0 / se);
      }
      // MTP repair for a depth-0 commit (spec_loop): ring back to post-step-1,
      // position q (written by step 1) stays, mtp_pos = q + 1.
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4, hipMemcpyDeviceToDevice, g_str));
      mtp_pos = q + 1;
      CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      df2_propose(cur, kDf2Block - 1, drafts.data());
      fprintf(dense_f, "%d %d |", q + 1, cur);
      for (int j = 0; j < kDf2Block - 1; j++) fprintf(dense_f, " %d", drafts[j]);
      fprintf(dense_f, " |");
      for (int j = 0; j < (kDf2Block - 1) * 3; j++) fprintf(dense_f, " %.4f", df2.conf[j]);
      fprintf(dense_f, " |");
      for (int j = 0; j < kDf2Block - 1; j++) fprintf(dense_f, " %d", md[j]);
      fprintf(dense_f, " |");
      for (int j = 0; j < kDf2Block - 1; j++) fprintf(dense_f, " %.4f", mp[j]);
      fputc('\n', dense_f);
      // Advance by one: decode cur, next = argmax.
      std::vector<int> vt{cur};
      spec_snap = true;
      vsnap_row0_ = 0;
      df2.cap_mode = 0;
      prefill_batch(vt, q + 1, false, true);
      batch_argmax(1, v.data());
      spec_snap = false;
      gdn_commit(1);
      df2_inject(q + 1, 1, true);
      rounds++;
      const bool ok = !emit || emit(v[0]);
      if (!ok) break;
      out.push_back(v[0]);
      cur = v[0];
    }
    fprintf(dense_f, "END");
    for (int t : out) fprintf(dense_f, " %d", t);
    fputc('\n', dense_f);
    fflush(dense_f);
    last_spec_rounds = rounds;
    last_spec_commit = (int)out.size();
    return out;
  }
  while ((int)out.size() < gen) {
    yield_point();
    const int q = pos - 1;  // trunk processed through q
    const int gp = std::max(1, std::min(gamma, gen - (int)out.size()));
    size_t chist_sz = chist.size();
    const size_t out0 = out.size();
    mtp_ckpt_save();
    auto t0 = now();
    // ---- pick the drafter ----
    int k = 0;
    if (hybrid) {
      if (hyb_force == 2) k = rounds & 1;  // debug: strict alternation
      else if (hyb_force >= 0) k = hyb_force ? 1 : 0;
      else if (arm[0].n < 2 || arm[1].n < 2) k = arm[0].n <= arm[1].n ? 0 : 1;
      else if (rounds - arm[0].last >= hyb_explore) k = 0;
      else if (rounds - arm[1].last >= hyb_explore) k = 1;
      else k = arm[1].tok * arm[0].ms > arm[0].tok * arm[1].ms ? 1 : 0;
    }
    int g;
    if (k == 0) {
      if (hybrid)  // stash h_q: MTP position q pairs it with cur after commit
        CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice, g_str));
      df2_propose(cur, gp, drafts.data());
      df2_round_dump(rounds + 1, cur, gp, drafts.data());
      g = gp;
      if (df2.conf_th > 0.f) {
        float cum = 1.f;
        g = 0;
        for (int j = 0; j < gp; j++) {
          cum *= df2.conf[j * 3 + 1];
          if (cum < df2.conf_th) break;
          g++;
        }
        g = std::max(1, g);
      }
      glen_hist[g]++;
    } else {
      // MTP draft burst (spec_loop): step 1 pairs the committed (h_q, cur);
      // its indexer ring rows are truth, later steps pollute -> checkpoint.
      g = std::max(1, std::min({gm.cur, gen - (int)out.size(), kDf2Block - 1}));
      if (draft_desync) {
        mtp_step(d_R, cur, nullptr, true, nullptr, d_argmax_arr);
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        for (int j = 1; j < g; j++)
          mtp_step(d_mR, 0, nullptr, true, d_argmax_arr + j - 1, d_argmax_arr + j);
        CK(hipMemcpy(drafts.data(), d_argmax_arr, (size_t)g * sizeof(int),
                     hipMemcpyDeviceToHost));
      } else {
        drafts[0] = mtp_step(d_R, cur);
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        for (int j = 1; j < g; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
      }
    }
    auto t1 = now();
    std::vector<int> vt;
    vt.reserve((size_t)g + 1);
    vt.push_back(cur);
    for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
    spec_snap = true;
    vsnap_row0_ = 0;
    df2.cap_mode = 0;  // verify rows -> vtap (spec_snap)
    if (!verify_graph_round_try(vt, q + 1, v.data())) {
      prefill_batch(vt, q + 1, false, true);
      batch_argmax(g + 1, v.data());
    }
    spec_snap = false;
    if (!spec_force.empty())
      for (int j = 0; j <= g; j++)
        if (out.size() + (size_t)j < spec_force.size()) v[j] = spec_force[out.size() + j];
    auto t2 = now();
    int a = 0;
    while (a < g && drafts[a] == v[a]) a++;
    {
      // QWENOX_DF2_TRACE=<file>: "base gp a draft_ms verify_ms g" + 3 conf/draft
      // (gp = drafted, g = verified, a = accepted of g).
      static FILE* tf = [] {
        const char* p = getenv("QWENOX_DF2_TRACE");
        return p && *p ? fopen(p, "a") : nullptr;
      }();
      if (tf && k == 0) {
        fprintf(tf, "%d %d %d %.3f %.3f", q + 1, g, a, ms(t0, t1), ms(t1, t2));
        for (int j = 0; j < g; j++)
          fprintf(tf, " %.4f %.4f %.4f", df2.conf[j * 3], df2.conf[j * 3 + 1],
                  df2.conf[j * 3 + 2]);
        fputc('\n', tf);
        fflush(tf);
      }
    }
    if (k == 0)
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
    else {
      gm.update(g, a);
      for (int j = 0; j < g && j < kDf2Block; j++) {
        mdepth_tot[j]++;
        if (j < a) mdepth_hits[j]++;
      }
    }
    rounds++;
    live_spec_rounds = rounds;
    live_spec_proposed += g;
    live_spec_commits = (int)out.size();
    int d = a;
    bool stop = false, keep_v = true;
    if (emit) {
      for (int j = 0; j < a; j++)
        if (!emit(drafts[j])) {
          d = j;
          stop = true;
          keep_v = false;
          break;
        }
      if (!stop && !emit(v[a])) {
        stop = true;
        keep_v = false;
      }
    }
    for (int j = 0; j < d; j++) out.push_back(drafts[j]);
    if (keep_v) out.push_back(v[a]);
    auto t3 = now();
    if (!stop && d == g) {
      gdn_commit(g + 1);
      cur = v[g];
    } else {
      rollbacks++;
      const int ngdn = g_cfg.layers - g_cfg.layers / 4;
      gdn_commit(d + 1);
      CK(hipMemcpyAsync(d_convst,
                        d_csnap + (size_t)(vsnap_row0_ + d) * ngdn * 3 * 10240,
                        (size_t)ngdn * 3 * 10240 * 4, hipMemcpyDeviceToDevice,
                        g_str));
      if (ckpt_iraw)
        k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
            d_iproj_snap + (size_t)vsnap_row0_ * 128,
            d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4, d_iraw, q + 1,
            q + d + 1, kVSnapRows);
      const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
      for (int p = q + d + 2; p <= q + g + 1; p++)
        CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                          ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      chist.resize(chist_sz + d + 1);
      pos = q + d + 2;
      ple_ringpos = pos % 9;
      CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
      CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                        g_str));
      restore_dR_row(d, vsnap_row0_);
      cur = v[d];
    }
    if (k == 1) {
      // MTP self-repair over committed positions q+1..q+d (spec_loop).
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4, hipMemcpyDeviceToDevice,
                        g_str));
      if (d > 0) {
        std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
        mtp_ingest_b(spec_taps(d, vsnap_row0_), mtoks, d, q + 1);
      } else {
        mtp_pos = q + 1;
        CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      }
    } else if (hybrid) {
      // Patch the MTP KV over q..q+d (spec_loop_chain's ngram rounds): row 0
      // = stashed h_q with cur, rows 1..d = verify rows 0..d-1 with the
      // committed drafts. Restores mtp_pos = pos-1.
      if (d > 0)
        CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(d, vsnap_row0_),
                          (size_t)d * 10240 * 4, hipMemcpyDeviceToDevice, g_str));
      mtp_ingest_b(d_mtap, std::vector<int>(vt.begin(), vt.begin() + d + 1), d + 1, q);
    }
    // Committed target rows 0..d (positions q+1 .. q+d+1): context features.
    df2_inject(q + 1, d + 1, /*verify_rows=*/true);
    if (hybrid) {
      // Charge the whole round (incl. the other drafter's upkeep) to arm k.
      CK(hipStreamSynchronize(g_str));
      const double rms = ms(t0, now());
      const int tk = (int)(out.size() - out0);
      Arm& A = arm[k];
      if (A.n == 0) {
        A.tok = tk;
        A.ms = rms;
      } else {
        A.tok += hyb_alpha * (tk - A.tok);
        A.ms += hyb_alpha * (rms - A.ms);
      }
      A.n++;
      A.last = rounds;
      arm_rounds[k]++;
      arm_tok[k] += tk;
      arm_ms[k] += rms;
      static const bool htr = getenv("QWENOX_DF2_HYB_TRACE") != nullptr;
      if (htr)
        fprintf(stderr, "hyb r%d k=%d g=%d tok=%d %.1fms | est df2 %.2f/%.1f mtp %.2f/%.1f\n",
                rounds, k, g, tk, rms, arm[0].tok, arm[0].ms, arm[1].tok, arm[1].ms);
    }
    auto t5 = now();
    t_draft += ms(t0, t1);
    t_verify += ms(t1, t2);
    t_roll += ms(t3, t5);
    if (stop) break;
  }
  double sec = std::chrono::duration<double>(now() - T0).count();
  last_spec_rounds = rounds;
  last_spec_commit = (int)out.size();
  last_spec_proposed = live_spec_proposed;
  last_spec_rollbacks = rollbacks;
  if (!bench_quiet) {
    fprintf(stderr,
            "\ndflash2 spec: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "commit/round=%.2f rollbacks=%d\n",
            (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0, rounds,
            rounds ? (double)out.size() / rounds : 0.0, rollbacks);
    fprintf(stderr, "dflash2 depth acc:");
    for (int j = 0; j < kDf2Block && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr, "\ndflash2 verified gamma (conf %.2f):", df2.conf_th);
    for (int j = 1; j <= kDf2Block; j++)
      if (glen_hist[j]) fprintf(stderr, " %d:%d", j, glen_hist[j]);
    fprintf(stderr, "\ndflash2 time: draft=%.0fms verify=%.0fms rollback=%.0fms\n",
            t_draft, t_verify, t_roll);
    if (hybrid)
      fprintf(stderr,
              "dflash2 hybrid: dflash2 %d rounds %d tok %.1f ms/tok | mtp %d rounds "
              "%d tok %.1f ms/tok\n",
              arm_rounds[0], arm_tok[0], arm_tok[0] ? arm_ms[0] / arm_tok[0] : 0.0,
              arm_rounds[1], arm_tok[1], arm_tok[1] ? arm_ms[1] / arm_tok[1] : 0.0);
    if (hybrid && mdepth_tot[0]) {
      fprintf(stderr, "dflash2 hybrid mtp depth acc:");
      for (int j = 0; j < kDf2Block && mdepth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1, (double)mdepth_hits[j] / mdepth_tot[j]);
      fputc('\n', stderr);
    }
    if (df2.prof_n)
      fprintf(stderr,
              "dflash2 prof (ms/round, %d rounds): inject %.2f blocks %.2f "
              "lm_head+topk %.2f select %.2f | verify %.2f\n",
              df2.prof_n, df2.prof_ms[0] / df2.prof_n, df2.prof_ms[1] / df2.prof_n,
              df2.prof_ms[2] / df2.prof_n, df2.prof_ms[3] / df2.prof_n,
              t_verify / std::max(1, rounds));
  }
  for (double& x : df2.prof_ms) x = 0;
  df2.prof_n = 0;
  return out;
}

// Sampled DFlash2 (upstream dflash_generate with temperature > 0): position
// j's draft distribution is q_j = softmax(selector scores / T) over its 16
// candidates, the chain is SAMPLED (the drawn candidate is the next
// predecessor), and the target accepts draft j with min(1, p_j(x)/q_j(x));
// a rejection draws from normalized max(p - q, 0), full acceptance draws the
// verify continuation — the exact spec_loop_sample rules, so the output is a
// sample of the target distribution. The verified length stops where the
// running product of q_j(x_j) drops below QWENOX_DF2_SCONF (default 0.3).
template <typename Sampler>
std::vector<int> GpuModel::df2_spec_loop_sample(
    int cur, int gen, Sampler& sampler,
    const std::function<bool(int, float)>& emit) {
  last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
  live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
  std::vector<int> out;
  out.reserve((size_t)std::max(0, gen) + 1);
  static const float sconf = [] {
    const char* e = getenv("QWENOX_DF2_SCONF");
    return e ? (float)atof(e) : 0.3f;
  }();
  static const int g_env = [] {
    const char* e = getenv("QWENOX_DF2_GAMMA");
    return e ? atoi(e) : 0;
  }();
  const int gamma = std::max(1, std::min(g_env > 0 ? g_env : df2.block_size - 1,
                                         df2.block_size - 1));
  const float T = sampler.temp > 0.f ? sampler.temp : 1.f;
  std::vector<int> drafts(kDf2Block);
  std::vector<float> draft_lps(kDf2Block), verify_logits, p_row(g_cfg.vocab), q_row;
  std::vector<typename Sampler::SDist> q_sd(kDf2Block);
  int depth_hits[kDf2Block] = {0}, depth_tot[kDf2Block] = {0};
  int glen_hist[kDf2Block + 1] = {0};
  int rounds = 0, rollbacks = 0;
  double t_draft = 0, t_verify = 0, t_roll = 0;
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto T0 = now();
  while ((int)out.size() < gen) {
    yield_point();
    const int q = pos - 1;
    const int gp = std::max(1, std::min(gamma, gen - (int)out.size()));
    size_t chist_sz = chist.size();
    mtp_ckpt_save();
    Sampler draft_sampler = sampler.fork_stream(0x6a09e667f3bcc909ull ^ (uint64_t)(q + 1));
    Sampler accept_sampler = sampler.fork_stream(0xbb67ae8584caa73bull ^ (uint64_t)(q + 1));
    auto t0 = now();
    df2_propose(cur, gp, drafts.data(), false);
    // Sampled selector chain.
    int g = 0;
    {
      int pred = cur;
      float cum = 1.f;
      float pg[kDf2Rank];
      for (int j = 0; j < gp; j++) {
        const float* row = df2.h_topk + (size_t)j * kDf2TopkRow;
        const int* ids = (const int*)row + kDf2TopK;
        const float* gate = df2.h_selh + (size_t)j * kDf2Rank;
        const uint16_t* pr = (const uint16_t*)df2.sel_pred_host + (size_t)pred * kDf2Rank;
        for (int r = 0; r < kDf2Rank; r++) pg[r] = hgn::bf16_to_f32(pr[r]) * gate[r];
        float sc[kDf2TopK], mx = -INFINITY;
        int nk = 0;
        for (int k = 0; k < kDf2TopK; k++) {
          const int cand = ids[k];
          if (cand < 0 || cand >= g_df2_vocab || !(row[k] > -INFINITY)) break;
          const uint16_t* sr = (const uint16_t*)df2.sel_succ_host + (size_t)cand * kDf2Rank;
          float dot = 0.f;
          for (int r = 0; r < kDf2Rank; r++) dot += pg[r] * hgn::bf16_to_f32(sr[r]);
          sc[k] = (row[k] + dot) / T;
          mx = std::max(mx, sc[k]);
          nk++;
        }
        auto& sd = q_sd[j];
        sd.ids.assign(ids, ids + nk);
        sd.pr.resize(nk);
        double z = 0;
        for (int k = 0; k < nk; k++) z += std::exp((double)sc[k] - mx);
        for (int k = 0; k < nk; k++) sd.pr[k] = (float)(std::exp((double)sc[k] - mx) / z);
        float qlp = 0.f;
        drafts[j] = draft_sampler.draw_sparse(sd, &qlp, false);
        pred = drafts[j];
        g++;
        cum *= std::exp(qlp);
        if (sconf > 0.f && cum < sconf) break;
      }
    }
    glen_hist[g]++;
    auto t1 = now();
    std::vector<int> vt;
    vt.reserve((size_t)g + 1);
    vt.push_back(cur);
    for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
    spec_snap = true;
    vsnap_row0_ = 0;
    df2.cap_mode = 0;
    prefill_batch(vt, q + 1, true, true);
    spec_snap = false;
    const bool smg = sms_gpu_active(sampler) && sms_round_shared(sampler);
    bool smg_v = false;
    if (smg) {
      const bool pfx = sms_prefix_upload(sampler, drafts.data(), 0, g + 1);
      batch_logits_device(g + 1);
      smg_v = sms_topk_rows(d_logits_b, g_cfg.vocab, g + 1, sampler.top_k, pfx,
                            sms_out_ids.data(), sms_out_vals.data());
    }
    if (!smg_v) batch_logits(g + 1, verify_logits);
    Sampler target_view = sampler;
    std::vector<typename Sampler::SDist> p_sd(smg_v ? g + 1 : 0);
    int a = 0;
    for (; a < g; a++) {
      float px, qx = Sampler::sdist_prob(q_sd[a], drafts[a]);
      if (smg_v) {
        sms_prep(target_view, a, false, p_sd[a]);
        px = Sampler::sdist_prob(p_sd[a], drafts[a]);
      } else {
        std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab, g_cfg.vocab,
                    p_row.data());
        target_view.prepare(p_row);
        px = p_row[drafts[a]];
      }
      draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
      const double th = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
      if (accept_sampler.uniform() >= th) break;
      target_view.remember(drafts[a]);
    }
    int next;
    float next_lp = 0.0f;
    target_view.rng = sampler.rng;
    if (a < g) {
      if (smg_v) {
        next = target_view.sample_residual_sparse(p_sd[a], q_sd[a], &next_lp);
      } else {
        q_row.assign(g_cfg.vocab, 0.0f);
        for (size_t u = 0; u < q_sd[a].ids.size(); u++) q_row[q_sd[a].ids[u]] = q_sd[a].pr[u];
        next = target_view.sample_residual(p_row, q_row, &next_lp);
      }
    } else if (smg_v) {
      sms_prep(target_view, g, false, p_sd[g]);
      next = target_view.draw_sparse(p_sd[g], &next_lp, false);
    } else {
      std::copy_n(verify_logits.data() + (size_t)g * g_cfg.vocab, g_cfg.vocab,
                  p_row.data());
      target_view.prepare(p_row);
      next = target_view.draw_prepared(p_row, &next_lp, false);
    }
    sampler.rng = target_view.rng;
    auto t2 = now();
    for (int j = 0; j < g; j++) {
      depth_tot[j]++;
      if (j < a) depth_hits[j]++;
    }
    rounds++;
    live_spec_rounds = rounds;
    live_spec_proposed += g;
    live_spec_commits = (int)out.size();
    int d = a, stopped_token = -1;
    bool stop = false, keep_next = true;
    if (emit) {
      for (int j = 0; j < a; j++)
        if (!emit(drafts[j], draft_lps[j])) {
          d = j;
          stopped_token = drafts[j];
          stop = true;
          keep_next = false;
          break;
        }
      if (!stop && !emit(next, next_lp)) {
        stop = true;
        keep_next = false;
      }
    }
    for (int j = 0; j < d; j++) {
      out.push_back(drafts[j]);
      sampler.remember(drafts[j]);
    }
    if (keep_next) {
      out.push_back(next);
      sampler.remember(next);
    }
    auto t3 = now();
    if (!stop && d == g) {
      gdn_commit(g + 1);
      cur = next;
    } else {
      rollbacks++;
      const int ngdn = g_cfg.layers - g_cfg.layers / 4;
      gdn_commit(d + 1);
      CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)(vsnap_row0_ + d) * ngdn * 3 * 10240,
                        (size_t)ngdn * 3 * 10240 * 4, hipMemcpyDeviceToDevice, g_str));
      if (ckpt_iraw)
        k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
            d_iproj_snap + (size_t)vsnap_row0_ * 128,
            d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4, d_iraw, q + 1, q + d + 1,
            kVSnapRows);
      const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
      for (int p = q + d + 2; p <= q + g + 1; p++)
        CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240, ckring + (size_t)(p % 9) * 10240,
                          10240 * 4, hipMemcpyDeviceToDevice, g_str));
      chist.resize(chist_sz + d + 1);
      pos = q + d + 2;
      ple_ringpos = pos % 9;
      CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
      CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
      restore_dR_row(d);
      cur = stopped_token >= 0 ? stopped_token : next;
    }
    df2_inject(q + 1, d + 1, /*verify_rows=*/true);
    auto t5 = now();
    t_draft += ms(t0, t1);
    t_verify += ms(t1, t2);
    t_roll += ms(t3, t5);
    if (stop) break;
  }
  double sec = std::chrono::duration<double>(now() - T0).count();
  last_spec_rounds = rounds;
  last_spec_commit = (int)out.size();
  last_spec_proposed = live_spec_proposed;
  last_spec_rollbacks = rollbacks;
  if (!bench_quiet) {
    fprintf(stderr,
            "\ndflash2 spec-sample: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "commit/round=%.2f rollbacks=%d\n",
            (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0, rounds,
            rounds ? (double)out.size() / rounds : 0.0, rollbacks);
    fprintf(stderr, "dflash2 spec-sample depth acc:");
    for (int j = 0; j < kDf2Block && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr, "\ndflash2 spec-sample verified gamma (sconf %.2f):", sconf);
    for (int j = 1; j <= kDf2Block; j++)
      if (glen_hist[j]) fprintf(stderr, " %d:%d", j, glen_hist[j]);
    fprintf(stderr, "\ndflash2 spec-sample time: draft=%.0fms verify=%.0fms rollback=%.0fms\n",
            t_draft, t_verify, t_roll);
  }
  return out;
}
