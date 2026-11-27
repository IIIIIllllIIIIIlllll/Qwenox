// sampgpu_test.cu: standalone validation of the GPU sampling tail kernels
// k_samp_prepare_draw / k_corr_apply_hist (src/gpu/parts/21_kernels_ple.inc)
// against the HostSampler CPU reference (src/gpu/parts/51_host_cfg.inc).
// The two kernel bodies below are verbatim copies of the .inc (keep in sync).
//   a) splitmix64 stream: 1000 next64 bit-exact
//   b) candidate sort: (val desc, id asc) order matches std::sort
//   c) prepare pr: bit-mismatch rate + max relative error (expf ulp source)
//   d) draw over synthetic corpus: token agreement >= 99%, divergent cases
//      traced to expf bit differences
//   e) lp relative error
//   f) corr (bias + legacy presence/frequency penalty): bit-exact
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <climits>
#include <vector>
#include <algorithm>
#include <random>

#define CK(x)                                                            \
  do {                                                                   \
    hipError_t e = (x);                                                  \
    if (e != hipSuccess) {                                               \
      fprintf(stderr, "CK fail %s:%d %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(e));                                     \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

// ---- HostSampler CPU reference (verbatim from 51_host_cfg.inc) --------------
struct HostSamplerRef {
  float temp, top_p, min_p;
  uint64_t rng;
  float presence, frequency;
  std::vector<int> cand;
  std::vector<float> work;

  static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  uint64_t next64() {  // splitmix64
    return mix64(rng += 0x9E3779B97F4A7C15ull);
  }

  double uniform() {
    return (double)(next64() >> 11) * (1.0 / 9007199254740992.0);
  }

  // Legacy penalty phase of correct(): bias adds are tested separately via
  // the same expression; volatile keeps clang from fusing the mul+add into
  // an FMA (the MSVC host build never contracts, and the kernel uses
  // __fmul_rn/__fadd_rn to match).
  void correct(std::vector<float>& logits, const std::vector<std::pair<int, float>>& bias,
               const std::vector<uint16_t>& counts) {
    const int n = (int)logits.size();
    for (auto& b : bias)
      if (b.first >= 0 && b.first < n) logits[b.first] += b.second;
    if (presence != 0.0f || frequency != 0.0f) {
      for (int id = 0; id < n; id++) {
        const unsigned c = counts[id];
        if (c) {
          volatile float prod = frequency * (float)c;
          float pen = presence + (float)prod;
          logits[id] -= pen;
        }
      }
    }
  }

  struct SDist {
    std::vector<int> ids;
    std::vector<float> pr;
  };

  void prepare_sparse(const int* ids, const float* vals, int k, SDist& out) {
    float lmax = -INFINITY;
    for (int j = 0; j < k; j++) lmax = fmaxf(lmax, vals[j]);
    cand.assign(ids, ids + k);
    work.resize(k);
    for (int j = 0; j < k; j++) work[j] = expf((vals[j] - lmax) / temp);
    if (top_p > 0.0f && top_p < 1.0f && !cand.empty()) {
      double total = 0;
      for (float w : work) total += w;
      double lo = 0, hi = 1.0;  // probs are scaled so max == 1
      for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi), m = 0;
        for (float w : work)
          if (w >= (float)mid) m += w;
        if (m >= top_p * total) lo = mid; else hi = mid;
      }
      std::vector<int> keep;
      std::vector<float> kwork;
      keep.reserve(cand.size());
      kwork.reserve(cand.size());
      for (size_t j = 0; j < cand.size(); j++)
        if (work[j] >= (float)lo) {
          keep.push_back(cand[j]);
          kwork.push_back(work[j]);
        }
      cand.swap(keep);
      work.swap(kwork);
    }
    if (min_p > 0.0f && !cand.empty()) {
      float pmax = 0;
      for (float w : work) pmax = fmaxf(pmax, w);
      float thr = min_p * pmax;
      std::vector<int> keep;
      std::vector<float> kwork;
      for (size_t j = 0; j < cand.size(); j++)
        if (work[j] >= thr) {
          keep.push_back(cand[j]);
          kwork.push_back(work[j]);
        }
      if (!keep.empty()) {
        cand.swap(keep);  // never empty the candidate set
        work.swap(kwork);
      }
    }
    double z = 0;
    for (float w : work) z += w;
    out.ids = cand;
    out.pr.resize(cand.size());
    for (size_t j = 0; j < cand.size(); j++) out.pr[j] = (float)(work[j] / z);
  }

  int draw_sparse(const SDist& d, float* lp) {
    double z = 0;
    for (float p : d.pr) z += p;
    double u = uniform() * z;
    size_t si = d.ids.size() - 1;
    for (size_t j = 0; j < d.ids.size(); j++) {
      u -= d.pr[j];
      if (u <= 0) { si = j; break; }
    }
    const int sel = d.ids[si];
    if (lp) *lp = (float)(log((double)d.pr[si]) - log(z));
    return sel;
  }
};

// ---- GPU kernels (verbatim copies from 21_kernels_ple.inc; keep in sync) ----
constexpr int SAMP_CAP = 512;
__device__ __forceinline__ uint64_t samp_mix64(uint64_t z) {  // splitmix64 finalizer
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__global__ void k_samp_prepare_draw(const int* __restrict__ cand_ids,
                                    const float* __restrict__ cand_vals,
                                    const int* __restrict__ cand_n,
                                    const int* __restrict__ hdr,
                                    uint64_t* __restrict__ rng_state,
                                    uint16_t* __restrict__ hist_counts,
                                    int* __restrict__ out_tok,
                                    float* __restrict__ out_lp,
                                    int* __restrict__ out_flag,
                                    int* __restrict__ dbg_ids,
                                    float* __restrict__ dbg_pr,
                                    int* __restrict__ dbg_n) {
  __shared__ int s_id[SAMP_CAP];
  __shared__ float s_val[SAMP_CAP];
  __shared__ float s_w[SAMP_CAP];
  const int cn = cand_n[0];
  const int k = hdr[2];
  if (cn <= 0 || cn > SAMP_CAP || k <= 0) {
    if (threadIdx.x == 0) {
      *out_flag = 1;
      *out_tok = -1;
    }
    return;
  }
  for (int i = threadIdx.x; i < SAMP_CAP; i += blockDim.x) {
    s_id[i] = i < cn ? cand_ids[i] : INT_MAX;
    s_val[i] = i < cn ? cand_vals[i] : -INFINITY;
  }
  __syncthreads();
  // bitonic sort by (val desc, id asc): a strict total order (ids are
  // unique), so any correct sort yields the same permutation as std::sort
  for (int kk = 2; kk <= SAMP_CAP; kk <<= 1)
    for (int stride = kk >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < SAMP_CAP; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const float vi = s_val[i], vj = s_val[j];
          const int ii = s_id[i], ij = s_id[j];
          const bool i_later = vi < vj || (vi == vj && ii > ij);
          if (((i & kk) == 0) == i_later) {
            s_val[i] = vj;
            s_id[i] = ij;
            s_val[j] = vi;
            s_id[j] = ii;
          }
        }
      }
      __syncthreads();
    }
  if (threadIdx.x == 0) {
    const float temp = __int_as_float(hdr[4]);
    const float top_p = __int_as_float(hdr[5]);
    const float min_p = __int_as_float(hdr[6]);
    const int n0 = k < cn ? k : cn;
    float lmax = -INFINITY;
    for (int j = 0; j < n0; j++) lmax = fmaxf(lmax, s_val[j]);
    for (int j = 0; j < n0; j++) s_w[j] = expf((s_val[j] - lmax) / temp);
    int n = n0;
    if (top_p > 0.0f && top_p < 1.0f && n > 0) {
      // largest threshold t with mass(p >= t) >= top_p * total
      double total = 0;
      for (int j = 0; j < n; j++) total += s_w[j];
      double lo = 0, hi = 1.0;  // probs are scaled so max == 1
      for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi), m = 0;
        for (int j = 0; j < n; j++)
          if (s_w[j] >= (float)mid) m += s_w[j];
        if (m >= top_p * total) lo = mid; else hi = mid;
      }
      int w = 0;
      for (int j = 0; j < n; j++)
        if (s_w[j] >= (float)lo) {
          s_w[w] = s_w[j];
          s_id[w] = s_id[j];
          w++;
        }
      n = w;
    }
    if (min_p > 0.0f && n > 0) {
      float pmax = 0;
      for (int j = 0; j < n; j++) pmax = fmaxf(pmax, s_w[j]);
      const float thr = min_p * pmax;
      int w = 0;
      for (int j = 0; j < n; j++)
        if (s_w[j] >= thr) {
          s_w[w] = s_w[j];
          s_id[w] = s_id[j];
          w++;
        }
      if (w > 0) n = w;  // never empty the candidate set
    }
    double z = 0;
    for (int j = 0; j < n; j++) z += s_w[j];
    for (int j = 0; j < n; j++) s_w[j] = (float)(s_w[j] / z);
    if (dbg_ids && dbg_pr && dbg_n) {
      *dbg_n = n;
      for (int j = 0; j < n; j++) {
        dbg_ids[j] = s_id[j];
        dbg_pr[j] = s_w[j];
      }
    }
    double z2 = 0;
    for (int j = 0; j < n; j++) z2 += s_w[j];
    const uint64_t st = rng_state[0] + 0x9E3779B97F4A7C15ull;  // splitmix64 next64
    rng_state[0] = st;
    double u = (double)(samp_mix64(st) >> 11) * (1.0 / 9007199254740992.0) * z2;
    int si = n - 1;
    for (int j = 0; j < n; j++) {
      u -= s_w[j];
      if (u <= 0) {
        si = j;
        break;
      }
    }
    const int sel = s_id[si];
    *out_tok = sel;
    *out_lp = (float)(log((double)s_w[si]) - log(z2));
    *out_flag = 0;
    if ((hdr[7] & 1) && hist_counts) hist_counts[sel] += 1;
  }
}

__global__ void k_corr_apply_hist(float* __restrict__ x, int stride,
                                  const int* __restrict__ ci,
                                  const float* __restrict__ cv,
                                  const int* __restrict__ hdr,
                                  const uint16_t* __restrict__ counts) {
  const int nb = hdr[0];
  const float presence = __int_as_float(hdr[8]);
  const float frequency = __int_as_float(hdr[9]);
  if (nb == 0 && presence == 0.0f && frequency == 0.0f) return;
  float* xr = x + (size_t)blockIdx.y * stride;
  for (int i = threadIdx.x; i < nb; i += blockDim.x) xr[ci[i]] += cv[i];
  __syncthreads();
  if (presence != 0.0f || frequency != 0.0f)
    for (int id = threadIdx.x; id < stride; id += blockDim.x) {
      const unsigned c = counts[id];
      if (c) xr[id] -= __fadd_rn(presence, __fmul_rn(frequency, (float)c));
    }
}

// ---- test-only probes --------------------------------------------------------
// Splitmix64 stream probe: n next64 outputs, one thread.
__global__ void k_rng_probe(uint64_t* __restrict__ state, uint64_t* __restrict__ out,
                            int n) {
  uint64_t st = state[0];
  for (int i = 0; i < n; i++) {
    st += 0x9E3779B97F4A7C15ull;
    out[i] = samp_mix64(st);
  }
  state[0] = st;
}

// expf probe over arbitrary inputs (divergence attribution).
__global__ void k_expf_probe(const float* __restrict__ in, float* __restrict__ out,
                             int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expf(in[i]);
}

// ---- helpers -----------------------------------------------------------------
static uint32_t g_lcg = 12345;
static float rnd() {  // [0,1)
  g_lcg = g_lcg * 1664525u + 1013904223u;
  return (g_lcg >> 8) / 16777216.0f;
}
static int rnd_int(int lo, int hi) {  // inclusive
  return lo + (int)(rnd() * (float)(hi - lo + 1));
}

static void* dalloc(size_t bytes) {
  void* p = nullptr;
  CK(hipMalloc(&p, bytes));
  return p;
}
static void dfree(void* p) { CK(hipFree(p)); }

static uint32_t f2u(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

// Random unique-id candidate group: cn ids sampled from [0, VOCAB).
struct Group {
  int cn, k;
  float temp, top_p, min_p;
  std::vector<int> ids;
  std::vector<float> vals;
};
static const int VOCAB = 248320;

static Group gen_group() {
  Group g;
  g.cn = rnd_int(1, SAMP_CAP);
  g.k = rnd_int(1, SAMP_CAP + 64);  // includes k > cn cases
  const float temps[] = {0.05f, 0.2f, 0.5f, 0.7f, 1.0f, 1.5f, 2.0f};
  const float tops[] = {0.0f, 0.5f, 0.8f, 0.9f, 0.95f, 1.0f};
  const float mins[] = {0.0f, 0.01f, 0.05f, 0.2f};
  g.temp = temps[rnd_int(0, 6)];
  g.top_p = tops[rnd_int(0, 5)];
  g.min_p = mins[rnd_int(0, 3)];
  std::vector<int> pool(2048);
  for (int i = 0; i < 2048; i++) pool[i] = rnd_int(0, VOCAB - 1);
  std::sort(pool.begin(), pool.end());
  pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
  while ((int)pool.size() < g.cn) {  // extremely unlikely; top up with fresh ids
    int id = rnd_int(0, VOCAB - 1);
    if (!std::binary_search(pool.begin(), pool.end(), id)) {
      pool.push_back(id);
      std::sort(pool.begin(), pool.end());
    }
  }
  std::shuffle(pool.begin(), pool.end(),
               std::mt19937(g_lcg));
  g.ids.assign(pool.begin(), pool.begin() + g.cn);
  for (int j = 0; j < g.cn; j++) g.vals.push_back(-30.0f + 40.0f * rnd());
  return g;
}

// Host replay: sort (val desc, id asc), take first k, prepare, draw.
static void host_sorted_topk(const Group& g, std::vector<int>& ids,
                             std::vector<float>& vals) {
  ids = g.ids;
  vals = g.vals;
  std::vector<int> ord(g.cn);
  for (int i = 0; i < g.cn; i++) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](int a, int b) {
    if (g.vals[a] != g.vals[b]) return g.vals[a] > g.vals[b];
    return g.ids[a] < g.ids[b];
  });
  const int n0 = g.k < g.cn ? g.k : g.cn;
  std::vector<int> si(n0);
  std::vector<float> sv(n0);
  for (int j = 0; j < n0; j++) {
    si[j] = g.ids[ord[j]];
    sv[j] = g.vals[ord[j]];
  }
  ids.swap(si);
  vals.swap(sv);
}

int main() {
  printf("[sampgpu] validation start\n");
  bool pass = true;

  // ---- (a) splitmix64 stream ------------------------------------------------
  {
    const int N = 1000;
    const uint64_t seed = 0x123456789abcdefULL;
    HostSamplerRef ref{};
    ref.rng = seed;
    std::vector<uint64_t> h(N);
    for (int i = 0; i < N; i++) h[i] = ref.next64();
    uint64_t* d_st = (uint64_t*)dalloc(8);
    uint64_t* d_out = (uint64_t*)dalloc(N * 8);
    CK(hipMemcpy(d_st, &seed, 8, hipMemcpyHostToDevice));
    k_rng_probe<<<1, 1>>>(d_st, d_out, N);
    std::vector<uint64_t> g(N);
    CK(hipMemcpy(g.data(), d_out, N * 8, hipMemcpyDeviceToHost));
    uint64_t st_end = 0;
    CK(hipMemcpy(&st_end, d_st, 8, hipMemcpyDeviceToHost));
    int mm = 0;
    for (int i = 0; i < N; i++) mm += h[i] != g[i];
    const bool state_ok = st_end == ref.rng;
    printf("[a] rng stream: %d/%d outputs bit-exact (mismatches=%d), state "
           "advance %s\n", N - mm, N, mm, state_ok ? "ok" : "MISMATCH");
    if (mm || !state_ok) pass = false;
    dfree(d_st);
    dfree(d_out);
  }

  // ---- (b) sort --------------------------------------------------------------
  {
    struct C {
      int cn;
      bool ties;
    } cases[] = {{1, false}, {2, true}, {7, false}, {64, true}, {128, false},
                 {300, true}, {511, false}, {512, true}};
    int* d_ids = (int*)dalloc(SAMP_CAP * 4);
    float* d_vals = (float*)dalloc(SAMP_CAP * 4);
    int* d_cn = (int*)dalloc(4);
    int* d_hdr = (int*)dalloc(8 * 4);
    uint64_t* d_rng = (uint64_t*)dalloc(8);
    int* d_tok = (int*)dalloc(4);
    float* d_lp = (float*)dalloc(4);
    int* d_flag = (int*)dalloc(4);
    int* d_dbg_ids = (int*)dalloc(SAMP_CAP * 4);
    float* d_dbg_pr = (float*)dalloc(SAMP_CAP * 4);
    int* d_dbg_n = (int*)dalloc(4);
    int bad = 0;
    for (auto c : cases) {
      Group g;
      g.cn = c.cn;
      g.k = c.cn;
      g.temp = 1.0f;
      g.top_p = 0.0f;
      g.min_p = 0.0f;
      for (int j = 0; j < g.cn; j++) g.ids.push_back(rnd_int(0, VOCAB - 1));
      // force unique ids (sort comparator requires a strict total order)
      std::sort(g.ids.begin(), g.ids.end());
      g.ids.erase(std::unique(g.ids.begin(), g.ids.end()), g.ids.end());
      while ((int)g.ids.size() < g.cn) g.ids.push_back(g.ids.back() + 1);
      std::shuffle(g.ids.begin(), g.ids.end(), std::mt19937(g_lcg));
      for (int j = 0; j < g.cn; j++)
        g.vals.push_back(c.ties ? (float)rnd_int(-3, 3) : -30.0f + 40.0f * rnd());
      std::vector<int> hs_ids;
      std::vector<float> hs_vals;
      host_sorted_topk(g, hs_ids, hs_vals);
      CK(hipMemcpy(d_ids, g.ids.data(), g.cn * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_vals, g.vals.data(), g.cn * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_cn, &g.cn, 4, hipMemcpyHostToDevice));
      int hdr[8] = {};
      hdr[2] = g.k;
      hdr[4] = (int)f2u(g.temp);
      hdr[5] = (int)f2u(g.top_p);
      hdr[6] = (int)f2u(g.min_p);
      hdr[7] = 0;
      CK(hipMemcpy(d_hdr, hdr, sizeof hdr, hipMemcpyHostToDevice));
      const uint64_t seed = 7;
      CK(hipMemcpy(d_rng, &seed, 8, hipMemcpyHostToDevice));
      k_samp_prepare_draw<<<1, 256>>>(d_ids, d_vals, d_cn, d_hdr, d_rng, nullptr,
                                      d_tok, d_lp, d_flag, d_dbg_ids, d_dbg_pr,
                                      d_dbg_n);
      std::vector<int> gi(g.cn);
      int gn = 0, flag = 1;
      CK(hipMemcpy(gi.data(), d_dbg_ids, g.cn * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(&gn, d_dbg_n, 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(&flag, d_flag, 4, hipMemcpyDeviceToHost));
      // with no filters the kernel emits the full sorted top-k: id order must
      // equal the host std::sort order (pr bits are checked in (c))
      bool ok = flag == 0 && gn == g.cn;
      for (int j = 0; j < gn && ok; j++) ok = gi[j] == hs_ids[j];
      if (!ok) {
        bad++;
        printf("    sort case cn=%d ties=%d MISMATCH (gn=%d flag=%d)\n", c.cn,
               (int)c.ties, gn, flag);
      }
    }
    printf("[b] sort: %zu/8 cases exact order (val desc, id asc), bad=%d\n",
           8 - (size_t)bad, bad);
    if (bad) pass = false;
    dfree(d_ids);
    dfree(d_vals);
    dfree(d_cn);
    dfree(d_hdr);
    dfree(d_rng);
    dfree(d_tok);
    dfree(d_lp);
    dfree(d_flag);
    dfree(d_dbg_ids);
    dfree(d_dbg_pr);
    dfree(d_dbg_n);
  }

  // ---- (c)+(d)+(e) corpus: prepare pr bits, draw agreement, lp error --------
  {
    const int G = 600;
    std::vector<Group> gs(G);
    for (auto& g : gs) g = gen_group();
    int* d_ids = (int*)dalloc((size_t)G * SAMP_CAP * 4);
    float* d_vals = (float*)dalloc((size_t)G * SAMP_CAP * 4);
    int* d_cn = (int*)dalloc(G * 4);
    int* d_hdr = (int*)dalloc((size_t)G * 8 * 4);
    uint64_t* d_rng = (uint64_t*)dalloc(8);
    int* d_tok = (int*)dalloc(G * 4);
    float* d_lp = (float*)dalloc(G * 4);
    int* d_flag = (int*)dalloc(G * 4);
    std::vector<int> h_ids((size_t)G * SAMP_CAP, 0);
    std::vector<float> h_vals((size_t)G * SAMP_CAP, 0.f);
    std::vector<int> h_cn(G), h_hdr((size_t)G * 8);
    for (int g = 0; g < G; g++) {
      for (int j = 0; j < gs[g].cn; j++) {
        h_ids[(size_t)g * SAMP_CAP + j] = gs[g].ids[j];
        h_vals[(size_t)g * SAMP_CAP + j] = gs[g].vals[j];
      }
      h_cn[g] = gs[g].cn;
      h_hdr[(size_t)g * 8 + 2] = gs[g].k;
      h_hdr[(size_t)g * 8 + 4] = (int)f2u(gs[g].temp);
      h_hdr[(size_t)g * 8 + 5] = (int)f2u(gs[g].top_p);
      h_hdr[(size_t)g * 8 + 6] = (int)f2u(gs[g].min_p);
      h_hdr[(size_t)g * 8 + 7] = 0;
    }
    CK(hipMemcpy(d_ids, h_ids.data(), h_ids.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_vals, h_vals.data(), h_vals.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_cn, h_cn.data(), G * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_hdr, h_hdr.data(), h_hdr.size() * 4, hipMemcpyHostToDevice));
    const uint64_t seed = 0xdeadbeefcafeULL;
    CK(hipMemcpy(d_rng, &seed, 8, hipMemcpyHostToDevice));
    int* d_dbg_ids = (int*)dalloc((size_t)G * SAMP_CAP * 4);
    float* d_dbg_pr = (float*)dalloc((size_t)G * SAMP_CAP * 4);
    int* d_dbg_n = (int*)dalloc(G * 4);
    for (int g = 0; g < G; g++)
      k_samp_prepare_draw<<<1, 256>>>(d_ids + (size_t)g * SAMP_CAP,
                                      d_vals + (size_t)g * SAMP_CAP, d_cn + g,
                                      d_hdr + (size_t)g * 8, d_rng, nullptr,
                                      d_tok + g, d_lp + g, d_flag + g,
                                      d_dbg_ids + (size_t)g * SAMP_CAP,
                                      d_dbg_pr + (size_t)g * SAMP_CAP,
                                      d_dbg_n + g);
    CK(hipDeviceSynchronize());
    std::vector<int> g_tok(G), g_flag(G), g_dbg_n(G);
    std::vector<float> g_lp(G);
    std::vector<int> g_dbg_ids((size_t)G * SAMP_CAP);
    std::vector<float> g_dbg_pr((size_t)G * SAMP_CAP);
    CK(hipMemcpy(g_tok.data(), d_tok, G * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_lp.data(), d_lp, G * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_flag.data(), d_flag, G * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_dbg_ids.data(), d_dbg_ids, g_dbg_ids.size() * 4,
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_dbg_pr.data(), d_dbg_pr, g_dbg_pr.size() * 4,
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(g_dbg_n.data(), d_dbg_n, G * 4, hipMemcpyDeviceToHost));

    // host replay
    HostSamplerRef ref{};
    ref.rng = seed;
    long pr_total = 0, pr_mm = 0;
    double pr_maxrel = 0, pr_maxrel_big = 0;
    long pr_denormal_mm = 0;
    int tok_match = 0, set_div = 0, flags_bad = 0;
    double lp_maxrel = 0, lp_sum = 0;
    int lp_n = 0;
    std::vector<int> div_groups;
    for (int g = 0; g < G; g++) {
      ref.temp = gs[g].temp;
      ref.top_p = gs[g].top_p;
      ref.min_p = gs[g].min_p;
      std::vector<int> si;
      std::vector<float> sv;
      host_sorted_topk(gs[g], si, sv);
      HostSamplerRef::SDist d;
      ref.prepare_sparse(si.data(), sv.data(), (int)si.size(), d);
      float h_lp = 0;
      const int h_tok = ref.draw_sparse(d, &h_lp);
      // (c) pr comparison over the common set
      const int gn = g_dbg_n[g];
      bool set_same = gn == (int)d.ids.size();
      for (int j = 0; j < gn && set_same; j++)
        set_same = g_dbg_ids[(size_t)g * SAMP_CAP + j] == d.ids[j];
      if (!set_same) set_div++;
      const int ncmp = (int)d.pr.size() < gn ? (int)d.pr.size() : gn;
      for (int j = 0; j < ncmp; j++) {
        const float hp = d.pr[j];
        const float gp = g_dbg_pr[(size_t)g * SAMP_CAP + j];
        pr_total++;
        if (f2u(hp) != f2u(gp)) {
          pr_mm++;
          const double rel = hp != 0.f ? fabs((double)gp - hp) / fabs((double)hp)
                                       : fabs((double)gp);
          if (rel > pr_maxrel) pr_maxrel = rel;
          if (fabs((double)hp) < 1e-30) pr_denormal_mm++;  // expf denormal/flush
          else if (rel > pr_maxrel_big) pr_maxrel_big = rel;
        }
      }
      // (d) token
      if (g_flag[g] != 0) flags_bad++;
      if (g_tok[g] == h_tok) tok_match++;
      else div_groups.push_back(g);
      // (e) lp
      if (g_tok[g] == h_tok) {
        const double rel = fabs((double)g_lp[g] - h_lp) /
                           (fabs((double)h_lp) > 1e-9 ? fabs((double)h_lp) : 1e-9);
        if (rel > lp_maxrel) lp_maxrel = rel;
        lp_sum += rel;
        lp_n++;
      }
    }
    printf("[c] pr: %ld values compared, bit mismatches %ld (%.4f%%), max rel "
           "err %.3e (%ld of them denormal-scale <1e-30), max rel err "
           "non-denormal %.3e; candidate-set divergent groups %d/%d\n",
           pr_total, pr_mm, 100.0 * pr_mm / (double)pr_total, pr_maxrel,
           pr_denormal_mm, pr_maxrel_big, set_div, G);
    printf("[d] draw: %d/%d tokens match (%.2f%%), divergent %d, flag errors "
           "%d\n", tok_match, G, 100.0 * tok_match / (double)G,
           (int)div_groups.size(), flags_bad);
    printf("[e] lp: max rel err %.3e, mean %.3e over %d matching tokens\n",
           lp_maxrel, lp_n ? lp_sum / lp_n : 0.0, lp_n);
    if ((double)tok_match / G < 0.99 || flags_bad) pass = false;

    // divergence attribution: every divergent group must show an expf bit
    // difference between UCRT host expf and device expf on that group's
    // prepare inputs
    float* d_ein = (float*)dalloc(SAMP_CAP * 4);
    float* d_eout = (float*)dalloc(SAMP_CAP * 4);
    int unattributed = 0;
    for (int g : div_groups) {
      std::vector<int> si;
      std::vector<float> sv;
      host_sorted_topk(gs[g], si, sv);
      float lmax = -INFINITY;
      for (float v : sv) lmax = fmaxf(lmax, v);
      std::vector<float> ein(sv.size()), hexp(sv.size());
      for (size_t j = 0; j < sv.size(); j++) {
        ein[j] = (sv[j] - lmax) / gs[g].temp;
        hexp[j] = expf(ein[j]);
      }
      std::vector<float> gexp(sv.size());
      CK(hipMemcpy(d_ein, ein.data(), ein.size() * 4, hipMemcpyHostToDevice));
      k_expf_probe<<<dim3((unsigned)((sv.size() + 255) / 256)), 256>>>(
          d_ein, d_eout, (int)sv.size());
      CK(hipMemcpy(gexp.data(), d_eout, gexp.size() * 4, hipMemcpyDeviceToHost));
      int ediff = 0;
      double emaxrel = 0;
      for (size_t j = 0; j < sv.size(); j++)
        if (f2u(hexp[j]) != f2u(gexp[j])) {
          ediff++;
          const double rel = fabs((double)gexp[j] - hexp[j]) /
                             (fabs((double)hexp[j]) > 0 ? fabs((double)hexp[j]) : 1);
          if (rel > emaxrel) emaxrel = rel;
        }
      if (!ediff) unattributed++;
      if (div_groups.size() <= 12 || unattributed == 0)
        printf("    div group %d: cn=%d k=%d temp=%g top_p=%g min_p=%g -> expf "
               "bit diffs %d/%zu (max rel %.2e)%s\n", g, gs[g].cn, gs[g].k,
               gs[g].temp, gs[g].top_p, gs[g].min_p, ediff, sv.size(), emaxrel,
               ediff ? "" : "  <-- NOT explained by expf!");
    }
    printf("[d'] divergence attribution: %d/%d divergent groups traced to expf "
           "ulp%s\n", (int)div_groups.size() - unattributed,
           (int)div_groups.size(), unattributed ? "  <-- FAIL" : "");
    if (unattributed) pass = false;
    dfree(d_ids);
    dfree(d_vals);
    dfree(d_cn);
    dfree(d_hdr);
    dfree(d_rng);
    dfree(d_tok);
    dfree(d_lp);
    dfree(d_flag);
    dfree(d_dbg_ids);
    dfree(d_dbg_pr);
    dfree(d_dbg_n);
    dfree(d_ein);
    dfree(d_eout);
  }

  // ---- overflow fallback + hist bump -----------------------------------------
  {
    int* d_ids = (int*)dalloc(SAMP_CAP * 4);
    float* d_vals = (float*)dalloc(SAMP_CAP * 4);
    int* d_cn = (int*)dalloc(4);
    int* d_hdr = (int*)dalloc(8 * 4);
    uint64_t* d_rng = (uint64_t*)dalloc(8);
    int* d_tok = (int*)dalloc(4);
    float* d_lp = (float*)dalloc(4);
    int* d_flag = (int*)dalloc(4);
    uint16_t* d_counts = (uint16_t*)dalloc(VOCAB * 2);
    CK(hipMemset(d_counts, 0, VOCAB * 2));
    // overflow: cand_n > SAMP_CAP -> flag 1, tok -1
    const int cn_over = SAMP_CAP + 1;
    CK(hipMemcpy(d_cn, &cn_over, 4, hipMemcpyHostToDevice));
    int hdr[8] = {};
    hdr[2] = 40;
    hdr[4] = (int)f2u(1.0f);
    hdr[7] = 1;
    CK(hipMemcpy(d_hdr, hdr, sizeof hdr, hipMemcpyHostToDevice));
    const uint64_t seed = 11;
    CK(hipMemcpy(d_rng, &seed, 8, hipMemcpyHostToDevice));
    k_samp_prepare_draw<<<1, 256>>>(d_ids, d_vals, d_cn, d_hdr, d_rng, d_counts,
                                    d_tok, d_lp, d_flag, nullptr, nullptr,
                                    nullptr);
    int tok = 0, flag = 0;
    CK(hipMemcpy(&tok, d_tok, 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&flag, d_flag, 4, hipMemcpyDeviceToHost));
    const bool over_ok = flag == 1 && tok == -1;
    // hist bump: normal draw with flags bit0 set
    Group g;
    g.cn = 64;
    g.k = 64;
    for (int j = 0; j < g.cn; j++) {
      g.ids.push_back(1000 + j);
      g.vals.push_back(-10.0f + 10.0f * rnd());
    }
    CK(hipMemcpy(d_ids, g.ids.data(), g.cn * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_vals, g.vals.data(), g.cn * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(d_cn, &g.cn, 4, hipMemcpyHostToDevice));
    k_samp_prepare_draw<<<1, 256>>>(d_ids, d_vals, d_cn, d_hdr, d_rng, d_counts,
                                    d_tok, d_lp, d_flag, nullptr, nullptr,
                                    nullptr);
    CK(hipMemcpy(&tok, d_tok, 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&flag, d_flag, 4, hipMemcpyDeviceToHost));
    std::vector<uint16_t> hc(VOCAB);
    CK(hipMemcpy(hc.data(), d_counts, VOCAB * 2, hipMemcpyDeviceToHost));
    int nonzero = 0;
    bool bump_ok = flag == 0 && tok >= 0;
    for (int i = 0; i < VOCAB; i++) {
      if (hc[i]) {
        nonzero++;
        bump_ok = bump_ok && i == tok && hc[i] == 1;
      }
    }
    bump_ok = bump_ok && nonzero == 1;
    printf("[g] overflow fallback (flag=1, tok=-1): %s; hist bump: %s\n",
           over_ok ? "ok" : "FAIL", bump_ok ? "ok" : "FAIL");
    if (!over_ok || !bump_ok) pass = false;
    dfree(d_ids);
    dfree(d_vals);
    dfree(d_cn);
    dfree(d_hdr);
    dfree(d_rng);
    dfree(d_tok);
    dfree(d_lp);
    dfree(d_flag);
    dfree(d_counts);
  }

  // ---- (f) corr: bias + legacy penalty bit-exact -----------------------------
  {
    struct C {
      int nb;
      float presence, frequency;
      int ncounted;
    } cases[] = {{64, 0.75f, 0.5f, 2000},
                 {3, 0.0f, 0.0f, 0},
                 {128, 1.5f, -0.25f, 5000}};
    float* d_x = (float*)dalloc((size_t)VOCAB * 4);
    int* d_ci = (int*)dalloc(2048 * 4);
    float* d_cv = (float*)dalloc(2048 * 4);
    int* d_hdr = (int*)dalloc(16 * 4);
    uint16_t* d_counts = (uint16_t*)dalloc(VOCAB * 2);
    long total_mm = 0;
    for (auto c : cases) {
      std::vector<float> x0(VOCAB);
      for (auto& v : x0) v = -20.0f + 30.0f * rnd();
      std::vector<std::pair<int, float>> bias;
      for (int j = 0; j < c.nb; j++)
        bias.push_back({rnd_int(0, VOCAB - 1), -5.0f + 10.0f * rnd()});
      std::vector<uint16_t> counts(VOCAB, 0);
      for (int j = 0; j < c.ncounted; j++) counts[rnd_int(0, VOCAB - 1)] =
          (uint16_t)rnd_int(1, 20);
      // host reference
      HostSamplerRef ref{};
      ref.presence = c.presence;
      ref.frequency = c.frequency;
      std::vector<float> href = x0;
      ref.correct(href, bias, counts);
      // gpu
      CK(hipMemcpy(d_x, x0.data(), (size_t)VOCAB * 4, hipMemcpyHostToDevice));
      std::vector<int> ci(c.nb);
      std::vector<float> cv(c.nb);
      for (int j = 0; j < c.nb; j++) {
        ci[j] = bias[j].first;
        cv[j] = bias[j].second;
      }
      int hdr[16] = {};
      hdr[0] = c.nb;
      hdr[8] = (int)f2u(c.presence);
      hdr[9] = (int)f2u(c.frequency);
      CK(hipMemcpy(d_ci, ci.data(), ci.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_cv, cv.data(), cv.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_hdr, hdr, sizeof hdr, hipMemcpyHostToDevice));
      CK(hipMemcpy(d_counts, counts.data(), VOCAB * 2, hipMemcpyHostToDevice));
      k_corr_apply_hist<<<dim3(1, 1), 256>>>(d_x, VOCAB, d_ci, d_cv, d_hdr,
                                             d_counts);
      std::vector<float> gx(VOCAB);
      CK(hipMemcpy(gx.data(), d_x, (size_t)VOCAB * 4, hipMemcpyDeviceToHost));
      long mm = 0;
      for (int i = 0; i < VOCAB; i++) mm += f2u(gx[i]) != f2u(href[i]);
      total_mm += mm;
      printf("    corr case nb=%d presence=%g frequency=%g counted=%d: %ld "
             "mismatches / %d logits\n", c.nb, c.presence, c.frequency,
             c.ncounted, mm, VOCAB);
    }
    printf("[f] corr: total bit mismatches %ld (must be 0)\n", total_mm);
    if (total_mm) pass = false;
    dfree(d_x);
    dfree(d_ci);
    dfree(d_cv);
    dfree(d_hdr);
    dfree(d_counts);
  }

  printf("[sampgpu] %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
