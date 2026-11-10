// ht2_wmma_test.cu — k_ht_wmma vs its frozen pre-rework copy (tools/ht_wmma_ref.inc).
//   bash tools/ht_wmma_test.sh ht2       (Linux / Windows Git Bash, ends PASS/FAIL)
// 1. bit-exact: every v2 ht shape of the decode path, P = 1..48, f32 and bf16
//    x, single and pair launch: the current kernel must reproduce the frozen
//    one bit for bit (the rework only changes scheduling / operand plumbing);
// 2. timing ref vs new (+ the fp32 k_ht_gemv at P <= 8 for scale), weights
//    rotated over copies larger than the 32 MB MALL (DRAM-streamed, as decode);
// 3. serial-decode estimate: the per-token call mix of the 11-10 rocprof trace.
#define main qwenox_real_main
#include "qwenox.cpp"
#undef main
#include "ht_wmma_ref.inc"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#define TCK(x)                                                                        \
  do {                                                                                \
    hipError_t e_ = (x);                                                              \
    if (e_ != hipSuccess) {                                                           \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,     \
              __LINE__);                                                              \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

static std::mt19937 rng(20261110);
static int fails = 0;

// reference launch: the pre-rework host math (P rows staged, no zero row)
template <int NPT>
static void ref_p(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y, uint64_t ys,
                  int P) {
  const int K = (int)w.cols, ntx = K / 128;
  const int nty = (int)((w.rows + 127) / 128);
  int ks, tpb;
  ht_wmma_split(ntx, nty, ks, tpb);
  const int tpc = std::max(1, std::min(tpb, 24 * 1024 / (P * 128 * 2)));
  const size_t lds = std::max((size_t)P * (tpc * 128 + 8) * 2, (size_t)P * 128 * 4);
  const dim3 grid((unsigned)nty, (unsigned)ks);
  if (xbf16)
    k_ht_wmma_ref<NPT, uint16_t><<<grid, 256, lds, g_str>>>(
        w.data, w.suh, w.svh, (const uint16_t*)x, xs, y, ys, nty * 128, K, (int)w.rows, tpb, tpc,
        P, g_ht_zs, g_ht_cnt);
  else
    k_ht_wmma_ref<NPT, float><<<grid, 256, lds, g_str>>>(w.data, w.suh, w.svh, (const float*)x,
                                                         xs, y, ys, nty * 128, K, (int)w.rows,
                                                         tpb, tpc, P, g_ht_zs, g_ht_cnt);
}
static void run_ref(const WView& w, const void* x, bool xbf, float* y, int P) {
  if (P <= 16) ref_p<1>(w, x, xbf, w.cols, y, w.rows, P);
  else if (P <= 32) ref_p<2>(w, x, xbf, w.cols, y, w.rows, P);
  else ref_p<3>(w, x, xbf, w.cols, y, w.rows, P);
}
static void run_new(const WView& w, const void* x, bool xbf, float* y, int P) {
  if (!ht_wmma(w, x, xbf, w.cols, y, w.rows, P, 0, w.rows)) throw std::runtime_error("ht_wmma");
}
static void run_old(const WView& w, const float* x, float* y, int P) {
  for (int off = 0; off < P; off += 8)
    ht_gemv(w, x + (size_t)off * w.cols, false, w.cols, y + (size_t)off * w.rows, w.rows,
            std::min(8, P - off), 0, w.rows);
}
static void ref_pair(const WView& a, const WView& b, const float* x, float* ya, float* yb, int P) {
  const int K = (int)a.cols, ntx = K / 128;
  const int na = (int)(a.rows / 128), nb = (int)(b.rows / 128), nty = na + nb;
  int ks, tpb;
  ht_wmma_split(ntx, nty, ks, tpb);
  const int tpc = std::max(1, std::min(tpb, 24 * 1024 / (P * 128 * 2)));
  const size_t lds = std::max((size_t)P * (tpc * 128 + 8) * 2, (size_t)P * 128 * 4);
  const dim3 grid((unsigned)nty, (unsigned)ks);
#define REFP(NPT)                                                                            \
  k_ht_wmma_ref<NPT, float><<<grid, 256, lds, g_str>>>(                                      \
      a.data, a.suh, a.svh, x, (uint64_t)K, ya, a.rows, nty * 128, K, (int)a.rows, tpb, tpc, \
      P, g_ht_zs, g_ht_cnt, na, b.data, b.suh, b.svh, yb, (int)b.rows)
  if (P <= 16) REFP(1);
  else if (P <= 32) REFP(2);
  else REFP(3);
#undef REFP
}
static std::vector<float> dget(const float* d, size_t n) {
  std::vector<float> h(n);
  TCK(hipMemcpy(h.data(), d, n * 4, hipMemcpyDeviceToHost));
  return h;
}
static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7FFF + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}

struct Weights {
  int O, K, ncopy;
  size_t wbytes;
  uint8_t* codes;
  float *suh, *svh;
  WView view(int copy) const {
    return WView{16, codes + wbytes * (copy % ncopy), (uint64_t)O, (uint64_t)K, 0, suh, svh};
  }
};
static Weights make_weights(int O, int K) {
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  Weights w{O, K, 0, (size_t)O * K / 2, nullptr, nullptr, nullptr};
  w.ncopy = (int)std::max<size_t>(1, std::min<size_t>(32, (384ull << 20) / w.wbytes));
  std::vector<uint8_t> hc(w.wbytes * w.ncopy);
  for (auto& b : hc) b = (uint8_t)(rng() & 0xff);
  std::vector<float> hsuh(K), hsvh(O);
  for (auto& v : hsuh) v = (rng() & 1 ? 1.f : -1.f) * (0.75f + 0.5f * ud(rng));
  for (auto& v : hsvh) v = (rng() & 1 ? 1.f : -1.f) * (0.01f + 0.04f * ud(rng));
  TCK(hipMalloc(&w.codes, hc.size()));
  TCK(hipMemcpy(w.codes, hc.data(), hc.size(), hipMemcpyHostToDevice));
  TCK(hipMalloc(&w.suh, K * 4));
  TCK(hipMalloc(&w.svh, O * 4));
  TCK(hipMemcpy(w.suh, hsuh.data(), K * 4, hipMemcpyHostToDevice));
  TCK(hipMemcpy(w.svh, hsvh.data(), O * 4, hipMemcpyHostToDevice));
  return w;
}
static void free_weights(Weights& w) {
  TCK(hipFree(w.codes));
  TCK(hipFree(w.suh));
  TCK(hipFree(w.svh));
}

static hipEvent_t e0, e1;
template <typename F>
static double time_us(const Weights& w, F&& fn, int P) {
  std::vector<float> ts;
  for (int r = 0; r < 53; r++) {
    TCK(hipEventRecord(e0, g_str));
    fn(w.view(r), P);
    TCK(hipEventRecord(e1, g_str));
    TCK(hipEventSynchronize(e1));
    float ms;
    TCK(hipEventElapsedTime(&ms, e0, e1));
    if (r >= 3) ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2] * 1e3;
}

struct Shape {
  const char* name;
  int O, K;
  int calls;  // per serial decode token (11-10 trace)
};
static const Shape kShapes[] = {
    {"gdn.qkv", 10240, 2560, 36}, {"gdn.z", 6144, 2560, 36},  {"gdn.out", 2560, 6144, 36},
    {"qsa.q", 12288, 2560, 12},   {"qsa.o", 2560, 6144, 12},  {"qsa.kv", 512, 2560, 24},
    {"x640", 640, 2560, 12},      {"se.down", 2560, 640, 48},
};

int main() {
  TCK(hipMalloc(&g_ht_zs, kHtZs * 4));
  TCK(hipMemset(g_ht_zs, 0, kHtZs * 4));
  TCK(hipMalloc(&g_ht_cnt, 4096 * 4));
  TCK(hipMemset(g_ht_cnt, 0, 4096 * 4));
  TCK(hipEventCreate(&e0));
  TCK(hipEventCreate(&e1));
  const int MAXP = 48;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  const int Pc[] = {1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 24, 33, 40, 48};
  const int Pt[] = {1, 2, 4, 5, 8, 16, 40};
  double est[2][2] = {{0, 0}, {0, 0}};  // [P=1|P=5][ref|new] us per token
  printf("%-8s %6s %5s | exact | timing us (ref / new / old fp32)\n", "shape", "O", "K");
  for (const Shape& sh : kShapes) {
    const int O = sh.O, K = sh.K;
    Weights w = make_weights(O, K);
    std::vector<float> hx((size_t)MAXP * K);
    for (auto& v : hx) v = nd(rng) * (ud(rng) < 0.002f ? 30.f : 1.f);
    std::vector<uint16_t> hxb(hx.size());
    for (size_t i = 0; i < hx.size(); i++) hxb[i] = f2bf(hx[i]);
    float *dx, *yr, *yn;
    uint16_t* dxb;
    TCK(hipMalloc(&dx, hx.size() * 4));
    TCK(hipMalloc(&dxb, hx.size() * 2));
    TCK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    TCK(hipMemcpy(dxb, hxb.data(), hxb.size() * 2, hipMemcpyHostToDevice));
    TCK(hipMalloc(&yr, (size_t)MAXP * O * 4));
    TCK(hipMalloc(&yn, (size_t)MAXP * O * 4));
    long bad = 0;
    for (int P : Pc)
      for (int xb = 0; xb < 2; xb++) {
        const void* xp = xb ? (const void*)dxb : (const void*)dx;
        TCK(hipMemset(yr, 0xff, (size_t)P * O * 4));
        TCK(hipMemset(yn, 0xee, (size_t)P * O * 4));
        run_ref(w.view(0), xp, xb, yr, P);
        run_new(w.view(0), xp, xb, yn, P);
        TCK(hipDeviceSynchronize());
        if (memcmp(dget(yr, (size_t)P * O).data(), dget(yn, (size_t)P * O).data(),
                   (size_t)P * O * 4)) {
          if (bad < 3) printf("  %s P=%d %s: new != ref\n", sh.name, P, xb ? "bf16" : "f32");
          bad++;
        }
      }
    if (bad) fails++;
    auto fr = [&](const WView& v, int P) { run_ref(v, dx, false, yr, P); };
    auto fn = [&](const WView& v, int P) { run_new(v, dx, false, yn, P); };
    auto fo = [&](const WView& v, int P) { run_old(v, dx, yr, P); };
    printf("%-8s %6d %5d | %s  |", sh.name, O, K, bad ? "BAD" : "ok ");
    for (int P : Pt) printf(" %7d", P);
    printf("\n%32s ref", "");
    std::vector<double> tr, tn;
    for (int P : Pt) printf(" %7.1f", (tr.push_back(time_us(w, fr, P)), tr.back()));
    printf("\n%32s new", "");
    for (int P : Pt) printf(" %7.1f", (tn.push_back(time_us(w, fn, P)), tn.back()));
    printf("\n%32s old", "");
    for (int P : Pt) printf(" %7.1f", P <= 8 ? time_us(w, fo, P) : 0.0);
    const double gbs = (double)w.wbytes / (tn[0] * 1e-6) / 1e9;
    printf("\n%32s new P=1 %.0f GB/s (ref %.0f)\n", "", gbs,
           (double)w.wbytes / (tr[0] * 1e-6) / 1e9);
    est[0][0] += sh.calls * tr[0];
    est[0][1] += sh.calls * tn[0];
    est[1][0] += sh.calls * tr[3];
    est[1][1] += sh.calls * tn[3];
    TCK(hipFree(dx));
    TCK(hipFree(dxb));
    TCK(hipFree(yr));
    TCK(hipFree(yn));
    free_weights(w);
  }
  // shared expert gate + up: pair launch, 48 per token
  {
    const int O = 640, K = 2560;
    Weights a = make_weights(O, K), b = make_weights(O, K);
    std::vector<float> hx((size_t)MAXP * K);
    for (auto& v : hx) v = nd(rng);
    float *dx, *ra, *rb, *na, *nb;
    TCK(hipMalloc(&dx, hx.size() * 4));
    TCK(hipMemcpy(dx, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    for (float** p : {&ra, &rb, &na, &nb}) TCK(hipMalloc(p, (size_t)MAXP * O * 4));
    long bad = 0;
    for (int P : Pc) {
      ref_pair(a.view(0), b.view(0), dx, ra, rb, P);
      ht_wmma_pair(a.view(0), b.view(0), dx, false, K, na, nb, O, P);
      TCK(hipDeviceSynchronize());
      if (memcmp(dget(ra, (size_t)P * O).data(), dget(na, (size_t)P * O).data(),
                 (size_t)P * O * 4) ||
          memcmp(dget(rb, (size_t)P * O).data(), dget(nb, (size_t)P * O).data(),
                 (size_t)P * O * 4))
        bad++;
    }
    if (bad) fails++;
    auto tp = [&](bool ref, int P) {
      std::vector<float> ts;
      for (int r = 0; r < 53; r++) {
        TCK(hipEventRecord(e0, g_str));
        if (ref) ref_pair(a.view(r), b.view(r), dx, ra, rb, P);
        else ht_wmma_pair(a.view(r), b.view(r), dx, false, K, na, nb, O, P);
        TCK(hipEventRecord(e1, g_str));
        TCK(hipEventSynchronize(e1));
        float ms;
        TCK(hipEventElapsedTime(&ms, e0, e1));
        if (r >= 3) ts.push_back(ms);
      }
      std::sort(ts.begin(), ts.end());
      return ts[ts.size() / 2] * 1e3;
    };
    printf("se.pair  2x%d %5d | %s  |", O, K, bad ? "BAD" : "ok ");
    for (int P : Pt) printf(" %7d", P);
    std::vector<double> tr, tn;
    printf("\n%32s ref", "");
    for (int P : Pt) printf(" %7.1f", (tr.push_back(tp(true, P)), tr.back()));
    printf("\n%32s new", "");
    for (int P : Pt) printf(" %7.1f", (tn.push_back(tp(false, P)), tn.back()));
    printf("\n");
    est[0][0] += 48 * tr[0];
    est[0][1] += 48 * tn[0];
    est[1][0] += 48 * tr[3];
    est[1][1] += 48 * tn[3];
  }
  printf("per-token ht (excl. lm_head, trace call mix): P=1 ref %.2f ms new %.2f ms | "
         "P=5 ref %.2f ms new %.2f ms\n",
         est[0][0] / 1e3, est[0][1] / 1e3, est[1][0] / 1e3, est[1][1] / 1e3);
  printf(fails ? "HT_WMMA2 TEST: FAIL (%d)\n" : "HT_WMMA2 TEST: PASS\n", fails);
  return fails ? 1 : 0;
}
