// tg_kernels_test.cu — unit test for the TG/MTP decode kernels in
// src/gpu/parts/21_kernels_ple.inc ([tgk-begin]..[tgk-end], extracted to
// tg_kernels.inc by tools/tg_verify.sh):
//   * two-stage argmax (k_argmax_p1/p2 via argmax_fast_rows) vs CPU
//     first-occurrence argmax: random, max at the last element / in the last
//     chunk, exact ties, all -inf row; rows 1/4/9/65
//   * radix top-k (k_tk_init/hist/scan/collect, same launch sequence as
//     Model::sms_topk_rows) vs CPU (value desc, id asc) top-k, k = 1/20/64,
//     including heavy-duplicate rows (overflow -> counted, not a failure) and
//     rows with -inf / negative values
//   * k_corr_apply (bias add, hist subtract, per-row prefix subtract) vs CPU
// Last line: PASS / FAIL.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                              \
  do {                                                                     \
    hipError_t e_ = (x);                                                   \
    if (e_ != hipSuccess) {                                                \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),   \
              __FILE__, __LINE__);                                         \
      printf("FAIL\n");                                                    \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// argmax_rows (inside the extracted block) falls back to this for >65 rows.
__global__ void k_argmax(const float* __restrict__, int, int* __restrict__) {}

#include "tg_kernels.inc"

static int fails = 0;
static void bad(const char* what) {
  printf("  FAIL: %s\n", what);
  fails++;
}

static int cpu_argmax(const float* x, int n) {
  int b = 0;
  for (int i = 1; i < n; i++)
    if (x[i] > x[b]) b = i;
  return b;
}

static void test_argmax(int n) {
  std::mt19937 rng(1234 + n);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  const int rows_list[] = {1, 4, 9, 65};
  float *d_x;
  int* d_out;
  unsigned long long* d_cand;
  CK(hipMalloc(&d_x, (size_t)65 * n * 4));
  CK(hipMalloc(&d_out, 65 * 4));
  CK(hipMalloc(&d_cand, (size_t)65 * ARGMAX_B1 * 8));
  int total = 0, wrong = 0;
  for (int rows : rows_list) {
    std::vector<float> h((size_t)rows * n);
    for (auto& v : h) v = nd(rng);
    for (int r = 0; r < rows; r++) {
      float* x = h.data() + (size_t)r * n;
      switch (r % 7) {
        case 0: x[rng() % n] = 40.0f; break;                    // random spot
        case 1: x[n - 1] = 40.0f; break;                        // last element
        case 2: x[n - 1 - (int)(rng() % (n / 8))] = 40.0f; break;  // tail chunk
        case 3: {                                               // exact tie
          int a = rng() % n, b = rng() % n;
          x[a] = 40.0f;
          x[b] = 40.0f;
        } break;
        case 4: x[0] = 40.0f; x[n - 1] = 40.0f; break;          // tie ends
        case 5:
          for (int i = 0; i < n; i++) x[i] = -INFINITY;         // all -inf
          break;
        default: break;                                         // plain random
      }
    }
    CK(hipMemcpy(d_x, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemset(d_out, 0xFF, 65 * 4));
    argmax_fast_rows(d_x, n, rows, d_cand, d_out, 0);
    CK(hipDeviceSynchronize());
    std::vector<int> out(rows);
    CK(hipMemcpy(out.data(), d_out, rows * 4, hipMemcpyDeviceToHost));
    for (int r = 0; r < rows; r++) {
      const int ref = cpu_argmax(h.data() + (size_t)r * n, n);
      total++;
      if (out[r] != ref) {
        if (wrong < 5)
          printf("    argmax n=%d rows=%d row %d (case %d): gpu %d cpu %d\n",
                 n, rows, r, r % 7, out[r], ref);
        wrong++;
      }
    }
  }
  // timing: 1 row and 4 rows, fast vs single-block
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  float t_fast = 0, t_old = 0;
  for (int it = 0; it < 2; it++) {
    CK(hipEventRecord(e0));
    for (int i = 0; i < 50; i++) argmax_fast_rows(d_x, n, 4, d_cand, d_out, 0);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    CK(hipEventElapsedTime(&t_fast, e0, e1));
  }
  printf("  argmax n=%d: %d/%d rows correct; 4 rows fast %.1f us/call\n", n,
         total - wrong, total, t_fast * 1000 / 50);
  (void)t_old;
  if (wrong) bad("two-stage argmax differs from CPU first-occurrence argmax");
  CK(hipFree(d_x));
  CK(hipFree(d_out));
  CK(hipFree(d_cand));
}

static const int TCAP = 512;

// Same launch sequence as Model::sms_topk_rows.
static void gpu_topk(const float* d_src, int n, int rows, int k, int* d_ctrl,
                     unsigned* d_hist, int* d_ids, float* d_vals, int* d_cn) {
  k_tk_init<<<rows, 32, 0, 0>>>(d_ctrl, k);
  CK(hipMemsetAsync(d_hist, 0, (size_t)rows * 4096 * 4, 0));
  k_tk_hist<0><<<dim3(8, rows), 256, 0, 0>>>(d_src, n, n, d_ctrl, d_hist);
  k_tk_scan<0><<<dim3(1, rows), 256, 0, 0>>>(d_hist, d_ctrl, d_cn);
  CK(hipMemsetAsync(d_hist, 0, (size_t)rows * 4096 * 4, 0));
  k_tk_hist<1><<<dim3(8, rows), 256, 0, 0>>>(d_src, n, n, d_ctrl, d_hist);
  k_tk_scan<1><<<dim3(1, rows), 256, 0, 0>>>(d_hist, d_ctrl, d_cn);
  k_tk_collect<<<dim3(8, rows), 256, 0, 0>>>(d_src, n, n, d_ctrl, d_ids,
                                             d_vals, d_cn, TCAP);
}

static void test_topk(int n) {
  std::mt19937 rng(99 + n);
  std::normal_distribution<float> nd(0.0f, 3.0f);
  const int R = 9;
  float *d_x, *d_vals;
  int *d_ctrl, *d_ids, *d_cn;
  unsigned* d_hist;
  CK(hipMalloc(&d_x, (size_t)R * n * 4));
  CK(hipMalloc(&d_ctrl, R * 8 * 4));
  CK(hipMalloc(&d_hist, (size_t)R * 4096 * 4));
  CK(hipMalloc(&d_ids, (size_t)R * TCAP * 4));
  CK(hipMalloc(&d_vals, (size_t)R * TCAP * 4));
  CK(hipMalloc(&d_cn, R * 4));
  int checked = 0, wrong = 0, overflow = 0;
  for (int k : {1, 20, 64}) {
    std::vector<float> h((size_t)R * n);
    for (int r = 0; r < R; r++) {
      float* x = h.data() + (size_t)r * n;
      for (int i = 0; i < n; i++) x[i] = nd(rng);
      switch (r) {
        case 1:  // heavy duplicates at the top (quantized)
          for (int i = 0; i < n; i++) x[i] = std::round(x[i] * 4) / 4;
          break;
        case 2:  // all negative, some -inf
          for (int i = 0; i < n; i++)
            x[i] = (i % 17 == 0) ? -INFINITY : -fabsf(x[i]) - 5.0f;
          break;
        case 3:  // winners in the tail
          for (int j = 0; j < 80; j++) x[n - 1 - j] = 30.0f - j * 0.01f;
          break;
        case 4:  // exact ties straddling the k-th place
          for (int j = 0; j < 100; j++) x[(j * 2477) % n] = 25.0f;
          break;
        case 5:  // every value identical -> must overflow, not crash
          for (int i = 0; i < n; i++) x[i] = 1.0f;
          break;
        default: break;
      }
    }
    CK(hipMemcpy(d_x, h.data(), h.size() * 4, hipMemcpyHostToDevice));
    gpu_topk(d_x, n, R, k, d_ctrl, d_hist, d_ids, d_vals, d_cn);
    CK(hipDeviceSynchronize());
    std::vector<int> cn(R), ids((size_t)R * TCAP);
    std::vector<float> vals((size_t)R * TCAP);
    CK(hipMemcpy(cn.data(), d_cn, R * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(ids.data(), d_ids, ids.size() * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(vals.data(), d_vals, vals.size() * 4, hipMemcpyDeviceToHost));
    for (int r = 0; r < R; r++) {
      const float* x = h.data() + (size_t)r * n;
      std::vector<int> ref(n);
      for (int i = 0; i < n; i++) ref[i] = i;
      auto cmp = [&](int a, int b) {
        if (x[a] != x[b]) return x[a] > x[b];
        return a < b;
      };
      std::partial_sort(ref.begin(), ref.begin() + k, ref.end(), cmp);
      // expected candidate count: all keys >= key of the k-th value's 24-bit
      // prefix bucket; only checked as "cn >= k" here
      if (cn[r] > TCAP) {
        overflow++;
        if (r != 5 && r != 1 && r != 4) {
          printf("    topk k=%d row %d: unexpected overflow cn=%d\n", k, r, cn[r]);
          wrong++;
        }
        continue;
      }
      checked++;
      if (r == 5) {
        printf("    topk k=%d row 5 (all equal): expected overflow, got cn=%d\n", k, cn[r]);
        wrong++;
        continue;
      }
      if (cn[r] < k) {
        printf("    topk k=%d row %d: only %d candidates\n", k, r, cn[r]);
        wrong++;
        continue;
      }
      const int m = cn[r];
      const int* ci = ids.data() + (size_t)r * TCAP;
      const float* cv = vals.data() + (size_t)r * TCAP;
      std::vector<int> ord(m);
      for (int j = 0; j < m; j++) ord[j] = j;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (cv[a] != cv[b]) return cv[a] > cv[b];
        return ci[a] < ci[b];
      });
      bool ok = true;
      for (int j = 0; j < k; j++)
        if (ci[ord[j]] != ref[j] || cv[ord[j]] != x[ref[j]]) ok = false;
      for (int j = 0; j < m; j++)
        if (x[ci[j]] != cv[j]) ok = false;
      if (!ok) {
        printf("    topk k=%d row %d: top-k mismatch (cn=%d)\n", k, r, m);
        wrong++;
      }
    }
  }
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  float t = 0;
  for (int it = 0; it < 2; it++) {
    CK(hipEventRecord(e0));
    for (int i = 0; i < 50; i++)
      gpu_topk(d_x, n, 4, 20, d_ctrl, d_hist, d_ids, d_vals, d_cn);
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    CK(hipEventElapsedTime(&t, e0, e1));
  }
  printf("  topk n=%d: %d rows checked, %d overflow (fallback), %d wrong; "
         "4 rows k=20 %.1f us/call\n", n, checked, overflow, wrong, t * 1000 / 50);
  if (wrong) bad("radix top-k differs from CPU top-k");
  CK(hipFree(d_x));
  CK(hipFree(d_ctrl));
  CK(hipFree(d_hist));
  CK(hipFree(d_ids));
  CK(hipFree(d_vals));
  CK(hipFree(d_cn));
}

static void test_corr(int n, float repeat) {
  std::mt19937 rng(7);
  const int R = 4, PC = 64;
  std::vector<float> h((size_t)R * n);
  for (auto& v : h) v = (float)(rng() % 1000) * 0.01f;
  std::vector<int> bi = {5, 17, n - 1}, hi = {17, 300, 9000};
  std::vector<float> bv = {1.5f, -2.0f, 3.0f}, hv = {0.25f, 1.0f, 0.5f};
  h[5] = -1.0f;
  h[17] = 1.0f;
  h[n - 1] = -3.0f;
  std::vector<int> pi((size_t)R * PC, 0), pn = {0, 1, 2, 3};
  std::vector<float> pv((size_t)R * PC, 0.0f);
  for (int r = 0; r < R; r++)
    for (int j = 0; j < pn[r]; j++) {
      pi[(size_t)r * PC + j] = 300 + j * 11;
      pv[(size_t)r * PC + j] = 0.125f * (j + 1);
    }
  std::vector<float> ref = h;
  auto penalize = [&](float& x, float p) {
    if (repeat != 1.0f) x = x <= 0.0f ? x * repeat : x / repeat;
    x -= p;
  };
  for (int r = 0; r < R; r++) {
    float* x = ref.data() + (size_t)r * n;
    for (size_t j = 0; j < bi.size(); j++) x[bi[j]] += bv[j];
    for (size_t j = 0; j < hi.size(); j++) penalize(x[hi[j]], hv[j]);
    for (int j = 0; j < pn[r]; j++) penalize(x[pi[(size_t)r * PC + j]], pv[(size_t)r * PC + j]);
  }
  float *d_x, *d_bv, *d_hv, *d_pv;
  int *d_bi, *d_hi, *d_pi, *d_pn;
  CK(hipMalloc(&d_x, h.size() * 4));
  CK(hipMalloc(&d_bi, 64));
  CK(hipMalloc(&d_bv, 64));
  CK(hipMalloc(&d_hi, 64));
  CK(hipMalloc(&d_hv, 64));
  CK(hipMalloc(&d_pi, pi.size() * 4));
  CK(hipMalloc(&d_pv, pv.size() * 4));
  CK(hipMalloc(&d_pn, R * 4));
  CK(hipMemcpy(d_x, h.data(), h.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_bi, bi.data(), bi.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_bv, bv.data(), bv.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_hi, hi.data(), hi.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_hv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_pi, pi.data(), pi.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_pv, pv.data(), pv.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_pn, pn.data(), R * 4, hipMemcpyHostToDevice));
  k_corr_apply<<<dim3(1, R), 256>>>(d_x, n, d_bi, d_bv, (int)bi.size(), d_hi,
                                     d_hv, (int)hi.size(), d_pi, d_pv, d_pn, PC, repeat);
  CK(hipDeviceSynchronize());
  std::vector<float> out(h.size());
  CK(hipMemcpy(out.data(), d_x, out.size() * 4, hipMemcpyDeviceToHost));
  const bool ok = memcmp(out.data(), ref.data(), out.size() * 4) == 0;
  printf("  corr_apply repeat=%g: %s\n", repeat, ok ? "bit-exact vs CPU" : "MISMATCH");
  if (!ok) bad("k_corr_apply differs from CPU");
  for (void* p : {(void*)d_x, (void*)d_bi, (void*)d_bv, (void*)d_hi, (void*)d_hv,
                  (void*)d_pi, (void*)d_pv, (void*)d_pn})
    CK(hipFree(p));
}

int main() {
  for (int n : {248320, 151937, 4099}) test_argmax(n);
  for (int n : {248320, 151937}) test_topk(n);
  for (float repeat : {1.0f, 1.03f, 0.8f}) test_corr(248320, repeat);
  printf(fails ? "FAIL\n" : "PASS\n");
  return fails ? 1 : 0;
}
