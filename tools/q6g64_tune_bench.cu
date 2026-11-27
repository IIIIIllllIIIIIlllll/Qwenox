// q6g64_tune_bench.cu — q6g64 GEMV kernel variants: bit-identity check vs the
// production kernel + timing on the G3 decode shapes.
//   q6g64_tune_bench [iters]
// kernels: q6g64 section of 28_kernels_hgn_v2.inc, cut out at build time:
//   sed -n "/^\/\/ --- q6g64 (dtype 24)/,/^\/\/ --- k_q6g64_wmma/p" \
//     src/gpu/parts/28_kernels_hgn_v2.inc > build/q6g64_part.inc
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace hv2 {
__device__ __forceinline__ void ldx8(const float* x, float* o) {
  const float4 a = *(const float4*)x, b = *(const float4*)(x + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
  o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
__device__ __forceinline__ void ldx8(const uint16_t* x, float* o) {
  const uint4 v = *(const uint4*)x;
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    o[2 * i] = __uint_as_float(w[i] << 16);
    o[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
}  // namespace hv2

#include "q6g64_part.inc"

// ---------------- V4: prefetch + 2 rows/warp (bit-identical per row) -------
// Warp handles rows r0 and r0+1 interleaved: each row keeps its per-lane
// group order (g = li; g += LPR) and reduction tree of k_q6g64_gemv, so both
// rows are bit-identical to the production kernel. Doubles per-lane work and
// halves the grid. Odd row counts read row0 twice; the duplicate is not
// stored (as k_q8g32_gemv_mr).
template <int P, typename TX, int LPR>
__global__ __launch_bounds__(256) void k_q6g64_gemv_pf(const uint8_t* __restrict__ wq,
                                                       const TX* __restrict__ x, uint64_t xs,
                                                       float* __restrict__ y, uint64_t ys, int R,
                                                       int C, int st) {
  const int tid = threadIdx.x;
  const int r0 = (blockIdx.x * (256 / LPR) + tid / LPR) * 2;
  const int li = tid % LPR;
  if (r0 >= R) return;
  const bool has1 = r0 + 1 < R;
  const uint8_t* row0 = wq + (size_t)r0 * st;
  const uint8_t* row1 = wq + (has1 ? r0 + 1 : r0) * st;
  const int ng = C >> 6;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) { acc0[p] = 0.f; acc1[p] = 0.f; }
  uint4 a_l0, a_l1, a_hp, b_l0, b_l1, b_hp;
  uint32_t a_sb = 0, b_sb = 0;
  const bool has0 = li < ng;
  if (has0) {
    a_l0 = *(const uint4*)(row0 + li * 32);
    a_l1 = *(const uint4*)(row0 + li * 32 + 16);
    a_hp = *(const uint4*)(row0 + C / 2 + li * 16);
    a_sb = *(const uint32_t*)(row0 + 3 * C / 4 + li * 4);
    b_l0 = *(const uint4*)(row1 + li * 32);
    b_l1 = *(const uint4*)(row1 + li * 32 + 16);
    b_hp = *(const uint4*)(row1 + C / 2 + li * 16);
    b_sb = *(const uint32_t*)(row1 + 3 * C / 4 + li * 4);
  }
  for (int g = li; g < ng; g += LPR) {
    const uint4 al0 = a_l0, al1 = a_l1, ahp = a_hp, bl0 = b_l0, bl1 = b_l1, bhp = b_hp;
    const uint32_t asb = a_sb, bsb = b_sb;
    const int gn = g + LPR;
    if (gn < ng) {
      a_l0 = *(const uint4*)(row0 + gn * 32);
      a_l1 = *(const uint4*)(row0 + gn * 32 + 16);
      a_hp = *(const uint4*)(row0 + C / 2 + gn * 16);
      a_sb = *(const uint32_t*)(row0 + 3 * C / 4 + gn * 4);
      b_l0 = *(const uint4*)(row1 + gn * 32);
      b_l1 = *(const uint4*)(row1 + gn * 32 + 16);
      b_hp = *(const uint4*)(row1 + C / 2 + gn * 16);
      b_sb = *(const uint32_t*)(row1 + 3 * C / 4 + gn * 4);
    }
    const float sca = __half2float(__ushort_as_half((unsigned short)(asb & 0xffffu)));
    const float bia = __half2float(__ushort_as_half((unsigned short)(asb >> 16)));
    const float scb = __half2float(__ushort_as_half((unsigned short)(bsb & 0xffffu)));
    const float bib = __half2float(__ushort_as_half((unsigned short)(bsb >> 16)));
    const uint32_t lw_a[8] = {al0.x, al0.y, al0.z, al0.w, al1.x, al1.y, al1.z, al1.w};
    const uint32_t hw_a[4] = {ahp.x, ahp.y, ahp.z, ahp.w};
    const uint32_t lw_b[8] = {bl0.x, bl0.y, bl0.z, bl0.w, bl1.x, bl1.y, bl1.z, bl1.w};
    const uint32_t hw_b[4] = {bhp.x, bhp.y, bhp.z, bhp.w};
    float dx0[P], sx0[P], dx1[P], sx1[P];
#pragma unroll
    for (int p = 0; p < P; p++) { dx0[p] = sx0[p] = 0.f; dx1[p] = sx1[p] = 0.f; }
#pragma unroll
    for (int i8 = 0; i8 < 8; i8++) {
      float cf_a[8], cf_b[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t lo = (lw_a[i8] >> (4 * e)) & 15u;
        const uint32_t hi = (hw_a[i8 >> 1] >> (2 * ((i8 & 1) * 8 + e))) & 3u;
        cf_a[e] = (float)(lo | (hi << 4));
        const uint32_t lob = (lw_b[i8] >> (4 * e)) & 15u;
        const uint32_t hib = (hw_b[i8 >> 1] >> (2 * ((i8 & 1) * 8 + e))) & 3u;
        cf_b[e] = (float)(lob | (hib << 4));
      }
#pragma unroll
      for (int p = 0; p < P; p++) {
        float xv[8];
        hv2::ldx8(x + (size_t)p * xs + g * 64 + i8 * 8, xv);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          dx0[p] = fmaf(cf_a[e], xv[e], dx0[p]);
          sx0[p] += xv[e];
          dx1[p] = fmaf(cf_b[e], xv[e], dx1[p]);
          sx1[p] += xv[e];
        }
      }
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      acc0[p] += sca * dx0[p] + bia * sx0[p];
      acc1[p] += scb * dx1[p] + bib * sx1[p];
    }
  }
  if constexpr (LPR <= 32) {
#pragma unroll
    for (int p = 0; p < P; p++) {
#pragma unroll
      for (int off = LPR / 2; off > 0; off >>= 1) {
        acc0[p] += __shfl_xor(acc0[p], off, LPR);
        acc1[p] += __shfl_xor(acc1[p], off, LPR);
      }
    }
    if (li == 0)
#pragma unroll
      for (int p = 0; p < P; p++) {
        y[(size_t)p * ys + r0] = acc0[p];
        if (has1) y[(size_t)p * ys + r0 + 1] = acc1[p];
      }
  }
}

#define CK(x)                                                                    \
  do {                                                                           \
    hipError_t e_ = (x);                                                         \
    if (e_ != hipSuccess) {                                                      \
      fprintf(stderr, "HIP %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 200;
  struct Shape { int rows, cols; const char* what; };
  const Shape shapes[] = {{10240, 2560, "in_proj_qkv"}, {6144, 2560, "in_proj_z"},
                          {2560, 6144, "out_proj"},       {12288, 2560, "q_proj"},
                          {512, 2560, "k/v proj"},        {10240, 320, "HC up"},
                          {2560, 640, "shexp down"},      {640, 2560, "shexp gate/up"},
                          {32768, 2560, "lm_head slice"}};
  std::mt19937 rng(1);
  bool allok = true;
  for (const Shape& sh : shapes) {
    const uint64_t R = sh.rows, C = sh.cols, st = q6g64_stride(C);
    const uint64_t wb = R * st;
    const int ncopy = (int)std::max<uint64_t>(1, (96ull << 20) / wb + 1);
    uint8_t* dw;
    CK(hipMalloc(&dw, wb * ncopy));
    std::vector<uint8_t> h(wb);
    for (size_t i = 0; i < wb; i++) h[i] = (uint8_t)(rng() & 0xFF);
    for (int c = 0; c < ncopy; c++) CK(hipMemcpy(dw + wb * c, h.data(), wb, hipMemcpyHostToDevice));
    std::vector<float> x(C);
    for (auto& v : x) v = (float)((int)(rng() % 2001) - 1000) / 1000.f;
    float *dx, *dy, *dr;
    CK(hipMalloc(&dx, C * 4));
    CK(hipMalloc(&dy, R * 4));
    CK(hipMalloc(&dr, R * 4));
    CK(hipMemcpy(dx, x.data(), C * 4, hipMemcpyHostToDevice));
    const int LPR = C < 1024 ? 8 : 32;
    auto old_k = [&](int c, float* out) {
      const uint8_t* wq = dw + (size_t)c * wb;
      if (LPR == 8)
        k_q6g64_gemv<1, float, 8><<<(unsigned)((R + 31) / 32), 256>>>(wq, dx, C, out, R, (int)R, (int)C, (int)st);
      else
        k_q6g64_gemv<1, float, 32><<<(unsigned)((R + 7) / 8), 256>>>(wq, dx, C, out, R, (int)R, (int)C, (int)st);
    };
    auto new_k = [&](int c, float* out) {
      const uint8_t* wq = dw + (size_t)c * wb;
      if (LPR == 8)
        k_q6g64_gemv<1, float, 8><<<(unsigned)((R + 31) / 32), 256>>>(wq, dx, C, out, R, (int)R, (int)C, (int)st);
      else
        k_q6g64_gemv_r2<1, float><<<(unsigned)((R + 15) / 16), 256>>>(wq, dx, C, out, R, (int)R, (int)C, (int)st);
    };
    old_k(0, dr);
    new_k(0, dy);
    CK(hipDeviceSynchronize());
    std::vector<float> hy(R), hr(R);
    CK(hipMemcpy(hy.data(), dy, R * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hr.data(), dr, R * 4, hipMemcpyDeviceToHost));
    uint64_t bad = 0;
    for (uint64_t i = 0; i < R; i++)
      if (*(uint32_t*)&hy[i] != *(uint32_t*)&hr[i]) bad++;
    auto time_k = [&](auto& k) {
      for (int i = 0; i < 8; i++) k(i % ncopy, dy);
      CK(hipDeviceSynchronize());
      std::vector<float> ts(iters);
      for (int i = 0; i < iters; i++) {
        hipEvent_t t0, t1;
        CK(hipEventCreate(&t0));
        CK(hipEventCreate(&t1));
        CK(hipEventRecord(t0));
        k(i % ncopy, dy);
        CK(hipEventRecord(t1));
        CK(hipEventSynchronize(t1));
        float ms;
        CK(hipEventElapsedTime(&ms, t0, t1));
        ts[i] = ms * 1000.f;
        CK(hipEventDestroy(t0));
        CK(hipEventDestroy(t1));
      }
      std::sort(ts.begin(), ts.end());
      return (double)ts[iters / 2];
    };
    const double old_us = time_k(old_k), new_us = time_k(new_k);
    printf("%-16s %5d x %-5d: old %7.1fus (%3.0f GB/s)  pf %7.1fus (%3.0f GB/s)  %+5.1f%%  bits-mismatch=%llu %s\n",
           sh.what, sh.rows, sh.cols, old_us, wb / old_us / 1e3, new_us, wb / new_us / 1e3,
           (old_us / new_us - 1) * 100.0, (unsigned long long)bad, bad ? "FAIL" : "OK");
    if (bad) allok = false;
    CK(hipFree(dw));
    CK(hipFree(dx));
    CK(hipFree(dy));
    CK(hipFree(dr));
  }
  printf(allok ? "ALL OK\n" : "BIT MISMATCH PRESENT\n");
  return allok ? 0 : 1;
}
