// ---------------- Lightning indexer: fused score + top-K select kernels ----
// Self-contained (HIP only, no engine globals) so tools/index_fast_test.cu can
// include it next to the production kernels it is checked against.
#include <hip/hip_fp16.h>
#ifndef INDEX_BUDGET
#define INDEX_BUDGET 1024
#endif
__device__ int g_index_budget_d = 512;
//
// Old (reference) kernels, moved verbatim from 10_ple_io.inc:
//   k_index_select_rs      n <= 8192, LDS row cache, 4 MSD radix passes
//   k_index_select_stream  n  > 8192, 5 global passes over the score row
//   k_index_scores_tiled   16q x 32k tile, fp32 FMA, rocBLAS K-loop start
// Fast bit-exact replacements (2026-09-28, 128K prefill work):
//   k_index_scores_t64     64q x 128k tile, 4q x 4h x 8k per thread
//   k_index_select_2p      2 global passes (8192-bin histogram + LDS candidates)
// C1 (2026-10-03): k_index_scores_bf16 (bf16 WMMA) + ST-templated selectors
// reading bf16 score rows (NOT bit-identical to fp32; overlap/KLD gated).

// score-load helpers: fp32 identity / bf16 -> fp32 (exact, bits << 16) /
// f16 -> fp32 (exact __half2float; f16 subset of f32). Non-negative f16 bit
// patterns are already monotone as raw uint16 (scores are ReLU-clamped), but
// converting to exact f32 keeps the histogram/radix code identical to fp32.
__device__ __forceinline__ float isel_ldf(float f) { return f; }
__device__ __forceinline__ float isel_ldf(uint16_t u) {
  return __uint_as_float((unsigned)u << 16);
}
__device__ __forceinline__ float isel_ldf(__half h) { return __half2float(h); }
__device__ __forceinline__ unsigned isel_bits(float f) {
  return __float_as_uint(f);
}
__device__ __forceinline__ unsigned isel_bits(uint16_t u) {
  return (unsigned)u << 16;
}
__device__ __forceinline__ unsigned isel_bits(__half h) {
  return __float_as_uint(__half2float(h));
}

// select f16 向量化（§10-5 小项）：uint4 一次 8 个 f16 + 对齐 peel，替代
// ST=__half 行的 2B 散读。仅对 __half 启用（uint16_t 是 bf16，位义不同）。
// 纯读法改动：每元素仍走 __half2float 精确转换，只用于顺序无关的消费点
// （直方图计数 / atomic 位置发射），输出逐 bit 不变。
template <class T> struct isel_is_h16 { enum { value = false }; };
template <> struct isel_is_h16<__half> { enum { value = true }; };
struct IselH16Row {
  const __half* rp;
  int lead, nv, n;
  __device__ __forceinline__ void init(const __half* r, int n_) {
    rp = r;
    n = n_;
    lead = min((int)(((16 - ((uintptr_t)rp & 15)) & 15) >> 1), n);
    nv = (n - lead) >> 3;
  }
  __device__ __forceinline__ const uint4* vec() const {
    return reinterpret_cast<const uint4*>(rp + lead);
  }
  __device__ __forceinline__ int tail0() const { return lead + (nv << 3); }
};
__device__ __forceinline__ void isel_h16x8_bits(uint4 u, unsigned bits[8]) {
  const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    bits[2 * k] = __float_as_uint(
        __half2float(__ushort_as_half((unsigned short)(w[k] & 0xffffu))));
    bits[2 * k + 1] = __float_as_uint(
        __half2float(__ushort_as_half((unsigned short)(w[k] >> 16))));
  }
}

// Exact top-K selection, bit-identical output to k_index_select (same
// combined-score formula and the same total order: score desc, then smaller
// block id), but sort-free. Four MSD radix passes over the fp32 bit pattern
// (scores are nonnegative, so bits are monotone) resolve the exact Kth
// score threshold s*; the selected set is {s > s*} plus the smallest-id
// elements of {s == s*}, which is exactly the old kernel's uint64-key order.
// Emission scans b ascending, so ids come out in the same ascending order the
// old kernel's final IdSort produced. Requires n <= 8192 (LDS combined-score
// cache); larger n keeps the old kernel. n <= g_index_budget_d fills directly (same as old).
template <bool Combined = false, class ST = float>
__global__ void k_index_select_rs(const ST* __restrict__ scores, int stride,
                                  int* __restrict__ selected, int first,
                                  const int* __restrict__ pos = nullptr) {
  __shared__ float sbuf[8192];
  __shared__ unsigned hist[256];
  __shared__ unsigned shr_cnt[8];
  __shared__ unsigned shr_eq[8];
  __shared__ int s_passcnt, s_passbyte, s_cntgt;
  const int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= g_index_budget_d) {
    for (int i = tid; i < g_index_budget_d; i += 256)
      selected[(size_t)q * g_index_budget_d + i] = i < n ? i : -1;
    return;
  }
  // combine (ReLU per head, ordered sum, scale) into LDS + top-byte histogram
  hist[tid] = 0;
  __syncthreads();
  const ST* qb = scores + (size_t)q * 4 * stride;
  if constexpr (Combined && isel_is_h16<ST>::value) {
    // f16 向量化读（顺序无关：sbuf 按位写 + 直方图计数）
    IselH16Row R;
    R.init(reinterpret_cast<const __half*>(scores + (size_t)q * stride), n);
    for (int b = tid; b < R.lead; b += 256) {
      float s = isel_ldf(R.rp[b]);
      sbuf[b] = s;
      atomicAdd(&hist[__float_as_uint(s) >> 24], 1u);
    }
    const uint4* rv = R.vec();
    for (int c = tid; c < R.nv; c += 256) {
      unsigned bits[8];
      isel_h16x8_bits(rv[c], bits);
      const int b0 = R.lead + (c << 3);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sbuf[b0 + j] = __uint_as_float(bits[j]);
        atomicAdd(&hist[bits[j] >> 24], 1u);
      }
    }
    for (int b = R.tail0() + tid; b < n; b += 256) {
      float s = isel_ldf(R.rp[b]);
      sbuf[b] = s;
      atomicAdd(&hist[__float_as_uint(s) >> 24], 1u);
    }
  } else
  for (int b = tid; b < n; b += 256) {
    float s = 0.f;
    if constexpr (Combined) {
      s = isel_ldf(scores[(size_t)q * stride + b]);
    } else {
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, qb[(size_t)h * stride + b]);
      s *= 0.08838834764831845f;
    }
    sbuf[b] = s;
    atomicAdd(&hist[__float_as_uint(s) >> 24], 1u);
  }
  __syncthreads();
  // 4 MSD radix passes -> exact Kth-largest score bits in prefix
  unsigned prefix = 0;
  int K = g_index_budget_d;
  if (tid == 0) s_cntgt = 0;
  for (int p = 0; p < 4; p++) {
    const int shift = (3 - p) * 8;
    if (p) {
      __syncthreads();  // hist reuse
      hist[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        unsigned bits = __float_as_uint(sbuf[b]);
        if ((bits >> (shift + 8)) == prefix)
          atomicAdd(&hist[(bits >> shift) & 0xffu], 1u);
      }
      __syncthreads();
    }
    if (tid == 0) {
      int acc = 0, chosen = 255;
      for (int i = 255; i >= 0; i--) {
        int c = (int)hist[i];
        if (acc + c >= K) {
          chosen = i;
          break;
        }
        acc += c;
      }
      s_passcnt = acc;
      s_passbyte = chosen;
    }
    __syncthreads();
    K -= s_passcnt;
    if (tid == 0) s_cntgt += s_passcnt;
    prefix = (prefix << 8) | (unsigned)s_passbyte;
    __syncthreads();
  }
  const unsigned sstar = prefix;
  const int cnt_gt = s_cntgt;  // < g_index_budget_d by construction
  const int need = g_index_budget_d - cnt_gt;  // tie slots taken by smallest ids
  // Single ascending-b emit: sel = (s > s*) || (s == s* && tie-rank < need).
  // The old kernel's final IdSort yields ascending ids, so ties and strict
  // greaters must interleave in one id-ordered scan (not two phases).
  int outbase = 0, eqbase = 0;
  const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int b = c0 + tid;
    unsigned bits = b < n ? __float_as_uint(sbuf[b]) : 0u;
    bool gt = b < n && bits > sstar;
    bool eq = b < n && bits == sstar;
    unsigned long long bal_eq = __ballot_sync(~0ull, eq);
    int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
    if (lane == 0) shr_eq[w] = __popcll(bal_eq);
    __syncthreads();
    int eq_woff = 0;
    for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
    bool sel = gt || (eq && eqbase + eq_woff + eq_pre < need);
    unsigned long long bal_sel = __ballot_sync(~0ull, sel);
    if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
    __syncthreads();
    int woff = 0, wtot = 0, eqtot = 0;
    for (int i = 0; i < nw; i++) {
      int c = (int)shr_cnt[i];
      woff += i < w ? c : 0;
      wtot += c;
      eqtot += (int)shr_eq[i];
    }
    if (sel)
      selected[(size_t)q * g_index_budget_d + outbase + woff +
               __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
    outbase += wtot;
    eqbase = min(eqbase + eqtot, need);
    __syncthreads();  // shr reuse
  }
}


// Stream the existing combined scores instead of storing an unbounded row in LDS.
// Four exact radix passes find the threshold, then an ascending scan emits IDs.
// The total ordering and tie handling are identical to the existing selectors.
template <bool Combined>
__device__ __forceinline__ float index_stream_score(const float* scores, int stride, int q, int b) {
  if constexpr (Combined) return scores[(size_t)q * stride + b];
  float s = 0.f;
  #pragma unroll
  for (int h = 0; h < 4; h++) s += fmaxf(0.f, scores[((size_t)q * 4 + h) * stride + b]);
  return s * 0.08838834764831845f;
}
template <bool Combined = false>
__global__ void k_index_select_stream(const float* __restrict__ scores, int stride,
                                  int* __restrict__ selected, int first,
                                  const int* __restrict__ pos = nullptr) {
  __shared__ unsigned s_bits[8192];
  __shared__ unsigned wh[8][256];
  __shared__ unsigned hist[256];
  __shared__ unsigned shr_cnt[8];
  __shared__ unsigned shr_eq[8];
  __shared__ int s_passcnt, s_passbyte, s_cntgt;
  const int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
  if (n <= g_index_budget_d) {
    for (int i = tid; i < g_index_budget_d; i += 256)
      selected[(size_t)q * g_index_budget_d + i] = i < n ? i : -1;
    return;
  }
  const bool use_lds = (n <= 8192);
  for (int rw = 0; rw < 8; rw++) wh[rw][tid] = 0;
  __syncthreads();
  const float* qb = scores + (size_t)q * 4 * stride;
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int b = c0 + tid;
    unsigned bits = 0u;
    if (b < n) {
      float s = 0.f;
      if constexpr (Combined) {
        s = scores[(size_t)q * stride + b];
      } else {
#pragma unroll
        for (int h = 0; h < 4; h++) s += fmaxf(0.f, qb[(size_t)h * stride + b]);
        s *= 0.08838834764831845f;
      }
      bits = __float_as_uint(s);
      if (use_lds) s_bits[b] = bits;
    }
    const unsigned topb = bits >> 24;
    const unsigned long long zmask = __ballot_sync(~0ull, b < n && topb == 0u);
    if (zmask && lane == (__ffsll(zmask) - 1)) wh[w][0] += (unsigned)__popcll(zmask);
    if (b < n && topb != 0u) atomicAdd(&wh[w][topb], 1u);
  }
  __syncthreads();
  {
    unsigned hs = 0;
#pragma unroll
    for (int rw = 0; rw < 8; rw++) hs += wh[rw][tid];
    hist[tid] = hs;
  }
  __syncthreads();
  // 4 MSD radix passes -> exact Kth-largest score bits in prefix
  unsigned prefix = 0;
  int K = g_index_budget_d;
  if (tid == 0) s_cntgt = 0;
  for (int p = 0; p < 4; p++) {
    const int shift = (3 - p) * 8;
    if (p) {
      for (int rw = 0; rw < 8; rw++) wh[rw][tid] = 0;
      __syncthreads();
      for (int c0 = 0; c0 < n; c0 += 256) {
        const int b = c0 + tid;
        unsigned bits = 0u;
        bool match = false;
        if (b < n) {
          bits = use_lds ? s_bits[b]
                         : __float_as_uint(index_stream_score<Combined>(scores, stride, q, b));
          match = ((bits >> (shift + 8)) == prefix);
        }
        const unsigned bin = (bits >> shift) & 0xffu;
        const unsigned long long zmask = __ballot_sync(~0ull, match && bin == 0u);
        if (zmask && lane == (__ffsll(zmask) - 1)) wh[w][0] += (unsigned)__popcll(zmask);
        if (match && bin != 0u) atomicAdd(&wh[w][bin], 1u);
      }
      __syncthreads();
      unsigned hs = 0;
#pragma unroll
      for (int rw = 0; rw < 8; rw++) hs += wh[rw][tid];
      hist[tid] = hs;
      __syncthreads();
    }
    if (tid == 0) {
      int acc = 0, chosen = 255;
      for (int i = 255; i >= 0; i--) {
        int c = (int)hist[i];
        if (acc + c >= K) {
          chosen = i;
          break;
        }
        acc += c;
      }
      s_passcnt = acc;
      s_passbyte = chosen;
    }
    __syncthreads();
    K -= s_passcnt;
    if (tid == 0) s_cntgt += s_passcnt;
    prefix = (prefix << 8) | (unsigned)s_passbyte;
    __syncthreads();
  }
  const unsigned sstar = prefix;
  const int cnt_gt = s_cntgt;  // < g_index_budget_d by construction
  const int need = g_index_budget_d - cnt_gt;  // tie slots taken by smallest ids
  // Single ascending-b emit: sel = (s > s*) || (s == s* && tie-rank < need).
  // The old kernel's final IdSort yields ascending ids, so ties and strict
  // greaters must interleave in one id-ordered scan (not two phases).
  int outbase = 0, eqbase = 0;
#pragma unroll 1
  for (int c0 = 0; c0 < n; c0 += 256) {
    const int b = c0 + tid;
    unsigned bits = b < n ? (use_lds ? s_bits[b]
                                     : __float_as_uint(index_stream_score<Combined>(scores, stride, q, b)))
                          : 0u;
    bool gt = b < n && bits > sstar;
    bool eq = b < n && bits == sstar;
    unsigned long long bal_eq = __ballot_sync(~0ull, eq);
    int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
    if (lane == 0) shr_eq[w] = __popcll(bal_eq);
    __syncthreads();
    int eq_woff = 0;
    for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
    bool sel = gt || (eq && eqbase + eq_woff + eq_pre < need);
    unsigned long long bal_sel = __ballot_sync(~0ull, sel);
    if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
    __syncthreads();
    int woff = 0, wtot = 0, eqtot = 0;
    for (int i = 0; i < nw; i++) {
      int c = (int)shr_cnt[i];
      woff += i < w ? c : 0;
      wtot += c;
      eqtot += (int)shr_eq[i];
    }
    if (sel)
      selected[(size_t)q * g_index_budget_d + outbase + woff +
               __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
    outbase += wtot;
    eqbase = min(eqbase + eqtot, need);
    __syncthreads();  // shr reuse
  }
}



// 16 queries x 32 keys: fuse FP32 heads, ReLU and combine, skipping
// causally unreachable tiles. rocBLAS gfx1151 MT32x32x8 SU32 SUS256
// starts the K loop at ((key / 32) & 1) * 64 and wraps at 128.
// Keep one FMA chain per head and the same ordered head reduction.
__global__ void __launch_bounds__(256)
k_index_scores_tiled(const float* __restrict__ q,
                     const float* __restrict__ keys,
                     float* __restrict__ out, int nb, int count, int first) {
  const int q0 = (int)blockIdx.y * 16, k0 = (int)blockIdx.x * 32;
  if (q0 >= count || k0 >= min(nb, (first + min(q0 + 16, count)) / 4)) return;
  __shared__ float qs[32][65];
  __shared__ float ks[32][33];
  const int tid = threadIdx.x, qr = tid / 16, kr = (tid % 16) * 2;
  float a[4][2] = {};
  for (int stage = 0; stage < 4; stage++) {
    const int d0 = (stage * 32 + ((int)blockIdx.x & 1) * 64) & 127;
    for (int j = tid; j < 512; j += 256) {
      int row = j / 8, d = (j % 8) * 4;
      float4 qv = make_float4(0, 0, 0, 0), kv = qv;
      if (q0 + row / 4 < count)
        qv = *reinterpret_cast<const float4*>(q + (size_t)(q0 * 4 + row) * 128 + d0 + d);
      if (row < 32 && k0 + row < nb)
        kv = *reinterpret_cast<const float4*>(keys + (size_t)(k0 + row) * 128 + d0 + d);
      qs[d][row] = qv.x; qs[d + 1][row] = qv.y;
      qs[d + 2][row] = qv.z; qs[d + 3][row] = qv.w;
      if (row < 32) {
        ks[d][row] = kv.x; ks[d + 1][row] = kv.y;
        ks[d + 2][row] = kv.z; ks[d + 3][row] = kv.w;
      }
    }
    __syncthreads();
#pragma unroll
    for (int d = 0; d < 32; d++) {
      float qv[4], kv[2];
#pragma unroll
      for (int h = 0; h < 4; h++) qv[h] = qs[d][qr * 4 + h];
#pragma unroll
      for (int c = 0; c < 2; c++) kv[c] = ks[d][kr + c];
#pragma unroll
      for (int h = 0; h < 4; h++)
#pragma unroll
        for (int c = 0; c < 2; c++) a[h][c] = fmaf(qv[h], kv[c], a[h][c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int c = 0; c < 2; c++) {
    const int t = q0 + qr, b = k0 + kr + c;
    if (t < count && b < nb && b < (first + t + 1) / 4) {
      float s = 0.f;
#pragma unroll
      for (int h = 0; h < 4; h++) s += fmaxf(0.f, a[h][c]);
      out[(size_t)t * nb + b] = s * 0.08838834764831845f;
    }
  }
}

// ---------------------------------------------------------------------------
// k_index_scores_t64: bit-exact replacement for k_index_scores_tiled.
//
// The old kernel is LDS-bandwidth bound: 6 scalar LDS reads per 8 FMA = 3 B
// per FMA, which is exactly the ~128 B/clk/WGP LDS limit at its measured
// 4.5 TFLOPS. Here a block covers 64 queries x 128 keys and a thread
// 4 queries x 4 heads x 8 keys (128 accumulators): per d it issues 4 + 2
// float4 LDS reads for 128 FMA (0.75 B/FMA).
// All 128 keys of a block share one K-loop parity: they are the four 32-key
// groups of parity p inside a 256-key super tile,
//   blockIdx.x = 2*m + p  ->  keys 256m + 32p + 64g + [0, 32),  g = 0..3
// so every key has the same rocBLAS K-loop start ((b / 32) & 1) * 64, and
// every (query, head, key) keeps exactly the old single FMA chain
//   a = fmaf(q[d], k[d], a),  d = start, start+1, ... (mod 128),  a0 = 0
// (16-d stages here vs 32-d there: same d sequence). The epilogue is the
// same expression: s = 0; s += fmaxf(0, a_h) for h = 0..3; s * 1/sqrt(128).
// The next stage's global loads are prefetched into registers while the
// current stage computes. Output / skip rules are unchanged:
// out[t * nb + b] written only for t < count, b < nb, b < (first + t + 1) / 4.
// Tile key j (0..127) -> key S + 64 * (j >> 5) + (j & 31), S = 256m + 32p.
// grid = (2 * ceil(nb / 256), ceil(count / 64)), block = 256.
__global__ void __launch_bounds__(256)
k_index_scores_t64(const float* __restrict__ q, const float* __restrict__ keys,
                   float* __restrict__ out, int nb, int count, int first) {
  const int par = (int)blockIdx.x & 1;
  const int S = ((int)blockIdx.x >> 1) * 256 + par * 32;
  const int q0 = (int)blockIdx.y * 64;
  if (q0 >= count || S >= min(nb, (first + min(q0 + 64, count)) / 4)) return;
  __shared__ __align__(16) float qs[16][260];  // [d][query*4 + head]
  __shared__ __align__(16) float ks[16][132];  // [d][tile key]
  const int tid = threadIdx.x;
  const int qg = tid >> 4, kg = tid & 15;  // thread keys: kg*4 + c + 64*e
  // global -> LDS: q = 256 rows x 4 float4 (4 per thread),
  //                k = 128 rows x 4 float4 (2 per thread)
  const float* kp[2];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int j = (tid + 256 * i) >> 2, key = S + 64 * (j >> 5) + (j & 31);
    kp[i] = key < nb ? keys + (size_t)key * 128 + (tid & 3) * 4 : nullptr;
  }
  float4 pq[4], pk[2];
  auto load = [&](int d0) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int j = tid + 256 * i, row = j >> 2, dd = (j & 3) * 4;
      pq[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (q0 + (row >> 2) < count)
        pq[i] = *reinterpret_cast<const float4*>(q + (size_t)(q0 * 4 + row) * 128 + d0 + dd);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      pk[i] = make_float4(0.f, 0.f, 0.f, 0.f);
      if (kp[i]) pk[i] = *reinterpret_cast<const float4*>(kp[i] + d0);
    }
  };
  float acc[4][4][8];  // [query][head][key]
#pragma unroll
  for (int a = 0; a < 4; a++)
#pragma unroll
    for (int h = 0; h < 4; h++)
#pragma unroll
      for (int c = 0; c < 8; c++) acc[a][h][c] = 0.f;
  load((par * 64) & 127);
#pragma unroll 1
  for (int s = 0; s < 8; s++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const int j = tid + 256 * i, row = j >> 2, dd = (j & 3) * 4;
      qs[dd][row] = pq[i].x; qs[dd + 1][row] = pq[i].y;
      qs[dd + 2][row] = pq[i].z; qs[dd + 3][row] = pq[i].w;
    }
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int j = (tid + 256 * i) >> 2, dd = (tid & 3) * 4;
      ks[dd][j] = pk[i].x; ks[dd + 1][j] = pk[i].y;
      ks[dd + 2][j] = pk[i].z; ks[dd + 3][j] = pk[i].w;
    }
    __syncthreads();
    if (s < 7) load(((s + 1) * 16 + par * 64) & 127);
#pragma unroll
    for (int d = 0; d < 16; d++) {
      const float4 a0 = *reinterpret_cast<const float4*>(&qs[d][qg * 16]);
      const float4 a1 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 4]);
      const float4 a2 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 8]);
      const float4 a3 = *reinterpret_cast<const float4*>(&qs[d][qg * 16 + 12]);
      const float4 k0 = *reinterpret_cast<const float4*>(&ks[d][kg * 4]);
      const float4 k1 = *reinterpret_cast<const float4*>(&ks[d][64 + kg * 4]);
      const float qv[16] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w,
                            a2.x, a2.y, a2.z, a2.w, a3.x, a3.y, a3.z, a3.w};
      const float kk[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
      for (int a = 0; a < 4; a++)
#pragma unroll
        for (int h = 0; h < 4; h++)
#pragma unroll
          for (int c = 0; c < 8; c++)
            acc[a][h][c] = fmaf(qv[a * 4 + h], kk[c], acc[a][h][c]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int a = 0; a < 4; a++) {
    const int t = q0 + qg * 4 + a;
    if (t >= count) continue;
    const int lim = min(nb, (first + t + 1) / 4);
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const int j = kg * 4 + (c & 3) + 64 * (c >> 2);
      const int b = S + 64 * (j >> 5) + (j & 31);
      if (b < lim) {
        float s = 0.f;
#pragma unroll
        for (int h = 0; h < 4; h++) s += fmaxf(0.f, acc[a][h][c]);
        out[(size_t)t * nb + b] = s * 0.08838834764831845f;
      }
    }
  }
}

static inline dim3 index_t64_grid(int nb, int count) {
  return dim3(2 * ((nb + 255) / 256), (count + 63) / 64);
}

// ---------------------------------------------------------------------------
// C1: bf16 WMMA score kernel + bf16 score storage. NOT bit-identical to the
// fp32 path (bf16 q/k rounding, wmma f32 accumulation order, bf16 score out);
// gated by selection overlap vs fp32 (tools/index_fast_test.cu, target
// >= 99.5%) + long-context KLD + needle retrieval.
using isel_shortx16 = __attribute__((ext_vector_type(16))) short;
using isel_floatx8 = __attribute__((ext_vector_type(8))) float;
__device__ __forceinline__ isel_shortx16 isel_ld16(const uint16_t* p) {
  short tmp[16];
  *(uint4*)&tmp[0] = *(const uint4*)p;
  *(uint4*)&tmp[8] = *(const uint4*)(p + 8);
  isel_shortx16 v;
  memcpy(&v, tmp, 32);
  return v;
}
__device__ __forceinline__ float isel_bf2f(uint16_t u) {
  return __uint_as_float((unsigned)u << 16);
}
__device__ __forceinline__ uint16_t isel_f2bf(float f) {  // round-to-nearest-even
  const unsigned u = __float_as_uint(f);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}
__device__ __forceinline__ uint16_t isel_f2h(float f) {  // RNE f16 bit pattern
  return __half_as_ushort(__float2half_rn(f));
}

// k_index_scores_wmma16<BF16>: score[t][b] (bf16/f16) = (sum_h max(0,
// q[t*4+h] . k[b])) * 1/sqrt(128), same output/skip rules as
// k_index_scores_t64: written only for t < count, b < nb,
// b < (first + t + 1) / 4; grid = (ceil(nb/128), ceil(count/16)),
// block = 256.
// CTA: 16 tokens (64 head-rows) x 128 keys, K=128 staged whole (no double
// buffer). Warp grid 2x4, each warp 32x32 (2x2 frags of wmma 16x16x16).
// The 8 frag rows a lane holds are exactly 2 tokens x 4 heads (row =
// token*4+head, rh a multiple of 8), so the ReLU+head-sum is intra-lane.
// wmma accumulates in f32; only the q/k inputs and the final score are
// rounded (bf16: RNE via isel_f2bf; f16: __float2half_rn). Inputs are
// RMS-normed indexer vectors (~unit scale), scores <= ~hundreds << 65504,
// so f16 range is ample; f16's 11-bit mantissa is 8x finer than bf16's.
template <bool BF16>
__global__ void __launch_bounds__(256)
k_index_scores_wmma16(const uint16_t* __restrict__ q,
                      const uint16_t* __restrict__ keys,
                      uint16_t* __restrict__ out, int nb, int count, int first) {
  constexpr int LSA = 128 + 8;        // LDS row stride (elems)
  constexpr int BREG = 8 * LSA + 8;   // keys 8-row region stride
  const int q0 = (int)blockIdx.y * 16, n0 = (int)blockIdx.x * 128;
  if (q0 >= count || n0 >= min(nb, (first + min(q0 + 16, count)) / 4)) return;
  __shared__ __align__(16) uint16_t As[64 * LSA];
  __shared__ __align__(16) uint16_t Bs[(128 / 8) * BREG];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  // stage q: 64 rows x 16 uint4 (4 per thread); keys: 128 x 16 (8 per thread)
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int idx = tid + j * 256, row = idx >> 4, c8 = (idx & 15) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (q0 + (row >> 2) < count)
      v = *(const uint4*)(q + (size_t)(q0 * 4 + row) * 128 + c8);
    *(uint4*)&As[row * LSA + c8] = v;
  }
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int idx = tid + j * 256, row = idx >> 4, c8 = (idx & 15) * 8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (n0 + row < nb)
      v = *(const uint4*)(keys + (size_t)(n0 + row) * 128 + c8);
    *(uint4*)&Bs[(row >> 3) * BREG + (row & 7) * LSA + c8] = v;
  }
  __syncthreads();
  // warp tile: 2x2 frags (32 rows x 32 keys)
  const int wm0 = (w >> 2) * 32, wp0 = (w & 3) * 32;
  // wmma A lane mapping identical to k_gemm_wmma (lane 2r -> row r, lanes
  // 17+2(r-8) -> row 8+r, complementary dummy rows on ignored lanes)
  const int base_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? base_ : ((base_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((base_ + 4) & 7) : base_);
  uint32_t aoff[2], boff[2];
#pragma unroll
  for (int i = 0; i < 2; i++) aoff[i] = ((wm0 + i * 16 + qrowA) * LSA) * 2;
#pragma unroll
  for (int j = 0; j < 2; j++) {
    const int pl = wp0 + j * 16 + (lane & 15);
    boff[j] = ((pl >> 3) * BREG + (pl & 7) * LSA) * 2;
  }
  isel_floatx8 acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
      for (int e = 0; e < 8; e++) acc[i][j][e] = 0.f;
#pragma unroll
  for (int kh = 0; kh < 8; kh++) {
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const isel_shortx16 af =
          isel_ld16((const uint16_t*)((const char*)&As[0] + aoff[i] + kh * 32));
#pragma unroll
      for (int j = 0; j < 2; j++) {
        const isel_shortx16 bfj =
            isel_ld16((const uint16_t*)((const char*)&Bs[0] + boff[j] + kh * 32));
        if constexpr (BF16)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bfj, acc[i][j]);
        else
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bfj, acc[i][j]);
      }
    }
  }
  // epilogue: lane covers rows rh..rh+7 (= 2 tokens x 4 heads), col cl
  const int rh = (lane >> 4) * 8, cl = lane & 15;
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int r0 = wm0 + i * 16 + rh;  // CTA-local head-row base (mult of 8)
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const int b = n0 + wp0 + j * 16 + cl;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int e = 0; e < 4; e++) {
        s0 += fmaxf(0.f, acc[i][j][e]);
        s1 += fmaxf(0.f, acc[i][j][4 + e]);
      }
      const int t0 = q0 + (r0 >> 2), t1 = t0 + 1;
      const float sc0 = s0 * 0.08838834764831845f;
      const float sc1 = s1 * 0.08838834764831845f;
      if (t0 < count && b < min(nb, (first + t0 + 1) / 4))
        out[(size_t)t0 * nb + b] = BF16 ? isel_f2bf(sc0) : isel_f2h(sc0);
      if (t1 < count && b < min(nb, (first + t1 + 1) / 4))
        out[(size_t)t1 * nb + b] = BF16 ? isel_f2bf(sc1) : isel_f2h(sc1);
    }
  }
}
static inline dim3 index_wmma16_grid(int nb, int count) {
  return dim3((nb + 127) / 128, (count + 15) / 16);
}

// ---------------------------------------------------------------------------
// k_index_select_2p: exact top-K over combined score rows, bit-identical to
// k_index_select_stream<true> / k_index_select_rs<true> / k_index_select
// (score desc, ties -> smaller block id; output = the K ids ascending;
// n <= g_index_budget_d -> 0..n-1 then -1). Two global passes instead of five:
//   1. 8192-bin histogram of bits >> 18 (scores are >= 0, so fp32 bits are
//      monotone and bit 31 is 0) -> threshold bin T, cnt_gt = #(bin > T)
//   2. bin > T -> selected directly; bin == T -> (bits, id) candidates in LDS
//   then, all in LDS: exact s* = (g_index_budget_d - cnt_gt)-th largest candidate (4 radix
//   passes), ties at s* resolved by the smallest ids (radix on ~id), and a
//   runtime-budget bitonic sort for the ascending output.
// If the threshold bin holds more than `cap` (<= ISEL_CAP) candidates the
// block falls back to exact streaming radix passes over the remaining 18
// bits + one ascending emit scan (same algorithm as k_index_select_stream).
// grid = count, block = 256. Row q: scores[q * stride + b], b < n.
constexpr int ISEL_BINS = 8192;
constexpr int ISEL_CAP = 4096;
// Histogram bin of a combined score: bits >> 18 (< 8192 for any score >= 0,
// which the engine always produces: 0 + ReLU terms). Clamped so a NaN /
// negative input can never index outside the LDS histogram.
__device__ __forceinline__ unsigned isel_bin(unsigned bits) { return min(bits >> 18, 8191u); }

// Block-wide: with hist[0 .. 256*PER) counting items per bin (ascending
// values), find the bin T holding the K-th largest item:
//   cum(bins > T) < K <= cum(bins >= T).
// Result: sh[0] = cum(bins > T), sh[1] = T. All 256 threads must call; the
// function ends with a barrier.
template <int PER>
__device__ __forceinline__ void isel_find_desc(const unsigned* hist, int K, int* sh,
                                               unsigned* wsum) {
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int hi = 256 * PER - 1 - tid * PER;  // this thread: bins hi, hi-1, ... (desc)
  unsigned part = 0;
#pragma unroll
  for (int i = 0; i < PER; i++) part += hist[hi - i];
  unsigned x = part;  // inclusive scan in tid order (= descending bins)
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const unsigned y = __shfl_up(x, o, 32);
    if (lane >= o) x += y;
  }
  if (lane == 31) wsum[w] = x;
  __syncthreads();
  unsigned off = 0;
  for (int i = 0; i < w; i++) off += wsum[i];
  const unsigned incl = x + off, excl = incl - part;
  if (excl < (unsigned)K && (unsigned)K <= incl) {  // exactly one thread
    unsigned acc = excl;
    for (int i = 0; i < PER; i++) {
      const unsigned c = hist[hi - i];
      if (acc + c >= (unsigned)K) {
        sh[0] = (int)acc;
        sh[1] = hi - i;
        break;
      }
      acc += c;
    }
  }
  __syncthreads();
}

// K-th largest (1-based, with multiplicity) 32-bit key among LDS items
// i < cnt for which get(i, key) is true (K must be <= that count). Returns the
// key; gt = number of those items with a strictly greater key.
template <class Get>
__device__ __forceinline__ unsigned isel_kth(int cnt, int K, Get get, unsigned* h256,
                                             int* sh, unsigned* wsum, int& gt) {
  unsigned prefix = 0;
  gt = 0;
#pragma unroll 1
  for (int p = 0; p < 4; p++) {
    const int shift = 24 - 8 * p;
    h256[threadIdx.x] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < cnt; i += 256) {
      unsigned key;
      if (get(i, key) && (p == 0 || (key >> (shift + 8)) == prefix))
        atomicAdd(&h256[(key >> shift) & 0xffu], 1u);
    }
    __syncthreads();
    isel_find_desc<1>(h256, K, sh, wsum);
    K -= sh[0];
    gt += sh[0];
    prefix = (prefix << 8) | (unsigned)sh[1];
    __syncthreads();  // sh / h256 reuse
  }
  return prefix;
}

template <class ST = float>
__global__ void __launch_bounds__(256)
k_index_select_2p(const ST* __restrict__ scores, int stride,
                  int* __restrict__ selected, int first, int cap = ISEL_CAP,
                  const int* __restrict__ pos = nullptr) {
  __shared__ unsigned big[ISEL_BINS];  // pass 1 histogram; then cand bits | cand ids
  __shared__ int sel[INDEX_BUDGET];
  __shared__ unsigned h256[256];
  __shared__ unsigned wsum[8];
  __shared__ unsigned shr_cnt[8], shr_eq[8];
  __shared__ int sh[2];
  __shared__ int s_nsel, s_ncand, s_eq;
  const int q = blockIdx.x, t = pos ? *pos : first + q, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= g_index_budget_d) {
    for (int i = tid; i < g_index_budget_d; i += 256)
      selected[(size_t)q * g_index_budget_d + i] = i < n ? i : -1;
    return;
  }
  cap = min(cap, ISEL_CAP);
  const ST* row = scores + (size_t)q * stride;
  for (int i = tid; i < ISEL_BINS; i += 256) big[i] = 0;
  if (tid == 0) s_nsel = s_ncand = s_eq = 0;
  __syncthreads();
  // ---- pass 1: top-13-bit histogram (bit 31 is 0 -> 8192 bins)
  if constexpr (isel_is_h16<ST>::value) {
    // f16 向量化读（顺序无关的直方图计数）
    IselH16Row R;
    R.init(reinterpret_cast<const __half*>(row), n);
    for (int b = tid; b < R.lead; b += 256)
      atomicAdd(&big[isel_bin(isel_bits(R.rp[b]))], 1u);
    const uint4* rv = R.vec();
    for (int c0 = tid; c0 < R.nv; c0 += 256 * 2) {
      unsigned bits[16];
#pragma unroll
      for (int k = 0; k < 2; ++k)
        if (c0 + 256 * k < R.nv) isel_h16x8_bits(rv[c0 + 256 * k], bits + 8 * k);
#pragma unroll
      for (int k = 0; k < 2; ++k)
        if (c0 + 256 * k < R.nv)
#pragma unroll
          for (int j = 0; j < 8; ++j)
            atomicAdd(&big[isel_bin(bits[8 * k + j])], 1u);
    }
    for (int b = R.tail0() + tid; b < n; b += 256)
      atomicAdd(&big[isel_bin(isel_bits(R.rp[b]))], 1u);
  } else {
  // 8 loads in flight per thread before the LDS atomics (latency bound)
  for (int b0 = tid; b0 < n; b0 += 256 * 8) {
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = b0 + 256 * k < n ? isel_bits(row[b0 + 256 * k]) : 0u;
#pragma unroll
    for (int k = 0; k < 8; k++)
      if (b0 + 256 * k < n) atomicAdd(&big[isel_bin(v[k])], 1u);
  }
  }
  __syncthreads();
  isel_find_desc<ISEL_BINS / 256>(big, g_index_budget_d, sh, wsum);
  const int cnt_gt = sh[0];
  const unsigned T = (unsigned)sh[1];
  const int ncand = (int)big[T];
  __syncthreads();  // everyone has big[T] before the candidates overwrite big
  if (ncand > cap) {
    // ---- fallback: exact streaming radix over the remaining 18 bits
    // (bits 17..10, 9..2, 1..0), then one ascending emit scan (as
    // k_index_select_stream).
    unsigned prefix = T;
    int K = g_index_budget_d - cnt_gt, cgt = cnt_gt;
    const int shifts[3] = {10, 2, 0}, widths[3] = {8, 8, 2};
#pragma unroll 1
    for (int p = 0; p < 3; p++) {
      const int shift = shifts[p], wdt = widths[p];
      h256[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        const unsigned bits = isel_bits(row[b]);
        if ((p == 0 ? isel_bin(bits) : bits >> (shift + wdt)) == prefix)
          atomicAdd(&h256[(bits >> shift) & ((1u << wdt) - 1u)], 1u);
      }
      __syncthreads();
      isel_find_desc<1>(h256, K, sh, wsum);
      K -= sh[0];
      cgt += sh[0];
      prefix = (prefix << wdt) | (unsigned)sh[1];
      __syncthreads();
    }
    const unsigned sstar = prefix;
    const int need = g_index_budget_d - cgt;
    int outbase = 0, eqbase = 0;
    const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
    for (int c0 = 0; c0 < n; c0 += 256) {
      const int b = c0 + tid;
      unsigned bits = b < n ? isel_bits(row[b]) : 0u;
      bool gt = b < n && bits > sstar;
      bool eq = b < n && bits == sstar;
      unsigned long long bal_eq = __ballot_sync(~0ull, eq);
      int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
      if (lane == 0) shr_eq[w] = __popcll(bal_eq);
      __syncthreads();
      int eq_woff = 0;
      for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
      bool pick = gt || (eq && eqbase + eq_woff + eq_pre < need);
      unsigned long long bal_sel = __ballot_sync(~0ull, pick);
      if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
      __syncthreads();
      int woff = 0, wtot = 0, eqtot = 0;
      for (int i = 0; i < nw; i++) {
        int c = (int)shr_cnt[i];
        woff += i < w ? c : 0;
        wtot += c;
        eqtot += (int)shr_eq[i];
      }
      if (pick)
        selected[(size_t)q * g_index_budget_d + outbase + woff +
                 __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
      outbase += wtot;
      eqbase = min(eqbase + eqtot, need);
      __syncthreads();
    }
    return;
  }
  // ---- pass 2: definite ids (bin > T) + candidates (bin == T)
  unsigned* cbits = big;
  int* cid = reinterpret_cast<int*>(big + ISEL_CAP);
  if constexpr (isel_is_h16<ST>::value) {
    // f16 向量化读（atomic 位置发射，顺序无关；最终 bitonic sort 定序）
    IselH16Row R;
    R.init(reinterpret_cast<const __half*>(row), n);
    auto emit = [&](int b, unsigned v) {
      const unsigned bin = isel_bin(v);
      if (bin > T) {
        sel[atomicAdd(&s_nsel, 1)] = b;
      } else if (bin == T) {
        const int i = atomicAdd(&s_ncand, 1);
        cbits[i] = v;
        cid[i] = b;
      }
    };
    for (int b = tid; b < R.lead; b += 256) emit(b, isel_bits(R.rp[b]));
    const uint4* rv = R.vec();
    for (int c = tid; c < R.nv; c += 256) {
      unsigned bits[8];
      isel_h16x8_bits(rv[c], bits);
      const int b0 = R.lead + (c << 3);
#pragma unroll
      for (int j = 0; j < 8; ++j) emit(b0 + j, bits[j]);
    }
    for (int b = R.tail0() + tid; b < n; b += 256) emit(b, isel_bits(R.rp[b]));
  } else
  for (int b0 = tid; b0 < n; b0 += 256 * 8) {
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = b0 + 256 * k < n ? isel_bits(row[b0 + 256 * k]) : 0u;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int b = b0 + 256 * k;
      const unsigned bin = isel_bin(v[k]);
      if (b >= n) continue;
      if (bin > T) {
        sel[atomicAdd(&s_nsel, 1)] = b;
      } else if (bin == T) {
        const int i = atomicAdd(&s_ncand, 1);
        cbits[i] = v[k];
        cid[i] = b;
      }
    }
  }
  __syncthreads();
  // ---- exact s* among the candidates
  int gt2;
  const unsigned sstar = isel_kth(
      ncand, g_index_budget_d - cnt_gt, [&](int i, unsigned& k) { k = cbits[i]; return true; },
      h256, sh, wsum, gt2);
  const int need = g_index_budget_d - cnt_gt - gt2;  // >= 1 slots for ties at s*
  for (int i = tid; i < ncand; i += 256)
    if (cbits[i] == sstar) atomicAdd(&s_eq, 1);
  __syncthreads();
  const int E = s_eq;
  unsigned idk = 0u;  // eq ids with ~id >= idk are taken (all when E == need)
  if (E > need) {
    int g;
    idk = isel_kth(
        ncand, need,
        [&](int i, unsigned& k) { k = ~(unsigned)cid[i]; return cbits[i] == sstar; },
        h256, sh, wsum, g);
  }
  for (int i = tid; i < ncand; i += 256) {
    const unsigned bits = cbits[i];
    if (bits > sstar || (bits == sstar && ~(unsigned)cid[i] >= idk))
      sel[atomicAdd(&s_nsel, 1)] = cid[i];
  }
  __syncthreads();
  // ---- ascending ids: bitonic sort of the selected (distinct) ids
#pragma unroll 1
  for (int k = 2; k <= g_index_budget_d; k <<= 1) {
#pragma unroll 1
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int c = tid; c < g_index_budget_d / 2; c += 256) {
        const int i = (c / j) * 2 * j + (c % j), p = i + j;
        const int a = sel[i], b = sel[p];
        if ((a > b) == ((i & k) == 0)) {
          sel[i] = b;
          sel[p] = a;
        }
      }
      __syncthreads();
    }
  }
  for (int i = tid; i < g_index_budget_d; i += 256)
    selected[(size_t)q * g_index_budget_d + i] = sel[i];
}

// k_index_select_p1: P=1 专用的单内核精确 top-K（ceil_probe 版）——
// n<=budget 直接恒等；n<=8192 走 LDS 直方图三遍收束；更大走 ISEL 直方图 +
// 候选 + 位排序兜底。输出与 k_index_select_2p 逐位一致（探针 10K/43.5K
// ABBA diff=0）。与 2p 的区别：单 launch、一次全局读，省一遍往返。
__global__ __launch_bounds__(256) void k_index_select_p1(
    const float* __restrict__ row, int* __restrict__ selected,
    const int* __restrict__ pos) {
  __shared__ unsigned big[ISEL_BINS];
  __shared__ int sel[INDEX_BUDGET];
  __shared__ unsigned h256[256];
  __shared__ unsigned wsum[8];
  __shared__ unsigned shr_cnt[8], shr_eq[8];
  __shared__ int sh[2];
  __shared__ int s_nsel, s_ncand, s_eq, s_passcnt, s_passbyte, s_cntgt;

  const int t = *pos, n = (t + 1) / 4;
  const int tid = threadIdx.x;
  if (n <= g_index_budget_d) {
    for (int i = tid; i < g_index_budget_d; i += 256)
      selected[i] = i < n ? i : -1;
    return;
  }
  if (n <= 8192) {
    float* sbuf = reinterpret_cast<float*>(big);
    h256[tid] = 0;
    __syncthreads();
    for (int b = tid; b < n; b += 256) {
      float s = row[b];
      sbuf[b] = s;
      atomicAdd(&h256[__float_as_uint(s) >> 24], 1u);
    }
    __syncthreads();
    unsigned prefix = 0;
    int K = g_index_budget_d;
    if (tid == 0) s_cntgt = 0;
    for (int p = 0; p < 4; p++) {
      const int shift = (3 - p) * 8;
      if (p) {
        __syncthreads();
        h256[tid] = 0;
        __syncthreads();
        for (int b = tid; b < n; b += 256) {
          unsigned bits = __float_as_uint(sbuf[b]);
          if ((bits >> (shift + 8)) == prefix)
            atomicAdd(&h256[(bits >> shift) & 0xffu], 1u);
        }
        __syncthreads();
      }
      if (tid == 0) {
        int acc = 0, chosen = 255;
        for (int i = 255; i >= 0; i--) {
          int c = (int)h256[i];
          if (acc + c >= K) { chosen = i; break; }
          acc += c;
        }
        s_passcnt = acc;
        s_passbyte = chosen;
      }
      __syncthreads();
      K -= s_passcnt;
      if (tid == 0) s_cntgt += s_passcnt;
      prefix = (prefix << 8) | (unsigned)s_passbyte;
      __syncthreads();
    }
    const unsigned sstar = prefix;
    const int need = g_index_budget_d - s_cntgt;
    int outbase = 0, eqbase = 0;
    const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
    for (int c0 = 0; c0 < n; c0 += 256) {
      const int b = c0 + tid;
      unsigned bits = b < n ? __float_as_uint(sbuf[b]) : 0u;
      bool gt = b < n && bits > sstar;
      bool eq = b < n && bits == sstar;
      unsigned long long bal_eq = __ballot_sync(~0ull, eq);
      int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
      if (lane == 0) shr_eq[w] = __popcll(bal_eq);
      __syncthreads();
      int eq_woff = 0;
      for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
      bool pick = gt || (eq && eqbase + eq_woff + eq_pre < need);
      unsigned long long bal_sel = __ballot_sync(~0ull, pick);
      if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
      __syncthreads();
      int woff = 0, wtot = 0, eqtot = 0;
      for (int i = 0; i < nw; i++) {
        int c = (int)shr_cnt[i];
        woff += i < w ? c : 0;
        wtot += c;
        eqtot += (int)shr_eq[i];
      }
      if (pick)
        selected[outbase + woff + __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
      outbase += wtot;
      eqbase = min(eqbase + eqtot, need);
      __syncthreads();
    }
    return;
  }

  const int cap = ISEL_CAP;
  for (int i = tid; i < ISEL_BINS; i += 256) big[i] = 0;
  if (tid == 0) s_nsel = s_ncand = s_eq = 0;
  __syncthreads();
  for (int b0 = tid; b0 < n; b0 += 256 * 8) {
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = b0 + 256 * k < n ? isel_bits(row[b0 + 256 * k]) : 0u;
#pragma unroll
    for (int k = 0; k < 8; k++)
      if (b0 + 256 * k < n) atomicAdd(&big[isel_bin(v[k])], 1u);
  }
  __syncthreads();
  isel_find_desc<ISEL_BINS / 256>(big, g_index_budget_d, sh, wsum);
  const int cnt_gt = sh[0];
  const unsigned T = (unsigned)sh[1];
  const int ncand = (int)big[T];
  __syncthreads();
  if (ncand > cap) {
    unsigned prefix = T;
    int K = g_index_budget_d - cnt_gt, cgt = cnt_gt;
    const int shifts[3] = {10, 2, 0}, widths[3] = {8, 8, 2};
#pragma unroll 1
    for (int p = 0; p < 3; p++) {
      const int shift = shifts[p], wdt = widths[p];
      h256[tid] = 0;
      __syncthreads();
      for (int b = tid; b < n; b += 256) {
        const unsigned bits = isel_bits(row[b]);
        if ((p == 0 ? isel_bin(bits) : bits >> (shift + wdt)) == prefix)
          atomicAdd(&h256[(bits >> shift) & ((1u << wdt) - 1u)], 1u);
      }
      __syncthreads();
      isel_find_desc<1>(h256, K, sh, wsum);
      K -= sh[0];
      cgt += sh[0];
      prefix = (prefix << wdt) | (unsigned)sh[1];
      __syncthreads();
    }
    const unsigned sstar = prefix;
    const int need = g_index_budget_d - cgt;
    int outbase = 0, eqbase = 0;
    const int lane = tid % warpSize, w = tid / warpSize, nw = 256 / warpSize;
#pragma unroll 1
    for (int c0 = 0; c0 < n; c0 += 256) {
      const int b = c0 + tid;
      unsigned bits = b < n ? isel_bits(row[b]) : 0u;
      bool gt = b < n && bits > sstar;
      bool eq = b < n && bits == sstar;
      unsigned long long bal_eq = __ballot_sync(~0ull, eq);
      int eq_pre = __popcll(bal_eq & (lane ? (~0ull) >> (64 - lane) : 0ull));
      if (lane == 0) shr_eq[w] = __popcll(bal_eq);
      __syncthreads();
      int eq_woff = 0;
      for (int i = 0; i < nw; i++) eq_woff += i < w ? (int)shr_eq[i] : 0;
      bool pick = gt || (eq && eqbase + eq_woff + eq_pre < need);
      unsigned long long bal_sel = __ballot_sync(~0ull, pick);
      if (lane == 0) shr_cnt[w] = __popcll(bal_sel);
      __syncthreads();
      int woff = 0, wtot = 0, eqtot = 0;
      for (int i = 0; i < nw; i++) {
        int c = (int)shr_cnt[i];
        woff += i < w ? c : 0;
        wtot += c;
        eqtot += (int)shr_eq[i];
      }
      if (pick)
        selected[outbase + woff + __popcll(bal_sel & (lane ? (~0ull) >> (64 - lane) : 0ull))] = b;
      outbase += wtot;
      eqbase = min(eqbase + eqtot, need);
      __syncthreads();
    }
    return;
  }
  unsigned* cbits = big;
  int* cid = reinterpret_cast<int*>(big + ISEL_CAP);
  for (int b0 = tid; b0 < n; b0 += 256 * 8) {
    unsigned v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = b0 + 256 * k < n ? isel_bits(row[b0 + 256 * k]) : 0u;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const int b = b0 + 256 * k;
      const unsigned bin = isel_bin(v[k]);
      if (b >= n) continue;
      if (bin > T) sel[atomicAdd(&s_nsel, 1)] = b;
      else if (bin == T) {
        const int i = atomicAdd(&s_ncand, 1);
        cbits[i] = v[k];
        cid[i] = b;
      }
    }
  }
  __syncthreads();
  int gt2;
  const unsigned sstar = isel_kth(
      ncand, g_index_budget_d - cnt_gt, [&](int i, unsigned& k) { k = cbits[i]; return true; },
      h256, sh, wsum, gt2);
  const int need = g_index_budget_d - cnt_gt - gt2;
  for (int i = tid; i < ncand; i += 256)
    if (cbits[i] == sstar) atomicAdd(&s_eq, 1);
  __syncthreads();
  const int E = s_eq;
  unsigned idk = 0u;
  if (E > need) {
    int g;
    idk = isel_kth(
        ncand, need,
        [&](int i, unsigned& k) { k = ~(unsigned)cid[i]; return cbits[i] == sstar; },
        h256, sh, wsum, g);
  }
  for (int i = tid; i < ncand; i += 256) {
    const unsigned bits = cbits[i];
    if (bits > sstar || (bits == sstar && ~(unsigned)cid[i] >= idk))
      sel[atomicAdd(&s_nsel, 1)] = cid[i];
  }
  __syncthreads();
#pragma unroll 1
  for (int k = 2; k <= g_index_budget_d; k <<= 1) {
#pragma unroll 1
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int c = tid; c < g_index_budget_d / 2; c += 256) {
        const int i = (c / j) * 2 * j + (c % j), p = i + j;
        const int a = sel[i], b = sel[p];
        if ((a > b) == ((i & k) == 0)) { sel[i] = b; sel[p] = a; }
      }
      __syncthreads();
    }
  }
  for (int i = tid; i < g_index_budget_d; i += 256) selected[i] = sel[i];
}
