// 26_kernels_dense_gguf.inc — GGUF dense weights in their native block format
// (dtype 100 + GGML type: Q4_K 112, Q5_K 113, Q6_K 114, IQ4_NL 120, IQ4_XS 123;
// src/gguf_map.h add_q8 with g_native_dense). Decode is bandwidth-bound, and
// the q8g32 requant used before reads 1.3-2.0x the native bytes of these types.
//
// Data: rows x row_bytes raw GGUF rows (row map already applied on the host).
// Every 32-element "unit" (one K-quant sub-block / one IQ4_NL block) is decoded
// with exactly the float ops of gguf::dequant_row (explicit _rn intrinsics, no
// contraction), so the device weights equal the CPU reference bit for bit.
//
// k_ggd_gemv<T, NP, XB>: one wave per output row, 4 rows per 128-thread
// block. The wave walks the row in segments of up to 96 units (a 2560-col row
// is one segment): all dword loads of a segment are issued at once, stored to
// the wave's LDS slice (Q6_K re-laid on 16-aligned 224 B slots), then lane i
// decodes units i, i+32, i+64 with vector LDS reads and adds w . x_p for NP
// tokens (fixed 4-chain fmaf order), and a fixed xor-shuffle tree reduces the
// 32 lanes. A token's result depends on the shape only -- not on NP or on the
// other tokens of the launch -- so any P <= 48 split into <= 8-token passes is
// bit-identical to P = 1 (the row-invariance contract of the dense P <= 48
// paths, see q8g32_wmma). Measured (8060S, DRAM-resident): output.weight
// Q6_K 219 GB/s, 12288x2560 Q4_K/Q5_K/Q6_K/IQ4_XS 188-215 GB/s.
// k_ggd_deq_bf16<T>: prefill GEMM operand (byte decode, f2bf RNE).

namespace ggd {

constexpr int kQ4K = 12, kQ5K = 13, kQ6K = 14, kIQ4NL = 20, kIQ4XS = 23;
constexpr int kDtypeBase = 100;

__host__ __device__ constexpr bool type_ok(int t) {
  return t == kQ4K || t == kQ5K || t == kQ6K || t == kIQ4NL || t == kIQ4XS;
}
// bytes / units of one GGUF block (K-quants and IQ4_XS: 256-element super-block)
template <int T>
__host__ __device__ constexpr int blk_bytes() {
  return T == kQ4K ? 144 : T == kQ5K ? 176 : T == kQ6K ? 210 : T == kIQ4XS ? 136 : 18;
}
template <int T>
__host__ __device__ constexpr int blk_units() {
  return T == kIQ4NL ? 1 : 8;
}
__host__ __device__ inline uint64_t row_bytes(int t, uint64_t cols) {
  return t == kIQ4NL ? cols / 32 * 18
       : (cols / 256) * (t == kQ4K ? 144 : t == kQ5K ? 176 : t == kQ6K ? 210 : 136);
}

__device__ __forceinline__ uint32_t ld16(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8);
}
__device__ __forceinline__ float h2f(uint32_t h) {
  return __half2float(__ushort_as_half((unsigned short)h));
}

__constant__ int8_t c_iq4nl_kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                      1,    13,   25,  38,  53,  69,  89,  113};

// unit ib (0..blk_units-1) of the block at b -> w[32] (gguf::dequant_row order)
template <int T>
__device__ __forceinline__ void unit_deq(const uint8_t* __restrict__ b, int ib, float* w) {
  if constexpr (T == kIQ4NL || T == kIQ4XS) {
    const float d0 = h2f(ld16(b));
    float dl;
    const uint8_t* qs;
    if constexpr (T == kIQ4NL) {
      dl = d0;
      qs = b + 2;
    } else {
      const uint32_t sh = ld16(b + 2);
      const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
      dl = __fmul_rn(d0, (float)(ls - 32));
      qs = b + 8 + 16 * ib;
    }
#pragma unroll
    for (int j = 0; j < 16; j++) {
      const uint32_t q = qs[j];
      w[j] = __fmul_rn(dl, (float)c_iq4nl_kv[q & 0xF]);
      w[j + 16] = __fmul_rn(dl, (float)c_iq4nl_kv[q >> 4]);
    }
  } else if constexpr (T == kQ4K || T == kQ5K) {
    const float d = h2f(ld16(b)), dmin = h2f(ld16(b + 2));
    const uint8_t* sc = b + 4;
    uint32_t s, m;
    if (ib < 4) {
      s = sc[ib] & 63;
      m = sc[ib + 4] & 63;
    } else {
      s = (sc[ib + 4] & 0xF) | ((sc[ib - 4] >> 6) << 4);
      m = (sc[ib + 4] >> 4) | ((sc[ib] >> 6) << 4);
    }
    const float d1 = __fmul_rn(d, (float)s), dm1 = __fmul_rn(dmin, (float)m);
    const int g = ib >> 1, hi = ib & 1;
    const uint8_t* ql = b + (T == kQ5K ? 48 : 16) + 32 * g;
#pragma unroll
    for (int l = 0; l < 32; l++) {
      int q = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
      if constexpr (T == kQ5K) q += ((b[16 + l] >> ib) & 1) ? 16 : 0;
      w[l] = __fsub_rn(__fmul_rn(d1, (float)q), dm1);
    }
  } else {  // Q6_K: ql[128], qh[64], scales[16] (int8), d at 208
    const float d = h2f(ld16(b + 208));
    const int h = ib >> 2, qq = ib & 3;
    const uint8_t* ql = b + 64 * h + 32 * (qq & 1);
    const uint8_t* qh = b + 128 + 32 * h;
    const int8_t* sc = (const int8_t*)(b + 192 + 8 * h) + 2 * qq;
    const float ds0 = __fmul_rn(d, (float)sc[0]), ds1 = __fmul_rn(d, (float)sc[1]);
#pragma unroll
    for (int l = 0; l < 32; l++) {
      const uint32_t nib = (qq >> 1) ? (ql[l] >> 4) : (ql[l] & 0xF);
      const int q = (int)(nib | (((qh[l] >> (2 * qq)) & 3) << 4)) - 32;
      w[l] = __fmul_rn(l < 16 ? ds0 : ds1, (float)q);
    }
  }
}

// w[0..31] . x[off..off+31]: four independent fmaf chains (element j -> chain
// j % 4), combined (a0 + a1) + (a2 + a3) and added to acc. Fixed order.
template <bool XB>
__device__ __forceinline__ float dot32(const float* w, const void* __restrict__ xv, uint64_t off,
                                       float acc) {
  float a[4] = {0.f, 0.f, 0.f, 0.f};
  if constexpr (XB) {
    const uint4* xp = (const uint4*)((const uint16_t*)xv + off);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const uint4 v = xp[i];
      const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        a[(2 * k) & 3] = fmaf(w[8 * i + 2 * k], __uint_as_float(u[k] << 16), a[(2 * k) & 3]);
        a[(2 * k + 1) & 3] =
            fmaf(w[8 * i + 2 * k + 1], __uint_as_float(u[k] & 0xFFFF0000u), a[(2 * k + 1) & 3]);
      }
    }
  } else {
    const float4* xp = (const float4*)((const float*)xv + off);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 v = xp[i];
      a[0] = fmaf(w[4 * i + 0], v.x, a[0]);
      a[1] = fmaf(w[4 * i + 1], v.y, a[1]);
      a[2] = fmaf(w[4 * i + 2], v.z, a[2]);
      a[3] = fmaf(w[4 * i + 3], v.w, a[3]);
    }
  }
  return acc + ((a[0] + a[1]) + (a[2] + a[3]));
}

// LDS slot of one GGUF block in the GEMV stage: Q6_K blocks (210 B) are
// re-laid on 224 B (16-aligned) so the decoder can use vector LDS reads; the
// other types already sit on 8/16-byte multiples (IQ4_NL keeps byte reads).
template <int T>
__host__ __device__ constexpr int lds_slot() {
  return T == kQ6K ? 224 : blk_bytes<T>();
}

// unit ib of the block at LDS slot b -> w[32]; same values as unit_deq
// (identical float ops), vector LDS reads + 4-byte-parallel bit unpacking.
template <int T>
__device__ __forceinline__ void unit_deq_lds(const uint8_t* b, int ib, float* w) {
  if constexpr (T == kQ6K) {
    const int h = ib >> 2, qq = ib & 3;
    const uint4* qlp = (const uint4*)(b + 64 * h + 32 * (qq & 1));
    const uint4* qhp = (const uint4*)(b + 128 + 32 * h);
    const uint4 l0 = qlp[0], l1 = qlp[1], h0 = qhp[0], h1 = qhp[1];
    const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    const uint32_t hw[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
    const float d = h2f(*(const uint16_t*)(b + 208));
    const int8_t* sc = (const int8_t*)(b + 192 + 8 * h + 2 * qq);
    const float ds0 = __fmul_rn(d, (float)sc[0]), ds1 = __fmul_rn(d, (float)sc[1]);
    const int ns = 4 * (qq >> 1), hs = 2 * qq;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const uint32_t q4 = ((lw[k] >> ns) & 0x0F0F0F0Fu) | (((hw[k] >> hs) & 0x03030303u) << 4);
#pragma unroll
      for (int j = 0; j < 4; j++)
        w[4 * k + j] = __fmul_rn(k < 4 ? ds0 : ds1, (float)((int)((q4 >> (8 * j)) & 0xFF) - 32));
    }
  } else if constexpr (T == kQ4K || T == kQ5K) {
    const uint32_t hd = *(const uint32_t*)b;
    const float d = h2f(hd & 0xFFFF), dmin = h2f(hd >> 16);
    const uint8_t* sc = b + 4;
    uint32_t s, m;
    if (ib < 4) {
      s = sc[ib] & 63;
      m = sc[ib + 4] & 63;
    } else {
      s = (sc[ib + 4] & 0xF) | ((sc[ib - 4] >> 6) << 4);
      m = (sc[ib + 4] >> 4) | ((sc[ib] >> 6) << 4);
    }
    const float d1 = __fmul_rn(d, (float)s), dm1 = __fmul_rn(dmin, (float)m);
    const int g = ib >> 1, hi = ib & 1;
    const uint4* qlp = (const uint4*)(b + (T == kQ5K ? 48 : 16) + 32 * g);
    const uint4 l0 = qlp[0], l1 = qlp[1];
    const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    uint32_t hw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if constexpr (T == kQ5K) {
      const uint4* qhp = (const uint4*)(b + 16);
      const uint4 h0 = qhp[0], h1 = qhp[1];
      const uint32_t t8[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
#pragma unroll
      for (int k = 0; k < 8; k++) hw[k] = ((t8[k] >> ib) & 0x01010101u) << 4;
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      const uint32_t q4 = ((lw[k] >> (4 * hi)) & 0x0F0F0F0Fu) | hw[k];
#pragma unroll
      for (int j = 0; j < 4; j++)
        w[4 * k + j] = __fsub_rn(__fmul_rn(d1, (float)((q4 >> (8 * j)) & 0xFF)), dm1);
    }
  } else if constexpr (T == kIQ4XS) {
    const uint2 hd = *(const uint2*)b;  // d | scales_h << 16, scales_l[4]
    const float d0 = h2f(hd.x & 0xFFFF);
    const uint32_t sh = hd.x >> 16;
    const int ls = ((hd.y >> (8 * (ib / 2) + 4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
    const float dl = __fmul_rn(d0, (float)(ls - 32));
    const uint2* qp = (const uint2*)(b + 8 + 16 * ib);
    const uint2 q0 = qp[0], q1 = qp[1];
    const uint32_t qw[4] = {q0.x, q0.y, q1.x, q1.y};
#pragma unroll
    for (int k = 0; k < 4; k++)
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t q = (qw[k] >> (8 * j)) & 0xFF;
        w[4 * k + j] = __fmul_rn(dl, (float)c_iq4nl_kv[q & 0xF]);
        w[4 * k + j + 16] = __fmul_rn(dl, (float)c_iq4nl_kv[q >> 4]);
      }
  } else {
    unit_deq<T>(b, ib, w);
  }
}

}  // namespace ggd

// y[p * ys + r] = W[r] . x[p * xs + :] for r < R, p < NP. 128 threads = 4 rows
// (one wave each). A row is walked in segments of up to kSegU units: the whole
// segment is loaded with dword loads into registers (all in flight at once),
// stored to the wave's LDS slice (Q6_K re-laid on 224 B slots), then lane i
// takes units i, i+32, i+64. Rows must be 4-byte aligned with row_bytes % 4 == 0
// (host checks), so every segment is a whole number of dwords.
namespace ggd {
constexpr int kSegU = 96;
}
template <int T, int NP, bool XB>
__launch_bounds__(128) __global__
void k_ggd_gemv(const uint8_t* __restrict__ w, uint64_t rb, const void* __restrict__ x,
                uint64_t xs, float* __restrict__ y, uint64_t ys, int R, int C) {
  using namespace ggd;
  constexpr int BB = blk_bytes<T>(), BU = blk_units<T>(), SL = lds_slot<T>();
  constexpr int SEGB = kSegU / BU * BB;               // source bytes of a full segment
  constexpr int SEGL = (kSegU / BU * SL + 15) & ~15;  // LDS bytes per wave
  constexpr int NV = (SEGB / 4 + 31) / 32;            // dwords per lane
  constexpr int kUnitUnroll = (T == kQ6K && NP >= 6) ? 1 : kSegU / 32;
  __shared__ __attribute__((aligned(16))) uint8_t lds[4][SEGL];
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = (int)blockIdx.x * 4 + wave;
  const bool live = r < R;
  const uint8_t* row = w + (size_t)(live ? r : 0) * rb;
  const int nu = C / 32, nseg = (nu + kSegU - 1) / kSegU;
  uint8_t* L = lds[wave];
  float acc[NP];
#pragma unroll
  for (int p = 0; p < NP; p++) acc[p] = 0.f;
  for (int sg = 0; sg < nseg; sg++) {
    const int u0 = sg * kSegU;
    const int nun = min(kSegU, nu - u0);  // a multiple of BU (cols % 256 == 0 for K-quants)
    const int nw = nun / BU * BB / 4;
    const uint32_t* src = (const uint32_t*)(row + (size_t)(u0 / BU) * BB);
    uint32_t v[NV];
#pragma unroll
    for (int k = 0; k < NV; k++) {
      const int i = lane + 32 * k;
      v[k] = i < nw ? src[i] : 0u;
    }
#pragma unroll
    for (int k = 0; k < NV; k++) {
      const int i = lane + 32 * k;
      if (i < nw) {
        if constexpr (SL == BB) {
          ((uint32_t*)L)[i] = v[k];
        } else {  // two halves; block boundaries are even, so a half never straddles
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int o = 4 * i + 2 * hh, bk = o / BB;
            *(uint16_t*)(L + bk * SL + (o - bk * BB)) = (uint16_t)(v[k] >> (16 * hh));
          }
        }
      }
    }
    __syncthreads();
    // Q6_K with NP >= 6: the unrolled unit loop keeps three decoded units live and
    // spills (256 VGPRs, 51-70 spilled, occupancy 10 -> 5; NP = 8 took 3.4x NP = 5).
    // Rolled, the same per-token arithmetic in the same order runs ~2x faster.
#pragma unroll kUnitUnroll
    for (int k = 0; k < kSegU / 32; k++) {
      const int u = lane + 32 * k;
      if (u < nun) {
        float wv[32];
        unit_deq_lds<T>(L + (u / BU) * SL, u % BU, wv);
        const uint64_t e0 = (uint64_t)(u0 + u) * 32;
#pragma unroll
        for (int p = 0; p < NP; p++) acc[p] = dot32<XB>(wv, x, (uint64_t)p * xs + e0, acc[p]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int p = 0; p < NP; p++) {
    float v = acc[p];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    acc[p] = v;
  }
  if (live && lane == 0) {
#pragma unroll
    for (int p = 0; p < NP; p++) y[(uint64_t)p * ys + r] = acc[p];
  }
}

// D4: two native-GGUF GEMVs with the same dtype/cols in one launch. Rows
// [0, RA) read wA and write yA; rows [RA, RA+RB) read wB and write yB. Each
// row's K loop / reduction is exactly k_ggd_gemv's, so the values are
// bit-identical to two separate launches. P = 1 only (decode).
template <int T, bool XB>
__launch_bounds__(128) __global__
void k_ggd_gemv2(const uint8_t* __restrict__ wA, const uint8_t* __restrict__ wB,
                 int RA, int RB, uint64_t rb, const void* __restrict__ x,
                 float* __restrict__ yA, float* __restrict__ yB, int C) {
  using namespace ggd;
  constexpr int BB = blk_bytes<T>(), BU = blk_units<T>(), SL = lds_slot<T>();
  constexpr int SEGB = kSegU / BU * BB;               // source bytes of a full segment
  constexpr int SEGL = (kSegU / BU * SL + 15) & ~15;  // LDS bytes per wave
  constexpr int NV = (SEGB / 4 + 31) / 32;            // dwords per lane
  __shared__ __attribute__((aligned(16))) uint8_t lds[4][SEGL];
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = (int)blockIdx.x * 4 + wave;
  const int R = RA + RB;
  const bool live = r < R;
  const bool second = r >= RA;
  const int rr = live ? (second ? r - RA : r) : 0;
  const uint8_t* row = (second ? wB : wA) + (size_t)rr * rb;
  const int nu = C / 32, nseg = (nu + kSegU - 1) / kSegU;
  uint8_t* L = lds[wave];
  float acc = 0.f;
  for (int sg = 0; sg < nseg; sg++) {
    const int u0 = sg * kSegU;
    const int nun = min(kSegU, nu - u0);
    const int nw = nun / BU * BB / 4;
    const uint32_t* src = (const uint32_t*)(row + (size_t)(u0 / BU) * BB);
    uint32_t v[NV];
#pragma unroll
    for (int k = 0; k < NV; k++) {
      const int i = lane + 32 * k;
      v[k] = i < nw ? src[i] : 0u;
    }
#pragma unroll
    for (int k = 0; k < NV; k++) {
      const int i = lane + 32 * k;
      if (i < nw) {
        if constexpr (SL == BB) {
          ((uint32_t*)L)[i] = v[k];
        } else {
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            const int o = 4 * i + 2 * hh, bk = o / BB;
            *(uint16_t*)(L + bk * SL + (o - bk * BB)) = (uint16_t)(v[k] >> (16 * hh));
          }
        }
      }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kSegU / 32; k++) {
      const int u = lane + 32 * k;
      if (u < nun) {
        float wv[32];
        unit_deq_lds<T>(L + (u / BU) * SL, u % BU, wv);
        const uint64_t e0 = (uint64_t)(u0 + u) * 32;
        acc = dot32<XB>(wv, x, e0, acc);
      }
    }
    __syncthreads();
  }
  float v = acc;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  if (live && lane == 0) {
    if (second) {
      if (rr < RB) yB[rr] = v;
    } else {
      if (rr < RA) yA[rr] = v;
    }
  }
}

// W (R x C, native rows) -> bf16 [R][C], one thread per 32-element unit.
template <int T>
__global__ void k_ggd_deq_bf16(const uint8_t* __restrict__ w, uint64_t rb,
                               uint16_t* __restrict__ out, uint64_t R, int C) {
  using namespace ggd;
  constexpr int BB = blk_bytes<T>(), BU = blk_units<T>();
  const uint64_t nu = (uint64_t)C / 32;
  const uint64_t gid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (gid >= R * nu) return;
  const uint64_t r = gid / nu, u = gid % nu;
  float wv[32];
  unit_deq<T>(w + r * rb + (u / BU) * BB, (int)(u % BU), wv);
  uint16_t o[32];
#pragma unroll
  for (int j = 0; j < 32; j++) o[j] = f2bf(wv[j]);
  uint4* dst = (uint4*)(out + r * (uint64_t)C + u * 32);
#pragma unroll
  for (int i = 0; i < 4; i++) dst[i] = ((const uint4*)o)[i];
}
