// --- LUT-decoded 4-bit routed-expert F16 WMMA GEMM -------------------------
//
// Same pipeline / tiling / epilogues as k_moe_gg (26_kernels_moe_gguf.inc),
// for the 4-bit formats whose value is  lut[nibble] * scale  with one scale
// per 32-element block:
//   hgn q4cp (dtype 5): per-tensor 16 x fp32 codebook, fp16 scale per 32
//     (codes and scales in separate planes, byte b = elements 2b | 2b+1 << 4);
//   GGUF IQ4_NL (18 B / 32: fp16 d, qs[16]; element j = qs[j] & 15,
//     j + 16 = qs[j] >> 4) and IQ4_XS (136 B / 256: d, scales_h, scales_l[4],
//     qs[128]; 6-bit sub-block scale d * (ls - 32)), lut = kvalues_iq4nl.
//
// Each stage stages 16 code bytes per (row, K block) in "pair order" (byte b =
// elements 2b, 2b+1; IQ blocks are reordered with two v_perm on commit) plus a
// half2 (scale, scale). Decode = 16 lookups in an LDS pair table
// s_cbp[byte] = half2(lut[byte & 15], lut[byte >> 4]) and one packed multiply.
// Numerics: w = f16(f16(lut) * f16(scale)), F32 accumulate — like the GGUF path
// (IQ: lut / 256 and scale * 256, exact rescale keeping scales out of fp16 subnormals).
//
//   kPair (gate/up): hid F16 [slot][m] = up * silu(gate), grid = (m / 64, tiles)
//     kRaw: F16 [slot][2m] = gate | up unfused (hgn v2: rotation comes first)
//   !kPair (down):   pairs F32 [slot][m] (F16 with kF16Pairs), grid = (m / 128, tiles)
//
// hgn v2 i4r (dtype 23, kI4R): q4cp code plane, linear book -8..7 (p.cb), one
// fp16 scale per 128 elements, scale rows unpadded (sstride = k / 128 * 2).
// i4r decodes arithmetically instead of via s_cbp (f16 bits 0x6400+nib =
// 1024+nib exact; minus 1032 = nib-8 exact by Sterbenz; the hmul2 by the scale
// rounds identically, so output is bit-identical to the table path), and keeps
// one s_scale entry per row per stage: a BK=2 stage never crosses a
// 128-element scale block, so the scale is fetched once per 2 stages.

namespace moelut {

constexpr int kQ4CP = 100, kIQ4NL = 20, kIQ4XS = 23;  // IQ: gguf::Type values
constexpr int kI4R = 101;  // hgn v2 routed experts
// GGUF IQ3_S (110 B / 256: d, qs[64], qh[8], signs[32], scales[4]). Every
// value is  d * (1 + 2*sc) * (+-g),  g in {1,3,...,15} from iq3s_grid (all 512
// x 4 entries are odd and <= 15), so one 32-element sub-block is exactly a
// 4-bit code (bits 0-2 = g >> 1, bit 3 = sign) with the 16-entry book
// +-(2c+1) and one scale: the commit expands each sub-block's 8 grid lookups
// (LDS copy of the grid) + signs into the same 16 pair-order code bytes the
// IQ4 path stages, and decode / WMMA / epilogues are unchanged.
constexpr int kIQ3S = 21;
// GGUF IQ3_XXS (98 B / 256: d, qs[64], scales_and_signs[32]) and IQ2_S
// (82 B / 256: d, qs[64], qh[8], scales[8]). Both have a 32-element sub-block
// whose magnitudes fit a 4-bit code with sign: IQ3_XXS g in {4,12,...,62}
// (code = (g-4)/8), IQ2_S g in {8,25,43} (code = (g-8)/16, 0..2).
constexpr int kIQ3XXS = 18, kIQ2S = 22;
// IQ2_XXS (66 B / 256) and IQ2_XS (74 B / 256) use the same magnitudes as
// IQ2_S, with packed signs and one scale per 32 / 16 elements respectively.
constexpr int kIQ2XXS = 16, kIQ2XS = 17;
// GGUF Q2_0 (18 B / 64: fp16 d, qs[16] = 4 x 2-bit). One 64-element block is
// exactly one BK=2 stage; codes 0..3 decode to (-1,0,1,2) * d.
constexpr int kQ2_0 = 42;

}  // namespace moelut

// hv2 basics live here (not in 28_kernels_hgn_v2.inc) because the kMid i4r
// epilogue below needs them and this file is also included standalone by
// tools/moe_lut_test.cu.
namespace hv2 {

constexpr float kRs128 = 0.0883883476483184405f;  // 1 / sqrt(128)

// Unnormalized in-wave FWHT of 128 values (lo' = lo + hi, hi' = lo - hi).
__device__ __forceinline__ void fwht128(float& v0, float& v1, float& v2, float& v3,
                                        int lane) {
#pragma unroll
  for (int s = 1; s < 32; s <<= 1) {
    const float p0 = __shfl_xor(v0, s, 32), p1 = __shfl_xor(v1, s, 32);
    const float p2 = __shfl_xor(v2, s, 32), p3 = __shfl_xor(v3, s, 32);
    const bool hi = (lane & s) != 0;
    v0 = hi ? p0 - v0 : v0 + p0;
    v1 = hi ? p1 - v1 : v1 + p1;
    v2 = hi ? p2 - v2 : v2 + p2;
    v3 = hi ? p3 - v3 : v3 + p3;
  }
  const float a = v0 + v1, b = v0 - v1, c = v2 + v3, d = v2 - v3;  // stride 32
  v0 = a + c;  // stride 64
  v2 = a - c;
  v1 = b + d;
  v3 = b - d;
}

}  // namespace hv2

namespace moelut {

__constant__ int8_t c_iq4_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                        1, 13, 25, 38, 53, 69, 89, 113};

// Expert-stacked weights. Row index of (expert e, row r) inside `w` / `w_up`:
//   e * e_rows + r            (gate, down, IQ up)
//   e * e_rows + up_off + r   (kPair upper half: q4cp fused gate_up -> up_off = mid)
struct LutW {
  const uint8_t* w;        // q4cp: code plane (tensor data + 64); IQ: raw blocks
  const uint8_t* w_up;     // kPair up rows (q4cp fused gate_up: == w)
  const uint8_t* sc;       // q4cp scale plane (fp16 per 32, sstride bytes / row)
  const uint8_t* sc_up;    // q4cp: == sc for fused gate_up
  const float* cb;         // q4cp codebook (16 x fp32, tensor data)
  int sstride;             // q4cp scale row stride in bytes (16 B multiple, rows padded)
  int e_rows, up_off;
};

template <int T>
__host__ __device__ constexpr size_t lut_row_bytes(size_t k) {
  return (T == kQ4CP || T == kI4R) ? k / 2
         : T == kIQ4NL             ? (k / 32) * 18
         : T == kIQ3S              ? (k / 256) * 110
         : T == kIQ2XXS            ? (k / 256) * 66
         : T == kIQ2XS             ? (k / 256) * 74
         : T == kIQ3XXS            ? (k / 256) * 98
         : T == kIQ2S              ? (k / 256) * 82
         : T == kQ2_0              ? (k / 64) * 18
                                   : (k / 256) * 136;
}

// Bits 0-3 of `bits` -> bit 3 of bytes 0-3 (IQ3_S sign into the 4-bit code).
__device__ __forceinline__ uint32_t iq3s_sign4(uint32_t bits) {
  return (__umul24(bits & 0xFU, 0x00204081U) & 0x01010101U) << 3;
}
// One IQ3_S grid word (4 odd magnitudes) + 4 sign bits -> 4 code bytes, then
// two such (8 consecutive elements) -> one pair-order word.
__device__ __forceinline__ uint32_t iq3s_word(uint32_t g1, uint32_t g2, uint32_t s8) {
  const uint32_t c1 = ((g1 >> 1) & 0x07070707U) | iq3s_sign4(s8);
  const uint32_t c2 = ((g2 >> 1) & 0x07070707U) | iq3s_sign4(s8 >> 4);
  return __builtin_amdgcn_perm(c2 | (c2 >> 4), c1 | (c1 >> 4), 0x06040200U);
}
// IQ3_XXS: magnitudes {4,12,20,28,36,44,52,62} -> code (g-4)/8 (0..7), then
// the same 8-element sign split / pair packing as IQ3_S.
__device__ __forceinline__ uint32_t iq3xxs_word(uint32_t g1, uint32_t g2, uint32_t s8) {
  const uint32_t c1 = (((g1 - 0x04040404U) >> 3) & 0x07070707U) | iq3s_sign4(s8);
  const uint32_t c2 = (((g2 - 0x04040404U) >> 3) & 0x07070707U) | iq3s_sign4(s8 >> 4);
  return __builtin_amdgcn_perm(c2 | (c2 >> 4), c1 | (c1 >> 4), 0x06040200U);
}
// IQ2_S: magnitudes {8,25,43} -> code (g-8)/16 (0..2). Every byte is >= 8, so
// the packed subtract cannot borrow across bytes.
__device__ __forceinline__ uint32_t iq2s_word(uint32_t g1, uint32_t g2, uint32_t s8) {
  const uint32_t c1 = (((g1 - 0x08080808U) >> 4) & 0x03030303U) | iq3s_sign4(s8);
  const uint32_t c2 = (((g2 - 0x08080808U) >> 4) & 0x03030303U) | iq3s_sign4(s8 >> 4);
  return __builtin_amdgcn_perm(c2 | (c2 >> 4), c1 | (c1 >> 4), 0x06040200U);
}

// IQ block codes (qs[0..15]: low nibbles = elements 0..15, high = 16..31)
// -> pair order (byte b = elements 2b | 2b+1 << 4).
__device__ __forceinline__ uint32_t pair_lo(uint32_t w) {
  const uint32_t l = w & 0x0F0F0F0FU;
  return l | (l >> 4);  // bytes 0, 2 = (q0 | q1 << 4), (q2 | q3 << 4)
}
__device__ __forceinline__ uint32_t pair_hi(uint32_t w) {
  const uint32_t h = (w >> 4) & 0x0F0F0F0FU;
  return h | (h >> 4);
}
__device__ __forceinline__ uint4 iq_to_pairs(uint4 q) {
  constexpr uint32_t sel = 0x06040200U;  // bytes 0, 2 of b, then 0, 2 of a
  return make_uint4(__builtin_amdgcn_perm(pair_lo(q.y), pair_lo(q.x), sel),
                    __builtin_amdgcn_perm(pair_lo(q.w), pair_lo(q.z), sel),
                    __builtin_amdgcn_perm(pair_hi(q.y), pair_hi(q.x), sel),
                    __builtin_amdgcn_perm(pair_hi(q.w), pair_hi(q.z), sel));
}

// Q2_0: 16 elements per word, four 2-bit codes per byte and 4 bytes per word.
// Expand each byte's codes to pair order (byte b = element 2b | 2b+1 << 4):
//   pe/po hold the even/odd element pairs of bytes 0..3, two v_perm interleave
//   them into elements 0..7 and 8..15.
__device__ __forceinline__ void q2_0_pairs(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t pe = (w & 0x03030303U) | ((w & 0x0C0C0C0CU) << 2);
  const uint32_t po = ((w >> 4) & 0x03030303U) | ((w & 0xC0C0C0C0U) >> 2);
  lo = __builtin_amdgcn_perm(pe, po, 0x01050004U);  // [e0,o0,e1,o1]
  hi = __builtin_amdgcn_perm(pe, po, 0x03070206U);  // [e2,o2,e3,o3]
}

}  // namespace moelut

template <int kLut, bool kPair, int BN = 64, bool kRaw = false, bool kF16Pairs = false,
          bool kMid = false, bool kQ4Perm = false, bool kPkForce = false>
__launch_bounds__(256) __global__
    void k_moe_lut(moelut::LutW p, const __half* __restrict__ x, const int* __restrict__ tokidx,
                   const MoeTile* __restrict__ tiles, const int* __restrict__ ntiles,
                   float* __restrict__ out, __half* __restrict__ out_half, int m, int k,
                   const float* __restrict__ svh_gu = nullptr, const float* __restrict__ suh_dn = nullptr) {
  using namespace moegg;
  using namespace moelut;
  constexpr int BM = 128, BK = 2;
#ifdef MOE_LUT_SHARE34_PROTO
  // tools/moe_lut_test.cu 专用原型口（该文件 include 本 inc 前定义此宏）：
  // SHARE 3/4 已在 §13.7 否决（0 spill 但 occupancy 塌缩 −120%），仅留复测能力。
  static_assert(BN == 64 || BN == 128 || (BN > 128 && kLut == kQ4CP),
                "BN = MoeTile slots: 64, 128, or the q4cp-only 192/256 prototypes");
#else
  static_assert(BN == 64 || BN == 128, "BN = MoeTile slots: 64, or 128 for the merged super-tiles (mt)");
#endif
  static_assert(kLut == kQ4CP || kLut == kIQ4NL || kLut == kIQ4XS || kLut == kIQ3S || kLut == kI4R ||
                    kLut == kQ2_0 || kLut == kIQ2XXS || kLut == kIQ2XS || kLut == kIQ3XXS || kLut == kIQ2S,
                "LUT type");
  static_assert(!kRaw || kPair, "kRaw is a gate/up epilogue");
  static_assert(!kMid || (kLut == kI4R && !kPair && !kF16Pairs), "kMid is the i4r fused up+mid");
  static_assert(!kQ4Perm || kLut == kQ4CP, "kQ4Perm is a q4cp register-codebook decode");
  constexpr int kTokTiles = BN / 16;
  constexpr int kCodeBytes = BM * BK * 16;
  // IQ2_S and IQ2_XS carry two scales per 32-element K block (one per 16 elements).
  constexpr int kSPB = (kLut == kIQ2S || kLut == kIQ2XS) ? 2 : 1;
  constexpr int kScaleBytes = (kLut == kI4R ? 1 : BK) * kSPB * BM * 4;
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  constexpr int kStageBytes = kCodeBytes + kScaleBytes + kActBytes;
  // kMid: gate spill plane [slot][row] f16 at lds[0, 16K); the stage lives at
  // lds[16K, 16K+kStageBytes); the up spill plane reuses the stage's address
  // range after the last __syncthreads of pass 1. Row stride is BM+2 halves:
  // the pass epilogue writes column wave*16+2*l+half per slot, and a BM stride
  // puts all 16 slots of a wave in the same bank (16-way conflict).
  // BN=128 (mt super-tiles): the gate pass spills all 128 slots into two
  // adjacent planes at lds[0, 32K) and the stage moves to lds[32K, 32K+kStageBytes);
  // the up pass's 128 slots cannot coexist with those, so the up spill + mid
  // run in two 64-slot chunks through a single up plane over the dead stage.
  // kPk (hipfire-style packed epilogue, i4r up-mid / down): the weight fetch
  // puts global row 8h+l of each 16-row wave block into LDS slot 2l+h, so the
  // WMMA C fragment (acc[j][l] = LDS slot 2l + half_id) holds 8 CONSECUTIVE
  // rows half_id*8 + l per lane. The compute loop's LDS addressing is
  // unchanged and every output keeps its K order -> bit-identical; the
  // epilogues store 16 B per lane instead of transposing through LDS.
  // MOE_LUT_NOPK restores the old fetch order + epilogues.
#ifdef MOE_LUT_NOPK
  constexpr bool kPk = false;
#else
  constexpr bool kPk = kPkForce || (kLut == kI4R && !kPair);
#endif
  // kPk: row stride a multiple of 8 halves so the packed spill is one
  // ds_write_b128 (272 B stride = 68 dwords: 8 lanes of a phase hit
  // disjoint 4-bank groups).
  constexpr int kMidRow = BM + (kPk ? 8 : 2);
  constexpr int kMidPlaneB = 64 * kMidRow * 2;
  constexpr int kStageOff = kMid ? (BN > 64 ? 2 * kMidPlaneB : kMidPlaneB) : 0;
  // aggr 10-05: non-kMid kernels double-buffer the LDS stage (commit of
  // stage s+1 overlaps compute of stage s, one barrier per stage). Same K
  // order -> bit-identical. MOE_LUT_NODB restores the single buffer.
  // measured 10-05: down share2 15.98 -> 16.33 ms (LDS x2 halves occupancy) -> opt-in only
#ifdef MOE_LUT_DB
  constexpr bool kDB = !kMid && BN <= 128;
#else
  constexpr bool kDB = false;
#endif
  constexpr int kLdsStage = (kDB ? 2 : 1) * kStageBytes > 8 * 1024 ? (kDB ? 2 : 1) * kStageBytes
                                                                    : 8 * 1024;
  constexpr int kLdsBytes =
      kMid ? kStageOff + (kStageBytes > kMidPlaneB ? kStageBytes : kMidPlaneB) : kLdsStage;
  static_assert(!kMid || kLdsBytes <= 64 * 1024, "kMid LDS");
  __shared__ __attribute__((aligned(16))) uint8_t lds[kLdsBytes];
  __shared__ uint32_t s_cbp[kLut == kI4R || kQ4Perm ? 1 : 256];  // i4r: arithmetic; perm: registers
  __shared__ uint32_t s_grid[kLut == kIQ3S ? 512 : kLut == kIQ3XXS ? 256 : 1];
  __shared__ uint64_t s_grid64[kLut == kIQ2S ? 1024 : kLut == kIQ2XS ? 512 : kLut == kIQ2XXS ? 256 : 1];
  __shared__ uint8_t s_ksigns[(kLut == kIQ3XXS || kLut == kIQ2XXS || kLut == kIQ2XS) ? 128 : 1];
  uint4* s_codes = reinterpret_cast<uint4*>(lds + kStageOff);
  uint32_t* s_scale = reinterpret_cast<uint32_t*>(lds + kStageOff + kCodeBytes);
  uint4* s_act = reinterpret_cast<uint4*>(lds + kStageOff + kCodeBytes + kScaleBytes);
  const auto set_buf = [&](int b) {
    uint8_t* const sb = lds + kStageOff + b * kStageBytes;
    s_codes = reinterpret_cast<uint4*>(sb);
    s_scale = reinterpret_cast<uint32_t*>(sb + kCodeBytes);
    s_act = reinterpret_cast<uint4*>(sb + kCodeBytes + kScaleBytes);
  };

  if ((int)blockIdx.y >= *ntiles) return;
  const int tid = threadIdx.x;
  [[maybe_unused]] uint32_t r_lt[4], r_ht[4];  // kQ4Perm register codebook
  if constexpr (kLut != kI4R) {
    if constexpr (kQ4Perm) {
      // Codebook in registers instead of the s_cbp LDS table: r_lt[j] / r_ht[j]
      // hold the low / high f16 bytes of cb[4j..4j+3], one entry per byte
      // (perm pool = {b:0-3, a:4-7}). Same __floats2half2_rn rounding as the
      // table fill below, so the decode is bit-identical.
      uint32_t h2[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        h2[j] = __builtin_bit_cast(uint32_t, __floats2half2_rn(p.cb[2 * j], p.cb[2 * j + 1]));
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        r_lt[j] = __builtin_amdgcn_perm(h2[2 * j + 1], h2[2 * j], 0x06040200U);
        r_ht[j] = __builtin_amdgcn_perm(h2[2 * j + 1], h2[2 * j], 0x07050301U);
      }
    } else {
      float lo, hi;
      if constexpr (kLut == kQ4CP) {
        lo = p.cb[tid & 15];
        hi = p.cb[tid >> 4];
      } else if constexpr (kLut == kIQ3S) {  // code c: +-(2 (c & 7) + 1), sign = bit 3
        const int cl = tid & 15, ch = tid >> 4;
        lo = (float)((cl & 8) ? -(2 * (cl & 7) + 1) : 2 * cl + 1) * (1.0f / 256.0f);
        hi = (float)((ch & 8) ? -(2 * (ch & 7) + 1) : 2 * ch + 1) * (1.0f / 256.0f);
      } else if constexpr (kLut == kIQ3XXS) {  // code c: +-{4,12,...,52,62} (c == 7 is 62)
        const int cl = tid & 15, ch = tid >> 4;
        const int ml = 4 + 8 * (cl & 7) + ((cl & 7) == 7 ? 2 : 0);
        const int mh = 4 + 8 * (ch & 7) + ((ch & 7) == 7 ? 2 : 0);
        lo = (float)((cl & 8) ? -ml : ml) * (1.0f / 256.0f);
        hi = (float)((ch & 8) ? -mh : mh) * (1.0f / 256.0f);
      } else if constexpr (kLut == kIQ2S || kLut == kIQ2XXS || kLut == kIQ2XS) {
        // code 0..2 -> 8,25,43, sign = bit 3
        const int cl = tid & 15, ch = tid >> 4;
        const int ml = (cl & 3) == 0 ? 8 : (cl & 3) == 1 ? 25 : 43;
        const int mh = (ch & 3) == 0 ? 8 : (ch & 3) == 1 ? 25 : 43;
        lo = (float)((cl & 8) ? -ml : ml) * (1.0f / 256.0f);
        hi = (float)((ch & 8) ? -mh : mh) * (1.0f / 256.0f);
      } else if constexpr (kLut == kQ2_0) {  // pair nibbles are codes 0..3 -> -1..2
        lo = (float)((tid & 3) - 1);
        hi = (float)(((tid >> 4) & 3) - 1);
      } else {
        lo = (float)c_iq4_values[tid & 15] * (1.0f / 256.0f);  // exact; scale carries * 256
        hi = (float)c_iq4_values[tid >> 4] * (1.0f / 256.0f);
      }
      s_cbp[tid] = __builtin_bit_cast(uint32_t, __floats2half2_rn(lo, hi));
    }
  }  // visible after the first __syncthreads of the main loop
  if constexpr (kLut == kIQ3S) {
    // the grid is read by commit_stage, which runs before the loop's first
    // barrier: fill + sync here (the ntiles early return above is
    // block-uniform, so this barrier is not divergent)
    s_grid[tid] = moegg::c_iq3s_grid[tid];
    s_grid[tid + 256] = moegg::c_iq3s_grid[tid + 256];
    __syncthreads();
  } else if constexpr (kLut == kIQ3XXS) {
    s_grid[tid] = moegg::c_iq3xxs_grid[tid];
    if (tid < 128) s_ksigns[tid] = moegg::c_iq2xs_signs[tid];
    __syncthreads();
  } else if constexpr (kLut == kIQ2XXS) {
    s_grid64[tid] = moegg::c_iq2xxs_grid[tid];
    if (tid < 128) s_ksigns[tid] = moegg::c_iq2xs_signs[tid];
    __syncthreads();
  } else if constexpr (kLut == kIQ2XS) {
    s_grid64[tid] = moegg::c_iq2xs_grid[tid];
    s_grid64[tid + 256] = moegg::c_iq2xs_grid[tid + 256];
    if (tid < 128) s_ksigns[tid] = moegg::c_iq2xs_signs[tid];
    __syncthreads();
  } else if constexpr (kLut == kIQ2S) {
#pragma unroll
    for (int i = 0; i < 4; ++i) s_grid64[tid + 256 * i] = moegg::c_iq2s_grid[tid + 256 * i];
    __syncthreads();
  }

  const MoeTile tile = tiles[blockIdx.y];
  const int expert = tile.expert;
  const int slot0 = tile.first;
  const int nrows = tile.count;  // <= BN
  const int live_tok_tiles = (nrows + 15) / 16;
  const int num_kb = k / 32;
  const size_t rb = lut_row_bytes<kLut>((size_t)k);

  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = (int)blockIdx.x * kRows;

  // Weight fetch: thread = (row tid/2, K block kb0 + tid%2). kMid runs the K
  // pipeline twice: pass 0 = gate rows, pass 1 = the matching up rows.
  const int f_c = tid & 1;
  const uint8_t* f_ptr;
  const uint8_t* f_sp = nullptr;
  bool f_live;
  const auto set_fetch = [&](int pass) {
    const bool upper = (kPair && (tid >> 1) >= kRows) || (kMid && pass != 0);
    int lr = kPair ? (tid >> 1) % kRows : (tid >> 1);  // LDS slot of this fetch thread
    if constexpr (kPk) lr = (lr & ~15) | ((lr & 1) << 3) | ((lr & 15) >> 1);  // slot 2l+h <- row 8h+l
    const int r = r_block + lr;
    f_live = r < m;
    const size_t row = (size_t)expert * p.e_rows + (upper ? p.up_off : 0) + (f_live ? r : (m - 1));
    f_ptr = (upper ? p.w_up : p.w) + row * rb;
    if constexpr (kLut == kQ4CP || kLut == kI4R)
      f_sp = (upper ? p.sc_up : p.sc) + row * (size_t)p.sstride;
  };
  uint4 f_codes = make_uint4(0u, 0u, 0u, 0u);
  uint32_t f_s = 0;   // q4cp / IQ4_NL: fp16 scale bits; IQ4_XS: header word 0 (d | scales_h << 16)
  uint32_t f_sl = 0;  // IQ4_XS: scales_l[0..3]
  int f_ib = 0;       // IQ4_XS: sub-block
  uint4 f_scv = make_uint4(0u, 0u, 0u, 0u);  // q4cp: scales of K blocks kb0 & ~7 .. +7
  constexpr int kActFetch = 2 * (BN / 64);
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256, "activation fetch");
  uint4 a_data[kActFetch];
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + i * 256;
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    int src = -1;
    if (chunk < kActChunks && t < nrows) src = tokidx ? tokidx[slot0 + t] : slot0 + t;
    a_src[i] = src >= 0 ? x + (size_t)src * k + sub * 8 : nullptr;
    a_slot[i] = chunk < kActChunks ? sub * kActStride + t : -1;
  }

  const auto swizzle = [](int row, int c) { return row * BK + (c ^ ((row >> 2) & 1)); };

  const auto fetch_stage = [&](int kb0) {
    const int kb = kb0 + f_c;
    if constexpr (kLut == kQ4CP) {
      f_codes = *reinterpret_cast<const uint4*>(f_ptr + (size_t)kb * 16);
      // 8 scales (16 B, rows padded to 16 B) per 4 stages, kept in registers.
      if (kb0 % 8 == 0) f_scv = *reinterpret_cast<const uint4*>(f_sp + (size_t)kb0 * 2);
      const int si = (kb0 % 8) + f_c;
      const uint32_t sw = si < 2 ? f_scv.x : si < 4 ? f_scv.y : si < 6 ? f_scv.z : f_scv.w;
      f_s = (si & 1) ? sw >> 16 : sw & 0xFFFFU;
    } else if constexpr (kLut == kI4R) {
      f_codes = *reinterpret_cast<const uint4*>(f_ptr + (size_t)kb * 16);
      // one fp16 scale per 128 elements = per 2 stages: fetch once, keep in reg
      if ((kb0 & 3) == 0) f_s = *reinterpret_cast<const uint16_t*>(f_sp + (size_t)(kb0 >> 2) * 2);
    } else if constexpr (kLut == kIQ4NL) {
      // 18 B block at a 4 B (kb even) or 2 B (kb odd) boundary: 5 aligned
      // words cover it exactly (rows are 36-byte multiples, num_kb even).
      const uint8_t* blk = f_ptr + (size_t)kb * 18;
      const auto* wp = reinterpret_cast<const uint32_t*>(blk - 2 * f_c);
      const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3], w4 = wp[4];
      if (f_c) {
        f_s = w0 >> 16;
        f_codes = make_uint4(w1, w2, w3, w4);
      } else {
        f_s = w0 & 0xFFFFU;
        f_codes = make_uint4(__builtin_amdgcn_alignbyte(w1, w0, 2), __builtin_amdgcn_alignbyte(w2, w1, 2),
                             __builtin_amdgcn_alignbyte(w3, w2, 2), __builtin_amdgcn_alignbyte(w4, w3, 2));
      }
    } else if constexpr (kLut == kQ2_0) {
      // 18 B / 64, 2-byte aligned: BK=2 is exactly one block; f_c picks the
      // 32-element half (8 code bytes = four uint16).
      const uint8_t* blk = f_ptr + (size_t)(kb >> 1) * 18;
      const uint16_t* b16 = reinterpret_cast<const uint16_t*>(blk);
      f_s = b16[0];
      const uint16_t* qs = b16 + 1 + 4 * f_c;
      f_codes = make_uint4((uint32_t)qs[0] | ((uint32_t)qs[1] << 16),
                           (uint32_t)qs[2] | ((uint32_t)qs[3] << 16), 0u, 0u);
    } else if constexpr (kLut == kIQ3S) {
      // raw sub-block kb % 8 of super-block kb / 8 (110 B, 2-byte aligned):
      // f_codes = qs[8] (x, y), signs[4] (z), qh byte | scale nibble << 8 (w)
      const int ib = kb & 7;
      const uint8_t* blk = f_ptr + (size_t)(kb >> 3) * 110;
      const uint16_t* b16 = reinterpret_cast<const uint16_t*>(blk);
      const uint16_t* qs = b16 + 1 + 4 * ib;
      const uint32_t sc = (blk[106 + (ib >> 1)] >> (4 * (ib & 1))) & 0xFU;
      f_s = b16[0];
      f_codes = make_uint4((uint32_t)qs[0] | ((uint32_t)qs[1] << 16), (uint32_t)qs[2] | ((uint32_t)qs[3] << 16),
                           (uint32_t)b16[37 + 2 * ib] | ((uint32_t)b16[38 + 2 * ib] << 16),
                           (uint32_t)blk[66 + ib] | (sc << 8));
    } else if constexpr (kLut == kIQ2XXS) {
      // 66 B super-block: four grid indices (x) and the 4-byte scale+sign word (y).
      const int ib = kb & 7;
      const uint8_t* blk = f_ptr + (size_t)(kb >> 3) * 66;
      const uint16_t* b16 = reinterpret_cast<const uint16_t*>(blk);
      f_s = b16[0];
      f_codes = make_uint4((uint32_t)b16[1 + 4 * ib] | ((uint32_t)b16[2 + 4 * ib] << 16),
                           (uint32_t)b16[3 + 4 * ib] | ((uint32_t)b16[4 + 4 * ib] << 16), 0u, 0u);
    } else if constexpr (kLut == kIQ2XS) {
      // 74 B super-block: four 16-bit grid/sign words (x, y) and one scale byte (z).
      const int ib = kb & 7;
      const uint8_t* blk = f_ptr + (size_t)(kb >> 3) * 74;
      const uint16_t* b16 = reinterpret_cast<const uint16_t*>(blk);
      const uint16_t* qs = b16 + 1 + 4 * ib;
      f_s = b16[0];
      f_codes = make_uint4((uint32_t)qs[0] | ((uint32_t)qs[1] << 16),
                           (uint32_t)qs[2] | ((uint32_t)qs[3] << 16), blk[66 + ib], 0u);
    } else if constexpr (kLut == kIQ3XXS) {
      // 98 B super-block: qs[8] (x, y) and the 4-byte scale+sign word (z).
      const int ib = kb & 7;
      const uint8_t* blk = f_ptr + (size_t)(kb >> 3) * 98;
      const uint16_t* b16 = reinterpret_cast<const uint16_t*>(blk);
      f_s = b16[0];
      f_codes = make_uint4((uint32_t)b16[1 + 4 * ib] | ((uint32_t)b16[2 + 4 * ib] << 16),
                           (uint32_t)b16[3 + 4 * ib] | ((uint32_t)b16[4 + 4 * ib] << 16),
                           (uint32_t)b16[33 + 2 * ib] | ((uint32_t)b16[34 + 2 * ib] << 16), 0u);
    } else if constexpr (kLut == kIQ2S) {
      // 82 B super-block: qs[4] (x), signs[4] (y), qh byte | scale byte << 8 (z).
      const int ib = kb & 7;
      const uint8_t* blk = f_ptr + (size_t)(kb >> 3) * 82;
      const uint16_t* b16 = reinterpret_cast<const uint16_t*>(blk);
      f_s = b16[0];
      f_codes = make_uint4((uint32_t)b16[1 + 2 * ib] | ((uint32_t)b16[2 + 2 * ib] << 16),
                           (uint32_t)b16[17 + 2 * ib] | ((uint32_t)b16[18 + 2 * ib] << 16),
                           (uint32_t)blk[66 + ib] | ((uint32_t)blk[74 + ib] << 8), 0u);
    } else {
      const uint8_t* blk = f_ptr + (size_t)(kb / 8) * 136;
      const uint2 hdr = *reinterpret_cast<const uint2*>(blk);
      const auto* qs = reinterpret_cast<const uint2*>(blk + 8 + 16 * (kb % 8));
      const uint2 q0 = qs[0], q1 = qs[1];
      f_s = hdr.x;
      f_sl = hdr.y;
      f_ib = kb % 8;
      f_codes = make_uint4(q0.x, q0.y, q1.x, q1.y);
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      a_data[i] = a_src[i] ? *reinterpret_cast<const uint4*>(a_src[i] + kb0 * 32)
                           : make_uint4(0u, 0u, 0u, 0u);
  };

  const auto commit_stage = [&]() {
    const int row = tid >> 1;
    uint32_t s16, s16b = 0;
    if constexpr (kLut == kQ4CP || kLut == kI4R) {
      s_codes[swizzle(row, f_c)] = f_codes;
      s16 = f_s;
    } else if constexpr (kLut == kQ2_0) {
      uint32_t s0, s1, s2, s3;
      q2_0_pairs(f_codes.x, s0, s1);
      q2_0_pairs(f_codes.y, s2, s3);
      s_codes[swizzle(row, f_c)] = make_uint4(s0, s1, s2, s3);
      s16 = (uint16_t)(f_s & 0xFFFFU);  // levels are -1..2: scale is used as-is
    } else if constexpr (kLut == kIQ3S) {
      const uint32_t h = f_codes.w & 0xFFU;
      uint32_t wd[4];
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint32_t q2 = ((l < 2 ? f_codes.x : f_codes.y) >> (16 * (l & 1))) & 0xFFFFU;
        const uint32_t g1 = s_grid[(q2 & 0xFFU) | ((h << (8 - 2 * l)) & 256U)];
        const uint32_t g2 = s_grid[(q2 >> 8) | ((h << (7 - 2 * l)) & 256U)];
        wd[l] = iq3s_word(g1, g2, (f_codes.z >> (8 * l)) & 0xFFU);
      }
      s_codes[swizzle(row, f_c)] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
      const float d = __half2float(__builtin_bit_cast(__half, (uint16_t)(f_s & 0xFFFFU))) * 256.0f;
      s16 = __builtin_bit_cast(uint16_t, __float2half(d * (float)(1 + 2 * (int)(f_codes.w >> 8))));
    } else if constexpr (kLut == kIQ2XXS) {
      const uint32_t q = f_codes.x, aux = f_codes.y;
      uint32_t wd[4];
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint64_t gg = s_grid64[(q >> (8 * l)) & 0xFFU];
        wd[l] = iq2s_word((uint32_t)gg, (uint32_t)(gg >> 32), s_ksigns[(aux >> (7 * l)) & 127]);
      }
      s_codes[swizzle(row, f_c)] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
      const float d = __half2float(__builtin_bit_cast(__half, (uint16_t)(f_s & 0xFFFFU))) * 256.0f;
      s16 = __builtin_bit_cast(uint16_t,
                               __float2half(d * (0.5f + (float)(aux >> 28)) * 0.25f));
    } else if constexpr (kLut == kIQ2XS) {
      const uint32_t sc = f_codes.z & 0xFFU;
      uint32_t wd[4];
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint32_t q = ((l < 2 ? f_codes.x : f_codes.y) >> (16 * (l & 1))) & 0xFFFFU;
        const uint64_t gg = s_grid64[q & 511U];
        wd[l] = iq2s_word((uint32_t)gg, (uint32_t)(gg >> 32), s_ksigns[q >> 9]);
      }
      s_codes[swizzle(row, f_c)] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
      const float d = __half2float(__builtin_bit_cast(__half, (uint16_t)(f_s & 0xFFFFU))) * 256.0f;
      s16 = __builtin_bit_cast(uint16_t,
                               __float2half(d * (0.5f + (float)(sc & 0xFU)) * 0.25f));
      s16b = __builtin_bit_cast(uint16_t,
                                __float2half(d * (0.5f + (float)(sc >> 4)) * 0.25f));
    } else if constexpr (kLut == kIQ3XXS) {
      const uint32_t aux = f_codes.z;
      uint32_t wd[4];
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint32_t q2 = ((l < 2 ? f_codes.x : f_codes.y) >> (16 * (l & 1))) & 0xFFFFU;
        wd[l] = iq3xxs_word(s_grid[q2 & 0xFFU], s_grid[(q2 >> 8) & 0xFFU],
                            s_ksigns[(aux >> (7 * l)) & 127]);
      }
      s_codes[swizzle(row, f_c)] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
      const float d = __half2float(__builtin_bit_cast(__half, (uint16_t)(f_s & 0xFFFFU))) * 256.0f;
      s16 = __builtin_bit_cast(uint16_t,
                               __float2half(d * (0.5f + (float)(aux >> 28)) * 0.5f));
    } else if constexpr (kLut == kIQ2S) {
      const uint32_t qs = f_codes.x, sg = f_codes.y;
      const uint32_t qh = f_codes.z & 0xFFU, sc = f_codes.z >> 8;
      uint32_t wd[4];
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const uint64_t gg = s_grid64[((qs >> (8 * l)) & 0xFFU) | ((qh << (8 - 2 * l)) & 0x300U)];
        wd[l] = iq2s_word((uint32_t)gg, (uint32_t)(gg >> 32), (sg >> (8 * l)) & 0xFFU);
      }
      s_codes[swizzle(row, f_c)] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
      const float d = __half2float(__builtin_bit_cast(__half, (uint16_t)(f_s & 0xFFFFU))) * 256.0f;
      s16 = __builtin_bit_cast(uint16_t,
                               __float2half(d * (0.5f + (float)(sc & 0xFU)) * 0.25f));
      s16b = __builtin_bit_cast(uint16_t,
                                __float2half(d * (0.5f + (float)(sc >> 4)) * 0.25f));
    } else {
      s_codes[swizzle(row, f_c)] = iq_to_pairs(f_codes);
      // IQ scales are ~1e-5 (fp16 subnormal range): carry scale * 256 and
      // lut / 256 so the per-block scale keeps full fp16 precision.
      const float d = __half2float(__builtin_bit_cast(__half, (uint16_t)(f_s & 0xFFFFU))) * 256.0f;
      if constexpr (kLut == kIQ4NL) {
        s16 = __builtin_bit_cast(uint16_t, __float2half(d));
      } else {
        const int ib = f_ib;
        const int ls = (int)((f_sl >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xFU) |
                       (int)(((f_s >> (16 + 2 * ib)) & 3U) << 4);
        s16 = __builtin_bit_cast(uint16_t, __float2half(d * (float)(ls - 32)));
      }
    }
    const int sslot = (kLut == kI4R ? 0 : f_c) * kSPB;
    s_scale[sslot * BM + row] = f_live ? (s16 | (s16 << 16)) : 0U;
    if constexpr (kSPB == 2) s_scale[(sslot + 1) * BM + row] = f_live ? (s16b | (s16b << 16)) : 0U;
#pragma unroll
    for (int i = 0; i < kActFetch; ++i)
      if (a_slot[i] >= 0) s_act[a_slot[i]] = a_data[i];
  };

  v8f acc[kTokTiles];

  const auto compute_stage = [&]() {
    const int row = wave_id * 16 + sub_lane;
    uint4 raw[BK];
#pragma unroll
    for (int c = 0; c < BK; ++c) raw[c] = s_codes[swizzle(row, c)];
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      const __half2 scale2 = __builtin_bit_cast(
          __half2, s_scale[(kLut == kI4R ? row : kb * kSPB * BM + row)]);
      [[maybe_unused]] __half2 scale2b = scale2;
      if constexpr (kSPB == 2)
        scale2b = __builtin_bit_cast(__half2, s_scale[kb * kSPB * BM + BM + row]);
      const uint32_t words[4] = {raw[kb].x, raw[kb].y, raw[kb].z, raw[kb].w};
      __half2 h[16];
#ifdef MOE_LUT_ABLATE_DECODE
      if constexpr (kLut == kI4R) {  // ablation: raw bits as halves (wrong numerics, timing only)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[4 * i + 0] = __builtin_bit_cast(__half2, words[i]);
          h[4 * i + 1] = __builtin_bit_cast(__half2, words[i] ^ 0x11111111u);
          h[4 * i + 2] = __builtin_bit_cast(__half2, words[i] ^ 0x22222222u);
          h[4 * i + 3] = __builtin_bit_cast(__half2, words[i] ^ 0x33333333u);
        }
      } else
#endif
      if constexpr (kLut == kI4R) {
        // Arithmetic decode (bit-identical to the s_cbp table path): f16 bits
        // 0x6400+nib = 1024+nib (exact); minus 1032 -> nib-8 (exact,
        // Sterbenz); * scale rounds the same. v_perm pool = {b:0-3, a:4-7}.
        const __half2 bias = __builtin_bit_cast(__half2, 0x64086408U);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t w = words[i];
          const uint32_t lo = w & 0x0F0F0F0FU;
          const uint32_t hi = (w >> 4) & 0x0F0F0F0FU;
          const uint32_t p0 = __builtin_amdgcn_perm(hi, lo, 0x05040100U);  // [l0,l1,h0,h1]
          const uint32_t p1 = __builtin_amdgcn_perm(hi, lo, 0x07060302U);  // [l2,l3,h2,h3]
          const uint32_t q[4] = {p0, p0 >> 8, p1, p1 >> 8};
#pragma unroll
          for (int j = 0; j < 4; ++j)
            h[4 * i + j] =
                __hmul2(__hsub2(__builtin_bit_cast(__half2, (q[j] & 0x00FF00FFU) | 0x64006400U), bias),
                        scale2);
        }
      } else if constexpr (kQ4Perm) {
        // q4cp register decode: for each byte b the value is the f16 pair
        // (cb[b & 15], cb[b >> 4]) = 4 bytes from the r_lt / r_ht tables. Two
        // v_perm + one bfi pick a byte across the two table halves (selector
        // n vs n ^ 8 so both stay in range regardless of how out-of-range
        // selectors behave; the other half is masked), then perms interleave
        // low/high nibbles into half2 lanes. Bit-identical to s_cbp[b].
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t w = words[i];
          const uint32_t lo = w & 0x0F0F0F0FU;         // low nibbles of bytes 0..3
          const uint32_t hi = (w >> 4) & 0x0F0F0F0FU;  // high nibbles
          const uint32_t lob = lo ^ 0x08080808U, hib = hi ^ 0x08080808U;
          const uint32_t me = ((lo >> 3) & 0x01010101U) * 0xFFU;  // 0xFF where nibble >= 8
          const uint32_t mo = ((hi >> 3) & 0x01010101U) * 0xFFU;
          const uint32_t lte = (me & __builtin_amdgcn_perm(r_lt[3], r_lt[2], lob)) |
                               (~me & __builtin_amdgcn_perm(r_lt[1], r_lt[0], lo));
          const uint32_t hte = (me & __builtin_amdgcn_perm(r_ht[3], r_ht[2], lob)) |
                               (~me & __builtin_amdgcn_perm(r_ht[1], r_ht[0], lo));
          const uint32_t lto = (mo & __builtin_amdgcn_perm(r_lt[3], r_lt[2], hib)) |
                               (~mo & __builtin_amdgcn_perm(r_lt[1], r_lt[0], hi));
          const uint32_t hto = (mo & __builtin_amdgcn_perm(r_ht[3], r_ht[2], hib)) |
                               (~mo & __builtin_amdgcn_perm(r_ht[1], r_ht[0], hi));
          const uint32_t fe0 = __builtin_amdgcn_perm(hte, lte, 0x05010400U);  // [le0,he0,le1,he1]
          const uint32_t fe1 = __builtin_amdgcn_perm(hte, lte, 0x07030602U);  // [le2,he2,le3,he3]
          const uint32_t fo0 = __builtin_amdgcn_perm(hto, lto, 0x05010400U);  // [lo0,ho0,lo1,ho1]
          const uint32_t fo1 = __builtin_amdgcn_perm(hto, lto, 0x07030602U);
          h[4 * i + 0] = __hmul2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(fo0, fe0, 0x05040100U)), scale2);
          h[4 * i + 1] = __hmul2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(fo0, fe0, 0x07060302U)), scale2);
          h[4 * i + 2] = __hmul2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(fo1, fe1, 0x05040100U)), scale2);
          h[4 * i + 3] = __hmul2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(fo1, fe1, 0x07060302U)), scale2);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const uint32_t b = (words[i >> 2] >> (8 * (i & 3))) & 0xFFU;
          // IQ2_S: byte i < 8 covers elements 0..15 (scale0), i >= 8 elements
          // 16..31 (scale1).
          h[i] = __hmul2(__builtin_bit_cast(__half2, s_cbp[b]), i < 8 ? scale2 : scale2b);
        }
      }
      v16h a_lo, a_hi;
      __builtin_memcpy(&a_lo, &h[0], 32);
      __builtin_memcpy(&a_hi, &h[8], 32);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        asm volatile("" ::: "memory");  // keep one token tile's fragments live
        if (j >= live_tok_tiles) continue;
        const uint4* frag = s_act + (kb * 4) * kActStride + j * 16 + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) b[q] = frag[q * kActStride];
        v16h b_lo, b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
        acc[j] = wmma(a_lo, b_lo, acc[j]);
        acc[j] = wmma(a_hi, b_hi, acc[j]);
      }
    }
  };

  // kMid spill of one token tile row: plane[(t) * kMidRow + wave rows] = f16(acc).
  // kPk: this lane's 8 rows are consecutive -> one 16 B LDS store.
  [[maybe_unused]] const auto spill_row = [&](__half* plane, int t, const v8f& a) {
    if constexpr (kPk) {
      __half2 h2[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) h2[q] = __halves2half2(__float2half(a[2 * q]), __float2half(a[2 * q + 1]));
      uint4 v;
      __builtin_memcpy(&v, h2, 16);
      *reinterpret_cast<uint4*>(plane + t * kMidRow + wave_id * 16 + half_id * 8) = v;
    } else {
#pragma unroll
      for (int l = 0; l < 8; ++l) plane[t * kMidRow + wave_id * 16 + 2 * l + half_id] = __float2half(a[l]);
    }
  };

  for (int pass = 0; pass < (kMid ? 2 : 1); ++pass) {
    set_fetch(pass);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    fetch_stage(0);
    if constexpr (kDB) {
      set_buf(0);
      commit_stage();
      __syncthreads();
      for (int kb0 = 0, b = 0; kb0 < num_kb; kb0 += BK, b ^= 1) {
        const bool more = kb0 + BK < num_kb;
        if (more) fetch_stage(kb0 + BK);
        set_buf(b);
        compute_stage();
        if (more) {
          set_buf(b ^ 1);
          commit_stage();
        }
        __syncthreads();
      }
    } else {
      for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
        commit_stage();
        __syncthreads();
        if (kb0 + BK < num_kb) fetch_stage(kb0 + BK);
        compute_stage();
        __syncthreads();
      }
    }
    if constexpr (kMid) {
      // Spill acc to the f16 plane [slot][row]: the same rounding the kRaw
      // epilogue applies writing guv16, so the fused mid below is bit-identical
      // to k_i4r_mid reading guv16 back. Pass 1's plane overlaps the stage,
      // which is dead after the loop's last __syncthreads. C fragment lane
      // mapping (from the pair/down epilogues): acc[j][l] = (token j*16 +
      // sub_lane, LDS slot wave*16 + 2*l + half_id); with kPk that slot holds
      // row wave*16 + half_id*8 + l (spill_row stores the 8 rows at once).
      // BN=128: pass 0 spills all 128 gate slots into the two adjacent planes
      // at lds[0, 32K) (indexed by t directly); pass 1's up spill is chunked
      // with the mid phase below and happens there, not here.
      if constexpr (BN > 64) {
        if (pass == 0) {
          __half* plane = reinterpret_cast<__half*>(lds);
#pragma unroll
          for (int j = 0; j < kTokTiles; ++j) {
            if (j >= live_tok_tiles) break;
            const int t = j * 16 + sub_lane;
            if (t < nrows) spill_row(plane, t, acc[j]);
          }
        }
      } else {
        __half* plane = reinterpret_cast<__half*>(lds) + pass * 64 * kMidRow;
#pragma unroll
        for (int j = 0; j < kTokTiles; ++j) {
          if (j >= live_tok_tiles) break;
          const int t = j * 16 + sub_lane;
          if (t < nrows) spill_row(plane, t, acc[j]);
        }
      }
    }
  }

  if constexpr (kMid) {
    // Fused i4r mid (replaces the k_i4r_mid launch), one slot per wave:
    //   g|u = svh_gu . H(z)/sqrt128;  h = silu(g)*u*suh_dn;  out = H(h)/sqrt128
    // Same op order and f32 math as k_i4r_mid on guv16.
    const auto mid_slot = [&](int s, const __half* gr, const __half* ur) {
      float g[4], u[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        g[j] = __half2float(gr[j * 32 + lane_id]);
        u[j] = __half2float(ur[j * 32 + lane_id]);
      }
      hv2::fwht128(g[0], g[1], g[2], g[3], lane_id);
      hv2::fwht128(u[0], u[1], u[2], u[3], lane_id);
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = r_block + j * 32 + lane_id;
        const float gg = g[j] * hv2::kRs128 * svh_gu[i];
        const float uu = u[j] * hv2::kRs128 * svh_gu[m + i];
        v[j] = gg / (1.f + expf(-gg)) * uu * suh_dn[i];
      }
      hv2::fwht128(v[0], v[1], v[2], v[3], lane_id);
      __half* o = out_half + (size_t)(slot0 + s) * m + r_block;
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j * 32 + lane_id] = __float2half(v[j] * hv2::kRs128);
    };
    if constexpr (BN > 64) {
      // acc still holds the up pass's values; the gate pass's 128 slots sit in
      // the two planes at lds[0, 32K). The 128 up slots go through a single
      // plane over the dead stage in two 64-slot chunks: spill the chunk, mid
      // it against the matching gate plane, then overwrite with the next.
      __half* uplane = reinterpret_cast<__half*>(lds) + 2 * 64 * kMidRow;
      const int nchunk = (nrows + 63) / 64;
      for (int c = 0; c < nchunk; ++c) {
#pragma unroll
        for (int j = 0; j < kTokTiles; ++j) {
          if (j >= live_tok_tiles) break;
          const int t = j * 16 + sub_lane;
          if (t >= c * 64 && t < c * 64 + 64 && t < nrows) spill_row(uplane, t - c * 64, acc[j]);
        }
        __syncthreads();
        const __half* gplane = reinterpret_cast<const __half*>(lds) + c * 64 * kMidRow;
        const int send = nrows < (c + 1) * 64 ? nrows : (c + 1) * 64;
        for (int s = c * 64 + wave_id; s < send; s += 8)
          mid_slot(s, gplane + (s - c * 64) * kMidRow, uplane + (s - c * 64) * kMidRow);
        __syncthreads();  // mid readers done before the next chunk overwrites uplane
      }
      return;
    }
    __syncthreads();  // both planes fully written
    const __half* gplane = reinterpret_cast<const __half*>(lds);
    const __half* uplane = gplane + 64 * kMidRow;
    for (int s = wave_id; s < nrows; s += 8) mid_slot(s, gplane + s * kMidRow, uplane + s * kMidRow);
    return;
  }

  if constexpr (kPair) {
    // Waves 0-3 hold gate rows, 4-7 the matching up rows: pair them in LDS.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes, "pair epilogue LDS");
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      if (j >= live_tok_tiles) break;  // uniform across the block
#pragma unroll
      for (int l = 0; l < 8; ++l) scratch[sub_lane * stride + 2 * l + half_id] = acc[j][l];
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < nrows && r_block + r < m) {
          const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
          if constexpr (kRaw) {
            const size_t o = (size_t)(slot0 + t) * 2 * m + r_block + r;
            *reinterpret_cast<__half2*>(out_half + o) = __floats2half2_rn(base[idx], base[idx + 1]);
            *reinterpret_cast<__half2*>(out_half + o + m) =
                __floats2half2_rn(base[idx + 4 * plane], base[idx + 1 + 4 * plane]);
            continue;
          }
          __half vals[2];
#pragma unroll
          for (int v = 0; v < 2; ++v) {
            const float g = base[idx + v];
            const float u = base[idx + v + 4 * plane];
            vals[v] = __float2half(u * g * sigmoidf_(g));
          }
          const size_t o = (size_t)(slot0 + t) * m + r_block + r;
          *reinterpret_cast<__half2*>(out_half + o) = __halves2half2(vals[0], vals[1]);
        }
      }
      __syncthreads();
    }
    return;
  }

  const int r0 = r_block + wave_id * 16;
  if constexpr (kPk) {
    // Packed down epilogue: lane (token sub_lane, rows r0 + half_id*8 + 0..7).
    // f16 pairs: one 16 B store; f32: two 16 B stores. Same values and
    // rounding as the LDS-transpose path below.
    const int rr = r0 + half_id * 8;
    const bool vec = (m & 7) == 0 && rr + 8 <= m;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      if (j >= live_tok_tiles) break;
      const int t = j * 16 + sub_lane;
      if (t >= nrows) continue;
      const v8f& a = acc[j];
      if constexpr (kF16Pairs) {
        __half* o = reinterpret_cast<__half*>(out) + (size_t)(slot0 + t) * m + rr;
        if (vec) {
          __half2 h2[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) h2[q] = __floats2half2_rn(a[2 * q], a[2 * q + 1]);
          uint4 v;
          __builtin_memcpy(&v, h2, 16);
          *reinterpret_cast<uint4*>(o) = v;
        } else {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            if (rr + l < m) o[l] = __float2half(a[l]);
        }
      } else {
        float* o = out + (size_t)(slot0 + t) * m + rr;
        if (vec) {
          reinterpret_cast<float4*>(o)[0] = make_float4(a[0], a[1], a[2], a[3]);
          reinterpret_cast<float4*>(o)[1] = make_float4(a[4], a[5], a[6], a[7]);
        } else {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            if (rr + l < m) o[l] = a[l];
        }
      }
    }
    return;
  }
  // Down: transpose each 16x16 tile through LDS; one 64 B row segment per slot.
  float* tile_scratch = reinterpret_cast<float*>(lds) + wave_id * 256;
#pragma unroll
  for (int j = 0; j < kTokTiles; ++j) {
    if (j >= live_tok_tiles) break;
#pragma unroll
    for (int l = 0; l < 8; ++l) tile_scratch[sub_lane * 16 + 2 * l + half_id] = acc[j][l];
    __builtin_amdgcn_wave_barrier();
    if constexpr (kF16Pairs) {
      // f16 pairs (v2 i4r): one __half2 store per lane — 128 stores per tile in
      // 4 iterations. r0 is a multiple of 16 and m is even, so r is even and
      // every 4 B store is aligned; each 8-lane group covers a 32 B row segment.
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int idx = s * 32 + lane_id;  // (row, half2 pair) in the tile
        const int t = j * 16 + (idx >> 3);
        const int r = r0 + (idx & 7) * 2;
        if (t < nrows && r < m)
          *reinterpret_cast<__half2*>(reinterpret_cast<__half*>(out) +
                                      (size_t)(slot0 + t) * m + r) =
              __floats2half2_rn(tile_scratch[(idx >> 3) * 16 + (idx & 7) * 2],
                                tile_scratch[(idx >> 3) * 16 + (idx & 7) * 2 + 1]);
      }
    } else {
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = s * 32 + lane_id;
        const int t = j * 16 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < nrows && r < m) out[(size_t)(slot0 + t) * m + r] = tile_scratch[flat];
      }
    }
    __builtin_amdgcn_wave_barrier();
  }
}

namespace moelut {
static bool aligned(const void* ptr, uintptr_t a) { return ((uintptr_t)ptr & (a - 1)) == 0; }
static bool lut_ok(int type, const LutW& p, bool pair, int k) {
  if (type == kI4R)
    return k % 128 == 0 && p.cb && aligned(p.w, 16) && (!pair || aligned(p.w_up, 16)) &&
           aligned(p.sc, 2) && (!pair || aligned(p.sc_up, 2)) && p.sstride == k / 128 * 2;
  if (k % ((type == kIQ4XS || type == kIQ3S || type == kIQ3XXS || type == kIQ2S ||
            type == kIQ2XXS || type == kIQ2XS) ? 256 : 64))
    return false;
  const uintptr_t a = type == kQ4CP ? 16 : type == kIQ4NL ? 4
                      : (type == kIQ3S || type == kQ2_0 || type == kIQ3XXS || type == kIQ2S ||
                         type == kIQ2XXS || type == kIQ2XS) ? 2
                      : 8;
  if (!aligned(p.w, a) || (pair && !aligned(p.w_up, a))) return false;
  if (type == kQ4CP && (!p.cb || !aligned(p.sc, 16) || (pair && !aligned(p.sc_up, 16)) || p.sstride % 16 ||
                        p.sstride < (k / 256 + (k % 256 != 0)) * 16))
    return false;
  return type == kQ4CP || type == kIQ4NL || type == kIQ4XS || type == kIQ3S || type == kQ2_0 ||
         type == kIQ2XXS || type == kIQ2XS || type == kIQ3XXS || type == kIQ2S;
}
}  // namespace moelut

// gate/up: hid[slot][m] F16 = up * silu(gate), x = F16 [P][k] via tokidx.
// type: moelut::kQ4CP / kIQ4NL / kIQ4XS. Returns false when unsupported.
static bool moe_lut_up(int type, const moelut::LutW& p, const __half* x, const int* tokidx,
                       const MoeTile* tiles, const int* ntiles, int max_tiles, __half* hid, int m,
                       int k, hipStream_t st) {
  if (m % 64 || !moelut::lut_ok(type, p, true, k)) return false;
  dim3 grid(m / 64, max_tiles);
  if (type == moelut::kQ4CP)
    k_moe_lut<moelut::kQ4CP, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kIQ4NL)
    k_moe_lut<moelut::kIQ4NL, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kIQ3S)
    k_moe_lut<moelut::kIQ3S, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kQ2_0)
    k_moe_lut<moelut::kQ2_0, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kIQ2XXS)
    k_moe_lut<moelut::kIQ2XXS, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kIQ2XS)
    k_moe_lut<moelut::kIQ2XS, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kIQ3XXS)
    k_moe_lut<moelut::kIQ3XXS, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else if (type == moelut::kIQ2S)
    k_moe_lut<moelut::kIQ2S, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  else
    k_moe_lut<moelut::kIQ4XS, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  return true;
}

// 128-slot merged tiles (mt) for the generic IQ/GGUF LUT types: two 64-slot
// tiles of the same expert are decoded once per K stage (bit-identical to the
// BN=64 path, each (slot, row) keeps its K order and WMMA sequence).
static bool moe_lut_up_mt_lut(int type, const moelut::LutW& p, const __half* x,
                              const int* tokidx, const MoeTile* tiles, const int* ntiles,
                              int max_tiles, __half* hid, int m, int k, hipStream_t st) {
  if (m % 64 || !moelut::lut_ok(type, p, true, k)) return false;
  dim3 grid(m / 64, max_tiles);
  if (type == moelut::kQ4CP)
    k_moe_lut<moelut::kQ4CP, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                              hid, m, k);
  else if (type == moelut::kIQ4NL)
    k_moe_lut<moelut::kIQ4NL, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                               hid, m, k);
  else if (type == moelut::kIQ3S)
    k_moe_lut<moelut::kIQ3S, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                              hid, m, k);
  else if (type == moelut::kQ2_0)
    k_moe_lut<moelut::kQ2_0, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                              hid, m, k);
  else if (type == moelut::kIQ2XXS)
    k_moe_lut<moelut::kIQ2XXS, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                                hid, m, k);
  else if (type == moelut::kIQ2XS)
    k_moe_lut<moelut::kIQ2XS, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                               hid, m, k);
  else if (type == moelut::kIQ3XXS)
    k_moe_lut<moelut::kIQ3XXS, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                                hid, m, k);
  else if (type == moelut::kIQ2S)
    k_moe_lut<moelut::kIQ2S, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                              hid, m, k);
  else
    k_moe_lut<moelut::kIQ4XS, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                               hid, m, k);
  return true;
}

// hgn v2 gate/up without the fused silu: guv[slot][2m] F16 (gate | up), still
// in the rotated domain (k_i4r_mid finishes it). F16 halves the round trip
// through memory (k_i4r_mid is bandwidth-bound at P = 16K).
static bool moe_lut_up_raw(const moelut::LutW& p, const __half* x, const int* tokidx,
                           const MoeTile* tiles, const int* ntiles, int max_tiles, __half* guv,
                           int m, int k, hipStream_t st) {
  if (m % 64 || !moelut::lut_ok(moelut::kI4R, p, true, k)) return false;
  dim3 grid(m / 64, max_tiles);
  k_moe_lut<moelut::kI4R, true, 64, true><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                                guv, m, k);
  return true;
}

// hgn v2 gate/up + i4r mid fused (HANDOFF-V2-PREFILL direction 3): one block
// owns 128 gate + 128 up rows = exactly one FWHT group, runs the K pipeline
// twice (gate pass, up pass), spills each to an f16 [slot][row] LDS plane with
// the same rounding as the guv16 write, then applies k_i4r_mid's math per slot
// in registers and writes hid[slot][m] F16 directly. Bit-identical to
// moe_lut_up_raw -> k_i4r_mid, without the guv16 round trip.
static bool moe_lut_up_mid(const moelut::LutW& p, const __half* x, const int* tokidx,
                           const MoeTile* tiles, const int* ntiles, int max_tiles, __half* hid,
                           const float* svh_gu, const float* suh_dn, int m, int k, hipStream_t st) {
  if (m % 128 || !moelut::lut_ok(moelut::kI4R, p, true, k)) return false;
  dim3 grid(m / 128, max_tiles);
  k_moe_lut<moelut::kI4R, false, 64, false, false, true><<<grid, 256, 0, st>>>(
      p, x, tokidx, tiles, ntiles, nullptr, hid, m, k, svh_gu, suh_dn);
  return true;
}

// §11.5 multi-tile (mt) variants: the same kernels run on super-tiles of up to
// 128 slots (two of the baseline's 64-slot tiles of one expert, merged by
// k_moe_tiles with cap 128). Each weight K stage is fetched and decoded once
// for both tile halves. The per-(slot, row) K order and wmma sequence are the
// baseline's, so outputs are bit-identical; the kMid up spill + fused mid run
// in two 64-slot chunks through one up plane (LDS stays <= 54K).
static bool moe_lut_up_mid_mt(const moelut::LutW& p, const __half* x, const int* tokidx,
                              const MoeTile* tiles, const int* ntiles, int max_tiles, __half* hid,
                              const float* svh_gu, const float* suh_dn, int m, int k,
                              hipStream_t st) {
  if (m % 128 || !moelut::lut_ok(moelut::kI4R, p, true, k)) return false;
  dim3 grid(m / 128, max_tiles);
  k_moe_lut<moelut::kI4R, false, 128, false, false, true><<<grid, 256, 0, st>>>(
      p, x, tokidx, tiles, ntiles, nullptr, hid, m, k, svh_gu, suh_dn);
  return true;
}

static bool moe_lut_down_mt(const moelut::LutW& p, const __half* hid, const MoeTile* tiles,
                            const int* ntiles, int max_tiles, float* pairs, int m, int k,
                            hipStream_t st, bool f16) {
  if (m % 128 || !moelut::lut_ok(moelut::kI4R, p, false, k)) return false;
  dim3 grid(m / 128, max_tiles);
  if (f16)
    k_moe_lut<moelut::kI4R, false, 128, false, true><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles,
                                                                          ntiles, pairs, nullptr, m, k);
  else
    k_moe_lut<moelut::kI4R, false, 128><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs,
                                                             nullptr, m, k);
  return true;
}

// w4b q4cp 站的 mt 变体：同一 k_moe_tiles cap=128 合并契约（SHARE=2 固定；
// 3/4 的 kActFetch 6/8 VGPR 溢出是死路）。q4cp 无 kMid，up 保持 kPair
// epilogue（silu 融合不变），down 直接移。逐 bit 论证与 i4r 相同：每
// (slot, row) 的 K 顺序与 wmma 序列与 BN=64 逐字一致。
static bool moe_lut_up_mt_q4cp(const moelut::LutW& p, const __half* x, const int* tokidx,
                               const MoeTile* tiles, const int* ntiles, int max_tiles, __half* hid,
                               int m, int k, hipStream_t st) {
  if (m % 64 || !moelut::lut_ok(moelut::kQ4CP, p, true, k)) return false;
  dim3 grid(m / 64, max_tiles);
  k_moe_lut<moelut::kQ4CP, true, 128><<<grid, 256, 0, st>>>(p, x, tokidx, tiles, ntiles, nullptr,
                                                            hid, m, k);
  return true;
}

static bool moe_lut_down_mt_q4cp(const moelut::LutW& p, const __half* hid, const MoeTile* tiles,
                                 const int* ntiles, int max_tiles, float* pairs, int m, int k,
                                 hipStream_t st, bool f16) {
  if (m % 128 || !moelut::lut_ok(moelut::kQ4CP, p, false, k)) return false;
  dim3 grid(m / 128, max_tiles);
  if (f16)
    k_moe_lut<moelut::kQ4CP, false, 128, false, true><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles,
                                                                            ntiles, pairs, nullptr, m, k);
  else
    k_moe_lut<moelut::kQ4CP, false, 128><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs,
                                                               nullptr, m, k);
  return true;
}

// P3: packed down epilogue (kPk) for the GGUF LUT types. The fetch reorder is
// type-independent and the epilogue keeps the same values (measured rel-L2
// unchanged); gains -5..-7% on IQ4_NL/Q2_0 down. QWENOX_GG_KPK=0 disables.
static bool gg_kpk_env() {
  static const bool v = [] {
    const char* s = getenv("QWENOX_GG_KPK");
    return !(s && s[0] == '0' && s[1] == 0);
  }();
  return v;
}

// down: pairs[slot][m] F32 (unweighted), hid = F16 [slot][k]. f16 = true: pairs
// written as F16 (half the traffic; read back by k_v2_reduce_rot<..., __half> on
// v2 i4r / k_moe_reduce_pw*_sg<..., __half> on w4b q4cp — B1).
static bool moe_lut_down(int type, const moelut::LutW& p, const __half* hid, const MoeTile* tiles,
                         const int* ntiles, int max_tiles, float* pairs, int m, int k, hipStream_t st,
                         bool f16 = false) {
  if (type == moelut::kIQ3S || type == moelut::kIQ3XXS || type == moelut::kIQ2S ||
      type == moelut::kIQ2XXS || type == moelut::kIQ2XS)
    return false;  // no down instance (this model: down is IQ4_NL / Q2_0)
  if (m % 128 || !moelut::lut_ok(type, p, false, k)) return false;
  dim3 grid(m / 128, max_tiles);
  const bool kpk = gg_kpk_env();
  if (type == moelut::kQ4CP && f16)
    k_moe_lut<moelut::kQ4CP, false, 64, false, true><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles,
                                                                           ntiles, pairs, nullptr, m, k);
  else if (type == moelut::kQ4CP)
    k_moe_lut<moelut::kQ4CP, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else if (type == moelut::kI4R && f16)
    k_moe_lut<moelut::kI4R, false, 64, false, true><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs,
                                                                         nullptr, m, k);
  else if (type == moelut::kI4R)
    k_moe_lut<moelut::kI4R, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else if (type == moelut::kIQ4NL) {
    if (f16) {
      if (kpk)
        k_moe_lut<moelut::kIQ4NL, false, 64, false, true, false, false, true>
            <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
      else
        k_moe_lut<moelut::kIQ4NL, false, 64, false, true><<<grid, 256, 0, st>>>(
            p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    } else if (kpk)
      k_moe_lut<moelut::kIQ4NL, false, 64, false, false, false, false, true>
          <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    else
      k_moe_lut<moelut::kIQ4NL, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  }
  else if (type == moelut::kQ2_0) {
    if (f16) {
      if (kpk)
        k_moe_lut<moelut::kQ2_0, false, 64, false, true, false, false, true>
            <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
      else
        k_moe_lut<moelut::kQ2_0, false, 64, false, true><<<grid, 256, 0, st>>>(
            p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    } else if (kpk)
      k_moe_lut<moelut::kQ2_0, false, 64, false, false, false, false, true>
          <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    else
      k_moe_lut<moelut::kQ2_0, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  }
  else if (f16) {
    if (kpk)
      k_moe_lut<moelut::kIQ4XS, false, 64, false, true, false, false, true>
          <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    else
      k_moe_lut<moelut::kIQ4XS, false, 64, false, true><<<grid, 256, 0, st>>>(
          p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  } else if (kpk)
    k_moe_lut<moelut::kIQ4XS, false, 64, false, false, false, false, true>
        <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else
    k_moe_lut<moelut::kIQ4XS, false><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  return true;
}

// 128-slot merged tiles for down (same contract as moe_lut_up_mt_lut).
static bool moe_lut_down_mt_lut(int type, const moelut::LutW& p, const __half* hid,
                                const MoeTile* tiles, const int* ntiles, int max_tiles,
                                float* pairs, int m, int k, hipStream_t st, bool f16 = false) {
  if (type == moelut::kIQ3S || type == moelut::kIQ3XXS || type == moelut::kIQ2S ||
      type == moelut::kIQ2XXS || type == moelut::kIQ2XS)
    return false;
  if (m % 128 || !moelut::lut_ok(type, p, false, k)) return false;
  dim3 grid(m / 128, max_tiles);
  const bool kpk = gg_kpk_env();
  if (type == moelut::kQ4CP)
    if (f16)
      k_moe_lut<moelut::kQ4CP, false, 128, false, true><<<grid, 256, 0, st>>>(
          p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    else
      k_moe_lut<moelut::kQ4CP, false, 128><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles,
                                                                 pairs, nullptr, m, k);
  else if (type == moelut::kIQ4NL)
    if (f16) {
      if (kpk)
        k_moe_lut<moelut::kIQ4NL, false, 128, false, true, false, false, true>
            <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
      else
        k_moe_lut<moelut::kIQ4NL, false, 128, false, true><<<grid, 256, 0, st>>>(
            p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    } else if (kpk)
      k_moe_lut<moelut::kIQ4NL, false, 128, false, false, false, false, true>
          <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    else
      k_moe_lut<moelut::kIQ4NL, false, 128><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles,
                                                                  pairs, nullptr, m, k);
  else if (type == moelut::kQ2_0)
    if (f16) {
      if (kpk)
        k_moe_lut<moelut::kQ2_0, false, 128, false, true, false, false, true>
            <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
      else
        k_moe_lut<moelut::kQ2_0, false, 128, false, true><<<grid, 256, 0, st>>>(
            p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    } else if (kpk)
      k_moe_lut<moelut::kQ2_0, false, 128, false, false, false, false, true>
          <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    else
      k_moe_lut<moelut::kQ2_0, false, 128><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles,
                                                                 pairs, nullptr, m, k);
  else if (f16) {
    if (kpk)
      k_moe_lut<moelut::kIQ4XS, false, 128, false, true, false, false, true>
          <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
    else
      k_moe_lut<moelut::kIQ4XS, false, 128, false, true><<<grid, 256, 0, st>>>(
          p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  } else if (kpk)
    k_moe_lut<moelut::kIQ4XS, false, 128, false, false, false, false, true>
        <<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  else
    k_moe_lut<moelut::kIQ4XS, false, 128><<<grid, 256, 0, st>>>(p, hid, nullptr, tiles, ntiles,
                                                                pairs, nullptr, m, k);
  return true;
}

// q4cp register-perm codebook variants (A1): bit-identical decode via v_perm
// byte tables in registers instead of the s_cbp LDS lookups.
[[maybe_unused]] static bool moe_lut_up_perm(const moelut::LutW& p, const __half* x, const int* tokidx,
                                             const MoeTile* tiles, const int* ntiles, int max_tiles,
                                             __half* hid, int m, int k, hipStream_t st) {
  if (m % 64 || !moelut::lut_ok(moelut::kQ4CP, p, true, k)) return false;
  dim3 grid(m / 64, max_tiles);
  k_moe_lut<moelut::kQ4CP, true, 64, false, false, false, true><<<grid, 256, 0, st>>>(
      p, x, tokidx, tiles, ntiles, nullptr, hid, m, k);
  return true;
}

[[maybe_unused]] static bool moe_lut_down_perm(const moelut::LutW& p, const __half* hid,
                                               const MoeTile* tiles, const int* ntiles, int max_tiles,
                                               float* pairs, int m, int k, hipStream_t st) {
  if (m % 128 || !moelut::lut_ok(moelut::kQ4CP, p, false, k)) return false;
  dim3 grid(m / 128, max_tiles);
  k_moe_lut<moelut::kQ4CP, false, 64, false, false, false, true><<<grid, 256, 0, st>>>(
      p, hid, nullptr, tiles, ntiles, pairs, nullptr, m, k);
  return true;
}

// q4cp views of the hgn expert tensors (dtype 5; data = device pointer to
// [cb 64 B][codes][scales]). gate_up: [E][2*mid][d] fused, down: [E][d][mid].
static moelut::LutW lut_q4cp_view(const uint8_t* data, uint64_t rows, uint64_t cols,
                                  uint64_t scale_stride, int e_rows, int up_off) {
  moelut::LutW p{};
  p.cb = reinterpret_cast<const float*>(data);
  p.w = p.w_up = data + 64;
  p.sc = p.sc_up = data + 64 + rows * cols / 2;
  p.sstride = (int)scale_stride;
  p.e_rows = e_rows;
  p.up_off = up_off;
  return p;
}

// i4r view (dtype 23; data = device pointer to [codes][scales], no header).
static moelut::LutW lut_i4r_view(const uint8_t* data, uint64_t rows, uint64_t cols,
                                 const float* cb, int e_rows, int up_off) {
  moelut::LutW p{};
  p.cb = cb;
  p.w = p.w_up = data;
  p.sc = p.sc_up = data + rows * cols / 2;
  p.sstride = (int)(cols / 128 * 2);
  p.e_rows = e_rows;
  p.up_off = up_off;
  return p;
}

// Single-"expert" 64-slot tiles for the dense-LUT dispatch (A3/§11.3):
// tiles[i] = {0, i*64, min(64, P-i*64)}, *ntiles = ceil(P/64). cap 64 always —
// the MoE-mt BN=128 variant (moe_lut_down_mt) is the v2 MoE station's, not
// this path's.
__global__ void k_dense_tiles(MoeTile* __restrict__ tiles, int* __restrict__ ntiles, int P) {
  const int i = threadIdx.x;
  const int n = (P + 63) / 64;
  if (i == 0) *ntiles = n;
  if (i < n) tiles[i] = {0, i * 64, min(64, P - i * 64)};
}

// --- GGUF routed experts, any supported block type (WMMA, P > gemv cap) -----
// K-quant / Q5_1 / Q8_0 -> k_moe_gg (26); IQ4_NL / IQ4_XS / IQ3_S / Q2_0 -> k_moe_lut
// on the raw expert-stacked GGUF tensors ([E][m][row bytes], gate and up are
// separate tensors: up_off = 0). Same contracts as moe_gg_up / moe_gg_down.
static bool ggx_is_lut(int type) {
  return type == moelut::kIQ4NL || type == moelut::kIQ4XS || type == moelut::kIQ3S ||
         type == moelut::kQ2_0 || type == moelut::kIQ2XXS || type == moelut::kIQ2XS ||
         type == moelut::kIQ3XXS || type == moelut::kIQ2S;
}
static bool moe_ggx_up(int type, const uint8_t* gate, const uint8_t* up, const __half* x,
                       const int* tokidx, const MoeTile* tiles, const int* ntiles, int max_tiles,
                       __half* hid, int m, int k, hipStream_t st, bool mt = false) {
  if (!ggx_is_lut(type)) {
    if (mt) return false;  // k_moe_gg is 64-slot only
    return moe_gg_up(type, gate, up, x, tokidx, tiles, ntiles, max_tiles, hid, m, k, st);
  }
  moelut::LutW p{};
  p.w = gate;
  p.w_up = up;
  p.e_rows = m;
  p.up_off = 0;
  return mt ? moe_lut_up_mt_lut(type, p, x, tokidx, tiles, ntiles, max_tiles, hid, m, k, st)
            : moe_lut_up(type, p, x, tokidx, tiles, ntiles, max_tiles, hid, m, k, st);
}
static bool moe_ggx_down(int type, const uint8_t* dn, const __half* hid, const MoeTile* tiles,
                         const int* ntiles, int max_tiles, float* pairs, int m, int k,
                         hipStream_t st, bool mt = false, bool f16 = false) {
  if (!ggx_is_lut(type)) {
    if (mt || f16) return false;  // k_moe_gg is 64-slot / f32 only
    return moe_gg_down(type, dn, hid, tiles, ntiles, max_tiles, pairs, m, k, st);
  }
  moelut::LutW p{};
  p.w = p.w_up = dn;
  p.e_rows = m;
  return mt ? moe_lut_down_mt_lut(type, p, hid, tiles, ntiles, max_tiles, pairs, m, k, st, f16)
            : moe_lut_down(type, p, hid, tiles, ntiles, max_tiles, pairs, m, k, st, f16);
}

// --- W4A4 routed experts (QWENOX_MOE_A4, opt-in, default off) ---------------
// 可选 prefill 提速：64K +8.9%，prompt KLD +0.015~0.035，decode 不受影响；
// 取舍与并发说明见 40_model.inc moe_b_v2。
// hipfire-style IU4 route for hgn v2 i4r experts (10-14; prototype and A/B in
// tools/moe_iu4_proto.cu, notes dev-docs/HIPFIRE-PREFILL-GAP-GPUONLY.md §8).
// Activations are quantized to int4 per 128 elements (k_quant_a4) and the
// expert GEMMs run v_wmma_i32_16x16x16_iu4 (2x the f16 WMMA rate on gfx1151;
// i4r nibbles need only ^0x8, no decode). Trades KLD for MoE speed.
namespace moeiu4 {
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
}  // namespace moeiu4

// ---------------------------------------------------------------------------
// A4 activation quantization: one wave per (row, 128-element block).
__global__ void __launch_bounds__(256)
    k_quant_a4(const __half* __restrict__ x, int N, int K, uint8_t* __restrict__ q,
               float* __restrict__ s) {
  const int lane = threadIdx.x & 31;
  const int nb = K / 128;
  const long wid = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (wid >= (long)N * nb) return;
  const long row = wid / nb;
  const int blk = (int)(wid % nb);
  const uint2 raw = *reinterpret_cast<const uint2*>(x + row * K + blk * 128 + lane * 4);
  const __half2 h01 = __builtin_bit_cast(__half2, raw.x), h23 = __builtin_bit_cast(__half2, raw.y);
  float v[4] = {__low2float(h01), __high2float(h01), __low2float(h23), __high2float(h23)};
  float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor(amax, o, 32));
  float best_s = 0.f, best_e = INFINITY;
  if (amax > 0.f) {
    const float cand[5] = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f};
#pragma unroll
    for (int c = 0; c < 5; ++c) {
      const float sc = cand[c] * amax / 7.f, inv = 1.f / sc;
      float e = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float qi = fminf(fmaxf(rintf(v[i] * inv), -8.f), 7.f);
        const float d = v[i] - qi * sc;
        e += d * d;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) e += __shfl_xor(e, o, 32);
      if (e < best_e) {  // wave-uniform
        best_e = e;
        best_s = sc;
      }
    }
  }
  uint32_t packed = 0;
  if (best_s > 0.f) {
    const float inv = 1.f / best_s;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int qi = (int)fminf(fmaxf(rintf(v[i] * inv), -8.f), 7.f);
      packed |= (uint32_t)(qi & 15) << (4 * i);
    }
  }
  reinterpret_cast<uint16_t*>(q + row * (K / 2) + blk * 64)[lane] = (uint16_t)packed;
  if (lane == 0) s[row * nb + blk] = best_s;
}

// ---------------------------------------------------------------------------
// W4A4 routed-expert GEMM. Weight rows: i4r code plane (row stride k/2 bytes,
// byte b = elements 2b | 2b+1 << 4, value nib-8), scale plane fp16 per 128.
struct Iu4W {
  const uint8_t* codes;
  const uint8_t* sc;
  int e_rows, up_off;
};

template <bool kUp, bool kGguf = false>
__launch_bounds__(256) __global__
    void k_moe_iu4(Iu4W p, const uint8_t* __restrict__ x4, const float* __restrict__ xs,
                   const int* __restrict__ tokidx, const MoeTile* __restrict__ tiles,
                   const int* __restrict__ ntiles, __half* __restrict__ out, int m, int k,
                   const float* __restrict__ svh_gu, const float* __restrict__ suh_dn) {
  constexpr int BM = 128, BN = 64, T = BN / 16;
  constexpr int kActStrideB = 80;  // 64 B codes + 16 B pad: 8-lane phases conflict-free
  constexpr int kCodeBytes = BM * 64, kWscBytes = BM * 2, kActBytes = BN * kActStrideB,
                kSaBytes = BN * 4;
  constexpr int kStage = kCodeBytes + kWscBytes + kActBytes + kSaBytes;
  constexpr int kMidRow = BM + 8;
  constexpr int kPlaneB = BN * kMidRow * 2;
  constexpr int kStageOff = kUp ? kPlaneB : 0;
  constexpr int kLds = kUp ? kStageOff + (kStage > kPlaneB ? kStage : kPlaneB) : kStage;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kLds];
  uint4* s_codes = reinterpret_cast<uint4*>(lds + kStageOff);
  __half* s_wsc = reinterpret_cast<__half*>(lds + kStageOff + kCodeBytes);
  uint8_t* s_act = lds + kStageOff + kCodeBytes + kWscBytes;
  float* s_sa = reinterpret_cast<float*>(lds + kStageOff + kCodeBytes + kWscBytes + kActBytes);

  if ((int)blockIdx.y >= *ntiles) return;
  const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31, sub = lane & 15, half = lane >> 4;
  const MoeTile tile = tiles[blockIdx.y];
  const int slot0 = tile.first, nrows = tile.count, live = (nrows + 15) / 16;
  const int r_block = (int)blockIdx.x * BM;
  const int nst = k / 128, nb = k / 128;
  const size_t wrb = (size_t)k / 2, wsb = (size_t)nb * 2;

  // weight fetch: thread -> LDS slot fs (2 threads/slot, 32 B each); slot
  // 2l+h of a 16-row wave block holds local row 8h+l (packed C epilogue).
  const int fs = tid >> 1, fpart = tid & 1;
  const int flr = (fs & ~15) | ((fs & 1) << 3) | ((fs & 15) >> 1);
  const uint8_t* fw;
  const uint8_t* fsc;
  const auto set_fetch = [&](int pass) {
    const size_t grow = (size_t)tile.expert * p.e_rows + (pass ? p.up_off : 0) + r_block + flr;
    fw = p.codes + grow * wrb + fpart * 32;
    fsc = p.sc + grow * wsb;
  };
  // activation fetch: thread -> (token at, 16 B chunk ac)
  const int at = tid >> 2, ac = tid & 3;
  const uint8_t* fa = nullptr;
  const float* fas = nullptr;
  if (at < nrows) {
    const int src = tokidx ? tokidx[slot0 + at] : slot0 + at;
    fa = x4 + (size_t)src * wrb + ac * 16;
    fas = xs + (size_t)src * nb;
  }
  uint4 rw0, rw1, ra;
  uint16_t rws = 0;
  float ras = 0.f;
  const auto fetch = [&](int st) {
    rw0 = reinterpret_cast<const uint4*>(fw + st * 64)[0];
    rw1 = reinterpret_cast<const uint4*>(fw + st * 64)[1];
    if (fpart == 0) rws = reinterpret_cast<const uint16_t*>(fsc)[st];
    ra = fa ? *reinterpret_cast<const uint4*>(fa + st * 64) : make_uint4(0u, 0u, 0u, 0u);
    if (ac == 0) ras = fa ? fas[st] : 0.f;
  };
  const auto swz = [](int slot, int c) { return slot * 4 + (c ^ ((slot >> 1) & 3)); };
  const auto commit = [&]() {
    constexpr uint32_t kFlip = 0x88888888u;  // offset binary -> two's complement
    s_codes[swz(fs, fpart * 2)] = make_uint4(rw0.x ^ kFlip, rw0.y ^ kFlip, rw0.z ^ kFlip, rw0.w ^ kFlip);
    s_codes[swz(fs, fpart * 2 + 1)] = make_uint4(rw1.x ^ kFlip, rw1.y ^ kFlip, rw1.z ^ kFlip, rw1.w ^ kFlip);
    if (fpart == 0) s_wsc[flr] = __builtin_bit_cast(__half, rws);
    *reinterpret_cast<uint4*>(s_act + at * kActStrideB + ac * 16) = ra;
    if (ac == 0) s_sa[at] = ras;
  };

  moegg::v8f acc[T];
  const int slot = wave * 16 + sub;
  const auto compute = [&]() {
    uint4 a[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) a[c] = s_codes[swz(slot, c)];
    const uint4 wsv = *reinterpret_cast<const uint4*>(s_wsc + wave * 16 + half * 8);
    const uint32_t wsw[4] = {wsv.x, wsv.y, wsv.z, wsv.w};
    float ws[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const __half2 h = __builtin_bit_cast(__half2, wsw[q]);
      ws[2 * q] = __low2float(h);
      ws[2 * q + 1] = __high2float(h);
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
      if (j >= live) continue;  // block-uniform
      const int t = j * 16 + sub;
      const uint8_t* ap = s_act + t * kActStrideB;
      moeiu4::v8i ci = moeiu4::v8i{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint4 b = *reinterpret_cast<const uint4*>(ap + c * 16);
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, moeiu4::v2i{(int)a[c].x, (int)a[c].y}, true,
                                                         moeiu4::v2i{(int)b.x, (int)b.y}, ci, false);
        ci = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, moeiu4::v2i{(int)a[c].z, (int)a[c].w}, true,
                                                         moeiu4::v2i{(int)b.z, (int)b.w}, ci, false);
      }
      const float sa = s_sa[t];
#pragma unroll
      for (int l = 0; l < 8; ++l) acc[j][l] = fmaf((float)ci[l], ws[l] * sa, acc[j][l]);
    }
  };

  const auto spill = [&](__half* plane) {
#pragma unroll
    for (int j = 0; j < T; ++j) {
      if (j >= live) break;
      const int t = j * 16 + sub;
      if (t >= nrows) continue;
      __half2 h2[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        h2[q] = __halves2half2(__float2half(acc[j][2 * q]), __float2half(acc[j][2 * q + 1]));
      uint4 v;
      __builtin_memcpy(&v, h2, 16);
      *reinterpret_cast<uint4*>(plane + t * kMidRow + wave * 16 + half * 8) = v;
    }
  };

  for (int pass = 0; pass < (kUp ? 2 : 1); ++pass) {
    set_fetch(pass);
#pragma unroll
    for (int j = 0; j < T; ++j) acc[j] = moegg::v8f{0, 0, 0, 0, 0, 0, 0, 0};
    fetch(0);
    for (int st = 0; st < nst; ++st) {
      commit();
      __syncthreads();
      if (st + 1 < nst) fetch(st + 1);
      compute();
      __syncthreads();
    }
    if constexpr (kUp) spill(reinterpret_cast<__half*>(lds) + (pass ? kPlaneB / 2 : 0));
  }

  if constexpr (kUp) {
    __syncthreads();
    const __half* gplane = reinterpret_cast<const __half*>(lds);
    const __half* uplane = gplane + kPlaneB / 2;
    for (int s = wave; s < nrows; s += 8) {  // == k_moe_lut kMid mid_slot
      const __half* gr = gplane + s * kMidRow;
      const __half* ur = uplane + s * kMidRow;
      float g[4], u[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        g[j] = __half2float(gr[j * 32 + lane]);
        u[j] = __half2float(ur[j * 32 + lane]);
      }
      float v[4];
      if constexpr (kGguf) {  // GGUF: weights are already in the plain basis
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = g[j] / (1.f + expf(-g[j])) * u[j];
      } else {
        hv2::fwht128(g[0], g[1], g[2], g[3], lane);
        hv2::fwht128(u[0], u[1], u[2], u[3], lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int i = r_block + j * 32 + lane;
          const float gg = g[j] * hv2::kRs128 * svh_gu[i];
          const float uu = u[j] * hv2::kRs128 * svh_gu[m + i];
          v[j] = gg / (1.f + expf(-gg)) * uu * suh_dn[i];
        }
        hv2::fwht128(v[0], v[1], v[2], v[3], lane);
      }
      __half* o = out + (size_t)(slot0 + s) * m + r_block;
#pragma unroll
      for (int j = 0; j < 4; ++j) o[j * 32 + lane] = __float2half(v[j] * (kGguf ? 1.f : hv2::kRs128));
    }
  } else {
    const int rr = r_block + wave * 16 + half * 8;
#pragma unroll
    for (int j = 0; j < T; ++j) {
      if (j >= live) break;
      const int t = j * 16 + sub;
      if (t >= nrows) continue;
      __half2 h2[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) h2[q] = __floats2half2_rn(acc[j][2 * q], acc[j][2 * q + 1]);
      uint4 v;
      __builtin_memcpy(&v, h2, 16);
      *reinterpret_cast<uint4*>(out + (size_t)(slot0 + t) * m + rr) = v;
    }
  }
}
