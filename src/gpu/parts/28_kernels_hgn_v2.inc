// --- hgn v2 (halogen 0.15 "qwen38-flash-next-v2") native kernels ------------
//
// dtype 16 "ht"   dense 4-bit, hashed-trellis codes in the H128 rotated domain
// dtype 23 "i4r"  routed experts, (nib - 8) * f16 scale per 128 cols, rotated
// dtype 24 "q6g64" HC, 6-bit code * scale + bias per 64 cols, not rotated
// dtype 2  f16 sidecars (.suh / .svh / experts .su), uploaded as f32 at load
//
// Rotated weights: W = diag(svh) . H . G . H . diag(suh) / 128 with H the
// 128-block Sylvester Hadamard (natural order). Instead of rebuilding W the
// rotation moves onto the activations:
//   rot_in   x^ = H(suh . x) / sqrt(128)        (per 128 block of K)
//   GEMV     z  = G . x^
//   rot_out  y  = svh . H(z) / sqrt(128)       (per 128 block of O)
// One wave32 owns one 128-vector: element i = j * 32 + lane (j = 0..3).
// Format notes: HGN-V2.md (reverse-engineered, numpy-verified).

namespace hv2 {

// ht code -> value: 16-bit trellis window, hash alphabet (no stored codebook).
__device__ __forceinline__ float ht_val(uint32_t w16) {
  const uint32_t t = w16 * 0x83dcd12du;
  const uint32_t s = __builtin_amdgcn_sad_u8(t, 0u, 0u);  // byte sum
  const float h = (float)(1024u + s);                      // f16(s + 0x6400) exactly
  return __half2float(__float2half(fmaf(h, 1.732421875f / 256.f, -10.3828125f)));
}
// Two codes at once, bit-identical to ht_val: sad's accumulator adds the
// 0x6400 exponent, h * c + d is exact in f32 (22 significant bits), so one
// packed f16 fma rounds the same way as the f32 fma + f16 cast.
__device__ __forceinline__ void ht_val2(uint32_t a0, uint32_t a1, float& v0, float& v1) {
  const uint32_t s0 = __builtin_amdgcn_sad_u8(a0 * 0x83dcd12du, 0u, 0x6400u);
  const uint32_t s1 = __builtin_amdgcn_sad_u8(a1 * 0x83dcd12du, 0u, 0x6400u);
  const uint32_t hp = s0 | (s1 << 16);
  const __half2 r = __hfma2(*(const __half2*)&hp,
                            __half2half2(__float2half(1.732421875f / 256.f)),
                            __half2half2(__float2half(-10.3828125f)));
  v0 = __low2float(r);
  v1 = __high2float(r);
}

__device__ __forceinline__ float ldx(const float* x, size_t i) { return x[i]; }
__device__ __forceinline__ float ldx(const uint16_t* x, size_t i) {
  return __uint_as_float((uint32_t)x[i] << 16);
}
// pairs element loads: f32 (q4cp path) or f16 (v2 f16-pairs path)
__device__ __forceinline__ float ldp(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float ldp(const __half* p, size_t i) { return __half2float(p[i]); }
// 8 consecutive activations (16 B aligned for f32 pairs / bf16 octets).
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

// y[p*ys + i] = post[i] * H(pre . x[p*xs + ...]) / sqrt128 per 128 block;
// pre / post optional (nullptr = 1). P rows x n (n % 128 == 0). One wave per
// (row, block); in place is fine (x == y with equal strides).
template <typename TX, bool OUTH>
__global__ __launch_bounds__(256) void k_v2_rot(const TX* __restrict__ x, uint64_t xs,
                                                void* y, uint64_t ys,
                                                const float* __restrict__ pre,
                                                const float* __restrict__ post, int n,
                                                int P) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int nb = n >> 7;
  if (gw >= P * nb) return;
  const int p = gw / nb, b = gw - p * nb;
  const TX* xr = x + (size_t)p * xs + b * 128;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = j * 32 + lane;
    v[j] = hv2::ldx(xr, i) * (pre ? pre[b * 128 + i] : 1.f);
  }
  hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = b * 128 + j * 32 + lane;
    const float o = v[j] * hv2::kRs128 * (post ? post[i] : 1.f);
    if constexpr (OUTH)
      ((__half*)y)[(size_t)p * ys + i] = __float2half(o);
    else
      ((float*)y)[(size_t)p * ys + i] = o;
  }
}

// i4r expert middle, one block per slot, 2*mid/128 waves:
//   g|u = svh_gu . H(z) / sqrt128   (z = rotated gate_up output, 2*mid wide,
//                                    f32 from the GEMVs or f16 from the LUT)
//   h   = silu(g) * u
//   out = H(suh_dn . h) / sqrt128   (rotated input of down_proj)
// mid % 128 == 0, 2*mid <= 2048.
template <typename TG, bool OUTH>
__global__ __launch_bounds__(512) void k_i4r_mid(const TG* __restrict__ guv, uint64_t gs,
                                                 void* out, uint64_t os,
                                                 const float* __restrict__ svh_gu,
                                                 const float* __restrict__ suh_dn, int mid) {
  __shared__ float s[2048];
  const int slot = blockIdx.x;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  {
    const TG* z = guv + (size_t)slot * gs + w * 128;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = (float)z[j * 32 + lane];
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int i = w * 128 + j * 32 + lane;
      s[i] = v[j] * hv2::kRs128 * svh_gu[i];
    }
  }
  __syncthreads();
  if (w >= (mid >> 7)) return;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = w * 128 + j * 32 + lane;
    const float g = s[i], u = s[mid + i];
    v[j] = g / (1.f + expf(-g)) * u * suh_dn[i];
  }
  hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = w * 128 + j * 32 + lane;
    const float o = v[j] * hv2::kRs128;
    if constexpr (OUTH)
      ((__half*)out)[(size_t)slot * os + i] = __float2half(o);
    else
      ((float*)out)[(size_t)slot * os + i] = o;
  }
}

// i4r LUT prefill tail: k_moe_reduce_pw + rot_out (post = svh_dn) + optional
// shared-expert add (k_axpy_sg) in one pass, one wave per (row t, 128 block).
// Same slot order and expressions as the three separate kernels, so the result
// is bit-identical to reduce -> k_v2_rot -> k_axpy_sg. TP = float or __half
// (f16 pairs from moe_lut_down(f16=true) halve the read traffic).
// out[t][i] = svh[i] * H(sum_s pw[id_s] * pairs[id_s])[i] / sqrt128
//             (+ sigmoid(sg[t]) * ey[t][i])
template <int K, typename TP = float>
__global__ __launch_bounds__(256) void k_v2_reduce_rot(const TP* __restrict__ pairs,
                                                       const int* __restrict__ pairids,
                                                       const float* __restrict__ pw,
                                                       const float* __restrict__ svh,
                                                       const float* __restrict__ ey,
                                                       const float* __restrict__ sg,
                                                       float* __restrict__ out, int P, int d) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int nb = d >> 7;
  if (gw >= P * nb) return;
  const int t = gw / nb, b = gw - t * nb;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < K; s++) {
    const int id = pairids[t * K + s];
    const float w = pw[id];
    const TP* pr = pairs + (size_t)id * d + b * 128;
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] += w * hv2::ldp(pr, j * 32 + lane);
  }
  hv2::fwht128(v[0], v[1], v[2], v[3], lane);
  const float sgt = ey ? 1.f / (1.f + expf(-sg[t])) : 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = b * 128 + j * 32 + lane;
    float o = v[j] * hv2::kRs128 * svh[i];
    if (ey) o += sgt * ey[(size_t)t * d + i];
    out[(size_t)t * d + i] = o;
  }
}

// k_v2_reduce_rot + the HC write-back (see k_moe_reduce_pw_sg_hc): the
// rotated result is added straight into the bf16 HC state R with the same
// expression as k_gr_write_b_hc_bf16, skipping the f32 out round-trip.
template <int K, typename TP = float>
__global__ __launch_bounds__(256) void k_v2_reduce_rot_hc(
    const TP* __restrict__ pairs, const int* __restrict__ pairids,
    const float* __restrict__ pw, const float* __restrict__ svh,
    const float* __restrict__ ey, const float* __restrict__ sg,
    const float* __restrict__ w4, uint16_t* __restrict__ R, int P, int d) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int nb = d >> 7;
  if (gw >= P * nb) return;
  const int t = gw / nb, b = gw - t * nb;
  float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < K; s++) {
    const int id = pairids[t * K + s];
    const float w = pw[id];
    const TP* pr = pairs + (size_t)id * d + b * 128;
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] += w * hv2::ldp(pr, j * 32 + lane);
  }
  hv2::fwht128(v[0], v[1], v[2], v[3], lane);
  const float sgt = ey ? 1.f / (1.f + expf(-sg[t])) : 0.f;
  const float* wt = w4 + (size_t)t * 4;
  float inj[4];
#pragma unroll
  for (int b2 = 0; b2 < 4; b2++) inj[b2] = 2.f / (1.f + expf(-wt[b2] * 0.25f));
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = b * 128 + j * 32 + lane;
    float o = v[j] * hv2::kRs128 * svh[i];
    if (ey) o += sgt * ey[(size_t)t * d + i];
    uint16_t* Rt = R + (size_t)t * 4 * d + i;
#pragma unroll
    for (int b2 = 0; b2 < 4; b2++)
      Rt[(size_t)b2 * d] = f2bf(bf2f(Rt[(size_t)b2 * d]) + inj[b2] * o);
  }
}

// --- ht (dtype 16) multi-row GEMV with fused rot_in / rot_out ---------------
// Code layout: 128x128 tiles, tile (ty, tx) = 8 granules of 1 KiB at
// (ty*(K/128)*8 + tx*8 + g) * 1024; granule g = tile cols 16g..16g+15, its
// 128 B block q = tile rows 16q..16q+15; in a block, u32 word L belongs to
// row L%16 / column half L/16 and carries 8 values (window over words L, L-1).
//
// grid = (ceil(nrows/128), KS), block 256 = 8 waves; wave w decodes granule
// w (tile cols 16w..16w+15, all 128 rows) of tile row ty over the block's
// tile columns [kz*tpb, ...). A lane's 8 x values per P row are read from LDS
// once per tile and reused by its 8 row blocks (rows-per-wave layout read x
// from LDS once per 8 weights and ran 3-5x slower at P=4). The rotated x goes
// through LDS tpc tiles at a time; codes of tile t+1 are loaded while tile t
// is decoded. Wave partials are summed in a fixed order (deterministic).
// KS > 1: partial z atomically summed in zs[p*O + row] (all zero on entry and
// left zero), the last block of tile row ty (counter cnt[ty]) does rot_out.
// LDS: max(P * tpc, 8 * P) * 128 f32. y[p*ys + row] for row < nrows.
template <int P, typename TX>
__global__ __launch_bounds__(256) void k_ht_gemv(const uint8_t* __restrict__ codes,
                                                 const float* __restrict__ suh,
                                                 const float* __restrict__ svh,
                                                 const TX* __restrict__ x, uint64_t xs,
                                                 float* __restrict__ y, uint64_t ys, int O,
                                                 int K, int nrows, int tpb, int tpc,
                                                 float* __restrict__ zs, int* __restrict__ cnt) {
  extern __shared__ float hv2_lds[];
  float* xh = hv2_lds;  // rotated x chunk; after the loop wave partials, then z
  float* zl = hv2_lds;
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int ty = blockIdx.x, kz = blockIdx.y;
  const int ntx = K >> 7;
  const int tx0 = kz * tpb;
  const int nt = min(ntx, tx0 + tpb) - tx0;

  float acc[8][P];
#pragma unroll
  for (int q = 0; q < 8; q++)
#pragma unroll
    for (int p = 0; p < P; p++) acc[q][p] = 0.f;
  const int h = lane >> 4;
  const uint8_t* base = codes + ((size_t)ty * ntx + tx0) * 8192 + w * 1024 + lane * 4;
  uint32_t nxt[8];
#pragma unroll
  for (int q = 0; q < 8; q++) nxt[q] = *(const uint32_t*)(base + q * 128);
  for (int t = 0; t < nt; t++) {
    const int tc = t % tpc;
    if (tc == 0) {  // rotate the next chunk of x into LDS (t is block-uniform)
      const int nc = min(tpc, nt - t);
      if (t) __syncthreads();
      for (int c = w; c < P * nc; c += 8) {
        const int p = c / nc, u = c - p * nc;
        const int col0 = (tx0 + t + u) * 128;
        const TX* xr = x + (size_t)p * xs;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int i = col0 + j * 32 + lane;
          v[j] = hv2::ldx(xr, i) * suh[i];
        }
        hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
        for (int j = 0; j < 4; j++) xh[(p * tpc + u) * 128 + j * 32 + lane] = v[j] * hv2::kRs128;
      }
      __syncthreads();
    }
    uint32_t cw[8];
#pragma unroll
    for (int q = 0; q < 8; q++) cw[q] = nxt[q];
    if (t + 1 < nt) {
      const uint8_t* tb = base + (size_t)(t + 1) * 8192;
#pragma unroll
      for (int q = 0; q < 8; q++) nxt[q] = *(const uint32_t*)(tb + q * 128);
    }
    float xv[P][8];
    const float* xt = xh + tc * 128 + w * 16 + h * 8;
#pragma unroll
    for (int p = 0; p < P; p++) {
      const float4 a = *(const float4*)(xt + p * tpc * 128);
      const float4 b = *(const float4*)(xt + p * tpc * 128 + 4);
      xv[p][0] = a.x; xv[p][1] = a.y; xv[p][2] = a.z; xv[p][3] = a.w;
      xv[p][4] = b.x; xv[p][5] = b.y; xv[p][6] = b.z; xv[p][7] = b.w;
    }
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const uint32_t cur = cw[q];
      // word of lane - 1 (lane 0 <- 31): rotate within 16-lane rows (DPP
      // row_ror:1), each row's first lane takes the other row's value
      const uint32_t rr = (uint32_t)__builtin_amdgcn_mov_dpp((int)cur, 0x121, 0xf, 0xf, false);
      const uint32_t xx = (uint32_t)__builtin_amdgcn_permlanex16((int)rr, (int)rr, 0x76543210,
                                                                 (int)0xfedcba98, false, false);
      const uint32_t prev = (lane & 15) ? rr : xx;
      float wv[8];
#pragma unroll
      for (int e = 0; e < 8; e += 2)
        hv2::ht_val2(__builtin_amdgcn_alignbit(prev, cur, 28 - 4 * e) & 0xffffu,
                     __builtin_amdgcn_alignbit(prev, cur, 24 - 4 * e) & 0xffffu, wv[e], wv[e + 1]);
#pragma unroll
      for (int p = 0; p < P; p++) {
        float s = acc[q][p];
#pragma unroll
        for (int e = 0; e < 8; e++) s = fmaf(wv[e], xv[p][e], s);
        acc[q][p] = s;
      }
    }
  }
#pragma unroll
  for (int q = 0; q < 8; q++)
#pragma unroll
    for (int p = 0; p < P; p++) acc[q][p] += __shfl_xor(acc[q][p], 16, 32);
  __syncthreads();  // x chunk dead, LDS becomes zw[w][p][128]
  if (lane < 16)
#pragma unroll
    for (int q = 0; q < 8; q++)
#pragma unroll
      for (int p = 0; p < P; p++) hv2_lds[(w * P + p) * 128 + q * 16 + lane] = acc[q][p];
  __syncthreads();
  for (int i = tid; i < P * 128; i += 256) {
    float s = hv2_lds[i];
#pragma unroll
    for (int v = 1; v < 8; v++) s += hv2_lds[v * P * 128 + i];
    zl[i] = s;  // only thread i touches index i of wave 0's slice
  }
  __syncthreads();
  if (gridDim.y > 1) {
    // aggr 10-05: deterministic split-K. Each K slice stores its partial in
    // its own slot zs[(kz*P + p)*O + row]; the last-arriving block sums the
    // slots in kz order. (The float atomicAdd form made v2 decode / verify
    // depend on block arrival order: spec-decode and 4-slot serving outputs
    // were not reproducible.)
    for (int i = tid; i < P * 128; i += 256)
      zs[((size_t)kz * P + (i >> 7)) * O + ty * 128 + (i & 127)] = zl[i];
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(cnt + ty, 1) == (int)gridDim.y - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int i = tid; i < P * 128; i += 256) {
      const size_t o = (size_t)(i >> 7) * O + ty * 128 + (i & 127);
      float s = __builtin_nontemporal_load(zs + o);
      for (int k2 = 1; k2 < (int)gridDim.y; k2++)
        s += __builtin_nontemporal_load(zs + (size_t)k2 * P * O + o);
      zl[i] = s;
    }
    if (tid == 0) atomicExch(cnt + ty, 0);
    __syncthreads();
  }
  if (w < P) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = zl[w * 128 + j * 32 + lane];
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = ty * 128 + j * 32 + lane;
      if (r < nrows) y[(size_t)w * ys + r] = v[j] * hv2::kRs128 * svh[r];
    }
  }
}

// --- k_ht_wmma: row-invariant WMMA ht GEMV (dense WMMA, 10-12) --------------
// Same contract as k_ht_gemv (rot_in -> z = G.x^ -> split-K -> rot_out) for
// P = 1..16*NPT rows, but z runs on v_wmma_f32_16x16x16_f16: x^ is rounded to
// f16 (the only new rounding: ht_val is exactly f16 already), f32 accumulate.
// Row invariance (concurrent == single-run bit for bit): every output element
// (p, o) is the WMMA chain over (tile t, 16-col group g) in fixed order, its
// A row is row p's own x^ (rows >= P are never stored), the split-K count ks is a
// function of the shape only (host ht_wmma_ks), and the K-slice partials are
// summed in kz order. Nothing depends on P or on the row's position in the
// batch (tests/ht_wmma_test.cu checks shifted / reversed batches bitwise).
// Layout: wave w owns rows [w*16, +16) of the 128-row tile and walks all 8
// column groups g of each tile; a 16x16 code block (128 B at g*1024 + w*128)
// decodes to lane l: row l&15, cols (l>>4)*8..+7 = half the WMMA B column
// (gfx1151 B(k,c): lane c and lane 16+c both hold col c) -> one permlanex16
// swap completes it. A(r,k) (r<8: lane 2r, r>=8: lane 17+2(r-8)) is read
// from the f16 x^ chunk in LDS. C(r,c) = lane c + 16*(r>=8), elem r%8.
// NPT == 1 (P <= 16): every A lane loads (rows >= P read row P-1, then select
// zero), so the loop body has no exec branch per column group: one basic
// block per tile for the scheduler, 3-5% faster at P <= 8 than a guarded
// load (11-10 ht2 test). Built as a vector + select; plain clamped loads
// without the select, or A registers kept across groups, scheduled worse.
// LDS: max(P*(tpc*128+8) f16 x^ chunk, P*128 f32 z).
typedef _Float16 htw_v16h __attribute__((ext_vector_type(16)));
typedef float htw_v8f __attribute__((ext_vector_type(8)));
typedef uint32_t htw_v8u __attribute__((ext_vector_type(8)));  // A operand bits
namespace hv2 {
// packed f16 pair of ht_val2 (same bits, no f32 round trip)
__device__ __forceinline__ uint32_t ht_h2(uint32_t a0, uint32_t a1) {
  const uint32_t s0 = __builtin_amdgcn_sad_u8(a0 * 0x83dcd12du, 0u, 0x6400u);
  const uint32_t s1 = __builtin_amdgcn_sad_u8(a1 * 0x83dcd12du, 0u, 0x6400u);
  const uint32_t hp = s0 | (s1 << 16);
  const __half2 r = __hfma2(*(const __half2*)&hp,
                            __half2half2(__float2half(1.732421875f / 256.f)),
                            __half2half2(__float2half(-10.3828125f)));
  return *(const uint32_t*)&r;
}
}  // namespace hv2

template <int NPT, typename TX>
__global__ __launch_bounds__(256) void k_ht_wmma(const uint8_t* __restrict__ codes,
                                                 const float* __restrict__ suh,
                                                 const float* __restrict__ svh,
                                                 const TX* __restrict__ x, uint64_t xs,
                                                 float* __restrict__ y, uint64_t ys, int O,
                                                 int K, int nrows, int tpb, int tpc, int P,
                                                 float* __restrict__ zs, int* __restrict__ cnt,
                                                 int nty0 = 1 << 30,
                                                 const uint8_t* __restrict__ codes2 = nullptr,
                                                 const float* __restrict__ suh2 = nullptr,
                                                 const float* __restrict__ svh2 = nullptr,
                                                 float* __restrict__ y2 = nullptr,
                                                 int nrows2 = 0) {
  extern __shared__ __align__(16) unsigned char hw_lds[];
  __half* xh = (__half*)hw_lds;  // [P][RS] rotated x chunk (f16), rows >= P not stored
  float* zl = (float*)hw_lds;    // after the loop: z [P][128]
  __shared__ int s_last;
  const int RS = tpc * 128 + 8;  // row stride (halves), +16 B against bank aliasing
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int ty = blockIdx.x, kz = blockIdx.y;  // ty: global tile row (split-K slots)
  // Pair launch (shared-expert gate + up, same K and x): tile rows >= nty0
  // belong to the second weight, with its own suh / svh / y. tyl = tile row
  // within its weight.
  const bool sec = ty >= nty0;
  const int tyl = sec ? ty - nty0 : ty;
  if (sec) {
    codes = codes2;
    suh = suh2;
    svh = svh2;
    y = y2;
    nrows = nrows2;
  }
  const int ntx = K >> 7;
  const int tx0 = kz * tpb;
  const int nt = min(ntx, tx0 + tpb) - tx0;
  // A-fragment row of this lane (k_gemm_wmma mapping; ignored lanes get
  // complementary dummy rows so LDS phases spread over banks)
  const int b_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? b_ : ((b_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((b_ + 4) & 7) : b_);
  int arow[NPT];  // A row loaded by this lane: its own row, clamped to P-1
#pragma unroll
  for (int pt = 0; pt < NPT; pt++) arow[pt] = min(pt * 16 + qrowA, P - 1);
  htw_v8f acc[NPT];
#pragma unroll
  for (int i = 0; i < NPT; i++)
#pragma unroll
    for (int e = 0; e < 8; e++) acc[i][e] = 0.f;
  const uint8_t* base = codes + ((size_t)tyl * ntx + tx0) * 8192 + w * 128 + lane * 4;
  uint32_t nxt[8];
#pragma unroll
  for (int g = 0; g < 8; g++) nxt[g] = *(const uint32_t*)(base + g * 1024);
  for (int t = 0; t < nt; t++) {
    const int tc = t % tpc;
    if (tc == 0) {  // rotate the next chunk of x into LDS (t is block-uniform)
      const int nc = min(tpc, nt - t);
      if (t) __syncthreads();
      for (int c = w; c < P * nc; c += 8) {
        const int p = c / nc, u = c - p * nc;
        __half* dst = xh + (size_t)p * RS + u * 128;
        const int col0 = (tx0 + t + u) * 128;
        const TX* xr = x + (size_t)p * xs;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int i = col0 + j * 32 + lane;
          v[j] = hv2::ldx(xr, i) * suh[i];
        }
        hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
        for (int j = 0; j < 4; j++) dst[j * 32 + lane] = __float2half(v[j] * hv2::kRs128);
      }
      __syncthreads();
    }
    uint32_t cw[8];
#pragma unroll
    for (int g = 0; g < 8; g++) cw[g] = nxt[g];
    if (t + 1 < nt) {
      const uint8_t* tb = base + (size_t)(t + 1) * 8192;
#pragma unroll
      for (int g = 0; g < 8; g++) nxt[g] = *(const uint32_t*)(tb + g * 1024);
    }
#pragma unroll
    for (int g = 0; g < 8; g++) {
      const uint32_t cur = cw[g];
      // word of lane - 1 (lane 0 <- 31), exactly as k_ht_gemv
      const uint32_t rr = (uint32_t)__builtin_amdgcn_mov_dpp((int)cur, 0x121, 0xf, 0xf, false);
      const uint32_t xx = (uint32_t)__builtin_amdgcn_permlanex16((int)rr, (int)rr, 0x76543210,
                                                                 (int)0xfedcba98, false, false);
      const uint32_t prev = (lane & 15) ? rr : xx;
      uint32_t own[4], oth[4];
#pragma unroll
      for (int j = 0; j < 4; j++)
        own[j] = hv2::ht_h2(__builtin_amdgcn_alignbit(prev, cur, 28 - 8 * j) & 0xffffu,
                            __builtin_amdgcn_alignbit(prev, cur, 24 - 8 * j) & 0xffffu);
#pragma unroll
      for (int j = 0; j < 4; j++)
        oth[j] = (uint32_t)__builtin_amdgcn_permlanex16((int)own[j], (int)own[j], 0x76543210,
                                                        (int)0xfedcba98, false, false);
      const bool hi = lane >= 16;
      uint32_t bw[8];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        bw[j] = hi ? oth[j] : own[j];
        bw[4 + j] = hi ? own[j] : oth[j];
      }
      htw_v16h bf;
      memcpy(&bf, bw, 32);
#pragma unroll
      for (int pt = 0; pt < NPT; pt++) {
        htw_v8u av;
        if constexpr (NPT == 1) {  // P <= 16: branch-free (see header)
          const __half* ap = xh + (size_t)arow[pt] * RS + tc * 128 + g * 16;
          const uint4 lo = *(const uint4*)ap, hi = *(const uint4*)(ap + 8);
          av = (htw_v8u){lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
          if (pt * 16 + qrowA >= P) av = (htw_v8u)(0u);  // rows >= P: zero
        } else {  // larger batches: the guarded load is faster (3 selects per group)
          av = (htw_v8u)(0u);
          if (pt * 16 + qrowA < P) {
            const __half* ap = xh + (size_t)arow[pt] * RS + tc * 128 + g * 16;
            const uint4 lo = *(const uint4*)ap, hi = *(const uint4*)(ap + 8);
            av = (htw_v8u){lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
          }
        }
        acc[pt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((htw_v16h)av, bf, acc[pt]);
      }
    }
  }
  __syncthreads();  // x^ chunk dead, LDS becomes z [P][128]
#pragma unroll
  for (int pt = 0; pt < NPT; pt++)
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const int p = pt * 16 + (lane >> 4) * 8 + e;
      if (p < P) zl[p * 128 + w * 16 + (lane & 15)] = acc[pt][e];
    }
  __syncthreads();
  if (gridDim.y > 1) {  // deterministic split-K, as k_ht_gemv
    for (int i = tid; i < P * 128; i += 256)
      zs[((size_t)kz * P + (i >> 7)) * O + ty * 128 + (i & 127)] = zl[i];
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(cnt + ty, 1) == (int)gridDim.y - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int i = tid; i < P * 128; i += 256) {
      const size_t o = (size_t)(i >> 7) * O + ty * 128 + (i & 127);
      float s = __builtin_nontemporal_load(zs + o);
      for (int k2 = 1; k2 < (int)gridDim.y; k2++)
        s += __builtin_nontemporal_load(zs + (size_t)k2 * P * O + o);
      zl[i] = s;
    }
    if (tid == 0) atomicExch(cnt + ty, 0);
    __syncthreads();
  }
  for (int p = w; p < P; p += 8) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = zl[p * 128 + j * 32 + lane];
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = tyl * 128 + j * 32 + lane;
      if (r < nrows) y[(size_t)p * ys + r] = v[j] * hv2::kRs128 * svh[r];
    }
  }
}

// ht tile -> original-domain bf16 W[O][K] (prefill dequant into d_wbf16).
// One block per tile; 64 KiB LDS tile, swizzled [r][c ^ (r & 31)].
__global__ __launch_bounds__(256) void k_ht_deq_bf16(const uint8_t* __restrict__ codes,
                                                     const float* __restrict__ suh,
                                                     const float* __restrict__ svh,
                                                     uint16_t* __restrict__ out, int K) {
  __shared__ float t[128 * 128];
  const int ntx = K >> 7;
  const int tile = blockIdx.x, ty = tile / ntx, tx = tile - ty * ntx;
  const uint8_t* tb = codes + (size_t)tile * 8192;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int prev_lane = (lane + 31) & 31;
  for (int it = 0; it < 8; it++) {
    const int blk = it * 8 + w, g = blk >> 3, q = blk & 7;
    const uint32_t cur = *(const uint32_t*)(tb + g * 1024 + q * 128 + lane * 4);
    const uint32_t prev = __shfl(cur, prev_lane, 32);
    const int r = q * 16 + (lane & 15);
    const int c0 = g * 16 + (lane >> 4) * 8;
#pragma unroll
    for (int e = 0; e < 8; e++)
      t[r * 128 + ((c0 + e) ^ (r & 31))] =
          hv2::ht_val(__builtin_amdgcn_alignbit(prev, cur, 28 - 4 * e) & 0xffffu);
  }
  __syncthreads();
  for (int cc = 0; cc < 16; cc++) {  // H along rows (left multiply)
    const int c = w * 16 + cc;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = j * 32 + lane;
      v[j] = t[r * 128 + (c ^ (r & 31))];
    }
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = j * 32 + lane;
      t[r * 128 + (c ^ (r & 31))] = v[j];
    }
  }
  __syncthreads();
  for (int rr = 0; rr < 16; rr++) {  // H along columns, scale, emit
    const int r = w * 16 + rr;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int c = j * 32 + lane;
      v[j] = t[r * 128 + (c ^ (r & 31))];
    }
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
    const int R = ty * 128 + r;
    const float sv = svh[R] * (1.f / 128.f);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int C = tx * 128 + j * 32 + lane;
      out[(size_t)R * K + C] = f2bf(v[j] * sv * suh[C]);
    }
  }
}

// Same arithmetic as k_ht_deq_bf16 (bit-identical), half the LDS (32 KiB ->
// 2x resident blocks): decoded codes are exactly f16, so they are staged as
// f16; the column FWHT results stay in registers (16 columns x 128 rows per
// wave) and reach the row pass through a 64-row fp32 half tile, twice.
__global__ __launch_bounds__(256) void k_ht_deq_bf16_h(const uint8_t* __restrict__ codes,
                                                       const float* __restrict__ suh,
                                                       const float* __restrict__ svh,
                                                       uint16_t* __restrict__ out, int K) {
  __shared__ __attribute__((aligned(16))) float tt[64 * 128];  // 32 KiB, also the f16 stage
  __half* t16 = reinterpret_cast<__half*>(tt);                 // [128][128] f16
  const int ntx = K >> 7;
  const int tile = blockIdx.x, ty = tile / ntx, tx = tile - ty * ntx;
  const uint8_t* tb = codes + (size_t)tile * 8192;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int prev_lane = (lane + 31) & 31;
  // f16 stage swizzle: element (r, c) at r*128 + (c ^ 2(r&31)); a column read
  // across 32 lanes (rows) hits 32 distinct 4 B words.
  for (int it = 0; it < 8; it++) {
    const int blk = it * 8 + w, g = blk >> 3, q = blk & 7;
    const uint32_t cur = *(const uint32_t*)(tb + g * 1024 + q * 128 + lane * 4);
    const uint32_t prev = __shfl(cur, prev_lane, 32);
    const int r = q * 16 + (lane & 15);
    const int c0 = g * 16 + (lane >> 4) * 8;
    const int sw = 2 * (r & 31);
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      const __half h0 = __float2half(
          hv2::ht_val(__builtin_amdgcn_alignbit(prev, cur, 28 - 4 * e) & 0xffffu));
      const __half h1 = __float2half(
          hv2::ht_val(__builtin_amdgcn_alignbit(prev, cur, 28 - 4 * (e + 1)) & 0xffffu));
      const uint32_t pk = (uint32_t)__half_as_ushort(h0) | ((uint32_t)__half_as_ushort(h1) << 16);
      *(uint32_t*)(t16 + r * 128 + ((c0 + e) ^ sw)) = pk;
    }
  }
  __syncthreads();
  float cv[16][4];  // column pass results: cv[cc][j] = (row j*32+lane, col w*16+cc)
#pragma unroll
  for (int cc = 0; cc < 16; cc++) {
    const int c = w * 16 + cc;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = j * 32 + lane;
      cv[cc][j] = __half2float(t16[r * 128 + (c ^ (2 * (r & 31)))]);
    }
    hv2::fwht128(cv[cc][0], cv[cc][1], cv[cc][2], cv[cc][3], lane);
  }
  float su[4];
#pragma unroll
  for (int j = 0; j < 4; j++) su[j] = suh[tx * 128 + j * 32 + lane];
#pragma unroll
  for (int h = 0; h < 2; h++) {
    __syncthreads();  // previous readers of tt (f16 stage / last half) are done
#pragma unroll
    for (int cc = 0; cc < 16; cc++) {
      const int c = w * 16 + cc;
#pragma unroll
      for (int jj = 0; jj < 2; jj++) {
        const int rl = jj * 32 + lane;  // local row in this half; global row h*64+rl
        tt[rl * 128 + (c ^ (rl & 31))] = cv[cc][2 * h + jj];
      }
    }
    __syncthreads();
    for (int rr = 0; rr < 8; rr++) {
      const int rl = w * 8 + rr, r = h * 64 + rl;
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int c = j * 32 + lane;
        v[j] = tt[rl * 128 + (c ^ (rl & 31))];
      }
      hv2::fwht128(v[0], v[1], v[2], v[3], lane);
      const int R = ty * 128 + r;
      const float sv = svh[R] * (1.f / 128.f);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int C = tx * 128 + j * 32 + lane;
        out[(size_t)R * K + C] = f2bf(v[j] * sv * su[j]);
      }
    }
  }
}

// --- q6g64 (dtype 24) --------------------------------------------------------
// Row r at r * st (st = align16(13 C / 16)): C/2 B low nibbles (even col =
// low), C/4 B high-2-bit plane (byte k/4, bits 2(k%4)), C/64 x (f16 scale,
// f16 bias). W = code * scale + bias.
__host__ __device__ constexpr uint64_t q6g64_stride(uint64_t C) {
  return ((13 * C / 16) + 15) & ~15ull;
}

// LPR lanes per row, groups strided by LPR; y[p*ys + r].
template <int P, typename TX, int LPR>
__global__ __launch_bounds__(256) void k_q6g64_gemv(const uint8_t* __restrict__ wq,
                                                    const TX* __restrict__ x, uint64_t xs,
                                                    float* __restrict__ y, uint64_t ys, int R,
                                                    int C, int st) {
  const int tid = threadIdx.x;
  const int r = blockIdx.x * (256 / LPR) + tid / LPR;
  const int li = tid % LPR;
  if (r >= R) return;  // whole LPR groups exit together
  const uint8_t* row = wq + (size_t)r * st;
  const int ng = C >> 6;
  float acc[P];
#pragma unroll
  for (int p = 0; p < P; p++) acc[p] = 0.f;
  for (int g = li; g < ng; g += LPR) {
    const uint4 l0 = *(const uint4*)(row + g * 32), l1 = *(const uint4*)(row + g * 32 + 16);
    const uint4 hp = *(const uint4*)(row + C / 2 + g * 16);
    const uint32_t sb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
    const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
    const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
    const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    const uint32_t hw[4] = {hp.x, hp.y, hp.z, hp.w};
    float dx[P], sx[P];
#pragma unroll
    for (int p = 0; p < P; p++) dx[p] = sx[p] = 0.f;
#pragma unroll
    for (int i8 = 0; i8 < 8; i8++) {
      float cf[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t lo = (lw[i8] >> (4 * e)) & 15u;
        const uint32_t hi = (hw[i8 >> 1] >> (2 * ((i8 & 1) * 8 + e))) & 3u;
        cf[e] = (float)(lo | (hi << 4));
      }
#pragma unroll
      for (int p = 0; p < P; p++) {
        float xv[8];
        hv2::ldx8(x + (size_t)p * xs + g * 64 + i8 * 8, xv);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          dx[p] = fmaf(cf[e], xv[e], dx[p]);
          sx[p] += xv[e];
        }
      }
    }
#pragma unroll
    for (int p = 0; p < P; p++) acc[p] += sc * dx[p] + bi * sx[p];
  }
  if constexpr (LPR <= 32) {
#pragma unroll
    for (int p = 0; p < P; p++)
#pragma unroll
      for (int off = LPR / 2; off > 0; off >>= 1) acc[p] += __shfl_xor(acc[p], off, LPR);
    if (li == 0)
#pragma unroll
      for (int p = 0; p < P; p++) y[(size_t)p * ys + r] = acc[p];
  } else {
    // aggr 10-05: LPR = 256 (one row per block) for the K=10240 HC mix-down /
    // inject: the LPR=32 form ran 40 blocks for R=320 (192 us, 13 GB/s at
    // P=7). Wave shfl tree, then a fixed-order sum of the 8 wave partials.
    static_assert(LPR == 256, "LPR > 32 means one 256-lane row per block");
    __shared__ float red[P][8];
#pragma unroll
    for (int p = 0; p < P; p++)
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc[p] += __shfl_xor(acc[p], off, 32);
    if ((tid & 31) == 0)
#pragma unroll
      for (int p = 0; p < P; p++) red[p][tid >> 5] = acc[p];
    __syncthreads();
    if (tid < P) {
      float s = red[tid][0];
#pragma unroll
      for (int w = 1; w < 8; w++) s += red[tid][w];
      y[(size_t)tid * ys + r] = s;
    }
  }
}

// aggr 10-05: split-K q6g64 GEMV for the K=10240 HC mix-down at P<=8. The
// LPR=32 kernel ran 40 blocks (R=320 / 8 rows) with 5 serial groups per lane
// (~192 us, latency-bound). Here blockIdx.y = K slice of gps groups; partial
// part[s][p][r] (same per-lane group order inside a slice, then the slices are
// summed in order by k_q6g64_skred) -> deterministic, not bit-identical to
// the LPR=32 kernel (different summation tree).
template <int P, typename TX>
__global__ __launch_bounds__(256) void k_q6g64_gemv_sk(const uint8_t* __restrict__ wq,
                                                       const TX* __restrict__ x, uint64_t xs,
                                                       float* __restrict__ part, int R, int C,
                                                       int st, int gps) {
  // One wave per row; 8 lanes share a 64-col group (lane sub = 8-elem slice
  // i8), 4 groups per wave step -> a wave's x loads are 512 contiguous bytes
  // (the LPR=32 kernel's lanes were 128 B apart: 32 lines per load).
  constexpr int LPR = 32;
  const int tid = threadIdx.x;
  const int r = blockIdx.x * (256 / LPR) + tid / LPR;
  const int li = tid % LPR;
  const int sub = li & 7, gq = li >> 3;
  const int s = blockIdx.y;
  if (r >= R) return;
  const uint8_t* row = wq + (size_t)r * st;
  const int g0 = s * gps, g1 = min(C >> 6, g0 + gps);
  float acc[P];
#pragma unroll
  for (int p = 0; p < P; p++) acc[p] = 0.f;
  for (int g = g0 + gq; g < g1; g += 4) {
    const uint32_t lw = *(const uint32_t*)(row + g * 32 + sub * 4);
    const uint32_t hw = *(const uint32_t*)(row + C / 2 + g * 16 + (sub >> 1) * 4);
    const uint32_t sb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
    const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
    const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
    float cf[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const uint32_t lo = (lw >> (4 * e)) & 15u;
      const uint32_t hi = (hw >> (2 * ((sub & 1) * 8 + e))) & 3u;
      cf[e] = (float)(lo | (hi << 4));
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      float xv[8];
      hv2::ldx8(x + (size_t)p * xs + g * 64 + sub * 8, xv);
      float dx = 0.f, sx = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        dx = fmaf(cf[e], xv[e], dx);
        sx += xv[e];
      }
      acc[p] += sc * dx + bi * sx;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++)
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1) acc[p] += __shfl_xor(acc[p], off, LPR);
  if (li == 0)
#pragma unroll
    for (int p = 0; p < P; p++) part[((size_t)s * P + p) * R + r] = acc[p];
}
template <int P>
__global__ void k_q6g64_skred(const float* __restrict__ part, float* __restrict__ y,
                              uint64_t ys, int R, int ns) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * R) return;
  const int p = i / R, r = i % R;
  float v = part[(size_t)p * R + r];
  for (int s = 1; s < ns; s++) v += part[((size_t)s * P + p) * R + r];
  y[(size_t)p * ys + r] = v;
}

// --- k_q6g64_wmma: row-invariant WMMA q6g64 GEMV (dense WMMA, 10-12) -------
// y[p][r] = sum_g sc_g[r] * (sum_k c_k x_pk) + bi_g[r] * (sum_k x_pk), 64-col
// groups g. The 6-bit codes c (0..63) are exact in f16, so W is never rounded:
// the per-group dot runs on v_wmma_f32_16x16x16_f16 (x -> f16 is the only new
// rounding), scale / bias are applied per group in f32, and the bias term uses
// the f32 group sum of the ORIGINAL x (fixed 8-lane tree). Block = 8 waves as
// RW (16-row tiles) x KW (groups interleaved g = g0+kw, g0+kw+KW, ...), one K
// slice of gps groups per blockIdx.y; KW partials summed in kw order in LDS,
// slices summed in s order by k_q6g64_wred. RW/KW/gps/ns are functions of the
// shape only and every output (p, r) only reads x row p -> row-invariant for
// P = 1..16*NPT (any P, any position in the batch).
// WMMA operands as k_ht_wmma: A = x rows (A(r,k): r<8 lane 2r, r>=8 lane
// 17+2(r-8)), B = weight rows (lane c / 16+c: row c, each decodes 8 of the 16
// cols, one permlanex16 swap), C(r,c) = lane c + 16*(r>=8), elem r%8.
template <int NPT, int RW, int KW, typename TX>
__global__ __launch_bounds__(256) void k_q6g64_wmma(const uint8_t* __restrict__ wq,
                                                    const TX* __restrict__ x, uint64_t xs,
                                                    float* __restrict__ out, uint64_t ys, int R,
                                                    int C, int st, int gps, int P) {
  static_assert(RW * KW == 8, "8 waves");
  __shared__ float sxs[48 * 64];              // [p][gl] f32 group sums, gps <= 64
  __shared__ float red[8][16 * NPT][16];      // per-wave partial tiles
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int rw = w / KW, kw = w % KW;
  const int s = blockIdx.y;
  const int ng = C >> 6;
  const int g0 = s * gps, g1 = min(ng, g0 + gps), nsl = g1 - g0;
  const int n = lane & 15, h = lane >> 4;
  const int r0 = (blockIdx.x * RW + rw) * 16;
  const uint8_t* row = wq + (size_t)(r0 + n) * st;  // R % (16*RW) == 0 (host)
  // weight group data, double-buffered: group g+KW is loaded while g computes;
  // the first group is issued before the x group-sum prologue
  uint4 cl0, cl1, chp, nl0, nl1, nhp;
  uint32_t csb = 0, nsb = 0;
  int g = g0 + kw;
  if (g < g1) {
    cl0 = *(const uint4*)(row + g * 32);
    cl1 = *(const uint4*)(row + g * 32 + 16);
    chp = *(const uint4*)(row + C / 2 + g * 16);
    csb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
  }
  for (int i = tid; i < P * nsl * 8; i += 256) {  // 8-lane groups stay whole
    const int sub = i & 7, pg = i >> 3, p = pg / nsl, gl = pg - p * nsl;
    float xv[8];
    hv2::ldx8(x + (size_t)p * xs + (size_t)(g0 + gl) * 64 + sub * 8, xv);
    float t = ((xv[0] + xv[1]) + (xv[2] + xv[3])) + ((xv[4] + xv[5]) + (xv[6] + xv[7]));
    t += __shfl_xor(t, 1, 8);
    t += __shfl_xor(t, 2, 8);
    t += __shfl_xor(t, 4, 8);
    if (sub == 0) sxs[p * nsl + gl] = t;
  }
  __syncthreads();
  const int b_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? b_ : ((b_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((b_ + 4) & 7) : b_);
  htw_v8f acc[NPT];
#pragma unroll
  for (int i = 0; i < NPT; i++)
#pragma unroll
    for (int e = 0; e < 8; e++) acc[i][e] = 0.f;
  const __half2 k1024 = __half2half2(__float2half(1024.f));
  for (; g < g1; g += KW) {
    if (g + KW < g1) {
      const int gn = g + KW;
      nl0 = *(const uint4*)(row + gn * 32);
      nl1 = *(const uint4*)(row + gn * 32 + 16);
      nhp = *(const uint4*)(row + C / 2 + gn * 16);
      nsb = *(const uint32_t*)(row + 3 * C / 4 + gn * 4);
    }
    const uint4 l0 = cl0, l1 = cl1, hp = chp;
    const uint32_t sb = csb;
    cl0 = nl0;
    cl1 = nl1;
    chp = nhp;
    csb = nsb;
    const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
    const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
    const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    const uint32_t hw[4] = {hp.x, hp.y, hp.z, hp.w};
    htw_v8f d[NPT];
#pragma unroll
    for (int i = 0; i < NPT; i++)
#pragma unroll
      for (int e = 0; e < 8; e++) d[i][e] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      // this lane: weight row n, cols 16ks + 8h .. +7 of the group
      const uint32_t lo = h ? lw[2 * ks + 1] : lw[2 * ks];
      const uint32_t hi = hw[ks] >> (16 * h);
      uint32_t own[4], oth[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const uint32_t c0 = ((lo >> (8 * j)) & 15u) | (((hi >> (4 * j)) & 3u) << 4);
        const uint32_t c1 = ((lo >> (8 * j + 4)) & 15u) | (((hi >> (4 * j + 2)) & 3u) << 4);
        const uint32_t u = (0x6400u | c0) | ((0x6400u | c1) << 16);  // f16 1024 + c, exact
        const __half2 v = __hsub2(*(const __half2*)&u, k1024);       // c, exact
        own[j] = *(const uint32_t*)&v;
      }
#pragma unroll
      for (int j = 0; j < 4; j++)
        oth[j] = (uint32_t)__builtin_amdgcn_permlanex16((int)own[j], (int)own[j], 0x76543210,
                                                        (int)0xfedcba98, false, false);
      uint32_t bw[8];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        bw[j] = h ? oth[j] : own[j];
        bw[4 + j] = h ? own[j] : oth[j];
      }
      htw_v16h bf;
      memcpy(&bf, bw, 32);
#pragma unroll
      for (int pt = 0; pt < NPT; pt++) {
        const int p = pt * 16 + qrowA;
        uint32_t aw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (p < P) {
          const TX* xp = x + (size_t)p * xs + (size_t)g * 64 + ks * 16;
          float xv[16];
          hv2::ldx8(xp, xv);
          hv2::ldx8(xp + 8, xv + 8);
#pragma unroll
          for (int j = 0; j < 8; j++) {
            const __half2 hv = __floats2half2_rn(xv[2 * j], xv[2 * j + 1]);
            aw[j] = *(const uint32_t*)&hv;
          }
        }
        htw_v16h af;
        memcpy(&af, aw, 32);
        d[pt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, d[pt]);
      }
    }
#pragma unroll
    for (int pt = 0; pt < NPT; pt++)
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const int p = pt * 16 + h * 8 + e;
        const float sxv = p < P ? sxs[p * nsl + (g - g0)] : 0.f;
        acc[pt][e] = fmaf(bi, sxv, fmaf(sc, d[pt][e], acc[pt][e]));
      }
  }
#pragma unroll
  for (int pt = 0; pt < NPT; pt++)
#pragma unroll
    for (int e = 0; e < 8; e++) red[w][pt * 16 + h * 8 + e][n] = acc[pt][e];
  __syncthreads();
  for (int i = tid; i < RW * P * 16; i += 256) {
    const int nn = i & 15, rest = i >> 4, p = rest % P, rwi = rest / P;
    float v = red[rwi * KW][p][nn];
#pragma unroll
    for (int k = 1; k < KW; k++) v += red[rwi * KW + k][p][nn];
    const int r = (blockIdx.x * RW + rwi) * 16 + nn;
    if (gridDim.y > 1)
      out[((size_t)s * P + p) * R + r] = v;
    else
      out[(size_t)p * ys + r] = v;
  }
}
// split-K slices of k_q6g64_wmma summed in s order (runtime P)
__global__ void k_q6g64_wred(const float* __restrict__ part, float* __restrict__ y, uint64_t ys,
                             int R, int ns, int P) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P * R) return;
  const int p = i / R, r = i % R;
  float v = part[(size_t)p * R + r];
  for (int s = 1; s < ns; s++) v += part[((size_t)s * P + p) * R + r];
  y[(size_t)p * ys + r] = v;
}

// One thread per 64-col group -> bf16 W[R][C].
__global__ void k_q6g64_deq_bf16(const uint8_t* __restrict__ wq, uint16_t* __restrict__ out,
                                 int R, int C, int st) {
  const int ng = C >> 6;
  const size_t gi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= (size_t)R * ng) return;
  const int r = (int)(gi / ng), g = (int)(gi % ng);
  const uint8_t* row = wq + (size_t)r * st;
  const uint4 l0 = *(const uint4*)(row + g * 32), l1 = *(const uint4*)(row + g * 32 + 16);
  const uint4 hp = *(const uint4*)(row + C / 2 + g * 16);
  const uint32_t sb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
  const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
  const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
  const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
  const uint32_t hw[4] = {hp.x, hp.y, hp.z, hp.w};
  uint16_t* o = out + (size_t)r * C + g * 64;
#pragma unroll
  for (int i8 = 0; i8 < 8; i8++) {
    uint32_t pk[4];
#pragma unroll
    for (int e = 0; e < 8; e += 2) {
      uint32_t c0 = ((lw[i8] >> (4 * e)) & 15u) |
                    (((hw[i8 >> 1] >> (2 * ((i8 & 1) * 8 + e))) & 3u) << 4);
      uint32_t c1 = ((lw[i8] >> (4 * (e + 1))) & 15u) |
                    (((hw[i8 >> 1] >> (2 * ((i8 & 1) * 8 + e + 1))) & 3u) << 4);
      pk[e / 2] = (uint32_t)f2bf(fmaf((float)c0, sc, bi)) |
                  ((uint32_t)f2bf(fmaf((float)c1, sc, bi)) << 16);
    }
    *(uint4*)(o + i8 * 8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
}

// QWENOX_DEQ_BENCH: count differing 32-bit words between two buffers.
__global__ void k_u32_mismatch(const uint32_t* __restrict__ a, const uint32_t* __restrict__ b,
                               size_t n, unsigned long long* cnt) {
  unsigned long long c = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    c += a[i] != b[i];
  if (c) atomicAdd(cnt, c);
}

// One thread per 8 columns (same arithmetic, bit-identical): adjacent lanes
// read adjacent code bytes and write adjacent 16 B, instead of each lane
// walking its own 128 B group.
__global__ void k_q6g64_deq_bf16_c8(const uint8_t* __restrict__ wq, uint16_t* __restrict__ out,
                                    int R, int C, int st) {
  const int n8 = C >> 3;
  const size_t gi = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gi >= (size_t)R * n8) return;
  const int r = (int)(gi / n8), k8 = (int)(gi % n8), g = k8 >> 3, i8 = k8 & 7;
  const uint8_t* row = wq + (size_t)r * st;
  const uint32_t lw = *(const uint32_t*)(row + g * 32 + i8 * 4);
  const uint32_t hw = (uint32_t)*(const uint16_t*)(row + C / 2 + g * 16 + i8 * 2);
  const uint32_t sb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
  const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
  const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
  uint32_t pk[4];
#pragma unroll
  for (int e = 0; e < 8; e += 2) {
    const uint32_t c0 = ((lw >> (4 * e)) & 15u) | (((hw >> (2 * e)) & 3u) << 4);
    const uint32_t c1 = ((lw >> (4 * (e + 1))) & 15u) | (((hw >> (2 * (e + 1))) & 3u) << 4);
    pk[e / 2] = (uint32_t)f2bf(fmaf((float)c0, sc, bi)) |
                ((uint32_t)f2bf(fmaf((float)c1, sc, bi)) << 16);
  }
  *(uint4*)(out + (size_t)r * C + (size_t)k8 * 8) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
}

// --- k_q8g32_wmma: row-invariant WMMA q8g32 GEMV (GGUF Q8_0 dense, 10-13) ----
// q8g32 planar (dtype 8): int8 codes [R][C] row-major, then f16 scales [R][C/32]
// (the scale plane is indexed with the FULL tensor row count). y[p][r] =
// sum_g sc_g[r] * (sum_k c_k x_pk) over 32-col groups g. The int8 codes are
// exact in f16 (0x6400 | (c ^ 0x80) = 1024 + c + 128, minus 1152), so W is never
// rounded; the per-group dot runs on v_wmma_f32_16x16x16_f16 (x -> f16 is the
// only new rounding, as k_q6g64_wmma) and the scale is applied per group in f32.
// Block = 8 waves as RW 16-row tiles x KW group lanes (groups g0+kw, +KW, ...);
// one K slice of gps groups per blockIdx.y; KW partials summed in kw order,
// slices summed in s order (k_q6g64_wred). KW / gps are functions of the FULL
// tensor shape only (host q8g32_wmma), never of P, of the row slice or of the
// row's position in the batch -> every output (p, r) is bit-identical for any
// P = 1..16*NPT and any batch composition (concurrent == single-run).
// Lane layout: lane n = lane & 15 owns weight row r0+n; half h = lane >> 4 loads
// the 16 contiguous codes of K step h (cols 16h..16h+15 of the group) and the
// two halves swap with permlanex16 so both hold each step's 16 K values (gfx11
// wave32 B fragment: lanes c and 16+c both carry column c). A fragment and C
// mapping as k_q6g64_wmma. Rows >= R are clamped for loads and not stored.
namespace q8w {
__device__ __forceinline__ void dec4(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t x = w ^ 0x80808080u;
  const __half2 k = __half2half2(__float2half(1152.f));
  const uint32_t p0 = __builtin_amdgcn_perm(x, 0x64646464u, 0x01050004u);  // 1024+u0 | 1024+u1
  const uint32_t p1 = __builtin_amdgcn_perm(x, 0x64646464u, 0x03070206u);  // 1024+u2 | 1024+u3
  const __half2 h0 = __hsub2(*(const __half2*)&p0, k), h1 = __hsub2(*(const __half2*)&p1, k);
  lo = *(const uint32_t*)&h0;
  hi = *(const uint32_t*)&h1;
}
}  // namespace q8w

template <int NPT, int KW, typename TX, int U = 4>
__global__ __launch_bounds__(256) void k_q8g32_wmma(const int8_t* __restrict__ q,
                                                    const __half* __restrict__ sc, int scs,
                                                    const TX* __restrict__ x, uint64_t xs,
                                                    float* __restrict__ out, uint64_t ys, int R,
                                                    int C, int gps, int P) {
  constexpr int RW = 8 / KW;
  static_assert(RW * KW == 8, "8 waves");
  __shared__ float red[8][16 * NPT][16];  // per-wave partial tiles
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int rw = w / KW, kw = w % KW;
  const int s = blockIdx.y;
  const int ng = C >> 5;
  const int g0 = s * gps, g1 = min(ng, g0 + gps);
  const int n = lane & 15, h = lane >> 4;
  const int r0 = (blockIdx.x * RW + rw) * 16;
  const int rr = min(r0 + n, R - 1);
  const int8_t* row = q + (size_t)rr * C + 16 * h;
  const __half* srow = sc + (size_t)rr * scs;
  const int b_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? b_ : ((b_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((b_ + 4) & 7) : b_);
  htw_v8f acc[NPT];
#pragma unroll
  for (int i = 0; i < NPT; i++)
#pragma unroll
    for (int e = 0; e < 8; e++) acc[i][e] = 0.f;
  // groups of this wave: g0 + kw + KW*i, i < ni, processed in ascending order
  // (the accumulation order is part of the numerics). Loads run U groups
  // ahead (U uint4 + U scales per lane in flight): at P = 1 the kernel is
  // memory-latency bound and one 16 B load in flight per lane reached only
  // ~70-130 GB/s (U is a host choice: QWENOX_Q8W_U, no effect on numerics).
  const int gs = g0 + kw;
  const int ni = gs < g1 ? (g1 - gs + KW - 1) / KW : 0;
  uint4 cw[U], nw[U];
  __half cs[U], nsc[U];
  const auto load = [&](int i0, uint4* wv, __half* sv) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int i = i0 + u;
      if (i < ni) {
        const int g = gs + KW * i;
        wv[u] = *(const uint4*)(row + (size_t)g * 32);
        sv[u] = srow[g];
      }
    }
  };
  load(0, cw, cs);
  for (int i0 = 0; i0 < ni; i0 += U) {
    if (i0 + U < ni) load(i0 + U, nw, nsc);
#pragma unroll
    for (int u = 0; u < U; u++) {
      if (i0 + u >= ni) break;  // wave-uniform
      const int g = gs + KW * (i0 + u);
      const uint32_t ww[4] = {cw[u].x, cw[u].y, cw[u].z, cw[u].w};
      const float scale = __half2float(cs[u]);
      uint32_t own[8], oth[8];
#pragma unroll
      for (int i = 0; i < 4; i++) q8w::dec4(ww[i], own[2 * i], own[2 * i + 1]);
#pragma unroll
      for (int j = 0; j < 8; j++)
        oth[j] = (uint32_t)__builtin_amdgcn_permlanex16((int)own[j], (int)own[j], 0x76543210,
                                                        (int)0xfedcba98, false, false);
      htw_v8f d[NPT];
#pragma unroll
      for (int i = 0; i < NPT; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) d[i][e] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        // K step ks = cols 16ks..16ks+15 of the group: lanes of half ks hold
        // them in own[], the other half received them in oth[]
        uint32_t bw[8];
#pragma unroll
        for (int j = 0; j < 8; j++) bw[j] = (h == ks) ? own[j] : oth[j];
        htw_v16h bf;
        memcpy(&bf, bw, 32);
#pragma unroll
        for (int pt = 0; pt < NPT; pt++) {
          const int p = pt * 16 + qrowA;
          uint32_t aw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
          if (p < P) {
            const TX* xp = x + (size_t)p * xs + (size_t)g * 32 + ks * 16;
            float xv[16];
            hv2::ldx8(xp, xv);
            hv2::ldx8(xp + 8, xv + 8);
#pragma unroll
            for (int j = 0; j < 8; j++) {
              const __half2 hv = __floats2half2_rn(xv[2 * j], xv[2 * j + 1]);
              aw[j] = *(const uint32_t*)&hv;
            }
          }
          htw_v16h af;
          memcpy(&af, aw, 32);
          d[pt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, d[pt]);
        }
      }
#pragma unroll
      for (int pt = 0; pt < NPT; pt++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[pt][e] = fmaf(scale, d[pt][e], acc[pt][e]);
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      cw[u] = nw[u];
      cs[u] = nsc[u];
    }
  }
#pragma unroll
  for (int pt = 0; pt < NPT; pt++)
#pragma unroll
    for (int e = 0; e < 8; e++) red[w][pt * 16 + h * 8 + e][n] = acc[pt][e];
  __syncthreads();
  for (int i = tid; i < RW * P * 16; i += 256) {
    const int nn = i & 15, rest = i >> 4, p = rest % P, rwi = rest / P;
    float v = red[rwi * KW][p][nn];
#pragma unroll
    for (int k = 1; k < KW; k++) v += red[rwi * KW + k][p][nn];
    const int r = (blockIdx.x * RW + rwi) * 16 + nn;
    if (r >= R) continue;
    if (gridDim.y > 1)
      out[((size_t)s * P + p) * R + r] = v;
    else
      out[(size_t)p * ys + r] = v;
  }
}
