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
                                                 const float* __restrict__ suh_dn, int mid,
                                                 int n_routed = 0,
                                                 const float* __restrict__ se_guv = nullptr,
                                                 float* __restrict__ se_hid = nullptr,
                                                 int se_mid = 0) {
  __shared__ float s[2048];
  const int slot = blockIdx.x;
  if (n_routed > 0 && slot >= n_routed) {
    if (se_guv && se_hid && threadIdx.x < 256) {
      const int i = (slot - n_routed) * 256 + threadIdx.x;
      if (i < se_mid) {
        const float a = se_guv[i];
        se_hid[i] = a / (1.f + expf(-a)) * se_guv[se_mid + i];
      }
    }
    return;
  }
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

struct HtGemvDesc {
  const uint8_t* codes;
  const float* suh;
  const float* svh;
  float* y;
  float* zs;
  int* cnt;
  int nty, ks, O, nrows, tpb, tpc, b0;
};
struct HtGemvBatch {
  HtGemvDesc m[4];
  int nm;
  int K;
};

// Multi-matrix P=1 k_ht_gemv<1, float>: runs 2..4 independent projections sharing
// input x in a single launch while preserving each matrix's exact (nty, ks, tpb, tpc)
// tile mapping and deterministic kz reduction order (bit-identical to separate gemvs).
__global__ __launch_bounds__(256) void k_ht_gemv_multi(const float* __restrict__ x,
                                                       HtGemvBatch B) {
  extern __shared__ float hv2_lds[];
  float* xh = hv2_lds;
  float* zl = hv2_lds;
  __shared__ int s_last;
  const int bid = (int)blockIdx.x;
  int mi = 0;
#pragma unroll
  for (int i = 1; i < 4; i++) {
    if (i < B.nm && bid >= B.m[i].b0) mi = i;
  }
  const HtGemvDesc d = B.m[mi];
  const int sb = bid - d.b0;
  const int ty = sb % d.nty, kz = sb / d.nty;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int ntx = B.K >> 7;
  const int tx0 = kz * d.tpb;
  const int nt = min(ntx, tx0 + d.tpb) - tx0;

  float acc[8];
#pragma unroll
  for (int q = 0; q < 8; q++) acc[q] = 0.f;
  const int h = lane >> 4;
  const uint8_t* base = d.codes + ((size_t)ty * ntx + tx0) * 8192 + w * 1024 + lane * 4;
  uint32_t nxt[8];
#pragma unroll
  for (int q = 0; q < 8; q++) nxt[q] = *(const uint32_t*)(base + q * 128);
  for (int t = 0; t < nt; t++) {
    const int tc = t % d.tpc;
    if (tc == 0) {
      const int nc = min(d.tpc, nt - t);
      if (t) __syncthreads();
      for (int u = w; u < nc; u += 8) {
        const int col0 = (tx0 + t + u) * 128;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int i = col0 + j * 32 + lane;
          v[j] = x[i] * d.suh[i];
        }
        hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
        for (int j = 0; j < 4; j++) xh[u * 128 + j * 32 + lane] = v[j] * hv2::kRs128;
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
    float xv[8];
    const float* xt = xh + tc * 128 + w * 16 + h * 8;
    const float4 a = *(const float4*)xt;
    const float4 b = *(const float4*)(xt + 4);
    xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
    xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const uint32_t cur = cw[q];
      const uint32_t rr = (uint32_t)__builtin_amdgcn_mov_dpp((int)cur, 0x121, 0xf, 0xf, false);
      const uint32_t xx = (uint32_t)__builtin_amdgcn_permlanex16((int)rr, (int)rr, 0x76543210,
                                                                 (int)0xfedcba98, false, false);
      const uint32_t prev = (lane & 15) ? rr : xx;
      float wv[8];
#pragma unroll
      for (int e = 0; e < 8; e += 2)
        hv2::ht_val2(__builtin_amdgcn_alignbit(prev, cur, 28 - 4 * e) & 0xffffu,
                     __builtin_amdgcn_alignbit(prev, cur, 24 - 4 * e) & 0xffffu, wv[e], wv[e + 1]);
      float s = acc[q];
#pragma unroll
      for (int e = 0; e < 8; e++) s = fmaf(wv[e], xv[e], s);
      acc[q] = s;
    }
  }
#pragma unroll
  for (int q = 0; q < 8; q++) acc[q] += __shfl_xor(acc[q], 16, 32);
  __syncthreads();
  if (lane < 16)
#pragma unroll
    for (int q = 0; q < 8; q++) hv2_lds[w * 128 + q * 16 + lane] = acc[q];
  __syncthreads();
  if (tid < 128) {
    float s = hv2_lds[tid];
#pragma unroll
    for (int v = 1; v < 8; v++) s += hv2_lds[v * 128 + tid];
    zl[tid] = s;
  }
  __syncthreads();
  if (d.ks > 1) {
    if (tid < 128) d.zs[(size_t)kz * d.O + ty * 128 + tid] = zl[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(d.cnt + ty, 1) == d.ks - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (tid < 128) {
      const size_t o = (size_t)ty * 128 + tid;
      float s = __builtin_nontemporal_load(d.zs + o);
      for (int k2 = 1; k2 < d.ks; k2++)
        s += __builtin_nontemporal_load(d.zs + (size_t)k2 * d.O + o);
      zl[tid] = s;
    }
    if (tid == 0) atomicExch(d.cnt + ty, 0);
    __syncthreads();
  }
  if (w == 0) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = zl[j * 32 + lane];
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = ty * 128 + j * 32 + lane;
      if (r < d.nrows) d.y[r] = v[j] * hv2::kRs128 * d.svh[r];
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

// 11-27 tune (GGUF q6 dense, tools/q6g64_tune_bench.cu): 2 rows per warp
// with next-group prefetch for the mid-size LPR=32 shapes (C in [1024,4096),
// R >= 640). Each row keeps the per-lane group order (g = li; g += LPR) and
// the shfl reduction tree of k_q6g64_gemv, so every row is bit-identical to
// it; the odd-row duplicate read is never stored (as k_q8g32_gemv_mr).
// Measured gfx1151 P=1: qkv 10240x2560 +6.2%, in_z 6144x2560 +4.3%,
// q_proj 12288x2560 +5.2%, shexp gu 640x2560 +10.1%, lm_head +4.7%;
// R < 640 and C >= 4096 stay on k_q6g64_gemv (dispatch in q6g64_gemv_p).
template <int P, typename TX>
__global__ __launch_bounds__(256) void k_q6g64_gemv_r2(const uint8_t* __restrict__ wq,
                                                       const TX* __restrict__ x, uint64_t xs,
                                                       float* __restrict__ y, uint64_t ys, int R,
                                                       int C, int st) {
  constexpr int LPR = 32;
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

// ---- decode prefetch (QWENOX_DECODE_PF) -----------------------------------
// Read-only sweep over up to 12 weight ranges on a side stream so the bytes
// sit in MALL/L2 when the main chain reaches them. Results are XOR-folded and
// only "stored" on an impossible value, so the loads cannot be dropped.
struct DpfRange { const uint4* p; unsigned long long n; };
struct DpfSet { DpfRange r[12]; int nr; };
__global__ void __launch_bounds__(256) k_decode_pf(DpfSet s, unsigned* sink) {
  unsigned acc = 0;
  const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
  for (int r = 0; r < s.nr; r++) {
    const uint4* p = s.r[r].p;
    const unsigned long long n = s.r[r].n;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
      const uint4 v = p[i];
      acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (acc == 0x7f4a7c15u) atomicXor(sink, acc);
}

// ---- HC chain fusion for the P=1 decode step (QWENOX_HC_FUSE) --------------
// Bit-identical to the unfused chain: every value is produced by the same
// lane math and summed in the same order as k_q6g64_gemv_sk + k_q6g64_skred,
// k_gr_write, k_rmsnorm_zc_grouped<<<4,1024>>>, k_silu_scale,
// k_q6g64_gemv<1,float,8> and k_gr_combine<true>.
namespace hcf {
// k_q6g64_gemv_sk<1,float> lane body: row `row`, groups [g0, g1), lane li.
__device__ __forceinline__ float sk_lane(const uint8_t* __restrict__ row, const float* __restrict__ x,
                                         int C, int g0, int g1, int li) {
  const int sub = li & 7, gq = li >> 3;
  float acc = 0.f;
  int g = g0 + gq;
  if (g < g1) {
    uint32_t clw = *(const uint32_t*)(row + g * 32 + sub * 4);
    uint32_t chw = *(const uint32_t*)(row + C / 2 + g * 16 + (sub >> 1) * 4);
    uint32_t csb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
    for (; g < g1; g += 4) {
      uint32_t nlw = 0, nhw = 0, nsb = 0;
      const int gn = g + 4;
      if (gn < g1) {
        nlw = *(const uint32_t*)(row + gn * 32 + sub * 4);
        nhw = *(const uint32_t*)(row + C / 2 + gn * 16 + (sub >> 1) * 4);
        nsb = *(const uint32_t*)(row + 3 * C / 4 + gn * 4);
      }
      const uint32_t lw = clw, hw = chw, sb = csb;
      clw = nlw; chw = nhw; csb = nsb;
      const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
      const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
      float cf[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t lo = (lw >> (4 * e)) & 15u;
        const uint32_t hi = (hw >> (2 * ((sub & 1) * 8 + e))) & 3u;
        cf[e] = (float)(lo | (hi << 4));
      }
      float xv[8];
      hv2::ldx8(x + g * 64 + sub * 8, xv);
      float dx = 0.f, sx = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        dx = fmaf(cf[e], xv[e], dx);
        sx += xv[e];
      }
      // explicit form = what the backend emits for the unfused kernels
      acc += fmaf(sc, dx, bi * sx);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
  return acc;
}
}  // namespace hcf

// inject GEMV (4 x C) + gr_write + (kNorm) grouped zc rmsnorm. grid = branches
// blocks of 1024 threads (block b owns branch b: w4[b] needs row b only, the
// norm of group b needs R[b] only). rh_old is read whole by every block, so
// rh_new must be a different buffer.
template <bool kNorm>
__global__ __launch_bounds__(1024) void k_hcf_wn(const uint8_t* __restrict__ wq, int C, int st,
                                                 int gps, int ns, const float* __restrict__ rh_old,
                                                 float* __restrict__ R, const float* __restrict__ y,
                                                 const float* __restrict__ nw,
                                                 float* __restrict__ rh_new, float* __restrict__ w4,
                                                 int d, float eps) {
  const int b = blockIdx.x, tid = threadIdx.x, wv = tid >> 5, li = tid & 31;
  __shared__ float sp[32];
  __shared__ float sm[1024];
  __shared__ float s_it[5][8][32];
  if (C == 10240 && gps == 32 && ns == 5) {
    if (wv < 20) {
      const int s = wv >> 2, u = wv & 3;
      const int sub = li & 7, gq = li >> 3;
      const uint8_t* row = wq + (size_t)b * st;
#pragma unroll
      for (int p = 0; p < 2; p++) {
        const int it = u * 2 + p;
        const int g = s * 32 + it * 4 + gq;
        const uint32_t lw = *(const uint32_t*)(row + g * 32 + sub * 4);
        const uint32_t hw = *(const uint32_t*)(row + 5120 + g * 16 + (sub >> 1) * 4);
        const uint32_t sb = *(const uint32_t*)(row + 7680 + g * 4);
        const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
        const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
        float cf[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
          const uint32_t lo = (lw >> (4 * e)) & 15u;
          const uint32_t hi = (hw >> (2 * ((sub & 1) * 8 + e))) & 3u;
          cf[e] = (float)(lo | (hi << 4));
        }
        float xv[8];
        hv2::ldx8(rh_old + g * 64 + sub * 8, xv);
        float dx = 0.f, sx = 0.f;
#pragma unroll
        for (int e = 0; e < 8; e++) {
          dx = fmaf(cf[e], xv[e], dx);
          sx += xv[e];
        }
        s_it[s][it][li] = fmaf(sc, dx, bi * sx);
      }
    }
    __syncthreads();
    if (wv < 5) {
      float a = 0.f;
#pragma unroll
      for (int it = 0; it < 8; it++) a += s_it[wv][it][li];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
      if (li == 0) sp[wv] = a;
    }
  } else if (wv < ns) {
    const int g0 = wv * gps, g1 = min(C >> 6, g0 + gps);
    const float a = hcf::sk_lane(wq + (size_t)b * st, rh_old, C, g0, g1, li);
    if (li == 0) sp[wv] = a;
  }
  __syncthreads();
  float w = sp[0];
  for (int s = 1; s < ns; s++) w += sp[s];
  if (tid == 0) w4[b] = w;
  const float s2 = 2.f / (1.f + expf(-w * 0.25f));
  float* Rb = R + (size_t)b * d;
  float rv[4];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int i = tid + k * 1024;
    if (i < d) {
      float v = Rb[i];
      v += s2 * y[i];
      Rb[i] = v;
      rv[k] = v;
      ss += v * v;
    }
  }
  if (!kNorm) return;
  sm[tid] = ss;
  __syncthreads();
#pragma unroll
  for (int off = 512; off >= 32; off >>= 1) {
    if (tid < off) sm[tid] += sm[tid + off];
    __syncthreads();
  }
  if (tid < 32) {
    float v = sm[tid];
    v += __shfl_down(v, 16, 32);
    v += __shfl_down(v, 8, 32);
    v += __shfl_down(v, 4, 32);
    v += __shfl_down(v, 2, 32);
    v += __shfl_down(v, 1, 32);
    if (tid == 0) sm[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(sm[0] / d + eps);
  const float* wg = nw + (size_t)b * d;
  float* og = rh_new + (size_t)b * d;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int i = tid + k * 1024;
    if (i < d) og[i] = rv[k] * inv * (1.f + wg[i]);
  }
}

// k_hcf_wn + fused k_moe_post on waves 10..29 while waves 0..9 compute 2x-parallel sk_lane.
template <bool kNorm>
__global__ __launch_bounds__(1024) void k_hcf_wn_moe(
    const uint8_t* __restrict__ wq, int C, int st, int gps, int ns,
    const float* __restrict__ rh_old, float* __restrict__ R,
    const float* __restrict__ moe_acc, const float* __restrict__ dn_svh,
    const float* __restrict__ se_ey, const float* __restrict__ se_sg,
    const float* __restrict__ nw, float* __restrict__ rh_new,
    float* __restrict__ w4, int d, float eps) {
  const int b = blockIdx.x, tid = threadIdx.x, wv = tid >> 5, li = tid & 31;
  __shared__ float sp[32];
  __shared__ float sm[1024];
  __shared__ float sy[2560];
  __shared__ float s_it[5][8][32];
  const bool fast_sk = (C == 10240 && gps == 32 && ns == 5);
  if (fast_sk && wv < 10) {
    const int s = wv >> 1, u = wv & 1;
    const int sub = li & 7, gq = li >> 3;
    const uint8_t* row = wq + (size_t)b * st;
#pragma unroll
    for (int p = 0; p < 4; p++) {
      const int it = u * 4 + p;
      const int g = s * 32 + it * 4 + gq;
      const uint32_t lw = *(const uint32_t*)(row + g * 32 + sub * 4);
      const uint32_t hw = *(const uint32_t*)(row + 5120 + g * 16 + (sub >> 1) * 4);
      const uint32_t sb = *(const uint32_t*)(row + 7680 + g * 4);
      const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
      const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
      float cf[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t lo = (lw >> (4 * e)) & 15u;
        const uint32_t hi = (hw >> (2 * ((sub & 1) * 8 + e))) & 3u;
        cf[e] = (float)(lo | (hi << 4));
      }
      float xv[8];
      hv2::ldx8(rh_old + g * 64 + sub * 8, xv);
      float dx = 0.f, sx = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        dx = fmaf(cf[e], xv[e], dx);
        sx += xv[e];
      }
      s_it[s][it][li] = fmaf(sc, dx, bi * sx);
    }
  } else if (!fast_sk && wv < ns) {
    const int g0 = wv * gps, g1 = min(C >> 6, g0 + gps);
    const float a = hcf::sk_lane(wq + (size_t)b * st, rh_old, C, g0, g1, li);
    if (li == 0) sp[wv] = a;
  } else if (wv >= 10 && wv < 30) {
    const int eb = wv - 10;
    if (eb < (d >> 7)) {
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; j++) v[j] = moe_acc[eb * 128 + j * 32 + li] * 1.f;
      hv2::fwht128(v[0], v[1], v[2], v[3], li);
      const float s = 1.f / (1.f + expf(-se_sg[0]));
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int i = eb * 128 + j * 32 + li;
        float o = __fmul_rn(__fmul_rn(v[j], hv2::kRs128), dn_svh[i]);
        o = fmaf(s, se_ey[i], o);
        sy[i] = o;
      }
    }
  }
  __syncthreads();
  if (fast_sk) {
    if (wv < 5) {
      float a = 0.f;
#pragma unroll
      for (int it = 0; it < 8; it++) a += s_it[wv][it][li];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) a += __shfl_xor(a, off, 32);
      if (li == 0) sp[wv] = a;
    }
    __syncthreads();
  }
  float w = sp[0];
  for (int s = 1; s < ns; s++) w += sp[s];
  if (tid == 0) w4[b] = w;
  const float s2 = 2.f / (1.f + expf(-w * 0.25f));
  float* Rb = R + (size_t)b * d;
  float rv[4];
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int i = tid + k * 1024;
    if (i < d) {
      float v = Rb[i];
      v += s2 * sy[i];
      Rb[i] = v;
      rv[k] = v;
      ss += v * v;
    }
  }
  if (!kNorm) return;
  sm[tid] = ss;
  __syncthreads();
#pragma unroll
  for (int off = 512; off >= 32; off >>= 1) {
    if (tid < off) sm[tid] += sm[tid + off];
    __syncthreads();
  }
  if (tid < 32) {
    float v = sm[tid];
    v += __shfl_down(v, 16, 32);
    v += __shfl_down(v, 8, 32);
    v += __shfl_down(v, 4, 32);
    v += __shfl_down(v, 2, 32);
    v += __shfl_down(v, 1, 32);
    if (tid == 0) sm[0] = v;
  }
  __syncthreads();
  const float inv = rsqrtf(sm[0] / d + eps);
  const float* wg = nw + (size_t)b * d;
  float* og = rh_new + (size_t)b * d;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int i = tid + k * 1024;
    if (i < d) og[i] = rv[k] * inv * (1.f + wg[i]);
  }
}

// Fast specialized k_hcf_down for R=320, C=10240, gps=32 (ns=5): stages the 2048-float
// x slice and its 256 8-element sums once in LDS and unrolls all 8 group iterations.
__global__ __launch_bounds__(256) void k_hcf_down_c10k(const uint8_t* __restrict__ wq,
                                                       const float* __restrict__ x,
                                                       float* __restrict__ part,
                                                       float* __restrict__ t,
                                                       int st, int* __restrict__ cnt) {
  __shared__ float sxv[2048];
  __shared__ float ssx[256];
  __shared__ int s_last;
  const int tid = threadIdx.x, li = tid & 31;
  const int s = blockIdx.y;
  {
    float xv[8];
    hv2::ldx8(x + s * 2048 + tid * 8, xv);
    *(float4*)(sxv + tid * 8) = make_float4(xv[0], xv[1], xv[2], xv[3]);
    *(float4*)(sxv + tid * 8 + 4) = make_float4(xv[4], xv[5], xv[6], xv[7]);
    float sx = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) sx += xv[e];
    ssx[tid] = sx;
  }
  __syncthreads();
  const int r = blockIdx.x * 8 + (tid >> 5);
  if (r < 320) {
    const int sub = li & 7, gq = li >> 3;
    const uint8_t* row = wq + (size_t)r * st;
    const int g0 = s * 32;
    float acc = 0.f;
#pragma unroll
    for (int it = 0; it < 8; it++) {
      const int gl = it * 4 + gq;
      const int g = g0 + gl;
      const uint32_t lw = *(const uint32_t*)(row + g * 32 + sub * 4);
      const uint32_t hw = *(const uint32_t*)(row + 5120 + g * 16 + (sub >> 1) * 4);
      const uint32_t sb = *(const uint32_t*)(row + 7680 + g * 4);
      const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
      const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
      float cf[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t lo = (lw >> (4 * e)) & 15u;
        const uint32_t hi = (hw >> (2 * ((sub & 1) * 8 + e))) & 3u;
        cf[e] = (float)(lo | (hi << 4));
      }
      const float4 a = *(const float4*)(sxv + gl * 64 + sub * 8);
      const float4 b = *(const float4*)(sxv + gl * 64 + sub * 8 + 4);
      const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      float dx = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) dx = fmaf(cf[e], xv[e], dx);
      acc += fmaf(sc, dx, bi * ssx[gl * 8 + sub]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
    if (li == 0) part[(size_t)s * 320 + r] = acc;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(cnt + blockIdx.x, 1) == 4;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid < 8) {
    const int rr = blockIdx.x * 8 + tid;
    float v = __builtin_nontemporal_load(part + rr);
#pragma unroll
    for (int k = 1; k < 5; k++) v += __builtin_nontemporal_load(part + (size_t)k * 320 + rr);
    const float xx = v * 0.25f;
    t[rr] = xx / (1.f + expf(-xx));
  }
  if (tid == 0) atomicExch(cnt + blockIdx.x, 0);
}

// No-split-K variant for R=320, C=10240, gps=32 (ns=5): 80 blocks x 128 threads
// (4 waves/block, 1 wave = 1 row; 4 blocks/WGP, zero tail wave). Each wave walks
// the 5 K slices in order and accumulates v = slice0 + ... + slice4 with the same
// FP32 add order as k_hcf_down's part write + skred sum -> bit-identical, with no
// part buffer, threadfence or atomic at all.
__global__ __launch_bounds__(128) void k_hcf_down_c10k_nsk(const uint8_t* __restrict__ wq,
                                                           const float* __restrict__ x,
                                                           float* __restrict__ t, int st) {
  const int tid = threadIdx.x, li = tid & 31;
  const int r = blockIdx.x * 4 + (tid >> 5);
  const uint8_t* row = wq + (size_t)r * st;
  float v = hcf::sk_lane(row, x, 10240, 0, 32, li);
#pragma unroll
  for (int s = 1; s < 5; s++) v += hcf::sk_lane(row, x, 10240, s * 32, s * 32 + 32, li);
  if (li == 0) {
    const float xx = v * 0.25f;
    t[r] = xx / (1.f + expf(-xx));
  }
}

// mix-down GEMV (R x C split-K, as k_q6g64_gemv_sk<1,float>) + the skred sum
// (last-arriving K slice of each 8-row tile, slices in order) + silu(v/4).
__global__ __launch_bounds__(256) void k_hcf_down(const uint8_t* __restrict__ wq,
                                                  const float* __restrict__ x,
                                                  float* __restrict__ part, float* __restrict__ t,
                                                  int R, int C, int st, int gps,
                                                  int* __restrict__ cnt) {
  const int tid = threadIdx.x, li = tid & 31;
  const int r = blockIdx.x * 8 + (tid >> 5);
  const int s = blockIdx.y;
  if (r < R) {
    const int g0 = s * gps, g1 = min(C >> 6, g0 + gps);
    const float a = hcf::sk_lane(wq + (size_t)r * st, x, C, g0, g1, li);
    if (li == 0) part[(size_t)s * R + r] = a;
  }
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(cnt + blockIdx.x, 1) == (int)gridDim.y - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid < 8) {
    const int rr = blockIdx.x * 8 + tid;
    if (rr < R) {
      float v = __builtin_nontemporal_load(part + rr);
      for (int k = 1; k < (int)gridDim.y; k++) v += __builtin_nontemporal_load(part + (size_t)k * R + rr);
      const float xx = v * 0.25f;
      t[rr] = xx / (1.f + expf(-xx));
    }
  }
  if (tid == 0) atomicExch(cnt + blockIdx.x, 0);
}

// ---- verify HC fusion (QWENOX_VHC) -----------------------------------------
// gr_mix_b at verify sizes (P <= gemvmr_maxp, dense WMMA HC on) runs
//   k_q6g64_wmma(down 320x10240, split-K ns=3) + k_q6g64_wred + k_silu_scale
//   + k_q6g64_wmma(up 10240x320) + k_gr_combine_b_hc_bf16  (5 launches).
// k_vhc_wmma replays k_q6g64_wmma's per-wave math verbatim (same 16-row WMMA
// tiles, same KW group interleave and partial order) with fused epilogues:
//  MODE 1 (down, RW=1 KW=8): every block stores its slice partial; the
//    last-arriving block of a 16-row tile sums the slices in s order (as
//    k_q6g64_wred) and applies silu(v * 0.25) (as k_silu_scale) -> t [P][R].
//  MODE 2 (up, RW=4 KW=2, C = 320): wave row-tile rw takes rows
//    rw * dd + blockIdx.x * 16, i.e. the 4 branch gates of the same 16
//    columns, so the block finishes k_gr_combine_b_hc_bf16 for them -> x
//    (f32 + bf16). G never leaves LDS.
// XP > 0 (P <= XP): the block's x slice is staged once in LDS as the f16 A
// operand (the same __floats2half2_rn values), and MODE 2 prefetches its
// Rhat values at kernel start. Every output takes the old ops in the old
// order -> bit-identical to the 5-launch chain (tools: _claude_hgn/vhc).
template <int NPT, typename TX, bool XL>
__device__ __forceinline__ void vhc_grp(const uint4 l0, const uint4 l1, const uint4 hp,
                                        const uint32_t sb, const int g, const TX* __restrict__ x,
                                        uint64_t xs, int P, int qrowA, int h,
                                        const float* sxs, int nsl, int g0, htw_v8f* acc,
                                        const uint4* xl, int xlst) {
  const __half2 k1024 = __half2half2(__float2half(1024.f));
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
    const uint32_t lo = h ? lw[2 * ks + 1] : lw[2 * ks];
    const uint32_t hi = hw[ks] >> (16 * h);
    uint32_t own[4], oth[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint32_t c0 = ((lo >> (8 * j)) & 15u) | (((hi >> (4 * j)) & 3u) << 4);
      const uint32_t c1 = ((lo >> (8 * j + 4)) & 15u) | (((hi >> (4 * j + 2)) & 3u) << 4);
      const uint32_t u = (0x6400u | c0) | ((0x6400u | c1) << 16);
      const __half2 v = __hsub2(*(const __half2*)&u, k1024);
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
      if (XL) {
        if (p < P) {
          const uint4* q = xl + (size_t)p * xlst + (g - g0) * 8 + ks * 2;
          const uint4 a0 = q[0], a1 = q[1];
          aw[0] = a0.x; aw[1] = a0.y; aw[2] = a0.z; aw[3] = a0.w;
          aw[4] = a1.x; aw[5] = a1.y; aw[6] = a1.z; aw[7] = a1.w;
        }
      } else if (p < P) {
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

template <int NPT, int RW, int KW, typename TX, int MODE, int XP>
__global__ __launch_bounds__(256) void k_vhc_wmma(const uint8_t* __restrict__ wq,
                                                  const TX* __restrict__ x, uint64_t xs,
                                                  float* __restrict__ out, int R, int C, int st,
                                                  int gps, int P, float* __restrict__ tout,
                                                  int* __restrict__ cnt,
                                                  const uint16_t* __restrict__ rhat, int dd,
                                                  float* __restrict__ xo,
                                                  uint16_t* __restrict__ xo16) {
  static_assert(RW * KW == 8, "8 waves");
  constexpr int XLMAXG = MODE == 2 ? 5 : 54;  // groups per slice (C = 320 / gps = 54)
  constexpr int XLST = XLMAXG * 8 + 1;        // uint4 per staged x row (+16 B pad)
  constexpr int XLN = XP > 0 ? XP * XLST : 1;
  constexpr int REDN = 8 * 16 * NPT * 16;
  constexpr int UN = XP > 0 ? (XLN * 4 > REDN ? XLN * 4 : REDN) : REDN;
  __shared__ float sxs[XP > 0 ? XP * XLMAXG : (MODE == 2 ? 16 * NPT * 8 : 48 * 64)];
  __shared__ __attribute__((aligned(16))) float lds_u[UN];  // staged x, then red
  float(*red)[16 * NPT][16] = (float(*)[16 * NPT][16])lds_u;
  uint4* xl = (uint4*)lds_u;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int rw = w / KW, kw = w % KW;
  const int s = blockIdx.y;
  const int ng = C >> 6;
  const int g0 = s * gps, g1 = min(ng, g0 + gps), nsl = g1 - g0;
  const int n = lane & 15, h = lane >> 4;
  const int r0 = MODE == 2 ? rw * dd + blockIdx.x * 16 : (blockIdx.x * RW + rw) * 16;
  const uint8_t* row = wq + (size_t)(r0 + n) * st;
  uint4 cl0, cl1, chp, nl0, nl1, nhp;
  uint32_t csb = 0, nsb = 0;
  int g = g0 + kw;
  if (g < g1) {
    cl0 = *(const uint4*)(row + g * 32);
    cl1 = *(const uint4*)(row + g * 32 + 16);
    chp = *(const uint4*)(row + C / 2 + g * 16);
    csb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
  }
  uint16_t rh4[4] = {0, 0, 0, 0};
  if constexpr (MODE == 2 && XP > 0) {
    if (tid < P * 16) {
      const uint16_t* Rt = rhat + (size_t)(tid >> 4) * 4 * dd + blockIdx.x * 16 + (tid & 15);
#pragma unroll
      for (int b = 0; b < 4; b++) rh4[b] = Rt[b * dd];
    }
  }
  for (int i = tid; i < P * nsl * 8; i += 256) {
    const int sub = i & 7, pg = i >> 3, p = pg / nsl, gl = pg - p * nsl;
    float xv[8];
    hv2::ldx8(x + (size_t)p * xs + (size_t)(g0 + gl) * 64 + sub * 8, xv);
    if constexpr (XP > 0) {
      uint32_t hv[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const __half2 v2 = __floats2half2_rn(xv[2 * j], xv[2 * j + 1]);
        hv[j] = *(const uint32_t*)&v2;
      }
      xl[(size_t)p * XLST + gl * 8 + sub] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
    }
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
    vhc_grp<NPT, TX, (XP > 0)>(l0, l1, hp, sb, g, x, xs, P, qrowA, h, sxs, nsl, g0, acc, xl,
                               XLST);
  }
  if constexpr (XP > 0) __syncthreads();  // red aliases the staged x
#pragma unroll
  for (int pt = 0; pt < NPT; pt++)
#pragma unroll
    for (int e = 0; e < 8; e++) red[w][pt * 16 + h * 8 + e][n] = acc[pt][e];
  __syncthreads();
  if constexpr (MODE == 1) {
    for (int i = tid; i < RW * P * 16; i += 256) {
      const int nn = i & 15, rest = i >> 4, p = rest % P, rwi = rest / P;
      float v = red[rwi * KW][p][nn];
#pragma unroll
      for (int k = 1; k < KW; k++) v += red[rwi * KW + k][p][nn];
      const int r = (blockIdx.x * RW + rwi) * 16 + nn;
      out[((size_t)s * P + p) * R + r] = v;
    }
    __shared__ int s_last;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(cnt + blockIdx.x, 1) == (int)gridDim.y - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int ns = (int)gridDim.y;
    for (int i = tid; i < RW * P * 16; i += 256) {
      const int nn = i & 15, rest = i >> 4, p = rest % P, rwi = rest / P;
      const int r = (blockIdx.x * RW + rwi) * 16 + nn;
      float v = __builtin_nontemporal_load(out + (size_t)p * R + r);
      for (int k = 1; k < ns; k++)
        v += __builtin_nontemporal_load(out + ((size_t)k * P + p) * R + r);
      const float xx = v * 0.25f;
      tout[(size_t)p * R + r] = xx / (1.f + expf(-xx));
    }
    if (tid == 0) atomicExch(cnt + blockIdx.x, 0);
  } else {
    for (int i = tid; i < RW * P * 16; i += 256) {
      const int nn = i & 15, rest = i >> 4, p = rest % P, rwi = rest / P;
      float v = red[rwi * KW][p][nn];
#pragma unroll
      for (int k = 1; k < KW; k++) v += red[rwi * KW + k][p][nn];
      red[rwi * KW][p][nn] = v;  // G in place (one owner per element)
    }
    __syncthreads();
    for (int i = tid; i < P * 16; i += 256) {
      const int nn = i & 15, p = i >> 4;
      const int c = blockIdx.x * 16 + nn;
      const uint16_t* Rt = rhat + (size_t)p * 4 * dd;
      float a = 0.f;
#pragma unroll
      for (int b = 0; b < 4; b++) {
        float gate = 1.f / (1.f + expf(-red[b * KW][p][nn]));
        a += gate * bf2f(XP > 0 ? rh4[b] : Rt[b * dd + c]);
      }
      const float value = a / 4;
      xo[(size_t)p * dd + c] = value;
      xo16[(size_t)p * dd + c] = f2bf(value);
    }
  }
}

// Fast specialized k_hcf_upc_160 for d=2560, C=320 (ng=5): 320 blocks x 160 threads
// (5 waves/block, 6 blocks = 30 waves/WGP = 93.75% occupancy, 0% dead lanes vs 37.5%
// dead lanes in LPR=8), stages t[320] + ssx[5] in 2 KiB LDS, keeps low VGPR count
// via scalar 32b loads, and preserves the exact shuffle-tree & FMA order of k_hcf_upc.
__global__ __launch_bounds__(160, 6) void k_hcf_upc_160(const uint8_t* __restrict__ wq,
                                                        const float* __restrict__ t,
                                                        const float* __restrict__ rh,
                                                        float* __restrict__ xo, int st) {
  const int tid = threadIdx.x;
  __shared__ float st_sh[320];
  __shared__ float ssx[5];
  __shared__ float sterm[32][5];
  st_sh[tid] = t[tid];
  st_sh[tid + 160] = t[tid + 160];
  __syncthreads();
  if (tid < 5) {
    float sx = 0.f;
#pragma unroll
    for (int k = 0; k < 64; k++) sx += st_sh[tid * 64 + k];
    ssx[tid] = sx;
  }
  __syncthreads();
  {
    const int rr = tid / 5, g = tid - rr * 5;
    const int b = rr >> 3, j = rr & 7;
    const int r = b * 2560 + blockIdx.x * 8 + j;
    const uint8_t* row = wq + (size_t)r * st;
    const uint32_t sb = *(const uint32_t*)(row + 240 + g * 4);
    const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
    const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
    float dx = 0.f;
#pragma unroll 2
    for (int i8 = 0; i8 < 8; i8++) {
      const uint32_t lw = *(const uint32_t*)(row + g * 32 + i8 * 4);
      const uint32_t hw = (uint32_t)*(const uint16_t*)(row + 160 + g * 16 + i8 * 2);
      const float4 a = *(const float4*)(st_sh + g * 64 + i8 * 8);
      const float4 bv = *(const float4*)(st_sh + g * 64 + i8 * 8 + 4);
      const float xv[8] = {a.x, a.y, a.z, a.w, bv.x, bv.y, bv.z, bv.w};
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t lo = (lw >> (4 * e)) & 15u;
        const uint32_t hi = (hw >> (2 * e)) & 3u;
        dx = fmaf((float)(lo | (hi << 4)), xv[e], dx);
      }
    }
    sterm[rr][g] = fmaf(sc, dx, bi * ssx[g]);
  }
  __syncthreads();
  if (tid < 32) {
    float s0 = sterm[tid][0] + sterm[tid][4];
    float s2 = sterm[tid][2];
    float s1 = sterm[tid][1];
    float s3 = sterm[tid][3];
    float G = (s0 + s2) + (s1 + s3);
    float gv = 1.f / (1.f + expf(-G));
    const int b = tid >> 3, j = tid & 7;
    sterm[tid][0] = gv;
    sterm[tid][1] = rh[(size_t)b * 2560 + blockIdx.x * 8 + j];
  }
  __syncthreads();
  if (tid < 8) {
    float a = 0.f;
#pragma unroll
    for (int bb = 0; bb < 4; bb++) {
      a += sterm[bb * 8 + tid][0] * sterm[bb * 8 + tid][1];
    }
    xo[blockIdx.x * 8 + tid] = a / 4;
  }
}

// Coalesced high-wave HC mix-up + combine (QWENOX_HCF_UPC_COAL=1):
// 320 blocks x 256 threads (2560 waves, 32 rows/block, 8 lanes/row, 0% dead lanes).
// For each group g=0..4, 8 consecutive lanes load 32 contiguous bytes of lw
// and 16 contiguous bytes of hw (100% coalesced 32B/16B cacheline sectors!).
__global__ __launch_bounds__(256, 4) void k_hcf_upc_coal(const uint8_t* __restrict__ wq,
                                                         const float* __restrict__ t,
                                                         const float* __restrict__ rh,
                                                         float* __restrict__ xo, int st) {
  const int tid = threadIdx.x;
  __shared__ float st_sh[320];
  __shared__ float ssx[5];
  __shared__ float sG[32];
  st_sh[tid] = t[tid];
  if (tid < 64) st_sh[256 + tid] = t[256 + tid];
  __syncthreads();
  if (tid < 5) {
    float sx = 0.f;
#pragma unroll
    for (int k = 0; k < 64; k++) sx += st_sh[tid * 64 + k];
    ssx[tid] = sx;
  }
  __syncthreads();
  const int rr = tid >> 3, li = tid & 7;
  const int b = rr >> 3, j = rr & 7;
  const int r = b * 2560 + blockIdx.x * 8 + j;
  const uint8_t* row = wq + (size_t)r * st;
  float term[5];
#pragma unroll
  for (int g = 0; g < 5; g++) {
    const uint32_t lw = *(const uint32_t*)(row + g * 32 + li * 4);
    const uint32_t hw = (uint32_t)*(const uint16_t*)(row + 160 + g * 16 + li * 2);
    const float4 a = *(const float4*)(st_sh + g * 64 + li * 8);
    const float4 bv = *(const float4*)(st_sh + g * 64 + li * 8 + 4);
    const float xv[8] = {a.x, a.y, a.z, a.w, bv.x, bv.y, bv.z, bv.w};
    float dx = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const uint32_t lo = (lw >> (4 * e)) & 15u;
      const uint32_t hi = (hw >> (2 * e)) & 3u;
      dx = fmaf((float)(lo | (hi << 4)), xv[e], dx);
    }
    dx += __shfl_xor(dx, 4, 32);
    dx += __shfl_xor(dx, 2, 32);
    dx += __shfl_xor(dx, 1, 32);
    if (li == 0) {
      const uint32_t sb = *(const uint32_t*)(row + 240 + g * 4);
      const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
      const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
      term[g] = fmaf(sc, dx, bi * ssx[g]);
    }
  }
  if (li == 0) {
    float s0 = term[0] + term[4];
    float G = (s0 + term[2]) + (term[1] + term[3]);
    sG[rr] = 1.f / (1.f + expf(-G));
  }
  __syncthreads();
  if (tid < 8) {
    const int i = blockIdx.x * 8 + tid;
    float a = 0.f;
#pragma unroll
    for (int bb = 0; bb < 4; bb++) a += sG[bb * 8 + tid] * rh[(size_t)bb * 2560 + i];
    xo[i] = a / 4;
  }
}

// mix-up GEMV (branches*d rows x C, C < 1024, k_q6g64_gemv<1,float,8> lane
// math) + sigmoid combine. Block = 8 output columns i for all 4 branches.
__global__ __launch_bounds__(256) void k_hcf_upc(const uint8_t* __restrict__ wq,
                                                 const float* __restrict__ t,
                                                 const float* __restrict__ rh,
                                                 float* __restrict__ xo, int d, int C, int st) {
  constexpr int LPR = 8;
  const int tid = threadIdx.x, rr = tid / LPR, li = tid % LPR;
  const int b = rr >> 3, j = rr & 7;
  const int r = b * d + blockIdx.x * 8 + j;
  __shared__ float sG[32];
  const uint8_t* row = wq + (size_t)r * st;
  const int ng = C >> 6;
  float acc = 0.f;
  for (int g = li; g < ng; g += LPR) {
    const uint4 l0 = *(const uint4*)(row + g * 32), l1 = *(const uint4*)(row + g * 32 + 16);
    const uint4 hp = *(const uint4*)(row + C / 2 + g * 16);
    const uint32_t sb = *(const uint32_t*)(row + 3 * C / 4 + g * 4);
    const float sc = __half2float(__ushort_as_half((unsigned short)(sb & 0xffffu)));
    const float bi = __half2float(__ushort_as_half((unsigned short)(sb >> 16)));
    const uint32_t lw[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    const uint32_t hw[4] = {hp.x, hp.y, hp.z, hp.w};
    float dx = 0.f, sx = 0.f;
#pragma unroll
    for (int i8 = 0; i8 < 8; i8++) {
      float cf[8];
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const uint32_t lo = (lw[i8] >> (4 * e)) & 15u;
        const uint32_t hi = (hw[i8 >> 1] >> (2 * ((i8 & 1) * 8 + e))) & 3u;
        cf[e] = (float)(lo | (hi << 4));
      }
      float xv[8];
      hv2::ldx8(t + g * 64 + i8 * 8, xv);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        dx = fmaf(cf[e], xv[e], dx);
        sx += xv[e];
      }
    }
    // explicit form = what the backend emits for the unfused kernels
    acc += fmaf(sc, dx, bi * sx);
  }
#pragma unroll
  for (int off = LPR / 2; off > 0; off >>= 1) acc += __shfl_xor(acc, off, LPR);
  if (li == 0) sG[rr] = acc;
  __syncthreads();
  if (tid < 8) {
    const int i = blockIdx.x * 8 + tid;
    float a = 0.f;
    for (int bb = 0; bb < 4; bb++) {
      float g = sG[bb * 8 + tid];
      g = 1.f / (1.f + expf(-g));
      a += g * rh[(size_t)bb * d + i];
    }
    xo[i] = a / 4;
  }
}

// ---- MoE pre/post fusion for the P=1 decode step (QWENOX_MOE_PP) -----------
// k_moe_pre: grid 64 x 512 (16 waves). Wave w of block b: router row b*8+w/2
// ... same row->wave map as k_bf16_gemv<<<64, 512>>> (one wave per row, 16
// rows per block); blocks 0..4 additionally rotate 4 of the 20 x blocks
// (k_v2_rot<float,false>, pre=suh). The last-arriving block runs the
// k_router_topk_w32<10> body on wave 0.
__global__ __launch_bounds__(512) void k_moe_pre(const uint16_t* __restrict__ rw,
                                                 const float* __restrict__ x, int cols,
                                                 float* __restrict__ logits,
                                                 const float* __restrict__ suh,
                                                 float* __restrict__ xr, int n,
                                                 int* __restrict__ ids, float* __restrict__ ws,
                                                 int* __restrict__ cnt, VizTap vt,
                                                 const uint16_t* __restrict__ seg_w = nullptr,
                                                 float* __restrict__ seg_out = nullptr,
                                                 bool seg_rowfast = false,
                                                 const uint8_t* __restrict__ seg_q4c = nullptr,
                                                 const uint8_t* __restrict__ seg_q4s = nullptr,
                                                 const float* __restrict__ seg_q4cb = nullptr) {
  const int lane = threadIdx.x & 31, wv = threadIdx.x >> 5;
  const int nw = blockDim.x >> 5;
  const int n_rt_blk = 512 / nw;
  if ((seg_w || seg_q4c) && (int)blockIdx.x == n_rt_blk) {
    if (seg_q4c) {
      __shared__ float cbf[16];
      if (threadIdx.x < 16) cbf[threadIdx.x] = seg_q4cb[threadIdx.x];
      __syncthreads();
      if (wv == 0) {
        const uint64_t groups_per_row = (uint64_t)cols / 32;
        float acc0 = 0.f;
        for (uint64_t g = lane; g < groups_per_row; g += 32) {
          uint4 pk0 = *(const uint4*)(seg_q4c + g * 16);
          float s0 = __half2float(*(const __half*)(seg_q4s + g * 2));
          const float* xg = x + g * 32;
          float xv[32];
#pragma unroll
          for (int i = 0; i < 8; i++) {
            float4 t = *(const float4*)(xg + i * 4);
            xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z; xv[i * 4 + 3] = t.w;
          }
          uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
          float a0 = 0.f;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            uint32_t v0 = w0[i];
#pragma unroll
            for (int j = 0; j < 8; j++) {
              a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
              v0 >>= 4;
            }
          }
          acc0 += a0 * s0;
        }
        for (int off = 16; off > 0; off >>= 1)
          acc0 += __shfl_down_sync(~0ull, acc0, off);
        if (lane == 0) seg_out[0] = acc0;
      }
      return;
    }
    if (seg_rowfast) {
      __shared__ float red[8];
      float acc = 0.f;
      if (threadIdx.x < 256) {
        for (uint64_t c = (uint64_t)threadIdx.x * 8; c < (uint64_t)cols; c += 256 * 8) {
          uint4 pk = *(const uint4*)(seg_w + c);
          float4 xa = *(const float4*)(x + c);
          float4 xb = *(const float4*)(x + c + 4);
          const uint32_t pw[4] = {pk.x, pk.y, pk.z, pk.w};
          const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
          for (int j = 0; j < 4; j++) {
            acc += __uint_as_float(pw[j] << 16) * xv[2 * j];
            acc += __uint_as_float(pw[j] & 0xFFFF0000u) * xv[2 * j + 1];
          }
        }
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(~0ull, acc, off);
        if (lane == 0) red[wv] = acc;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        float t = 0.f;
        for (int i = 0; i < 8; i++) t += red[i];
        seg_out[0] = t;
      }
    } else if (wv == 0) {
      float acc = 0.f;
      for (uint64_t c = lane * 8; c < (uint64_t)cols; c += 32 * 8) {
        uint4 pk = *(const uint4*)(seg_w + c);
        float4 xa = *(const float4*)(x + c);
        float4 xb = *(const float4*)(x + c + 4);
        const uint16_t* h = (const uint16_t*)&pk;
        float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int j = 0; j < 8; j++) {
          uint32_t u = (uint32_t)h[j] << 16;
          float wf;
          memcpy(&wf, &u, 4);
          acc += wf * xv[j];
        }
      }
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off, 32);
      if (lane == 0) seg_out[0] = acc;
    }
    return;
  }
  __shared__ float sx[2560];
  const float* xsrc = x;
  if (cols == 2560) {
    for (int i = (int)threadIdx.x * 4; i < 2560; i += (int)blockDim.x * 4)
      *(float4*)(sx + i) = *(const float4*)(x + i);
    __syncthreads();
    xsrc = sx;
  }
  {
    const uint64_t row = (uint64_t)blockIdx.x * nw + wv;
    if (row < 512) {
      const uint16_t* r = rw + row * cols;
      float acc = 0.f;
      if (cols == 2560) {
        uint4 cpk = *(const uint4*)(r + lane * 8);
#pragma unroll
        for (int it = 0; it < 10; it++) {
          uint4 npk = {0, 0, 0, 0};
          if (it + 1 < 10) npk = *(const uint4*)(r + (it + 1) * 256 + lane * 8);
          const uint4 pk = cpk;
          cpk = npk;
          const uint64_t c = (uint64_t)it * 256 + lane * 8;
          const float4 xa = *(const float4*)(xsrc + c);
          const float4 xb = *(const float4*)(xsrc + c + 4);
          const uint16_t* h = (const uint16_t*)&pk;
          const float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
          for (int j = 0; j < 8; j++) {
            uint32_t u = (uint32_t)h[j] << 16;
            float wf;
            memcpy(&wf, &u, 4);
            acc += wf * xv[j];
          }
        }
      } else {
        for (uint64_t c = lane * 8; c < (uint64_t)cols; c += 32 * 8) {
          uint4 pk = *(const uint4*)(r + c);
          float4 xa = *(const float4*)(xsrc + c);
          float4 xb = *(const float4*)(xsrc + c + 4);
          const uint16_t* h = (const uint16_t*)&pk;
          float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
          for (int j = 0; j < 8; j++) {
            uint32_t u = (uint32_t)h[j] << 16;
            float wf;
            memcpy(&wf, &u, 4);
            acc += wf * xv[j];
          }
        }
      }
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off, 32);
      if (lane == 0) logits[row] = acc;
    }
  }
  {
    const int nb = n >> 7;
    const int gw = (int)blockIdx.x * nw + wv;
    if (gw < nb) {
      const int b = gw;
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int i = j * 32 + lane;
        v[j] = xsrc[b * 128 + i] * suh[b * 128 + i];
      }
      hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int i = b * 128 + j * 32 + lane;
        xr[i] = v[j] * hv2::kRs128 * 1.f;
      }
    }
  }
  __shared__ int s_last;
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) s_last = atomicAdd(cnt, 1) == n_rt_blk - 1;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (wv != 0) return;
  if (lane == 0) atomicExch(cnt, 0);
  constexpr int k = 10;
  float v[16], p[16];
#pragma unroll
  for (int j = 0; j < 16; j++) v[j] = __builtin_nontemporal_load(logits + lane + 32 * j);
  float mx = v[0];
#pragma unroll
  for (int j = 1; j < 16; j++) mx = fmaxf(mx, v[j]);
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
#pragma unroll
  for (int j = 0; j < 16; j++) p[j] = expf(v[j] - mx);
  float a[16];
#pragma unroll
  for (int j = 0; j < 16; j++) a[j] = p[j];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] += a[j + 8];
#pragma unroll
  for (int j = 0; j < 4; j++) a[j] += a[j + 4];
#pragma unroll
  for (int j = 0; j < 2; j++) a[j] += a[j + 2];
  a[0] += a[1];
  float s = a[0];
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_down(s, o, 32);
  s = __shfl(s, 0, 32);
  const float inv_se = 1.f / s;
#pragma unroll
  for (int j = 0; j < 16; j++) p[j] *= inv_se;
  float vz_ent = 0.f;
  if (vt.ring) {
    float e = 0.f;
#pragma unroll
    for (int j = 0; j < 16; j++)
      if (p[j] > 0.f) e -= p[j] * __logf(p[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) e += __shfl_xor(e, o, 32);
    vz_ent = e;
  }
  float raw[k];
#pragma unroll
  for (int r = 0; r < k; r++) {
    float bv = p[0];
    int bi = lane;
#pragma unroll
    for (int j = 1; j < 16; j++)
      if (p[j] > bv) { bv = p[j]; bi = lane + 32 * j; }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor(bv, o, 32);
      const int oi = __shfl_xor(bi, o, 32);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    raw[r] = bv;
    if (lane == 0) ids[r] = bi;
    if ((bi & 31) == lane) {
#pragma unroll
      for (int j = 0; j < 16; j++)
        if (j == (bi >> 5)) p[j] = -1e30f;
    }
  }
  if (lane == 0) {
    float sw = 0.f;
#pragma unroll
    for (int j = 0; j < k; j++) sw += raw[j];
#pragma unroll
    for (int j = 0; j < k; j++) ws[j] = raw[j] / sw;
    if (vt.ring) viz::store_route(vt, 0, ids, ws, k, sw, vz_ent);
  }
}

// ============================================================================
__global__ __launch_bounds__(256) void k_moe_pre_segu(
    const uint16_t* __restrict__ rw, const float* __restrict__ x, int cols,
    float* __restrict__ logits, const float* __restrict__ gu_suh,
    float* __restrict__ xr, int n, int* __restrict__ ids, float* __restrict__ ws,
    int* __restrict__ rt_cnt, VizTap vt,
    const uint8_t* __restrict__ seg_data, float* __restrict__ seg_out,
    int se_nty, int se_nty0, int se_tpb, int se_tpc,
    const uint8_t* __restrict__ sg_codes, const float* __restrict__ sg_suh,
    const float* __restrict__ sg_svh, float* __restrict__ sg_y, int sg_nrows,
    const uint8_t* __restrict__ su_codes, const float* __restrict__ su_suh,
    const float* __restrict__ su_svh, float* __restrict__ su_y, int su_nrows,
    float* __restrict__ ht_zs, int* __restrict__ ht_cnt) {
  extern __shared__ __align__(16) unsigned char sh_raw[];
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, wv = tid >> 5;
  const int bid = blockIdx.x;

  if (bid < 64) {
    const uint64_t row = (uint64_t)bid * 8 + wv;
    if (row < 512) {
      const uint16_t* r = rw + row * cols;
      float acc = 0.f;
      for (uint64_t c = lane * 8; c < (uint64_t)cols; c += 32 * 8) {
        uint4 pk = *(const uint4*)(r + c);
        float4 xa = *(const float4*)(x + c);
        float4 xb = *(const float4*)(x + c + 4);
        const uint16_t* h = (const uint16_t*)&pk;
        float xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
#pragma unroll
        for (int j = 0; j < 8; j++) {
          uint32_t u = (uint32_t)h[j] << 16;
          float wf;
          memcpy(&wf, &u, 4);
          acc += wf * xv[j];
        }
      }
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off, 32);
      if (lane == 0) logits[row] = acc;
    }
    const int nb = n >> 7;
    const int gw = bid * 8 + wv;
    if (gw < nb) {
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int i = j * 32 + lane;
        v[j] = x[gw * 128 + i] * gu_suh[gw * 128 + i];
      }
      hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int i = gw * 128 + j * 32 + lane;
        xr[i] = v[j] * hv2::kRs128 * 1.f;
      }
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(rt_cnt, 1) == 63);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (wv != 0) return;
    if (lane == 0) atomicExch(rt_cnt, 0);
    constexpr int k = 10;
    float v[16], p[16];
#pragma unroll
    for (int j = 0; j < 16; j++) v[j] = __builtin_nontemporal_load(logits + lane + 32 * j);
    float mx = v[0];
#pragma unroll
    for (int j = 1; j < 16; j++) mx = fmaxf(mx, v[j]);
#pragma unroll
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
#pragma unroll
    for (int j = 0; j < 16; j++) p[j] = expf(v[j] - mx);
    float a[16];
#pragma unroll
    for (int j = 0; j < 16; j++) a[j] = p[j];
#pragma unroll
    for (int j = 0; j < 8; j++) a[j] += a[j + 8];
#pragma unroll
    for (int j = 0; j < 4; j++) a[j] += a[j + 4];
#pragma unroll
    for (int j = 0; j < 2; j++) a[j] += a[j + 2];
    a[0] += a[1];
    float s = a[0];
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_down(s, o, 32);
    s = __shfl(s, 0, 32);
    const float inv_se = 1.f / s;
#pragma unroll
    for (int j = 0; j < 16; j++) p[j] *= inv_se;
    float vz_ent = 0.f;
    if (vt.ring) {
      float e = 0.f;
#pragma unroll
      for (int j = 0; j < 16; j++)
        if (p[j] > 0.f) e -= p[j] * __logf(p[j]);
#pragma unroll
      for (int o = 16; o; o >>= 1) e += __shfl_xor(e, o, 32);
      vz_ent = e;
    }
    float raw[k];
#pragma unroll
    for (int r = 0; r < k; r++) {
      float bv = p[0];
      int bi = lane;
#pragma unroll
      for (int j = 1; j < 16; j++)
        if (p[j] > bv) { bv = p[j]; bi = lane + 32 * j; }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor(bv, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
      }
      raw[r] = bv;
      if (lane == 0) ids[r] = bi;
      if ((bi & 31) == lane) {
#pragma unroll
        for (int j = 0; j < 16; j++)
          if (j == (bi >> 5)) p[j] = -1e30f;
      }
    }
    if (lane == 0) {
      float sw = 0.f;
#pragma unroll
      for (int j = 0; j < k; j++) sw += raw[j];
#pragma unroll
      for (int j = 0; j < k; j++) ws[j] = raw[j] / sw;
      if (vt.ring) viz::store_route(vt, 0, ids, ws, k, sw, vz_ent);
    }
    return;
  }

  if (bid == 64) {
    float* cbf = (float*)sh_raw;
    const float* cb = (const float*)seg_data;
    if (tid < 16) cbf[tid] = cb[tid];
    __syncthreads();
    if (wv != 0) return;
    const uint8_t* rc0 = seg_data + 64;
    const uint8_t* rs0 = seg_data + 64 + (cols / 2);
    const uint64_t groups_per_row = (uint64_t)cols / 32;
    float acc0 = 0.f;
    for (uint64_t g = lane; g < groups_per_row; g += 32) {
      uint4 pk0 = *(const uint4*)(rc0 + g * 16);
      float s0 = __half2float(*(const __half*)(rs0 + g * 2));
      const float* xg = x + g * 32;
      float xv[32];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float4 t = *(const float4*)(xg + i * 4);
        xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z; xv[i * 4 + 3] = t.w;
      }
      uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
      float a0 = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t v0 = w0[i];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
          v0 >>= 4;
        }
      }
      acc0 += a0 * s0;
    }
    for (int off = 16; off > 0; off >>= 1) acc0 += __shfl_down_sync(~0ull, acc0, off);
    if (lane == 0) seg_out[0] = acc0;
    return;
  }

  const int sb = bid - 65;
  const int ty = sb % se_nty;
  const int kz = sb / se_nty;
  const int ks_total = ((int)gridDim.x - 65) / se_nty;
  __half* xh = (__half*)sh_raw;
  float* zl = (float*)sh_raw;
  const bool sec = ty >= se_nty0;
  const int tyl = sec ? ty - se_nty0 : ty;
  const uint8_t* codes = sec ? su_codes : sg_codes;
  const float* suh = sec ? su_suh : sg_suh;
  const float* svh = sec ? su_svh : sg_svh;
  float* y = sec ? su_y : sg_y;
  const int nrows = sec ? su_nrows : sg_nrows;
  const int O = se_nty * 128;

  const int ntx = cols >> 7;
  const int tx0 = kz * se_tpb;
  const int nt = min(ntx, tx0 + se_tpb) - tx0;
  const int b_ = lane >> 1;
  const int qrowA = (lane & 1) ? ((lane & 16) ? b_ : ((b_ + 4) & 7))
                               : ((lane & 16) ? 8 + ((b_ + 4) & 7) : b_);
  htw_v8f acc;
#pragma unroll
  for (int e = 0; e < 8; e++) acc[e] = 0.f;
  const uint8_t* base = codes + ((size_t)tyl * ntx + tx0) * 8192 + wv * 128 + lane * 4;
  uint32_t nxt[8];
#pragma unroll
  for (int g = 0; g < 8; g++) nxt[g] = *(const uint32_t*)(base + g * 1024);
  for (int t = 0; t < nt; t++) {
    const int tc = t % se_tpc;
    if (tc == 0) {
      const int nc = min(se_tpc, nt - t);
      if (t) __syncthreads();
      for (int c = wv; c < nc; c += 8) {
        __half* dst = xh + c * 128;
        const int col0 = (tx0 + t + c) * 128;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int i = col0 + j * 32 + lane;
          v[j] = x[i] * suh[i];
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
      const __half* ap = xh + tc * 128 + g * 16;
      const uint4 lo = *(const uint4*)ap, hi4 = *(const uint4*)(ap + 8);
      htw_v8u av = (htw_v8u){lo.x, lo.y, lo.z, lo.w, hi4.x, hi4.y, hi4.z, hi4.w};
      if (qrowA >= 1) av = (htw_v8u)(0u);
      acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32((htw_v16h)av, bf, acc);
    }
  }
  __syncthreads();
#pragma unroll
  for (int e = 0; e < 8; e++) {
    const int p = (lane >> 4) * 8 + e;
    if (p < 1) zl[wv * 16 + (lane & 15)] = acc[e];
  }
  __syncthreads();
  if (ks_total > 1) {
    for (int i = tid; i < 128; i += 256)
      ht_zs[(size_t)kz * O + ty * 128 + i] = zl[i];
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(ht_cnt + ty, 1) == ks_total - 1);
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    for (int i = tid; i < 128; i += 256) {
      const size_t o = (size_t)ty * 128 + i;
      float s = __builtin_nontemporal_load(ht_zs + o);
      for (int k2 = 1; k2 < ks_total; k2++)
        s += __builtin_nontemporal_load(ht_zs + (size_t)k2 * O + o);
      zl[i] = s;
    }
    if (tid == 0) atomicExch(ht_cnt + ty, 0);
    __syncthreads();
  }
  if (wv == 0) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = zl[j * 32 + lane];
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = tyl * 128 + j * 32 + lane;
      if (r < nrows) y[r] = v[j] * hv2::kRs128 * svh[r];
    }
  }
}

// k_moe_post: out[i] = rot_out(acc, svh)[i] + sigmoid(sg) * ey[i]. One wave per
// 128 block (k_v2_rot<float,false> post=svh, pre=null), then the
// k_axpy_sg_out expression on the rotated value.
__global__ __launch_bounds__(256) void k_moe_post(const float* __restrict__ acc,
                                                  const float* __restrict__ svh,
                                                  const float* __restrict__ ey,
                                                  const float* __restrict__ sg,
                                                  float* __restrict__ out, int n) {
  const int lane = threadIdx.x & 31;
  const int b = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (b >= (n >> 7)) return;
  float v[4];
#pragma unroll
  for (int j = 0; j < 4; j++) v[j] = acc[b * 128 + j * 32 + lane] * 1.f;
  hv2::fwht128(v[0], v[1], v[2], v[3], lane);
  const float s = 1.f / (1.f + expf(-sg[0]));
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const int i = b * 128 + j * 32 + lane;
    // no contraction across the two original kernels' store boundary
    float o = __fmul_rn(__fmul_rn(v[j], hv2::kRs128), svh[i]);
    o = fmaf(s, ey[i], o);
    out[i] = o;
  }
}

// Fused P=1 routed down_proj (k_q4cp_gemv_gd_topk_h16<10, 2>, blocks 0..dn_blocks-1)
// + shared_expert down_proj (k_ht_gemv<1, float>, blocks dn_blocks..end).
// Both branches run at 256 threads/block and <=4 KiB LDS (zero occupancy loss).
__global__ __launch_bounds__(256) void k_moe_dn_se(
    int dn_blocks, const uint8_t* __restrict__ dn_codes,
    const uint8_t* __restrict__ dn_scales, const float* __restrict__ dn_cb,
    const float* __restrict__ dn_xs, float* __restrict__ dn_out,
    const int* __restrict__ ids, const float* __restrict__ ws,
    uint64_t rows_per, uint64_t dn_cols, uint64_t dn_scale_stride,
    uint64_t x_stride,
    const uint8_t* __restrict__ se_codes, const float* __restrict__ se_suh,
    const float* __restrict__ se_svh, const float* __restrict__ se_x,
    float* __restrict__ se_y, int se_nty, int se_ks, int se_O, int se_K,
    int se_nrows, int se_tpb, int se_tpc, float* __restrict__ zs,
    int* __restrict__ cnt) {
  extern __shared__ float hv2_lds[];
  if ((int)blockIdx.x < dn_blocks) {
    if (threadIdx.x < 16) hv2_lds[threadIdx.x] = dn_cb[threadIdx.x];
    __syncthreads();
    const uint64_t row_pairs = (rows_per + 1) / 2;
    const uint64_t gw = (uint64_t)blockIdx.x * 8 + (threadIdx.x >> 5);
    if (gw >= row_pairs) return;
    const int half = (threadIdx.x & 31) >> 4;
    const int lane = threadIdx.x & 15;
    const uint64_t r = gw * 2 + half;
    if (r >= rows_per) return;
    float acc[10] = {};
#pragma unroll
    for (int s = 0; s < 10; s++) {
      const uint64_t row = (uint64_t)ids[s] * rows_per + r;
      const uint8_t* rc = dn_codes + row * (dn_cols / 2);
      const uint8_t* rs = dn_scales + row * dn_scale_stride;
      const float* x = dn_xs + (size_t)s * x_stride;
      for (uint64_t g = lane; g < dn_cols / 32; g += 16) {
        const uint4 pk = *(const uint4*)(rc + g * 16);
        const float sc = __half2float(*(const __half*)(rs + (g >> 2) * 2));
        const float* xg = x + g * 32;
        float xv[32];
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const float4 t = *(const float4*)(xg + i * 4);
          xv[i * 4] = t.x;
          xv[i * 4 + 1] = t.y;
          xv[i * 4 + 2] = t.z;
          xv[i * 4 + 3] = t.w;
        }
        const uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          uint32_t v = w[i];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            a += hv2_lds[v & 0xf] * xv[i * 8 + j];
            v >>= 4;
          }
        }
        acc[s] += a * sc;
      }
    }
#pragma unroll
    for (int s = 0; s < 10; s++)
      for (int off = 8; off > 0; off >>= 1)
        acc[s] += __shfl_down_sync(~0ull, acc[s], off, 16);
    if (lane == 0) {
      float y = 0.f;
#pragma unroll
      for (int s = 0; s < 10; s++) y += ws[s] * acc[s];
      dn_out[r] = y;
    }
    return;
  }

  float* xh = hv2_lds;
  float* zl = hv2_lds;
  __shared__ int s_last;
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const int sb = (int)blockIdx.x - dn_blocks;
  const int ty = sb % se_nty, kz = sb / se_nty;
  const int ntx = se_K >> 7;
  const int tx0 = kz * se_tpb;
  const int nt = min(ntx, tx0 + se_tpb) - tx0;

  float acc[8];
#pragma unroll
  for (int q = 0; q < 8; q++) acc[q] = 0.f;
  const int h = lane >> 4;
  const uint8_t* base = se_codes + ((size_t)ty * ntx + tx0) * 8192 + w * 1024 + lane * 4;
  uint32_t nxt[8];
#pragma unroll
  for (int q = 0; q < 8; q++) nxt[q] = *(const uint32_t*)(base + q * 128);
  for (int t = 0; t < nt; t++) {
    const int tc = t % se_tpc;
    if (tc == 0) {
      const int nc = min(se_tpc, nt - t);
      if (t) __syncthreads();
      for (int u = w; u < nc; u += 8) {
        const int col0 = (tx0 + t + u) * 128;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const int i = col0 + j * 32 + lane;
          v[j] = se_x[i] * se_suh[i];
        }
        hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
        for (int j = 0; j < 4; j++) xh[u * 128 + j * 32 + lane] = v[j] * hv2::kRs128;
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
    float xv[8];
    const float* xt = xh + tc * 128 + w * 16 + h * 8;
    const float4 a = *(const float4*)xt;
    const float4 b = *(const float4*)(xt + 4);
    xv[0] = a.x; xv[1] = a.y; xv[2] = a.z; xv[3] = a.w;
    xv[4] = b.x; xv[5] = b.y; xv[6] = b.z; xv[7] = b.w;
#pragma unroll
    for (int q = 0; q < 8; q++) {
      const uint32_t cur = cw[q];
      const uint32_t rr = (uint32_t)__builtin_amdgcn_mov_dpp((int)cur, 0x121, 0xf, 0xf, false);
      const uint32_t xx = (uint32_t)__builtin_amdgcn_permlanex16((int)rr, (int)rr, 0x76543210,
                                                                 (int)0xfedcba98, false, false);
      const uint32_t prev = (lane & 15) ? rr : xx;
      float wv[8];
#pragma unroll
      for (int e = 0; e < 8; e += 2)
        hv2::ht_val2(__builtin_amdgcn_alignbit(prev, cur, 28 - 4 * e) & 0xffffu,
                     __builtin_amdgcn_alignbit(prev, cur, 24 - 4 * e) & 0xffffu, wv[e], wv[e + 1]);
      float s = acc[q];
#pragma unroll
      for (int e = 0; e < 8; e++) s = fmaf(wv[e], xv[e], s);
      acc[q] = s;
    }
  }
#pragma unroll
  for (int q = 0; q < 8; q++) acc[q] += __shfl_xor(acc[q], 16, 32);
  __syncthreads();
  if (lane < 16)
#pragma unroll
    for (int q = 0; q < 8; q++) hv2_lds[w * 128 + q * 16 + lane] = acc[q];
  __syncthreads();
  if (tid < 128) {
    float s = hv2_lds[tid];
#pragma unroll
    for (int v = 1; v < 8; v++) s += hv2_lds[v * 128 + tid];
    zl[tid] = s;
  }
  __syncthreads();
  if (se_ks > 1) {
    if (tid < 128) zs[(size_t)kz * se_O + ty * 128 + tid] = zl[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(cnt + ty, 1) == se_ks - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (tid < 128) {
      const size_t o = (size_t)ty * 128 + tid;
      float s = __builtin_nontemporal_load(zs + o);
      for (int k2 = 1; k2 < se_ks; k2++)
        s += __builtin_nontemporal_load(zs + (size_t)k2 * se_O + o);
      zl[tid] = s;
    }
    if (tid == 0) atomicExch(cnt + ty, 0);
    __syncthreads();
  }
  if (w == 0) {
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) v[j] = zl[j * 32 + lane];
    hv2::fwht128(v[0], v[1], v[2], v[3], lane);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int r = ty * 128 + j * 32 + lane;
      if (r < se_nrows) se_y[r] = v[j] * hv2::kRs128 * se_svh[r];
    }
  }
}
