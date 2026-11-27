// Standalone kernel microbench for the gfx1151 HGN v2 decode kernels.
// Extracted verbatim from src/gpu/parts/*.inc; synthetic data; bit-exact
// checks between variants + per-kernel timing/GB-s. No model needed.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                            \
  do {                                                                   \
    hipError_t e = (x);                                                  \
    if (e != hipSuccess) {                                               \
      fprintf(stderr, "CK fail %s:%d %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(e));                                     \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

namespace hv2 {
__device__ __forceinline__ void ldx8(const float* x, float* o) {
  const float4 a = *(const float4*)x, b = *(const float4*)(x + 4);
  o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
  o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
}
}  // namespace hv2

namespace hcf {
__device__ __forceinline__ float sk_lane(const uint8_t* __restrict__ row,
                                         const float* __restrict__ x, int C, int g0,
                                         int g1, int li) {
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
      acc += fmaf(sc, dx, bi * sx);
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off, 32);
  return acc;
}
}  // namespace hcf

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
      for (int k = 1; k < (int)gridDim.y; k++)
        v += __builtin_nontemporal_load(part + (size_t)k * R + rr);
      const float xx = v * 0.25f;
      t[rr] = xx / (1.f + expf(-xx));
    }
  }
  if (tid == 0) atomicExch(cnt + blockIdx.x, 0);
}

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

// old scalar k_hcf_upc_160 (git HEAD version)
__global__ __launch_bounds__(160, 6) void k_hcf_upc_160_old(const uint8_t* __restrict__ wq,
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
      const uint32_t hw = (uint32_t) * (const uint16_t*)(row + 160 + g * 16 + i8 * 2);
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

// new uint4 k_hcf_upc_160
__global__ __launch_bounds__(160, 6) void k_hcf_upc_160_new(const uint8_t* __restrict__ wq,
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
    const uint4 l0 = *(const uint4*)(row + g * 32);
    const uint4 l1 = *(const uint4*)(row + g * 32 + 16);
    const uint4 hp = *(const uint4*)(row + 160 + g * 16);
    const uint32_t lw8[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
    const uint32_t hw4[4] = {hp.x, hp.y, hp.z, hp.w};
    float dx = 0.f;
#pragma unroll
    for (int i8 = 0; i8 < 8; i8++) {
      const uint32_t lw = lw8[i8];
      const uint32_t hw = (hw4[i8 >> 1] >> (16 * (i8 & 1))) & 0xffffu;
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

// ---- GDN kernels (from 20_kernels_gemv.inc) --------------------------------
__global__ __launch_bounds__(128, 8) void k_gdn_step_gn_128(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ S,
    const float* __restrict__ z, const float* __restrict__ w_gn,
    float* __restrict__ ytmp, float* __restrict__ out,
    int* __restrict__ cnt, float eps) {
  const int h = blockIdx.x >> 2, bh = blockIdx.x & 3;
  const int t = threadIdx.x;
  const int il = t & 31, jc = t >> 5;
  const int i = bh * 32 + il, j0 = jc * 32;
  __shared__ float skh[128];
  __shared__ float sqh[128];
  __shared__ float part[4][32];
  __shared__ int s_last;
  skh[t] = k[(h / 3) * 128 + t];
  sqh[t] = q[(h / 3) * 128 + t];
  __syncthreads();

  float* Sh = S + (size_t)h * 128 * 128;
  const float alpha = expf(g[h]);
  const float be = beta[h];
  float s_reg[32];
  float kv = 0.f;
#pragma unroll
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float s = Sh[j * 128 + i] * alpha;
    asm volatile("" : "+v"(s));
    s_reg[u] = s;
    kv += s * skh[j];
  }
  part[jc][il] = kv;
  __syncthreads();
  const float kvf = part[0][il] + part[1][il] + part[2][il] + part[3][il];
  const float delta = (v[h * 128 + i] - kvf) * be;
  float o = 0.f;
#pragma unroll
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float s = s_reg[u] + skh[j] * delta;
    Sh[j * 128 + i] = s;
    o += s * sqh[j];
  }
  __syncthreads();
  part[jc][il] = o;
  __syncthreads();
  if (jc == 0) ytmp[h * 128 + i] = part[0][il] + part[1][il] + part[2][il] + part[3][il];
  __threadfence();
  __syncthreads();
  if (t == 0) s_last = atomicAdd(cnt + h, 1) == 3;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  const float yv = __builtin_nontemporal_load(ytmp + h * 128 + t);
  skh[t] = __fmul_rn(yv, yv);
  __syncthreads();
  if (t < 64) skh[t] += skh[t + 64];
  __syncthreads();
  if (t < 32) {
    float sv = skh[t] + skh[t + 32];
    sv += __shfl_down(sv, 16, 32);
    sv += __shfl_down(sv, 8, 32);
    sv += __shfl_down(sv, 4, 32);
    sv += __shfl_down(sv, 2, 32);
    sv += __shfl_down(sv, 1, 32);
    if (t == 0) skh[0] = sv;
  }
  if (t == 0) atomicExch(cnt + h, 0);
  __syncthreads();
  const float inv = rsqrtf(skh[0] / 128 + eps);
  const float zz = z[h * 128 + t];
  out[h * 128 + t] = yv * inv * w_gn[t] / (1.f + expf(-zz));
}

__global__ __launch_bounds__(128, 8) void k_gdn_step_gn_yt(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ S,
    const float* __restrict__ z, float* __restrict__ ytmp) {
  const int h = blockIdx.x >> 2, bh = blockIdx.x & 3;
  const int t = threadIdx.x;
  const int il = t & 31, jc = t >> 5;
  const int i = bh * 32 + il, j0 = jc * 32;
  __shared__ float skh[128];
  __shared__ float sqh[128];
  __shared__ float part[4][32];
  skh[t] = k[(h / 3) * 128 + t];
  sqh[t] = q[(h / 3) * 128 + t];
  __syncthreads();

  float* Sh = S + (size_t)h * 128 * 128;
  const float alpha = expf(g[h]);
  const float be = beta[h];
  float s_reg[32];
  float kv = 0.f;
#pragma unroll
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float s = Sh[j * 128 + i] * alpha;
    asm volatile("" : "+v"(s));
    s_reg[u] = s;
    kv += s * skh[j];
  }
  part[jc][il] = kv;
  __syncthreads();
  const float kvf = part[0][il] + part[1][il] + part[2][il] + part[3][il];
  const float delta = (v[h * 128 + i] - kvf) * be;
  float o = 0.f;
#pragma unroll
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float s = s_reg[u] + skh[j] * delta;
    Sh[j * 128 + i] = s;
    o += s * sqh[j];
  }
  __syncthreads();
  part[jc][il] = o;
  __syncthreads();
  if (jc == 0) ytmp[h * 128 + i] = part[0][il] + part[1][il] + part[2][il] + part[3][il];
}

__global__ __launch_bounds__(128) void k_gdn_norm_128(const float* __restrict__ ytmp,
                                                      const float* __restrict__ z,
                                                      const float* __restrict__ w_gn,
                                                      float* __restrict__ out, float eps) {
  const int h = blockIdx.x;
  const int t = threadIdx.x;
  __shared__ float sred[128];
  const float yv = __builtin_nontemporal_load(ytmp + h * 128 + t);
  sred[t] = __fmul_rn(yv, yv);
  __syncthreads();
  if (t < 64) sred[t] += sred[t + 64];
  __syncthreads();
  if (t < 32) {
    float sv = sred[t] + sred[t + 32];
    sv += __shfl_down(sv, 16, 32);
    sv += __shfl_down(sv, 8, 32);
    sv += __shfl_down(sv, 4, 32);
    sv += __shfl_down(sv, 2, 32);
    sv += __shfl_down(sv, 1, 32);
    if (t == 0) sred[0] = sv;
  }
  __syncthreads();
  const float inv = rsqrtf(sred[0] / 128 + eps);
  const float zz = z[h * 128 + t];
  out[h * 128 + t] = yv * inv * w_gn[t] / (1.f + expf(-zz));
}

__global__ __launch_bounds__(256) void k_gdn_step_gn_f2(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ S,
    const float* __restrict__ z, const float* __restrict__ w_gn,
    float* __restrict__ out, float eps) {
  const int h = blockIdx.x;
  const int t = threadIdx.x;
  const int i2 = (t & 63) * 2, jc = t >> 6;
  const int j0 = jc * 32;
  __shared__ float skh[128];
  __shared__ float sqh[128];
  __shared__ float part[4][128];
  __shared__ float sdelta[128];
  if (t < 128) {
    skh[t] = k[(h / 3) * 128 + t];
    sqh[t] = q[(h / 3) * 128 + t];
  }
  __syncthreads();

  float* Sh = S + (size_t)h * 128 * 128;
  const float alpha = expf(g[h]);
  const float be = beta[h];
  float2 sr[32];
  float kvx = 0.f, kvy = 0.f;
#pragma unroll
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float2 s = *(const float2*)(Sh + j * 128 + i2);
    s.x *= alpha;
    s.y *= alpha;
    asm volatile("" : "+v"(s.x), "+v"(s.y));
    sr[u] = s;
    kvx += s.x * skh[j];
    kvy += s.y * skh[j];
  }
  part[jc][i2] = kvx;
  part[jc][i2 + 1] = kvy;
  __syncthreads();
  const float kvfx = part[0][i2] + part[1][i2] + part[2][i2] + part[3][i2];
  const float kvfy =
      part[0][i2 + 1] + part[1][i2 + 1] + part[2][i2 + 1] + part[3][i2 + 1];
  const float dx = (v[h * 128 + i2] - kvfx) * be;
  const float dy = (v[h * 128 + i2 + 1] - kvfy) * be;
  float ox = 0.f, oy = 0.f;
#pragma unroll
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float2 s;
    s.x = sr[u].x + skh[j] * dx;
    s.y = sr[u].y + skh[j] * dy;
    *(float2*)(Sh + j * 128 + i2) = s;
    ox += s.x * sqh[j];
    oy += s.y * sqh[j];
  }
  __syncthreads();
  part[jc][i2] = ox;
  part[jc][i2 + 1] = oy;
  __syncthreads();
  if (t < 128) {
    const float yv = part[0][t] + part[1][t] + part[2][t] + part[3][t];
    sdelta[t] = __fmul_rn(yv, yv);
  }
  __syncthreads();
  if (t < 64) sdelta[t] += sdelta[t + 64];
  __syncthreads();
  if (t < 32) {
    float sv = sdelta[t] + sdelta[t + 32];
    sv += __shfl_down(sv, 16, 32);
    sv += __shfl_down(sv, 8, 32);
    sv += __shfl_down(sv, 4, 32);
    sv += __shfl_down(sv, 2, 32);
    sv += __shfl_down(sv, 1, 32);
    if (t == 0) sdelta[0] = sv;
  }
  __syncthreads();
  if (t < 128) {
    const float yv = part[0][t] + part[1][t] + part[2][t] + part[3][t];
    const float inv = rsqrtf(sdelta[0] / 128 + eps);
    const float zz = z[h * 128 + t];
    out[h * 128 + t] = yv * inv * w_gn[t] / (1.f + expf(-zz));
  }
}

// ---- bench harness ----------------------------------------------------------
static float g_hash = 12345.f;
static float frand() {
  g_hash = g_hash * 0.749874883f + 0.532165861f;
  return g_hash - floorf(g_hash);
}

template <typename F>
static double time_kernel(const char* name, int reps, double bytes, F&& fn,
                            hipStream_t st) {
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  for (int i = 0; i < 5; i++) fn();
  CK(hipStreamSynchronize(st));
  CK(hipEventRecord(e0, st));
  for (int i = 0; i < reps; i++) fn();
  CK(hipEventRecord(e1, st));
  CK(hipStreamSynchronize(st));
  float ms = 0;
  CK(hipEventElapsedTime(&ms, e0, e1));
  CK(hipEventDestroy(e0));
  CK(hipEventDestroy(e1));
  const double us = ms * 1000.0 / reps;
  printf("  %-24s %8.2f us  %7.1f GB/s\n", name, us, bytes / us / 1000.0);
  return us;
}

int cmp_f(const float* a, const float* b, size_t n, const char* nm) {
  std::vector<float> ha(n), hb(n);
  CK(hipMemcpy(ha.data(), a, n * 4, hipMemcpyDeviceToHost));
  CK(hipMemcpy(hb.data(), b, n * 4, hipMemcpyDeviceToHost));
  int bad = 0;
  for (size_t i = 0; i < n; i++) bad += memcmp(&ha[i], &hb[i], 4) != 0;
  printf("  cmp %-24s %d / %zu differ\n", nm, bad, n);
  return bad;
}

int main() {
  CK(hipSetDevice(0));
  srand(7);
  g_hash = 12345.f;

  // ---- HC_down: R=320, C=10240, st=8320 ------------------------------------
  {
    const int R = 320, C = 10240, st = 8320;
    uint8_t *wq, *wq2;
    float *x, *part, *tA, *tB, *tC;
    int* cnt;
    CK(hipMalloc(&wq, (size_t)R * st));
    CK(hipMalloc(&x, C * 4));
    CK(hipMalloc(&part, 5 * R * 4));
    CK(hipMalloc(&tA, R * 4));
    CK(hipMalloc(&tB, R * 4));
    CK(hipMalloc(&tC, R * 4));
    CK(hipMalloc(&cnt, 64 * 4));
    CK(hipMemset(cnt, 0, 64 * 4));
    std::vector<uint8_t> hw((size_t)R * st);
    std::vector<float> hx(C);
    for (auto& b : hw) b = (uint8_t)(rand() & 63);
    for (auto& v : hx) v = (float)(rand() % 1024) / 256.f;
    CK(hipMemcpy(wq, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(x, hx.data(), C * 4, hipMemcpyHostToDevice));
    const double bytes = (double)R * st + C * 4;
    printf("HC_down 320x10240 (%.2f MB)\n", bytes / 1e6);
    const int reps = 200;
    double us_gen = time_kernel("gen (old prod)", reps, bytes, [&] {
      k_hcf_down<<<dim3(40, 5), 256>>>(wq, x, part, tA, R, C, st, 32, cnt);
    }, 0);
    double us_c10k = time_kernel("c10k atomic", reps, bytes, [&] {
      k_hcf_down_c10k<<<dim3(40, 5), 256>>>(wq, x, part, tB, st, cnt);
    }, 0);
    double us_nsk = time_kernel("c10k_nsk (new)", reps, bytes, [&] {
      k_hcf_down_c10k_nsk<<<80, 128>>>(wq, x, tC, st);
    }, 0);
    cmp_f(tA, tB, R, "gen vs c10k");
    cmp_f(tA, tC, R, "gen vs nsk");
    CK(hipFree(wq));
    CK(hipFree(x));
    CK(hipFree(part));
    CK(hipFree(tA));
    CK(hipFree(tB));
    CK(hipFree(tC));
    CK(hipFree(cnt));
    (void)us_gen; (void)us_c10k; (void)us_nsk;
  }

  // ---- HC_upc: 10240x320, st=260 -------------------------------------------
  {
    const int d = 2560, C = 320, st = 260;
    uint8_t* wq;
    float *t, *rh, *xoA, *xoB;
    CK(hipMalloc(&wq, (size_t)10240 * st));
    CK(hipMalloc(&t, C * 4));
    CK(hipMalloc(&rh, 4 * d * 4));
    CK(hipMalloc(&xoA, d * 4));
    CK(hipMalloc(&xoB, d * 4));
    std::vector<uint8_t> hw((size_t)10240 * st);
    std::vector<float> ht(C), hrh(4 * d);
    for (auto& b : hw) b = (uint8_t)(rand() & 63);
    for (auto& v : ht) v = (float)(rand() % 1024) / 256.f;
    for (auto& v : hrh) v = (float)(rand() % 512) / 256.f;
    CK(hipMemcpy(wq, hw.data(), hw.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(t, ht.data(), C * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(rh, hrh.data(), hrh.size() * 4, hipMemcpyHostToDevice));
    const double bytes = (double)10240 * st + C * 4 + 4 * d * 4;
    printf("HC_upc 10240x320 (%.2f MB)\n", bytes / 1e6);
    const int reps = 200;
    double us_old = time_kernel("upc_160 scalar (old)", reps, bytes, [&] {
      k_hcf_upc_160_old<<<320, 160>>>(wq, t, rh, xoA, st);
    }, 0);
    double us_new = time_kernel("upc_160 uint4 (new)", reps, bytes, [&] {
      k_hcf_upc_160_new<<<320, 160>>>(wq, t, rh, xoB, st);
    }, 0);
    cmp_f(xoA, xoB, d, "old vs new uint4");
    CK(hipFree(wq));
    CK(hipFree(t));
    CK(hipFree(rh));
    CK(hipFree(xoA));
    CK(hipFree(xoB));
    (void)us_old; (void)us_new;
  }

  // ---- GDN step: 48 heads x 128x128 fp32 ------------------------------------
  {
    const int H = 48;
    float *q, *k, *v, *g, *beta, *S, *z, *w, *ytmp, *outA, *outB, *outC;
    int* cnt;
    CK(hipMalloc(&q, 16 * 128 * 4));
    CK(hipMalloc(&k, 16 * 128 * 4));
    CK(hipMalloc(&v, H * 128 * 4));
    CK(hipMalloc(&g, H * 4));
    CK(hipMalloc(&beta, H * 4));
    CK(hipMalloc(&S, (size_t)H * 128 * 128 * 4));
    CK(hipMalloc(&z, H * 128 * 4));
    CK(hipMalloc(&w, 128 * 4));
    CK(hipMalloc(&ytmp, H * 128 * 4));
    CK(hipMalloc(&outA, H * 128 * 4));
    CK(hipMalloc(&outB, H * 128 * 4));
    CK(hipMalloc(&outC, H * 128 * 4));
    CK(hipMalloc(&cnt, 64 * 4));
    CK(hipMemset(cnt, 0, 64 * 4));
    std::vector<float> hq(16 * 128), hk(16 * 128), hv(H * 128), hg(H), hb(H),
        hS((size_t)H * 128 * 128), hz(H * 128), hw(128);
    for (auto& x : hq) x = (float)(rand() % 512) / 256.f;
    for (auto& x : hk) x = (float)(rand() % 512) / 256.f;
    for (auto& x : hv) x = (float)(rand() % 512) / 256.f;
    for (auto& x : hg) x = -2.f + (float)(rand() % 128) / 64.f;
    for (auto& x : hb) x = (float)(rand() % 128) / 128.f;
    for (auto& x : hS) x = (float)(rand() % 512) / 256.f;
    for (auto& x : hz) x = (float)(rand() % 512) / 256.f;
    for (auto& x : hw) x = (float)(rand() % 512) / 256.f;
    CK(hipMemcpy(q, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(k, hk.data(), hk.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(v, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(g, hg.data(), hg.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(beta, hb.data(), hb.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(z, hz.data(), hz.size() * 4, hipMemcpyHostToDevice));
    CK(hipMemcpy(w, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
    const double bytes = 2.0 * H * 128 * 128 * 4;
    printf("GDN_step 48 heads (%.2f MB r+w)\n", bytes / 1e6);
    const int reps = 50;
    // bit-exact: run each variant once from the same S, compare S and out
    auto run_once = [&](int variant) {
      CK(hipMemcpy(S, hS.data(), hS.size() * 4, hipMemcpyHostToDevice));
      CK(hipMemset(cnt, 0, 64 * 4));
      if (variant == 0)
        k_gdn_step_gn_128<<<192, 128>>>(q, k, v, g, beta, S, z, w, ytmp, outA, cnt, 1e-5f);
      else if (variant == 1) {
        k_gdn_step_gn_yt<<<192, 128>>>(q, k, v, g, beta, S, z, ytmp);
        k_gdn_norm_128<<<48, 128>>>(ytmp, z, w, outB, 1e-5f);
      } else
        k_gdn_step_gn_f2<<<48, 256>>>(q, k, v, g, beta, S, z, w, outC, 1e-5f);
      CK(hipDeviceSynchronize());
      std::vector<float> hs(hS.size());
      CK(hipMemcpy(hs.data(), S, hs.size() * 4, hipMemcpyDeviceToHost));
      return hs;
    };
    std::vector<float> sA = run_once(0);
    std::vector<float> sB = run_once(1);
    std::vector<float> sC = run_once(2);
    int bad = 0;
    for (size_t i = 0; i < sA.size(); i++) bad += memcmp(&sA[i], &sB[i], 4) != 0;
    printf("  cmp %-24s %d / %zu differ\n", "S _128 vs yt+norm", bad, sA.size());
    bad = 0;
    for (size_t i = 0; i < sA.size(); i++) bad += memcmp(&sA[i], &sC[i], 4) != 0;
    printf("  cmp %-24s %d / %zu differ\n", "S _128 vs f2", bad, sA.size());
    cmp_f(outA, outB, H * 128, "out _128 vs yt+norm");
    cmp_f(outA, outC, H * 128, "out _128 vs f2");
    {
      std::vector<float> oa(H * 128), ob(H * 128), oc(H * 128);
      CK(hipMemcpy(oa.data(), outA, oa.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(ob.data(), outB, ob.size() * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(oc.data(), outC, oc.size() * 4, hipMemcpyDeviceToHost));
      for (int i = 0; i < 4; i++)
        printf("  dbg out[%d]: A=%.9g B=%.9g C=%.9g\n", i, oa[i], ob[i], oc[i]);
      {
        std::vector<float> hy(H * 128);
        CK(hipMemcpy(hy.data(), ytmp, hy.size() * 4, hipMemcpyDeviceToHost));
        for (int i = 0; i < 4; i++) printf("  dbg ytmp[%d]=%.9g\n", i, hy[i]);
      }
    }
    double us_128 = time_kernel("gdn_128 (prod)", reps, bytes, [&] {
      k_gdn_step_gn_128<<<192, 128>>>(q, k, v, g, beta, S, z, w, ytmp, outA, cnt, 1e-5f);
    }, 0);
    double us_yt = time_kernel("gdn yt+norm (f3)", reps, bytes, [&] {
      k_gdn_step_gn_yt<<<192, 128>>>(q, k, v, g, beta, S, z, ytmp);
      k_gdn_norm_128<<<48, 128>>>(ytmp, z, w, outB, 1e-5f);
    }, 0);
    double us_f2 = time_kernel("gdn f2", reps, bytes, [&] {
      k_gdn_step_gn_f2<<<48, 256>>>(q, k, v, g, beta, S, z, w, outC, 1e-5f);
    }, 0);
    CK(hipFree(q));
    CK(hipFree(k));
    CK(hipFree(v));
    CK(hipFree(g));
    CK(hipFree(beta));
    CK(hipFree(S));
    CK(hipFree(z));
    CK(hipFree(w));
    CK(hipFree(ytmp));
    CK(hipFree(outA));
    CK(hipFree(outB));
    CK(hipFree(outC));
    CK(hipFree(cnt));
    (void)us_128; (void)us_yt; (void)us_f2;
  }
  printf("done\n");
  return 0;
}
