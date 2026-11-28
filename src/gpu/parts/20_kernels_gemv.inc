// ============================ kernels =======================================

// --- Q4C-P GEMV: y[r] = sum_c cb[nib]*scale[r][c/32]*x[c]; one warp per row.
// v3 (decode GEMV governance): cb staged in LDS, fp32 x, 2 rows per warp.
// Bit-exact vs the original 1-row/warp version: same per-lane fma order,
// same shuffle reduction tree (verified d=0 in tools/decode_gemv_bench.cu).
__global__ void k_q4cp_gemv(const uint8_t* __restrict__ codes,
                            const uint8_t* __restrict__ scales,
                            const float* __restrict__ cb,
                            const float* __restrict__ x, float* __restrict__ y,
                            uint64_t rows, uint64_t cols, uint64_t scale_stride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t groups_per_row = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = row0 + 1 < rows;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < groups_per_row; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    const float* xg = x + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    y[row0] = acc0;
    if (has1) y[row0 + 1] = acc1;
  }
}

// --- q4cp multi-row GEMV: y[P,N] = x[P,K] . W[N,K]^T, weight streamed once ---
// Same dequant math as k_q4cp_gemv (16-entry codebook + fp16 group scale),
// but each warp applies its 2 weight rows to all P input rows. x is small and
// stays L2-resident across warps, so DRAM traffic ~ N*K/2 bytes regardless of
// P — this replaces the dequant->bf16->hipBLASLt path for small P, which
// costs ~4.5x the weight bytes in extra traffic.
template <int P, bool XBF16 = false>
__global__ void k_q4cp_gemv_mr(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const void* __restrict__ x,
                               float* __restrict__ y, uint64_t rows,
                               uint64_t cols, uint64_t scale_stride,
                               uint64_t xstride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t groups_per_row = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = row0 + 1 < rows;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < groups_per_row; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + g * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + g * 2));
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
#pragma unroll
    for (int p = 0; p < P; p++) {
      float xv[32];
      if (XBF16) {
        const uint16_t* xg =
            (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          uint4 t = *(const uint4*)(xg + i * 8);
          const uint16_t* th = (const uint16_t*)&t;
#pragma unroll
          for (int j = 0; j < 8; j++)
            xv[i * 8 + j] = __bfloat162float(*(__hip_bfloat16*)(&th[j]));
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          float4 t = *(const float4*)(xg + i * 4);
          xv[i * 4] = t.x;
          xv[i * 4 + 1] = t.y;
          xv[i * 4 + 2] = t.z;
          xv[i * 4 + 3] = t.w;
        }
      }
      uint32_t u0[4] = {w0[0], w0[1], w0[2], w0[3]};
      uint32_t u1[4] = {w1[0], w1[1], w1[2], w1[3]};
      float b0 = 0.f, b1 = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
          b0 += cbf[u0[i] & 0xf] * xv[i * 8 + j];
          b1 += cbf[u1[i] & 0xf] * xv[i * 8 + j];
          u0[i] >>= 4;
          u1[i] >>= 4;
        }
      }
      acc0[p] += b0 * s0;
      acc1[p] += b1 * s1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- bf16-weight multi-row GEMV: y[P,N] = x[P,K] . W[N,K]^T (bf16 weights) ---
template <int P>
__global__ void k_bf16_gemv_mr(const uint16_t* __restrict__ w,
                               const float* __restrict__ x,
                               float* __restrict__ y, uint64_t rows,
                               uint64_t cols, uint64_t xstride) {
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const uint16_t* r0 = w + row0 * cols;
  const uint16_t* r1 = w + (row0 + 1) * cols;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t c = lane * 8; c < cols; c += warpSize * 8) {
    uint4 pk0 = *(const uint4*)(r0 + c);
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    if (has1) pk1 = *(const uint4*)(r1 + c);
    const uint16_t* h0 = (const uint16_t*)&pk0;
    const uint16_t* h1 = (const uint16_t*)&pk1;
    float w0[8], w1[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      w0[j] = __bfloat162float(*(__hip_bfloat16*)(&h0[j]));
      w1[j] = __bfloat162float(*(__hip_bfloat16*)(&h1[j]));
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      const float* xg = x + (size_t)p * xstride + c;
      float4 xa = *(const float4*)xg;
      float4 xb = *(const float4*)(xg + 4);
      float b0 = w0[0] * xa.x + w0[1] * xa.y + w0[2] * xa.z + w0[3] * xa.w +
                 w0[4] * xb.x + w0[5] * xb.y + w0[6] * xb.z + w0[7] * xb.w;
      float b1 = w1[0] * xa.x + w1[1] * xa.y + w1[2] * xa.z + w1[3] * xa.w +
                 w1[4] * xb.x + w1[5] * xb.y + w1[6] * xb.z + w1[7] * xb.w;
      acc0[p] += b0;
      acc1[p] += b1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- fp32-weight multi-row GEMV: y[P,N] = x[P,K] . W[N,K]^T (fp32 weights) ---
// One warp per row, lane-strided float4. For small-N fp32 projections at
// decode P (indexer iproj N=640 K=2560) where rocBLAS picks latency-bound
// tiles (MT32x32x8, 20 blocks, ~110 us); this streams W once (~36 us).
// NOT bit-exact vs rocblas_sgemm (lane-split + tree reduce reorder the sum).
template <int P>
__global__ void k_f32_gemv_mr(const float* __restrict__ w,
                              const float* __restrict__ x,
                              float* __restrict__ y, uint64_t rows,
                              uint64_t cols) {
  uint64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const float* rw = w + row * cols;
  float acc[P];
#pragma unroll
  for (int p = 0; p < P; p++) acc[p] = 0.f;
  for (uint64_t c = lane * 4; c < cols; c += warpSize * 4) {
    float4 wv = *(const float4*)(rw + c);
#pragma unroll
    for (int p = 0; p < P; p++) {
      float4 xv = *(const float4*)(x + (size_t)p * cols + c);
      acc[p] += wv.x * xv.x;
      acc[p] += wv.y * xv.y;
      acc[p] += wv.z * xv.z;
      acc[p] += wv.w * xv.w;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1)
      acc[p] += __shfl_down_sync(~0ull, acc[p], off);
    if (lane == 0) y[(size_t)p * rows + row] = acc[p];
  }
}

// --- q8g64 multi-row GEMV (overlay o_proj): [codes uint8 x cols][(s,m) fp16 per 64] ---
template <int P, bool XBF16 = false>
__global__ void k_q8g64_gemv_mr(const uint8_t* __restrict__ w,
                                const void* __restrict__ x,
                                float* __restrict__ y, uint64_t rows,
                                uint64_t cols, uint64_t xstride) {
  uint64_t stride = cols + cols / 64 * 4;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const uint8_t* r0 = w + row0 * stride;
  const uint8_t* r1 = w + (row0 + 1) * stride;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < cols / 64; g += warpSize) {
    const uint8_t* c0 = r0 + g * 64;
    const __half* sm0 = (const __half*)(r0 + cols + g * 4);
    float s0 = __half2float(sm0[0]), m0 = __half2float(sm0[1]);
    float s1 = 0.f, m1 = 0.f;
    const uint8_t* c1 = (has1 ? r1 : r0) + g * 64;  // !has1: dead, s1=m1=0
    const __half* sm1 = (const __half*)(r1 + cols + g * 4);
    if (has1) {
      s1 = __half2float(sm1[0]);
      m1 = __half2float(sm1[1]);
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      float b0 = 0.f, b1 = 0.f, xs = 0.f;
      if (XBF16) {
        const uint16_t* xg = (const uint16_t*)x + (size_t)p * xstride + g * 64;
#pragma unroll
        for (int j = 0; j < 64; j += 8) {
          uint4 t = *(const uint4*)(xg + j);
          const uint16_t* th = (const uint16_t*)&t;
#pragma unroll
          for (int k = 0; k < 8; k++) {
            float v = __bfloat162float(*(__hip_bfloat16*)(&th[k]));
            b0 += (float)c0[j + k] * v;
            b1 += (float)c1[j + k] * v;
            xs += v;
          }
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 64;
#pragma unroll
        for (int j = 0; j < 64; j += 4) {
          float4 t = *(const float4*)(xg + j);
          b0 += (float)c0[j] * t.x + (float)c0[j + 1] * t.y +
                (float)c0[j + 2] * t.z + (float)c0[j + 3] * t.w;
          b1 += (float)c1[j] * t.x + (float)c1[j + 1] * t.y +
                (float)c1[j + 2] * t.z + (float)c1[j + 3] * t.w;
          xs += t.x + t.y + t.z + t.w;
        }
      }
      // q8g64 dequant: w = c*s + m  ->  w.x = s*(c.x) + m*sum(x)
      acc0[p] += b0 * s0 + m0 * xs;
      acc1[p] += b1 * s1 + m1 * xs;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- q8g64 GEMV (overlay o_proj): per row [cols uint8][cols/64 x (fp16 s, fp16 m)]
__global__ void k_q8g64_gemv(const uint8_t* __restrict__ w, const float* __restrict__ x,
                             float* __restrict__ y, uint64_t rows, uint64_t cols) {
  uint64_t stride = cols + cols / 64 * 4;
  uint64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rp = w + row * stride;
  float acc = 0.f;
  for (uint64_t g = lane; g < cols / 64; g += warpSize) {
    const uint8_t* codes = rp + g * 64;
    const __half* sm = (const __half*)(rp + cols + g * 4);
    float s = __half2float(sm[0]), mn = __half2float(sm[1]);
    float a = 0.f, cs = 0.f;
    const float* xg = x + g * 64;
#pragma unroll
    for (int j = 0; j < 64; j += 4) {
      float4 xv = *(const float4*)(xg + j);
      a += codes[j] * xv.x + codes[j + 1] * xv.y + codes[j + 2] * xv.z +
           codes[j + 3] * xv.w;
      cs += xv.x + xv.y + xv.z + xv.w;
    }
    acc += a * s + mn * cs;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(~0ull, acc, off);
  if (lane == 0) y[row] = acc;
}

// --- q8g32 planar (GGUF Q8_0 repack, hgn.h dtype 8) --------------------------
// q: [rows][cols] int8 codes, s: [rows][cols/32] fp16 scales; w = q * s.
// y[P,N] = x[P,K] . W[N,K]^T, 2 rows per warp, lanes stride over 32-col
// groups (32 contiguous code bytes = 2 x uint4 per lane), weight read once
// for all P input rows. P=1 doubles as the plain decode GEMV.
__device__ inline float q8_b(uint32_t v, int k) {
  return (float)(int8_t)(uint8_t)(v >> (8 * k));
}
template <int P, bool XBF16 = false>
__global__ void k_q8g32_gemv_mr(const int8_t* __restrict__ q,
                                const __half* __restrict__ s,
                                const void* __restrict__ x,
                                float* __restrict__ y, uint64_t rows,
                                uint64_t cols, uint64_t xstride) {
  const uint64_t gpr = cols / 32;
  uint64_t row0 =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * 2;
  if (row0 >= rows) return;
  int lane = threadIdx.x % warpSize;
  bool has1 = row0 + 1 < rows;
  const int8_t* q0 = q + row0 * cols;
  const int8_t* q1 = q + (has1 ? row0 + 1 : row0) * cols;
  const __half* s0p = s + row0 * gpr;
  const __half* s1p = s + (has1 ? row0 + 1 : row0) * gpr;
  float acc0[P], acc1[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    acc0[p] = 0.f;
    acc1[p] = 0.f;
  }
  for (uint64_t g = lane; g < gpr; g += warpSize) {
    const uint4* c0 = (const uint4*)(q0 + g * 32);
    const uint4* c1 = (const uint4*)(q1 + g * 32);
    uint4 a0 = c0[0], a1 = c0[1], b0 = c1[0], b1 = c1[1];
    uint32_t w0[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    uint32_t w1[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
    float sc0 = __half2float(s0p[g]);
    float sc1 = has1 ? __half2float(s1p[g]) : 0.f;
#pragma unroll
    for (int p = 0; p < P; p++) {
      float xv[32];
      if (XBF16) {
        const uint16_t* xg = (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          uint4 t = *(const uint4*)(xg + i * 8);
          const uint16_t* th = (const uint16_t*)&t;
#pragma unroll
          for (int j = 0; j < 8; j++)
            xv[i * 8 + j] = __bfloat162float(*(__hip_bfloat16*)(&th[j]));
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          float4 t = *(const float4*)(xg + i * 4);
          xv[i * 4] = t.x;
          xv[i * 4 + 1] = t.y;
          xv[i * 4 + 2] = t.z;
          xv[i * 4 + 3] = t.w;
        }
      }
      float d0 = 0.f, d1 = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) {
          d0 += q8_b(w0[i], k) * xv[i * 4 + k];
          d1 += q8_b(w1[i], k) * xv[i * 4 + k];
        }
      acc0[p] += d0 * sc0;
      acc1[p] += d1 * sc1;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      acc0[p] += __shfl_down_sync(~0ull, acc0[p], off);
      acc1[p] += __shfl_down_sync(~0ull, acc1[p], off);
    }
    if (lane == 0) {
      y[(size_t)p * rows + row0] = acc0[p];
      if (has1) y[(size_t)p * rows + row0 + 1] = acc1[p];
    }
  }
}

// --- q8g32 embedding gather: block b dequants row tok[b] into
// out + b*outstride (fp32), or with HC4 into four bf16 branch copies at
// out + b*4*cols (same layout as k_q4cp_row_b_hc_bf16). cols/32 threads.
template <bool HC4>
__global__ void k_q8g32_row_b(const int8_t* __restrict__ q, const __half* __restrict__ s,
                              const int* __restrict__ d_tokens, uint64_t cols,
                              void* __restrict__ out, uint64_t outstride) {
  uint64_t row = (uint64_t)d_tokens[blockIdx.x];
  uint64_t g = threadIdx.x;
  if (g * 32 >= cols) return;
  const uint4* c = (const uint4*)(q + row * cols + g * 32);
  uint4 a = c[0], b = c[1];
  uint32_t w[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  float sc = __half2float(s[row * (cols / 32) + g]);
#pragma unroll
  for (int i = 0; i < 8; i++)
#pragma unroll
    for (int k = 0; k < 4; k++) {
      uint64_t col = g * 32 + i * 4 + k;
      float v = q8_b(w[i], k) * sc;
      if (HC4) {
        uint32_t u;
        memcpy(&u, &v, 4);
        u += 0x7FFF + ((u >> 16) & 1);
        uint16_t h = (uint16_t)(u >> 16);
        uint16_t* o = (uint16_t*)out + (size_t)blockIdx.x * 4 * cols;
        o[col] = h;
        o[cols + col] = h;
        o[2 * cols + col] = h;
        o[3 * cols + col] = h;
      } else {
        ((float*)out)[(size_t)blockIdx.x * outstride + col] = v;
      }
    }
}

// --- BF16 GEMV: one warp per row, float4 (8 bf16) loads
__global__ void k_bf16_gemv(const uint16_t* __restrict__ w, const float* __restrict__ x,
                            float* __restrict__ y, uint64_t rows, uint64_t cols) {
  uint64_t row = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint16_t* rw = w + row * cols;
  float acc = 0.f;
  for (uint64_t c = lane * 8; c < cols; c += warpSize * 8) {
    uint4 pk = *(const uint4*)(rw + c);  // 8 bf16
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
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(~0ull, acc, off);
  if (lane == 0) y[row] = acc;
}

__global__ void k_bf16_gemv_2m(const uint16_t* __restrict__ w0, float* __restrict__ y0,
                               uint64_t rows0, unsigned b0,
                               const uint16_t* __restrict__ w1, float* __restrict__ y1,
                               uint64_t rows1, const float* __restrict__ x, uint64_t cols) {
  const bool second = blockIdx.x >= b0;
  const unsigned bx = second ? (blockIdx.x - b0) : blockIdx.x;
  const uint64_t rows = second ? rows1 : rows0;
  uint64_t row = (uint64_t)bx * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= rows) return;
  int lane = threadIdx.x % warpSize;
  const uint16_t* rw = (second ? w1 : w0) + row * cols;
  float acc = 0.f;
  for (uint64_t c = lane * 8; c < cols; c += warpSize * 8) {
    uint4 pk = *(const uint4*)(rw + c);
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
  for (int off = warpSize / 2; off > 0; off >>= 1)
    acc += __shfl_down_sync(~0ull, acc, off);
  if (lane == 0) (second ? y1 : y0)[row] = acc;
}

__global__ void k_f32_copy_vec4(float* __restrict__ dst, const float* __restrict__ src, size_t n) {
  size_t n4 = n / 4;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (size_t)gridDim.x * blockDim.x) {
    ((float4*)dst)[i] = ((const float4*)src)[i];
  }
  for (size_t i = n4 * 4 + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

// --- q8g32 P=1 GEMV, LPR lanes per row (32/LPR rows per wave), TPB-thread
// blocks. For the decode shapes where k_q8g32_gemv_mr<1> (2 rows per wave,
// 512-thread blocks) runs below bandwidth: few rows (2560 x 6144: only 80
// blocks on 40 CUs) or short rows (640 cols = 20 groups < 32 lanes).
template <int LPR, bool kSilu = false>
__global__ void k_q8g32_gemv_lpr(const int8_t* __restrict__ q, const __half* __restrict__ s,
                                 const float* __restrict__ x, float* __restrict__ y,
                                 uint64_t rows, uint64_t cols) {
  constexpr int RPW = 32 / LPR;
  const uint64_t gpr = cols / 32;
  const int lane = threadIdx.x % warpSize, ll = lane % LPR;
  const uint64_t row =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * RPW + lane / LPR;
  const bool live = row < rows;
  const int8_t* qr = q + (live ? row : 0) * cols;
  const __half* sr = s + (live ? row : 0) * gpr;
  float acc = 0.f;
#pragma unroll 2
  for (uint64_t g = ll; g < gpr; g += LPR) {
    const uint4* c = (const uint4*)(qr + g * 32);
    const uint4 a0 = c[0], a1 = c[1];
    const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float* xg = x + g * 32;
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float4 t = *(const float4*)(xg + i * 4);
      d += q8_b(w[i], 0) * t.x + q8_b(w[i], 1) * t.y + q8_b(w[i], 2) * t.z + q8_b(w[i], 3) * t.w;
    }
    acc += __half2float(sr[g]) * d;
  }
#pragma unroll
  for (int off = LPR / 2; off > 0; off >>= 1) acc += __shfl_down_sync(~0ull, acc, off, LPR);
  if (ll == 0 && live) {
    float v = acc;
    if (kSilu) {  // same expression as k_silu_scale(v, 0.25f)
      const float z = v * 0.25f;
      v = z / (1.f + expf(-z));
    }
    y[row] = v;
  }
}
template <int LPR, int TPB, bool kSilu = false>
static void q8g32_gemv_small(const int8_t* q, const __half* s, const float* x, float* y,
                             uint64_t rows, uint64_t cols, hipStream_t st) {
  constexpr uint64_t rpb = (TPB / 32) * (32 / LPR);
  k_q8g32_gemv_lpr<LPR, kSilu>
      <<<(unsigned)((rows + rpb - 1) / rpb), TPB, 0, st>>>(q, s, x, y, rows, cols);
}

// --- q8g32 multi-row (P input rows, spec verify / small batches) with LPR
// lanes per weight row: same layout as k_q8g32_gemv_lpr, codes decoded to
// float once per 32-col group and reused for all P x rows (weight read once).
// x: P rows of stride xstride, fp32 or bf16 (XBF16). y[p*rows + r].
template <int P, bool XBF16, int LPR>
__global__ void k_q8g32_gemv_mlpr(const int8_t* __restrict__ q, const __half* __restrict__ s,
                                  const void* __restrict__ x, float* __restrict__ y,
                                  uint64_t rows, uint64_t cols, uint64_t xstride) {
  constexpr int RPW = 32 / LPR;
  const uint64_t gpr = cols / 32;
  const int lane = threadIdx.x % warpSize, ll = lane % LPR;
  const uint64_t row =
      (blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize) * RPW + lane / LPR;
  const bool live = row < rows;
  const int8_t* qr = q + (live ? row : 0) * cols;
  const __half* sr = s + (live ? row : 0) * gpr;
  float acc[P];
#pragma unroll
  for (int p = 0; p < P; p++) acc[p] = 0.f;
  for (uint64_t g = ll; g < gpr; g += LPR) {
    const uint4* c = (const uint4*)(qr + g * 32);
    const uint4 a0 = c[0], a1 = c[1];
    const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float sc = __half2float(sr[g]);
    float wf[32];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int k = 0; k < 4; k++) wf[i * 4 + k] = q8_b(w[i], k);
#pragma unroll
    for (int p = 0; p < P; p++) {
      float d = 0.f;
      if (XBF16) {
        const uint16_t* xg = (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const uint4 t = *(const uint4*)(xg + i * 8);
          const uint32_t tv[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
          for (int j = 0; j < 4; j++) {
            d += wf[i * 8 + 2 * j] * __uint_as_float(tv[j] << 16);
            d += wf[i * 8 + 2 * j + 1] * __uint_as_float(tv[j] & 0xFFFF0000u);
          }
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const float4 t = *(const float4*)(xg + i * 4);
          d += wf[i * 4] * t.x + wf[i * 4 + 1] * t.y + wf[i * 4 + 2] * t.z + wf[i * 4 + 3] * t.w;
        }
      }
      acc[p] += sc * d;
    }
  }
#pragma unroll
  for (int p = 0; p < P; p++) {
#pragma unroll
    for (int off = LPR / 2; off > 0; off >>= 1) acc[p] += __shfl_down_sync(~0ull, acc[p], off, LPR);
    if (ll == 0 && live) y[(size_t)p * rows + row] = acc[p];
  }
}
// Few long rows (HC down 320 x 10240): one TPB-thread block per row, block
// reduce, so 320 rows still fill the GPU (the wave-per-row kernel gets only
// 320 waves on 40 CUs and stays latency-bound).
template <int P, bool XBF16, int TPB>
__launch_bounds__(TPB) __global__
void k_q8g32_gemv_brow(const int8_t* __restrict__ q, const __half* __restrict__ s,
                       const void* __restrict__ x, float* __restrict__ y, uint64_t rows,
                       uint64_t cols, uint64_t xstride) {
  const uint64_t gpr = cols / 32, row = blockIdx.x;
  const int8_t* qr = q + row * cols;
  const __half* sr = s + row * gpr;
  float acc[P];
#pragma unroll
  for (int p = 0; p < P; p++) acc[p] = 0.f;
  for (uint64_t g = threadIdx.x; g < gpr; g += TPB) {
    const uint4* c = (const uint4*)(qr + g * 32);
    const uint4 a0 = c[0], a1 = c[1];
    const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const float sc = __half2float(sr[g]);
    float wf[32];
#pragma unroll
    for (int i = 0; i < 8; i++)
#pragma unroll
      for (int k = 0; k < 4; k++) wf[i * 4 + k] = q8_b(w[i], k);
#pragma unroll
    for (int p = 0; p < P; p++) {
      float d = 0.f;
      if (XBF16) {
        const uint16_t* xg = (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const uint4 t = *(const uint4*)(xg + i * 8);
          const uint32_t tv[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
          for (int j = 0; j < 4; j++) {
            d += wf[i * 8 + 2 * j] * __uint_as_float(tv[j] << 16);
            d += wf[i * 8 + 2 * j + 1] * __uint_as_float(tv[j] & 0xFFFF0000u);
          }
        }
      } else {
        const float* xg = (const float*)x + (size_t)p * xstride + g * 32;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const float4 t = *(const float4*)(xg + i * 4);
          d += wf[i * 4] * t.x + wf[i * 4 + 1] * t.y + wf[i * 4 + 2] * t.z + wf[i * 4 + 3] * t.w;
        }
      }
      acc[p] += sc * d;
    }
  }
  __shared__ float red[P][TPB / 32];
  const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) acc[p] += __shfl_down_sync(~0ull, acc[p], off);
    if (lane == 0) red[p][wid] = acc[p];
  }
  __syncthreads();
  if (threadIdx.x < P) {
    float t = 0.f;
    for (int i = 0; i < TPB / 32; i++) t += red[threadIdx.x][i];
    y[(size_t)threadIdx.x * rows + row] = t;
  }
}
// Row-tiled q8g32 block-GEMV (bf16 x): block covers RPB consecutive rows so
// each x group is fetched once per block instead of once per row (brow
// re-reads all of x per row — e.g. HC down 320x10240 at P=4 re-reads 80 KB x
// 320 times = 26 MB of L2 traffic; RPB=4 cuts it to 6.5 MB). Per (row,p) the
// group order, d expression, acc chain, warp tree and smem sum replicate
// k_q8g32_gemv_brow exactly: bit-exact by construction.
template <int P, int TPB, int RPB>
__launch_bounds__(TPB) __global__
void k_q8g32_gemv_brow_r(const int8_t* __restrict__ q, const __half* __restrict__ s,
                         const void* __restrict__ x, float* __restrict__ y, uint64_t rows,
                         uint64_t cols, uint64_t xstride) {
  const uint64_t gpr = cols / 32, row0 = (uint64_t)blockIdx.x * RPB;
  float acc[RPB][P];
#pragma unroll
  for (int r = 0; r < RPB; r++)
#pragma unroll
    for (int p = 0; p < P; p++) acc[r][p] = 0.f;
  for (uint64_t g = threadIdx.x; g < gpr; g += TPB) {
    uint4 xv[P][4];  // x group (32 bf16 per p), loaded once per (g, block)
#pragma unroll
    for (int p = 0; p < P; p++) {
      const uint16_t* xg = (const uint16_t*)x + (size_t)p * xstride + g * 32;
#pragma unroll
      for (int i = 0; i < 4; i++) xv[p][i] = *(const uint4*)(xg + i * 8);
    }
#pragma unroll
    for (int r = 0; r < RPB; r++) {
      const uint64_t row = row0 + r;
      if (row >= rows) break;  // block-uniform, does not desync barriers
      const uint4* c = (const uint4*)(q + row * cols + g * 32);
      const uint4 a0 = c[0], a1 = c[1];
      const uint32_t w[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
      const float sc = __half2float(s[row * gpr + g]);
      float wf[32];
#pragma unroll
      for (int i = 0; i < 8; i++)
#pragma unroll
        for (int k = 0; k < 4; k++) wf[i * 4 + k] = q8_b(w[i], k);
#pragma unroll
      for (int p = 0; p < P; p++) {
        float d = 0.f;
        const uint32_t* tv = (const uint32_t*)xv[p];
#pragma unroll
        for (int i = 0; i < 4; i++) {
#pragma unroll
          for (int j = 0; j < 4; j++) {
            d += wf[i * 8 + 2 * j] * __uint_as_float(tv[i * 4 + j] << 16);
            d += wf[i * 8 + 2 * j + 1] * __uint_as_float(tv[i * 4 + j] & 0xFFFF0000u);
          }
        }
        acc[r][p] += sc * d;
      }
    }
  }
  __shared__ float red[RPB][P][TPB / 32];
  const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
#pragma unroll
  for (int r = 0; r < RPB; r++)
#pragma unroll
    for (int p = 0; p < P; p++) {
      for (int off = warpSize / 2; off > 0; off >>= 1)
        acc[r][p] += __shfl_down_sync(~0ull, acc[r][p], off);
      if (lane == 0) red[r][p][wid] = acc[r][p];
    }
  __syncthreads();
  if (threadIdx.x < RPB * P) {
    const int r = threadIdx.x / P, p = threadIdx.x % P;
    if (row0 + r < rows) {
      float t = 0.f;
      for (int i = 0; i < TPB / 32; i++) t += red[r][p][i];
      y[(size_t)p * rows + row0 + r] = t;
    }
  }
}
template <int P, bool XBF16, int LPR, int TPB>
static void q8g32_gemv_multi(const int8_t* q, const __half* s, const void* x, float* y,
                             uint64_t rows, uint64_t cols, uint64_t xstride, hipStream_t st) {
  constexpr uint64_t rpb = (TPB / 32) * (32 / LPR);
  k_q8g32_gemv_mlpr<P, XBF16, LPR><<<(unsigned)((rows + rpb - 1) / rpb), TPB, 0, st>>>(
      q, s, x, y, rows, cols, xstride);
}

// --- BF16 GEMV for few-row matrices (rows < 64: HC inject 4x10240, shared-
// expert gate 1x2560, GDN in_proj_a/b 48x2560 from GGUF): one 256-thread
// block per row + block reduce. The warp-per-row kernel above leaves those
// launches latency-bound (4 waves each walking 40 sequential steps, ~10 us).
__launch_bounds__(256) __global__
void k_bf16_gemv_row(const uint16_t* __restrict__ w, const float* __restrict__ x,
                     float* __restrict__ y, uint64_t cols) {
  const uint16_t* rw = w + (uint64_t)blockIdx.x * cols;
  float acc = 0.f;
  for (uint64_t c = threadIdx.x * 8; c < cols; c += 256 * 8) {
    uint4 pk = *(const uint4*)(rw + c);
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
  for (int off = warpSize / 2; off > 0; off >>= 1) acc += __shfl_down_sync(~0ull, acc, off);
  __shared__ float red[256 / 32];
  const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
  if (lane == 0) red[wid] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x / warpSize); i++) t += red[i];
    y[blockIdx.x] = t;
  }
}
// Multi-row (P input rows, stride xstride, fp32 or bf16) version: the spec
// verify HC inject (4 x 10240, bf16 x) otherwise falls to the rocBLAS LT
// path (~66 us per call on a 128-workgroup tile). y[p*rows + blockIdx.x].
template <int P, bool XBF16>
__launch_bounds__(256) __global__
void k_bf16_gemv_row_mp(const uint16_t* __restrict__ w, const void* __restrict__ x,
                        float* __restrict__ y, uint64_t rows, uint64_t cols, uint64_t xstride) {
  const uint16_t* rw = w + (uint64_t)blockIdx.x * cols;
  float acc[P];
#pragma unroll
  for (int p = 0; p < P; p++) acc[p] = 0.f;
  for (uint64_t c = threadIdx.x * 8; c < cols; c += 256 * 8) {
    const uint4 pk = *(const uint4*)(rw + c);
    const uint32_t pw[4] = {pk.x, pk.y, pk.z, pk.w};
    float wv[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      wv[2 * j] = __uint_as_float(pw[j] << 16);
      wv[2 * j + 1] = __uint_as_float(pw[j] & 0xFFFF0000u);
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      float xv[8];
      if (XBF16) {
        const uint4 t = *(const uint4*)((const uint16_t*)x + (uint64_t)p * xstride + c);
        const uint32_t tv[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          xv[2 * j] = __uint_as_float(tv[j] << 16);
          xv[2 * j + 1] = __uint_as_float(tv[j] & 0xFFFF0000u);
        }
      } else {
        const float* xp = (const float*)x + (uint64_t)p * xstride + c;
        const float4 xa = *(const float4*)xp, xb = *(const float4*)(xp + 4);
        xv[0] = xa.x; xv[1] = xa.y; xv[2] = xa.z; xv[3] = xa.w;
        xv[4] = xb.x; xv[5] = xb.y; xv[6] = xb.z; xv[7] = xb.w;
      }
#pragma unroll
      for (int j = 0; j < 8; j++) acc[p] += wv[j] * xv[j];
    }
  }
  __shared__ float red[P][256 / 32];
  const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
#pragma unroll
  for (int p = 0; p < P; p++) {
    for (int off = warpSize / 2; off > 0; off >>= 1) acc[p] += __shfl_down_sync(~0ull, acc[p], off);
    if (lane == 0) red[p][wid] = acc[p];
  }
  __syncthreads();
  if (threadIdx.x < P) {
    float t = 0.f;
    for (int i = 0; i < (int)(blockDim.x / warpSize); i++) t += red[threadIdx.x][i];
    y[(uint64_t)threadIdx.x * rows + blockIdx.x] = t;
  }
}

// --- Q4C-P single-row dequant (embedding gather): out[cols]
__global__ void k_q4cp_row(const uint8_t* __restrict__ codes,
                           const uint8_t* __restrict__ scales,
                           const float* __restrict__ cb, const int* __restrict__ d_token,
                           uint64_t cols, uint64_t scale_stride, float* __restrict__ out) {
  uint64_t row = (uint64_t)*d_token;
  uint64_t g = threadIdx.x;  // one block, cols/32 threads
  if (g * 32 >= cols) return;
  const uint8_t* rc = codes + row * (cols / 2) + g * 16;
  const uint8_t* rs = scales + row * scale_stride;
  float s = __half2float(*(const __half*)(rs + g * 2));
  uint4 pk = *(const uint4*)rc;
  uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint32_t v = w[i];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      out[g * 32 + i * 8 + j] = __ldg(cb + (v & 0xf)) * s;
      v >>= 4;
    }
  }
}

// --- zero-centered RMSNorm, optionally grouped. grid = one block per group.
__global__ void k_rmsnorm_zc(const float* __restrict__ x, const float* __restrict__ w,
                             float* __restrict__ out, int n, float eps) {
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < n; i += blockDim.x) ss += x[i] * x[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / n + eps);
  for (int i = t; i < n; i += blockDim.x) out[i] = x[i] * inv * (1.f + w[i]);
}

__device__ inline uint16_t f2bf(float f);
__device__ inline float bf2f(uint16_t b);

// grouped zc rmsnorm over flat (ngroups x gsize): one block per group.
// Batched: grid may exceed ngroups; weight index wraps with % ngroups.
template <bool Bf16 = false>
__global__ void k_rmsnorm_zc_grouped(const float* __restrict__ x,
                                     const float* __restrict__ w,
                                     float* __restrict__ out, int gsize, float eps,
                                     int ngroups, uint16_t* __restrict__ bf16 = nullptr) {
  const float* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)(blockIdx.x % ngroups) * gsize;
  float* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) ss += xg[i] * xg[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x) {
    float value = xg[i] * inv * (1.f + wg[i]);
    og[i] = value;
    if (Bf16) bf16[(size_t)blockIdx.x * gsize + i] = f2bf(value);
  }
}

// --- GDN conv update: state (3,qkv) shifts; out = silu(sum_j w[j]*hist[j])
// hist[0..2] = previous three inputs (oldest first), cur = current input.
__global__ void k_gdn_conv(const float* __restrict__ cur, const float* __restrict__ cw,
                           float* __restrict__ convstate, float* __restrict__ out,
                           int qkv) {
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= qkv) return;
  float* st = convstate;  // 3 x qkv
  float a = cw[ch * 4 + 0] * st[ch] + cw[ch * 4 + 1] * st[qkv + ch] +
            cw[ch * 4 + 2] * st[2 * qkv + ch] + cw[ch * 4 + 3] * cur[ch];
  st[ch] = st[qkv + ch];
  st[qkv + ch] = st[2 * qkv + ch];
  st[2 * qkv + ch] = cur[ch];
  float sg = 1.f / (1.f + expf(-a));
  out[ch] = a * sg;
}

// --- GDN recurrence, one block per v-head (128 threads).
// S layout: [h][j(128 k)][i(128 v)] row-major. q,k already l2normed; q prescaled.
__global__ void k_gdn_step(const float* __restrict__ q, const float* __restrict__ k,
                           const float* __restrict__ v, const float* __restrict__ g,
                           const float* __restrict__ beta, float* __restrict__ S,
                           float* __restrict__ out, int dk, int dv, int hv, int hk) {
  int h = blockIdx.x;
  int group = hv / hk;
  const float* kh = k + (h / group) * dk;
  const float* qh = q + (h / group) * dk;
  const float* vh = v + h * dv;
  float* Sh = S + (size_t)h * dk * dv;
  float alpha = expf(g[h]);
  float be = beta[h];
  // 4 j-chunks x dv threads: thread t owns column i=t%dv, j-chunk jc=t/dv
  int t = threadIdx.x;
  int i = t % dv, jc = t / dv;
  int j0 = jc * (dk / 4), j1 = j0 + dk / 4;
  __shared__ float part[4][128];  // partial kv / o per j-chunk
  __shared__ float sdelta[128];
  // pass 1: decay + kv partial
  float kv = 0.f;
  for (int j = j0; j < j1; j++) {
    float s = Sh[j * dv + i] * alpha;
    Sh[j * dv + i] = s;
    kv += s * kh[j];
  }
  part[jc][i] = kv;
  __syncthreads();
  float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
  float delta = (vh[i] - kvf) * be;
  if (jc == 0) sdelta[i] = delta;
  __syncthreads();
  delta = sdelta[i];
  // pass 2: rank-1 update + pass 3: output partial
  float o = 0.f;
  for (int j = j0; j < j1; j++) {
    float s = Sh[j * dv + i] + kh[j] * delta;
    Sh[j * dv + i] = s;
    o += s * qh[j];
  }
  __syncthreads();
  part[jc][i] = o;
  __syncthreads();
  if (jc == 0) out[h * dv + i] = part[0][i] + part[1][i] + part[2][i] + part[3][i];
}

// --- Fused P=1 GDN recurrence + gated RMSNorm (dk=128, dv=128, hv=48, hk=16) ---
// Keeps decayed S[32] in VGPRs across pass 1 -> pass 2 (halving S DRAM traffic
// from 432 MB to 216 MB/tok) and runs k_gdn_gatednorm in-place before storing out.
__global__ __launch_bounds__(512, 2) void k_gdn_step_gn(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, float* __restrict__ S,
    const float* __restrict__ z, const float* __restrict__ w_gn,
    float* __restrict__ out, float eps) {
  const int h = blockIdx.x;
  const float* kh = k + (h / 3) * 128;
  const float* qh = q + (h / 3) * 128;
  const float* vh = v + h * 128;
  float* Sh = S + (size_t)h * 128 * 128;
  const float alpha = expf(g[h]);
  const float be = beta[h];
  const int t = threadIdx.x;
  const int i = t & 127, jc = t >> 7;
  const int j0 = jc * 32;
  __shared__ float part[4][128];
  __shared__ float sdelta[128];
  float s_reg[32];
  float kv = 0.f;
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float s = Sh[j * 128 + i] * alpha;
    asm volatile("" : "+v"(s));
    s_reg[u] = s;
    kv += s * kh[j];
  }
  part[jc][i] = kv;
  __syncthreads();
  const float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
  float delta = (vh[i] - kvf) * be;
  if (jc == 0) sdelta[i] = delta;
  __syncthreads();
  delta = sdelta[i];
  float o = 0.f;
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    float s = s_reg[u] + kh[j] * delta;
    Sh[j * 128 + i] = s;
    o += s * qh[j];
  }
  __syncthreads();
  part[jc][i] = o;
  __syncthreads();
  float yv = 0.f;
  if (t < 128) {
    yv = part[0][t] + part[1][t] + part[2][t] + part[3][t];
    sdelta[t] = yv * yv;
  }
  __syncthreads();
  for (int off = 64; off > 0; off >>= 1) {
    if (t < off) sdelta[t] += sdelta[t + off];
    __syncthreads();
  }
  if (t < 128) {
    const float inv = rsqrtf(sdelta[0] / 128 + eps);
    const float zz = z[h * 128 + t];
    out[h * 128 + t] = yv * inv * w_gn[t] / (1.f + expf(-zz));
  }
}

// Wave-aligned 192x128 variant of k_gdn_step_gn (4 blocks of 32 columns per v-head):
// eliminates the 48-on-40 WGP tail wave (160 + 32 blocks across 20 WGPs vs 40 + 8)
// and unrolls u=0..31 so s_reg[32] stays in static VGPRs with full MLP. Bit-identical.
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

// BF16-storage variant of k_gdn_step_gn_128 (QWENOX_GDN_STATE_BF16=1):
// identical 192x128 (768-wave) high-occupancy grid and FP32 register math,
// halving S DRAM traffic from 226.5 MB/tok to 113.25 MB/tok.
__global__ __launch_bounds__(128, 8) void k_gdn_step_gn_128_bf16(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ g,
    const float* __restrict__ beta, uint16_t* __restrict__ S,
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

  uint16_t* Sh = S + (size_t)h * 128 * 128;
  const float alpha = expf(g[h]);
  const float be = beta[h];
  float s_reg[32];
  float kv = 0.f;
#pragma unroll
  for (int u = 0; u < 32; u++) {
    const int j = j0 + u;
    const uint32_t su = (uint32_t)Sh[j * 128 + i] << 16;
    float s = __uint_as_float(su) * alpha;
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
    uint32_t u32 = __float_as_uint(s);
    uint16_t bf = (uint16_t)((u32 + 0x7fffu + ((u32 >> 16) & 1u)) >> 16);
    Sh[j * 128 + i] = bf;
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

__global__ void k_f32_to_bf16_vec(const float* __restrict__ src, uint16_t* __restrict__ dst,
                                  size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) {
    uint32_t u = __float_as_uint(src[i]);
    dst[i] = (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
}

__global__ void k_bf16_to_f32_vec(const uint16_t* __restrict__ src, float* __restrict__ dst,
                                  size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t st = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += st) {
    dst[i] = __uint_as_float((uint32_t)src[i] << 16);
  }
}

// Two-kernel split of k_gdn_step_gn_128 (QWENOX_GDN_STEP_F3): phase kernel
// (192 blocks, same wave-aligned occupancy) ends after the yv partial write —
// no threadfence/atomic at all — and the tail RMSNorm runs as a separate tiny
// 48-block kernel on the same ytmp buffer with the exact same offset/shfl
// reduction order as _128's winner block. Bit-identical to k_gdn_step_gn_128.
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

// Tail RMSNorm of the F3 split: identical reduction order to _128's winner.
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

// Single-block-per-head variant of k_gdn_step_gn_128 (48 blocks of 256 threads;
// thread = 2 consecutive columns x its jc 32-row shard with float2 S access).
// Per-column partials keep the exact 32-row chains and the p0+p1+p2+p3 add
// order, and the tail RMSNorm runs in LDS replicating the same offset-64/32
// adds + shfl chain -> bit-identical, with 0 threadfence/atomic/ytmp per head.
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
  const float kvfy = part[0][i2 + 1] + part[1][i2 + 1] + part[2][i2 + 1] + part[3][i2 + 1];
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

// Short speculative batches: keep the entire recurrent state in registers.
// The four partial reductions and accumulation order follow k_gdn_step.
// Thread map: the four jc shards of one column i sit in adjacent lanes of
// one warp (i = warp * warpSize/4 + lane/4, jc = lane%4), so both part[0..3]
// reductions run as four __shfl_sync fetches summed in the same left-to-right
// order (bit-identical to the shared-memory version) and the row loop needs
// no __syncthreads at all. qkv and out may alias: each row's value is read
// before the first shfl of that row and the output store happens after the
// second, inside the same warp.
// Gated delta recurrence over P rows, one block (512 threads) per v-head h —
// the single source of the GDN verify / replay arithmetic (10-12):
//   k_gdn_verify        in place (ngram verify: output + S write-back)
//   k_gdn_verify_ro     spec verify: output only, S left at the pre-round state
//   k_gdn_commit        after acceptance: replay the committed rows, write S
// so a committed S is bit-identical whichever kernel produced it (and to the
// former k_gdn_snap rows: same per-row update expressions and order).
// Row t inputs: k at kb + t*rs + (h/3)*128, v[i] at vb + t*rs + h*128 + i,
// log-decay g at gb + t*gs + h, beta at bb + t*gs + h; OUT: q at
// qb + t*rs + (h/3)*128 and out row at ob + t*rs + h*128 (may alias v).
template <bool kOut, bool kWs, typename ST = float>
__device__ __forceinline__ void gdn_rec_body(const int h, const float* qb, const float* kb,
                                             const float* vb, size_t rs, const float* gb,
                                             const float* bb, int gs, ST* S, float* ob,
                                             int P) {
  const int lane = threadIdx.x % warpSize;
  const int i = (threadIdx.x / warpSize) * (warpSize / 4) + (lane >> 2);
  const int jc = lane & 3, j0 = jc * 32, jb = lane & ~3;
  ST* Sh = S + (size_t)h * 128 * 128;
  float state[32];
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    if constexpr (sizeof(ST) == 2) state[j] = bf2f(Sh[(j0 + j) * 128 + i]);
    else state[j] = Sh[(j0 + j) * 128 + i];
  }
  for (int row = 0; row < P; ++row) {
    const float* k = kb + (size_t)row * rs + (h / 3) * 128;
    const float value = vb[(size_t)row * rs + h * 128 + i];
    const float alpha = expf(gb[(size_t)row * gs + h]);
    const float beta = bb[(size_t)row * gs + h];
    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      state[j] *= alpha;
      kv += state[j] * k[j0 + j];
    }
    const float kv0 = __shfl_sync(~0ull, kv, jb + 0);
    const float kv1 = __shfl_sync(~0ull, kv, jb + 1);
    const float kv2 = __shfl_sync(~0ull, kv, jb + 2);
    const float kv3 = __shfl_sync(~0ull, kv, jb + 3);
    const float delta = (value - (kv0 + kv1 + kv2 + kv3)) * beta;
    if constexpr (kOut) {
      const float* q = qb + (size_t)row * rs + (h / 3) * 128;
      float result = 0.f;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        state[j] += k[j0 + j] * delta;
        result += state[j] * q[j0 + j];
      }
      const float r0 = __shfl_sync(~0ull, result, jb + 0);
      const float r1 = __shfl_sync(~0ull, result, jb + 1);
      const float r2 = __shfl_sync(~0ull, result, jb + 2);
      const float r3 = __shfl_sync(~0ull, result, jb + 3);
      if (jc == 0) ob[(size_t)row * rs + h * 128 + i] = r0 + r1 + r2 + r3;
    } else {
#pragma unroll
      for (int j = 0; j < 32; ++j) state[j] += k[j0 + j] * delta;
    }
  }
  if constexpr (kWs) {
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      if constexpr (sizeof(ST) == 2) Sh[(j0 + j) * 128 + i] = f2bf(state[j]);
      else Sh[(j0 + j) * 128 + i] = state[j];
    }
  }
}
// qkv-row form of the body (rows of [q 2048 | k 2048 | v 6144], stride).
template <bool kOut, bool kWs, typename ST = float>
__device__ __forceinline__ void gdn_verify_body(const int h, const float* qkv,
                                                const float* g48, const float* b48, ST* S,
                                                float* out, int P, int stride) {
  gdn_rec_body<kOut, kWs, ST>(h, qkv, qkv + 2048, qkv + 4096, (size_t)stride, g48, b48, 48, S,
                              out + 4096, P);
}
__global__ void k_gdn_verify(const float* qkv, const float* g48,
                             const float* b48, float* S, float* out,
                             int P, int stride) {
  gdn_verify_body<true, true>(blockIdx.x, qkv, g48, b48, S, out, P, stride);
}
// Spec verify: outputs only; S stays at the round start until k_gdn_commit.
__global__ void k_gdn_verify_ro(const float* qkv, const float* g48, const float* b48,
                                float* S, float* out, int P, int stride) {
  gdn_verify_body<true, false>(blockIdx.x, qkv, g48, b48, S, out, P, stride);
}
__global__ void k_gdn_verify_ro_bf16(const float* qkv, const float* g48, const float* b48,
                                     uint16_t* S, float* out, int P, int stride) {
  gdn_verify_body<true, false, uint16_t>(blockIdx.x, qkv, g48, b48, S, out, P, stride);
}
// Per-row replay inputs of a spec verify round, layout per (row, GDN layer):
// [k 2048 | v 6144 | g 48 | beta 48] floats (post conv/l2norm, post gates).
constexpr int kGdnRowF = 8288;
__global__ void k_gdn_rowsave(const float* __restrict__ convb, const float* __restrict__ g48,
                              const float* __restrict__ b48, float* __restrict__ dst,
                              size_t rs, int P) {
  const int t = blockIdx.x;
  if (t >= P) return;
  const float* src = convb + (size_t)t * 10240 + 2048;
  float* d = dst + (size_t)t * rs;
  for (int c = threadIdx.x; c < 2048; c += blockDim.x)  // k|v as float4
    *(float4*)(d + c * 4) = *(const float4*)(src + c * 4);
  if (threadIdx.x < 48) {
    d[8192 + threadIdx.x] = g48[t * 48 + threadIdx.x];
    d[8240 + threadIdx.x] = b48[t * 48 + threadIdx.x];
  }
}
// Install the state after the first n committed rows: grid (48, ngdn), rows
// at rows + t*rs + gi*kGdnRowF, S layer gi at S + gi*48*128*128.
__global__ void k_gdn_commit(const float* __restrict__ rows, size_t rs, float* S, int n) {
  const int gi = blockIdx.y;
  const float* r = rows + (size_t)gi * kGdnRowF;
  gdn_rec_body<false, true>(blockIdx.x, nullptr, r, r + 2048, rs, r + 8192, r + 8240, (int)rs,
                            S + (size_t)gi * 48 * 128 * 128, nullptr, n);
}
__global__ void k_gdn_commit_bf16(const float* __restrict__ rows, size_t rs, uint16_t* S, int n) {
  const int gi = blockIdx.y;
  const float* r = rows + (size_t)gi * kGdnRowF;
  gdn_rec_body<false, true, uint16_t>(blockIdx.x, nullptr, r, r + 2048, rs, r + 8192, r + 8240,
                                      (int)rs, S + (size_t)gi * 48 * 128 * 128, nullptr, n);
}

// --- indexer proj snapshot: keep every verify row's raw ring value (the
// 128-float tail of the 640-wide proj) so a rollback can rebuild the 4-slot
// raw ring at any accepted depth.
__global__ void k_iproj_snap(const float* __restrict__ proj,
                             float* __restrict__ snap, int P) {
  int t = blockIdx.x, d = threadIdx.x;
  if (t < P) snap[(size_t)t * 128 + d] = proj[(size_t)t * 640 + 512 + d];
}

// --- GDN snapshot: serially advance S over the chunk's P rows (same math as
// k_gdn_step, no output), writing S after each row into snap[t]. S lives in
// registers (32/thread) and d_S is left untouched — the chunked scan after
// this kernel still sees and advances the pre-chunk state. Lets speculative
// rollback install the state at any accepted depth without a re-prefill.
// snap t-stride = ngdn*48*128*128.
__device__ __forceinline__ void gdn_snap_body(
    const int h, const float* __restrict__ qkv, const float* __restrict__ g48,
    const float* __restrict__ b48, const float* __restrict__ S,
    float* __restrict__ snap, int P, int qkvstride, size_t snaptstride) {
  // h: 48 v-heads, group of 3 over 16 k-heads
  const int dk = 128, dv = 128;
  const float* Sh = S + (size_t)h * dk * dv;
  float* Sn = snap + (size_t)h * dk * dv;
  int t = threadIdx.x;
  int i = t % dv, jc = t / dv;
  int j0 = jc * (dk / 4);
  float sreg[dk / 4];
#pragma unroll
  for (int j = 0; j < dk / 4; j++) sreg[j] = Sh[(j0 + j) * dv + i];
  __shared__ float part[4][128];
  __shared__ float sdelta[128];
  for (int tt = 0; tt < P; tt++) {
    const float* row = qkv + (size_t)tt * qkvstride;
    const float* kh = row + 2048 + (h / 3) * dk;
    const float* vh = row + 4096 + h * dv;
    float alpha = expf(g48[tt * 48 + h]);
    float be = b48[tt * 48 + h];
    float kv = 0.f;
#pragma unroll
    for (int j = 0; j < dk / 4; j++) {
      sreg[j] *= alpha;
      kv += sreg[j] * kh[j0 + j];
    }
    part[jc][i] = kv;
    __syncthreads();
    float kvf = part[0][i] + part[1][i] + part[2][i] + part[3][i];
    float delta = (vh[i] - kvf) * be;
    if (jc == 0) sdelta[i] = delta;
    __syncthreads();
    delta = sdelta[i];
#pragma unroll
    for (int j = 0; j < dk / 4; j++) {
      sreg[j] += kh[j0 + j] * delta;
      Sn[(size_t)tt * snaptstride + (j0 + j) * dv + i] = sreg[j];
    }
    __syncthreads();
  }
}
__global__ void k_gdn_snap(const float* __restrict__ qkv,
                           const float* __restrict__ g48,
                           const float* __restrict__ b48,
                           const float* __restrict__ S, float* __restrict__ snap,
                           int P, int qkvstride, size_t snaptstride) {
  gdn_snap_body(blockIdx.x, qkv, g48, b48, S, snap, P, qkvstride, snaptstride);
}

// --- conv state snapshot: convst after row t = rows (t-2,t-1,t) of the raw
// qkv stream, with the live (pre-chunk) convst filling rows < 0. snap layout
// per t: [3][qkv], t-stride = ngdn*3*qkv.
__device__ __forceinline__ void convst_snap_body(
    const int t, const float* __restrict__ qkvb, const float* __restrict__ convst,
    float* __restrict__ snap, int qkv, size_t snaptstride) {
  float* out = snap + (size_t)t * snaptstride;
  for (int c = threadIdx.x; c < qkv; c += blockDim.x) {
    out[c] = t >= 2   ? qkvb[(size_t)(t - 2) * qkv + c]
             : t == 1 ? convst[2 * qkv + c]
                      : convst[qkv + c];
    out[qkv + c] = t >= 1 ? qkvb[(size_t)(t - 1) * qkv + c] : convst[2 * qkv + c];
    out[2 * qkv + c] = qkvb[(size_t)t * qkv + c];
  }
}
__global__ void k_convst_snap(const float* __restrict__ qkvb,
                              const float* __restrict__ convst,
                              float* __restrict__ snap, int P, int qkv,
                              size_t snaptstride) {
  convst_snap_body(blockIdx.x, qkvb, convst, snap, qkv, snaptstride);
}

// --- indexer raw-ring rebuild after rollback: slot s must hold the proj row
// of the largest position p <= qpos with p%4==s (qpos = q+a+1). Rows from the
// verify chunk come from iproj_snap[qi][p-qbase], older rows from the
// checkpoint ring. One block per QSA layer.
__global__ void k_iraw_restore(const float* __restrict__ iproj_snap,
                               const float* __restrict__ ckpt_iraw,
                               float* __restrict__ iraw, int qbase, int qpos,
                               int snapstride) {
  int qi = blockIdx.x, d = threadIdx.x;
  const float* snapl = iproj_snap + (size_t)qi * snapstride * 128;
  for (int s = 0; s < 4; s++) {
    int p = qpos - ((qpos % 4 - s + 4) % 4);
    const float* src = p >= qbase
                           ? snapl + (size_t)(p - qbase) * 128
                           : ckpt_iraw + (size_t)qi * 512 + (size_t)(p % 4) * 128;
    iraw[(size_t)qi * 512 + s * 128 + d] = src[d];
  }
}

// --- GDN q/k L2 norm + q scale: one block per qk head
__global__ void k_l2norm_qk(float* __restrict__ q, float* __restrict__ k, int dk,
                            float qscale) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  float* qh = q + h * dk;
  float* kh = k + h * dk;
  float v = (t < dk) ? qh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float iq = rsqrtf(sm[0] + 1e-6f);
  __syncthreads();
  float kv = (t < dk) ? kh[t] : 0.f;
  sm[t] = kv * kv;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float ik = rsqrtf(sm[0] + 1e-6f);
  if (t < dk) {
    qh[t] = v * iq * qscale;
    kh[t] = kv * ik;
  }
}

// --- GDN gates: alpha_decay[g] = -exp(A_log)*softplus(a+dt_bias) (= log decay),
//     beta = sigmoid(b).  hv threads.  Batched: blockIdx.x = token row (stride).
__global__ void k_gdn_gates(const float* __restrict__ a, const float* __restrict__ b,
                            const float* __restrict__ A_log,
                            const float* __restrict__ dt_bias,
                            float* __restrict__ g_out, float* __restrict__ beta_out,
                            int hv, int stride) {
  int h = threadIdx.x;
  if (h >= hv) return;
  size_t off = (size_t)blockIdx.x * stride;
  a += off;
  b += off;
  g_out += off;
  beta_out += off;
  float sp = a[h] + dt_bias[h];
  sp = sp > 20.f ? sp : log1pf(expf(sp));
  g_out[h] = -expf(A_log[h]) * sp;
  beta_out[h] = 1.f / (1.f + expf(-b[h]));
}

// --- GDN gated norm: per v-head plain RMSNorm(128) * sigmoid(z)
// Batched: grid.y = token row; y/out row stride ys, z row stride zs.
// Bf16: additionally emit compact [grid.y, gridDim.x*dv] bf16 (same f2bf RNE
// as k_f32_to_bf16 on out, so the consumer GEMM can skip its conversion).
template <bool Bf16 = false>
__global__ void k_gdn_gatednorm(const float* __restrict__ y, const float* __restrict__ z,
                                const float* __restrict__ w, float* __restrict__ out,
                                int dv, float eps, int ys, int zs,
                                uint16_t* __restrict__ bf16 = nullptr) {
  __shared__ float sm[128];
  int h = blockIdx.x;
  int t = threadIdx.x;
  const float* yh = y + (size_t)blockIdx.y * ys + h * dv;
  const float* zh = z + (size_t)blockIdx.y * zs + h * dv;
  float* oh = out + (size_t)blockIdx.y * ys + h * dv;
  float v = (t < dv) ? yh[t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / dv + eps);
  if (t < dv) {
    float zz = zh[t];
    float value = v * inv * w[t] / (1.f + expf(-zz));
    oh[t] = value;
    if (Bf16) bf16[((size_t)blockIdx.y * gridDim.x + h) * dv + t] = f2bf(value);
  }
}

// --- GDN gated norm, warp-per-token (dv = 128 fixed): one wave32 per
// (token, head); lane owns dims lane, lane+32, lane+64, lane+96. Reduction
// reproduces the smem tree of k_gdn_gatednorm exactly: the off=64 layer
// pairs (d, d+64), off=32 adds the (d+32, d+96) pair, shfl_down 16..1 does
// the rest -> bit-exact, but one block covers 32 tokens instead of 1.
// grid (heads, ceil(P/32)), block 128 (4 warps); warp w takes tokens
// blockIdx.y*32 + k*4 + w, k = 0..7.
template <bool Bf16 = false, bool Split2 = false>
__global__ void k_gdn_gatednorm_b(const float* __restrict__ y,
                                  const float* __restrict__ z,
                                  const float* __restrict__ w,
                                  float* __restrict__ out, float eps, int ys, int zs,
                                  uint16_t* __restrict__ bf16, int P) {
  const int h = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const float w0 = w[lane], w1 = w[lane + 32], w2 = w[lane + 64],
              w3 = w[lane + 96];
#pragma unroll 2
  for (int k = 0; k < 8; k++) {
    int t = blockIdx.y * 32 + k * 4 + warp;  // warp-uniform
    if (t >= P) return;
    const float* yh = y + (size_t)t * ys + h * 128;
    const float* zh = z + (size_t)t * zs + h * 128;
    float* oh = out ? out + (size_t)t * ys + h * 128 : nullptr;
    float v0 = yh[lane], v1 = yh[lane + 32], v2 = yh[lane + 64],
          v3 = yh[lane + 96];
    // Bit-exactness vs the smem tree: products must round individually before
    // the adds, but ffp-contract=fast would fuse mul+add into FMA (the old
    // kernel is saved by its smem round-trip). The empty asm barrier
    // materializes each product as a rounded VGPR value, no memory traffic.
    float q0 = v0 * v0, q1 = v1 * v1, q2 = v2 * v2, q3 = v3 * v3;
    asm volatile("" : "+v"(q0), "+v"(q1), "+v"(q2), "+v"(q3));
    float s64 = q0 + q2;
    float s96 = q1 + q3;
    float s = s64 + s96;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(~0ull, s, off);
    s = __shfl_sync(~0ull, s, 0);
    float inv = rsqrtf(s / 128 + eps);
    float r0 = v0 * inv * w0 / (1.f + expf(-zh[lane]));
    float r1 = v1 * inv * w1 / (1.f + expf(-zh[lane + 32]));
    float r2 = v2 * inv * w2 / (1.f + expf(-zh[lane + 64]));
    float r3 = v3 * inv * w3 / (1.f + expf(-zh[lane + 96]));
    if (out) {  // fp32 out is dead when the consumer GEMM reads only the bf16
      oh[lane] = r0;
      oh[lane + 32] = r1;
      oh[lane + 64] = r2;
      oh[lane + 96] = r3;
    }
    if (Bf16) {
      uint16_t* bp = Split2
          ? (bf16 + (h >= 24 ? (size_t)P * 3072 : 0) + (size_t)t * 3072 +
             (h >= 24 ? h - 24 : h) * 128)
          : (bf16 + ((size_t)t * gridDim.x + h) * 128);
      bp[lane] = f2bf(r0);
      bp[lane + 32] = f2bf(r1);
      bp[lane + 64] = f2bf(r2);
      bp[lane + 96] = f2bf(r3);
    }
  }
}

// --- RoPE partial: rotate first rotary_dim dims of each head (half-split).
// vec: (nheads, dh). One warp per head... simple: one thread per pair.
// d_rdelta (M-RoPE decode): rope angle position is *d_pos + *d_rdelta;
// absent/zero for text-only requests (bitwise identical to before).
__global__ void k_rope(float* __restrict__ vec, int nheads, int dh, int rotary,
                       const int* __restrict__ d_pos, double theta,
                       const int* __restrict__ d_rdelta = nullptr) {
  int pos = *d_pos;
  if (d_rdelta) pos += *d_rdelta;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  // over nheads * rotary/2
  int half = rotary / 2;
  if (idx >= nheads * half) return;
  int h = idx / half, i = idx % half;
  double ang = pos * g_rope_tab[i];
  float cs = (float)cos(ang), sn = (float)sin(ang);
  if (g_rope_mscale != 1.f) cs *= g_rope_mscale, sn *= g_rope_mscale;
  float* p = vec + h * dh;
  float x0 = p[i], x1 = p[i + half];
  p[i] = x0 * cs - x1 * sn;
  p[i + half] = x0 * sn + x1 * cs;
  (void)theta;
}

// --- QSA attention step: one block per q head, threads over positions.
// kcache/vcache: (pos, hkv*dh) accumulated; current token's k/v already appended.
template <typename KVT = float>
__global__ void k_qsa_step(const float* __restrict__ q, const KVT* __restrict__ kc,
                           const KVT* __restrict__ vc, float* __restrict__ out,
                           const int* __restrict__ d_pos, int dh, int hkv, int hq,
                           const int* selected = nullptr, bool skip_sparse = false,
                           const int* __restrict__ ptab = nullptr) {
  int visible = *d_pos + 1;
  const int tokens = qsa_index_tokens();
  bool sparse = selected && visible >= tokens + 4;
  if (sparse && skip_sparse) return;
  int ntok = sparse ? tokens + visible % 4 : visible;
  int h = blockIdx.x;
  int kvh = h / (hq / hkv);
  const float* qh = q + h * dh;
  float scale = 1.f / sqrtf((float)dh);
  // pass 1: scores + running max (threads strided over positions)
  int t = threadIdx.x;
  float mx = -1e30f;
  for (int p = t; p < ntok; p += blockDim.x) {
    const auto kp = kv_view(kc) + ((size_t)kv_row(ptab, qsa_source_pos(p, visible, selected)) * hkv + kvh) * dh;
    float acc = 0.f;
    for (int i = 0; i < dh; i++) acc += qh[i] * kv_ld1(kp + i);
    mx = fmaxf(mx, acc * scale);
  }
  // block max reduce
  __shared__ float red[1024];
  red[t] = mx;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) red[t] = fmaxf(red[t], red[t + off]);
    __syncthreads();
  }
  mx = red[0];
  __syncthreads();
  // pass 2: exp sum
  float se = 0.f;
  for (int p = t; p < ntok; p += blockDim.x) {
    const auto kp = kv_view(kc) + ((size_t)kv_row(ptab, qsa_source_pos(p, visible, selected)) * hkv + kvh) * dh;
    float acc = 0.f;
    for (int i = 0; i < dh; i++) acc += qh[i] * kv_ld1(kp + i);
    se += expf(acc * scale - mx);
  }
  red[t] = se;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) red[t] += red[t + off];
    __syncthreads();
  }
  float invsum = 1.f / red[0];
  // pass 3: weighted v — threads over dh? need per-dim sums. dh=256, block 256.
  // recompute weights per position serially but parallel over dh.
  if (t < dh) {
    float acc = 0.f;
    for (int p = 0; p < ntok; p++) {
      int source = kv_row(ptab, qsa_source_pos(p, visible, selected));
      const auto kp = kv_view(kc) + ((size_t)source * hkv + kvh) * dh;
      float s = 0.f;
      for (int i = 0; i < dh; i++) s += qh[i] * kv_ld1(kp + i);
      float w = expf(s * scale - mx) * invsum;
      acc += w * kv_ld1(kv_view(vc) + ((size_t)source * hkv + kvh) * dh + t);
    }
    out[h * dh + t] = acc;
  }
}

// --- per-head zc rmsnorm for QSA q/k (dh=256, one block per head)
// (reuse k_rmsnorm_zc with n=256 per block over head pointer)

// --- sigmoid gate multiply: out[i] = a[i] * sigmoid(g[i])
__global__ void k_sigmoid_gate(const float* __restrict__ a, const float* __restrict__ g,
                               float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] / (1.f + expf(-g[i]));
}

// --- GR read: needs gemv results; combine on device.
// t = silu(Wd Rhat / 4) computed via gemv into t320 then k_silu_scale.
__global__ void k_silu_scale(float* __restrict__ v, float inv_div, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = v[i] * inv_div;
    v[i] = x / (1.f + expf(-x));
  }
}
__global__ void k_sigmoid_inplace(float* __restrict__ v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = 1.f / (1.f + expf(-v[i]));
}
// x_out[i] = mean_b G[b*d+i] * Rhat[b*d+i]
// kSig: apply the k_sigmoid_inplace expression to each G element before the
// product (same expression/order -> bit-identical to the separate launch).
template <bool kSig = false>
__global__ void k_gr_combine(const float* __restrict__ G, const float* __restrict__ Rhat,
                             float* __restrict__ x, int d, int branches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  float acc = 0.f;
  for (int b = 0; b < branches; b++) {
    float g = G[b * d + i];
    if (kSig) g = 1.f / (1.f + expf(-g));
    acc += g * Rhat[b * d + i];
  }
  x[i] = acc / branches;
}
// GR write: s = 2*sigmoid(w/4); R[b*d+i] += s[b] * y[i]
__global__ void k_gr_write(const float* __restrict__ w4, float* __restrict__ R,
                           const float* __restrict__ y, int d, int branches) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= d) return;
  for (int b = 0; b < branches; b++) {
    float s = 2.f / (1.f + expf(-w4[b] * 0.25f));
    R[b * d + i] += s * y[i];
  }
}

// --- MoE router: softmax + top-k done on host (512 floats download is cheap),
//     experts gathered per selected expert via row-offset GEMV.
// silu(gate)*up for mid dims
__global__ void k_silu_mul(const float* __restrict__ g, const float* __restrict__ u,
                           float* __restrict__ out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a = g[i];
    out[i] = a / (1.f + expf(-a)) * u[i];
  }
}
// acc[r] += w * y[r]
__global__ void k_axpy(float* __restrict__ acc, const float* __restrict__ y, float w,
                       int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += w * y[i];
}

// --- GPU router: softmax + top-k over n logits, one block. ------------------
// out: ids[k], w[k] with w = p_e / sum(selected p) (official norm_topk_prob).
// Batched: one block per token, ids/ws rows of 16.
__global__ void k_router_topk(const float* __restrict__ logits, int* __restrict__ ids,
                              float* __restrict__ ws, int n, int k, VizTap vt) {
  logits += (size_t)blockIdx.x * n;
  ids += (size_t)blockIdx.x * 16;
  ws += (size_t)blockIdx.x * 16;
  __shared__ float p[512];
  __shared__ float sval[512];
  __shared__ int sidx[512];
  __shared__ float raw[16];
  int t = threadIdx.x;
  float v = logits[t];  // n == 512 == blockDim
  // block max
  sval[t] = v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sval[t] = fmaxf(sval[t], sval[t + off]);
    __syncthreads();
  }
  float mx = sval[0];
  p[t] = expf(v - mx);
  __syncthreads();
  sval[t] = p[t];
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sval[t] += sval[t + off];
    __syncthreads();
  }
  float inv_se = 1.f / sval[0];
  p[t] *= inv_se;
  __syncthreads();
  float vz_ent = 0.f;
  if (vt.ring) {  // QWENOX_VIZ only: router softmax entropy (outputs untouched)
    const float q = p[t];
    sval[t] = q > 0.f ? -q * __logf(q) : 0.f;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (t < off) sval[t] += sval[t + off];
      __syncthreads();
    }
    vz_ent = sval[0];
    __syncthreads();
  }
  for (int j = 0; j < k; j++) {
    sval[t] = p[t];
    sidx[t] = t;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
      if (t < off && sval[t + off] > sval[t]) {
        sval[t] = sval[t + off];
        sidx[t] = sidx[t + off];
      }
      __syncthreads();
    }
    if (t == 0) {
      ids[j] = sidx[0];
      raw[j] = sval[0];
      p[sidx[0]] = -1e30f;
    }
    __syncthreads();
  }
  if (t == 0) {
    float sw = 0.f;
    for (int j = 0; j < k; j++) sw += raw[j];
    for (int j = 0; j < k; j++) ws[j] = raw[j] / sw;
    if (vt.ring) viz::store_route(vt, blockIdx.x, ids, ws, k, sw, vz_ent);
  }
}

// Batched k_router_topk, one wave32 per token (aggr 10-05). n == 512, k <= 16.
// Bit-identical to k_router_topk: lane l holds logits l + 32*j (j = 0..15);
// the softmax sum replays the 512-wide LDS tree order exactly (off 256..32 are
// in-lane j-pair adds, off 16..1 are shfl_down adds on lane 0's chain), max is
// order-free, and each top-k round is the same argmax (largest p, ties -> the
// lowest index, which is what the strict-> tree keeps). The old kernel spends
// ~100 block barriers per token (2.25 ms/layer at P=16K); this one none.
template <int k>
__global__ void __launch_bounds__(256) k_router_topk_w32(const float* __restrict__ logits,
                                                         int* __restrict__ ids,
                                                         float* __restrict__ ws, int P,
                                                         VizTap vt) {
  const int lane = threadIdx.x & 31;
  const int tok = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (tok >= P) return;
  logits += (size_t)tok * 512;
  ids += (size_t)tok * 16;
  ws += (size_t)tok * 16;
  float v[16], p[16];
#pragma unroll
  for (int j = 0; j < 16; j++) v[j] = logits[lane + 32 * j];
  float mx = v[0];
#pragma unroll
  for (int j = 1; j < 16; j++) mx = fmaxf(mx, v[j]);
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
#pragma unroll
  for (int j = 0; j < 16; j++) p[j] = expf(v[j] - mx);
  // tree sum in the LDS kernel's order: s[t] += s[t+off] for off = 256..1
  float a[16];
#pragma unroll
  for (int j = 0; j < 16; j++) a[j] = p[j];
#pragma unroll
  for (int j = 0; j < 8; j++) a[j] += a[j + 8];   // off 256
#pragma unroll
  for (int j = 0; j < 4; j++) a[j] += a[j + 4];   // off 128
#pragma unroll
  for (int j = 0; j < 2; j++) a[j] += a[j + 2];   // off 64
  a[0] += a[1];                                    // off 32
  float s = a[0];
#pragma unroll
  for (int o = 16; o; o >>= 1) s += __shfl_down(s, o, 32);  // off 16..1 (lane 0 chain)
  s = __shfl(s, 0, 32);
  const float inv_se = 1.f / s;
#pragma unroll
  for (int j = 0; j < 16; j++) p[j] *= inv_se;
  float vz_ent = 0.f;
  if (vt.ring) {  // QWENOX_VIZ only: router softmax entropy (outputs untouched)
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
    if (vt.ring) viz::store_route(vt, tok, ids, ws, k, sw, vz_ent);
  }
}

// Sort the P*k (expert, slot) pairs by expert id (ties by slot index) so
// same-expert slots become adjacent in k_q4cp_gemv_gg's warp order: the
// duplicate expert weight reads then execute in the same scheduling window
// and get served by L2 / in-flight misses instead of DRAM (verify P=5:
// union/slots ≈ 0.63, unsorted L2 catch only ~35%). Pure scheduling
// permutation — every output element is computed by the same expression, so
// results are bit-identical. Single block, bitonic over 256 padded keys
// (key = expert << 8 | slot; sentinel UINT32_MAX sinks to the tail).
__global__ void k_moe_sort_slots(const int* __restrict__ ids, int P, int k,
                                 int id_stride, int* __restrict__ order) {
  __shared__ unsigned keys[256];
  const int n = P * k, t = threadIdx.x;
  for (int i = t; i < 256; i += blockDim.x) {
    unsigned key = 0xFFFFFFFFu;
    if (i < n)
      key = ((unsigned)ids[(i / k) * id_stride + (i % k)] << 8) | (unsigned)i;
    keys[i] = key;
  }
  __syncthreads();
  for (int size = 2; size <= 256; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = t; i < 256; i += blockDim.x) {
        int j = i ^ stride;
        if (j > i) {
          bool up = (i & size) == 0;
          unsigned a = keys[i], b = keys[j];
          if ((a > b) == up) {
            keys[i] = b;
            keys[j] = a;
          }
        }
      }
      __syncthreads();
    }
  for (int i = t; i < n; i += blockDim.x) order[i] = (int)(keys[i] & 0xFFu);
}

// --- grouped expert GEMV (shared x): slot-major y[(slot, r)] ---------------
// Batched: nslots = P*kpt; slot -> token tok = slot/kpt, expert s = slot%kpt;
// x = xs + tok*x_stride, expert id = ids[tok*id_stride + s].
// v3 (decode GEMV governance): cb in LDS + 2 rows per warp; bit-exact vs the
// original (same per-lane fma order and reduction). Grid must now cover
// nslots * ceil(rows_per/2) warps, not nslots * rows_per.
template <int GSH = 0>
__global__ void k_q4cp_gemv_gg(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const float* __restrict__ x, float* __restrict__ y,
                               const int* __restrict__ ids, uint64_t rows_per,
                               uint64_t cols, uint64_t scale_stride, int nslots,
                               uint64_t x_stride, int id_stride, int kpt,
                               const int* __restrict__ slot_order = nullptr) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t pairs_per_slot = (rows_per + 1) / 2;
  uint64_t gw = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)nslots * pairs_per_slot) return;
  int slot = gw / pairs_per_slot;
  if (slot_order) slot = slot_order[slot];  // 同专家相邻（纯调度排列，逐 bit）
  int tok = slot / kpt, s = slot - tok * kpt;
  uint64_t row0 =
      (uint64_t)ids[tok * id_stride + s] * rows_per + (gw % pairs_per_slot) * 2;
  int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = (gw % pairs_per_slot) * 2 + 1 < rows_per;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  const float* xt = x + (size_t)tok * x_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < cols / 32; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + (g >> GSH) * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + (g >> GSH) * 2));
    }
    const float* xg = xt + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    uint64_t out = (uint64_t)slot * rows_per + (gw % pairs_per_slot) * 2;
    y[out] = acc0;
    if (has1) y[out + 1] = acc1;
  }
}

// diagnostic (QWENOX_MOE_UNIQ): hist[P*161 + D] += 1, D = distinct expert ids among P*k slots
__global__ void k_moe_uniq_hist(const int* __restrict__ ids, int P, int k, int id_stride,
                                unsigned* __restrict__ hist) {
  __shared__ unsigned char seen[512];
  for (int i = threadIdx.x; i < 512; i += blockDim.x) seen[i] = 0;
  __syncthreads();
  for (int i = threadIdx.x; i < P * k; i += blockDim.x) seen[ids[(i / k) * id_stride + (i % k)] & 511] = 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    int d = 0;
    for (int i = 0; i < 512; i++) d += seen[i];
    if (P <= 16) atomicAdd(hist + P * 161 + d, 1u);
  }
}

// ---- expert-grouped verify gate_up (QWENOX_MOE_GG) -------------------------
// k_moe_group: one block. Sort the n = P*k slots by (expert, slot) and cut runs:
// group g = expert gexp[g] with gn[g] slots gsl[g*16 + 0..gn-1] (ascending slot,
// hence ascending token; a token's top-k experts are distinct so gn <= P <= 16).
__global__ void k_moe_group(const int* __restrict__ ids, int P, int k, int id_stride,
                            int* __restrict__ gexp, int* __restrict__ gn,
                            int* __restrict__ gsl, int* __restrict__ ngrp) {
  __shared__ unsigned keys[256];
  const int n = P * k, t = threadIdx.x;
  int m = 1;
  while (m < n) m <<= 1;
  for (int i = t; i < m; i += blockDim.x)
    keys[i] = i < n ? ((unsigned)ids[(i / k) * id_stride + (i % k)] << 8) | (unsigned)i : 0xFFFFFFFFu;
  __syncthreads();
  for (int size = 2; size <= m; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = t; i < m; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const bool up = (i & size) == 0;
          const unsigned a = keys[i], b = keys[j];
          if ((a > b) == up) { keys[i] = b; keys[j] = a; }
        }
      }
      __syncthreads();
    }
  if (t == 0) {
    int g = -1, last = -1;
    for (int i = 0; i < n; i++) {
      const int e = (int)(keys[i] >> 8);
      if (e != last) { g++; last = e; gexp[g] = e; gn[g] = 0; }
      gsl[g * 16 + gn[g]++] = (int)(keys[i] & 0xFFu);
    }
    *ngrp = g + 1;
  }
}

// Same output as k_moe_group (groups by ascending expert, slots ascending),
// by counting: cnt[e] via LDS atomics, group index = number of present experts
// below e (block scan of the 512 flags), rank in group = earlier slots with the
// same expert. blockDim 256, n = P*k <= 256, 512 experts.
__global__ __launch_bounds__(256) void k_moe_group2(const int* __restrict__ ids, int P, int k,
                                                    int id_stride, int* __restrict__ gexp,
                                                    int* __restrict__ gn, int* __restrict__ gsl,
                                                    int* __restrict__ ngrp) {
  __shared__ int cnt[512];
  __shared__ int gid[512];
  __shared__ int es[256];
  __shared__ int wsum[8];
  const int t = threadIdx.x, n = P * k;
  cnt[t] = 0;
  cnt[t + 256] = 0;
  __syncthreads();
  int e = -1;
  if (t < n) {
    e = ids[(t / k) * id_stride + (t % k)] & 511;
    es[t] = e;
    atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  // exclusive scan of flags over 512 experts: thread t owns experts 2t, 2t+1
  const int f0 = cnt[2 * t] > 0, f1 = cnt[2 * t + 1] > 0;
  int v = f0 + f1, incl = v;
  const int lane = t & 31, w = t >> 5;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_up(incl, o, 32);
    if (lane >= o) incl += u;
  }
  if (lane == 31) wsum[w] = incl;
  __syncthreads();
  int base = 0;
  for (int i = 0; i < w; i++) base += wsum[i];
  const int ex = base + incl - v;
  gid[2 * t] = ex;
  gid[2 * t + 1] = ex + f0;
  if (t == 255) *ngrp = base + incl;
  __syncthreads();
  if (t < n) {
    int rank = 0;
    for (int j = 0; j < t; j++) rank += es[j] == e;
    const int g = gid[e];
    gsl[g * 16 + rank] = t;
    if (rank == 0) { gexp[g] = e; gn[g] = cnt[e]; }
  }
}

// grid (ceil(pairs/16), max groups); block-uniform group -> dead groups exit as
// whole blocks. Warp = rows (2rp, 2rp+1) of expert gexp[g] for up to TM slots
// at a time (more slots: another pass over the same rows).
template <int GSH, int TM>
__global__ __launch_bounds__(512) void k_q4cp_gemv_ggx(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const float* __restrict__ x, float* __restrict__ y,
    uint64_t rows_per, uint64_t cols, uint64_t scale_stride, uint64_t x_stride, int kpt,
    const int* __restrict__ gexp, const int* __restrict__ gnum, const int* __restrict__ gsl,
    const int* __restrict__ ngrp) {
  const int g = blockIdx.y;
  if (g >= *ngrp) return;
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  const uint64_t pairs_per_slot = (rows_per + 1) / 2;
  const uint64_t rp = (uint64_t)blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (rp >= pairs_per_slot) return;
  const int lane = threadIdx.x % warpSize;
  const uint64_t row0 = (uint64_t)gexp[g] * rows_per + rp * 2;
  const bool has1 = rp * 2 + 1 < rows_per;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  const int ns = gnum[g];
  const int* sl = gsl + g * 16;
  for (int b0 = 0; b0 < ns; b0 += TM) {
    const int nb = min(TM, ns - b0);
    const float* xt[TM];
#pragma unroll
    for (int q = 0; q < TM; q++) xt[q] = x + (size_t)(sl[b0 + (q < nb ? q : 0)] / kpt) * x_stride;
    float acc0[TM], acc1[TM];
#pragma unroll
    for (int q = 0; q < TM; q++) acc0[q] = acc1[q] = 0.f;
    for (uint64_t gi = lane; gi < cols / 32; gi += warpSize) {
      const uint4 pk0 = *(const uint4*)(rc0 + gi * 16);
      const float s0 = __half2float(*(const __half*)(rs0 + (gi >> GSH) * 2));
      uint4 pk1 = make_uint4(0, 0, 0, 0);
      float s1 = 0.f;
      if (has1) {
        pk1 = *(const uint4*)(rc1 + gi * 16);
        s1 = __half2float(*(const __half*)(rs1 + (gi >> GSH) * 2));
      }
      const uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
      const uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
      float a0[TM], a1[TM];
#pragma unroll
      for (int q = 0; q < TM; q++) a0[q] = a1[q] = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float c0[8], c1[8];
        uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          c0[j] = cbf[v0 & 0xf];
          c1[j] = cbf[v1 & 0xf];
          v0 >>= 4;
          v1 >>= 4;
        }
#pragma unroll
        for (int q = 0; q < TM; q++) {
          if (q < nb) {
            const float* xg = xt[q] + gi * 32 + i * 8;
            const float4 t0 = *(const float4*)xg, t1 = *(const float4*)(xg + 4);
            const float xv[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
#pragma unroll
            for (int j = 0; j < 8; j++) {
              a0[q] += c0[j] * xv[j];
              a1[q] += c1[j] * xv[j];
            }
          }
        }
      }
#pragma unroll
      for (int q = 0; q < TM; q++) {
        acc0[q] += a0[q] * s0;
        acc1[q] += a1[q] * s1;
      }
    }
#pragma unroll
    for (int q = 0; q < TM; q++)
      for (int off = warpSize / 2; off > 0; off >>= 1) {
        acc0[q] += __shfl_down_sync(~0ull, acc0[q], off);
        acc1[q] += __shfl_down_sync(~0ull, acc1[q], off);
      }
    if (lane == 0) {
#pragma unroll
      for (int q = 0; q < TM; q++) {
        if (q < nb) {
          const uint64_t out = (uint64_t)sl[b0 + q] * rows_per + rp * 2;
          y[out] = acc0[q];
          if (has1) y[out + 1] = acc1[q];
        }
      }
    }
  }
}

// --- pair slots that picked the same expert --------------------------------
// Sorts the P*kpt (token, expert) slots by expert id (bitonic, padded to the
// next power of two) and walks the sorted run pairing neighbours that share an
// expert. pa[g]/pb[g] are the slot ids of group g (pb < 0 = lone slot); *ngrp
// is the live group count. Every slot lands in exactly one group, so the gg2
// kernel still writes all P*kpt outputs. Two slots of one group always come
// from different tokens (a token's top-k experts are distinct).
__global__ void k_moe_pair_slots(const int* __restrict__ ids, int P, int k,
                                 int id_stride, int* __restrict__ pa,
                                 int* __restrict__ pb, int* __restrict__ ngrp) {
  __shared__ unsigned keys[256];
  const int n = P * k, t = threadIdx.x;
  int m = 1;
  while (m < n) m <<= 1;
  for (int i = t; i < m; i += blockDim.x) {
    unsigned key = 0xFFFFFFFFu;
    if (i < n)
      key = ((unsigned)ids[(i / k) * id_stride + (i % k)] << 8) | (unsigned)i;
    keys[i] = key;
  }
  __syncthreads();
  for (int size = 2; size <= m; size <<= 1)
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = t; i < m; i += blockDim.x) {
        int j = i ^ stride;
        if (j > i) {
          bool up = (i & size) == 0;
          unsigned a = keys[i], b = keys[j];
          if ((a > b) == up) {
            keys[i] = b;
            keys[j] = a;
          }
        }
      }
      __syncthreads();
    }
  if (t == 0) {
    int g = 0;
    for (int i = 0; i < n;) {
      unsigned ka = keys[i];
      if (i + 1 < n && (keys[i + 1] >> 8) == (ka >> 8)) {
        pa[g] = (int)(ka & 0xFFu);
        pb[g] = (int)(keys[i + 1] & 0xFFu);
        i += 2;
      } else {
        pa[g] = (int)(ka & 0xFFu);
        pb[g] = -1;
        i += 1;
      }
      g++;
    }
    *ngrp = g;
  }
}

// --- paired expert GEMV: one weight pass serves one or two slots -----------
// Bit-exact with k_q4cp_gemv_gg per slot (identical loop nest, fma order and
// reduction); the pair only shares the weight load, the scale and the codebook
// lookups. x is read per slot inside the i loop to keep the register footprint
// close to the single-slot kernel. The grid stays sized for the worst case
// (P*kpt groups) so it can live in a captured decode graph; warps above *ngrp
// exit immediately, which is where the saving comes from.
template <int GSH = 0>
__global__ void k_q4cp_gemv_gg2(const uint8_t* __restrict__ codes,
                                const uint8_t* __restrict__ scales,
                                const float* __restrict__ cb,
                                const float* __restrict__ x, float* __restrict__ y,
                                const int* __restrict__ ids, uint64_t rows_per,
                                uint64_t cols, uint64_t scale_stride,
                                uint64_t x_stride, int id_stride, int kpt,
                                const int* __restrict__ pa,
                                const int* __restrict__ pb,
                                const int* __restrict__ ngrp) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t pairs_per_slot = (rows_per + 1) / 2;
  uint64_t gw = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  int grp = (int)(gw / pairs_per_slot);
  if (grp >= *ngrp) return;
  const int sa = pa[grp], sb = pb[grp];
  const bool has_b = sb >= 0;
  const int toka = sa / kpt, ea = sa - toka * kpt;
  const uint64_t rp = gw % pairs_per_slot;
  const uint64_t row0 = (uint64_t)ids[toka * id_stride + ea] * rows_per + rp * 2;
  const int lane = threadIdx.x % warpSize;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  const bool has1 = rp * 2 + 1 < rows_per;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  const float* xta = x + (size_t)toka * x_stride;
  const float* xtb = has_b ? x + (size_t)(sb / kpt) * x_stride : xta;
  float acc0a = 0.f, acc1a = 0.f, acc0b = 0.f, acc1b = 0.f;
  for (uint64_t g = lane; g < cols / 32; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + (g >> GSH) * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + (g >> GSH) * 2));
    }
    const float* xga = xta + g * 32;
    const float* xgb = xtb + g * 32;
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0a = 0.f, a1a = 0.f, a0b = 0.f, a1b = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float xa[8], xb[8];
      {
        float4 t0 = *(const float4*)(xga + i * 8);
        float4 t1 = *(const float4*)(xga + i * 8 + 4);
        xa[0] = t0.x; xa[1] = t0.y; xa[2] = t0.z; xa[3] = t0.w;
        xa[4] = t1.x; xa[5] = t1.y; xa[6] = t1.z; xa[7] = t1.w;
      }
      if (has_b) {
        float4 t0 = *(const float4*)(xgb + i * 8);
        float4 t1 = *(const float4*)(xgb + i * 8 + 4);
        xb[0] = t0.x; xb[1] = t0.y; xb[2] = t0.z; xb[3] = t0.w;
        xb[4] = t1.x; xb[5] = t1.y; xb[6] = t1.z; xb[7] = t1.w;
      }
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const float c0 = cbf[v0 & 0xf], c1 = cbf[v1 & 0xf];
        a0a += c0 * xa[j];
        a1a += c1 * xa[j];
        if (has_b) {
          a0b += c0 * xb[j];
          a1b += c1 * xb[j];
        }
        v0 >>= 4;
        v1 >>= 4;
      }
    }
    acc0a += a0a * s0;
    acc1a += a1a * s1;
    acc0b += a0b * s0;
    acc1b += a1b * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0a += __shfl_down_sync(~0ull, acc0a, off);
    acc1a += __shfl_down_sync(~0ull, acc1a, off);
    acc0b += __shfl_down_sync(~0ull, acc0b, off);
    acc1b += __shfl_down_sync(~0ull, acc1b, off);
  }
  if (lane == 0) {
    uint64_t oa = (uint64_t)sa * rows_per + rp * 2;
    y[oa] = acc0a;
    if (has1) y[oa + 1] = acc1a;
    if (has_b) {
      uint64_t ob = (uint64_t)sb * rows_per + rp * 2;
      y[ob] = acc0b;
      if (has1) y[ob + 1] = acc1b;
    }
  }
}

// --- grouped silu(gate)*up: guv (nslots, 2*mid) -> hid (nslots, mid) --------
__global__ void k_silu_mul_g(const float* __restrict__ guv, float* __restrict__ hid,
                             int mid, int nslots) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nslots * mid) return;
  int slot = i / mid, r = i % mid;
  float a = guv[slot * 2 * mid + r];
  hid[i] = a / (1.f + expf(-a)) * guv[slot * 2 * mid + mid + r];
}

// --- grouped expert GEMV (per-slot x) with fused weighted accumulate --------
// acc[tok*acc_stride + r] += ws[tok*id_stride+s] * dot(W[ids[...]*rows_per + r],
// xs + slot*x_stride). v3: full warp per row, 2 rows per warp, cb in LDS;
// per-row math bit-exact vs the LPW=32 original on wave32 (the atomicAdd set
// per output element is unchanged). Grid must cover nslots*ceil(rows_per/2)
// warps.
template <int GSH = 0>
__global__ void k_q4cp_gemv_gd(const uint8_t* __restrict__ codes,
                               const uint8_t* __restrict__ scales,
                               const float* __restrict__ cb,
                               const float* __restrict__ xs, float* __restrict__ acc,
                               const int* __restrict__ ids, const float* __restrict__ ws,
                               uint64_t rows_per, uint64_t cols, uint64_t scale_stride,
                               uint64_t x_stride, int nslots, int id_stride,
                               uint64_t acc_stride, int kpt) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t pairs_per_slot = (rows_per + 1) / 2;
  int lane = threadIdx.x % warpSize;
  uint64_t gw = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)nslots * pairs_per_slot) return;
  int slot = gw / pairs_per_slot;
  uint64_t rp = gw % pairs_per_slot;
  int tok = slot / kpt, s = slot - tok * kpt;
  uint64_t row0 = (uint64_t)ids[tok * id_stride + s] * rows_per + rp * 2;
  const float* x = xs + (uint64_t)slot * x_stride;
  const uint8_t* rc0 = codes + row0 * (cols / 2);
  const uint8_t* rs0 = scales + row0 * scale_stride;
  bool has1 = rp * 2 + 1 < rows_per;
  const uint8_t* rc1 = codes + (row0 + 1) * (cols / 2);
  const uint8_t* rs1 = scales + (row0 + 1) * scale_stride;
  float acc0 = 0.f, acc1 = 0.f;
  for (uint64_t g = lane; g < cols / 32; g += warpSize) {
    uint4 pk0 = *(const uint4*)(rc0 + g * 16);
    float s0 = __half2float(*(const __half*)(rs0 + (g >> GSH) * 2));
    uint4 pk1 = make_uint4(0, 0, 0, 0);
    float s1 = 0.f;
    if (has1) {
      pk1 = *(const uint4*)(rc1 + g * 16);
      s1 = __half2float(*(const __half*)(rs1 + (g >> GSH) * 2));
    }
    const float* xg = x + g * 32;
    float xv[32];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float4 t = *(const float4*)(xg + i * 4);
      xv[i * 4] = t.x; xv[i * 4 + 1] = t.y; xv[i * 4 + 2] = t.z;
      xv[i * 4 + 3] = t.w;
    }
    uint32_t w0[4] = {pk0.x, pk0.y, pk0.z, pk0.w};
    uint32_t w1[4] = {pk1.x, pk1.y, pk1.z, pk1.w};
    float a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      uint32_t v0 = w0[i], v1 = w1[i];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        a0 += cbf[v0 & 0xf] * xv[i * 8 + j];
        a1 += cbf[v1 & 0xf] * xv[i * 8 + j];
        v0 >>= 4; v1 >>= 4;
      }
    }
    acc0 += a0 * s0;
    acc1 += a1 * s1;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc0 += __shfl_down_sync(~0ull, acc0, off);
    acc1 += __shfl_down_sync(~0ull, acc1, off);
  }
  if (lane == 0) {
    float w = ws[tok * id_stride + s];
    atomicAdd(acc + (size_t)tok * acc_stride + rp * 2, w * acc0);
    if (has1) atomicAdd(acc + (size_t)tok * acc_stride + rp * 2 + 1, w * acc1);
  }
}

// Small-P verify down projection: one wave covers two output rows and keeps
// all top-10 expert partials in registers. Two 16-lane subgroups avoid the
// 12 idle lanes of the K=640 full-wave mapping and eliminate atomic output
// accumulation. This is intentionally limited to topk=10 by the host route.
template <int KPT, int GSH = 0>
__global__ void k_q4cp_gemv_gd_topk_h16(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const float* __restrict__ xs,
    float* __restrict__ out, const int* __restrict__ ids,
    const float* __restrict__ ws, uint64_t rows_per, uint64_t cols,
    uint64_t scale_stride, uint64_t x_stride, int P, int id_stride,
    uint64_t out_stride) {
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  uint64_t row_pairs = (rows_per + 1) / 2;
  uint64_t gw =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (gw >= (uint64_t)P * row_pairs) return;
  int tok = gw / row_pairs;
  uint64_t rp = gw % row_pairs;
  int half = (threadIdx.x % warpSize) / 16;
  int lane = threadIdx.x & 15;
  uint64_t r = rp * 2 + half;
  if (r >= rows_per) return;
  float acc[KPT] = {};
#pragma unroll
  for (int s = 0; s < KPT; s++) {
    uint64_t row = (uint64_t)ids[tok * id_stride + s] * rows_per + r;
    const uint8_t* rc = codes + row * (cols / 2);
    const uint8_t* rs = scales + row * scale_stride;
    const float* x = xs + ((size_t)tok * KPT + s) * x_stride;
    for (uint64_t g = lane; g < cols / 32; g += 16) {
      uint4 pk = *(const uint4*)(rc + g * 16);
      float sc = __half2float(*(const __half*)(rs + (g >> GSH) * 2));
      const float* xg = x + g * 32;
      float xv[32];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        float4 t = *(const float4*)(xg + i * 4);
        xv[i * 4] = t.x;
        xv[i * 4 + 1] = t.y;
        xv[i * 4 + 2] = t.z;
        xv[i * 4 + 3] = t.w;
      }
      uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t v = w[i];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          a += cbf[v & 0xf] * xv[i * 8 + j];
          v >>= 4;
        }
      }
      acc[s] += a * sc;
    }
  }
#pragma unroll
  for (int s = 0; s < KPT; s++)
    for (int off = 8; off > 0; off >>= 1)
      acc[s] += __shfl_down_sync(~0ull, acc[s], off, 16);
  if (lane == 0) {
    float y = 0.f;
#pragma unroll
    for (int s = 0; s < KPT; s++) y += ws[tok * id_stride + s] * acc[s];
    out[(size_t)tok * out_stride + r] = y;
  }
}

// High-occupancy zero-idle P=1 MoE down_proj (QWENOX_MOE_DN_FAST=1):
// 320 blocks x 160 threads (5 warps/block = 1600 waves, 8 rows/block, 2 experts/warp).
// Within each warp: 8 rows x 4 lanes/row; cols=640 -> 20 groups = 4 lanes x 5 groups
// (100% lane utilization vs 75% idle on iter 2 in h16), only 2 accumulators/thread
// (cuts VGPRs), and LDS 5-warp reduction (zero global atomics).
template <int GSH = 2>
__global__ __launch_bounds__(160, 6) void k_q4cp_gemv_gd_topk_5w(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const float* __restrict__ xs,
    float* __restrict__ out, const int* __restrict__ ids,
    const float* __restrict__ ws, uint64_t rows_per, uint64_t cols,
    uint64_t scale_stride, uint64_t x_stride) {
  __shared__ float cbf[16];
  __shared__ float sp[5][8];
  const int tid = threadIdx.x;
  if (tid < 16) cbf[tid] = cb[tid];
  __syncthreads();
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int ri = lane >> 2;
  const int sub = lane & 3;
  const uint64_t r = (uint64_t)blockIdx.x * 8 + ri;
  if (r < rows_per) {
    float sum_w = 0.f;
#pragma unroll
    for (int e2 = 0; e2 < 2; e2++) {
      const int s = w * 2 + e2;
      const uint64_t row = (uint64_t)ids[s] * rows_per + r;
      const uint8_t* rc = codes + row * (cols / 2);
      const uint8_t* rs = scales + row * scale_stride;
      const float* x = xs + (size_t)s * x_stride;
      float acc = 0.f;
#pragma unroll
      for (int it = 0; it < 5; it++) {
        const uint64_t g = (uint64_t)(it * 4 + sub);
        const uint4 pk = *(const uint4*)(rc + g * 16);
        const float sc = __half2float(*(const __half*)(rs + (g >> GSH) * 2));
        const float* xg = x + g * 32;
        const uint32_t wv[4] = {pk.x, pk.y, pk.z, pk.w};
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          const float4 t0 = *(const float4*)(xg + i * 8);
          const float4 t1 = *(const float4*)(xg + i * 8 + 4);
          uint32_t v = wv[i];
          a = fmaf(cbf[v & 0xf], t0.x, a); v >>= 4;
          a = fmaf(cbf[v & 0xf], t0.y, a); v >>= 4;
          a = fmaf(cbf[v & 0xf], t0.z, a); v >>= 4;
          a = fmaf(cbf[v & 0xf], t0.w, a); v >>= 4;
          a = fmaf(cbf[v & 0xf], t1.x, a); v >>= 4;
          a = fmaf(cbf[v & 0xf], t1.y, a); v >>= 4;
          a = fmaf(cbf[v & 0xf], t1.z, a); v >>= 4;
          a = fmaf(cbf[v & 0xf], t1.w, a);
        }
        acc += a * sc;
      }
      acc += __shfl_xor(acc, 2, 32);
      acc += __shfl_xor(acc, 1, 32);
      sum_w += ws[s] * acc;
    }
    if (sub == 0) sp[w][ri] = sum_w;
  }
  __syncthreads();
  if (tid < 8) {
    const uint64_t rr = (uint64_t)blockIdx.x * 8 + tid;
    if (rr < rows_per) {
      out[rr] = sp[0][tid] + sp[1][tid] + sp[2][tid] + sp[3][tid] + sp[4][tid];
    }
  }
}

// Grouped small-P verify down_proj (QWENOX_MOE_GDX, default on): the down
// weights of a verify batch's duplicate experts are read once per group
// instead of once per slot (groups from k_moe_group2, shared with the up
// phase). Per-(slot,row) arithmetic replicates k_q4cp_gemv_gd_topk_h16
// exactly — same 16-lane half-warp mapping, same x layout, same sequential
// i/j dot order, same shfl reduce tree, fp32 acc. Slot partials ("pairs")
// land in a [slot][rows] fp32 buffer; the fused kernel's epilogue then runs
// as k_topk_wsum (router order, fp32), so the result is bit-identical.
template <int KPT, int GSH, int TM>
__global__ __launch_bounds__(512) void k_q4cp_gemv_gdx_h16(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
    const float* __restrict__ cb, const float* __restrict__ xs,
    float* __restrict__ pairs, uint64_t rows_per, uint64_t cols,
    uint64_t scale_stride, uint64_t x_stride,
    const int* __restrict__ gexp, const int* __restrict__ gnum,
    const int* __restrict__ gsl, const int* __restrict__ ngrp) {
  const int g = blockIdx.y;
  if (g >= *ngrp) return;
  __shared__ float cbf[16];
  if (threadIdx.x < 16) cbf[threadIdx.x] = cb[threadIdx.x];
  __syncthreads();
  const uint64_t row_pairs = (rows_per + 1) / 2;
  const uint64_t rp =
      (uint64_t)blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (rp >= row_pairs) return;
  const int half = (threadIdx.x % warpSize) / 16;
  const int lane = threadIdx.x & 15;
  const uint64_t r = rp * 2 + half;
  if (r >= rows_per) return;
  const uint64_t row = (uint64_t)gexp[g] * rows_per + r;
  const uint8_t* rc = codes + row * (cols / 2);
  const uint8_t* rs = scales + row * scale_stride;
  const int ns = gnum[g];
  const int* sl = gsl + g * 16;
  for (int b0 = 0; b0 < ns; b0 += TM) {
    const int nb = min(TM, ns - b0);
    float acc[TM];
#pragma unroll
    for (int q = 0; q < TM; q++) acc[q] = 0.f;
    for (uint64_t gi = lane; gi < cols / 32; gi += 16) {
      const uint4 pk = *(const uint4*)(rc + gi * 16);
      const float sc = __half2float(*(const __half*)(rs + (gi >> GSH) * 2));
      uint32_t w[4] = {pk.x, pk.y, pk.z, pk.w};
      float c[32];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t v = w[i];
#pragma unroll
        for (int j = 0; j < 8; j++) {
          c[i * 8 + j] = cbf[v & 0xf];
          v >>= 4;
        }
      }
#pragma unroll
      for (int q = 0; q < TM; q++) {
        if (q < nb) {
          const float* xg = xs + (size_t)sl[b0 + q] * x_stride + gi * 32;
          float xv[32];
#pragma unroll
          for (int i = 0; i < 8; i++) {
            float4 t = *(const float4*)(xg + i * 4);
            xv[i * 4] = t.x;
            xv[i * 4 + 1] = t.y;
            xv[i * 4 + 2] = t.z;
            xv[i * 4 + 3] = t.w;
          }
          float a = 0.f;
#pragma unroll
          for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) a += c[i * 8 + j] * xv[i * 8 + j];
          acc[q] += a * sc;
        }
      }
    }
#pragma unroll
    for (int q = 0; q < TM; q++)
      if (q < nb)
        for (int off = 8; off > 0; off >>= 1)
          acc[q] += __shfl_down_sync(~0ull, acc[q], off, 16);
    if (lane == 0)
#pragma unroll
      for (int q = 0; q < TM; q++)
        if (q < nb) pairs[(size_t)sl[b0 + q] * rows_per + r] = acc[q];
  }
}

// Epilogue of k_q4cp_gemv_gd_topk_h16 as a standalone pass over grouped
// pairs: out[tok][r] = sum_s ws[s] * pairs[tok*KPT+s][r], fp32, router order.
template <int KPT>
__global__ void k_topk_wsum(const float* __restrict__ pairs,
                            const float* __restrict__ ws, float* __restrict__ out,
                            uint64_t rows_per, int id_stride, uint64_t out_stride,
                            int P) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (uint64_t)P * rows_per) return;
  const int tok = (int)(i / rows_per);
  const uint64_t r = i % rows_per;
  float y = 0.f;
#pragma unroll
  for (int s = 0; s < KPT; s++)
    y += ws[tok * id_stride + s] * pairs[((size_t)tok * KPT + s) * rows_per + r];
  out[(size_t)tok * out_stride + r] = y;
}

// --- PLE conv (dilation 3, k=4): ring 9 x 10240; tap j pairs t-(3-j)*3
__global__ void k_ple_conv(const float* __restrict__ ring,
                           const int* __restrict__ d_ringpos,
                           const float* __restrict__ cw, float* __restrict__ out,
                           int n) {
  int ringpos = *d_ringpos;
  int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= n) return;
  float a = 0.f;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    int back = (3 - j) * 3;
    int rp = ((ringpos - back) % 9 + 9) % 9;
    a += cw[ch * 4 + j] * ring[(size_t)rp * n + ch];
  }
  out[ch] = a / (1.f + expf(-a));  // silu
}

// grouped zc rmsnorm into PLE ring slot *d_ringpos (ring = 9 x 10240)
__global__ void k_rmsnorm_zc_ring(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ ring,
                                  const int* __restrict__ d_ringpos, int gsize,
                                  float eps) {
  float* out = ring + (size_t)(*d_ringpos) * 10240;
  const float* xg = x + (size_t)blockIdx.x * gsize;
  const float* wg = w + (size_t)blockIdx.x * gsize;
  float* og = out + (size_t)blockIdx.x * gsize;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float ss = 0.f;
  for (int i = t; i < gsize; i += blockDim.x) ss += xg[i] * xg[i];
  sm[t] = ss;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / gsize + eps);
  for (int i = t; i < gsize; i += blockDim.x) og[i] = xg[i] * inv * (1.f + wg[i]);
}

// --- QSA kv cache append at *d_pos (replaces pos-dependent memcpy) ----------
__global__ void k_store_kv(const float* __restrict__ kb, const float* __restrict__ vb,
                           float* __restrict__ kc, float* __restrict__ vc,
                           const int* __restrict__ d_pos,
                           const int* __restrict__ ptab = nullptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 512) return;
  size_t off = (size_t)kv_row(ptab, *d_pos) * 512 + i;
  kc[off] = kb[i];
  vc[off] = vb[i];
}

// BF16 variant of the above: same layout, values rounded RNE via f2bf.
// vct (optional, QWENOX_QSA_WMMA_BTV): also write V into the block-transposed
// prefill cache [block][kvh][dim][slot] (layout of k_f32_to_bf16_v4_bt, with
// i = kvh*256 + dim), so later sparse prefill / verify reads of decoded
// positions see the same bits as the row-major cache.
// ptab (A1c, optional): paged addressing. The physical row is kv_row(pos) and
// the physical vct block is kv_row(pos) >> 2 == ik_blk(pos >> 2) (a page holds
// whole 4-token blocks), slot pos & 3 unchanged.
__global__ void k_store_kv_bf16(const float* __restrict__ kb,
                                const float* __restrict__ vb,
                                uint16_t* __restrict__ kc, uint16_t* __restrict__ vc,
                                const int* __restrict__ d_pos,
                                uint16_t* __restrict__ vct = nullptr,
                                const int* __restrict__ ptab = nullptr) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 512) return;
  const int pos = *d_pos;
  const int row = kv_row(ptab, pos);
  size_t off = (size_t)row * 512 + i;
  kc[off] = f2bf(kb[i]);
  const uint16_t v = f2bf(vb[i]);
  vc[off] = v;
  if (vct) vct[((size_t)(row >> 2) * 512 + i) * 4 + (pos & 3)] = v;
}

// Native FP16 rows, shared by prefill, decode, MTP and graph replay. Convert
// directly from FP32 with RNE; never pass through BF16 or keep a BF16 twin.
// vct (optional, QWENOX_QSA_WMMA_BTV): also write V into the block-transposed
// prefill cache [block][kvh][dim][slot], same addressing as k_store_kv_bf16.
__global__ void k_store_kv_fp16(const float* __restrict__ k,
                                const float* __restrict__ v,
                                __half* __restrict__ kc,
                                __half* __restrict__ vc, int P, int base,
                                const int* __restrict__ ptab = nullptr,
                                const int* __restrict__ dbase = nullptr,
                                uint16_t* __restrict__ vct = nullptr) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)P * 512) return;
  const int pos = (dbase ? *dbase : base) + (int)(i / 512);
  const size_t row = (size_t)kv_row(ptab, pos);
  const size_t dst = row * 512 + i % 512;
  kc[dst] = __float2half_rn(k[i]);
  const __half hv = __float2half_rn(v[i]);
  vc[dst] = hv;
  if (vct) vct[((row >> 2) * 512 + i % 512) * 4 + (pos & 3)] = __half_as_ushort(hv);
}

// Quantize K and V together: one wave owns one independent 32-dim group.
// Works for prefill, decode, MTP, non-bound slots and captured verify graphs.
// ptab maps logical rows; dbase supplies the graph's replay-time base.
__global__ void k_store_kv_q8(const float* __restrict__ k,
                              const float* __restrict__ v,
                              KvQ8Block* __restrict__ kc,
                              KvQ8Block* __restrict__ vc, int P, int base,
                              const int* __restrict__ ptab = nullptr,
                              const int* __restrict__ dbase = nullptr) {
  const int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (g >= P * 16) return;
  const int lane = threadIdx.x & 31, row = g / 16, group = g % 16;
  const size_t i = (size_t)row * 512 + group * 32 + lane;
  const float a = k[i], b = v[i];
  float ak = fabsf(a), av = fabsf(b);
#pragma unroll
  for (int d = 16; d; d >>= 1) {
    ak = fmaxf(ak, __shfl_xor_sync(~0ull, ak, d));
    av = fmaxf(av, __shfl_xor_sync(~0ull, av, d));
  }
  // Quantize with the stored (rounded) scale. Clamp tiny nonzero scales to
  // the minimum fp16 subnormal, so a finite input never divides by zero.
  const __half sk = __float2half_rn(ak == 0 ? 0.f : fmaxf(ak / 127.f, 0x1p-24f));
  const __half sv = __float2half_rn(av == 0 ? 0.f : fmaxf(av / 127.f, 0x1p-24f));
  const float dk = __half2float(sk), dv = __half2float(sv);
  const int qk = dk == 0 ? 0 : max(-127, min(127, __float2int_rn(a / dk)));
  const int qv = dv == 0 ? 0 : max(-127, min(127, __float2int_rn(b / dv)));
  const int pos = (dbase ? *dbase : base) + row;
  const size_t dst = (size_t)kv_row(ptab, pos) * 16 + group;
  kc[dst].q[lane] = (int8_t)qk;
  vc[dst].q[lane] = (int8_t)qv;
  if (lane == 0) { kc[dst].scale = sk; vc[dst].scale = sv; }
}

// --- end-of-step: pos++, ringpos = (ringpos+1)%9 ----------------------------
__global__ void k_step_incr(int* __restrict__ d_pos, int* __restrict__ d_ringpos) {
  *d_pos += 1;
  *d_ringpos = (*d_ringpos + 1) % 9;
}
__global__ void k_step_incr_amx(int* __restrict__ d_pos, int* __restrict__ d_ringpos,
                                const int* __restrict__ d_amx, int* __restrict__ p_amx) {
  *d_pos += 1;
  *d_ringpos = (*d_ringpos + 1) % 9;
  if (p_amx) {
    *p_amx = *d_amx;
    __threadfence_system();
  }
}
// ceil_probe R2 移植：图尾在写完 zero-copy argmax 后递增 host 可见的完成序号，
// 让宿主侧自旋等旗标替代 hipStreamSynchronize（WDDM 上 sync 成本 ~0.4-1ms）。
__global__ void k_step_incr_amx_seq(int* __restrict__ d_pos, int* __restrict__ d_ringpos,
                                    const int* __restrict__ d_amx, int* __restrict__ p_amx,
                                    uint32_t* __restrict__ p_seq) {
  *d_pos += 1;
  *d_ringpos = (*d_ringpos + 1) % 9;
  if (p_amx) *p_amx = *d_amx;
  __threadfence_system();
  if (p_seq) {
    uint32_t nx = *p_seq + 1u;
    __hip_atomic_store(p_seq, nx, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

// Emit one pipeline token (+ its logprob) into the coherent ring, then publish
// the write counter. Runs right after k_samp_prepare_draw in the graph tail;
// tok == -1 marks a candidate-overflow marker for the host fallback.
__global__ void k_pipe_emit(const int* __restrict__ tok, const float* __restrict__ lp,
                            const int* __restrict__ flag, int* __restrict__ ring_tok,
                            float* __restrict__ ring_lp, uint32_t* __restrict__ ring_w) {
  const uint32_t w = *ring_w;
  ring_tok[w & 127] = *flag ? -1 : *tok;
  ring_lp[w & 127] = *lp;
  __threadfence_system();
  __hip_atomic_store(ring_w, w + 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
// R[ch] += U[ch] + convout[ch]
__global__ void k_add2(float* __restrict__ R, const float* __restrict__ a,
                       const float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) R[i] += a[i] + b[i];
}
__global__ void k_acc(float* __restrict__ acc, const float* __restrict__ a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) acc[i] += a[i];
}
// Copy P contiguous d-rows into the heads of 4d-stride rows (k_repeat4_b then
// expands them in place).
__global__ void k_row_head_copy(const float* __restrict__ x,
                                float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  R[(size_t)t * 4 * d + c] = x[(size_t)t * d + c];
}
// Copy P contiguous d-rows into branch slot b of 4d-stride rows.
__global__ void k_row_branch_copy(const float* __restrict__ x,
                                  float* __restrict__ R, int d, int b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  R[(size_t)t * 4 * d + (size_t)b * d + c] = x[(size_t)t * d + c];
}
// Broadcast-add P contiguous d-rows onto all four branches of 4d-stride rows.
__global__ void k_acc_broadcast4(const float* __restrict__ x,
                                 float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  float v = x[(size_t)t * d + c];
  size_t base = (size_t)t * 4 * d + c;
  R[base] += v;
  R[base + d] += v;
  R[base + 2 * d] += v;
  R[base + 3 * d] += v;
}
// PLE per-branch gate: g_m = sum(norm_k_m * norm_q_m)/sqrt(d) -> signed sqrt ->
// sigmoid; U[m*d+i] = alpha_m * v[i]
__global__ void k_ple_gate(const float* __restrict__ kn, const float* __restrict__ qn,
                           const float* __restrict__ v, float* __restrict__ U, int d,
                           int branches) {
  int m = blockIdx.x;
  __shared__ float sm[1024];
  int t = threadIdx.x;
  float acc = 0.f;
  for (int i = t; i < d; i += blockDim.x) acc += kn[m * d + i] * qn[m * d + i];
  sm[t] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float g = sm[0] / sqrtf((float)d);
  g = (g < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(g), 1e-6f));
  float alpha = 1.f / (1.f + expf(-g));
  for (int i = t; i < d; i += blockDim.x) U[m * d + i] = alpha * v[i];
}

// --- Fused P=1 PLE tail: 2x rmsnorm_zc + ple_gate + rmsnorm_zc_ring + ple_conv + add2 ---
// Each block m in 0..3 owns branch slice [m*2560 .. (m+1)*2560) independently.
__global__ __launch_bounds__(1024) void k_ple_tail(
    const float* __restrict__ keys, const float* __restrict__ v,
    float* __restrict__ R, const float* __restrict__ nk,
    float* __restrict__ ring, const int* __restrict__ d_ringpos,
    const float* __restrict__ cw, float eps) {
  const int m = blockIdx.x, t = threadIdx.x;
  const float* kg = keys + (size_t)m * 2560;
  const float* wk = nk + (size_t)m * 2560;
  float* Rg = R + (size_t)m * 2560;
  const float* wq = nk + 10240 + (size_t)m * 2560;
  const float* wu = nk + 20480 + (size_t)m * 2560;
  __shared__ float sm[1024];

  float ssk = 0.f;
  for (int i = t; i < 2560; i += 1024) ssk += kg[i] * kg[i];
  sm[t] = ssk;
  __syncthreads();
  for (int off = 512; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  const float inv_k = rsqrtf(sm[0] / 2560 + eps);

  float ssq = 0.f;
  for (int i = t; i < 2560; i += 1024) ssq += Rg[i] * Rg[i];
  sm[t] = ssq;
  __syncthreads();
  for (int off = 512; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  const float inv_q = rsqrtf(sm[0] / 2560 + eps);

  float acc = 0.f;
  for (int i = t; i < 2560; i += 1024) {
    float kn_i = kg[i] * inv_k * (1.f + wk[i]);
    float qn_i = Rg[i] * inv_q * (1.f + wq[i]);
    asm volatile("" : "+v"(kn_i), "+v"(qn_i));
    acc += kn_i * qn_i;
  }
  sm[t] = acc;
  __syncthreads();
  for (int off = 512; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float g = sm[0] / sqrtf(2560.f);
  g = (g < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(g), 1e-6f));
  const float alpha = 1.f / (1.f + expf(-g));

  float uv[3];
  float ssu = 0.f;
  for (int k = 0; k < 3; k++) {
    const int i = t + k * 1024;
    if (i < 2560) {
      float ui = alpha * v[i];
      asm volatile("" : "+v"(ui));
      uv[k] = ui;
      ssu += ui * ui;
    }
  }
  sm[t] = ssu;
  __syncthreads();
  for (int off = 512; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  const float inv_u = rsqrtf(sm[0] / 2560 + eps);
  const int ringpos = *d_ringpos;
  float* rcur = ring + (size_t)ringpos * 10240;

  for (int k = 0; k < 3; k++) {
    const int i = t + k * 1024;
    if (i < 2560) {
      const int ch = m * 2560 + i;
      float rni = uv[k] * inv_u * (1.f + wu[i]);
      asm volatile("" : "+v"(rni));
      rcur[ch] = rni;
      float a = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int back = (3 - j) * 3;
        const int rp = ((ringpos - back) % 9 + 9) % 9;
        const float rv = (rp == ringpos) ? rni : ring[(size_t)rp * 10240 + ch];
        a += cw[ch * 4 + j] * rv;
      }
      float co = a / (1.f + expf(-a));
      asm volatile("" : "+v"(co));
      Rg[i] += uv[k] + co;
    }
  }
}

// --- Fused P=1 QSA prepare: qsplit + knorm + 2x rope + KV append (26x256) ---
// Blocks 0..23: Q head h (zcnorm + gate split + rotary 64).
// Blocks 24..25: KV head kh = blockIdx.x - 24 (zcnorm + rotary 64 + KV store).
// KvMode: 0 = bf16, 1 = fp16, 2 = fp32, 3 = no store (caller runs k_store_kv_q8).
template <int KvMode>
__global__ __launch_bounds__(256) void k_qsa_prep_p1(
    const float* __restrict__ qg, const float* __restrict__ qnw,
    float* __restrict__ qs, float* __restrict__ gs,
    float* __restrict__ kb, const float* __restrict__ vb,
    const float* __restrict__ knw, void* __restrict__ kc, void* __restrict__ vc,
    uint16_t* __restrict__ vct, const int* __restrict__ ptab,
    const int* __restrict__ d_pos, const int* __restrict__ d_rdelta, float eps) {
  __shared__ float sm[256];
  const int b = blockIdx.x, t = threadIdx.x;
  const int pos = *d_pos;
  const int rpos = d_rdelta ? pos + *d_rdelta : pos;
  if (b < 24) {
    const int h = b;
    const float v = qg[h * 512 + t];
    sm[t] = v * v;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
      if (t < off) sm[t] += sm[t + off];
      __syncthreads();
    }
    const float inv = rsqrtf(sm[0] / 256 + eps);
    gs[h * 256 + t] = qg[h * 512 + 256 + t];
    const float qn = v * inv * (1.f + qnw[t]);
    __syncthreads();
    sm[t] = qn;
    __syncthreads();
    float* q = qs + h * 256;
    if (t < 32) {
      const double ang = rpos * g_rope_tab[t];
      float cs = (float)cos(ang), sn = (float)sin(ang);
      if (g_rope_mscale != 1.f) cs *= g_rope_mscale, sn *= g_rope_mscale;
      const float x0 = sm[t], x1 = sm[t + 32];
      q[t] = x0 * cs - x1 * sn;
      q[t + 32] = x0 * sn + x1 * cs;
    } else if (t >= 64) {
      q[t] = qn;
    }
    return;
  }
  const int kh = b - 24;
  const int idx = kh * 256 + t;
  const float v = kb[idx];
  sm[t] = v * v;
  __syncthreads();
  for (int off = 128; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  const float inv = rsqrtf(sm[0] / 256 + eps);
  const float kn = v * inv * (1.f + knw[t]);
  __syncthreads();
  sm[t] = kn;
  __syncthreads();
  if (t < 32) {
    const double ang = rpos * g_rope_tab[t];
    float cs = (float)cos(ang), sn = (float)sin(ang);
    if (g_rope_mscale != 1.f) cs *= g_rope_mscale, sn *= g_rope_mscale;
    const float x0 = sm[t], x1 = sm[t + 32];
    sm[t] = x0 * cs - x1 * sn;
    sm[t + 32] = x0 * sn + x1 * cs;
  }
  __syncthreads();
  const float kf = sm[t];
  kb[idx] = kf;
  if constexpr (KvMode == 0) {
    const int row = kv_row(ptab, pos);
    const size_t off = (size_t)row * 512 + idx;
    ((uint16_t*)kc)[off] = f2bf(kf);
    const uint16_t vv = f2bf(vb[idx]);
    ((uint16_t*)vc)[off] = vv;
    if (vct) vct[((size_t)(row >> 2) * 512 + idx) * 4 + (pos & 3)] = vv;
  } else if constexpr (KvMode == 1) {
    const size_t row = (size_t)kv_row(ptab, pos);
    const size_t dst = row * 512 + idx;
    ((__half*)kc)[dst] = __float2half_rn(kf);
    const __half hv = __float2half_rn(vb[idx]);
    ((__half*)vc)[dst] = hv;
    if (vct) vct[((row >> 2) * 512 + idx) * 4 + (pos & 3)] = __half_as_ushort(hv);
  } else if constexpr (KvMode == 2) {
    const size_t off = (size_t)kv_row(ptab, pos) * 512 + idx;
    ((float*)kc)[off] = kf;
    ((float*)vc)[off] = vb[idx];
  } else if constexpr (KvMode == 3) {
    if (kc && vc) {
      const int lane = t & 31, group = kh * 8 + (t >> 5);
      const float b_val = vb[idx];
      float ak = fabsf(kf), av = fabsf(b_val);
#pragma unroll
      for (int d = 16; d; d >>= 1) {
        ak = fmaxf(ak, __shfl_xor_sync(~0ull, ak, d));
        av = fmaxf(av, __shfl_xor_sync(~0ull, av, d));
      }
      const __half sk = __float2half_rn(ak == 0 ? 0.f : fmaxf(ak / 127.f, 0x1p-24f));
      const __half sv = __float2half_rn(av == 0 ? 0.f : fmaxf(av / 127.f, 0x1p-24f));
      const float dk = __half2float(sk), dv = __half2float(sv);
      const int qk = dk == 0 ? 0 : max(-127, min(127, __float2int_rn(kf / dk)));
      const int qv = dv == 0 ? 0 : max(-127, min(127, __float2int_rn(b_val / dv)));
      const size_t dst = (size_t)kv_row(ptab, pos) * 16 + group;
      auto* kcq = (KvQ8Block*)kc;
      auto* vcq = (KvQ8Block*)vc;
      kcq[dst].q[lane] = (int8_t)qk;
      vcq[dst].q[lane] = (int8_t)qv;
      if (lane == 0) { kcq[dst].scale = sk; vcq[dst].scale = sv; }
    }
  }
}

