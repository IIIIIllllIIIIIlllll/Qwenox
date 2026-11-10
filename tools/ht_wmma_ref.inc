// Frozen copy of k_ht_wmma as of 99c4673 (before the 11-10 small-P rework):
// the bit-exact reference for tools/ht2_wmma_test.cu. Do not edit.
template <int NPT, typename TX>
__global__ __launch_bounds__(256) void k_ht_wmma_ref(const uint8_t* __restrict__ codes,
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
  extern __shared__ __align__(16) unsigned char hwr_lds[];
  __half* xh = (__half*)hwr_lds;  // [P][RS] rotated x chunk (f16), rows >= P not stored
  float* zl = (float*)hwr_lds;    // after the loop: z [P][128]
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
    const __half* xa = xh + (size_t)qrowA * RS + tc * 128;
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
        uint4 a2[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};  // rows >= P: zero
        if (pt * 16 + qrowA < P) {
          const __half* ap = xa + (size_t)pt * 16 * RS + g * 16;
          a2[0] = *(const uint4*)ap;
          a2[1] = *(const uint4*)(ap + 8);
        }
        htw_v16h af;
        memcpy(&af, a2, 32);
        acc[pt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc[pt]);
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
