// ================== PLE batched prefill (Phase 3d-1) kernels =================

// --- k_ple_gate over P tokens: block (m, t); kn/qn/U rows are [P, branches*d]
__global__ void k_ple_gate_b(const float* __restrict__ kn,
                             const float* __restrict__ qn,
                             const float* __restrict__ v, float* __restrict__ U, int d,
                             int branches) {
  int m = blockIdx.x, t = blockIdx.y;
  size_t row = ((size_t)t * branches + m) * d;
  const float* knm = kn + row;
  const float* qnm = qn + row;
  const float* vt = v + (size_t)t * d;
  float* Um = U + row;
  __shared__ float sm[1024];
  int i = threadIdx.x;
  float acc = 0.f;
  for (int j = i; j < d; j += blockDim.x) acc += knm[j] * qnm[j];
  sm[i] = acc;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (i < off) sm[i] += sm[i + off];
    __syncthreads();
  }
  float g = sm[0] / sqrtf((float)d);
  g = (g < 0 ? -1.f : 1.f) * sqrtf(fmaxf(fabsf(g), 1e-6f));
  float alpha = 1.f / (1.f + expf(-g));
  for (int j = i; j < d; j += blockDim.x) Um[j] = alpha * vt[j];
}

// --- batched PLE conv (dilation 3, k=4): the 9-slot ring aliases tap back-9
// to the current position, so out[t] = silu(cw0*Un[t] + cw1*Un[t-6] +
// cw2*Un[t-3] + cw3*Un[t]); missing (negative) positions read as zero.
__global__ void k_ple_conv_b(const float* __restrict__ Un,
                             const float* __restrict__ cw, float* __restrict__ out,
                             int P, int n, int base = 0,
                             const float* __restrict__ ring = nullptr,
                             const int* __restrict__ d_vb = nullptr) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= (int64_t)P * n) return;
  if (d_vb) base = *d_vb;  // P3-A: device-mirrored base (graph-replayable)
  int t = (int)(i / n), ch = (int)(i % n);
  int a = base + t;  // absolute position
  const float* c4 = cw + (size_t)ch * 4;
  float v = c4[0] * Un[i];
  // taps reach back 6/3 positions; before the batch start they read the ring
  // (holds the last 9 absolute positions), at sequence start they vanish
  if (a >= 6) v += c4[1] * (t >= 6 ? Un[i - (size_t)6 * n]
                                    : ring[(size_t)((a - 6) % 9) * n + ch]);
  if (a >= 3) v += c4[2] * (t >= 3 ? Un[i - (size_t)3 * n]
                                    : ring[(size_t)((a - 3) % 9) * n + ch]);
  v += c4[3] * Un[i];
  out[i] = v / (1.f + expf(-v));
}

// --- PLE ring handoff: ring slot (base+P-9+j)%9 = Un[P-9+j] (negative pos skipped)
__global__ void k_ple_ring_wr(const float* __restrict__ Un, float* __restrict__ ring,
                              int P, int n, int base = 0,
                              const int* __restrict__ d_vb = nullptr) {
  int pos = P - 9 + blockIdx.x;
  if (pos < 0) return;
  if (d_vb) base = *d_vb;  // P3-A: device-mirrored base (graph-replayable)
  const float* src = Un + (size_t)pos * n;
  float* dst = ring + (size_t)((base + pos) % 9) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// --- PLE fp8 staging dequant: raw e4m3 bytes (16 rows x 160 per token, packed
// 2560/token) -> f32 embeddings. Bit-exact with hgn::fp8e4m3_to_f32 * scale.
__device__ inline float fp8e4m3_f32_dev(uint8_t v) {
  uint32_t s = v >> 7, e = (v >> 3) & 0xf, m = v & 7;
  float r;
  if (e == 0) r = ldexpf((float)m, -9);
  else if (e == 15 && m == 7) r = __int_as_float(0x7fc00000);
  else r = ldexpf((float)(8 + m), (int)e - 10);
  return s ? -r : r;
}

__device__ __forceinline__ void ple_wait_ready(const uint32_t* ple_ready) {
  if (ple_ready) {
    if (threadIdx.x == 0) {
      while (__hip_atomic_load(ple_ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0u) {}
    }
    __syncthreads();
  }
}

__global__ void k_ple_fp8_dequant(const uint8_t* __restrict__ src,
                                  float* __restrict__ dst, float scale,
                                  size_t n16,
                                  const uint32_t* ple_ready = nullptr) {  // n16 = elements/16
  ple_wait_ready(ple_ready);
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  uint4 raw = ((const uint4*)src)[i];
  const uint8_t* b = (const uint8_t*)&raw;
  float t[16];
#pragma unroll
  for (int j = 0; j < 16; j++) t[j] = fp8e4m3_f32_dev(b[j]) * scale;
  float4* d4 = (float4*)dst + i * 4;
  d4[0] = make_float4(t[0], t[1], t[2], t[3]);
  d4[1] = make_float4(t[4], t[5], t[6], t[7]);
  d4[2] = make_float4(t[8], t[9], t[10], t[11]);
  d4[3] = make_float4(t[12], t[13], t[14], t[15]);
}

// Pipeline variant of k_ple_fp8_dequant (QWENOX_SAMP_GPU): the graph ordinal
// in the request is d_pos - req_base + 1 (d_pos only advances at graph tails,
// so it is stable mid-graph); wait for the monotonic gather counter to reach
// that value, then read this graph's parity rows buffer (double buffered, so
// the host may gather step t+1 while step t is still in flight).
__global__ void k_ple_fp8_dequant_pipe(const uint8_t* __restrict__ src2,
                                       float* __restrict__ dst, float scale,
                                       size_t n16, size_t rows_bytes,
                                       const uint32_t* cnt,
                                       const int* __restrict__ d_pos,
                                       const int* __restrict__ req_base) {
  if (threadIdx.x == 0) {
    const uint32_t exp = (uint32_t)(*d_pos - *req_base + 1);
    while (__hip_atomic_load(cnt, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < exp) {
    }
  }
  __syncthreads();
  const int ord = *d_pos - *req_base;  // ordinal - 1
  const uint8_t* src = src2 + (size_t)(ord & 1) * rows_bytes;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  uint4 raw = ((const uint4*)src)[i];
  const uint8_t* b = (const uint8_t*)&raw;
  float t[16];
#pragma unroll
  for (int j = 0; j < 16; j++) t[j] = fp8e4m3_f32_dev(b[j]) * scale;
  float4* d4 = (float4*)dst + i * 4;
  d4[0] = make_float4(t[0], t[1], t[2], t[3]);
  d4[1] = make_float4(t[4], t[5], t[6], t[7]);
  d4[2] = make_float4(t[8], t[9], t[10], t[11]);
  d4[3] = make_float4(t[12], t[13], t[14], t[15]);
}

// --- PLE IQ4_NL staging dequant (GGUF table, dtype 11): raw 18 B blocks
// [fp16 d][16 B nibbles], 5 per 160-wide row, packed 16 rows per token ->
// f32 embeddings (block b -> dst[32b..32b+31]). Bit-exact with
// gguf::dequant_row(IQ4_NL) (fp16->f32 exact, one f32 multiply).
__constant__ int8_t c_kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                           1,    13,   25,  38,  53,  69,  89,  113};
__global__ void k_ple_iq4nl_dequant(const uint8_t* __restrict__ src,
                                    float* __restrict__ dst, size_t nblk,
                                    const uint32_t* ple_ready = nullptr) {
  ple_wait_ready(ple_ready);
  size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk) return;
  const uint8_t* p = src + b * 18;
  const float d = __half2float(__ushort_as_half((uint16_t)(p[0] | (p[1] << 8))));
  float* o = dst + b * 32;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const uint8_t q = p[2 + j];
    o[j] = d * (float)c_kvalues_iq4nl[q & 0xF];
    o[j + 16] = d * (float)c_kvalues_iq4nl[q >> 4];
  }
}

// --- PLE Q5_0 staging dequant (GGUF table, dtype 12): raw 22 B blocks
// [fp16 d][4 B qh][16 B qs], 5 per 160-wide row, packed 16 rows per token ->
// f32 embeddings (block b -> dst[32b..32b+31]). Bit-exact with
// gguf::dequant_row(Q5_0) (fp16->f32 exact, one f32 multiply per element).
__global__ void k_ple_q50_dequant(const uint8_t* __restrict__ src,
                                  float* __restrict__ dst, size_t nblk,
                                  const uint32_t* ple_ready = nullptr) {
  ple_wait_ready(ple_ready);
  size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nblk) return;
  const uint8_t* p = src + b * 22;
  const float d = __half2float(__ushort_as_half((uint16_t)(p[0] | (p[1] << 8))));
  uint32_t qh;
  memcpy(&qh, p + 2, 4);
  const uint8_t* qs = p + 6;
  float* o = dst + b * 32;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int x0 = (qs[j] & 0xF) | (int)(((qh >> j) & 1) << 4);
    const int x1 = (qs[j] >> 4) | (int)(((qh >> (j + 16)) & 1) << 4);
    o[j] = d * (float)(x0 - 16);
    o[j + 16] = d * (float)(x1 - 16);
  }
}

// --- PLE bf16 staging dequant (dtype 0 table): raw bf16 rows (16 rows x 160
// per token, packed 5120 B/token) -> f32 embeddings. Bit-exact with
// hgn::bf16_to_f32 (zero-extend << 16); no scale.
__device__ __forceinline__ float ple_bf16_f32(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
__global__ void k_ple_bf16_dequant(const uint8_t* __restrict__ src,
                                   float* __restrict__ dst,
                                   size_t n16,
                                   const uint32_t* ple_ready = nullptr) {  // n16 = elements/16
  ple_wait_ready(ple_ready);
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n16) return;
  uint4 raw0 = ((const uint4*)src)[i * 2];
  uint4 raw1 = ((const uint4*)src)[i * 2 + 1];
  const uint16_t* b0 = (const uint16_t*)&raw0;
  const uint16_t* b1 = (const uint16_t*)&raw1;
  float4* d4 = (float4*)dst + i * 4;
  d4[0] = make_float4(ple_bf16_f32(b0[0]), ple_bf16_f32(b0[1]),
                      ple_bf16_f32(b0[2]), ple_bf16_f32(b0[3]));
  d4[1] = make_float4(ple_bf16_f32(b0[4]), ple_bf16_f32(b0[5]),
                      ple_bf16_f32(b0[6]), ple_bf16_f32(b0[7]));
  d4[2] = make_float4(ple_bf16_f32(b1[0]), ple_bf16_f32(b1[1]),
                      ple_bf16_f32(b1[2]), ple_bf16_f32(b1[3]));
  d4[3] = make_float4(ple_bf16_f32(b1[4]), ple_bf16_f32(b1[5]),
                      ple_bf16_f32(b1[6]), ple_bf16_f32(b1[7]));
}

// --- argmax over n floats -> writes int id
__global__ void k_argmax(const float* __restrict__ x, int n, int* __restrict__ out) {
  __shared__ float bv[1024];
  __shared__ int bi[1024];
  int t = threadIdx.x;
  float best = -1e30f;
  int bidx = 0;
  for (int i = t; i < n; i += blockDim.x) {
    if (x[i] > best) {
      best = x[i];
      bidx = i;
    }
  }
  bv[t] = best;
  bi[t] = bidx;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off && bv[t + off] > bv[t]) {
      bv[t] = bv[t + off];
      bi[t] = bi[t + off];
    }
    __syncthreads();
  }
  if (t == 0) out[0] = bi[0];
}

// [tgk-begin] (extracted by tools/tg_kernels_test.cu via tools/tg_verify.sh)
// Two-stage argmax: stage 1 reduces each contiguous chunk to one (val, id)
// candidate packed as a sortable u64 (monotonic fp32 bits high, ~id low, so
// max key = largest value, ties -> smallest id = first occurrence). Stage 2
// reduces the few candidates. Same output value as k_argmax except on exact
// fp32 ties, where the old kernel's modulo-thread order is replaced by a
// first-occurrence rule (k_argmax stays for bit-sensitive A/B).
// Rows are blockIdx.y; cand layout [row][40].
__device__ __forceinline__ unsigned long long amx_pack(float v, int id) {
  unsigned u = __float_as_uint(v);
  u ^= (u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
  return ((unsigned long long)u << 32) | (unsigned)(0xFFFFFFFFu - (unsigned)id);
}

__global__ void k_argmax_p1(const float* __restrict__ x, int n,
                            unsigned long long* __restrict__ cand) {
  const int chunk = (n + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
  const float* xr = x + (size_t)blockIdx.y * n;
  unsigned long long best = 0;  // packs below any finite/inf value
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    unsigned long long k = amx_pack(xr[i], i);
    if (k > best) best = k;
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    unsigned long long o = __shfl_down_sync(~0ull, best, off);
    if (o > best) best = o;
  }
  __shared__ unsigned long long wv[32];
  const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
  if (lane == 0) wv[w] = best;
  __syncthreads();
  if (w == 0) {
    best = lane < (blockDim.x + warpSize - 1) / warpSize ? wv[lane] : 0;
    for (int off = 16; off > 0; off >>= 1) {
      unsigned long long o = __shfl_down_sync(~0ull, best, off);
      if (o > best) best = o;
    }
    if (lane == 0) cand[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = best;
  }
}

__global__ void k_argmax_p2(const unsigned long long* __restrict__ cand,
                            int mcand, int* __restrict__ out) {
  const unsigned long long* cr = cand + (size_t)blockIdx.y * mcand;
  unsigned long long best = 0;
  for (int i = threadIdx.x; i < mcand; i += blockDim.x)
    if (cr[i] > best) best = cr[i];
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    unsigned long long o = __shfl_down_sync(~0ull, best, off);
    if (o > best) best = o;
  }
  if (threadIdx.x == 0)
    out[blockIdx.y] = (int)(0xFFFFFFFFu - (unsigned)(best & 0xFFFFFFFFull));
}

// 40 blocks/row: one per CU on gfx1151; cand scratch is rows * 40 * 8 bytes.
#define ARGMAX_B1 40
static void argmax_fast_rows(const float* x, int n, int rows,
                             unsigned long long* d_cand, int* out,
                             hipStream_t st) {
  k_argmax_p1<<<dim3(ARGMAX_B1, rows), 256, 0, st>>>(x, n, d_cand);
  // p2 is a single-wave reduction (no cross-wave step): launch exactly one
  // wave. A 64-thread block on wave32 dropped candidates 32..39.
  k_argmax_p2<<<dim3(1, rows), 32, 0, st>>>(d_cand, ARGMAX_B1, out);
}

// Default: two-stage. QWENOX_ARGMAX_OLD restores the single-block kernel
// (identical values except on exact fp32 ties; see k_argmax_p1).
static const bool g_argmax_old = getenv("QWENOX_ARGMAX_OLD") != nullptr;
static void argmax_rows(const float* x, int n, int rows,
                        unsigned long long* d_cand, int* out, hipStream_t st) {
  if (!g_argmax_old && rows <= 65) {
    argmax_fast_rows(x, n, rows, d_cand, out, st);
    return;
  }
  for (int r = 0; r < rows; r++)
    k_argmax<<<1, 1024, 0, st>>>(x + (size_t)r * n, n, out + r);
}

// --- GPU sampling prep (spec sampling fast path) ---
// k_corr_apply replays HostSampler::prepare's correction phases on device:
// bias adds first, then penalty subtracts, then this row's draft-prefix
// penalty subtracts. Each list has unique ids, so within a phase the adds are
// independent; phases are ordered by __syncthreads (one block per row).
// Llama rows carry their full window in either the shared or the prefix list;
// repeat is applied once per corrected id before the additive penalty.
__global__ void k_corr_apply(float* __restrict__ x, int stride,
                             const int* __restrict__ b_ids,
                             const float* __restrict__ b_vals, int nb,
                             const int* __restrict__ h_ids,
                             const float* __restrict__ h_vals, int nh,
                             const int* __restrict__ p_ids,
                             const float* __restrict__ p_vals,
                             const int* __restrict__ p_n, int pcap, float repeat) {
  float* xr = x + (size_t)blockIdx.y * stride;
  for (int i = threadIdx.x; i < nb; i += blockDim.x) xr[b_ids[i]] += b_vals[i];
  __syncthreads();
  for (int i = threadIdx.x; i < nh; i += blockDim.x) {
    float& v = xr[h_ids[i]];
    if (repeat != 1.0f) v = v <= 0.0f ? v * repeat : v / repeat;
    v -= h_vals[i];
  }
  __syncthreads();
  const int np = p_n ? p_n[blockIdx.y] : 0;
  const int* pr_ids = p_ids + (size_t)blockIdx.y * pcap;
  const float* pr_vals = p_vals + (size_t)blockIdx.y * pcap;
  for (int i = threadIdx.x; i < np; i += blockDim.x) {
    float& v = xr[pr_ids[i]];
    if (repeat != 1.0f) v = v <= 0.0f ? v * repeat : v / repeat;
    v -= pr_vals[i];
  }
}

// Graph-capture variant of k_corr_apply for the decode sampling tail: the list
// head and counts live in a coherent control header (hdr[0]=nb, hdr[1]=nh,
// hdr[3]=repeat bits) instead of launch params, so one captured node serves
// every round. ids/vals are the coherent round-shared list (bias [0,nb) then
// hist [nb,nb+nh)); serial decode has no draft-prefix list. Same per-phase
// math and order as k_corr_apply.
__global__ void k_corr_apply_hdr(float* __restrict__ x, int stride,
                                 const int* __restrict__ ci,
                                 const float* __restrict__ cv,
                                 const int* __restrict__ hdr) {
  const int nb = hdr[0], nh = hdr[1];
  if ((nb | nh) == 0) return;
  float* xr = x + (size_t)blockIdx.y * stride;
  const float repeat = __int_as_float(hdr[3]);
  for (int i = threadIdx.x; i < nb; i += blockDim.x) xr[ci[i]] += cv[i];
  __syncthreads();
  for (int i = threadIdx.x; i < nh; i += blockDim.x) {
    float& v = xr[ci[nb + i]];
    if (repeat != 1.0f) v = v <= 0.0f ? v * repeat : v / repeat;
    v -= cv[nb + i];
  }
}

// k_corr_apply_hdr variant for the GPU sampling tail (QWENOX_SAMP_GPU): bias
// phase identical (coherent ci/cv list, hdr[0]=nb); the penalty phase scans
// the full vocab histogram instead of a per-row list: counts[id] > 0 ->
// logits[id] -= presence + frequency * count, bit-exact with
// HostSampler::correct's legacy penalty (__fmul_rn/__fadd_rn keep the mul and
// add unfused, as on the host). hdr[8]=presence bits, hdr[9]=frequency bits.
// Single block like k_corr_apply_hdr: every bias add must complete before any
// penalty touches the same id, which a multi-block grid cannot order.
__global__ void k_corr_apply_hist(float* __restrict__ x, int stride,
                                  const int* __restrict__ ci,
                                  const float* __restrict__ cv,
                                  const int* __restrict__ hdr,
                                  const uint16_t* __restrict__ counts) {
  const int nb = hdr[0];
  const float presence = __int_as_float(hdr[8]);
  const float frequency = __int_as_float(hdr[9]);
  if (nb == 0 && presence == 0.0f && frequency == 0.0f) return;
  float* xr = x + (size_t)blockIdx.y * stride;
  for (int i = threadIdx.x; i < nb; i += blockDim.x) xr[ci[i]] += cv[i];
  __syncthreads();
  if (presence != 0.0f || frequency != 0.0f)
    for (int id = threadIdx.x; id < stride; id += blockDim.x) {
      const unsigned c = counts[id];
      if (c) xr[id] -= __fadd_rn(presence, __fmul_rn(frequency, (float)c));
    }
}

// --- GPU sampling tail (QWENOX_SAMP_GPU): prepare_sparse + draw_sparse ---
// Replays HostSampler::prepare_sparse/draw_sparse (51_host_cfg.inc) on the
// radix top-k collect output: cand_n[0] unordered (id, corrected logit)
// pairs, capacity SAMP_CAP. Bit-exact with the host except expf/log (device
// libm vs UCRT); the splitmix64 rng, the (val desc, id asc) sort and every
// fp64 accumulation order match the host. hdr: [2]=k, [4]=temp bits,
// [5]=top_p bits, [6]=min_p bits, [7]=flags (bit0: bump hist_counts).
// Collect overflow (cand_n > SAMP_CAP) or empty input: out_flag=1,
// out_tok=-1 (host fallback). dbg_ids/dbg_pr/dbg_n (nullable) receive the
// filtered candidates and their normalized probabilities.
constexpr int SAMP_CAP = 512;
__device__ __forceinline__ uint64_t samp_mix64(uint64_t z) {  // splitmix64 finalizer
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
__global__ void k_samp_prepare_draw(const int* __restrict__ cand_ids,
                                    const float* __restrict__ cand_vals,
                                    const int* __restrict__ cand_n,
                                    const int* __restrict__ hdr,
                                    uint64_t* __restrict__ rng_state,
                                    uint16_t* __restrict__ hist_counts,
                                    int* __restrict__ out_tok,
                                    float* __restrict__ out_lp,
                                    int* __restrict__ out_flag,
                                    int* __restrict__ dbg_ids,
                                    float* __restrict__ dbg_pr,
                                    int* __restrict__ dbg_n) {
  __shared__ int s_id[SAMP_CAP];
  __shared__ float s_val[SAMP_CAP];
  __shared__ float s_w[SAMP_CAP];
  const int cn = cand_n[0];
  const int k = hdr[2];
  if (cn <= 0 || cn > SAMP_CAP || k <= 0) {
    if (threadIdx.x == 0) {
      *out_flag = 1;
      // Feed the top candidate as the dummy state input: the next graph must
      // embed a valid token while the host recovers this step densely.
      *out_tok = cn > 0 ? cand_ids[0] : 0;
    }
    return;
  }
  int sort_n = 32;
  while (sort_n < cn && sort_n < SAMP_CAP) sort_n <<= 1;
  for (int i = threadIdx.x; i < sort_n; i += blockDim.x) {
    s_id[i] = i < cn ? cand_ids[i] : INT_MAX;
    s_val[i] = i < cn ? cand_vals[i] : -INFINITY;
  }
  __syncthreads();
  // bitonic sort by (val desc, id asc) over dynamic power-of-2 sort_n <= SAMP_CAP
  for (int kk = 2; kk <= sort_n; kk <<= 1)
    for (int stride = kk >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < sort_n; i += blockDim.x) {
        const int j = i ^ stride;
        if (j > i) {
          const float vi = s_val[i], vj = s_val[j];
          const int ii = s_id[i], ij = s_id[j];
          const bool i_later = vi < vj || (vi == vj && ii > ij);
          if (((i & kk) == 0) == i_later) {
            s_val[i] = vj;
            s_id[i] = ij;
            s_val[j] = vi;
            s_id[j] = ii;
          }
        }
      }
      __syncthreads();
    }
  if (threadIdx.x == 0) {
    const float temp = __int_as_float(hdr[4]);
    const float top_p = __int_as_float(hdr[5]);
    const float min_p = __int_as_float(hdr[6]);
    const int n0 = k < cn ? k : cn;
    const float lmax = s_val[0];
    const float inv_t = 1.0f / temp;
    float total = 0.0f;
    for (int j = 0; j < n0; j++) {
      const float wv = expf((s_val[j] - lmax) * inv_t);
      s_w[j] = wv;
      total += wv;
    }
    int n = n0;
    if (top_p > 0.0f && top_p < 1.0f && n > 0) {
      // s_w[0..n-1] is already sorted descending: single-pass FP32 prefix sum
      // (avoids 50-iter serial FP64 binary search on 1/16-rate RDNA3 FP64).
      const float target = top_p * total;
      float cum = 0.0f;
      int keep = n;
      for (int j = 0; j < n; j++) {
        cum += s_w[j];
        if (cum >= target) {
          keep = j + 1;
          while (keep < n && s_w[keep] == s_w[j]) keep++;
          break;
        }
      }
      n = keep;
    }
    if (min_p > 0.0f && n > 0) {
      const float thr = min_p * s_w[0];
      int w = 0;
      for (int j = 0; j < n; j++)
        if (s_w[j] >= thr) {
          s_w[w] = s_w[j];
          s_id[w] = s_id[j];
          w++;
        }
      if (w > 0) n = w;
    }
    float z = 0.0f;
    for (int j = 0; j < n; j++) z += s_w[j];
    const float inv_z = 1.0f / z;
    for (int j = 0; j < n; j++) s_w[j] *= inv_z;
    if (dbg_ids && dbg_pr && dbg_n) {
      *dbg_n = n;
      for (int j = 0; j < n; j++) {
        dbg_ids[j] = s_id[j];
        dbg_pr[j] = s_w[j];
      }
    }
    const uint64_t st = rng_state[0] + 0x9E3779B97F4A7C15ull;  // splitmix64 next64
    rng_state[0] = st;
    float u = (float)((double)(samp_mix64(st) >> 11) * (1.0 / 9007199254740992.0));
    int si = n - 1;
    for (int j = 0; j < n; j++) {
      u -= s_w[j];
      if (u <= 0.0f) {
        si = j;
        break;
      }
    }
    const int sel = s_id[si];
    *out_tok = sel;
    *out_lp = __logf(fmaxf(s_w[si], 1e-30f));
    *out_flag = 0;
    if ((hdr[7] & 1) && hist_counts) hist_counts[sel] += 1;
  }
}

__global__ void k_f32_copy_vec4_hdr(float* __restrict__ dst, const float* __restrict__ src,
                                    size_t n, const int* __restrict__ hdr) {
  if ((hdr[0] | hdr[1]) == 0) return;
  size_t n4 = n >> 2;
  float4* d4 = (float4*)dst;
  const float4* s4 = (const float4*)src;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += (size_t)gridDim.x * blockDim.x)
    d4[i] = s4[i];
  if (blockIdx.x == 0) {
    for (size_t i = (n4 << 2) + threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
  }
}

__global__ void k_zero_u32(unsigned* __restrict__ dst, int n) {
  for (int i = (int)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (int)gridDim.x * blockDim.x)
    dst[i] = 0u;
}

// Radix top-k over fp32 logits via their monotonic u32 keys (same map as
// amx_pack). Two 12-bit histogram passes narrow the k-th value to a 24-bit
// prefix; collect then writes every key >= T (the true top-k is a subset).
// ctrl per row (8 ints): 0=prefix, 1=strictly-in count, 2=k, 3=done, 4=T,
// 5=expected collect count.
__device__ __forceinline__ unsigned tk_key(float v) {
  unsigned u = __float_as_uint(v);
  return u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

template <int PASS>
__global__ void k_tk_hist_hdr(const float* __restrict__ logits,
                              const float* __restrict__ scratch, int n, int stride,
                              const int* __restrict__ ctrl,
                              unsigned* __restrict__ hist,
                              const int* __restrict__ hdr) {
  __shared__ unsigned sh[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  const int row = blockIdx.y;
  if (!PASS || !ctrl[(size_t)row * 8 + 3]) {
    const unsigned pfx = PASS ? (unsigned)ctrl[(size_t)row * 8 + 0] : 0u;
    const float* base = ((hdr[0] | hdr[1]) != 0) ? scratch : logits;
    const float* xr = base + (size_t)row * stride;
    const int chunk = (n + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      const unsigned k = tk_key(xr[i]);
      if (!PASS) {
        atomicAdd(&sh[k >> 20], 1u);
      } else if ((k >> 20) == pfx) {
        atomicAdd(&sh[(k >> 8) & 0xFFFu], 1u);
      }
    }
    __syncthreads();
    unsigned* h = hist + (size_t)row * 4096;
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
      const unsigned c = sh[i];
      if (c) atomicAdd(&h[i], c);
    }
  }
}

template <int PASS>
__global__ void k_tk_hist(const float* __restrict__ x, int n, int stride,
                          const int* __restrict__ ctrl,
                          unsigned* __restrict__ hist) {
  __shared__ unsigned sh[4096];
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) sh[i] = 0;
  __syncthreads();
  const int row = blockIdx.y;
  if (!PASS || !ctrl[(size_t)row * 8 + 3]) {
    const unsigned pfx = PASS ? (unsigned)ctrl[(size_t)row * 8 + 0] : 0u;
    const float* xr = x + (size_t)row * stride;
    const int chunk = (n + gridDim.x - 1) / gridDim.x;
    const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      const unsigned u = tk_key(xr[i]);
      if (PASS && (u >> 20) != pfx) continue;
      atomicAdd(&sh[PASS ? ((u >> 8) & 0xFFFu) : (u >> 20)], 1u);
    }
  }
  __syncthreads();
  unsigned* h = hist + (size_t)row * 4096;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x)
    if (sh[i]) atomicAdd(&h[i], sh[i]);
}

template <int PASS>
__global__ void k_tk_scan(const unsigned* __restrict__ hist,
                          int* __restrict__ ctrl, int* __restrict__ cand_n) {
  const int row = blockIdx.y;
  int* c = ctrl + (size_t)row * 8;
  if (PASS && c[3]) return;
  const unsigned* h = hist + (size_t)row * 4096;
  const int t = threadIdx.x;
  const int needed = c[2] - c[1];
  __shared__ unsigned wsum[8];
  __shared__ unsigned s_d, s_gt, s_bin;
  if (t == 0) s_d = 0xFFFFFFFFu;
  unsigned v[16];
  unsigned lsum = 0;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    v[j] = h[t * 16 + j];
    lsum += v[j];
  }
  unsigned x = lsum;
  // Inclusive scan within each 32-lane group (all 256 threads active), then a
  // serial scan of the 8 group sums. No *_sync shuffles in a partial wave:
  // HIP traps when the mask names inactive lanes.
  for (int off = 1; off < 32; off <<= 1) {
    unsigned y = __shfl_up(x, off, 32);
    if ((t & 31) >= off) x += y;
  }
  if ((t & 31) == 31) wsum[t >> 5] = x;
  __syncthreads();
  if (t == 0)
    for (int w = 1; w < 8; w++) wsum[w] += wsum[w - 1];
  __syncthreads();
  const unsigned total = wsum[7];
  unsigned run = x - lsum + (t >= 32 ? wsum[(t >> 5) - 1] : 0);
  unsigned best = 0xFFFFFFFFu, bgt = 0, bbin = 0;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const unsigned s_ge = total - run, s_gt = s_ge - v[j];
    if (s_ge >= (unsigned)needed && s_gt < (unsigned)needed) {
      best = t * 16 + j;
      bgt = s_gt;
      bbin = v[j];
    }
    run += v[j];
  }
  if (best != 0xFFFFFFFFu) {
    s_d = best;
    s_gt = bgt;
    s_bin = bbin;
  }
  __syncthreads();
  if (t == 0) {
    if (s_d == 0xFFFFFFFFu) {  // total < needed: impossible for k <= n
      c[4] = 0;
      c[5] = 1 << 30;
      c[3] = 1;
      cand_n[row] = 0;
      return;
    }
    const unsigned pfx = PASS ? (((unsigned)c[0] << 12) | s_d) : s_d;
    c[0] = (int)pfx;
    c[1] += (int)s_gt;
    if (PASS) {
      c[4] = (int)(pfx << 8);
      c[5] = c[1] + (int)s_bin;
      c[3] = 1;
      cand_n[row] = 0;
    }
  }
}

__global__ void k_tk_init(int* __restrict__ ctrl, int k) {
  int* c = ctrl + (size_t)blockIdx.x * 8;
  if (threadIdx.x == 0) {
    c[0] = 0;
    c[1] = 0;
    c[2] = k;
    c[3] = 0;
    c[4] = 0;
    c[5] = 0;
  }
}

// Graph-capture variant of k_tk_init: k from the coherent header (hdr[2]).
__global__ void k_tk_init_hdr(int* __restrict__ ctrl, const int* __restrict__ hdr) {
  int* c = ctrl + (size_t)blockIdx.x * 8;
  if (threadIdx.x == 0) {
    c[0] = 0;
    c[1] = 0;
    c[2] = hdr[2];
    c[3] = 0;
    c[4] = 0;
    c[5] = 0;
  }
}

__global__ void k_tk_collect_hdr(const float* __restrict__ logits,
                                 const float* __restrict__ scratch, int n, int stride,
                                 const int* __restrict__ ctrl,
                                 int* __restrict__ ids, float* __restrict__ vals,
                                 int* __restrict__ cand_n, int tcap,
                                 const int* __restrict__ hdr) {
  const int row = blockIdx.y;
  const unsigned T = (unsigned)ctrl[(size_t)row * 8 + 4];
  const float* base = ((hdr[0] | hdr[1]) != 0) ? scratch : logits;
  const float* xr = base + (size_t)row * stride;
  const int chunk = (n + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const float xv = xr[i];
    if (tk_key(xv) >= T) {
      const int idx = atomicAdd(&cand_n[row], 1);
      if (idx < tcap) {
        ids[(size_t)row * tcap + idx] = i;
        vals[(size_t)row * tcap + idx] = xv;
      }
    }
  }
}

__global__ void k_tk_collect(const float* __restrict__ x, int n, int stride,
                             const int* __restrict__ ctrl,
                             int* __restrict__ ids, float* __restrict__ vals,
                             int* __restrict__ cand_n, int tcap) {
  const int row = blockIdx.y;
  const unsigned T = (unsigned)ctrl[(size_t)row * 8 + 4];
  const float* xr = x + (size_t)row * stride;
  const int chunk = (n + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * chunk, hi = min(n, lo + chunk);
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const float xv = xr[i];
    if (tk_key(xv) >= T) {
      const int idx = atomicAdd(&cand_n[row], 1);
      if (idx < tcap) {
        ids[(size_t)row * tcap + idx] = i;
        vals[(size_t)row * tcap + idx] = xv;
      }
    }
  }
}

// [tgk-end]

__global__ void k_batch_nll(const float* __restrict__ logits, int vocab,
                            const int* __restrict__ targets,
                            float* __restrict__ losses) {
  __shared__ float reduction[256];
  int row = blockIdx.x;
  int lane = threadIdx.x;
  const float* x = logits + (size_t)row * vocab;
  float maximum = -INFINITY;
  for (int i = lane; i < vocab; i += blockDim.x)
    maximum = fmaxf(maximum, x[i]);
  reduction[lane] = maximum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) reduction[lane] = fmaxf(reduction[lane], reduction[lane + offset]);
    __syncthreads();
  }
  maximum = reduction[0];
  float total = 0.0f;
  for (int i = lane; i < vocab; i += blockDim.x)
    total += expf(x[i] - maximum);
  reduction[lane] = total;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) reduction[lane] += reduction[lane + offset];
    __syncthreads();
  }
  if (lane == 0) losses[row] = maximum + logf(reduction[0]) - x[targets[row]];
}

// vector copy / fill helpers
__global__ void k_repeat4(const float* __restrict__ x, float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < d) {
    R[i] = x[i];
    R[d + i] = x[i];
    R[2 * d + i] = x[i];
    R[3 * d + i] = x[i];
  }
}

// batched embedding gather for prefill: one block per token, out rows are
// outstride apart (the 4*d Rb row). Same math as k_q4cp_row.
__global__ void k_q4cp_row_b(const uint8_t* __restrict__ codes,
                             const uint8_t* __restrict__ scales,
                             const float* __restrict__ cb,
                             const int* __restrict__ d_tokens, uint64_t cols,
                             uint64_t scale_stride, float* __restrict__ out,
                             uint64_t outstride) {
  uint64_t row = (uint64_t)d_tokens[blockIdx.x];
  out += blockIdx.x * outstride;
  uint64_t g = threadIdx.x;  // one block per token, cols/32 threads
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

// batched repeat4 over P rows of 4*d contiguous floats (x == R in-place safe)
__global__ void k_repeat4_b(const float* __restrict__ x, float* __restrict__ R, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // over P * d
  int t = i / d, c = i - t * d;
  size_t base = (size_t)t * 4 * d + c;
  float v = x[base];
  R[base] = v;
  R[base + d] = v;
  R[base + 2 * d] = v;
  R[base + 3 * d] = v;
}

// --- Greedy draft confidence: top-1 softmax probability of a logits row ---
// 1 / sum(exp(x - max)). One block, two grid-stride sweeps; x is not
// modified. Block reduce follows k_argmax_p1 (wave width from warpSize).
__global__ void k_top1_prob(const float* __restrict__ x, int n,
                            float* __restrict__ out) {
  __shared__ float wv[64];
  const int lane = threadIdx.x % warpSize, w = threadIdx.x / warpSize;
  const int nw = (blockDim.x + warpSize - 1) / warpSize;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, x[i]);
  for (int off = warpSize / 2; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_down_sync(~0ull, m, off));
  if (lane == 0) wv[w] = m;
  __syncthreads();
  if (w == 0) {
    m = lane < nw ? wv[lane] : -INFINITY;
    for (int off = 16; off > 0; off >>= 1)
      m = fmaxf(m, __shfl_down_sync(~0ull, m, off));
    if (lane == 0) wv[0] = m;
  }
  __syncthreads();
  const float mx = wv[0];
  __syncthreads();  // wv is reused below
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += __expf(x[i] - mx);
  for (int off = warpSize / 2; off > 0; off >>= 1)
    s += __shfl_down_sync(~0ull, s, off);
  if (lane == 0) wv[w] = s;
  __syncthreads();
  if (w == 0) {
    s = lane < nw ? wv[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1)
      s += __shfl_down_sync(~0ull, s, off);
    if (lane == 0) *out = 1.f / s;
  }
}

// Enqueue-only top-1 probability: the result lands in out_dev (stream-ordered,
// no host sync). Batched draft bursts read the whole array back at once.
static void top1_prob_enq(const float* x, int n, float* out_dev,
                          hipStream_t st) {
  k_top1_prob<<<1, 256, 0, st>>>(x, n, out_dev);
}

// 草稿子词表回映射：子表 argmax 下标 -> 全局 token id（就地，单线程）。
__global__ void k_dv_map(const int* __restrict__ ids, int* __restrict__ io) {
  *io = ids[*io];
}
