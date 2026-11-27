// expf host(UCRT via clang) vs GPU device expf bit-equivalence probe.
// Sweeps the actual prepare_sparse input range [(vals-lmax)/temp in [-large, 0]]
// plus a random corpus; prints mismatch count.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <cstdio>
#include <cstdlib>
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

__global__ void k_expf(const float* __restrict__ in, float* __restrict__ out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expf(in[i]);
}

int main() {
  const size_t N = 4 * 1024 * 1024;
  std::vector<float> hin(N), hg(N), hh(N);
  // corpus: dense grid of the relevant range + pseudo-random samples
  unsigned rng = 12345;
  auto rnd = [&] {
    rng = rng * 1664525u + 1013904223u;
    return (rng >> 8) / 16777216.0f;
  };
  for (size_t i = 0; i < N; i++) {
    if (i < 1024 * 1024)
      hin[i] = -100.f + 100.f * (float)i / 1048576.f;  // [-100, 0)
    else
      hin[i] = -rnd() * (30.f + 90.f * rnd());  // [-120, 0)
  }
  float *din, *dout;
  CK(hipMalloc(&din, N * 4));
  CK(hipMalloc(&dout, N * 4));
  CK(hipMemcpy(din, hin.data(), N * 4, hipMemcpyHostToDevice));
  k_expf<<<dim3((unsigned)((N + 255) / 256)), 256>>>(din, dout, N);
  CK(hipMemcpy(hg.data(), dout, N * 4, hipMemcpyDeviceToHost));
  CK(hipDeviceSynchronize());
  size_t bad = 0;
  int first = -1;
  size_t b_lt87 = 0, b_20_87 = 0, b_10_20 = 0, b_5_10 = 0, b_ge5 = 0;
  for (size_t i = 0; i < N; i++) {
    hh[i] = expf(hin[i]);
    if (hh[i] != hg[i]) {
      if (first < 0) first = (int)i;
      bad++;
      const float x = hin[i];
      if (x < -87.f) b_lt87++;
      else if (x < -20.f) b_20_87++;
      else if (x < -10.f) b_10_20++;
      else if (x < -5.f) b_5_10++;
      else b_ge5++;
    }
  }
  printf("mismatch: %zu / %zu\n", bad, N);
  printf("  x<-87:     %zu\n", b_lt87);
  printf("  -87..-20:  %zu\n", b_20_87);
  printf("  -20..-10:  %zu\n", b_10_20);
  printf("  -10..-5:   %zu\n", b_5_10);
  printf("  >=-5:      %zu\n", b_ge5);
  if (first >= 0)
    printf("first @ %d: x=%.9g host=%.9g gpu=%.9g\n", first, hin[first], hh[first],
           hg[first]);
  else
    printf("host expf == gpu expf on entire corpus\n");
  return 0;
}
