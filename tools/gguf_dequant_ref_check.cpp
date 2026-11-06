// gguf_dequant_ref_check.cpp — bit-exact check of gguf::dequant_row
// (Q5_0, IQ2_XXS, IQ2_XS, IQ3_S, Q6_K) against llama.cpp ggml-quants.c reference
// functions copied verbatim below
// (llama.cpp commit 85c5522, MIT License, Copyright (c) 2023-2026 The ggml authors).
// Build: g++ -O1 -std=c++17 -I src tools/gguf_dequant_ref_check.cpp -o /tmp/gdrc && /tmp/gdrc
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <random>
#include <vector>
#include "gguf.h"
#define QK_K 256
#define GGML_RESTRICT
typedef uint16_t ggml_half;
#define GGML_FP16_TO_FP32(x) gguf::h2f(x)
static const uint32_t iq3s_grid[512] = GGUF_IQ3S_GRID_INIT;
static const uint64_t iq2xxs_grid[256] = GGUF_IQ2XXS_GRID_INIT;
static const uint64_t iq2xs_grid[512] = GGUF_IQ2XS_GRID_INIT;
static const uint8_t ksigns_iq2xs[128] = GGUF_KSIGNS_IQ2XS_INIT;
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == 66, "wrong iq2_xxs block size/padding");
typedef struct {
    ggml_half d;
    uint16_t qs[QK_K/8];
    uint8_t scales[QK_K/32];
} block_iq2_xs;
static_assert(sizeof(block_iq2_xs) == 74, "wrong iq2_xs block size/padding");
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}
void dequantize_row_iq2_xs(const block_iq2_xs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    float db[2];

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (x[i].qs[4*ib32 + l] & 511));
                const uint8_t  signs = ksigns_iq2xs[x[i].qs[4*ib32 + l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db[l/2] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}
void dequantize_row_iq3_s(const block_iq3_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}
void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
#define QK5_0 32
typedef struct {
    ggml_half d;
    uint8_t qh[4];
    uint8_t qs[QK5_0/2];
} block_q5_0;
static_assert(sizeof(block_q5_0) == 22, "wrong q5_0 block size/padding");
void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK5_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}

int main() {
  using RefFn = void (*)(const uint8_t*, float*, int64_t);
  struct TestCase {
    const char* name;
    uint32_t type;
    int be;
    int bs;
    int d_off;
    RefFn ref;
  };
  const TestCase tests[] = {
    {"Q5_0", gguf::Q5_0, 32, 22, 0, [](const uint8_t* p, float* y, int64_t k) {
      dequantize_row_q5_0((const block_q5_0*)p, y, k);
    }},
    {"IQ2_XXS", gguf::IQ2_XXS, 256, 66, 0, [](const uint8_t* p, float* y, int64_t k) {
      dequantize_row_iq2_xxs((const block_iq2_xxs*)p, y, k);
    }},
    {"IQ2_XS", gguf::IQ2_XS, 256, 74, 0, [](const uint8_t* p, float* y, int64_t k) {
      dequantize_row_iq2_xs((const block_iq2_xs*)p, y, k);
    }},
    {"IQ3_S", gguf::IQ3_S, 256, 110, 0, [](const uint8_t* p, float* y, int64_t k) {
      dequantize_row_iq3_s((const block_iq3_s*)p, y, k);
    }},
    {"Q6_K", gguf::Q6_K, 256, 210, 208, [](const uint8_t* p, float* y, int64_t k) {
      dequantize_row_q6_K((const block_q6_K*)p, y, k);
    }},
  };

  constexpr int nb = 4096;
  std::mt19937 rng(1);
  long total_bad = 0;
  for (const auto& tc : tests) {
    std::vector<uint8_t> buf(tc.bs * nb);
    for (auto& b : buf) b = rng();
    for (int i = 0; i < nb; i++) {  // finite fp16 d in [2^-10, 2^-6)
      uint16_t h = (uint16_t)(((5 + rng() % 4) << 10) | (rng() & 0x3ff)) | (rng() & 1 ? 0x8000 : 0);
      memcpy(&buf[i * tc.bs + tc.d_off], &h, 2);
    }
    std::vector<float> a(tc.be * nb), b(tc.be * nb);
    tc.ref(buf.data(), a.data(), tc.be * nb);
    gguf::dequant_row(tc.type, buf.data(), b.data(), tc.be * nb);

    long bad = 0;
    uint32_t be, bb;
    gguf::block_info(tc.type, be, bb);
    if (be != (uint32_t)tc.be || bb != (uint32_t)tc.bs) {
      printf("%s layout: got {%u, %u}, expected {%d, %d} -> FAIL\n", tc.name, be, bb, tc.be, tc.bs);
      bad++;
    }
    double mx = 0;
    for (size_t i = 0; i < a.size(); i++) {
      if (memcmp(&a[i], &b[i], sizeof(float)) != 0) bad++;
      mx = fmax(mx, fabs(a[i]));
    }
    total_bad += bad;
    printf("%s: %d blocks, %zu elems, mismatches %ld, max|y| %.4g -> %s\n",
           tc.name, nb, a.size(), bad, mx, bad ? "FAIL" : "PASS");
  }
  printf("TOTAL: mismatches %ld -> %s\n", total_bad, total_bad ? "FAIL" : "PASS");
  return total_bad ? 1 : 0;
}
