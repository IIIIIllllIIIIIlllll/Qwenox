// ---------------- config (matches ref.cpp / config.json) --------------------
#include "../../rope.h"

__device__ double g_rope_tab[64];
__device__ float g_rope_mscale;

struct Cfg {
  int d = 2560;
  int layers = 48;
  int branches = 4;
  int gdn_hk = 16, gdn_dk = 128;
  int gdn_hv = 48, gdn_dv = 128;
  int gdn_conv = 4;
  int qsa_hq = 24, qsa_dh = 256;
  int qsa_hkv = 2;
  int rotary_dim = 64;
  int gr_rank = 320;
  int experts = 512, topk = 10, moe_mid = 640;
  float norm_eps = 1e-6f;
  double rope_theta = 1e7;
  RopeConfig rope;
  int vocab = 248320;
};
static Cfg g_cfg;

enum class KvType { F32, BF16, Q8 };
static KvType kv_type_env() {
  const char* e = getenv("QWENOX_KV_TYPE");
  if (!e || !*e)
    return getenv("QWENOX_QSA_KV_BF16") ? KvType::BF16 : KvType::F32;
  if (!strcmp(e, "bf16")) return KvType::BF16;
  if (!strcmp(e, "fp32") || !strcmp(e, "f32")) return KvType::F32;
  if (!strcmp(e, "q8") || !strcmp(e, "q8_0")) return KvType::Q8;
  throw std::runtime_error("QWENOX_KV_TYPE must be bf16, fp32, or q8");
}

// Native KV format: symmetric int8, 32 consecutive dimensions per group,
// independent K/V scales. Every token owns all its scales: appending or
// rejecting a draft cannot change the quantization of an accepted prefix.
struct KvQ8Block {
  __half scale;
  int8_t q[32];
};
static_assert(sizeof(KvQ8Block) == 34, "Q8 KV block layout");
static constexpr size_t KV_Q8_ROW_BYTES = 16 * sizeof(KvQ8Block);
// One layer's logical BF16 K + transposed V workspace, reused across layers.
// Keep this gate shared with the Windows arena estimate.
static bool q8_prefill_expand_env() {
  const char* e = getenv("QWENOX_Q8_PREFILL_EXPAND");
  return kv_type_env() == KvType::Q8 && !(e && !strcmp(e, "0")) &&
         getenv("QWENOX_QSA_WMMA") && getenv("QWENOX_QSA_WMMA_BTV") &&
         !getenv("QWENOX_QSA_DENSE") && !getenv("QWENOX_QSA_LOOP") &&
         !getenv("QWENOX_QSA_GLOBAL_V");
}
static int q8_prefill_min_env() {
  const char* e = getenv("QWENOX_Q8_PREFILL_MIN");
  return e ? std::max(1, atoi(e)) : 256;
}
static size_t kv_format_row_bytes(int format) {
  // Snapshot format tags: 2 = BF16, 4 = FP32, 8 = native Q8 group32.
  return format == 8 ? KV_Q8_ROW_BYTES : (size_t)512 * format;
}
static int g_index_budget = 512;
static int g_index_tokens = 2048;
static int g_index_first = 2051;
static int g_index_sparse = 2052;
static hipStream_t g_str = nullptr;  // capture stream during graph build, else default
// D1b: >0 while the GPU is lent out mid-prefill-chunk; during the window only
// light rounds (chunks of at most QWENOX_CONC_RMAX rows) may start on it.
static std::atomic<int> g_borrow_light{0};

static bool is_qsa(int layer) { return layer % 4 == 3; }
// QWENOX_TURN_PROF diagnostics fed by the batched verify (40_model.inc
// prefill_chunk_vseg), printed with the [turnprof] line (51_host_cfg.inc).
static double g_vseg_ple_ms = 0;  // host time in the per-segment PLE gathers
static double g_vseg_pleh_ms = 0;  // PLE gather spawn -> done (batched verify)
static long g_vseg_rows = 0, g_vseg_calls = 0;
static double g_vseg_enq_ms = 0;  // host enqueue time of batched verifies

