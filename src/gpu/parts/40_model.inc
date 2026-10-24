// Concurrent serve (QWENOX_PARALLEL > 1): thrown by kv_reserve / kv_cow when the
// shared KV pool cannot back this sequence's rows (the later request loses),
// and by the serve yield hook when another request aborted this one. Always
// raised before any device write of the step, so the slot can simply reset.
struct KvOverflow : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// 客户端断连/取消打断 prompt 预填（prefill_batch 分段循环在段间检查 abort_flag——
// 与 yield_point 同一个"只有持久槽位状态存活"的安全点）。抛点前已写入的 KV 页
// 属于槽位持久状态，serve 的 catch 走槽位 reset 回收，与 KvOverflow 相同。
// decode/spec 路径不 armed 它：verify 批次有回合本地状态，中途抛会破坏回滚语义，
// 那边的取消仍走 emit/cancel 标志的回合边界回滚。
struct ReqCancelled : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct GpuModel {
  const Checkpoint& ckpt;
  bool ple_on = true;
  int maxctx;
  int maxbatch;  // prefill chunk size: batch buffers are allocated at
                 // maxbatch_cap (maxbatch + tail-merge slack)
  int maxbatch_cap;
  int pos = 0;
  NgramMod ngram;
  // Adaptive ngram verify chunk rows, shared by both chain loops. Per slot
  // and reset per request (ng_reset, 10-12): a value carried over from other
  // requests (or concurrent peers) changes this request's round structure,
  // so the same seed would not reproduce. 0 = re-parse QWENOX_NGRAM_CHUNK.
  int ng_cs = 0;
  bool ng_cs_fixed = false;
  bool ng_legacy = false;  // QWENOX_NGRAM_CHUNK=0: legacy single-pass verify
  void ng_chunk_init() {
    if (ng_cs) return;
    const char* e = getenv("QWENOX_NGRAM_CHUNK");
    const int v = e ? atoi(e) : -1;
    ng_legacy = v == 0;
    ng_cs_fixed = v > 0;
    ng_cs = ng_cs_fixed ? std::max(1, std::min(64, v)) : 16;
  }
  // Adaptive ngram gate (chain drafters): the ngram leg only pays on
  // repetitive content. Track the draft-token accept ratio with an EMA over
  // ngram rounds, close the leg for 32 rounds, then probe again. Like ng_cs
  // it is per slot and reset per request, so a request's output depends only
  // on its own prompt/seed (not on what ran before or alongside it).
  // QWENOX_NGRAM_GATE=0 pins the leg open.
  struct NgGate {
    double ema = 1.0;  // decayed draft-token accept ratio (starts optimistic)
    int cd = 0;        // rounds the leg stays closed (probes when it hits 0)
    int rounds = 0;
    bool last_full = false;  // previous ngram round committed its whole draft
    static bool disabled() {
      static const int v = [] {
        const char* e = getenv("QWENOX_NGRAM_GATE");
        return e ? atoi(e) : 1;
      }();
      return v == 0;
    }
    bool open() {
      if (disabled()) return true;
      if (cd > 0) { --cd; return false; }
      return true;
    }
    // A probe round measures the content with a short capped draft; a good
    // result reopens the leg to full-length drafts on the next round.
    bool probing() const { return !disabled() && rounds >= 4 && ema < 0.7; }
    // QWENOX_NGRAM_SAMPLE_CAP=N (sampled requests, chain-sample; default 0 =
    // off): until an ngram round commits its whole draft, cap the next draft
    // at N tokens. A sampled ngram round costs checkpoint + verify + replay +
    // sync whatever the draft length (Windows 10-16: ~115 ms capped at 8 vs
    // ~135 ms for 64 at depth 0, ~130 vs ~215 ms at 64K; an MTP round is
    // ~61 ms), so the cap only trims part of it and gives up long accepts on
    // repetitive text — A/B (code + prose, 0/64K) was within noise.
    static int sample_cap() {
      static const int v = [] {
        const char* e = getenv("QWENOX_NGRAM_SAMPLE_CAP");
        return e ? std::max(0, std::min(64, atoi(e))) : 0;
      }();
      return v;
    }
    void update(int keep, int proposed) {
      ++rounds;
      last_full = keep >= proposed;
      ema += ((double)keep / std::max(1, proposed) - ema) * 0.25;
      if (rounds >= 4 && ema < 0.7 && cd == 0) cd = 32;
    }
  };
  NgGate ng_gate;
  // Request start (serve): fresh gate and chunk level.
  void ng_reset() {
    ng_gate = NgGate{};
    ng_cs = 0;
  }

  // device state
  float *d_R, *d_Rhat, *d_x, *d_y, *d_emb;
  float *d_t320, *d_G, *d_w4;
  float *d_qkv, *d_g48, *d_beta48, *d_z;  // GDN scratch (z also used by QSA gate)
  float* d_S = nullptr;                   // (gdn_layers, 48, 128, 128)
  float* d_convst = nullptr;              // (gdn_layers, 3, 10240)
  float* d_convw = nullptr;               // (gdn_layers, 10240, 4) fp32
  float *d_Alog, *d_dtbias;               // (gdn_layers, 48)
  float* d_gdnnorm = nullptr;             // (gdn_layers, 128)
  float *d_kc = nullptr, *d_vc = nullptr;  // (qsa_layers, maxctx, 512), FP32 mode
  uint16_t *d_kcb = nullptr, *d_vcb = nullptr;  // BF16 KV cache (default mode)
  uint16_t *d_vcbt = nullptr;  // optional BF16 V cache: [block][kvh][dim][slot]
  float *d_qg, *d_qs, *d_gs, *d_kb, *d_vb, *d_attn;
  float *d_pacc = nullptr, *d_pml = nullptr;  // QSA decode flash-decoding partials
  float *d_iproj = nullptr, *d_iq = nullptr, *d_ik = nullptr;
  float *d_iraw = nullptr, *d_inorm = nullptr, *d_iscores = nullptr;
  float* d_iw = nullptr;
  // C1 f16 indexer: f16 twins of d_iq / d_ik / d_ikg（RNE 自同一份 fp32 值），
  // 打分走 k_index_scores_wmma16<false>，score 复用 d_iscores 内存按 f16 存，
  // select 走 ST=__half 变体。原型（logs/c1_f16_proto1.log）：重合率
  // 0.99886-0.99992 ≥ 0.995 门，x1.93-2.00 vs t64。QWENOX_INDEX_F16=0 退出；
  // 其余 index 旧开关（OLDSEL/SCORE64=0/SEL2P=0）任一出现即整路回 fp32。
  uint16_t *d_iqh = nullptr, *d_ikh = nullptr, *d_ikgh = nullptr;
  bool index_f16() const {
    return d_iqh != nullptr;
  }
  // Block ranking is sensitive to projection rounding; retain FP32 here.
  // aggr 10-05: bf16 indexer projection by default (the fp32 SGEMM ran at
  // ~2.4 TFLOPS: 264 ms per 16K chunk; bf16 gemm path 16K +2.6%, ids equal).
  // QWENOX_INDEX_F32=1 (or QWENOX_INDEX_BF16=0) restores the fp32 projection.
  bool index_f32 = getenv("QWENOX_INDEX_F32") != nullptr || env_zero("QWENOX_INDEX_BF16");
  bool index_fused2 = getenv("QWENOX_INDEX_FUSED2") != nullptr;
  bool index_stream_select = getenv("QWENOX_INDEX_STREAM_SELECT") != nullptr;
  // Fused-path fast kernels (09_kernels_index.inc), bit-exact, default on:
  // QWENOX_INDEX_SCORE64=0 -> old k_index_scores_tiled; QWENOX_INDEX_SEL2P=0 ->
  // old selector for n > 8192 (stream / rocPRIM per QWENOX_INDEX_STREAM_SELECT).
  bool index_score64 = !env_zero("QWENOX_INDEX_SCORE64");
  bool index_sel2p = !env_zero("QWENOX_INDEX_SEL2P");
  static bool env_zero(const char* k) {
    const char* v = getenv(k);
    return v && v[0] == '0' && v[1] == 0;
  }
  // ---- dense dequant look-ahead (dpf, 10-14) ----------------------------------
  // Prefill GEMMs read the dense weights as bf16. gemm_wbf16 used to dequantize
  // each weight into d_wbf16 right before its GEMM, serialized on g_str (a
  // per-chunk fixed cost: 15.7 us/tok at chunk 4096, 7.9 at 8192). dpf
  // dequantizes the whole of layer l+1 on a side stream into one of two slots
  // while layer l computes; gemm_wbf16 then returns the slot copy. Same
  // dequant kernels and bytes -> bit-identical. Events: side waits dpf_free_[s]
  // (g_str past the slot's previous layer) before writing slot s; g_str waits
  // dpf_ready_[s] before the layer that reads it. QWENOX_DEQ_AHEAD=0 = off.
  const uint16_t* wpre_override_ = nullptr;  // one-shot gemm_wbf16 result
  bool se_merge_active_ = false;
  // Shared expert gate + up as ONE N=1280 GEMM into the [P][1280] gate|up
  // layout k_silu_mul_g reads (10-14). Only where gemm() would run both N=640
  // halves on the same k_gemm_wmma 128x128 tile (P >= 1024): per-column K
  // order is tile-independent, so gate/up/hid are bit-identical to the two
  // calls. Weights: the two bf16 halves must be adjacent (dpf slot, or both
  // dequantized into d_wbf16). QWENOX_SE_MERGE=0 = two GEMMs.
  bool se_gate_up_merged(const WView& g, const WView& u, int P) {
    static const bool on = !env_zero("QWENOX_SE_MERGE") && getenv("QWENOX_GEMM_WMMA") &&
                           !getenv("QWENOX_GEMM_RB") && !getenv("QWENOX_GEMM_D9_OLD");
    if (!on || P < 1024 || g.dtype == 0 || g.dtype != u.dtype || g.rows != 640 ||
        u.rows != 640 || g.cols != 2560 || u.cols != 2560)
      return false;
    const size_t half = (size_t)640 * 2560;
    const uint16_t* gp = dpf_lookup(g);
    if (gp) {
      if (dpf_lookup(u) != gp + half) return false;
    } else {
      dpf_deq(g, d_wbf16, g_str);
      dpf_deq(u, d_wbf16 + half, g_str);
      gp = d_wbf16;
    }
    WView m = g;
    m.rows = 1280;
    wpre_override_ = gp;
    se_merge_active_ = true;
    gemm(m, d_xb, d_guvb, P);
    se_merge_active_ = false;
    wpre_override_ = nullptr;
    return true;
  }
  struct DpfW {
    WView w;
    size_t off;  // bytes into the slot
  };
  std::vector<std::vector<DpfW>> dpf_list_;                      // per layer
  std::unordered_map<const uint8_t*, std::pair<int, size_t>> dpf_idx_;  // data -> (layer, off)
  uint8_t* dpf_slot_[2] = {nullptr, nullptr};
  int dpf_slot_layer_[2] = {-1, -1};
  hipStream_t dpf_st_ = nullptr;
  hipEvent_t dpf_free_[2] = {}, dpf_ready_[2] = {};
  int dpf_cur_ = -1;     // layer whose slot gemm_wbf16 may read (main waited its ready event)
  bool dpf_busy_ = false;  // a chunk's pipeline owns the slots (nested borrower chunks skip dpf)
  size_t dpf_bytes_ = 0;   // slot size
  // Opt-in dense bf16 shadow (QWENOX_DENSE_SHADOW=1): all ht/q6g64 dense
  // weights dequantized once at load into pinned host memory; gemm_wbf16 then
  // returns the shadow instead of re-running the dequant every chunk. Same
  // kernels, same bytes -> bit-identical. Cost: ~7 GiB locked host RAM, and
  // the dense GEMMs read the shadow over the CPU/GPU fabric instead of VRAM.
  // Measured 128K (Windows, paged 512K, c6144): 1709.5 vs 1779-1791 tok/s
  // without it (-4%) -- the host read costs more than the 46 ms/chunk dequant.
  // Kept as a documented negative result; production stays off.
  uint8_t* dense_shadow_host_ = nullptr;
  uint8_t* dense_shadow_dev_ = nullptr;
  size_t dense_shadow_bytes_ = 0;
  std::unordered_map<const uint8_t*, size_t> dense_shadow_off_;
  // Measured 10-14 (Windows, v2, 64K): ON is SLOWER -- c4096 1522 -> 1488,
  // c8192 1603 -> 1524, single-chunk 8K 1430 -> 1359, 2K 1241 -> 1205: the
  // side-stream dequant kernels steal CUs / bandwidth from the main GEMMs
  // (same finding as gufo's side-stream overlap). Opt-in QWENOX_DEQ_AHEAD=1.
  static bool dpf_env() {
    static const bool on = getenv("QWENOX_DEQ_AHEAD") && !env_zero("QWENOX_DEQ_AHEAD");
    return on;
  }
  static int dpf_minp() {
    static const int v = [] {
      const char* e = getenv("QWENOX_DEQ_AHEAD_MINP");
      return e && *e ? atoi(e) : 64;
    }();
    return v;
  }
  static bool dpf_dtype(int dt) {
    return dt == 5 || dt == 7 || dt == 8 || dt == 16 || dt == 24 || ggd_dtype(dt);
  }
  // Build the per-layer lists (call once ckpt is loaded); returns slot bytes.
  size_t dpf_plan() {
    dpf_list_.assign(g_cfg.layers, {});
    dpf_idx_.clear();
    std::vector<std::pair<std::string, const Tensor*>> ts;
    for (const auto& kv : ckpt.tensors()) {
      const Tensor& t = kv.second;
      if (!dpf_dtype((int)t.dtype)) continue;
      if (kv.first.rfind("layers.", 0) != 0) continue;
      ts.emplace_back(kv.first, &t);
    }
    std::sort(ts.begin(), ts.end());
    size_t mx = 0;
    std::vector<size_t> used(g_cfg.layers, 0);
    for (const auto& e : ts) {
      const int l = atoi(e.first.c_str() + 7);
      if (l < 0 || l >= g_cfg.layers) continue;
      const WView w = wview(*e.second);
      const size_t b = (size_t)w.rows * w.cols * 2;
      if (b > ((size_t)64 << 20) || w.rows < 2) continue;  // gemm_wbf16 would throw / gemv rows
      dpf_list_[l].push_back({w, used[l]});
      dpf_idx_[w.data] = {l, used[l]};
      used[l] += (b + 255) & ~(size_t)255;
      mx = std::max(mx, used[l]);
    }
    return mx;
  }
  // dequant w into out on stream st (same kernels as gemm_wbf16).
  static void dpf_deq(const WView& w, uint16_t* out, hipStream_t st) {
    const uint64_t N = w.rows, K = w.cols;
    if (w.dtype == 5) {
      const uint64_t tot = N * K / 32;
      k_dequant_q4cp_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(
          w.data + 64, w.data + 64 + N * K / 2, (const float*)w.data, out, N, K, w.scale_stride);
    } else if (w.dtype == 7) {
      const uint64_t tot = N * K / 64;
      k_dequant_q8g64_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(w.data, out, N, K);
    } else if (w.dtype == 8) {
      const uint64_t tot = N * K / 32;
      k_dequant_q8g32_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, st>>>(
          (const int8_t*)w.data, (const __half*)(w.data + N * K), out, N, K);
    } else if (ggd_dtype(w.dtype)) {
      ggd_deq_bf16(w, out, st);
    } else {
      v2_deq_bf16(w, out, st);
    }
  }
  void dpf_issue(int l) {
    if (l < 0 || l >= g_cfg.layers || dpf_list_[l].empty()) return;
    const int s = l & 1;
    CK(hipEventRecord(dpf_free_[s], g_str));  // g_str is past every earlier reader of slot s
    CK(hipStreamWaitEvent(dpf_st_, dpf_free_[s], 0));
    for (const DpfW& e : dpf_list_[l]) dpf_deq(e.w, (uint16_t*)(dpf_slot_[s] + e.off), dpf_st_);
    CK(hipEventRecord(dpf_ready_[s], dpf_st_));
    dpf_slot_layer_[s] = l;
  }
  void dpf_begin(int l) {
    const int s = l & 1;
    if (l < 0 || l >= g_cfg.layers || dpf_slot_layer_[s] != l) {
      dpf_cur_ = -1;
      return;
    }
    CK(hipStreamWaitEvent(g_str, dpf_ready_[s], 0));
    dpf_cur_ = l;
  }
  const uint16_t* dpf_lookup(const WView& w) const {
    if (dpf_cur_ < 0) return nullptr;
    auto it = dpf_idx_.find(w.data);
    if (it == dpf_idx_.end() || it->second.first != dpf_cur_) return nullptr;
    return (const uint16_t*)(dpf_slot_[dpf_cur_ & 1] + it->second.second);
  }
  struct DpfScope {  // owns the slots for one prefill chunk's layer loop
    GpuModel& m;
    bool on;
    DpfScope(GpuModel& mm, int P) : m(mm) {
      on = m.dpf_st_ && !m.dpf_busy_ && P >= dpf_minp() && !g_kprof_sync;
      if (!on) return;
      m.dpf_busy_ = true;
      m.dpf_slot_layer_[0] = m.dpf_slot_layer_[1] = -1;
      m.dpf_issue(0);
      m.dpf_issue(1);
    }
    void layer(int l) {  // call right before enq_layer_b(l)
      if (!on) return;
      if (l >= 1) m.dpf_issue(l + 1);
      m.dpf_begin(l);
    }
    void pause() { m.dpf_cur_ = -1; }  // before a borrow / anything outside the layer
    ~DpfScope() {
      if (!on) return;
      m.dpf_cur_ = -1;
      m.dpf_busy_ = false;
      // the side stream may still be writing slots: later chunks re-record
      // free events on g_str, so no extra sync is needed for correctness.
    }
  };

  // QWENOX_ABL=moe,deq,gdn,qsa,hc2 — timing-only ablation (skips the named
  // prefill stage; outputs are garbage). Diagnostic for cost attribution.
  static bool abl_has(const char* what) {
    static const std::string abl = [] {
      const char* e = getenv("QWENOX_ABL");
      return e ? "," + std::string(e) + "," : std::string();
    }();
    return !abl.empty() && abl.find("," + std::string(what) + ",") != std::string::npos;
  }
  // B3 split-K 门控（d_skx 分配与 gemm() dispatch 共用）：k1 (2560,6144)
  // 在 P>=8192 时把 X 重排成 2 个连续 K=3072 切片跑两次 + slice1 原子加。
  // proto（tools/b3_splitk_proto.cu，logs/b3_splitk3.log）：16K 下
  // 14.5+0.5 ms vs 16.4 ms（x1.13）；收益来自 X 连续切片，strided 版 x0.84
  // 已否决。默认开，QWENOX_GEMM_SPLITK=0 退出。
  // B3 split-K: Windows +1.3% (10-03); Linux 16K −1.6% (aggr 10-05: unsplit
  // k1 16.35 ms vs split+repack ~18.8) -> Linux default off, QWENOX_GEMM_SPLITK=1 on.
  static bool gemm_splitk_on() {
#ifdef _WIN32
    static const bool want = !env_zero("QWENOX_GEMM_SPLITK");
#else
    static const bool want = getenv("QWENOX_GEMM_SPLITK") && !env_zero("QWENOX_GEMM_SPLITK");
#endif
    static const bool on = getenv("QWENOX_GEMM_WMMA") != nullptr &&
                           getenv("QWENOX_GEMM_NO_K64") == nullptr &&
                           getenv("QWENOX_GEMM_RB") == nullptr && want;
    return on;
  }
  int* d_iselected = nullptr;
  int index_blocks = 0;
  // Draft lm_head override (default on, QWENOX_DRAFT_LM_Q4=0 opts out): when an
  // overlay upgraded lm_head to a wider dtype, drafting reads the base file's
  // 4-bit copy instead (~0.5x bytes per draft step). Verify/output exactness
  // is unchanged -- only the proposal distribution shifts slightly.
  WView w_dlm{};
  bool dlm_on = false;
  // Paged QSA KV (memory model Phase A1). The trunk FP32 KV cache (d_kc/d_vc)
  // and indexer keys (d_ik) are addressed through a logical->physical page
  // table (KV_PAGE = 256 tokens = 64 indexer blocks; one table shared by all
  // QSA layers). QWENOX_KV_PAGED=1: identity table, =2: reversed table
  // (self-test: output must stay bitwise identical to unpaged). Unset/0: legacy
  // addressing (ptab == nullptr in every kernel). A1c: the BF16 cache
  // (d_kcb/d_vcb and the transposed d_vcbt) is paged too when the sparse
  // prefill kernel is k_qsa_wmma (no WMMA6/UNION). B0: the MTP layer's KV and
  // indexer keys are layer nqsa of the same arrays (kv_nl = nqsa + 1), so
  // they share the page table, the pool, checkpoint pins and COW; d_mkcf/
  // d_mvcf/d_mkcb/d_mvcb/d_mik are views of that slice (no transposed V).
  // A2: pages are allocated on demand from a pool (KvPagePool, 10_ple_io.inc)
  // as the live sequence grows (kv_reserve) and returned when it shrinks or
  // resets (kv_trim). QWENOX_KV_PAGED=1: lowest-free-first allocation, =2:
  // scrambled order + FIFO reuse (self-test). Physical page kv_pool.guard()
  // backs every unmapped logical page.
  // A3: RAM checkpoints (49_rckpt.inc) pin their prefix pages by refcount
  // (kv_pin / kv_unpin) and are restored by remapping (kv_adopt) instead of
  // requiring their tokens to be a prefix of the live stream. A write into a
  // page still shared with a checkpoint copies it first (kv_prepare -> kv_cow).
  // When the pool runs dry, kv_evict_fn drops the LRU checkpoint.
  bool kv_paged = false;
  int kv_lpages = 0;            // logical pages per sequence = ceil(maxctx / 256)
  int kv_pool_pages = 0;        // allocatable physical pages (kv_pool_pages_for)
  int kv_mode = 0;              // QWENOX_KV_PAGED value (2 = self-test logging)
  bool (*kv_evict_fn)(GpuModel&) = nullptr;  // A3: drop one pinned checkpoint
  // Concurrent serve: called once no checkpoint is left to evict. Frees pages
  // held by other sequence slots (true -> retry) or throws KvOverflow.
  bool (*kv_starve_fn)(GpuModel&) = nullptr;
  // Concurrent serve: GPU hand-off point between sequence slots (nullptr =
  // single sequence). Called only where no round-local state is live: spec
  // round tops, between prompt prefill chunks, between serial decode steps.
  void (*yield_fn)(GpuModel&) = nullptr;
  void yield_point() {
    if (yield_fn) yield_fn(*this);
  }
  // 客户端断连/取消标志（serve 模式下 gen_one 指向 Conn::cancel，平时为
  // nullptr）：仅 prompt 预填期间 armed，prefill_batch 在分段间检查并抛
  // ReqCancelled。投机 verify 也走 prefill_batch，但那时该指针为空——
  // verify 中途不能抛（回合本地状态存活），取消由回合边界处理。
  std::atomic<bool>* abort_flag = nullptr;
  // D1b mid-chunk GPU lending (serve mode, nslots > 1): while a long prompt
  // chunk is enqueued layer by layer, queued decode workers may borrow the
  // GPU for light rounds (chunks of at most conc_rmax_ rows) running on a
  // second, small copy of the P-row workspaces. All callbacks null -> inert.
  bool (*borrow_query_fn)() = nullptr;      // a LIGHT worker queued on the turnstile?
  void (*borrow_fn)(GpuModel&) = nullptr;   // release+reacquire under g_borrow_light
  void (*borrow_skip_fn)(GpuModel&) = nullptr;  // heavy chunk during borrow: requeue
  // 10-13 prefill scheduling between concurrent long prompts. A heavy worker
  // (its next GPU work is a prompt chunk of more than conc_rmax_ rows) queues
  // on the turnstile tagged with its remaining prompt tokens (heavy_rem_,
  // read by borrow_fn/borrow_skip_fn right before they re-acquire), so the
  // owner lends mid-chunk only when a light worker waits (lending to a heavy
  // one was a per-layer sync + a hand-off it immediately gave back), and
  // heavy_gate_fn(m, rem) = "a heavy waiter has less prompt left than rem":
  // the chunk about to start requeues behind it (shortest remaining first;
  // ties keep the running one, so equal prompts run first-come-first-served).
  bool (*heavy_gate_fn)(GpuModel&, int) = nullptr;
  int heavy_rem_ = 0;
  // Remaining prompt tokens of the bound slot's prompt ingest (0 = not
  // ingesting, or only a light tail of <= conc_rmax_ rows left). pf_end is the
  // prompt end set by serve around the ingest (per-slot field).
  int pf_end = 0;
  int pf_rem() const {
    if (pf_end <= 0) return 0;
    const int r = pf_end - pos;
    return r > conc_rmax_ ? r : 0;
  }
  // D2-S2a: verify rendezvous pool (serve mode, nslots > 1). Non-null
  // vseg_pool_fn reroutes the spec loop's eager verify into the pool;
  // false = the pool is unavailable for this round, caller falls back to
  // single-segment verify_segments. vseg_active_fn reports how many workers
  // can submit a round now (gamma clamp for the shared row budget).
  struct VSeg;
  // post (optional, sampled spec loops): run by the executor right after the
  // batched verify, still holding the GPU, with this segment's first row in
  // d_logits_b — the worker's logits rows are only valid until the next
  // batch, so their sampling prep (top-k / D2H) happens there.
  bool (*vseg_pool_fn)(GpuModel&, const VSeg&, int*,
                       const std::function<void(int)>*) = nullptr;
  int (*vseg_active_fn)() = nullptr;
  // D2-S2b/c: chain mode joins/leaves the rendezvous membership via these
  // (loop-scope count; ngram rounds excuse themselves per round — they never
  // submit, so counting them would stall the all-arrived trigger).
  void (*vseg_round_enter_fn)() = nullptr;
  void (*vseg_round_exit_fn)() = nullptr;
  void vseg_round_enter() { if (vseg_round_enter_fn) vseg_round_enter_fn(); }
  void vseg_round_exit() { if (vseg_round_exit_fn) vseg_round_exit_fn(); }
  // D2-S3c: draft rendezvous pool (serve mode). Non-null dr_pool_fn reroutes
  // the spec loops' serial per-slot draft burst into the pool (one batched
  // cross-slot pass per step, bit-identical); false = the caller runs the
  // serial burst itself. dr_round_enter/exit_fn: loop-scope membership, same
  // discipline as the verify pool.
  struct DrSeg;
  bool (*dr_pool_fn)(GpuModel&, const DrSeg&) = nullptr;
  void (*dr_round_enter_fn)() = nullptr;
  void (*dr_round_exit_fn)() = nullptr;
  void dr_round_enter() { if (dr_round_enter_fn) dr_round_enter_fn(); }
  void dr_round_exit() { if (dr_round_exit_fn) dr_round_exit_fn(); }
  // S3b: 48 = kVSnapRows。会合 verify 批作为 borrower 在借用窗口内直接跑
  // verify_segments（不经 conc_rmax_ 闸门），S3a 把行预算升到 48 后 32 行的
  // 小工作区会被越界写（B=8 下 "unspecified launch failure"）。
  int conc_rmax_ = 48;                      // borrow round row cap (= small ws rows)
  float conc_quantum_ms_ = 300;             // min GPU-busy ms between lends
  int kv_nl = 0;                // layers in d_kc/d_vc/d_kcb/d_vcb/d_ik (+1 = MTP)
  size_t kv_layer_rows = 0;     // per-layer KV row stride (== maxctx unpaged)
  size_t ik_layer_blocks = 0;   // per-layer d_ik block stride (== index_blocks unpaged)
  int* d_ptab = nullptr;        // [kv_lpages] live logical page -> physical page
  KvPagePool kv_pool;
  KvSeqTable kv_seq;            // live sequence; kv_seq.tab is d_ptab's host mirror
  float* d_ikg = nullptr;       // [index_blocks][128] gathered keys (paged prefill)
  const int* kvpt() const { return kv_paged ? d_ptab : nullptr; }
  bool qsa_dense = getenv("QWENOX_QSA_DENSE") != nullptr;
  // BF16 KV cache: measured parity at 32K (QSA is latency-bound, not DRAM-bound)
  // and flips the 8192 oracle -> opt-in experiment, default stays FP32.
  bool qsa_kv_fp32 = getenv("QWENOX_QSA_KV_BF16") == nullptr;  // default: FP32 cache
  // WMMA sparse prefill attention (k_qsa_wmma): opt-in on top of BF16 KV mode.
  // bf16 P + WMMA accumulation is not bitwise fp32 — FP32 kernels stay default.
  bool qsa_wmma = getenv("QWENOX_QSA_WMMA") != nullptr;
  // Keep the established row-major cache for decode, MTP, snapshots, and
  // the default WMMA path; this second cache is a prefill-only opt-in.
  bool qsa_wmma_btv = qsa_wmma && !qsa_kv_fp32 &&
                      getenv("QWENOX_QSA_WMMA_BTV") != nullptr;
  // QWENOX_QSA_WMMA6=1: dim-split 128-thread rewrite (k_qsa_wmma6); layered on
  // the transposed-V path since it reads B fragments from that cache.
  bool qsa_wmma6 = qsa_wmma_btv && getenv("QWENOX_QSA_WMMA6") != nullptr;
  // QWENOX_QSA_UNION=1: 4-query 块并集 WMMA prefill（k_qsa_union_merge +
  // k_qsa_wmma_u），叠加在转置 V 路径上；组边缘回退 k_qsa_wmma<true>。
  // aggr 10-05: default ON (paged support + tail-edge sel-offset fix added;
  // 16K +0.9% paged, +2.2% unpaged; NOT bit-identical — key order — KLD
  // pending). QWENOX_QSA_UNION=0 restores the per-token kernel.
  // D2-S1b: spec verify chunks (spec_snap) never take the union path — group
  // membership depends on (base+P)&~3, i.e. on gamma, breaking row invariance
  // (γ=2 vs γ=3 near-tie flip at pos 8193; d2s1b_vgd_* probes pcL3).
  bool qsa_union = qsa_wmma_btv && !env_zero("QWENOX_QSA_UNION");
  uint16_t* d_ublk = nullptr;  // [ngroups][QSA_UCAP] 并集块号
  uint16_t* d_umask = nullptr; // [ngroups][QSA_UCAP] 4 query x 4 slot 掩码
  int* d_ucount = nullptr;     // [ngroups] 并集块数（含 pad，4 的倍数）
  size_t qsa_vt_blocks = 0;
  // Per-QSA-layer base of the transposed V cache (nullptr unless BTV).
  uint16_t* vct_layer(int qi) const {
    return qsa_wmma_btv ? d_vcbt + (size_t)qi * qsa_vt_blocks * 2 * 256 * 4 : nullptr;
  }
  bool prefill_fused = getenv("QWENOX_PREFILL_UNFUSED") == nullptr;
  bool gr_bf16 = getenv("QWENOX_GR_BF16") != nullptr && prefill_fused;
  bool gr_scatter_norm = getenv("QWENOX_GR_SCATTER_NORM") != nullptr || gr_bf16;
  // Independent of QWENOX_QSA_DENSE: dense used to force atomic MoE (and so lose
  // MoE_LT), which made it 2x slower in MoE and not a clean attention-only A/B.
  bool moe_deterministic = getenv("QWENOX_MOE_ATOMIC") == nullptr;
  float* d_router;                        // 512
  int* d_topids;                          // 16
  float* d_topw;                          // 16
  float *d_guv, *d_hid, *d_ey, *d_moeacc, *d_sg;
  float *d_e, *d_keys, *d_v, *d_kn, *d_qn, *d_U, *d_ring, *d_convout;
  float* d_ple_convw;                     // 10240*4
  float *d_ple_nk, *d_ple_nq, *d_ple_nc;  // 10240 each
  float* d_logits;
  float* d_logits_b = nullptr;  // [65][vocab] speculative verify lm_head scratch
  int* d_argmax;
  float* d_hnorm_mixer;  // 10240

  // ---- prefill batch state (Phase 3a), sized maxctx ----
  rocblas_handle rbh = nullptr;
  hipblasLtHandle_t lth = nullptr;   // prefill GEMMs (Phase 3e)
  void* d_ltws = nullptr;            // hipBLASLt workspace
  size_t lt_ws_bytes = (size_t)64 << 20;
  // (N,K,P[,flags]) -> full LT descriptor set. Descriptors are shape-implied
  // and self-contained, so cache them for process lifetime instead of
  // create/destroy per call (~55 ms per 8-token chunk of pure host time).
  struct LtEntry {
    hipblasLtMatmulAlgo_t algo;
    hipblasLtMatmulDesc_t opd = nullptr;
    hipblasLtMatrixLayout_t Al = nullptr, Bl = nullptr, Cl = nullptr,
                          Dl = nullptr;
  };
  std::unordered_map<uint64_t, LtEntry> lt_algos;
  // A1: (N,K,Pb) -> validated algo; Pb buckets P (ceil 64 below 1024, 256
  // above). See the A1 comment in gemm(). Cleared together with lt_algos.
  std::unordered_map<uint64_t, hipblasLtMatmulAlgo_t> lt_buckets;
  int* d_posarr;   // [maxctx] d_posarr[i] = i
  int* d_ringarr;  // [maxctx] d_ringarr[i] = i % 9
  int* d_tokarr;   // [maxctx] prefill token ids
  float2* d_ropecs = nullptr;  // [maxbatch][32] per-token rope cos/sin table
  int ropecs_base_ = -1, ropecs_P_ = -1;  // cache key of the last k_rope_cs fill
  unsigned ropecs_gen_ = 0;  // mrope state generation, part of the cache key
  // ---- M-RoPE (vision) request state ----
  // Empty mrope_pos_ = plain text request: rope is untouched (device-table
  // fill, decode delta 0) and bitwise identical to the pre-mrope engine.
  bool mrope_active_ = false;
  int rope_delta_ = 0;                       // host mirror of d_rdelta
  unsigned mrope_gen_ = 0;                   // bumps on set/clear
  std::vector<std::array<int, 3>> mrope_pos_;  // per-prompt-token [T,H,W]
  std::vector<std::array<int, 3>> mrope_grids_;  // request grids (kvsnap meta)
  std::vector<float2> h_ropecs_;             // host fill staging
  int* d_rdelta = nullptr;                   // device scalar: decode rope delta
  float2* d_cs1 = nullptr;                   // dump scratch (32 cos/sin)
  int* d_rpos1 = nullptr;                    // dump scratch (rope position)
  FILE* mrope_dump_ = nullptr;               // QWENOX_MROPE_DUMP sink
  bool mrope_dump_meta_ = false;
  // ---- Vision tower (stage 4) ----
  // vision_on only via load_vision (--vision-tower). Per-request state lives
  // and dies with the mrope request lifecycle (clear_mrope drops it): the
  // stitched bf16 embeds and the (src row, prompt pos) injection map.
  VisionTower vt;
  bool vision_on = false;
  uint16_t* d_img_emb = nullptr;   // [img_emb_rows, 2560] bf16
  size_t img_emb_rows = 0, img_emb_cap = 0;
  std::vector<int2> img_inj;       // per image token: (embed row, prompt pos)
  uint64_t img_hash_ = 0;          // FNV-1a of the request's patch bytes (cont guard)
  int2* d_img_inj = nullptr;       // [maxbatch] chunk-local upload scratch
  std::vector<int2> h_img_inj;     // host staging for the chunk slice
  float *d_Rb, *d_Rhatb, *d_xb, *d_yb, *d_Gb, *d_t320b, *d_w4b;
  // HC inject fusion (gr_write_read_b): double-buffered [BP][4][4] partials of
  // the site injp_for's w4, valid while injp_cur >= 0 (see inj_fuse_on()).
  float* d_injp[2] = {nullptr, nullptr};
  int injp_cur = -1;
  std::string injp_for;
  // B2-1: site whose HC write was already applied by the moe reduce
  // (k_*_hc); the next gr_write_read_b on it runs y=nullptr (norm only).
  std::string hcw_site_;
  std::unordered_map<std::string, uint16_t*> inj_wcache_;  // bf16 inject W per site
  float *d_qkvb, *d_zb, *d_g48b, *d_beta48b;
  float *d_qgb, *d_qsb, *d_gsb, *d_kbb, *d_vbb, *d_attnb;
  float* d_routerb;
  int* d_topidsb;
  float* d_topwb;
  int* d_slot_order = nullptr;  // verify MoE 槽序（k_moe_sort_slots，按专家排序）
  int moe_dump_base_ = -1;      // QWENOX_MOE_TOPIDS_DUMP：记录用的当前 chunk base
  float *d_guvb, *d_hidb, *d_moeaccb, *d_eyb, *d_sgb;
  float* d_eb;              // [maxctx, 2560] PLE embeddings
  float* ple_eb = nullptr;  // pinned staging [maxctx, 2560]
  uint8_t* ple_eb_dev = nullptr;  // device alias of ple_eb (zero-copy fp8 staging)
  uint16_t* d_wbf16;        // dequant workspace (64 MiB)
  uint16_t* d_xbf16;        // [maxctx, 10240] bf16 activations
  uint16_t* d_Rhatbf16 = nullptr;  // survives intermediate projection conversions
  float* d_hcsc = nullptr;  // B2-2: [BP,4] rms inv scales（Rhat 不落盘时由 GEMM 现场归一化）
  // Non-cacheable gemm inputs (d_hidb, d_t320b, attn/conv outs) convert here so
  // they don't evict the d_xb/d_Rhatb cache entries in d_xbf16.
  uint16_t* d_xbf16b = nullptr;  // [maxctx, 6144] bf16
  // A3 dense-LUT dispatch (§11.3): single-"expert" 64-slot tiles scratch for
  // the q4cp dense d3/d9 diversion in gemm() (tiny; (BP/64+2) tiles).
  MoeTile* d_dlut_tiles = nullptr;
  int* d_dlut_nt = nullptr;
  // B3 split-K: [BP, 6144] bf16，k1 (2560,6144) 的 X 重排成 2 个连续 K=3072
  // 切片。层边界即死（仅单次 gemm 内活跃），与 d_wbf16 同理性不入 WS 小拷贝。
  uint16_t* d_skx = nullptr;
  // QWENOX_MOE_LT buffers (allocated only when enabled): gathered bf16 pair rows
  // and per-group dequant scratch.
  __hip_bfloat16* d_moexg = nullptr;   // [BP*topk, d]
  __hip_bfloat16* d_moewup = nullptr;  // [MOE_LT_GRP(x2 if OVL), 2*moe_mid, d]
  __hip_bfloat16* d_moewdn = nullptr;  // [MOE_LT_GRP(x2 if OVL), d, moe_mid]
  uint16_t* d_hidb2 = nullptr;  // QWENOX_MOE_LT_CMP: fused-reference packed hid
  float* d_pairs2 = nullptr;    // QWENOX_MOE_LT_CMP: fused-reference weighted pairs
  int* d_cmpi = nullptr;        // QWENOX_MOE_LT_CMP: [max|diff|, max|ref|] ordered int
  // bf16 conversion cache: d_xb/d_Rhatb are converted once per layer section
  // instead of once per gemm; invalidated where those buffers are written
  const float* xf16_src = nullptr;
  uint64_t xf16_xs = 0;
  int xf16_P = 0, xf16_K = 0;
  // Expert-bucketed MoE with stable GPU routing and independently scheduled tiles.
  int* d_tokidx;       // [maxctx*topk] expert-sorted token indices
  float* d_pweight;    // [maxctx*topk] expert-sorted router weights
  int* d_eoff;         // [experts+1] expert pair offsets
  int* d_pairids = nullptr;  // [maxctx*topk] pair rows in ascending expert order
  GpuMoeRouting moe_routing;
  MoeTile* d_moetiles = nullptr;
  int* d_moentiles = nullptr;
  MoeTile* d_moetilesA = nullptr;  // hybrid tiling: <=64-slot remainder tiles
  int* d_moentilesA = nullptr;
  // QWENOX_MOE_A4 (moe_b_v2): int4 activations + per-128 scales, P-row
  // prefill scratch like d_hidb (written and consumed inside one MoE call).
  uint8_t *d_a4x = nullptr, *d_a4h = nullptr;  // [P][d/2], [P*topk][mid/2]
  float *d_a4xs = nullptr, *d_a4hs = nullptr;  // [P][d/128], [P*topk][mid/128]
  // GGUF W4A4: [IQ4_NL | IQ4_XS | IQ3_S][sign][16 ratio levels][256 pair bytes]
  uint8_t *d_gg_qt = nullptr;
  // GGUF routed experts (gg_on()): P == 1 scratch used by the decode graph;
  // batches reuse the prefill workspaces (see moe_gguf).
  __half* d_gg_x16 = nullptr;   // [d] F16 x
  __half* d_gg_hid = nullptr;   // [topk][moe_mid] F16 up*silu(gate)
  float* d_gg_pairs = nullptr;  // [topk][d] unweighted expert outputs
  float* d_gg_pw = nullptr;     // [topk]
  int *d_gg_tok = nullptr, *d_gg_pid = nullptr, *d_gg_nt = nullptr;
  MoeTile* d_gg_tiles = nullptr;  // [topk]
  int* d_gg_uq = nullptr;         // expert-dedup metadata (moegg::kUqInts), 2 <= P <= 48
  // GDN chunked prefill (Phase 3b)
  float* d_convb;    // [maxctx, 10240] post-conv qkv (conv is not in-place)
  float* d_gdn_ws;   // [48, 2, 64, 128] per-head chunk scratch
  float* d_gdn_split_ws = nullptr;  // [nchunks, 48] intra->inter staging (80.25 KiB each)
  // PLE batched prefill (Phase 3d-1)
  float* d_keysb;     // [maxctx, 10240] key_proj out, normed in place
  float* d_plevb;     // [maxctx, 2560] value_proj out
  float* d_qnb;       // [maxctx, 10240] normed R rows
  float* d_Ub;        // [maxctx, 10240] gated values (pre-norm)
  float* d_Unb;       // [maxctx, 10240] normed U (conv input / ring source)
  float* d_convoutb;  // [maxctx, 10240] conv outputs

  // PLE host state
  std::vector<int64_t> chist;
  int64_t ple_mult[3], ple_vsz[16], ple_voff[16];
  const Tensor* ple_table = nullptr;
  float ple_tscale = 1.f;
  bool ple_iq4 = false;  // table is the GGUF IQ4_NL view (dtype 11, g_ple_rb 90)
  bool ple_bf16 = false;  // table is raw bf16 (dtype 0, g_ple_rb 320, no scale)
  int ple_ringpos = 0;           // host mirror for direct mode
  int* d_token = nullptr;        // device scalars for graph-replayable kernels
  int* d_pos = nullptr;
  int* d_ringpos = nullptr;
  int* p_token = nullptr;        // pinned staging for token id
  int* d_vbase = nullptr;        // P3-A: device mirror of the batch base position
  // QWENOX_VIZ taps (19_viz.inc). Batched rows are tagged with the position
  // source the engine kernels of the current batch use: set by enq_layer_b
  // (dvb ? *dvb : base) and vgraph_run; enq_layer_vseg builds a per-row
  // (slot, position) table. Single-token steps read *d_pos.
  int vz_base_ = 0;
  const int* vz_dvb_ = nullptr;
  const int* vz_rowtab_ = nullptr;
  bool vz_rows_ = false;
  // host = the pinned host rings (fenced, top-5 and publish target); else the
  // device mirror the router taps write.
  // rows: the batch's row count (wrap guard, see VizTap::skip_rows).
  // QWENOX_VIZ_PARTS bitmask (diagnostics): 1 routing, 2 top-5; default 3.
  VizTap vz_tap(int layer, bool batched, bool host = false, int rows = 1) const {
    VizTap t;
    if (!viz::on()) return t;
    static const int parts = [] {
      const char* e = getenv("QWENOX_VIZ_PARTS");
      return e && *e ? atoi(e) : 3;
    }();
    if (!(parts & (layer == g_cfg.layers ? 2 : 1))) return t;
    uint32_t* r = host ? viz::ring(bound_) : viz::dev(bound_);
    if (!r) return t;
    if (batched) {
      if (!vz_rows_) return t;
      t.rowtab = vz_rowtab_;
      t.dpos = vz_rowtab_ ? nullptr : vz_dvb_;
      t.base = vz_rowtab_ || vz_dvb_ ? 0 : vz_base_;
    } else {
      t.dpos = d_pos;
    }
    t.ring = r;
    t.sys = host;
    t.slot = bound_;
    t.layer = layer;
    t.recs = viz::recs();
    t.skip_rows = rows > viz::kRing ? rows - viz::kRing : 0;
    return t;
  }
  // After the last layer's router: copy this pass's rows x layers records
  // from the device mirror to the host rings (one launch per pass).
  void vz_publish(int rows, bool batched) {
    if (!viz::on()) return;
    const VizTap d = vz_tap(0, batched, false, rows), h = vz_tap(0, batched, true, rows);
    if (!d.ring || !h.ring) return;
    const int n = rows * g_cfg.layers;
    k_viz_publish<<<(n + 255) / 256, 256, 0, g_str>>>(d, h, rows, g_cfg.layers);
  }
  int* p_vbase = nullptr;        // pinned staging for d_vbase
  int pad_id = 248044;

  // PLE chunk-ahead prefetch: row addresses are pure functions of the last
  // two token ids, so the next chunks' disk gathers run on host threads while
  // this chunk's layers occupy the GPU (the gather otherwise leaves the GPU
  // idle behind embed+L0). Two flights (N+1, N+2) with three pinned staging
  // buffers: a multi-second disk stall is hidden behind two chunks of layer
  // compute instead of one. Same rows, same bytes, same consumers.
  float* ple_eb2 = nullptr;        // second pinned staging [maxbatch_cap, 2560]
  uint8_t* ple_eb_dev2 = nullptr;  // its device alias
  float* ple_eb3 = nullptr;        // third pinned staging (lookahead)
  uint8_t* ple_eb_dev3 = nullptr;
  hipEvent_t ple_stg_ev[3] = {};   // recorded after each staging consume
  bool ple_stg_ev_ok[3] = {};
  int ple_par = 0;                 // next staging index (rotates 0..2)
  uint8_t* ple_stgbuf(int b) const {
    return (uint8_t*)(b == 0 ? ple_eb : b == 1 ? ple_eb2 : ple_eb3);
  }
  uint8_t* ple_stgdev(int b) const {
    return b == 0 ? ple_eb_dev : b == 1 ? ple_eb_dev2 : ple_eb_dev3;
  }
  int ple_take_buf() {
    const int b = ple_par;
    ple_par = (ple_par + 1) % 3;
    return b;
  }
  struct PleFlight {
    std::vector<std::thread> pool;
    std::vector<uint64_t> addrs, keys;
    std::vector<int> toks;       // the chunk's tokens (cat() window)
    std::atomic<int> barrier{0};
    std::atomic<int> workers{0};
    int64_t pre0 = 0, pre1 = 0;  // tokens before the window (pad_id at start)
    uint8_t* fp8 = nullptr;      // staging this flight fills
    int base = -1, P = 0, buf = 0;
    // Owner slot (bound_ at spawn). ple_fl is NOT per-slot state: with two
    // concurrent prefills in lockstep (same chunk grid from base 0) slot B's
    // chunk-ahead flight has the same base/P as slot A's next chunk, so a
    // base/P-only match fed A with B's PLE rows (10-13). Hits now require
    // owner + tokens + history pair (ple_flight_hit).
    int slot = -1;
    bool valid = false;
    bool inline_io = false;  // Windows small batch: reads posted, no threads (ple_join waits)
    std::chrono::steady_clock::time_point start, done;
    // Batched verify (prefill_chunk_vseg): toks = nseg concatenated segments
    // of different slots; segment s spans [seg_start[s], seg_start[s+1]) and
    // has its own history pair. nseg == 0: one window (pre0/pre1 above).
    int nseg = 0;
    int seg_start[9] = {};
    int64_t seg_pre0[8] = {}, seg_pre1[8] = {};
  } ple_fls[2];
  int ple_cur_i = 0;
  PleFlight& ple_cur() { return ple_fls[ple_cur_i]; }
  PleFlight& ple_next() { return ple_fls[ple_cur_i ^ 1]; }
  void ple_drop_one(PleFlight& fl) {
    if (fl.valid || fl.inline_io || !fl.pool.empty()) ple_join(fl);
    fl.valid = false;
  }
  void ple_drop_all() {
    ple_drop_one(ple_fls[0]);
    ple_drop_one(ple_fls[1]);
  }
  struct PleSegs {  // ple_spawn argument for a multi-segment flight
    int n = 0;
    int start[9] = {};
    int64_t pre0[8] = {}, pre1[8] = {};
  };

  // ---- MTP speculative draft head (Phase A) ----
  // Single QSA+MoE layer on top of the trunk's 48-layer stream; predicts
  // t_{p+2} from (trunk h_p, emb(t_{p+1})) (DeepSeek-style wiring). Owns its
  // stream/KV/indexer state; reuses trunk scratch (strictly serial use).
  bool mtp_avail = false;      // checkpoint has mtp.* tensors and env is default
  // mtp_avail's condition, known before the KV pool is sized (B0: the MTP
  // layer is pool layer nqsa).
  bool mtp_moe_default() const {
    return moe_deterministic && !moe_ordered && !moe_host_route && !moe_untiled &&
           !moe_fp32_io;
  }
  bool mtp_kv_wanted() const {
    return ckpt.find("mtp.fc_hidden.weight") && mtp_moe_default();
  }
  bool ngram_verify_active = false;
  bool ngram_legacy_gdn = getenv("QWENOX_NGRAM_LEGACY_GDN") != nullptr;
  bool ngram_legacy_final = getenv("QWENOX_NGRAM_LEGACY_FINAL") != nullptr;
  bool mtp_tap_capture = false;  // prefill_chunk copies d_Rb rows into d_mtap
  bool bench_quiet = false;      // offline bench: suppress per-layer/per-run diagnostics
  int mtp_norm_mode = 0;       // 0 = full-10240 RMSNorm (H1), 1 = 4x2560 (H2)
  int mtp_hagg_mode = 0;       // 0 = sum branches + repeat4, 1 = per-branch + bcast e
  int mtp_pos = 0;             // host mirror of the MTP stream position
  float* d_mR = nullptr;       // MTP hyper stream (10240)
  float* d_mRhat = nullptr;    // MTP normed stream (10240)
  // S3c: d_mR 入 SLOT_FIELDS 后指向本池的第 bound_ 行（[nslots][10240] 连续，
  // 供 draft 跨槽合批按行直读）；nslots==1 时为空、d_mR 用构造器独立分配。
  float* d_mRpool = nullptr;
  float* d_mh = nullptr;       // input h-path accumulator (2560)
  float* d_me = nullptr;       // input e-path normed embed (2560)
  float* d_meo = nullptr;      // input e-path fc_embedding out (2560)
  float* d_mt = nullptr;       // input per-branch gemv tmp (2560)
  float* d_mnorm_e = nullptr;  // mtp.pre_fc_norm_embedding [2560] fp32
  float* d_mnorm_h = nullptr;  // mtp.pre_fc_norm_hidden [10240] fp32
  int* d_mtok = nullptr;       // device scalar: draft input token
  int* d_mpos = nullptr;       // device scalar: MTP position
  // MTP KV / indexer keys: views of pool layer nqsa (B0), not allocations
  float* d_mkcf = nullptr;     // FP32 mode (kv_layer_rows, 512)
  float* d_mvcf = nullptr;
  uint16_t* d_mkcb = nullptr;  // BF16 mode
  uint16_t* d_mvcb = nullptr;
  float* d_mik = nullptr;      // pooled keys (ik_layer_blocks, 128)
  float* d_miraw = nullptr;    // MTP indexer raw ring (4, 128)
  float* d_minorm = nullptr;   // MTP indexer norms q[128] + k[128]
  float* d_miw = nullptr;      // MTP indexer fp32 proj (640, 2560)
  float* d_mRb = nullptr;      // batch stream (maxbatch, 10240)
  float* d_mRhatb = nullptr;   // batch normed stream
  float* d_mtap = nullptr;     // captured trunk taps (maxbatch, 10240)
  // ---- DFlash2 draft head (sidecar GGUF, 41_dflash2.inc) ----
  // Independent of MTP: a 5-block masked draft with its own KV ring and a
  // selector. Context features are HC-mean taps of trunk layers 5/15/25/35/45.
  Df2State df2;
  bool df2_init();
  void df2_capture_layer(int l, int P, int base, int mode);
  void df2_hc_capture(const std::string& prefix, int P);
  void df2_inject(int base, int P, bool verify_rows);
  void df2_inject_chunk(int base, int P);
  void df2_propose(int anchor, int g, int* drafts, bool select = true);
  template <typename Sampler>
  std::vector<int> df2_spec_loop_sample(int cur, int gen, Sampler& sampler,
                                        const std::function<bool(int, float)>& emit);
  void df2_round_dump(int round, int anchor, int g, const int* drafts);
  std::vector<int> df2_spec_loop(int cur, int gen, int gamma,
                                 const std::function<bool(int)>& emit,
                                 bool hybrid = false, int mtp_gamma = 4);
  // Offline test hook (--spec-gen + QWENOX_SPEC_FORCE=<ids file>): greedy
  // spec_loop verifies drafts against these ids instead of the trunk argmax,
  // so every γ policy commits the same trajectory (paired γ A/B without greedy
  // near-tie divergence; verify cost unchanged). Empty = off.
  std::vector<int> spec_force;
  // Stats for the most recent speculative request, emitted on the terminal
  // serve D line so clients can distinguish MTP from serial decode.
  int last_spec_rounds = 0;
  int last_spec_commit = 0;
  int last_spec_rollbacks = 0;
  int last_spec_proposed = 0;  // sum of drafts offered (D line trailing field)
  // Live progress reporting (2026-09-13): 1s-cadence prefill/decode stats on
  // stderr. Prefill is armed by serve around the prompt ingest only (never
  // for spec-verify batches) and uses NON-BLOCKING event probes — no stream
  // sync, so the pipeline is undisturbed. Decode reads the live spec
  // counters the loops bump every round.
  int prog_total = 0;         // prompt tokens to prefill this request (0=off)
  int64_t prog_base = 0;      // prompt tokens completed before current chunk
  int64_t prog_last_est = 0;  // est tokens at the last report (interval rate)
  int64_t prog_req = -1;      // request id stamped on progress lines
  std::chrono::steady_clock::time_point prog_t0, prog_last;
  hipEvent_t prog_ev[64] = {};  // per-layer completion probes (DisableTiming)
  int live_spec_rounds = 0;
  int live_spec_commits = 0;
  int live_spec_proposed = 0;
  // Trunk rollback checkpoint for verify: S + convst + PLE ring + trunk iraw,
  // plus a separate tiny slot for the MTP indexer ring (saved post-draft-1).
  float* d_ckpt = nullptr;
  size_t ckpt_S = 0, ckpt_convst = 0, ckpt_ring = 0, ckpt_iraw = 0;
  float* d_mring_ckpt = nullptr;  // 512 floats
  int* d_argmax_arr = nullptr;    // per-row verify argmax (<= 65 rows)
  int* d_tokmat = nullptr;  // D2-S3c: [8 step][8 row] batched draft argmax
  unsigned long long* d_amx_cand = nullptr;  // two-stage argmax scratch
  // GPU sampling prep (spec sampling fast path): correction lists, radix
  // top-k scratch and per-row candidate blocks. Rows are capped at 65 (ngram
  // <= 64 + 1); the dense path stays as fallback (top_k unset/>64, candidate
  // overflow, QWENOX_SMS_GPU=0).
  static constexpr int SMS_ROWS = 65, SMS_TCAP = 512, SMS_PCAP = 64;
  static constexpr int SMS_BIAS_CAP = 20480, SMS_HIST_CAP = 120000;
  int* d_sm_ctrl = nullptr;        // [SMS_ROWS][8]
  unsigned* d_sm_hist = nullptr;   // [SMS_ROWS][4096]
  int* d_sm_cand_ids = nullptr;    // [SMS_ROWS][SMS_TCAP]
  float* d_sm_cand_vals = nullptr; // [SMS_ROWS][SMS_TCAP]
  int* d_sm_cand_n = nullptr;      // [SMS_ROWS]
  int* d_sm_bids = nullptr;        // bias list (shared per round)
  float* d_sm_bvals = nullptr;
  int* d_sm_hids = nullptr;        // penalty list from sampler hist (shared)
  float* d_sm_hvals = nullptr;
  int* d_sm_pids = nullptr;        // per-row draft-prefix penalties
  float* d_sm_pvals = nullptr;     // [SMS_ROWS][SMS_PCAP]
  int* d_sm_pn = nullptr;          // [SMS_ROWS]
  float* d_sm_scratch = nullptr;   // corrected-logits copy (lazy, corr only)
  int sm_nb = 0, sm_nh = 0;        // uploaded shared correction counts
  std::vector<int> h_sm_ids;       // D2H landing + host truncation staging
  std::vector<float> h_sm_vals;
  std::vector<int> h_sm_ci;        // shared-correction upload staging
  std::vector<float> h_sm_cv;
  std::vector<int> h_sm_pi;        // prefix upload staging (separate: the
  std::vector<float> h_sm_pv;      //   shared H2D may still be in flight)
  std::vector<int> sms_out_ids;    // truncated top-k results [rows][k]
  std::vector<float> sms_out_vals;
  // Rollback snapshots (spec verify): S/convst per accepted depth, indexer
  // proj rows per QSA layer. Written only when spec_snap is set. One slot's
  // verify chunk needs gamma+1 <= 9 rows (gamma <= 8); kVSnapRows reserves
  // extra rows so several slots' chunks can coexist (D2 rendezvous), each
  // slot addressing its segment via vsnap_row0_.
  // 48 = 8 slots × (gamma 5 + 1): B=8 rendezvous no longer clamps gamma
  // (24-row budget forced gamma<=2 there). Cost: +2.8 GiB arena (S0 的
  // 9→24 为 +1.73 GiB)，devarena estimate 86.4→89.2 + 4 slack = 93.2 < 95。
  // Runtime since 10-12: 48 with several slots (rendezvous), else 9 — a
  // single sequence paid ~4.4 GiB of arena for rows it never used.
  const int kVSnapRows = vsnap_rows_cfg();
  // GDN replay inputs per verify row (k_gdn_rowsave): [kVSnapRows][ngdn]
  // [kGdnRowF]; replaced the per-row S snapshots [kVSnapRows][ngdn][3.1 MB].
  float* d_grow = nullptr;
  float* d_csnap = nullptr;    // [kVSnapRows][ngdn][3*10240]
  float* d_iproj_snap = nullptr;  // [nqsa][kVSnapRows][128]
  int vsnap_row0_ = 0;         // row base of the bound slot's snap segment
  bool spec_snap = false;         // set around spec verify chunks only
  // aggr 10-05 in-pass snapshot cut: the serve loop used to split the prompt
  // at the frontend snapshot point cc (prefill [0,cc), checkpoint, then a tiny
  // [cc,N) pass ~70 ms of extra TTFT). With cut_abs = cc the single pass
  // assembles the recurrent state at cc into shadow buffers instead (GDN S by
  // splitting the fused scan at the cut row, conv / PLE ring / indexer rings
  // by replaying their update with the first r rows, d_R = stream row cc-1,
  // MTP ring at cc-1); with_cut_state() swaps them in for rckpt / kvsnap.
  int cut_abs = -1;   // absolute cut position, -1 = off
  int cut_r_ = 0;     // chunk-local cut row this chunk (0 = none)
  bool cut_hit = false, cut_mtp_hit = false;
  float *d_cS = nullptr, *d_cconvst = nullptr, *d_cring = nullptr, *d_ciraw = nullptr,
        *d_cR = nullptr, *d_cmiraw = nullptr;
  bool cut_busy = false;  // NOT per slot: one shadow set, one owner at a time
  hipGraphExec_t g1_exec = nullptr, g2_exec = nullptr;  // per-slot decode graphs

  // ---- Sequence slots (concurrent serve, QWENOX_PARALLEL) ----
  // Everything one sequence carries from step to step: host mirrors, GDN /
  // conv / PLE ring / indexer ring state, the page table (the KV pages live
  // in the shared pool, MTP layer included), the MTP indexer ring and
  // position, the verify rollback checkpoints (d_ckpt / d_mring_ckpt — D2
  // swaps slots between checkpoint save and restore), per-slot decode graphs
  // (they bake these
  // pointers) and the request-scoped flags/stats. Weights, prefill/decode
  // scratch, logits and the KV pool are shared: slots
  // time-slice the GPU and hand it over only at yield_point(), where none of
  // the shared buffers holds live data. bind(s) swaps slot s into the model
  // members, so every kernel path runs unchanged on the bound slot.
  // mrope_gen_ is NOT per slot: it is a global generation for the ropecs
  // cache key (bind also drops that cache).
#define QWENOX_SLOT_FIELDS(X)                                                    \
  X(pos) X(ple_ringpos) X(chist) X(ngram) X(ng_gate) X(ng_cs) X(mtp_pos)     \
  X(kv_seq) X(d_ptab)                                                         \
  X(mrope_active_) X(rope_delta_) X(mrope_pos_) X(mrope_grids_) X(d_rdelta)   \
  X(img_inj) X(img_hash_) X(d_img_emb) X(img_emb_rows) X(img_emb_cap)         \
  X(mtp_tap_capture) X(spec_snap) X(vsnap_row0_) X(ngram_verify_active)       \
  X(cut_abs) X(cut_r_) X(cut_hit) X(cut_mtp_hit) X(d_mR)                      \
  X(prog_total) X(prog_base) X(prog_last_est) X(prog_req) X(prog_t0) X(pf_end) \
  X(prog_last) X(last_spec_rounds) X(last_spec_commit) X(last_spec_rollbacks) \
  X(last_spec_proposed) X(live_spec_rounds) X(live_spec_commits)              \
  X(live_spec_proposed) X(g1_exec) X(g2_exec)                                 \
  X(d_S) X(d_convst) X(d_ring) X(d_iraw) X(d_R) X(d_pos) X(d_ringpos)         \
  X(d_miraw) X(d_mpos) X(d_ckpt) X(d_mring_ckpt)
  struct SeqState {
#define QWENOX_SLOT_DECL(f) decltype(GpuModel::f) f{};
    QWENOX_SLOT_FIELDS(QWENOX_SLOT_DECL)
#undef QWENOX_SLOT_DECL
  };
  // Invariant: the bound slot's state is in the members and slots_[bound_]
  // holds a placeholder; every other slots_[i] holds slot i's state.
  std::vector<SeqState> slots_;
  int nslots = 1, bound_ = 0;
  void swap_slot(SeqState& s) {
    using std::swap;
#define QWENOX_SLOT_SWAP(f) swap(f, s.f);
    QWENOX_SLOT_FIELDS(QWENOX_SLOT_SWAP)
#undef QWENOX_SLOT_SWAP
  }
  void bind(int s) {
    if (s == bound_) return;
    CK(hipDeviceSynchronize());  // nothing of the outgoing slot still in flight
    swap_slot(slots_[(size_t)bound_]);  // members <- placeholder
    swap_slot(slots_[(size_t)s]);       // members <- slot s
    bound_ = s;
    ropecs_base_ = -1;   // rope cos/sin table belongs to the previous slot
    xf16_src = nullptr;  // so does the bf16 conversion cache
    viz::ensure(s);      // QWENOX_VIZ: slot ring exists before any capture on it
  }

  // D2-S1c: per-slot state accessors for multi-segment verify. The bound
  // slot's state lives in the members (bind() swaps it in), every other
  // slot's in slots_[s]. slot_f(bound_) reads the member, so code written
  // against these accessors behaves exactly like the member path whenever
  // the segment's slot is the bound one.
#define QWENOX_SLOT_ACC(f)                                                     \
  auto& slot_##f(int s) { return s == bound_ ? f : slots_[(size_t)s].f; }      \
  const auto& slot_##f(int s) const {                                         \
    return s == bound_ ? f : slots_[(size_t)s].f;                              \
  }
  QWENOX_SLOT_FIELDS(QWENOX_SLOT_ACC)
#undef QWENOX_SLOT_ACC
  // Paged-KV fast path for large prefill chunks: when the live sequence's
  // pages are the identity map, the kernels' legacy ptab==nullptr addressing
  // reads/writes the same bytes and skips the per-block lookup. Decode,
  // verify and graph-captured paths (P < 256) keep the real table so baked
  // pointers stay valid; the scrambled self-test (kv_mode==2) also keeps it.
  const int* slot_ptab_eff(int s, int P) {
    if (!kv_paged) return nullptr;
    if (kv_mode == 2 || P < 256) return slot_d_ptab(s);
    if (env_zero("QWENOX_KV_IDENTITY")) return slot_d_ptab(s);  // A/B escape
    const bool id = slot_kv_seq(s).identity;
    if (id) {
      static bool logged = false;
      if (!logged && getenv("QWENOX_KV_DEBUG")) {
        logged = true;
        fprintf(stderr, "[kvpage] identity ptab fast path on (P=%d)\n", P);
      }
      return nullptr;
    }
    return slot_d_ptab(s);
  }
  // Greedy draft chain without per-step D2H: step j reads step j-1's argmax
  // via D2D, one blocking D2H collects the chain. 0 restores per-step sync.
  const bool draft_desync =
      !getenv("QWENOX_DRAFT_DESYNC") || atoi(getenv("QWENOX_DRAFT_DESYNC")) != 0;
  // Draft-only vocab subset (first N rows of lm_head, FR-Spec style): greedy
  // draft argmax over the subset; verify is untouched, so committed output is
  // unchanged even when the true argmax lies outside. 0 = full vocab.
  // Greedy argmax path only; sampling drafts keep the full distribution.
  const int draft_vocab = [] {
    const char* e = getenv("QWENOX_DRAFT_VOCAB");
    return e ? atoi(e) : 0;
  }();
  // Dump top-2 verify logits for rows near absolute position N (draft
  // dependence forensics: tiny margin at a fork = fp tie-flip, large = leak).
  const int fork_dump = [] {
    const char* e = getenv("QWENOX_FORK_DUMP");
    return e ? atoi(e) : 0;
  }();

  explicit GpuModel(const Checkpoint& c, int mc) : ckpt(c), maxctx(mc) {
    if (qsa_union && maxctx > 262144) {  // block ids uint16: 65536 blocks max
      fprintf(stderr, "qsa: disabling QWENOX_QSA_UNION above 256K because its "
                      "block ids are uint16\n");
      qsa_union = false;
    }
    ensure_rope_tab(g_cfg.rope_theta, g_cfg.rotary_dim);
    CK(hipStreamSynchronize(nullptr));
    // Prefill workspace is sized by maxbatch, NOT maxctx: long prompts are
    // processed in maxbatch-token chunks via the same continuation path as
    // KV reuse, so a 256K context costs chunk-sized workspace (~4.5 GB at
    // 8192) instead of ~541 KB/token of context.
    maxbatch = eff_maxbatch(maxctx);
    maxbatch_cap = std::min(maxctx, maxbatch + eff_tail_slack());
    int rb = g_cfg.branches * g_cfg.d;
    DALLOC(&d_R, rb * 4);
    DALLOC(&d_Rhat, rb * 4);
    DALLOC(&d_x, rb * 4);
    DALLOC(&d_y, rb * 4);
    DALLOC(&d_emb, g_cfg.d * 4);
    DALLOC(&d_t320, g_cfg.gr_rank * 4);
    DALLOC(&d_G, rb * 4);
    DALLOC(&d_w4, 16);
    DALLOC(&d_qkv, 10240 * 4);
    DALLOC(&d_g48, 48 * 4);
    DALLOC(&d_beta48, 48 * 4);
    DALLOC(&d_z, 6144 * 4);
    int ngdn = g_cfg.layers - g_cfg.layers / 4;  // 36
    DALLOC(&d_S, (size_t)ngdn * 48 * 128 * 128 * 4);
    CK(hipMemset(d_S, 0, (size_t)ngdn * 48 * 128 * 128 * 4));
    DALLOC(&d_convst, (size_t)ngdn * 3 * 10240 * 4);
    CK(hipMemset(d_convst, 0, (size_t)ngdn * 3 * 10240 * 4));
    DALLOC(&d_convw, (size_t)ngdn * 10240 * 4 * 4);
    DALLOC(&d_Alog, (size_t)ngdn * 48 * 4);
    DALLOC(&d_dtbias, (size_t)ngdn * 48 * 4);
    DALLOC(&d_gdnnorm, (size_t)ngdn * 128 * 4);
    // ---- prefill batch buffers (Phase 3a), allocated once at maxctx ----
    // 提到最前（紧随权重 arena）：批缓冲的 gemm 速度随 arena 内绝对地址升高
    // 单调变差（139264 pads 1→12 GiB 实测稳态 1310→1240 tok/s、gdn:proj
    // 2257→2639 ms，无周期）；而 maxctx 尺寸的 KV/ik/vcbt/iscores 对位置
    // 不敏感（qsa:flash/idx 在两位形间不变）。固定低位摆放同时让布局与
    // maxctx 无关——此前 262144 仅因这些缓冲被 maxctx 增长顶高 5.21 GiB
    // 就掉 14%（1464.5→1253.8 tok/s @128k，E1 用 QWENOX_PAD_BATCH 复现）。
    // QWENOX_PAD_BATCH (bytes)：调试开关——哑分配整体推移批缓冲的 arena 偏移。
    {
      const char* pbs = getenv("QWENOX_PAD_BATCH");
      if (pbs && atoll(pbs) > 0) {
        void* pad_dummy = nullptr;
        DALLOC(&pad_dummy, (size_t)atoll(pbs));
      }
    }
    {
      size_t BP = (size_t)maxbatch_cap;
      DALLOC(&d_Rb, BP * 10240 * 4);
      DALLOC(&d_Rhatb, BP * 10240 * 4);
      DALLOC(&d_xb, BP * 2560 * 4);
      DALLOC(&d_yb, BP * 2560 * 4);
      DALLOC(&d_Gb, BP * 10240 * 4);
      DALLOC(&d_t320b, BP * 320 * 4);
      DALLOC(&d_w4b, BP * 4 * 4);
      DALLOC(&d_injp[0], BP * 16 * 4);
      DALLOC(&d_injp[1], BP * 16 * 4);
      DALLOC(&d_qkvb, BP * 10240 * 4);
      DALLOC(&d_zb, BP * 6144 * 4);
      DALLOC(&d_g48b, BP * 48 * 4);
      DALLOC(&d_beta48b, BP * 48 * 4);
      DALLOC(&d_qgb, BP * 12288 * 4);
      DALLOC(&d_qsb, BP * 6144 * 4);
      DALLOC(&d_gsb, BP * 6144 * 4);
      DALLOC(&d_kbb, BP * 512 * 4);
      DALLOC(&d_vbb, BP * 512 * 4);
      DALLOC(&d_ropecs, BP * 32 * sizeof(float2));
      DALLOC(&d_attnb, BP * 6144 * 4);
      DALLOC(&d_routerb, BP * 512 * 4);
      DALLOC(&d_topidsb, BP * 16 * 4);
      DALLOC(&d_topwb, BP * 16 * 4);
      DALLOC(&d_slot_order, 256 * 4);  // k_moe_sort_slots 输出（P*k ≤ 256）
      // Fused down reuses the otherwise idle gate/up workspace for pair outputs.
      // B1：v2+w4b pairs 全 f16（guvb_f16_env，与 30_host_util 估算同步）后按
      // f16 尺寸分配，省一半；任一回退 env 在位则保持 f32 全尺寸。
      DALLOC(&d_guvb,
             BP * g_cfg.topk * (moe_deterministic ? 2560 : 1280) *
                 (guvb_f16_env() ? 2 : 4));
      DALLOC(&d_hidb, BP * g_cfg.topk * 640 * 4);
      DALLOC(&d_moeaccb, BP * 2560 * 4);
      DALLOC(&d_eyb, BP * 2560 * 4);
      DALLOC(&d_sgb, BP * 4);
      DALLOC(&d_eb, BP * 2560 * 4);
      CK(hipHostMalloc((void**)&ple_eb, BP * 2560 * 4, hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ple_eb_dev, ple_eb, 0));
      CK(hipHostMalloc((void**)&ple_eb2, BP * 2560 * 4, hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ple_eb_dev2, ple_eb2, 0));
      CK(hipHostMalloc((void**)&ple_eb3, BP * 2560 * 4, hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ple_eb_dev3, ple_eb3, 0));
      CK(hipEventCreateWithFlags(&ple_stg_ev[0], hipEventDisableTiming));
      CK(hipEventCreateWithFlags(&ple_stg_ev[1], hipEventDisableTiming));
      CK(hipEventCreateWithFlags(&ple_stg_ev[2], hipEventDisableTiming));
      DALLOC(&d_wbf16, (size_t)64 << 20);  // fits q_proj 12288x2560 bf16
      DALLOC(&d_xbf16, BP * 10240 * 2);
      DALLOC(&d_xbf16b, BP * 6144 * 2);  // largest non-cacheable K = 6144
      if (gemm_splitk_on()) DALLOC(&d_skx, BP * 6144 * 2);
      if (prefill_fused) DALLOC(&d_Rhatbf16, BP * 10240 * 2);
      if (prefill_fused) DALLOC(&d_hcsc, BP * 4 * 4);
      DALLOC(&d_tokidx, BP * g_cfg.topk * 4);
      DALLOC(&d_pweight, BP * g_cfg.topk * 4);
      DALLOC(&d_eoff, (g_cfg.experts + 1) * 4);
      if (moe_deterministic) DALLOC(&d_pairids, BP * g_cfg.topk * 4);
      if (!moe_host_route && !moe_ordered) moe_routing.init(BP * g_cfg.topk, g_cfg.experts);
      if (!moe_untiled && !moe_ordered) {
        DALLOC(&d_moetiles, ((BP * g_cfg.topk + 63) / 64 + g_cfg.experts) * sizeof(MoeTile));
        DALLOC(&d_moentiles, sizeof(int));
      }
      if (moe_lt && g_cfg.topk == 10 && moe_deterministic && !moe_host_route && !moe_ordered) {
        // 4-expert dequant groups: scratch stays MALL-resident between dequant
        // and its GEMMs (measured in tools/moe_lt_bench.cu). QWENOX_MOE_LT_OVL
        // doubles it for the A/B slot rotation of the overlap pipeline.
        DALLOC(&d_moexg, (size_t)BP * g_cfg.topk * g_cfg.d * 2);
        DALLOC(&d_moewup, (size_t)4 * 2 * g_cfg.moe_mid * g_cfg.d * 2 * (moe_lt_ovl ? 2 : 1));
        DALLOC(&d_moewdn, (size_t)4 * g_cfg.d * g_cfg.moe_mid * 2 * (moe_lt_ovl ? 2 : 1));
        if (moe_lt_cmp) {
          DALLOC(&d_hidb2, (size_t)BP * g_cfg.topk * g_cfg.moe_mid * 2);
          DALLOC(&d_pairs2, (size_t)BP * g_cfg.topk * g_cfg.d * 4);
          DALLOC(&d_cmpi, 8);
        }
      }
      DALLOC(&d_convb, BP * 10240 * 4);
      DALLOC(&d_gdn_ws, (size_t)48 * 2 * 64 * 128 * 4);
      if (!gdn_nosplit && !gdn_loop && !gdn_fused)  // fused needs no ws; streaming caps it at window chunks (15.05 MiB @4); pipeline double-buffers the slots
        DALLOC(&d_gdn_split_ws,
                     (size_t)(gdn_stream ? std::min<size_t>(gdn_window_chunks, (BP + 63) / 64)
                                         : (BP + 63) / 64) * 48 * GDN_SPLIT_WS_FLOATS * 4 *
                         (gdn_pipe ? 2 : 1));
      DALLOC(&d_keysb, BP * 10240 * 4);
      DALLOC(&d_plevb, BP * 2560 * 4);
      DALLOC(&d_qnb, BP * 10240 * 4);
      DALLOC(&d_Ub, BP * 10240 * 4);
      DALLOC(&d_Unb, BP * 10240 * 4);
      DALLOC(&d_convoutb, BP * 10240 * 4);
      // d_posarr/d_ringarr are indexed by ABSOLUTE position (qsa_loop debug,
      // ple_gpu_t): they stay maxctx-sized. d_tokarr only ever holds one
      // prefill chunk.
      DALLOC(&d_posarr, (size_t)maxctx * 4);
      DALLOC(&d_ringarr, (size_t)maxctx * 4);
      DALLOC(&d_tokarr, BP * 4);
      DALLOC(&d_img_inj, BP * sizeof(int2));  // vision injection map
      std::vector<int> h(maxctx);
      for (int i = 0; i < maxctx; i++) h[i] = i;
      CK(hipMemcpy(d_posarr, h.data(), (size_t)maxctx * 4, hipMemcpyHostToDevice));
      for (int i = 0; i < maxctx; i++) h[i] = i % 9;
      CK(hipMemcpy(d_ringarr, h.data(), (size_t)maxctx * 4, hipMemcpyHostToDevice));
      if (rocblas_create_handle(&rbh) != rocblas_status_success) {
        fprintf(stderr, "rocblas_create_handle failed\n");
        exit(1);
      }
      if (hipblasLtCreate(&lth) != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "hipblasLtCreate failed\n");
        exit(1);
      }
      DALLOC(&d_ltws, lt_ws_bytes);
      // dlut 小分配一律放 WS 末尾：插在中间会把其后所有缓冲推开（256 B 对齐后
      // 768 B），gdn/gr 工作区地址敏感（§13.8：P=2048 hc +20 µs/tok 的成因）。
      DALLOC(&d_dlut_tiles, (BP / 64 + 2) * sizeof(MoeTile));
      DALLOC(&d_dlut_nt, sizeof(int));
      if (!moe_untiled && !moe_ordered) {  // hybrid MoE tiling (WS end: keep the layout above)
        DALLOC(&d_moetilesA, (size_t)g_cfg.experts * sizeof(MoeTile));
        DALLOC(&d_moentilesA, sizeof(int));
      }
      if (dpf_env()) {  // dense dequant look-ahead slots (WS end, see dpf_*)
        dpf_bytes_ = dpf_plan();
        if (dpf_bytes_) {
          DALLOC(&dpf_slot_[0], dpf_bytes_);
          DALLOC(&dpf_slot_[1], dpf_bytes_);
          int lo = 0, hi = 0;
          CK(hipDeviceGetStreamPriorityRange(&lo, &hi));
          if (getenv("QWENOX_DPF_LOWPRIO"))
            CK(hipStreamCreateWithPriority(&dpf_st_, hipStreamNonBlocking, lo));
          else
            CK(hipStreamCreateWithFlags(&dpf_st_, hipStreamNonBlocking));
          fprintf(stderr, "dpf: stream priority range least=%d greatest=%d%s\n", lo, hi,
                  getenv("QWENOX_DPF_LOWPRIO") ? " (side stream = least)" : "");
          for (int i = 0; i < 2; i++) {
            CK(hipEventCreateWithFlags(&dpf_free_[i], hipEventDisableTiming));
            CK(hipEventCreateWithFlags(&dpf_ready_[i], hipEventDisableTiming));
          }
          fprintf(stderr, "dpf: dense dequant look-ahead on, 2 x %.1f MiB slots\n",
                  dpf_bytes_ / 1048576.0);
        }
      }
      if ((moe_a4_env() && g_hgn_v2 && moe_q4w) || gg_a4_env()) {  // A4 (WS end, opt-in)
        const size_t d = g_cfg.d, km = (size_t)g_cfg.topk * g_cfg.moe_mid;
        DALLOC(&d_a4x, BP * d / 2);
        DALLOC(&d_a4xs, BP * (d / 128) * 4);
        DALLOC(&d_a4h, BP * km / 2);
        DALLOC(&d_a4hs, BP * (km / 128) * 4);
        if (gg_on()) {
          fprintf(stderr,
                  "警告: QWENOX_GG_A4 已废弃（KLD +0.13 换 prefill +4%%，见 "
                  "29_kernels_moe_gg_iu4.inc），禁止用于生产或测速口径\n");
          DALLOC(&d_gg_qt, 3 * 8192);
          gg_init_qtab(moegg::kIq4nl, d_gg_qt + 0 * 8192, 9.0f, g_str);
          gg_init_qtab(moegg::kIq4xs, d_gg_qt + 1 * 8192, 9.0f, g_str);
          gg_init_qtab(moegg::kIq3s, d_gg_qt + 2 * 8192, 9.0f, g_str);
        }
      }
    }
    int nqsa = g_cfg.layers / 4;  // 12
    kv_nl = nqsa + (mtp_kv_wanted() ? 1 : 0);
    {
      const char* pe = getenv("QWENOX_KV_PAGED");
      int mode = pe ? atoi(pe) : 0;
      // A1c: the BF16 cache is paged on the k_qsa_flash (dense / decode) and
      // k_qsa_wmma<*, true> (sparse prefill, with or without BTV) kernels.
      // The union / wmma6 / k_qsa_flash_bf16 sparse kernels are not paged.
      const bool bf16_ok = qsa_dense || (qsa_wmma && !qsa_wmma6);  // union: paged since aggr
      if (mode && !qsa_kv_fp32 && !bf16_ok) {
        fprintf(stderr, "[kvpage] QWENOX_KV_PAGED ignored: with QWENOX_QSA_KV_BF16, "
                        "paging needs QWENOX_QSA_WMMA (with or without _BTV) and no "
                        "QWENOX_QSA_WMMA6 / QWENOX_QSA_UNION\n");
        mode = 0;
      }
      kv_paged = mode != 0;
      kv_mode = mode;
      kv_lpages = (maxctx + KV_PAGE - 1) / KV_PAGE;
      // A3: at least one full-length sequence; pinned checkpoints share the
      // pool (QWENOX_KV_POOL_TOKENS enlarges it); +1 physical guard page.
      kv_pool_pages = kv_pool_pages_for(maxctx);
      kv_layer_rows =
          kv_paged ? (size_t)(kv_pool_pages + 1) * KV_PAGE : (size_t)maxctx;
      if (kv_paged) {
        kv_pool.init(kv_pool_pages, mode == 2 ? 2 : 1);
        kv_seq.init(kv_lpages, kv_pool.guard());
        DALLOC(&d_ptab, (size_t)kv_lpages * 4);
        CK(hipMemcpy(d_ptab, kv_seq.tab.data(), (size_t)kv_lpages * 4,
                     hipMemcpyHostToDevice));
        fprintf(stderr,
                "[kvpage] paged QSA KV on: %d pages x %d tokens (+1 guard), "
                "on-demand alloc=%s, kv=%s\n",
                kv_pool_pages, KV_PAGE,
                mode == 2 ? "scrambled+fifo(self-test)" : "lowest-first",
                qsa_kv_fp32 ? "fp32" : (qsa_wmma_btv ? "bf16+btv" : "bf16"));
        if (kv_pool_pages > kv_lpages)
          fprintf(stderr, "[kvpage] pool enlarged: %d pages = %.2fx one sequence "
                          "(QWENOX_KV_POOL_TOKENS)\n",
                  kv_pool_pages, (double)kv_pool_pages / kv_lpages);
      }
    }
    if (!qsa_dense) {
      index_blocks = (maxctx + 3) / 4;
      ik_layer_blocks = kv_paged ? (size_t)(kv_pool_pages + 1) * KV_PAGE_BLOCKS
                                 : (size_t)index_blocks;
      // A1b: per-query indexer buffers are batch-local (row = abs - base), so
      // they are sized by one prefill chunk (maxbatch_cap), not by maxctx.
      // At 256K ctx / 16K chunk this is ~1.6 GB less. d_ik (per-block keys
      // of the whole history) stays maxctx-sized.
      DALLOC(&d_iproj, (size_t)maxbatch_cap * 640 * 4);
      DALLOC(&d_iq, (size_t)maxbatch_cap * 512 * 4);
      DALLOC(&d_ik, (size_t)kv_nl * ik_layer_blocks * 128 * 4);
      if (kv_paged) DALLOC(&d_ikg, (size_t)index_blocks * 128 * 4);
      // C1 f16 twins（门控与 30_host_util.inc 的 arena 估算同步）
      if (!env_zero("QWENOX_INDEX_F16") && index_score64 && index_sel2p &&
          getenv("QWENOX_INDEX_OLDSEL") == nullptr) {
        DALLOC(&d_iqh, (size_t)maxbatch_cap * 512 * 2);
        DALLOC(&d_ikh, (size_t)kv_nl * ik_layer_blocks * 128 * 2);
        if (kv_paged) DALLOC(&d_ikgh, (size_t)index_blocks * 128 * 2);
      }
      DALLOC(&d_iraw, (size_t)nqsa * 4 * 128 * 4);
      CK(hipMemset(d_iraw, 0, (size_t)nqsa * 4 * 128 * 4));
      DALLOC(&d_inorm, (size_t)nqsa * 256 * 4);
      if (index_f32) DALLOC(&d_iw, (size_t)nqsa * 640 * 2560 * 4);
      DALLOC(&d_iscores, (size_t)std::min(maxctx, INDEX_BATCH) * 4 * index_blocks * 4);
      DALLOC(&d_iselected, (size_t)maxbatch_cap * INDEX_BUDGET * 4);
    }
    if (qsa_kv_fp32) {
      DALLOC(&d_kc, (size_t)kv_nl * kv_layer_rows * 512 * 4);
      DALLOC(&d_vc, (size_t)kv_nl * kv_layer_rows * 512 * 4);
    } else {
      DALLOC(&d_kcb, (size_t)kv_nl * kv_layer_rows * 512 * 2);
      DALLOC(&d_vcb, (size_t)kv_nl * kv_layer_rows * 512 * 2);
      if (qsa_wmma_btv) {
        // paged: kv_layer_rows = (pool + 1) * 256 -> (pool + 1) * 64 blocks,
        // and physical page p owns physical blocks [p*64, p*64+64)
        qsa_vt_blocks = (kv_layer_rows + 3) / 4;
        DALLOC(&d_vcbt, (size_t)nqsa * qsa_vt_blocks * 2 * 256 * 4 * 2);
      }
      if (qsa_union) {
        // 并集工作区按单 chunk 的组数分配（maxbatch_cap/4 组，每组 ≤ 2052 块）
        const size_t ng = (size_t)maxbatch_cap / 4 + 1;
        DALLOC(&d_ublk, ng * QSA_UCAP * 2);
        DALLOC(&d_umask, ng * QSA_UCAP * 2);
        DALLOC(&d_ucount, ng * 4);
      }
    }
    if (kv_paged)  // A2: the guard page must read as finite data (zeros)
      kv_guard_regions([&](void* p, size_t n) { CK(hipMemset(p, 0, n)); });
    DALLOC(&d_qg, 12288 * 4);
    DALLOC(&d_qs, 24 * 256 * 4);
    DALLOC(&d_gs, 24 * 256 * 4);
    DALLOC(&d_kb, 512 * 4);
    DALLOC(&d_vb, 512 * 4);
    DALLOC(&d_attn, 24 * 256 * 4);
    DALLOC(&d_pacc, (size_t)2 * QSA_DEC_NSPLIT * 12 * 256 * 4);
    DALLOC(&d_pml, (size_t)2 * QSA_DEC_NSPLIT * 12 * 2 * 4);
    DALLOC(&d_router, 512 * 4);
    DALLOC(&d_topids, 16 * 4);
    DALLOC(&d_topw, 16 * 4);
    DALLOC(&d_guv, 16 * 1280 * 4);
    DALLOC(&d_hid, 16 * 640 * 4);
    DALLOC(&d_ey, g_cfg.d * 4);
    DALLOC(&d_moeacc, g_cfg.d * 4);
    if (gg_on()) {
      const int k = g_cfg.topk;
      if (k != 10 || !moe_deterministic || moe_host_route || moe_ordered || moe_untiled)
        throw std::runtime_error(
            "QWENOX_GGUF needs top-10 and the default MoE routing (no QWENOX_MOE_ATOMIC / "
            "HOST_ROUTE / ORDERED / UNTILED)");
      DALLOC(&d_gg_x16, (size_t)g_cfg.d * 2);
      DALLOC(&d_gg_hid, (size_t)k * g_cfg.moe_mid * 2);
      DALLOC(&d_gg_pairs, (size_t)k * g_cfg.d * 4);
      DALLOC(&d_gg_pw, k * 4);
      DALLOC(&d_gg_tok, k * 4);
      DALLOC(&d_gg_pid, k * 4);
      DALLOC(&d_gg_nt, 4);
      DALLOC(&d_gg_tiles, k * sizeof(MoeTile));
      DALLOC(&d_gg_uq, moegg::kUqInts * 4);
    }
    DALLOC(&d_sg, 4);
    DALLOC(&d_e, 2560 * 4);
    CK(hipHostMalloc((void**)&ple_e, 2560 * 4, hipHostMallocDefault));
    DALLOC(&d_token, 4);
    DALLOC(&d_pos, 4);
    DALLOC(&d_ringpos, 4);
    DALLOC(&d_rdelta, 4);
    DALLOC(&d_vbase, 4);
    CK(hipMemset(d_pos, 0, 4));
    CK(hipMemset(d_ringpos, 0, 4));
    CK(hipMemset(d_rdelta, 0, 4));
    if (getenv("QWENOX_MROPE_DUMP")) {
      DALLOC(&d_cs1, 32 * sizeof(float2));
      DALLOC(&d_rpos1, 4);
    }
    CK(hipHostMalloc((void**)&p_token, 4, hipHostMallocDefault));
    CK(hipHostMalloc((void**)&p_vbase, 4 * kVbRing, hipHostMallocDefault));
    DALLOC(&d_keys, 10240 * 4);
    DALLOC(&d_v, 2560 * 4);
    DALLOC(&d_kn, 10240 * 4);
    DALLOC(&d_qn, 10240 * 4);
    DALLOC(&d_U, 10240 * 4);
    DALLOC(&d_ring, (size_t)9 * 10240 * 4);
    CK(hipMemset(d_ring, 0, (size_t)9 * 10240 * 4));
    DALLOC(&d_convout, 10240 * 4);
    DALLOC(&d_ple_convw, 10240 * 4 * 4);
    DALLOC(&d_ple_nk, 10240 * 3 * 4);
    DALLOC(&d_logits, (size_t)g_cfg.vocab * 4);
    // Ngram verifies up to 64 drafts plus the current token; MTP needs nine
    // rows at most. Keep logits independent of the much larger prefill chunk.
    DALLOC(&d_logits_b, (size_t)g_cfg.vocab * 4 * 65);
    DALLOC(&d_argmax_arr, 65 * sizeof(int));
    DALLOC(&d_amx_cand, 65 * ARGMAX_B1 * sizeof(unsigned long long));
    DALLOC(&d_sm_ctrl, SMS_ROWS * 8 * sizeof(int));
    DALLOC(&d_sm_hist, (size_t)SMS_ROWS * 4096 * sizeof(unsigned));
    DALLOC(&d_sm_cand_ids, (size_t)SMS_ROWS * SMS_TCAP * sizeof(int));
    DALLOC(&d_sm_cand_vals, (size_t)SMS_ROWS * SMS_TCAP * sizeof(float));
    DALLOC(&d_sm_cand_n, SMS_ROWS * sizeof(int));
    DALLOC(&d_sm_bids, SMS_BIAS_CAP * sizeof(int));
    DALLOC(&d_sm_bvals, SMS_BIAS_CAP * sizeof(float));
    DALLOC(&d_sm_hids, SMS_HIST_CAP * sizeof(int));
    DALLOC(&d_sm_hvals, SMS_HIST_CAP * sizeof(float));
    DALLOC(&d_sm_pids, (size_t)SMS_ROWS * SMS_PCAP * sizeof(int));
    DALLOC(&d_sm_pvals, (size_t)SMS_ROWS * SMS_PCAP * sizeof(float));
    DALLOC(&d_sm_pn, SMS_ROWS * sizeof(int));
    h_sm_ids.resize((size_t)SMS_ROWS * SMS_TCAP);
    h_sm_vals.resize((size_t)SMS_ROWS * SMS_TCAP);
    // Trunk checkpoint belongs to the verifier, independently of MTP weights.
    {
      int ngdn = g_cfg.layers - g_cfg.layers / 4;
      ckpt_S = (size_t)ngdn * 48 * 128 * 128 * 4;
      ckpt_convst = (size_t)ngdn * 3 * 10240 * 4;
      ckpt_ring = (size_t)9 * 10240 * 4;
      ckpt_iraw = qsa_dense ? 0 : (size_t)(g_cfg.layers / 4) * 4 * 128 * 4;
      DALLOC(&d_ckpt, ckpt_S + ckpt_convst + ckpt_ring + ckpt_iraw);
    }
    DALLOC(&d_argmax, 4);
    DALLOC(&d_hnorm_mixer, 10240 * 4);

    // ---- upload small fp32 weights (norms, conv, A_log, dt_bias) ----
    char buf[128];
    std::vector<float> tmp;
    auto upload = [&](const char* name, float* dst, size_t n) {
      const Tensor& t = ckpt.at(name);
      if (t.dtype == 16 || t.dtype == 24) {  // hgn v2 (fp32 indexer projection): decode on GPU
        uint16_t* bf;
        CK(hipMalloc(&bf, n * 2));
        v2_deq_bf16(wview(t), bf, g_str);
        k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(bf, dst, n);
        CK(hipStreamSynchronize(g_str));
        CK(hipFree(bf));
        return;
      }
      tmp.resize(n);
      ckpt.dequant(t, tmp.data());
      CK(hipMemcpy(dst, tmp.data(), n * 4, hipMemcpyHostToDevice));
    };
    int gi = 0;
    for (int l = 0; l < g_cfg.layers; l++) {
      if (is_qsa(l)) {
        if (!qsa_dense) {
          snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.q_layernorm.weight", l);
          upload(buf, d_inorm + (l / 4) * 256, 128);
          snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.k_layernorm.weight", l);
          upload(buf, d_inorm + (l / 4) * 256 + 128, 128);
          if (index_f32) {
            snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
            upload(buf, d_iw + (size_t)(l / 4) * 640 * 2560, 640 * 2560);
          }
        }
        continue;
      }
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.conv1d.weight", l);
      upload(buf, d_convw + (size_t)gi * 10240 * 4, 10240 * 4);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.A_log", l);
      upload(buf, d_Alog + (size_t)gi * 48, 48);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.dt_bias", l);
      upload(buf, d_dtbias + (size_t)gi * 48, 48);
      snprintf(buf, sizeof buf, "layers.%d.linear_attn.norm.weight", l);
      upload(buf, d_gdnnorm + (size_t)gi * 128, 128);
      gi++;
    }
    upload("hyper_connection_mixer.hc_norm.weight", d_hnorm_mixer, 10240);
    upload("layers.1.ple.conv1d.weight", d_ple_convw, 10240 * 4);
    upload("layers.1.ple.norm_key.weight", d_ple_nk, 10240);
    upload("layers.1.ple.norm_query.weight", d_ple_nk + 10240, 10240);
    upload("layers.1.ple.norm_conv.weight", d_ple_nk + 20480, 10240);

    // PLE config tensors
    {
      const Tensor& mt = ckpt.at("layers.1.ple.ple_embedding.layer_multipliers");
      memcpy(ple_mult, mt.data, 24);
      const Tensor& vs = ckpt.at("layers.1.ple.ple_embedding.ngram_heads_vocab_sizes");
      memcpy(ple_vsz, vs.data, 128);
      const Tensor& vo = ckpt.at("layers.1.ple.ple_embedding.ngram_heads_offsets");
      memcpy(ple_voff, vo.data, 128);
      ple_table = ckpt.find("layers.1.ple.ngram_embedding.weight");
      if (!ple_table)
        throw std::runtime_error(
            "PLE n-gram table (layers.1.ple.ngram_embedding.weight) not found: hgn v2 keeps it "
            "in a separate file — set NGRAM_FILE (qwenox: pass that .hgn as an overlay)");
      ple_iq4 = ple_table->dtype == 11;
      ple_bf16 = ple_table->dtype == 0;
      if (ple_iq4) {
        // GGUF IQ4_NL table: 90 B rows, no global scale. The batch-read fd is
        // re-pointed at the shard that holds it (mmapped from offset 0).
        g_ple_rb = 90;
        if (g_gguf)
          for (const auto& mp : g_gguf->mappings())
            if (ple_table->data >= mp.base && ple_table->data < mp.base + mp.len) {
              ple_uring_open(mp.path.c_str());
              g_ple_foff = (uint64_t)(ple_table->data - mp.base);
#ifdef _WIN32
              g_ple_off_ok = g_ple_win.fh != INVALID_HANDLE_VALUE;
#else
              g_ple_off_ok = g_ple_fd >= 0;
#endif
              fprintf(stderr, "ple: IQ4_NL n-gram table from %s%s\n", mp.path.c_str(),
                      g_ple_off_ok ? "" : " (batch reader unavailable: mapped reads)");
            }
      } else {
      if (ple_bf16) {
        // Raw bf16 table (dtype 0): 320 B rows, values used as-is (no trailing
        // global scale — the fp8 layout's extra 4 B do not exist here).
        if (ple_table->ndims != 3 || ple_table->dims[2] != 160 ||
            ple_table->data_size != ple_table->numel() * 2)
          throw std::runtime_error("PLE bf16 table: bad shape/size");
        g_ple_rb = 320;
        ple_tscale = 1.f;
      } else if (ple_table->dtype == 10) {
      memcpy(&ple_tscale, ple_table->data + ple_table->numel(), 4);
      } else {
        throw std::runtime_error("PLE n-gram table: unsupported dtype " +
                                 std::to_string(ple_table->dtype));
      }
      // io_uring / IOCP path: file offset of the PLE table in the file that
      // holds it. open_base opened the base .hgn; a separate n-gram file
      // (NGRAM_FILE, passed as an overlay — hgn v2 ships the table on its
      // own) re-points the reader at that file. Every mapping maps its file
      // from offset 0.
      if (const auto* mp = ckpt.mapping_of(ple_table->data)) {
        if (!ckpt.mappings().empty() && mp != &ckpt.mappings()[0]) {
          ple_uring_open(mp->path.c_str());
          fprintf(stderr, "ple: n-gram table from %s\n", mp->path.c_str());
        }
#ifdef _WIN32
        if (g_ple_win.fh != INVALID_HANDLE_VALUE) {
#else
        if (g_ple_fd >= 0) {
#endif
          g_ple_foff = (uint64_t)(ple_table->data - mp->base);
          g_ple_off_ok = true;
        }
      }
      }
    }

    // ---- MTP draft head: detect weights, allocate own state ----
    if (ckpt.find("mtp.fc_hidden.weight")) {
      // moe_lt is compatible: it only activates at P >= moe_lt_min (4096) in
      // the trunk batch MoE, while MTP draft/ingest/verify run their own
      // fixed kernels at P <= gamma+1 <= 9. gr_bf16 taps are converted
      // bf16->f32 at capture/rollback/ingest (see spec_taps/restore_dR_row).
      bool moe_default = mtp_moe_default();
      if (!moe_default) {
        fprintf(stderr, "mtp: disabled (moe_default=%d)\n", (int)moe_default);
      } else {
        // Validated wiring (mtp-test 0.855): branch-grouped norm + per-branch
        // fc_hidden + broadcast e. Env flips back for debugging only.
        const char* nm = getenv("QWENOX_MTP_NORM");
        mtp_norm_mode = (nm && !strcmp(nm, "full")) ? 0 : 1;
        const char* hg = getenv("QWENOX_MTP_HAGG");
        mtp_hagg_mode = (hg && !strcmp(hg, "sum")) ? 0 : 1;
        DALLOC(&d_mR, rb * 4);
        DALLOC(&d_mRhat, rb * 4);
        DALLOC(&d_mh, g_cfg.d * 4);
        DALLOC(&d_me, g_cfg.d * 4);
        DALLOC(&d_meo, g_cfg.d * 4);
        DALLOC(&d_mt, g_cfg.d * 4);
        DALLOC(&d_mtok, 4);
        DALLOC(&d_mpos, 4);
        DALLOC(&d_mnorm_e, g_cfg.d * 4);
        DALLOC(&d_mnorm_h, rb * 4);
        upload("mtp.pre_fc_norm_embedding.weight", d_mnorm_e, g_cfg.d);
        upload("mtp.pre_fc_norm_hidden.weight", d_mnorm_h, rb);
        // B0: pool layer nqsa (allocated above since kv_nl == nqsa + 1)
        const size_t moff = (size_t)nqsa * kv_layer_rows * 512;
        if (qsa_kv_fp32) {
          d_mkcf = d_kc + moff;
          d_mvcf = d_vc + moff;
        } else {
          d_mkcb = d_kcb + moff;
          d_mvcb = d_vcb + moff;
        }
        DALLOC(&d_mRb, (size_t)maxbatch_cap * 10240 * 4);
        DALLOC(&d_mRhatb, (size_t)maxbatch_cap * 10240 * 4);
        DALLOC(&d_mtap, (size_t)maxbatch_cap * 10240 * 4);
        {
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          DALLOC(&d_mring_ckpt, 512 * 4);
          // Rollback snapshots: per-row GDN S / convst / indexer-proj state
          // captured during a spec verify chunk, indexed by accepted depth.
          // Spec chunks are gamma+1 <= 9 rows (gamma <= 8), not maxbatch;
          // kVSnapRows rows are allocated for the multi-slot rendezvous (D2).
          DALLOC(&d_grow, (size_t)kVSnapRows * ngdn * kGdnRowF * 4);
          DALLOC(&d_csnap, (size_t)kVSnapRows * ngdn * 3 * 10240 * 4);
          DALLOC(&d_iproj_snap, (size_t)(g_cfg.layers / 4) * kVSnapRows * 128 * 4);
        }
        if (!qsa_dense) {
          d_mik = d_ik + (size_t)nqsa * ik_layer_blocks * 128;
          DALLOC(&d_miraw, 4 * 128 * 4);
          CK(hipMemset(d_miraw, 0, 4 * 128 * 4));
          DALLOC(&d_minorm, 256 * 4);
          upload("mtp.layers.0.self_attn.indexer.q_layernorm.weight", d_minorm, 128);
          upload("mtp.layers.0.self_attn.indexer.k_layernorm.weight", d_minorm + 128,
                 128);
          if (index_f32) {
            DALLOC(&d_miw, (size_t)640 * 2560 * 4);
            upload("mtp.layers.0.self_attn.indexer.index_qk_proj.weight", d_miw,
                   (size_t)640 * 2560);
          }
        }
        mtp_avail = true;
        fprintf(stderr,
                "mtp: draft head enabled (norm=%s hagg=%s kv=%s fc_hidden dtype=%d)\n",
                mtp_norm_mode ? "branch" : "full",
                mtp_hagg_mode ? "branch" : "sum",
                qsa_kv_fp32 ? "fp32" : "bf16",
                (int)ckpt.at("mtp.fc_hidden.weight").dtype);
      }
    }
    df2_init();  // no-op unless QWENOX_DFLASH2 is set
    dense_shadow_init();  // no-op unless QWENOX_DENSE_SHADOW=1
    fprintf(stderr, "qwenox: state allocated (maxctx=%d)\n", maxctx);
  }

  ~GpuModel() {
    ple_drop_all();  // no joinable PLE worker may outlive the model
    if (dense_shadow_host_) {
      CK(hipDeviceSynchronize());  // no GEMM may still read the shadow
      CK(hipHostFree(dense_shadow_host_));
      dense_shadow_host_ = nullptr;
      dense_shadow_dev_ = nullptr;
      dense_shadow_off_.clear();
    }
    if (rbh) rocblas_destroy_handle(rbh);
    if (lth) hipblasLtDestroy(lth);
#ifndef _WIN32
    // Windows 下 d_Rhatbf16 来自 devarena bump 分配，不能单独 hipFree
    if (d_Rhatbf16) CK(hipFree(d_Rhatbf16));
#endif
  }

  // GR read: R -> Rhat -> t320 -> G -> x. prefix like
  // "layers.3.attn_hyper_connection" or "hyper_connection_mixer".
  void gr_read(const std::string& prefix, float* x_out) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.hc_norm.weight", prefix.c_str());
    // hc_norm: BF16 10240 — needs fp32; we did NOT upload per-layer norms!
    // per-layer hc_norm weights are uploaded lazily on first use (see ensure_hcnorm)
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_R, hw, d_Rhat, 2560, g_cfg.norm_eps, 4);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemv_silu(wview(ckpt.at(buf)), d_Rhat, d_t320);  // D3: silu*0.25 epilogue
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_t320, d_G);
    // D1: fold k_sigmoid_inplace into k_gr_combine (same expression). d_G is
    // not read after this point. QWENOX_GR_COMBINE_SIG=0 restores the launch.
    static const bool gr_sig = !env_zero("QWENOX_GR_COMBINE_SIG");
    if (gr_sig) {
      k_gr_combine<true><<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_Rhat, x_out,
                                                                g_cfg.d, g_cfg.branches);
    } else {
      k_sigmoid_inplace<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_G, 10240);
      k_gr_combine<false><<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_Rhat, x_out,
                                                                 g_cfg.d, g_cfg.branches);
    }
  }

  void gr_write(const std::string& prefix, const float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_Rhat, d_w4);
    k_gr_write<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_w4, d_R, y, g_cfg.d, g_cfg.branches);
  }

  // lazy cache of per-layer hc_norm + q/k_norm fp32 uploads
  std::unordered_map<std::string, float*> hcnorm_cache_;
  float* ensure_hcnorm(const std::string& prefix) {
    auto it = hcnorm_cache_.find(prefix);
    if (it != hcnorm_cache_.end()) return it->second;
    char buf[128];
    snprintf(buf, sizeof buf, "%s.hc_norm.weight", prefix.c_str());
    float* p;
    CK(hipMalloc(&p, 10240 * 4));
    std::vector<float> tmp(10240);
    ckpt.dequant(ckpt.at(buf), tmp.data());
    CK(hipMemcpy(p, tmp.data(), 10240 * 4, hipMemcpyHostToDevice));
    hcnorm_cache_[prefix] = p;
    return p;
  }
  float* ensure_norm256(const std::string& name) {
    auto it = hcnorm_cache_.find(name);
    if (it != hcnorm_cache_.end()) return it->second;
    float* p;
    CK(hipMalloc(&p, 256 * 4));
    std::vector<float> tmp(256);
    ckpt.dequant(ckpt.at(name), tmp.data());
    CK(hipMemcpy(p, tmp.data(), 256 * 4, hipMemcpyHostToDevice));
    hcnorm_cache_[name] = p;
    return p;
  }

  // Draft lm_head override: keep a private device copy of the BASE file's
  // q4cp lm_head when an overlay replaced it with a wider dtype. Drafting
  // reads it instead (~0.5x bytes per draft step); verify and serial decode
  // still use the production lm_head, so committed outputs are unchanged.
  void setup_draft_lmhead(const Tensor* base_lm) {
    if (env_zero("QWENOX_DRAFT_LM_Q4") || !base_lm || base_lm->dtype != 5) return;
    const Tensor& cur = ckpt.at("lm_head.weight");
    if (cur.data == base_lm->data) return;  // no overlay override
    uint64_t cols = base_lm->dims[base_lm->ndims - 1];
    uint64_t rows = base_lm->numel() / cols;
    if (rows != (uint64_t)g_cfg.vocab || cols != (uint64_t)g_cfg.d) return;
    uint8_t* p = nullptr;
    DALLOC(&p, base_lm->data_size);
    CK(hipMemcpy(p, base_lm->data, base_lm->data_size, hipMemcpyHostToDevice));
    w_dlm.dtype = base_lm->dtype;
    w_dlm.data = p;
    w_dlm.rows = rows;
    w_dlm.cols = cols;
    w_dlm.scale_stride = ((cols / 32 * 2) + 15) & ~15ull;
    dlm_on = true;
    fprintf(stderr,
            "draft lm_head: base q4cp %.0f MiB (overlay dtype %d skipped for drafts)\n",
            base_lm->data_size / 1048576.0, (int)cur.dtype);
  }

  // ---- M-RoPE (vision) ----
  // set_mrope expands the request's image grid_thw triples into per-token
  // 3-row positions, replicating reference get_rope_index (tools/mrope_ref.py):
  // text runs take arange+current_pos on all 3 rows; an image run consumes
  // one grid (t, h, w) -> t*(h/2)*(w/2) placeholder tokens on a 3D grid
  // (t outer, h mid, w inner; each row offset by the run's start), and
  // current_pos advances only max(h, w)//2. rope_delta = max(pos)+1-N is the
  // decode offset (reference modeling_qwen4_exp.py:2121,2237-2241).
  bool set_mrope(const std::vector<int>& ids,
                 const std::vector<std::array<int, 3>>& grids) {
    clear_mrope();
    const int IMG = 248056, VID = 248057, MERGE = 2;
    const int N = (int)ids.size();
    mrope_pos_.assign(N, {0, 0, 0});
    auto type = [&](int id) { return id == IMG ? 1 : id == VID ? 2 : 0; };
    size_t gi = 0;
    int cur = 0;
    long long pmax = -1;
    for (int i = 0; i < N;) {
      int tp = type(ids[i]), j = i + 1;
      while (j < N && type(ids[j]) == tp) j++;
      if (tp == 0) {
        for (int k = i; k < j; k++) {
          int p = cur + (k - i);
          mrope_pos_[k] = {p, p, p};
        }
        pmax = std::max<long long>(pmax, cur + (j - i) - 1);
        cur += j - i;
      } else {
        if (gi >= grids.size()) return false;
        auto g = grids[gi++];
        if (g[0] < 1 || g[1] <= 0 || g[2] <= 0 || g[1] % MERGE || g[2] % MERGE)
          return false;
        int t = g[0], h = g[1] / MERGE, w = g[2] / MERGE;
        if ((long long)t * h * w != j - i) return false;
        int k = i;
        for (int tt = 0; tt < t; tt++)
          for (int hh = 0; hh < h; hh++)
            for (int ww = 0; ww < w; ww++, k++)
              mrope_pos_[k] = {cur + tt, cur + hh, cur + ww};
        pmax = std::max<long long>(pmax, cur + std::max({t - 1, h - 1, w - 1}));
        cur += std::max(g[1], g[2]) / MERGE;
      }
      i = j;
    }
    if (gi != grids.size()) return false;  // unconsumed grids
    rope_delta_ = (int)(pmax + 1 - N);
    mrope_active_ = true;
    mrope_grids_ = grids;  // kvsnap meta: pixel-agnostic placeholders need this
    mrope_gen_++;
    CK(hipMemcpy(d_rdelta, &rope_delta_, 4, hipMemcpyHostToDevice));
    return true;
  }

  void clear_mrope() {
    img_inj.clear();  // vision per-request state dies with the mrope request
    img_emb_rows = 0;
    img_hash_ = 0;
    mrope_grids_.clear();
    if (!mrope_active_ && rope_delta_ == 0) return;
    mrope_active_ = false;
    rope_delta_ = 0;
    mrope_pos_.clear();
    mrope_gen_++;
    CK(hipMemcpy(d_rdelta, &rope_delta_, 4, hipMemcpyHostToDevice));
  }

  // ---- Vision tower wiring (stage 4) ----
  void load_vision(const char* path) {
    vt.lth = lth;
    vt.rbh = rbh;
    vt.d_ltws = d_ltws;
    vt.lt_ws = lt_ws_bytes;
    vt.load(path);
    vision_on = true;
  }

  // Run the ViT over each image's patches and build the injection map.
  // grids/ids were already validated by set_mrope (same run-walking logic
  // here); images are t==1 only at this stage. Returns false on any
  // patches/grid mismatch (stderr carries the position).
  bool vision_embeds(const std::vector<int>& ids,
                     const std::vector<std::array<int, 3>>& grids,
                     const std::vector<std::vector<float>>& patches,
                     int cached = 0) {
    img_inj.clear();
    img_emb_rows = 0;
    const int IMG = 248056, VID = 248057, MERGE = 2;
    const int N = (int)ids.size();
    auto type = [&](int id) { return id == IMG ? 1 : id == VID ? 2 : 0; };
    size_t gi = 0, emb_off = 0;
    for (int i = 0; i < N;) {
      int tp = type(ids[i]), j = i + 1;
      while (j < N && type(ids[j]) == tp) j++;
      if (tp == 0) {
        i = j;
        continue;
      }
      if (tp != 1 || gi >= grids.size() || gi >= patches.size()) {
        fprintf(stderr, "vision: unsupported run at prompt pos %d\n", i);
        return false;
      }
      auto g = grids[gi];
      const std::vector<float>& px = patches[gi];
      gi++;
      int gh = g[1], gw = g[2];
      if (g[0] != 1 || gh <= 0 || gw <= 0 || gh % MERGE || gw % MERGE) {
        fprintf(stderr, "vision: bad grid t=%d h=%d w=%d at prompt pos %d\n",
                g[0], g[1], g[2], i);
        return false;
      }
      int P = gh * gw, M = P / 4;
      if ((int)px.size() != P * 1536 || j - i != M) {
        fprintf(stderr,
                "vision: patches/grid mismatch at prompt pos %d "
                "(patches=%zu floats, grid P=%d, run=%d tokens)\n",
                i, px.size(), P, j - i);
        return false;
      }
      if (j <= cached) {
        // Fully inside the reused live prefix: the KV already carries this
        // image's injected embeds. Shape validation above still applies;
        // skip the forward (and the injection rows, which would all fall
        // below the chunk base and be filtered out anyway).
        fprintf(stderr, "vision: image %zu grid=(%d,%d,%d) cached, forward "
                "skipped\n", gi, g[0], gh, gw);
        i = j;
        continue;
      }
      if (emb_off + (size_t)M > img_emb_cap) {
        // Grow WITHOUT losing the rows already written for earlier images of
        // this request: they are only consumed at prefill (k_img_inject), so
        // the old free-then-malloc dropped images 1..k-1 whenever image k
        // overflowed (first multi-image request per slot past 4096 rows).
        size_t ncap = std::max<size_t>(emb_off + M, 4096);
        uint16_t* nbuf = nullptr;
        CK(hipMalloc(&nbuf, ncap * 2560 * 2));
        if (d_img_emb && emb_off)
          CK(hipMemcpy(nbuf, d_img_emb, emb_off * 2560 * 2,
                       hipMemcpyDeviceToDevice));
        if (d_img_emb) (void)hipFree(d_img_emb);
        d_img_emb = nbuf;
        img_emb_cap = ncap;
      }
      auto t0 = std::chrono::steady_clock::now();
      // Debug hook (QWENOX_VISION_EMBEDS_NPY=<f4 (M,2560) npy>): inject
      // precomputed embeds (host bf16 RNE) instead of running the ViT.
      // Isolates LLM/injection behavior from ViT numerics in A/B probes.
      const char* dbg_emb = getenv("QWENOX_VISION_EMBEDS_NPY");
      if (dbg_emb) {
        VitNpy e = vit_npy_read(dbg_emb);
        if (e.shape.size() != 2 || e.shape[0] != (uint64_t)M ||
            e.shape[1] != 2560) {
          fprintf(stderr, "vision: embeds npy shape mismatch (want (%d,2560))\n", M);
          return false;
        }
        const float* ef = (const float*)e.data.data();
        std::vector<uint16_t> bf((size_t)M * 2560);
        for (size_t k = 0; k < bf.size(); k++) {
          uint32_t b;
          memcpy(&b, ef + k, 4);
          b += 0x7FFFu + ((b >> 16) & 1u);
          bf[k] = (uint16_t)(b >> 16);
        }
        CK(hipMemcpy(d_img_emb + emb_off * 2560, bf.data(),
                     (size_t)M * 2560 * 2, hipMemcpyHostToDevice));
      } else {
      int Mr = vt.forward(px.data(), P, gh, gw);
      if (Mr != M) return false;
      // forward only enqueues; the stitch copy must observe the finished
      // embeds (and the log line should report real forward time).
      CK(hipStreamSynchronize(g_str));
      CK(hipMemcpy(d_img_emb + emb_off * 2560, vt.d_emb_bf16,
                   (size_t)M * 2560 * 2, hipMemcpyDeviceToDevice));
      }
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
      fprintf(stderr, "vision: image %zu grid=(%d,%d,%d) P=%d M=%d "
              "forward %.0f ms\n", gi, g[0], gh, gw, P, M, ms);
      for (int k = 0; k < M; k++)
        img_inj.push_back(make_int2((int)emb_off + k, i + k));
      emb_off += M;
      i = j;
    }
    if (gi != grids.size() || gi != patches.size()) {
      fprintf(stderr, "vision: %zu grids/patches for unconsumed runs\n",
              grids.size() - gi);
      return false;
    }
    img_emb_rows = emb_off;
    img_hash_ = fnv1a_patches(patches);
    return true;
  }


  // ---- M-RoPE debug dump (QWENOX_MROPE_DUMP=<file>; syncs the stream) ----
  // Format: "META rope_delta=<d>", then per table fill one line per row:
  //   ROW <base> <row> <pT> <pH> <pW> <cos0> <sin0> ... <cos31> <sin31>
  // and per decode step:
  //   DEC <tag> <rpos> <rdelta> <cos0> <sin0> ... <cos31> <sin31>
  // cos/sin as C %a hexfloats (exact binary value of the fp32 table entry).
  FILE* mrope_dump() {
    if (mrope_dump_ || !getenv("QWENOX_MROPE_DUMP")) return mrope_dump_;
    mrope_dump_ = fopen(getenv("QWENOX_MROPE_DUMP"), "w");
    if (mrope_dump_) fprintf(mrope_dump_, "# qwenox-mrope-dump v1\n");
    return mrope_dump_;
  }

  void dump_rope_cs(int base, int P) {
    FILE* f = mrope_dump();
    if (!f) return;
    if (!mrope_dump_meta_) {
      fprintf(f, "META rope_delta=%d\n", rope_delta_);
      mrope_dump_meta_ = true;
    }
    std::vector<float2> h((size_t)P * 32);
    CK(hipMemcpy(h.data(), d_ropecs, h.size() * sizeof(float2),
                 hipMemcpyDeviceToHost));
    for (int r = 0; r < P; r++) {
      int pT = base + r, pH = base + r, pW = base + r;
      if (mrope_active_ && base + r < (int)mrope_pos_.size()) {
        pT = mrope_pos_[base + r][0];
        pH = mrope_pos_[base + r][1];
        pW = mrope_pos_[base + r][2];
      } else if (mrope_active_) {
        pT = pH = pW = base + r + rope_delta_;
      }
      fprintf(f, "ROW %d %d %d %d %d", base, r, pT, pH, pW);
      for (int i = 0; i < 32; i++)
        fprintf(f, " %a %a", (double)h[(size_t)r * 32 + i].x,
                (double)h[(size_t)r * 32 + i].y);
      fprintf(f, "\n");
    }
    fflush(f);
  }

  void dump_decode_step(int tag) {
    FILE* f = mrope_dump();
    if (!f || !d_cs1) return;
    if (!mrope_dump_meta_) {
      fprintf(f, "META rope_delta=%d\n", rope_delta_);
      mrope_dump_meta_ = true;
    }
    k_rope_dump<<<1, 32, 0, g_str>>>(d_cs1, d_rpos1, d_pos, d_rdelta,
                                     g_cfg.rope_theta);
    float2 h[32];
    int rpos = -1;
    CK(hipMemcpy(h, d_cs1, sizeof h, hipMemcpyDeviceToHost));
    CK(hipMemcpy(&rpos, d_rpos1, 4, hipMemcpyDeviceToHost));
    fprintf(f, "DEC %d %d %d", tag, rpos, rope_delta_);
    for (int i = 0; i < 32; i++)
      fprintf(f, " %a %a", (double)h[i].x, (double)h[i].y);
    fprintf(f, "\n");
    fflush(f);
  }

  // P3-A: mirror the batch base position to device so batch-path kernels read
  // it from memory (a fixed pointer arg replays correctly under graph capture;
  // a baked scalar would freeze the capture-time base). Same value, same math
  // -> bit-identical. Stream-ordered: every consumer's public entry re-writes
  // it before launching. QWENOX_VBASE=0 falls back to scalar args everywhere.
  static bool vbase_on() { return !env_zero("QWENOX_VBASE"); }
  const int* vbptr() const { return vbase_on() ? d_vbase : nullptr; }
  // Staging is a small pinned ring, not one slot: a pinned async copy reads
  // host memory when the stream reaches it, so rewriting a single slot before
  // the GPU got there would hand an earlier chunk the later chunk's base.
  // Each slot's copy is fenced by an event and waited on before reuse (in
  // practice never blocks). NOTE (stage 2 / graph capture): hipEventSynchronize
  // is illegal while capturing -- a captured round needs its own staging slot
  // written before each replay, with the per-round readback as the fence.
  static constexpr int kVbRing = 4;
  hipEvent_t vb_ev[kVbRing] = {};
  int vb_slot = 0;
  void set_vbase(int base) {
    if (!vbase_on()) return;
    const int s = vb_slot;
    vb_slot = (vb_slot + 1) % kVbRing;
    if (vb_ev[s]) CK(hipEventSynchronize(vb_ev[s]));
    else CK(hipEventCreateWithFlags(&vb_ev[s], hipEventDisableTiming));
    p_vbase[s] = base;
    CK(hipMemcpyAsync(d_vbase, p_vbase + s, 4, hipMemcpyHostToDevice, g_str));
    CK(hipEventRecord(vb_ev[s], g_str));
    vb_str[s] = g_str;
  }
  // Before destroying a temporary stream (the graph-capture stream, whose
  // eager boundary ops between segments call set_vbase): retire the slots
  // last recorded on it. A later hipEventSynchronize on an event whose
  // stream is gone reads freed runtime memory (random access violation).
  hipStream_t vb_str[kVbRing] = {};
  void vb_forget_stream(hipStream_t st) {
    for (int s = 0; s < kVbRing; s++)
      if (vb_ev[s] && vb_str[s] == st) {
        CK(hipEventSynchronize(vb_ev[s]));  // stream still alive: completes
        CK(hipEventDestroy(vb_ev[s]));
        vb_ev[s] = nullptr;  // recreated on next use
        vb_str[s] = nullptr;
      }
  }

  // Refill the rope cos/sin table only when the chunk's (base, P) changes:
  // one compute per chunk, shared by all 12 QSA layers (stream-ordered).
  // M-RoPE: host fill with the request's per-token 3-row positions,
  // slot-interleaved; same fp64 expression as k_rope_cs (pow + libm cos/sin,
  // cast to float). Text requests keep the device-kernel fill untouched.
  void ensure_rope_cs(int base, int P) {
    if (base == ropecs_base_ && P == ropecs_P_ && mrope_gen_ == ropecs_gen_)
      return;
    if (mrope_active_) {
      double tab[32];
      for (int i = 0; i < 32; i++)
        tab[i] = rope_inv_freq(g_cfg.rope, g_cfg.rope_theta, 64, i);
      h_ropecs_.resize((size_t)P * 32);
      const int mropeN = (int)mrope_pos_.size();
      for (int r = 0; r < P; r++)
        for (int i = 0; i < 32; i++) {
          // Spec verify runs with base=q+1 >= N: mrope_pos_ only holds the
          // prompt's N rows. Verified positions keep advancing from the last
          // mrope row by +1 (same linear sequence mrope_step_ produces), so
          // fall back to rope_delta_ + absolute position out of range.
          int p = (base + r < mropeN)
                      ? mrope_pos_[base + r][mrope_slot_row(i)]
                      : (base + r + rope_delta_);
          double ang = (double)p * tab[i];
          h_ropecs_[(size_t)r * 32 + i] =
              make_float2((float)cos(ang), (float)sin(ang));
        }
      CK(hipMemcpy(d_ropecs, h_ropecs_.data(),
                   (size_t)P * 32 * sizeof(float2), hipMemcpyHostToDevice));
    } else {
      set_vbase(base);
      k_rope_cs<<<(P * 32 + 255) / 256, 256, 0, g_str>>>(d_ropecs, base, P,
                                                         vbptr());
    }
    ropecs_base_ = base;
    ropecs_P_ = P;
    ropecs_gen_ = mrope_gen_;
    dump_rope_cs(base, P);
  }

  void gdn(int l, int gi, const float* x, float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_qkv.weight", l);
    const WView qkvw = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    const WView zw = wview(ckpt.at(buf));
    // D4: qkv and z share x; merge when the dtypes match (mixed -> fallback).
    const bool gdn_pair = gemv2(qkvw, zw, x, d_qkv, d_z);
    if (!gdn_pair) gemv(qkvw, x, d_qkv);
    // conv + silu (state per layer)
    float* convst = d_convst + (size_t)gi * 3 * 10240;
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    k_gdn_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_qkv, cw, convst, d_qkv, 10240);
    // note: in-place (cur==out) is safe: read cur[ch] before write out[ch]
    // split: q [0,2048) k [2048,4096) v [4096,10240)
    float* qp = d_qkv;
    float* kp = d_qkv + 2048;
    float* vp = d_qkv + 4096;
    k_l2norm_qk<<<16, 128, 0, g_str>>>(qp, kp, 128, 1.f / sqrtf(128.f));
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_a.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_g48 /*tmp*/);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_b.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_beta48 /*tmp: raw b*/);
    k_gdn_gates<<<1, 64, 0, g_str>>>(d_g48, d_beta48, d_Alog + (size_t)gi * 48,
                           d_dtbias + (size_t)gi * 48, d_g48, d_beta48, 48, 48);
    k_gdn_step<<<48, 512, 0, g_str>>>(qp, kp, vp, d_g48, d_beta48,
                            d_S + (size_t)gi * 48 * 128 * 128, vp /*out in place*/,
                            128, 128, 48, 16);
    // out written over vp is safe: k_gdn_step reads vh before writing out
    if (!gdn_pair) gemv(zw, x, d_z);
    k_gdn_gatednorm<<<dim3(48, 1), 128, 0, g_str>>>(vp, d_z, d_gdnnorm + (size_t)gi * 128,
                                                    vp, 128, g_cfg.norm_eps, 10240,
                                                    6144);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
    gemv(wview(ckpt.at(buf)), vp, y);
  }

  // One decode step of QSA attention over the KV cache: flash-decoding split
  // over QSA_DEC_NSPLIT position slices plus a merge, for every token. Dense
  // tokens (< 2051) take the same path (ntok = token+1, identity source map);
  // the old 3-pass k_qsa_step dense kernel cost ~0.7ms->5.4ms/layer growing
  // linearly with pos (rocprofv3: 55.8% of all decode kernel time), while the
  // flash split stays flat. QWENOX_QSA_DENSE takes the same split kernel with no
  // selection list (all visible positions). k_qsa_step used to be its fallback,
  // but its pass 3 recomputes every q.k per output dim: 1170 ms/token at 8K.
  template <typename KVT>
  void qsa_attn(const KVT* kc, const KVT* vc, const int* ptab = nullptr) {
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, qsa_dense ? nullptr : d_iselected, 0,
        d_pos, d_pacc, d_pml, 0, ptab);
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_pos);
  }

  void qsa(int l, int qi, const float* x, float* y) {
    char buf[128];
    ensure_rope_tab(g_cfg.rope_theta, g_cfg.rotary_dim);
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      float* keys = d_ik + (size_t)qi * ik_layer_blocks * 128;
      uint16_t* keys16 =
          d_ikh ? d_ikh + (size_t)qi * ik_layer_blocks * 128 : nullptr;
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256, d_iq,
                                              1, g_cfg.norm_eps, g_cfg.rope_theta, d_pos,
                                              0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_inorm + qi * 256 + 128,
                                           d_iraw + qi * 512, keys, d_pos,
                                           g_cfg.norm_eps, g_cfg.rope_theta,
                                           d_rdelta, kvpt(), keys16);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(d_iq, keys, d_iscores,
                                                               index_blocks, d_pos,
                                                               kvpt());
      index_select(d_iscores, index_blocks, d_iselected, 0, 1, d_pos);
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_qg);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    const WView kw = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    const WView vw = wview(ckpt.at(buf));
    if (!gemv2(kw, vw, x, d_kb, d_vb)) {  // D4: k/v share the dtype in GGUF
      gemv(kw, x, d_kb);
      gemv(vw, x, d_vb);
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_norm.weight", l);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_norm.weight", l);
    float* knw = ensure_norm256(buf);
    // per-head: q[h] = zcnorm(qg[h*512 .. +256]); gate[h] = qg[h*512+256 ..]
    k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(d_qg, qnw, d_qs, d_gs, 256,
                                                 g_cfg.norm_eps);
    k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(d_kb, knw, d_kb, 256, g_cfg.norm_eps, 1);
    k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(d_qs, 24, 256, 64, d_pos,
                                                     g_cfg.rope_theta, d_rdelta);
    k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(d_kb, 2, 256, 64, d_pos,
                                                    g_cfg.rope_theta, d_rdelta);
    // append to cache at *d_pos
    if (qsa_kv_fp32) {
      float* kc = d_kc + (size_t)qi * kv_layer_rows * 512;
      float* vc = d_vc + (size_t)qi * kv_layer_rows * 512;
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos, kvpt());
      qsa_attn(kc, vc, kvpt());
    } else {
      uint16_t* kc = d_kcb + (size_t)qi * kv_layer_rows * 512;
      uint16_t* vc = d_vcb + (size_t)qi * kv_layer_rows * 512;
      // BTV: keep the transposed prefill V cache complete for decoded
      // positions (read later by sparse prefill/verify via k_qsa_wmma<true>)
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, kc, vc, d_pos, vct_layer(qi),
                                            kvpt());
      qsa_attn(kc, vc, kvpt());
    }
    k_sigmoid_gate<<<(24 * 256 + 255) / 256, 256, 0, g_str>>>(d_attn, d_gs, d_attn, 24 * 256);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemv(wview(ckpt.at(buf)), d_attn, y);
  }


  // ---- 专家权重直读(零拷贝,gfx1151 统一内存;唯一形态,全量 arena 已移除)
  // load_arena 已把 .mlp.experts. 张量登记为 host mmap 指针;kernel 直读
  // 页缓存实测 224 GB/s(tools/tgv2_probe,与 GTT 同一片 DRAM),gather
  // kernel 按专家 id 索引全量张量,读到的字节与 arena 逐位一致 → 数值
  // 不变、kernel 零改动。decode 无 D2H routing 同步、无 H2D staging,
  // 指针全程静态,图捕获照常。
  // (2026-09-12 staging 方案曾记"裸指针 Memory Fault",与今日实测矛盾;
  //  当时 fault 应另有原因,本路径以 oracle 验证为准。)
  // Single-token routed down projection (trunk decode + MTP draft) into
  // d_moeacc. The atomic kernel sums the 10 expert partials in wave-schedule
  // order, so a decode step was not bit-reproducible: timing changes (another
  // slot's work in between, a debug sync) flipped near-tie tokens thousands of
  // steps later. The top-k kernel keeps the partials in registers and sums in
  // expert order, overwriting every row (no memset). QWENOX_MOE_DOWN_ATOMIC /
  // QWENOX_MOE_ATOMIC restore the old path.
  bool moe_down_direct(int k, const WView& dn) const {
    return moe_down_topk && k == 10 && dn.cols == 640;
  }
  // Routed-expert tensor -> operands of the q4cp-style grouped kernels. q4cp
  // (dtype 5): 64 B codebook header, one f16 scale per 32 cols (GSH 0). hgn v2
  // i4r (dtype 23): codes at 0, linear -8..7 book, one scale per 128 (GSH 2).
  struct XCodes {
    const uint8_t* codes;
    const uint8_t* scales;
    const float* cb;
  };
  static XCodes xcodes(const WView& w) {
    if (w.dtype == 23) return {w.data, w.data + w.rows * w.cols / 2, g_i4r_cb};
    return {w.data + 64, w.data + 64 + w.rows * w.cols / 2, (const float*)w.data};
  }
  template <int GSH>
  void moe_down1_t(int k, const WView& dn, const int* topids) {
    const XCodes c = xcodes(dn);
    if (moe_down_direct(k, dn)) {
      const uint64_t row_pairs = (g_cfg.d + 1) / 2, warps_per_block = 128 / 32;
      k_q4cp_gemv_gd_topk_h16<10, GSH><<<(unsigned)((row_pairs + warps_per_block - 1) / warps_per_block),
                                    128, 0, g_str>>>(
          c.codes, c.scales, c.cb, d_hid,
          d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
          (uint64_t)g_cfg.moe_mid, 1, 16 /*id_stride*/, (uint64_t)g_cfg.d);
      return;
    }
    // v3 kernel: 2 rows per warp -> k * (d/2) warp-pairs
    uint64_t pairs = (uint64_t)k * ((g_cfg.d + 1) / 2);
    k_q4cp_gemv_gd<GSH><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
        c.codes, c.scales, c.cb, d_hid,
        d_moeacc, topids, d_topw, (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
        (uint64_t)g_cfg.moe_mid, k, 16 /*id_stride*/, 0 /*acc_stride*/, k);
  }
  void moe_down1(int k, const WView& dn, const int* topids) {
    if (dn.dtype == 23)
      moe_down1_t<2>(k, dn, topids);
    else
      moe_down1_t<0>(k, dn, topids);
  }
  // hgn v2 rotation helpers (k_v2_rot): P rows of n values, f32 in / f32 or
  // f16 out; pre = input plane (rot_in), post = output plane (rot_out).
  void v2_rot(const float* x, uint64_t xs, void* y, uint64_t ys, const float* pre,
              const float* post, int n, int P, bool outh = false) {
    const unsigned waves = (unsigned)P * (n / 128);
    if (outh)
      k_v2_rot<float, true><<<(waves + 7) / 8, 256, 0, g_str>>>(x, xs, y, ys, pre, post, n, P);
    else
      k_v2_rot<float, false><<<(waves + 7) / 8, 256, 0, g_str>>>(x, xs, y, ys, pre, post, n, P);
  }
  // i4r routed experts for one token (decode / MTP-free trunk): x -> d_moeacc.
  //   rot_in(x, su) -> gate_up (rotated) -> k_i4r_mid (rot_out, silu*u,
  //   rot_in(down.suh)) -> down (rotated, weighted sum over experts) ->
  //   rot_out(acc, down.svh). svh is shared by all experts of a layer, so the
  //   output rotation runs once on the weighted sum.
  void moe_v2_1(int k, const WView& gu, const WView& dn, const float* x, const int* topids) {
    const int d = g_cfg.d, mid = g_cfg.moe_mid;
    if (!moe_down_direct(k, dn)) CK(hipMemsetAsync(d_moeacc, 0, d * 4, g_str));
    v2_rot(x, 0, d_ey, 0, gu.suh, nullptr, d, 1);  // d_ey: free until the shared expert
    const XCodes c = xcodes(gu);
    const uint64_t pairs = (uint64_t)k * mid;
    k_q4cp_gemv_gg<2><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
        c.codes, c.scales, c.cb, d_ey, d_guv, topids, (uint64_t)2 * mid, gu.cols,
        gu.scale_stride, k, 0 /*x_stride*/, 16 /*id_stride*/, k);
    k_i4r_mid<float, false><<<k, 2 * mid / 128 * 32, 0, g_str>>>(d_guv, 2 * mid, d_hid, mid,
                                                                 gu.svh, dn.suh, mid);
    moe_down1(k, dn, topids);
    v2_rot(d_moeacc, 0, d_moeacc, 0, nullptr, dn.svh, d, 1);
  }
  // GGUF routed experts g (trunk layer g_ggx[l] or the MTP layer g_ggx_mtp;
  // needs gg_on() for the scratch buffers): x f32 [P][d] with top-k
  // rows (16 wide) -> acc f32 [P][d], every element overwritten (no memset).
  // P == 1 (decode graph): sort-free per-pair tiles on private scratch.
  // P > 1: stable GPU bucketing + 64-slot tiles on the prefill workspaces
  // (F16 x in d_xbf16b, which is never cached; hid F16 in d_hidb; unweighted
  // pairs in d_guvb, dead again once the weighted reduce has run).
  // ey/sg (optional, P > 48 only): the shared-expert output/gate logit are
  // added in the reduce (k_moe_reduce_pw_sg) instead of by k_axpy_sg after the
  // call. Returns true when they were consumed; false means the caller must
  // still add them (small-P GEMV path).
  bool moe_gguf(const GgExperts& g, const float* x, const int* topids, const float* topw,
                float* acc, int P, const float* ey = nullptr, const float* sg = nullptr) {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    // Small P (decode, spec verify): raw-block GEMVs, FP32 hid, weighted sum in
    // expert-rank order written straight to acc (26_kernels_moe_gguf.inc).
    // QWENOX_GG_GEMV=<max P> (default 48 = moegg::kUqMaxP, 0 = always WMMA).
    // Every P <= 48 (decode, spec verify, batched verify / drafts, small
    // prefill tails) takes the GEMV family, whose per-(token, slot) arithmetic
    // does not depend on P -> concurrent == single-run bit for bit (with the
    // row-invariant dense q8g32 WMMA). P > 48 is prefill: WMMA tiles.
    static const int gemv_max = [] {
      const char* s = getenv("QWENOX_GG_GEMV");
      return std::min(s && *s ? atoi(s) : moegg::kUqMaxP, moegg::kUqMaxP);
    }();
    // 2 <= P <= 48: experts shared between the P tokens are read once
    // (bit-identical to the plain path); unweighted pairs go to d_guvb.
    // QWENOX_GG_UQ=0 disables (then P > 8 re-reads per slot: slow).
    static const bool uq_on = [] {
      const char* s = getenv("QWENOX_GG_UQ");
      return !(s && *s == '0');
    }();
    const bool uq = uq_on && P >= 2;
    if (P <= gemv_max && P <= maxbatch_cap &&
        moe_gg_gemv(g.tgu, g.tdn, g.gate, g.up, g.down, x, topids, topw,
                    P == 1 ? d_hid : d_hidb, acc, P, d, mid, 16, g_str,
                    uq ? d_gg_uq : nullptr, uq ? d_guvb : nullptr))
      return false;  // GEMV path: ey/sg not consumed
    int *tokidx, *pairids, *ntiles;
    float *pw, *pairs;
    MoeTile* tiles;
    __half *x16, *hid;
    int max_tiles;
    bool mt_tiles = false;
    if (P == 1) {
      k_moe_gg_route1<<<1, 64, 0, g_str>>>(topids, topw, 1, k, 16, d_gg_tok, d_gg_pw,
                                           d_gg_pid, d_gg_tiles, d_gg_nt);
      tokidx = d_gg_tok, pairids = d_gg_pid, ntiles = d_gg_nt, pw = d_gg_pw;
      tiles = d_gg_tiles, x16 = d_gg_x16, hid = d_gg_hid, pairs = d_gg_pairs;
      max_tiles = k;
    } else {
      if (P > maxbatch_cap) throw std::runtime_error("moe_gguf: P exceeds the batch buffers");
      moe_routing.run(topids, topw, d_tokidx, d_pweight, d_eoff, d_pairids, P, k, E);
      // mt (§11.5 contract, hgn v2 / q4cp parity): merge the 64-slot tiles of
      // one expert into 128-slot super-tiles so each weight K stage is decoded
      // once for both halves. Only when both expert GEMMs are LUT (k_moe_gg
      // is 64-slot) and the average slots per expert justify it; per-(slot,
      // row) K order is unchanged -> bit-identical. QWENOX_GG_MT=0 reverts.
      static const bool gg_mt_env = [] {
        const char* s = getenv("QWENOX_GG_MT");
        return !(s && *s == '0');
      }();
      const bool mt = gg_mt_env && P > 1 && ggx_is_lut(g.tgu) && ggx_is_lut(g.tdn) &&
                      (int64_t)P * k >= 80 * E;
      k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, mt ? 128 : 64);
      tokidx = d_tokidx, pairids = d_pairids, ntiles = d_moentiles, pw = d_pweight;
      tiles = d_moetiles, x16 = (__half*)d_xbf16b, hid = (__half*)d_hidb, pairs = d_guvb;
      max_tiles = mt ? (P * k + 127) / 128 + E : (P * k + 63) / 64 + E;
      if (mt) {
        static bool logged = false;
        if (!logged) {
          logged = true;
          fprintf(stderr, "gguf moe prefill: LUT mt BN=128 (QWENOX_GG_MT=0 disables)\n");
        }
      }
      // The deprecated W4A4 path (QWENOX_GG_A4) needs 64-slot tiles: when mt
      // is on it is skipped (a4_*_ok below); QWENOX_GG_MT=0 brings it back.
      mt_tiles = mt;
    }
    const size_t n4 = (size_t)P * d / 4;
    k_f32_to_f16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(x, x16, n4);
    // QWENOX_GG_A4 (deprecated, forbidden for production/benchmarks, default
    // off): linearizes each 128-K weight group in-kernel (per-32 GGUF scales
    // -> one per-128 q map) and runs iu4 WMMA. Q8_0 down and MTP experts keep
    // the F16 path. Verdict/evidence: see 29_kernels_moe_gg_iu4.inc header.
    const int a4m = gg_a4_env();
    auto qt_for = [&](int t) {
      if (t == moegg::kIq4nl) return d_gg_qt + 0 * 8192;
      if (t == moegg::kIq4xs) return d_gg_qt + 1 * 8192;
      return d_gg_qt + 2 * 8192;
    };
    bool a4_up_ok = !mt_tiles && (a4m & 1) && d_a4x &&
                    (g.tgu == moegg::kIq3s || g.tgu == moegg::kIq4xs || g.tgu == moegg::kIq4nl);
    bool a4_dn_ok = !mt_tiles && (a4m & 2) && d_a4h &&
                    (g.tdn == moegg::kIq4nl || g.tdn == moegg::kIq4xs);
    if (a4_up_ok) {
      k_quant_a4<<<(unsigned)(((size_t)P * (d / 128) + 7) / 8), 256, 0, g_str>>>(x16, P, d,
                                                                                d_a4x, d_a4xs);
      a4_up_ok = moe_gg_iu4_up(g.tgu, g.gate, g.up, d_a4x, d_a4xs, tokidx, tiles, ntiles,
                               max_tiles, hid, mid, d, qt_for(g.tgu), 9.0f, g_str);
    }
    if (!a4_up_ok &&
        !moe_ggx_up(g.tgu, g.gate, g.up, x16, tokidx, tiles, ntiles, max_tiles, hid, mid, d,
                    g_str, mt_tiles))
      throw std::runtime_error("moe_gguf: unsupported expert type");
    if (a4_dn_ok) {
      k_quant_a4<<<(unsigned)(((size_t)P * k * (mid / 128) + 7) / 8), 256, 0, g_str>>>(
          hid, P * k, mid, d_a4h, d_a4hs);
      a4_dn_ok = moe_gg_iu4_down(g.tdn, g.down, d_a4h, d_a4hs, tiles, ntiles, max_tiles,
                                 (__half*)pairs, d, mid, qt_for(g.tdn), 9.0f, g_str);
    }
    // P2: GGUF LUT down writes f16 pairs (half the reduce traffic; reduce
    // widens exactly and accumulates in f32). Default on (Swift IQ3 KLD
    // .173408 vs f32 .173862); QWENOX_GG_PAIRS_F16=0 keeps f32 pairs.
    static const bool gg_pairs_f16_env = !env_zero("QWENOX_GG_PAIRS_F16");
    const bool pairs_f16 = gg_pairs_f16_env && !a4_dn_ok && ggx_is_lut(g.tdn);
    if (!a4_dn_ok &&
        !moe_ggx_down(g.tdn, g.down, hid, tiles, ntiles, max_tiles, pairs, d, mid, g_str,
                      mt_tiles, pairs_f16))
      throw std::runtime_error("moe_gguf: unsupported expert type");
    if (a4_up_ok || a4_dn_ok) {
      static bool logged = false;
      if (!logged) {
        logged = true;
        fprintf(stderr,
                "gguf moe prefill: W4A4 iu4 DEPRECATED (up=%d down=%d; per-32 GGUF scales -> per-128 qtab)\n",
                (int)a4_up_ok, (int)a4_dn_ok);
      }
    }
    if (a4_dn_ok || pairs_f16)
      if (ey)
        k_moe_reduce_pw_sg<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
            (const __half*)pairs, pairids, pw, ey, sg, acc, P, d);
      else
        k_moe_reduce_pw<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
            (const __half*)pairs, pairids, pw, acc, P, d);
    else if (ey)
      k_moe_reduce_pw_sg<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
          pairs, pairids, pw, ey, sg, acc, P, d);
    else
      k_moe_reduce_pw<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(pairs, pairids, pw, acc,
                                                                       P, d);
    return ey != nullptr;
  }

  // hgn q4cp routed experts through the LUT-decode WMMA kernels (QWENOX_MOE_Q4W):
  // same prefill workspaces / dataflow as moe_gguf's P > 1 branch; acc f32
  // [P][d] fully overwritten. Returns false (nothing launched) if unsupported.
  // Whether moe_q4w_run(gu, dn, ..., P) will take the LUT path (no launches).
  bool moe_q4w_ok(const WView& gu, const WView& dn, int P) const {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    if (!moe_q4w || k != 10 || !moe_deterministic || moe_host_route || moe_ordered ||
        moe_untiled || P > maxbatch_cap || gu.dtype != 5 || dn.dtype != 5 ||
        gu.cols != (uint64_t)d || gu.rows != (uint64_t)E * 2 * mid || dn.cols != (uint64_t)mid ||
        dn.rows != (uint64_t)E * d)
      return false;
    const moelut::LutW pgu = lut_q4cp_view(gu.data, gu.rows, gu.cols, gu.scale_stride, 2 * mid, mid);
    const moelut::LutW pdn = lut_q4cp_view(dn.data, dn.rows, dn.cols, dn.scale_stride, d, 0);
    return !(mid % 64 || d % 128 || !moelut::lut_ok(moelut::kQ4CP, pgu, true, d) ||
             !moelut::lut_ok(moelut::kQ4CP, pdn, false, mid));
  }
  // ey/sg (optional): shared-expert output + gate logit, added in the reduce
  // (k_moe_reduce_pw_sg) instead of by k_axpy_sg afterwards.
  // w4hc/Rhc (optional, requires ey): the reduce fuses the HC write-back
  // (k_moe_reduce_pw_sg_hc) and acc is left untouched.
  bool moe_q4w_run(const WView& gu, const WView& dn, const float* x, const int* topids,
                   const float* topw, float* acc, int P, const float* ey = nullptr,
                   const float* sg = nullptr, const float* w4hc = nullptr,
                   uint16_t* Rhc = nullptr) {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    if (!moe_q4w_ok(gu, dn, P)) return false;
    const moelut::LutW pgu = lut_q4cp_view(gu.data, gu.rows, gu.cols, gu.scale_stride, 2 * mid, mid);
    const moelut::LutW pdn = lut_q4cp_view(dn.data, dn.rows, dn.cols, dn.scale_stride, d, 0);
    static bool logged = false;
    if (!logged) {
      logged = true;
      fprintf(stderr, "moe prefill: q4cp LUT WMMA (QWENOX_MOE_Q4W=0 disables)\n");
    }
    KP("moe:route");
    moe_routing.run(topids, topw, d_tokidx, d_pweight, d_eoff, d_pairids, P, k, E);
    // §11.5 mt（q4cp 站）：同专家双 64-tile 合并 BN=128，每个 K stage 的权重
    // 解码一次供两个 tile 半用，逐 bit 不变；默认开，QWENOX_MOE_MT=0 回退。
    // 门槛同 v2：平均每专家 slot >= 80（P=2048 档 v2 实测 −5.2%，小 P 走原
    // 路径）。w4b 与 v2 不同：moe 在关键路径上（segsum==wall，见 §11.5）。
    static const bool moe_mt_env = !env_zero("QWENOX_MOE_MT");
    const bool mt = moe_mt_env && (int64_t)P * k >= 80 * E;
    k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, mt ? 128 : 64);
    const int max_tiles = mt ? (P * k + 127) / 128 + E : (P * k + 63) / 64 + E;
    __half* x16 = (__half*)d_xbf16b;
    __half* hid = (__half*)d_hidb;
    const size_t n4 = (size_t)P * d / 4;
    k_f32_to_f16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(x, x16, n4);
    KP("moe:up");
    if (mt)
      moe_lut_up_mt_q4cp(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, mid, d,
                         g_str);
    else
      moe_lut_up(moelut::kQ4CP, pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, mid, d,
                 g_str);
    KP("moe:down");
    // B1 w4b f16 pairs（与 v2 QWENOX_V2_PAIRS_F16 同款）：down 写 f16、reduce 读
    // f16（__half2float 精确拓宽，fp32 累加不变），reduce 读流量减半。默认开
    // （KLD 过线后翻转，=0 回退）。w4b_pairs_f16_env 与分配门控
    // guvb_f16_env 语义同步。
    static const bool pairs_f16 = w4b_pairs_f16_env();
    if (mt)
      moe_lut_down_mt_q4cp(pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid, g_str,
                           pairs_f16);
    else
      moe_lut_down(moelut::kQ4CP, pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid,
                   g_str, pairs_f16);
    KP("moe:reduce");
    if (ey)
      if (Rhc)
        if (pairs_f16)
          k_moe_reduce_pw_sg_hc<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
              (const __half*)d_guvb, d_pairids, d_pweight, ey, sg, w4hc, Rhc, P, d);
        else
          k_moe_reduce_pw_sg_hc<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, d_pweight, ey, sg, w4hc, Rhc, P, d);
      else if (pairs_f16)
        k_moe_reduce_pw_sg<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
            (const __half*)d_guvb, d_pairids, d_pweight, ey, sg, acc, P, d);
      else
        k_moe_reduce_pw_sg<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
            d_guvb, d_pairids, d_pweight, ey, sg, acc, P, d);
    else if (pairs_f16)
      k_moe_reduce_pw<10, __half><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(
          (const __half*)d_guvb, d_pairids, d_pweight, acc, P, d);
    else
      k_moe_reduce_pw<10><<<dim3((d + 255) / 256, P), 256, 0, g_str>>>(d_guvb, d_pairids,
                                                                       d_pweight, acc, P, d);
    return true;
  }

  void moe(int l, const float* x, float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.mlp.gate.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_router);
    int k = g_cfg.topk;
    k_router_topk<<<1, 512, 0, g_str>>>(d_router, d_topids, d_topw, g_cfg.experts, k,
                                        vz_tap(l, false));
    if (l == g_cfg.layers - 1) vz_publish(1, false);
    if (dbg) {
      int hi[16];
      float hw[16];
      CK(hipMemcpy(hi, d_topids, k * 4, hipMemcpyDeviceToHost));
      CK(hipMemcpy(hw, d_topw, k * 4, hipMemcpyDeviceToHost));
      fprintf(stderr, "gpu L%02d moe topk:", l);
      for (int i = 0; i < k; i++) fprintf(stderr, " %d:%.5f", hi[i], (double)hw[i]);
      fprintf(stderr, "\n");
    }

    if (gg_on()) {
      moe_gguf(g_ggx.at(l), x, d_topids, d_topw, d_moeacc, 1);
    } else {
      snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
      WView gu = wview(ckpt.at(buf));
      snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
      WView dn = wview(ckpt.at(buf));
      if (gu.dtype == 23) {
        moe_v2_1(k, gu, dn, x, d_topids);
      } else {
      if (!moe_down_direct(k, dn)) CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
      const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
      // fused experts: one grouped launch each for gate_up / silu / down+accum
      {
        // v3 kernel: 2 rows per warp -> k * moe_mid warp-pairs
        uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
        k_q4cp_gemv_gg<0><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
            gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data, x,
            d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride, k,
            0 /*x_stride*/, 16 /*id_stride*/, k);
      }
      k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(d_guv, d_hid,
                                                                       g_cfg.moe_mid, k);
      moe_down1(k, dn, topids);
      }
    }
    // shared expert
    {
      char ub[128];
      snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
      snprintf(ub, sizeof ub, "layers.%d.mlp.shared_expert.up_proj.weight", l);
      se_gate_up(buf, ub, x, d_guv, d_guv + 640, 1, true);
    }
    k_silu_mul<<<(640 + 255) / 256, 256, 0, g_str>>>(d_guv, d_guv + 640, d_hid, 640);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.down_proj.weight", l);
    gemv(wview(ckpt.at(buf)), d_hid, d_ey);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert_gate.weight", l);
    gemv(wview(ckpt.at(buf)), x, d_sg);
    // D2: write y directly instead of acc += ... followed by a 40 KiB D2D.
    static const bool axpy_out = !env_zero("QWENOX_AXPY_SG_OUT");
    if (axpy_out) {
      k_axpy_sg_out<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg, y,
                                                           g_cfg.d, g_cfg.d);
    } else {
      k_axpy_sg<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg, g_cfg.d,
                                                       g_cfg.d);
      CK(hipMemcpyAsync(y, d_moeacc, g_cfg.d * 4, hipMemcpyDeviceToDevice, g_str));
    }
  }

  // PLE host phase: hash + read 16 rows into pinned staging (disk latency — overlapped).
  float* ple_e = nullptr;  // pinned
  // one raw table row (g_ple_rb bytes) -> 160 f32
  void ple_deq(const uint8_t* p, float* dst) const {
    if (ple_iq4) {
      gguf::dequant_row(gguf::IQ4_NL, p, dst, 160);
      return;
    }
    if (ple_bf16) {
      const uint16_t* p16 = (const uint16_t*)p;
      for (int i = 0; i < 160; i++) dst[i] = hgn::bf16_to_f32(p16[i]);
      return;
    }
    for (int i = 0; i < 160; i++) dst[i] = hgn::fp8e4m3_to_f32(p[i]) * ple_tscale;
  }
  // reads the 16 PLE rows for chist position t into dst (pure read of chist;
  // safe to call concurrently for distinct t/dst).
  void ple_rows(int t, float* dst) {
    auto cat = [&](int back) -> int64_t {
      int j = t - back;
      return j < 0 ? pad_id : chist[j];
    };
    const uint8_t* addrs[16];
    // pass 1: compute rows, kick async readahead on all 16 pages (concurrent I/O)
    for (int h = 0; h < 16; h++) {
      int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
      if (h >= 8) mix ^= cat(2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * g_ple_rb;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      uint8_t stg[16 * 160];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, stg, (uint64_t)(uintptr_t)ple_table->data, 16);
      for (int h = 0; h < 16; h++) ple_deq(stg + h * g_ple_rb, dst + h * 160);
      return;
    }
#endif
    // pass 2: read (pages already in flight)
    for (int h = 0; h < 16; h++) ple_deq(addrs[h], dst + h * 160);
  }
  // raw variant: staging keeps the g_ple_rb row bytes per head (fp8 codes,
  // IQ4_NL blocks or bf16 values); dequant happens on GPU (k_ple_fp8_dequant /
  // k_ple_iq4nl_dequant / k_ple_bf16_dequant). ~4x less host work per token.
  void ple_rows_fp8(int t, uint8_t* dst) {
    auto cat = [&](int back) -> int64_t {
      int j = t - back;
      return j < 0 ? pad_id : chist[j];
    };
    const uint8_t* addrs[16];
    for (int h = 0; h < 16; h++) {
      int64_t mix = cat(0) * ple_mult[0] ^ (cat(1) * ple_mult[1]);
      if (h >= 8) mix ^= cat(2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * g_ple_rb;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, dst, (uint64_t)(uintptr_t)ple_table->data, 16);
      return;
    }
#endif
    for (int h = 0; h < 16; h++) memcpy(dst + h * g_ple_rb, addrs[h], g_ple_rb);
  }
  void ple_host(int token, float* dst) {
    chist.push_back(token);
    ple_rows((int)chist.size() - 1, dst);
  }

  // PLE device phase: upload + gate/conv kernels (must be enqueued after L0).
  void ple_gpu() {
    CK(hipMemcpyAsync(d_e, ple_e, 2560 * 4, hipMemcpyHostToDevice, g_str));
    gemv(wview(ckpt.at("layers.1.ple.key_proj.weight")), d_e, d_keys);
    gemv(wview(ckpt.at("layers.1.ple.value_proj.weight")), d_e, d_v);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_keys, d_ple_nk, d_kn, 2560,
                                                g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_R, d_ple_nk + 10240, d_qn, 2560,
                                                g_cfg.norm_eps, 4);
    k_ple_gate<<<4, 1024, 0, g_str>>>(d_kn, d_qn, d_v, d_U, 2560, 4);
    // ring[ringpos] = norm_conv(U)
    k_rmsnorm_zc_ring<<<4, 1024, 0, g_str>>>(d_U, d_ple_nk + 20480, d_ring, d_ringpos,
                                             2560, g_cfg.norm_eps);
    k_ple_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_ring, d_ringpos, d_ple_convw,
                                                       d_convout, 10240);
    k_add2<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_R, d_U, d_convout, 10240);
    // device ringpos advanced by k_step_incr (graph) or host upload (direct)
  }

  // one token; returns true + writes argmax id when want_token
  bool dbg = getenv("QWENOX_DEBUG") != nullptr;
  bool perf = getenv("QWENOX_PERF") != nullptr;
  bool qsa_loop = getenv("QWENOX_QSA_LOOP") != nullptr;  // prefill QSA fallback
  bool qsa_global_v = getenv("QWENOX_QSA_GLOBAL_V") != nullptr;
  bool moe_naive = getenv("QWENOX_MOE_NAIVE") != nullptr;  // prefill MoE fallback
  // Small batches: per-pair naive gemv reads ~the same expert bytes as the
  // packed path (few shared experts) but at ~2x its effective bandwidth
  // (measured: P=8 chunk 0.19s -> 0.14s). 0 disables the auto switch.
  int moe_naive_max = [] {
    const char* e = getenv("QWENOX_MOE_NAIVE_MAX");
    return e ? atoi(e) : 16;
  }();
  // Default small-P down path; QWENOX_MOE_DOWN_ATOMIC restores the old
  // per-expert atomic accumulation for numerical/performance comparisons.
  bool moe_down_topk =
      moe_deterministic && getenv("QWENOX_MOE_DOWN_ATOMIC") == nullptr;
  bool moe_ordered = getenv("QWENOX_MOE_ORDERED") != nullptr;  // diagnostic: expert-order scatter
  bool moe_host_route = getenv("QWENOX_MOE_HOST_ROUTE") != nullptr;
  bool moe_untiled = getenv("QWENOX_MOE_UNTILED") != nullptr;
  bool moe_fp32_io = getenv("QWENOX_MOE_FP32_IO") != nullptr;
  bool moe_up_legacy = getenv("QWENOX_MOE_UP_LEGACY") != nullptr;
  bool moe_up_no_table = getenv("QWENOX_MOE_UP_NO_TABLE") != nullptr;
  bool moe_down_no_table = getenv("QWENOX_MOE_DOWN_NO_TABLE") != nullptr;
  bool moe_reduce_old = getenv("QWENOX_MOE_REDUCE_OLD") != nullptr;
  // bf16 pair rows from the deterministic down kernel; the router weight is
  // applied in fp32 by k_moe_reduce_pw_bf16 instead of pre-multiplied in fp32.
  bool moe_pairs_bf16 = getenv("QWENOX_MOE_PAIRS_BF16") != nullptr;
  // hgn q4cp prefill MoE (P > moe_naive_max) via the LUT-decode F16 WMMA kernels
  // (27_kernels_moe_lut.inc, same dataflow as moe_gguf). Default on,
  // QWENOX_MOE_Q4W=0 restores QWENOX_MOE_LT / tiled k_moe_w4. When on, QWENOX_MOE_LT is
  // ignored (no LT buffers); decode (P <= moe_naive_max) is untouched.
  bool moe_q4w = moe_q4w_env();
  bool moe_lt = getenv("QWENOX_MOE_LT") != nullptr && !moe_q4w;  // prefill MoE: dequant + per-expert hipBLASLt
  // Validated PP dataflow changes. Unset each switch to restore its baseline.
  // Keep the P<=8 direct GEMV path's input precision and scratch lifetime.
  bool pp_moe_out = getenv("QWENOX_PP_MOE_OUT") != nullptr;
  bool moe_lt_bf16 = getenv("QWENOX_MOE_LT_BF16") != nullptr;  // bf16 GEMM outputs (LT path)
  bool moe_lt_cmp = getenv("QWENOX_MOE_LT_CMP") != nullptr;  // debug: in-run fused reference diff
  bool moe_lt_ovl = getenv("QWENOX_MOE_LT_OVL") != nullptr;  // overlap next group's dequant with current GEMMs (LT path)
  // LT loses below ~4K tokens (512 tiny per-expert launches + the d_eoff sync
  // cost more than the GEMM wins; measured 2026-09-10: +0.7s @2K, wash @8K).
  int moe_lt_min = getenv("QWENOX_MOE_LT_MIN") ? atoi(getenv("QWENOX_MOE_LT_MIN")) : 4096;
  bool gdn_loop = getenv("QWENOX_GDN_LOOP") != nullptr;  // prefill GDN fallback
  bool gdn_nosplit = getenv("QWENOX_GDN_NOSPLIT") != nullptr;  // single-kernel GDN fallback
  bool gdn_nostrip = getenv("QWENOX_GDN_NOSTRIP") != nullptr;  // old inter (no strip)
  bool gdn_stream = getenv("QWENOX_GDN_STREAM") != nullptr;  // four-chunk workspace window
  int gdn_window_chunks = [] {
    const char* e = getenv("QWENOX_GDN_WINDOW_CHUNKS");
    return e ? std::max(1, atoi(e)) : 4;
  }();
  bool gdn_wave = getenv("QWENOX_GDN_WAVE") != nullptr;  // column-local triangular solve
  bool gdn_pipe = getenv("QWENOX_GDN_PIPE") != nullptr;  // intra/strip two-stream window pipeline (strip path)
  bool gdn_pipe2 = getenv("QWENOX_GDN_PIPE2") != nullptr;  // fp16-resident WMMA intra (chunk-looped)
  bool gdn_fused = getenv("QWENOX_GDN_FUSED") != nullptr;  // intra+strip fused, block-per-head persistent (no ws)
  bool gr_scat4 = getenv("QWENOX_GR_SCAT4") != nullptr;  // one-block-per-token GR scatter+norm
  bool gdn_ut5 = getenv("QWENOX_GDN_UT5") != nullptr;  // blocked triangular inverse
  bool ple_loop = getenv("QWENOX_PLE_LOOP") != nullptr;  // prefill PLE fallback
  bool prof = getenv("QWENOX_PROF") != nullptr;  // host-side phase timing
  double p_topk_wait = 0, p_bucket = 0, p_gemm = 0, p_pleh = 0,
         p_ple_wait = 0;
  // perf accumulators (ms): embed, ple, attn, moe, head
  double t_acc[5] = {0, 0, 0, 0, 0};
  double t_enq = 0, t_sync = 0;
  int t_n = 0;
  std::chrono::steady_clock::time_point tick() {
    if (!perf) return std::chrono::steady_clock::time_point{};
    CK(hipDeviceSynchronize());
    return std::chrono::steady_clock::now();
  }
  void tock(int slot, std::chrono::steady_clock::time_point t0) {
    if (!perf) return;
    CK(hipDeviceSynchronize());
    t_acc[slot] +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
            .count();
  }
  void h_rms(const char* tag, int l, const char* sub, const float* p, int n) {
    if (!dbg) return;
    std::vector<float> h(n);
    CK(hipMemcpy(h.data(), (void*)p, n * 4, hipMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < n; i++) s += (double)h[i] * h[i];
    fprintf(stderr, "%s L%02d %s rms=%.4f [0..3]=%.5f %.5f %.5f %.5f\n", tag, l, sub,
            sqrt(s / n), h[0], h[1], h[2], h[3]);
  }
  // enqueue helpers — shared by direct execution and graph capture
  void enq_embed() {
    embed_rows(wview(ckpt.at("embed_tokens.weight")), d_token, 1, d_emb, 0);
    k_repeat4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_emb, d_R, g_cfg.d);
    h_rms("gpu", -1, "emb", d_emb, g_cfg.d);
  }
  void enq_layer(int l) {
    char prefix[128];
    snprintf(prefix, sizeof prefix, "layers.%d.attn_hyper_connection", l);
    auto t2 = tick();
    gr_read(prefix, d_x);
    if (is_qsa(l))
      qsa(l, l / 4, d_x, d_y);
    else
      gdn(l, l - (l + 1) / 4, d_x, d_y);
    h_rms("gpu", l, "attn-x", d_x, 2560);
    h_rms("gpu", l, "attn-y", d_y, 2560);
    gr_write(prefix, d_y);
    tock(2, t2);
    snprintf(prefix, sizeof prefix, "layers.%d.mlp_hyper_connection", l);
    auto t3 = tick();
    gr_read(prefix, d_x);
    moe(l, d_x, d_y);
    h_rms("gpu", l, "mlp-x", d_x, 2560);
    h_rms("gpu", l, "mlp-y", d_y, 2560);
    gr_write(prefix, d_y);
    tock(3, t3);
  }
  void enq_final() {
    gr_read("hyper_connection_mixer", d_x /*h*/);
    gemv(wview(ckpt.at("lm_head.weight")), d_x, d_logits);
    if (viz::on()) {
      const VizTap t = vz_tap(g_cfg.layers, false, true);
      if (t.ring) k_viz_top5<<<1, 256, 0, g_str>>>(d_logits, g_cfg.vocab, t);
    }
    argmax_rows(d_logits, g_cfg.vocab, 1, d_amx_cand, d_argmax, g_str);
  }

  // =================== prefill batch (Phase 3a) ==============================
  // Batched GEMM: Y[P,N] = X[P,K] . W[N,K]^T. W dequanted to bf16 workspace,
  // X converted to bf16; rocBLAS col-major mapping: C_cm[N,P] with
  // A = W (transa=T, lda=K), B = X (transb=N, ldb=K), C = Y (ldc=N).
  // W -> bf16 for the LT / WMMA paths: dense bf16 is used in place, the
  // quantized dtypes dequant into the shared d_wbf16 scratch (<= 64 MiB).
  // QWENOX_DENSE_SHADOW=1: one-time dequant of every ht/q6g64 dense weight into
  // a pinned host shadow. gemm_wbf16 returns the shadow entry, so the per-chunk
  // dequant (ht_deq) disappears. Bit-identical (same dequant kernels/bytes).
  void dense_shadow_init() {
    if (!getenv("QWENOX_DENSE_SHADOW")) return;
    if (!g_hgn_v2) {
      fprintf(stderr, "dense shadow: only hgn v2; off\n");
      return;
    }
    struct Item {
      const uint8_t* data;
      WView w;
      size_t off;
    };
    std::vector<Item> items;
    size_t total = 0;
    for (const auto& kv : ckpt.tensors()) {
      const Tensor& t = kv.second;
      if (t.dtype != 16 && t.dtype != 24) continue;
      if (kv.first.rfind("layers.", 0) != 0) continue;
      const WView w = wview(t);
      const size_t b = (size_t)w.rows * w.cols * 2;
      if (b > ((size_t)64 << 20) || w.rows < 2) continue;  // gemm_wbf16 would throw
      items.push_back({w.data, w, total});
      total += (b + 255) & ~(size_t)255;
    }
    if (items.empty() || total == 0) return;
    void* host = nullptr;
    if (hipHostMalloc(&host, total, hipHostMallocDefault) != hipSuccess) {
      fprintf(stderr, "dense shadow: hipHostMalloc %.2f GiB failed; off\n",
              total / 1073741824.0);
      return;
    }
    void* dev = nullptr;
    CK(hipHostGetDevicePointer(&dev, host, 0));
    for (const Item& e : items) {
      v2_deq_bf16(e.w, (uint16_t*)((char*)dev + e.off), g_str);
      dense_shadow_off_[e.data] = e.off;
    }
    CK(hipStreamSynchronize(g_str));
    dense_shadow_host_ = (uint8_t*)host;
    dense_shadow_dev_ = (uint8_t*)dev;
    dense_shadow_bytes_ = total;
    fprintf(stderr, "dense shadow: %zu weights, %.2f GiB pinned host\n",
            items.size(), total / 1073741824.0);
  }

  const uint16_t* gemm_wbf16(const WView& w) {
    uint64_t N = w.rows, K = w.cols;
    const uint16_t* wp = nullptr;
    if (wpre_override_) {  // se_gate_up_merged: caller staged the bf16 weight
      const uint16_t* p = wpre_override_;
      wpre_override_ = nullptr;
      return p;
    }
    if (w.dtype != 0 && dense_shadow_dev_) {
      auto it = dense_shadow_off_.find(w.data);
      if (it != dense_shadow_off_.end())
        return (const uint16_t*)(dense_shadow_dev_ + it->second);
    }
    if (w.dtype != 0)
      if (const uint16_t* pre = dpf_lookup(w)) return pre;  // dequantized ahead (dpf)
    if (w.dtype == 0) {
      wp = (const uint16_t*)w.data;
    } else if (w.dtype == 5) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 32;
      KP_SUB("ht_deq");
      k_dequant_q4cp_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data + 64, w.data + 64 + N * K / 2, (const float*)w.data, d_wbf16, N, K,
          w.scale_stride);
      KP_SUB_END();
      wp = d_wbf16;
    } else if (w.dtype == 7) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 64;
      KP_SUB("ht_deq");
      k_dequant_q8g64_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          w.data, d_wbf16, N, K);
      KP_SUB_END();
      wp = d_wbf16;
    } else if (w.dtype == 8) {
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      uint64_t tot = N * K / 32;
      KP_SUB("ht_deq");
      k_dequant_q8g32_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
          (const int8_t*)w.data, (const __half*)(w.data + N * K), d_wbf16, N, K);
      KP_SUB_END();
      wp = d_wbf16;
    } else if (w.dtype == 16 || w.dtype == 24) {  // hgn v2 ht / q6g64
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      KP_SUB("ht_deq");
      if (!abl_has("deq")) v2_deq_bf16(w, d_wbf16, g_str);
      KP_SUB_END();
      wp = d_wbf16;
    } else if (ggd_dtype(w.dtype)) {  // GGUF native dense
      if (N * K * 2 > ((size_t)64 << 20))
        throw std::runtime_error("gemm: weight too big for d_wbf16");
      KP_SUB("ht_deq");
      ggd_deq_bf16(w, d_wbf16, g_str);
      KP_SUB_END();
      wp = d_wbf16;
    } else {
      throw std::runtime_error("gemm: unsupported dtype");
    }
    return wp;
  }

  // QWENOX_DEQ_BENCH=1 (diagnostic, exits after): time one pass of the
  // per-chunk prefill dequant over every layer v2 dense weight (ht 16 / q6g64
  // 24) with the old and the fast kernels, and compare their bytes.
  int deq_bench() {
    std::vector<std::pair<std::string, WView>> ws;
    for (const auto& kv : ckpt.tensors()) {
      const Tensor& t = kv.second;
      if (t.dtype != 16 && t.dtype != 24) continue;
      if (kv.first.rfind("layers.", 0) != 0) continue;
      const WView w = wview(t);
      if (w.rows * w.cols * 2 > ((size_t)64 << 20)) continue;
      ws.emplace_back(kv.first, w);
    }
    std::sort(ws.begin(), ws.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    uint16_t* alt = nullptr;
    unsigned long long* d_cnt = nullptr;
    CK(hipMalloc(&alt, (size_t)64 << 20));
    CK(hipMalloc(&d_cnt, sizeof(unsigned long long)));
    double bytes[2] = {0, 0};  // read+write per dtype class (0 ht, 1 q6g64)
    int nw[2] = {0, 0};
    for (auto& e : ws) {
      const int c = e.second.dtype == 16 ? 0 : 1;
      const double out = (double)e.second.rows * e.second.cols * 2;
      const double in = c == 0 ? out / 4 : out * 13 / 32;  // 4 / 6.5 bits per weight
      bytes[c] += in + out;
      nw[c]++;
    }
    fprintf(stderr, "deq_bench: %d ht (%.2f GB r+w) + %d q6g64 (%.2f GB r+w) weights\n", nw[0],
            bytes[0] / 1e9, nw[1], bytes[1] / 1e9);
    const int reps = 5;
    for (int f = 0; f < 2; f++) {
      for (int c = 0; c < 2; c++) {
        auto pass = [&] {
          for (auto& e : ws)
            if ((e.second.dtype == 16) == (c == 0)) v2_deq_bf16(e.second, d_wbf16, g_str, f);
        };
        pass();
        CK(hipStreamSynchronize(g_str));
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; r++) pass();
        CK(hipStreamSynchronize(g_str));
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                .count() / reps;
        fprintf(stderr, "deq_bench: %s %s  %.1f ms/pass  %.0f GB/s\n", f ? "fast" : "old ",
                c ? "q6g64" : "ht   ", ms, bytes[c] / ms / 1e6);
      }
    }
    unsigned long long total = 0;
    int bad = 0;
    for (auto& e : ws) {
      v2_deq_bf16(e.second, d_wbf16, g_str, 0);
      v2_deq_bf16(e.second, alt, g_str, 1);
      CK(hipMemsetAsync(d_cnt, 0, sizeof(unsigned long long), g_str));
      const size_t n = e.second.rows * e.second.cols / 2;
      k_u32_mismatch<<<1024, 256, 0, g_str>>>((const uint32_t*)d_wbf16, (const uint32_t*)alt, n,
                                              d_cnt);
      unsigned long long h = 0;
      CK(hipMemcpy(&h, d_cnt, sizeof h, hipMemcpyDeviceToHost));
      if (h && bad++ < 5) fprintf(stderr, "deq_bench: MISMATCH %s: %llu words\n", e.first.c_str(), h);
      total += h;
    }
    fprintf(stderr, "deq_bench: compare %zu weights, %llu differing words -> %s\n", ws.size(),
            total, total ? "FAIL" : "PASS");
    CK(hipFree(alt));
    CK(hipFree(d_cnt));
    return total ? 1 : 0;
  }

  void gemm(const WView& w, const float* x, float* y, int P, uint64_t xstride = 0,
            const uint16_t* xbf = nullptr) {
    auto pt0 = std::chrono::steady_clock::now();
    uint64_t N = w.rows, K = w.cols;
    if (!xstride) xstride = K;
    // Small P: direct multi-row q4 gemv reads the weight once (~0.56x bytes
    // of the dequant->bf16->LT path's traffic) and skips 3 launches/gemm.
    // xbf inputs are bf16 with row stride K (the LT B layout).
    bool mr_done =
        xbf ? gemv_multi_q4cp(w, xbf, y, P, K, true)
            : gemv_multi_q4cp(w, x, y, P, xstride);
    if (mr_done) {
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
      return;
    }
    // A3 (§11.3): dense q4cp direct-read LUT for the d3/d9/dz/dq projection
    // shapes at P in [MINP, MAXP]. Harness ratios (deq+gemm_wmma)/lut with the
    // L2-hot dequant lower bound: d3 (2560,6144) 1.25/1.13 at P=1024/2048,
    // crossover ~4000; d9 (10240,2560) 1.11 at P=1024, crossover ~1500
    // (logs/a3_dense_lut.md); dz (6144,2560) 1.18@1024 crossover ~2000, dq
    // (12288,2560) 1.12@1024 crossover ~1800. In-engine (32k@1024 KPROF):
    // ht_deq -24 ms/chunk, gdn:proj -15, qsa:proj -4, qsa:oproj ~0 — the
    // engine's k1 WMMA config for (2560,6144) is ~1.3x faster than the
    // harness's old-d3 reference, so the true d3 ratio vs the OFF path is
    // ~1.0, not 1.25; end-to-end +1.8% < the 3% default-on bar, so MAXP
    // defaults are 0 (off). Opt in with QWENOX_DENSE_LUT_MAXP_{D3,D9,DZ,DQ}
    // (80%-of-crossover values 3200/1200/1600/1400). The weight runs as a
    // single-"expert" moe_lut_down with plain 64-slot tiles — the MoE-mt
    // BN=128 path (v2 MoE station) is not involved; the activation enters as
    // f16 (xbf calls: exact bf16->f16 of the live gated/normed input — the
    // f32 x there is the stale pre-gate/norm pre-image; f32 calls:
    // k_f32_to_f16_v4 rounding as in the MoE LUT station) — NOT bit-identical
    // vs the bf16 path, non-bit-exact by design, KLD-gated (c8192@4096:
    // mean 0.121637 vs off 0.121246, +0.0004 << 0.002). P<1024 keeps the Lt
    // fallback (MINP); =0 disables.
    static const int dlut_minp = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MINP");
      return s && *s ? atoi(s) : 1024;
    }();
    static const int dlut_d3 = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_D3");
      return s && *s ? atoi(s) : 0;
    }();
    static const int dlut_d9 = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_D9");
      return s && *s ? atoi(s) : 0;
    }();
    static const int dlut_dz = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_DZ");
      return s && *s ? atoi(s) : 0;
    }();
    static const int dlut_dq = [] {
      const char* s = getenv("QWENOX_DENSE_LUT_MAXP_DQ");
      return s && *s ? atoi(s) : 0;
    }();
    if (w.dtype == 5 && P >= dlut_minp && P <= maxbatch_cap && !(N % 128) &&
        K % 4 == 0 && xstride % 4 == 0 && (((uintptr_t)x & 15) == 0) &&
        (!xbf || ((((uintptr_t)xbf & 15) == 0) && K % 8 == 0))) {
      const int dlut_maxp = (N == 2560 && K == 6144) ? dlut_d3
                          : (N == 10240 && K == 2560) ? dlut_d9
                          : (N == 6144 && K == 2560) ? dlut_dz
                          : (N == 12288 && K == 2560) ? dlut_dq : 0;
      const moelut::LutW pdn =
          lut_q4cp_view(w.data, N, K, w.scale_stride, (int)N, 0);
      if (dlut_maxp && P <= dlut_maxp &&
          moelut::lut_ok(moelut::kQ4CP, pdn, false, (int)K)) {
        static bool dllog = false;
        if (!dllog) {
          dllog = true;
          fprintf(stderr,
                  "dense lut: q4cp d3<=%d d9<=%d dz<=%d dq<=%d minp=%d "
                  "(QWENOX_DENSE_LUT_MAXP_D3/D9/DZ/DQ, QWENOX_DENSE_LUT_MINP)\n",
                  dlut_d3, dlut_d9, dlut_dz, dlut_dq, dlut_minp);
        }
        const int nt = (P + 63) / 64;
        k_dense_tiles<<<1, 1024, 0, g_str>>>(d_dlut_tiles, d_dlut_nt, P);
        // x16 借用 d_xbf16b。xbf 调用点（qsa:oproj、fused gdn:oproj）的 f32 x
        // 是 gate/norm 前的旧值——活输入是 xbf（bf16, packed [P][K]），原地
        // bf16→f16（值精确）；非 xbf 调用点 x 即活 f32 输入，直接转。
        __half* x16 = (__half*)d_xbf16b;
        if (xbf) {
          const size_t n8 = (size_t)P * K / 8;
          k_bf16_to_f16_v8<<<(unsigned)((n8 + 255) / 256), 256, 0, g_str>>>(
              xbf, x16, n8);
        } else if (xstride == (uint64_t)K) {
          const size_t n4 = (size_t)P * K / 4;
          k_f32_to_f16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(
              x, x16, n4);
        } else {
          const size_t n4 = (size_t)P * (K / 4);
          k_f32_to_f16_v4s<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(
              x, x16, xstride, P, (int)K);
        }
        moe_lut_down(moelut::kQ4CP, pdn, x16, d_dlut_tiles, d_dlut_nt, nt, y,
                     (int)N, (int)K, g_str);
        if (prof)
          p_gemm += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt0)
                        .count();
        return;
      }
    }
    static const bool gemm_dbg = getenv("QWENOX_GEMM_DBG") != nullptr;
    if (gemm_dbg)
      fprintf(stderr, "gemm-LT: dtype=%d N=%llu K=%llu P=%d xbf=%d xs=%llu\n",
              w.dtype, (unsigned long long)N, (unsigned long long)K, P,
              (int)(xbf != nullptr), (unsigned long long)xstride);
    const uint16_t* wp = gemm_wbf16(w);
    uint64_t tot = (uint64_t)P * K;
    bool residual_ready =
        !xbf && prefill_fused && x == d_Rhatb && K == 10240 && xstride == K;
    bool cacheable = (x == d_xb || x == d_Rhatb);
    // Non-cacheable inputs convert into d_xbf16b, leaving the d_xb/d_Rhatb
    // cache entries (xf16_*) untouched; xbf bypasses conversion entirely.
    uint16_t* dst = cacheable ? d_xbf16 : d_xbf16b;
    const uint16_t* xp = xbf ? xbf : residual_ready ? d_Rhatbf16 : dst;
    if (!xbf && !residual_ready &&
        !(cacheable && xf16_src == x && xf16_xs == xstride && xf16_P == P &&
          xf16_K == (int)K)) {
      if (K % 4 == 0 && xstride % 4 == 0 &&
          (((uintptr_t)x | (uintptr_t)dst) & 15) == 0)
        k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
            x, dst, xstride, P, (int)K);
      else
        k_f32_to_bf16<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
            x, dst, xstride, P, (int)K);
      if (cacheable) {
        xf16_src = x;
        xf16_xs = xstride;
        xf16_P = P;
        xf16_K = (int)K;
      }
    }
    float one = 1.f, zero = 0.f;
    if (getenv("QWENOX_GEMM_RB")) {  // fallback: rocBLAS default algo
      rocblas_set_stream(rbh, g_str);
      rocblas_status st = rocblas_gemm_ex(
          rbh, rocblas_operation_transpose, rocblas_operation_none, (rocblas_int)N,
          (rocblas_int)P, (rocblas_int)K, &one, wp, rocblas_datatype_bf16_r,
          (rocblas_int)K, xp, rocblas_datatype_bf16_r, (rocblas_int)K, &zero, y,
          rocblas_datatype_f32_r, (rocblas_int)N, y, rocblas_datatype_f32_r,
          (rocblas_int)N, rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
      if (st != rocblas_status_success) {
        fprintf(stderr, "rocblas_gemm_ex failed: %d (N=%llu K=%llu P=%d)\n", (int)st,
                (unsigned long long)N, (unsigned long long)K, P);
        exit(1);
      }
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
      return;
    }
    // Phase 3g: self-written bf16 WMMA GEMM (QWENOX_GEMM_WMMA=1). Diverts the
    // winning projection shapes at P >= 1024 to k_gemm_wmma — d3 config
    // (512t w2x8 f4x2, gm=16) for N=2560 K=6144, d9 (256t w2x4 f4x4) for
    // N=6144/10240/12288 K=2560 (gm=4) and the narrow-K/N winners
    // N=10240 K=320 / N=512 K=2560 / N=2560 K=640 (gm=2, 1.25-2.0x vs Lt in
    // tools/gemm_lt_bench.cu); every other shape and the default flag-off
    // state fall through to hipBLASLt unchanged. Purely async. The kernel
    // requires K % 32 == 0; P and N tails are handled inside the kernel
    // (staging row clamps + predicated epilogue; the grid must be ceiled to
    // match). (640,2560) was measured and stays on Lt — 1.08-1.13x, under
    // the 1.15x bar. KST=64 family (128B per row per staging step, the
    // tools/gr320_proto.cu lesson; re-swept 2026-09-24 with
    // tools/gemm_lt_bench.cu at P=8192/16384, gm=1 best for both):
    //  - (320,10240) GR mix: h1 BM=64 BP=160 (num_n=2 exact), 3.9 ms vs
    //    5.3 ms Lt idx4 (x1.35); LDS 65,152B of 65,536 — any layout growth
    //    in the kernel breaks it.
    //  - (2560,6144) out-proj: k1 BM=64 BP=128, 16.4 ms vs 21.1-21.6 ms for
    //    the old d3 512t config and Lt (x1.3).
    // The K=2560 d9 shapes and (2560,640) stay on d9 (KST=64 configs lose
    // there). QWENOX_GEMM_NO_K64 restores the old routing (Lt / d3) for A/B.
    // QWENOX_GEMM_RB above takes precedence if both are set.
    static const bool gemm_wmma = getenv("QWENOX_GEMM_WMMA") != nullptr;
    static const bool gemm_k64 = getenv("QWENOX_GEMM_NO_K64") == nullptr;
    if (gemm_dbg)
      fprintf(stderr, "gemm-ptr: N=%llu K=%llu P=%d wp=%p xp=%p y=%p\n",
              (unsigned long long)N, (unsigned long long)K, P,
              (const void*)wp, (const void*)xp, (void*)y);
    if (gemm_wmma && P >= 1024) {
      if (gemm_k64 && ((N == 320 && K == 10240) || (N == 2560 && K == 6144))) {
        if (N == 320)
          k_gemm_wmma<64, 160, 2, 2, 2, 5, 64, 128>
              <<<(unsigned)(((P + 63) / 64) * 2), 128, 0, g_str>>>(
                  xp, wp, y, P, (int)K, (int)N, 1);
        else if (gemm_splitk_on() && P >= 8192 && d_skx) {
          // B3 split-K：X 重排成 2 个连续 K=3072 切片（W 维持原布局 strided
          // 读），slice0 写 Y，slice1 Epi=3 原子加。16K 下单 GEMM 省 ~1.5 ms
          //（proto x1.13 扣除 repack 0.23 ms）。P<8192 收益消失（x1.01）。
          const int ks = 3072;
          const unsigned grid = (unsigned)(((P + 63) / 64) * 20);
          k_splitk_repack2<<<1024, 256, 0, g_str>>>(xp, d_skx, P, (int)K);
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
              <<<grid, 128, 0, g_str>>>(d_skx, wp, y, P, ks, (int)N, 1,
                                        nullptr, nullptr, nullptr, nullptr, 0,
                                        (unsigned)K);
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128, 3>
              <<<grid, 128, 0, g_str>>>(d_skx + (size_t)P * ks, wp + ks, y, P,
                                        ks, (int)N, 1, nullptr, nullptr,
                                        nullptr, nullptr, 0, (unsigned)K);
        } else
          k_gemm_wmma<64, 128, 2, 2, 2, 4, 64, 128>
              <<<(unsigned)(((P + 63) / 64) * 20), 128, 0, g_str>>>(
                  xp, wp, y, P, (int)K, (int)N, 1);
        if (prof)
          p_gemm += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt0)
                        .count();
        return;
      }
      const bool gw_d3 = (N == 2560 && K == 6144);
      static const bool d9_old0 = getenv("QWENOX_GEMM_D9_OLD") != nullptr;
      // aggr: N=640 (shared gate/up, indexer proj; was hipBLASLt ~1.7 ms @16K)
      // and N=512 join the 128x128 tile route (1.33 / 1.06 ms @16K).
      const bool gw_d9 = (K == 2560 && (N == 6144 || N == 10240 || N == 12288 ||
                                        (!d9_old0 && (N == 640 || N == 512)) ||
                                        (N == 1280 && se_merge_active_)));
      const bool gw_d9n = !gw_d9 && ((N == 10240 && K == 320) ||
                                     (N == 512 && K == 2560) || (N == 2560 && K == 640));
      if (gw_d3 || gw_d9 || gw_d9n) {
        const unsigned gw_grid =
            (unsigned)(((P + 127) / 128) * (int)((N + 255) / 256));
        // aggr 10-05 (tools/gemm_tune d9 @16384): 128x128 w2x2 f4x4 128-thread
        // tile beats the 128x256 d9 one — qkv N=10240 22.0 vs 22.8 ms (gm=4),
        // z N=6144 12.8 vs 13.8 ms (gm=1). QWENOX_GEMM_D9_OLD=1 restores.
        static const bool d9_old = getenv("QWENOX_GEMM_D9_OLD") != nullptr;
        if (gw_d9 && !d9_old) {
          const unsigned g2 = (unsigned)(((P + 127) / 128) * (int)((N + 127) / 128));
          k_gemm_wmma<128, 128, 2, 2, 4, 4, 32, 128>
              <<<g2, 128, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N,
                                      (N == 10240 || N == 12288) ? 4 : 1);
        } else if (gw_d3)
          k_gemm_wmma<128, 256, 2, 8, 4, 2, 32, 512>
              <<<gw_grid, 512, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 16);
        else if (gw_d9)
          k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
              <<<gw_grid, 256, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 4);
        else
          k_gemm_wmma<128, 256, 2, 4, 4, 4, 32, 256>
              <<<gw_grid, 256, 0, g_str>>>(xp, wp, y, P, (int)K, (int)N, 2);
        if (prof)
          p_gemm += std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - pt0)
                        .count();
        return;
      }
    }
    // hipBLASLt path: ~30 TFLOPS vs ~5.6 for the rocBLAS default algo on
    // gfx1151. The heuristic's rank order is misleading — index 4 is the fast
    // tensile kernel for every projection shape measured (5-6x over index 0),
    // so use it after a validation run; fall back to a timing scan if the
    // list is short or index 4 fails. Full descriptor set cached per (N,K,P).
    // A1 (2026-10-17, HANDOFF-AGENT-SHORT-PREFILL): the algo is selected per
    // (N,K,Pb), Pb = P ceiled to 64 (256 at P>=1024) — agent increments arrive
    // with a fresh exact P every round and each exact-P miss paid a heuristic
    // plus a queue-draining validation sync (30-90 ms, R1 lt_miss log). The
    // layouts' leading dims (K,N,N) do not depend on P, so an algo validated
    // at one P in the bucket is valid at every P; the descriptor set is still
    // built per exact P (host-only, no sync). The heuristic runs with Pb
    // layouts so the pick is uniform within a bucket (deterministic per P).
    // QWENOX_LT_BUCKET=0 restores exact-P selection.
    static const bool lt_bucket = [] {
      const char* s = getenv("QWENOX_LT_BUCKET");
      return !s || strcmp(s, "0") != 0;
    }();
    const int Pb = !lt_bucket ? P
                   : P < 1024  ? ((P + 63) / 64) * 64
                               : ((P + 255) / 256) * 256;
    uint64_t key = (N << 40) ^ (K << 20) ^ (uint64_t)P;
    auto it = lt_algos.find(key);
    if (it == lt_algos.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16BF, K, P, K);
      hipblasLtMatrixLayoutCreate(&e.Cl, HIP_R_32F, N, P, N);
      hipblasLtMatrixLayoutCreate(&e.Dl, HIP_R_32F, N, P, N);
      const uint64_t bkey = (N << 40) ^ (K << 20) ^ (uint64_t)Pb;
      auto bt = lt_buckets.find(bkey);
      if (bt != lt_buckets.end()) {
        e.algo = bt->second;  // validated at the bucket's first miss, host-only
      } else {
        // QWENOX_LT_MISS_LOG=1: host time spent on this miss (heuristic + the
        // validation run's stream sync), to size the per-new-bucket cost.
        static const bool lt_miss_log = getenv("QWENOX_LT_MISS_LOG") != nullptr;
        const auto lt_m0 = std::chrono::steady_clock::now();
        struct LtMissLog {
          bool on;
          std::chrono::steady_clock::time_point t0;
          uint64_t N, K;
          int P, Pb;
          ~LtMissLog() {
            if (on)
              fprintf(stderr, "lt_miss N=%llu K=%llu P=%d Pb=%d %.2f ms\n",
                      (unsigned long long)N, (unsigned long long)K, P, Pb,
                      std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count());
          }
        } lt_ml{lt_miss_log, lt_m0, N, K, P, Pb};
        // Heuristic on bucket layouts so the pick is uniform within a bucket;
        // the validation run below uses the exact-P set (real operands).
        hipblasLtMatrixLayout_t bBl, bCl, bDl;
        hipblasLtMatrixLayoutCreate(&bBl, HIP_R_16BF, K, Pb, K);
        hipblasLtMatrixLayoutCreate(&bCl, HIP_R_32F, N, Pb, N);
        hipblasLtMatrixLayoutCreate(&bDl, HIP_R_32F, N, Pb, N);
        hipblasLtMatmulPreference_t pref;
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(
            pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
            sizeof(lt_ws_bytes));
        hipblasLtMatmulHeuristicResult_t hres[8];
        int nres = 0;
        hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, bBl, bCl, bDl, pref,
                                        8, hres, &nres);
        hipblasLtMatrixLayoutDestroy(bBl);
        hipblasLtMatrixLayoutDestroy(bCl);
        hipblasLtMatrixLayoutDestroy(bDl);
        auto try_run = [&](int i) {
          hipblasLtMatmul(lth, e.opd, &one, wp, e.Al, xp, e.Bl, &zero, y, e.Cl, y,
                          e.Dl, &hres[i].algo, d_ltws, lt_ws_bytes, g_str);
          if (hipStreamSynchronize(g_str) != hipSuccess) {
            (void)hipGetLastError();
            return false;
          }
          return true;
        };
        int bi = -1;
        if (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
            hres[4].workspaceSize <= lt_ws_bytes && try_run(4))
          bi = 4;
        else {  // fallback: time all candidates
          double best = 1e30;
          for (int i = 0; i < nres; i++) {
            if (hres[i].state != HIPBLAS_STATUS_SUCCESS) continue;
            if (hres[i].workspaceSize > lt_ws_bytes) continue;
            if (!try_run(i)) continue;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < 3; r++) try_run(i);
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
            if (ms < best) {
              best = ms;
              bi = i;
            }
          }
        }
        if (bi < 0) {
          fprintf(stderr, "hipBLASLt: no algo for N=%llu K=%llu P=%d\n",
                  (unsigned long long)N, (unsigned long long)K, P);
          exit(1);
        }
        e.algo = hres[bi].algo;
        hipblasLtMatmulPreferenceDestroy(pref);
        lt_buckets.emplace(bkey, e.algo);
      }
      it = lt_algos.emplace(key, e).first;
    }
    const LtEntry& e = it->second;
    hipblasStatus_t lst =
        hipblasLtMatmul(lth, e.opd, &one, wp, e.Al, xp, e.Bl, &zero, y, e.Cl, y,
                        e.Dl, &e.algo, d_ltws, lt_ws_bytes, g_str);
    if (lst != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "hipblasLtMatmul failed: %d (N=%llu K=%llu P=%d)\n", (int)lst,
              (unsigned long long)N, (unsigned long long)K, P);
      exit(1);
    }
    if (prof)
      p_gemm += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - pt0)
                    .count();
  }

  // gemm variant for inputs that are already bf16 was used by the 3a' grouped
  // MoE path; removed with it (3d-1 fused W4 kernels replace it).

  // bf16 A (weights, [N][K] k-contiguous -> op T) x bf16 B ([P][K] k-contiguous
  // -> op N) -> C[N,P] (n-contiguous), fp32 or bf16 out. Used by QWENOX_MOE_LT
  // per-expert GEMMs. Algo cached per (N,K,P,outbf16); on miss trust heuristic
  // index 4 (the fast Tensile kernel for every projection shape measured in
  // tools/moe_lt_bench.cu scan mode; per-M validation syncs would stall the
  // per-layer expert loop). Bit 62 keeps keys disjoint from dense gemm's.
  void gemm_bf16(const __hip_bfloat16* A, const __hip_bfloat16* B, void* C,
                 int N, int K, int P, bool outbf16) {
    float one = 1.f, zero = 0.f;
    uint64_t key = (1ULL << 62) ^ ((uint64_t)outbf16 << 63) ^ ((uint64_t)N << 40) ^
                   ((uint64_t)K << 20) ^ (uint64_t)P;
    auto it = lt_algos.find(key);
    hipDataType dtout = outbf16 ? HIP_R_16BF : HIP_R_32F;
    if (it == lt_algos.end()) {
      LtEntry e;
      hipblasLtMatmulDescCreate(&e.opd, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSA, &opT,
                                      sizeof(opT));
      hipblasLtMatmulDescSetAttribute(e.opd, HIPBLASLT_MATMUL_DESC_TRANSB, &opN,
                                      sizeof(opN));
      hipblasLtMatrixLayoutCreate(&e.Al, HIP_R_16BF, K, N, K);
      hipblasLtMatrixLayoutCreate(&e.Bl, HIP_R_16BF, K, P, K);
      hipblasLtMatrixLayoutCreate(&e.Cl, dtout, N, P, N);
      hipblasLtMatrixLayoutCreate(&e.Dl, dtout, N, P, N);
      hipblasLtMatmulPreference_t pref;
      hipblasLtMatmulPreferenceCreate(&pref);
      hipblasLtMatmulPreferenceSetAttribute(
          pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes,
          sizeof(lt_ws_bytes));
      hipblasLtMatmulHeuristicResult_t hres[8];
      int nres = 0;
      hipblasLtMatmulAlgoGetHeuristic(lth, e.opd, e.Al, e.Bl, e.Cl, e.Dl, pref,
                                      8, hres, &nres);
      if (nres <= 0) {
        fprintf(stderr, "moe_lt: no algo for N=%d K=%d P=%d\n", N, K, P);
        exit(1);
      }
      int bi = (nres > 4 && hres[4].state == HIPBLAS_STATUS_SUCCESS &&
                hres[4].workspaceSize <= lt_ws_bytes)
                   ? 4
                   : 0;
      e.algo = hres[bi].algo;
      hipblasLtMatmulPreferenceDestroy(pref);
      it = lt_algos.emplace(key, e).first;
    }
    const LtEntry& e = it->second;
    hipblasStatus_t lst =
        hipblasLtMatmul(lth, e.opd, &one, A, e.Al, B, e.Bl, &zero, C, e.Cl, C,
                        e.Dl, &e.algo, d_ltws, lt_ws_bytes, g_str);
    if (lst != HIPBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "moe_lt: hipblasLtMatmul failed: %d (N=%d K=%d P=%d)\n",
              (int)lst, N, K, P);
      exit(1);
    }
  }

  // #51 HC gate fusion: the mix-up GEMM (N=10240 K=320) with the
  // k_gr_combine_b_hc_bf16 epilogue folded in (k_gemm_wmma Epi=1, W rows
  // permuted at LDS staging so one lane holds the 4 branch gates of its
  // channel). Same X and W bf16 prep as the gw_d9n route in gemm(); the
  // tile / column->block assignment changes but not the K order -> G and x are
  // bit-identical, and the P x 10240 fp32 G never goes to DRAM. Writes d_xb
  // (fp32) + d_xbf16 exactly like the unfused pair. Returns false (caller
  // runs the unfused pair) unless that exact route would have been taken.
  // Opt-out QWENOX_HC_FUSE=0 (tools/hcmix_verify.sh).
  // B2-2 (QWENOX_HC_NORHAT=1 显式开启，默认关): sc 非空时 Rh 实参应是未归一化
  // 的 d_Rb，epilogue 现场乘 (R*sc)*(1+Hw)（见 k_gemm_wmma Epi=1 注释），Rhat
  // 不落盘。终判否决（2026-10-03，128k@16384 背靠背 1175.7 vs 1216.4 = -3.3%，
  // KPROF hc+hc2 +293ms/chunk）：Epi=2 staging 归一化的 VALU + Sc/Hw 额外读
  // 超过省下的 Rhat 落盘带宽；正确性全过（c8192 0.08129 vs 0.08249），代码留存。
  bool hc_norhat(int P) const {
    static const bool on = getenv("QWENOX_HC_NORHAT") != nullptr &&
                           !env_zero("QWENOX_HC_NORHAT") &&
                           !env_zero("QWENOX_HC_FUSE") &&
                           getenv("QWENOX_GEMM_WMMA") != nullptr &&
                           getenv("QWENOX_GEMM_RB") == nullptr &&
                           getenv("QWENOX_GEMM_NO_K64") == nullptr;
    return on && prefill_fused && gr_bf16 && P >= 1024 && g_cfg.branches == 4 &&
           (uint64_t)g_cfg.d * 4 == 10240 &&
           ((((uintptr_t)d_t320b | (uintptr_t)d_xbf16b) & 15) == 0);
  }
  bool hc_up_fused(const WView& w, int P, const float* sc = nullptr,
                   const float* hw = nullptr, bool x_ready = false) {
    static const bool on = !env_zero("QWENOX_HC_FUSE") &&
                           getenv("QWENOX_GEMM_WMMA") != nullptr &&
                           getenv("QWENOX_GEMM_RB") == nullptr;
    const uint64_t N = w.rows, K = w.cols;
    if (!on || P < 1024 || N != 10240 || K != 320 || g_cfg.branches != 4 ||
        (uint64_t)g_cfg.d * 4 != N)
      return false;
    if ((((uintptr_t)d_t320b | (uintptr_t)d_xbf16b) & 15) != 0) return false;
    auto pt0 = std::chrono::steady_clock::now();
    const uint16_t* wp = gemm_wbf16(w);
    const uint64_t tot = (uint64_t)P * K;
    if (!x_ready)  // aggr: k_hc_mixcomb already wrote the bf16 X into d_xbf16b
      k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
          d_t320b, d_xbf16b, K, P, (int)K);
    // 128x128 w4x2 f2x4 (tools/hcmix_proto.cu tile sweep): the fused epilogue
    // favours the narrower tile — 2.33 vs 2.80 ms at P=8192 (pair 4.92 ms),
    // still bit-identical (K accumulation order is tile-independent).
    const unsigned grid = (unsigned)(((P + 127) / 128) * (int)(N / 128));
    k_gemm_wmma<128, 128, 4, 2, 2, 4, 32, 256, 1><<<grid, 256, 0, g_str>>>(
        d_xbf16b, wp, d_xb, P, (int)K, (int)N, 2,
        sc ? (const uint16_t*)d_Rb : d_Rhatbf16, d_xbf16, sc, hw);
    if (prof)
      p_gemm += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - pt0)
                    .count();
    return true;
  }

  void gr_mix_b(const std::string& prefix, int P, float* hw = nullptr,
                bool norhat = false) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    static const bool mixsplit = !env_zero("QWENOX_HC_MIXSPLIT");
    const WView wdn = wview(ckpt.at(buf));
    if (norhat && mixsplit && wdn.dtype != 0 && wdn.rows == 320 && wdn.cols == 10240) {
      // aggr 10-05: branch-split mix-down. W' = W*(1+hw) (dequant scratch,
      // scaled in place), 4 K=2560 slice GEMMs in one launch on the raw R'
      // (blockIdx.y = branch, partials in the fused path's idle d_Gb), then
      // t = silu(0.25 * sum_b sc_b * part_b) straight to bf16 for the up GEMM.
      // Not bit-identical to the Rhat path (bf16 W' vs bf16 Rhat rounding).
      uint16_t* wp = (uint16_t*)gemm_wbf16(wdn);
      k_scale_cols_bf16<<<640, 256, 0, g_str>>>(wp, hw, 320, 10240);
      float* parts = d_Gb;
      k_gemm_wmma<64, 160, 2, 2, 2, 5, 64, 128>
          <<<dim3((unsigned)(((P + 63) / 64) * 2), 4), 128, 0, g_str>>>(
              (const uint16_t*)d_Rb, wp, parts, P, 2560, 320, 1, nullptr, nullptr,
              nullptr, nullptr, 10240, 10240, 2560, 2560, (size_t)P * 320);
      k_hc_mixcomb<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(parts, d_hcsc,
                                                            d_xbf16b, P, 320);
      const WView wup = wview(ckpt.at(std::string(prefix) + ".input_mix_weight_up.weight"));
      if (!hc_up_fused(wup, P, d_hcsc, hw, true)) {
        fprintf(stderr, "hc_norhat: up-fused route fell back with Rhat unwritten\n");
        exit(1);
      }
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
      return;
    }
    if (norhat) {
      // B2-2: mix-down 的 A = 未归一化的 d_Rb，Epi=2 在 staging 现场归一化
      //（与 rmsnorm 同一表达式 -> A tile 与被取代的 Rhat 逐 bit）。路线条件与
      // gemm() 的 (320,10240) k64 分流一致（hc_norhat 已核）。
      auto pt0 = std::chrono::steady_clock::now();
      const uint16_t* wp = gemm_wbf16(wview(ckpt.at(buf)));
      k_gemm_wmma<64, 160, 2, 2, 2, 5, 64, 128, 2>
          <<<(unsigned)(((P + 63) / 64) * 2), 128, 0, g_str>>>(
              (const uint16_t*)d_Rb, wp, d_t320b, P, 10240, 320, 1, nullptr,
              nullptr, d_hcsc, hw);
      if (prof)
        p_gemm += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - pt0)
                      .count();
    } else {
      gemm(wview(ckpt.at(buf)), d_Rhatb, d_t320b, P, 0,
           gr_bf16 ? d_Rhatbf16 : nullptr);
    }
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f, P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    const WView wup = wview(ckpt.at(buf));
    if (norhat) {
      // hc_norhat 与 hc_up_fused 条件逐一对应；仍失败说明 Rhat 未落盘而
      // 后续读者存在——不能静默走旧路（会读 stale Rhat），直接 fatal。
      if (!hc_up_fused(wup, P, d_hcsc, hw)) {
        fprintf(stderr, "hc_norhat: up-fused route fell back with Rhat unwritten\n");
        exit(1);
      }
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
      return;
    }
    if (prefill_fused && gr_bf16 && hc_up_fused(wup, P)) {
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
      return;
    }
    gemm(wup, d_t320b, d_Gb, P);
    if (prefill_fused) {
      if (gr_bf16) {
        // verify sizes: 4 column slices over P*4 blocks (per-column arithmetic
        // single-thread, unchanged). QWENOX_COMBINE_CS=0 keeps P blocks.
        static const bool comb_cs = !env_zero("QWENOX_COMBINE_CS");
        if (comb_cs && P <= 8)
          k_gr_combine_b_hc_bf16<<<dim3(P, 4), 256, 0, g_str>>>(
              d_Gb, d_Rhatbf16, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16,
              g_cfg.d / 4);
        else
          k_gr_combine_b_hc_bf16<<<P, 256, 0, g_str>>>(
              d_Gb, d_Rhatbf16, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16,
              g_cfg.d);
      } else
        k_gr_combine_b<true><<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
            d_Gb, d_Rhatb, d_xb, g_cfg.d, g_cfg.branches, P, d_xbf16);
      xf16_src = d_xb;
      xf16_xs = xf16_K = g_cfg.d;
      xf16_P = P;
    } else {
      k_sigmoid_inplace<<<(P * 10240 + 255) / 256, 256, 0, g_str>>>(d_Gb, P * 10240);
      k_gr_combine_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_Gb, d_Rhatb, d_xb, g_cfg.d, g_cfg.branches, P);
      if (xf16_src == d_xb) xf16_src = nullptr;
    }
  }

  void gr_read_b(const std::string& prefix, int P) {
    float* hw = ensure_hcnorm(prefix);
    if (gr_bf16)
      k_rmsnorm_zc_grouped_bf16<<<4 * P, 1024, 0, g_str>>>(
          (const uint16_t*)d_Rb, hw, d_Rhatbf16, 2560, g_cfg.norm_eps, 4);
    else if (prefill_fused)
      k_rmsnorm_zc_grouped<true><<<4 * P, 1024, 0, g_str>>>(
          d_Rb, hw, d_Rhatb, 2560, g_cfg.norm_eps, 4, d_Rhatbf16);
    else
      k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(
          d_Rb, hw, d_Rhatb, 2560, g_cfg.norm_eps, 4);
    if (xf16_src == d_Rhatb) xf16_src = nullptr;  // d_Rhatb overwritten
    injp_cur = -1;  // new Rhat without inject partials
    gr_mix_b(prefix, P);
  }

  // HC inject fusion (k_gr_scatter_norm_inj_bf16): the scatter+norm that
  // produces a site's Rhat also emits that site's inject projection as
  // per-branch partials, so the later N=4 inject GEMM (which re-read the
  // whole P x 10240 bf16 Rhat, ~0.97 ms at P=8192) is skipped. R / Rhat are
  // bit-identical; w4 differs in summation order only (fp32, err/sum|terms|
  // ~1e-8 vs ~1e-7 for any sequential fp32 dot; tools/injfuse_proto.cu).
  // Opt-out QWENOX_HC_INJ_FUSE=0 (tools/injfuse_verify.sh: KLD + speed).
  bool inj_fuse_on() const {
    static const bool on = !env_zero("QWENOX_HC_INJ_FUSE");
    return on && gr_bf16 && prefill_fused && !gr_scat4 && g_cfg.d <= 2560 &&
           g_cfg.branches == 4;
  }
  // Private bf16 copy of a site's inject weight (4 x 4d). gemm_wbf16 dequants
  // quantized weights into the shared d_wbf16 scratch, which later GEMMs
  // overwrite, so quantized ones are copied out once (~80 KB per site).
  const uint16_t* inj_wbf16(const std::string& prefix) {
    auto it = inj_wcache_.find(prefix);
    if (it != inj_wcache_.end()) return it->second;
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    const auto* ct = ckpt.find(buf);
    if (!ct) return inj_wcache_[prefix] = nullptr;
    const WView w = wview(*ct);
    if ((uint64_t)w.rows != 4 || (uint64_t)w.cols != (uint64_t)4 * g_cfg.d ||
        !(w.dtype == 0 || w.dtype == 5 || w.dtype == 7 || w.dtype == 8 || w.dtype == 24))
      return inj_wcache_[prefix] = nullptr;
    const uint16_t* src = gemm_wbf16(w);
    uint16_t* dst = (uint16_t*)src;
    if (w.dtype != 0) {
      CK(hipMalloc(&dst, (size_t)w.rows * w.cols * 2));
      CK(hipMemcpyAsync(dst, src, (size_t)w.rows * w.cols * 2,
                        hipMemcpyDeviceToDevice, g_str));
    }
    return inj_wcache_[prefix] = dst;
  }

  // First half of gr_write_b: make d_w4b for prefix (inject partials psum or
  // the N=4 GEMM). Bit-identical to what gr_write_b would compute; injp_cur
  // is left untouched so a non-fused fallback recomputes the same d_w4b.
  void gr_write_prep_b(const std::string& prefix, int P) {
    char buf[128];
    if (injp_cur >= 0 && injp_for == prefix) {
      k_inj_psum<<<(P * 4 + 255) / 256, 256, 0, g_str>>>(d_injp[injp_cur], d_w4b, P);
    } else {
      snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
      gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
           gr_bf16 ? d_Rhatbf16 : nullptr);
    }
  }

  void gr_write_b(const std::string& prefix, int P) {
    const bool had_p = injp_cur >= 0 && injp_for == prefix;
    gr_write_prep_b(prefix, P);
    if (had_p) injp_cur = -1;
    if (gr_bf16)
      k_gr_write_b_hc_bf16<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_w4b, (uint16_t*)d_Rb, d_yb, g_cfg.d, g_cfg.branches, P);
    else
      k_gr_write_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, g_cfg.d, g_cfg.branches, P);
    df2_hc_capture(prefix, P);
  }

  void gr_write_read_b(const std::string& write_prefix,
                       const std::string& read_prefix, int P) {
    char buf[128];
    // B2-1: the moe reduce already applied this site's write (R is
    // post-write); the scatter below only norms. pfull/pin go unused.
    const bool write_done = !hcw_site_.empty() && hcw_site_ == write_prefix;
    if (write_done) hcw_site_.clear();
    const bool have_p = injp_cur >= 0 && injp_for == write_prefix;
    if (!have_p && !write_done) {
      snprintf(buf, sizeof buf, "%s.block_inject_weight.weight",
               write_prefix.c_str());
      gemm(wview(ckpt.at(buf)), d_Rhatb, d_w4b, P, 0,
           gr_bf16 ? d_Rhatbf16 : nullptr);
    }
    float* hw = ensure_hcnorm(read_prefix);
    const uint16_t* wn = inj_fuse_on() ? inj_wbf16(read_prefix) : nullptr;
    if (!wn && write_done)  // R is already post-write; a writing scatter would double-apply
      throw std::runtime_error("gr_write_read_b: fused HC write but read site can't fuse");
    if (!wn && have_p && !write_done)  // partials exist but this site can't fuse (defensive)
      k_inj_psum<<<(P * 4 + 255) / 256, 256, 0, g_str>>>(d_injp[injp_cur], d_w4b, P);
    if (wn) {
      static bool said = false;
      if (!said) { said = true; fprintf(stderr, "[hc-inj-fuse] on (inject GEMM folded into scatter_norm)\n"); }
      const int nxt = injp_cur == 0 ? 1 : 0;  // pin != pout
      const float* pfull = (have_p || write_done) ? nullptr : d_w4b;
      const float* pin = (have_p && !write_done) ? d_injp[injp_cur] : nullptr;
      const float* y = write_done ? nullptr : d_yb;
      // B2-2: Rhat 不落盘，scatter 只写 P×4 inv scales（gr_read_b 站除外——
      // 那里的 Rhat 还要喂后续 inject GEMM，维持旧路，见 gr_read_b）。
      const bool nr = hc_norhat(P);
      uint16_t* const rh = nr ? nullptr : d_Rhatbf16;
      float* const sc = nr ? d_hcsc : nullptr;
      // Verify sizes (P<=8): 1 token/block -> 4P blocks instead of 4*ceil(P/4).
      // Per-(t,b) arithmetic is T-invariant (proto: R/Rhat bit-identical), the
      // 4-token serial loop was pure latency at 4 blocks. QWENOX_SCAT_T1=0: T=4.
      static const bool scat_t1 = !env_zero("QWENOX_SCAT_T1");
      // v2 (coalesced + pipelined): kernel −5% only (this op is at ~190 GB/s,
      // bandwidth-bound) and not bit-identical -> opt-in QWENOX_GR_SCAT_V2=1.
      static const bool scat_v2 = getenv("QWENOX_GR_SCAT_V2") && !env_zero("QWENOX_GR_SCAT_V2");
      static const bool scat_t8 = getenv("QWENOX_GR_SCAT_T8") != nullptr;
      if (scat_v2 && scat_t8 && g_cfg.d == 2560 && P > 8) {
        if (write_done)
          k_gr_scatter_norm_inj_bf16_v2<8, false><<<4 * ((P + 7) / 8), 256, 0, g_str>>>(
              nullptr, nullptr, (uint16_t*)d_Rb, nullptr, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
        else
          k_gr_scatter_norm_inj_bf16_v2<8><<<4 * ((P + 7) / 8), 256, 0, g_str>>>(
              pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
      } else if (scat_v2 && g_cfg.d == 2560 && P > 8) {
        if (write_done)
          k_gr_scatter_norm_inj_bf16_v2<4, false><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
              nullptr, nullptr, (uint16_t*)d_Rb, nullptr, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
        else
          k_gr_scatter_norm_inj_bf16_v2<4><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
              pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt], P,
              g_cfg.norm_eps, sc);
      } else if (scat_t1 && P <= 8 && !write_done)
        k_gr_scatter_norm_inj_bf16<1><<<4 * P, 256, 0, g_str>>>(
            pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt],
            g_cfg.d, P, g_cfg.norm_eps, sc);
      else if (write_done)
        k_gr_scatter_norm_inj_bf16<4, false><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
            nullptr, nullptr, (uint16_t*)d_Rb, nullptr, hw, rh, wn,
            d_injp[nxt], g_cfg.d, P, g_cfg.norm_eps, sc);
      else
        k_gr_scatter_norm_inj_bf16<4><<<4 * ((P + 3) / 4), 256, 0, g_str>>>(
            pfull, pin, (uint16_t*)d_Rb, y, hw, rh, wn, d_injp[nxt],
            g_cfg.d, P, g_cfg.norm_eps, sc);
      injp_cur = nxt;
      injp_for = read_prefix;
    } else if (gr_bf16)
      if (gr_scat4 && g_cfg.d <= 2560)
        k_gr_scatter_norm_b_hc_bf16_f4<<<P, 256, 0, g_str>>>(
            d_w4b, (uint16_t*)d_Rb, d_yb, hw, d_Rhatbf16, g_cfg.d,
            g_cfg.branches, P, g_cfg.norm_eps);
      else
        k_gr_scatter_norm_b_hc_bf16<<<4 * P, 256, 0, g_str>>>(
            d_w4b, (uint16_t*)d_Rb, d_yb, hw, d_Rhatbf16, g_cfg.d,
            g_cfg.branches, P, g_cfg.norm_eps);
    else if (prefill_fused)
      k_gr_scatter_norm_b<true><<<4 * P, 1024, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, hw, d_Rhatb, g_cfg.d, g_cfg.branches, P,
          g_cfg.norm_eps, d_Rhatbf16);
    else
      k_gr_scatter_norm_b<<<4 * P, 1024, 0, g_str>>>(
          d_w4b, d_Rb, d_yb, hw, d_Rhatb, g_cfg.d, g_cfg.branches, P,
          g_cfg.norm_eps);
    if (!wn) injp_cur = -1;
    if (xf16_src == d_Rhatb) xf16_src = nullptr;
    gr_mix_b(read_prefix, P, hw, wn && hc_norhat(P));
    df2_hc_capture(write_prefix, P);
  }

  // GDN: projections batched. Chunked path: conv/l2norm/gates batched, then
  // k_gdn_chunk (chunk=64, fp32) advances S and writes all outputs at once.
  // QWENOX_GDN_LOOP=1 falls back to the per-token recurrence loop.
  // GDN in/out projections over P rows starting at the given workspace rows.
  // Row-invariant gemv (ktest ri_*): a row's result does not depend on P, so
  // one R-row call equals per-segment calls bit for bit.
  void gdn_proj_in_b(int l, int P, float* xb, float* qkvb, float* g48b,
                     float* beta48b, float* zb) {
    char buf[128];
    KP_SUB("gdn:proj");
    int npu_z_nr = 0;  // M3：NPU 覆盖的 z 行数（0 = 全 GPU）
#ifdef QWENOX_NPU
    npu_z_nr = npu_z_submit(l, P, xb);  // 先提交，NPU 与下面三个 GEMM 重叠
#endif
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_qkv.weight", l);
    gemm(wview(ckpt.at(buf)), xb, qkvb, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_a.weight", l);
    gemm(wview(ckpt.at(buf)), xb, g48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_b.weight", l);
    gemm(wview(ckpt.at(buf)), xb, beta48b, P);
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
#ifdef QWENOX_NPU
    if (npu_z_nr > 0) {
      if (P > npu_z_nr)  // NPU 固定算满 4096 行，剩余行 GPU 补（GEMM 行独立）
        gemm(wview(ckpt.at(buf)), xb + (size_t)npu_z_nr * 2560,
             zb + (size_t)npu_z_nr * 6144, P - npu_z_nr);
      npu_pend_z_l_ = l;  // 收尾推迟到 conv+scan 之后（zb 首个消费点前）
      npu_pend_z_P_ = P;
    } else
#endif
    gemm(wview(ckpt.at(buf)), xb, zb, P);
    KP_SUB_END();
  }
  void gdn_proj_out_b(int l, int P, float* convb, float* yb, uint16_t* xbf16b) {
    char buf[128];
    KP_SUB("gdn:oproj");
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
    gemm(wview(ckpt.at(buf)), convb + 4096, yb, P, 10240 /*row stride*/,
         xbf16b);
    KP_SUB_END();
  }
  // Batched-verify GDN state ops for all segments of one layer: the same
  // kernel sequence gdn_b(state_only) runs per segment on the spec-verify
  // path (conv+l2n, conv snap, conv update, gates, S snap, k_gdn_verify,
  // gated norm), each as ONE launch over all segments (blockIdx picks the
  // segment; bodies shared with the per-segment kernels -> bit-identical).
  // Returns false (caller runs the per-segment loop) unless every segment is
  // on that path. Projections are the caller's (gdn_proj_in_b/out_b).
  bool gdn_state_vseg(int l, int gi, const int* Ps, const int* sl,
                      const int* voffs, int nseg, int R) {
    static const bool off = env_zero("QWENOX_VSEG_GDN_BATCH");
    static const bool convl2 = !env_zero("QWENOX_GDN_CONVL2");
    if (off || gdn_loop || !convl2 || ngram_legacy_gdn || nseg < 2 || R > 64)
      return false;
    const int ngdn = g_cfg.layers - g_cfg.layers / 4;
    GdnVSegs sg{};
    sg.n = nseg;
    for (int s = 0; s < nseg; s++) {
      const int t = sl[s];
      if (!slot_spec_snap(t) || slot_ngram_verify_active(t) || slot_cut_r_(t) ||
          Ps[s] > 65)
        return false;
      const int vrow0 = slot_vsnap_row0_(t);
      GdnVSeg& g = sg.s[s];
      g.convst = slot_d_convst(t) + (size_t)gi * 3 * 10240;
      g.S = slot_d_S(t) + (size_t)gi * 48 * 128 * 128;
      g.csnap = d_csnap + ((size_t)vrow0 * ngdn + gi) * 3 * 10240;
      g.r0 = voffs[s];
      g.P = Ps[s];
      for (int r = 0; r < Ps[s]; r++) sg.rowseg[voffs[s] + r] = (signed char)s;
    }
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    const float* Alog = d_Alog + (size_t)gi * 48;
    const float* dtb = d_dtbias + (size_t)gi * 48;
    const float* gnw = d_gdnnorm + (size_t)gi * 128;
    const float qscale = 1.f / sqrtf(128.f);
    KP_SUB("gdn:conv");
    k_gdn_conv_l2n_vseg<<<dim3(10, (unsigned)R), 256, 0, g_str>>>(
        d_qkvb, cw, d_convb, sg, 10240, 16, qscale);
    k_convst_snap_vseg<<<(unsigned)R, 512, 0, g_str>>>(
        d_qkvb, sg, 10240, (size_t)ngdn * 3 * 10240);
    k_convst_update_vseg<<<dim3((10240 + 255) / 256, (unsigned)nseg), 256, 0,
                           g_str>>>(d_qkvb, sg, 10240);
    k_gdn_gates<<<(unsigned)R, 64, 0, g_str>>>(d_g48b, d_beta48b, Alog, dtb,
                                               d_g48b, d_beta48b, 48, 48);
    KP_SUB_END();
    // replay inputs of every row (batch row r -> d_grow row r: segment s's
    // vsnap_row0_ == its batch offset), then the read-only verify
    k_gdn_rowsave<<<(unsigned)R, 256, 0, g_str>>>(
        d_convb, d_g48b, d_beta48b, d_grow + (size_t)gi * kGdnRowF,
        (size_t)ngdn * kGdnRowF, R);
    KP_SUB("gdn:scan");
    k_gdn_verify_vseg<<<dim3(48, (unsigned)nseg), 512, 0, g_str>>>(
        d_convb, d_g48b, d_beta48b, sg, 10240);
    KP_SUB_END();
#ifdef QWENOX_NPU
    npu_z_pend_finish(l, R, d_xb, d_zb);  // M3：合批 verify 路径同样收尾
#endif
    KP_SUB("gdn:norm");
    k_gdn_gatednorm_b<true><<<dim3(48, (unsigned)((R + 31) / 32)), 128, 0, g_str>>>(
        d_convb + 4096, d_zb, gnw, nullptr, g_cfg.norm_eps, 10240, 6144, d_xbf16b,
        R);
    KP_SUB_END();
    return true;
  }
  // state_only (batched verify): the caller runs the in/out projections over
  // all R rows at once (gdn_proj_in_b / gdn_proj_out_b); this call does only
  // the segment's conv/gates/scan/gatednorm on its slot's state.
  void gdn_b(int l, int gi, int P, int slot = -1, int r0 = 0,
             bool state_only = false) {
    char buf[128];
    // D2-S1c: slot >= 0 runs the segment on that slot's state (accessors
    // resolve bound_ back to the members); r0 shifts the batch-row bases of
    // the [BP] workspaces. slot == -1 && r0 == 0 is the old call, unchanged.
    const int bs = slot < 0 ? bound_ : slot;
    float* const xb = d_xb + (size_t)r0 * 2560;
    float* const qkvb = d_qkvb + (size_t)r0 * 10240;
    float* const g48b = d_g48b + (size_t)r0 * 48;
    float* const beta48b = d_beta48b + (size_t)r0 * 48;
    float* const zb = d_zb + (size_t)r0 * 6144;
    float* const convb = d_convb + (size_t)r0 * 10240;
    float* const yb = d_yb + (size_t)r0 * 2560;
    uint16_t* const xbf16b = d_xbf16b + (size_t)r0 * 6144;
    const bool usnap = slot_spec_snap(bs);
    const bool ung = slot_ngram_verify_active(bs);
    const int vrow0 = slot_vsnap_row0_(bs);
    const int cutr = slot_cut_r_(bs);
    // D2-S1b bisect: QWENOX_VGDUMP_GL=<layer> 时对该 GDN 层逐阶段打行 hash。
    const int vgl = [] {
      const char* e = getenv("QWENOX_VGDUMP_GL");
      return e ? atoi(e) : -1;
    }();
    auto vgd_g = [&](const char* tag, const void* p, size_t bytes) {
      if (l == vgl) vg_dump_buf(tag, P, p, bytes);
    };
    if (!state_only) gdn_proj_in_b(l, P, xb, qkvb, g48b, beta48b, zb);
    vgd_g("gq", qkvb, (size_t)P * 10240 * 4);
    vgd_g("ga", g48b, (size_t)P * 48 * 4);
    vgd_g("bt", beta48b, (size_t)P * 48 * 4);
#ifdef QWENOX_NPU
    if (vgl == l) npu_z_pend_finish(l, P, xb, zb);  // dump zb 前强制收尾
#endif
    vgd_g("zb", zb, (size_t)P * 6144 * 4);
    float* convst = slot_d_convst(bs) + (size_t)gi * 3 * 10240;
    const float* cw = d_convw + (size_t)gi * 10240 * 4;
    const float* Alog = d_Alog + (size_t)gi * 48;
    const float* dtb = d_dtbias + (size_t)gi * 48;
    float* S = slot_d_S(bs) + (size_t)gi * 48 * 128 * 128;
    const float* gnw = d_gdnnorm + (size_t)gi * 128;
    float qscale = 1.f / sqrtf(128.f);
    if (!gdn_loop) {
      KP_SUB("gdn:conv");
      // conv+silu and the q/k l2norm in one pass (k_gdn_conv_l2n_b, bit-exact
      // vs the pair, tools/convl2_verify.sh); opt-out QWENOX_GDN_CONVL2=0.
      static const bool convl2 = !env_zero("QWENOX_GDN_CONVL2");
      const bool cl2 = convl2 && P <= 65535;
      int64_t tot = (int64_t)P * 10240 / 4;  // 4 channels per thread (float4)
      if (cl2)
        k_gdn_conv_l2n_b<<<dim3(10, (unsigned)P), 256, 0, g_str>>>(
            qkvb, cw, convst, convb, P, 10240, 16, qscale);
      else
        k_gdn_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
            qkvb, cw, convst, convb, P, 10240);
      if (usnap) {
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_convst_snap<<<P, 512, 0, g_str>>>(
            qkvb, convst,
            d_csnap + ((size_t)vrow0 * ngdn + gi) * 3 * 10240, P, 10240,
            (size_t)ngdn * 3 * 10240);
      }
      if (cutr)  // in-pass cut: conv state after the first r rows
        k_convst_update<<<(10240 + 255) / 256, 256, 0, g_str>>>(
            d_cconvst + (size_t)gi * 3 * 10240, qkvb, cutr, 10240);
      k_convst_update<<<(10240 + 255) / 256, 256, 0, g_str>>>(convst, qkvb, P,
                                                              10240);
      if (!cl2)
        k_l2norm_qk_b<<<dim3(16, P), 128, 0, g_str>>>(convb, 128, qscale, 10240, 16);
      k_gdn_gates<<<P, 64, 0, g_str>>>(g48b, beta48b, Alog, dtb, g48b, beta48b,
                                       48, 48);
      KP_SUB_END();
      vgd_g("gc", convb, (size_t)P * 10240 * 4);
      vgd_g("ga2", g48b, (size_t)P * 48 * 4);
      vgd_g("bt2", beta48b, (size_t)P * 48 * 4);
      // D2-S1b: whole-state hash of S at scan entry (scalar line — row mode
      // would split it differently per P). Tells apart "S entered the round
      // different" (rollback path) from "scan kernel diverged mid-batch".
      if (l == vgl && !vg_capturing_ && getenv("QWENOX_VGDUMP")) {
        CK(hipStreamSynchronize(g_str));
        static std::vector<uint8_t> sh;
        sh.resize((size_t)48 * 128 * 128 * 4);
        CK(hipMemcpy(sh.data(), S, sh.size(), hipMemcpyDeviceToHost));
        uint64_t fnv = 1469598103934665603ULL;
        for (uint8_t b : sh) { fnv ^= b; fnv *= 1099511628211ULL; }
        fprintf(stderr, "[sS] P=%d base=%d bufhash=%016llx\n", P, moe_dump_base_,
                (unsigned long long)fnv);
      }
      KP_SUB("gdn:scan");
      // D2-S1b: spec verify rounds (spec_snap) always take k_gdn_verify.
      // Its per-row state recurrence is bit-identical to k_gdn_snap's (same
      // 512-thread layout, same reduction order), so the committed S has a
      // single provenance whether the round full-accepts (scan write-back)
      // or rolls back (snap restore). The fused/stream kernels are fp16-WMMA
      // / chunked-reassociated: their write-back differs from snap in LSBs,
      // and acceptance-pattern differences across gamma then leak into the
      // committed state (gs fork at abs 8196, γ2 P=3 vs γ3 P=4).
      // 10-12: spec verify no longer snapshots S per row (P x 3.1 MB per
      // layer). It saves the rows' replay inputs (k|v|g|beta, 33 KB/row) and
      // leaves S at the round start; gdn_commit(d+1) replays the committed
      // rows through the same recurrence body (gdn_rec_body) once the
      // acceptance is known -- rollback and full accept alike.
      if (usnap && P <= 65) {
        const int ngdn = g_cfg.layers - g_cfg.layers / 4;
        k_gdn_rowsave<<<P, 256, 0, g_str>>>(
            convb, g48b, beta48b, d_grow + ((size_t)vrow0 * ngdn + gi) * kGdnRowF,
            (size_t)ngdn * kGdnRowF, P);
        k_gdn_verify_ro<<<48, 512, 0, g_str>>>(convb, g48b, beta48b, S, convb, P, 10240);
      } else if (ung && !ngram_legacy_gdn && P <= 65) {
        k_gdn_verify<<<48, 512, 0, g_str>>>(
            convb, g48b, beta48b, S, convb, P, 10240);
      } else if (gdn_nosplit) {
        k_gdn_chunk<<<48, GDN_NT, 0, g_str>>>(convb, g48b, beta48b, S, convb,
                                              d_gdn_ws, P, 10240);
      } else if (gdn_fused) {
        // one launch for all of P: no window, no workspace; S in/out via global
        const int r = cutr;
        float* const cS = d_cS ? d_cS + (size_t)gi * 48 * 128 * 128 : nullptr;
        if (r > 0 && r < P) {
          // in-pass cut: scan [0, r), keep S(cut), scan [r, P) (chunk grid
          // restarts at r -> not bit-identical to the single launch)
          k_gdn_fused<<<48, 1024, 0, g_str>>>(convb, g48b, beta48b, S,
                                              convb, r, 10240);
          CK(hipMemcpyAsync(cS, S, (size_t)48 * 128 * 128 * 4, hipMemcpyDeviceToDevice, g_str));
          k_gdn_fused<<<48, 1024, 0, g_str>>>(
              convb + (size_t)r * 10240, g48b + (size_t)r * 48, beta48b + (size_t)r * 48,
              S, convb + (size_t)r * 10240, P - r, 10240);
        } else {
          k_gdn_fused<<<48, 1024, 0, g_str>>>(convb, g48b, beta48b, S,
                                              convb, P, 10240);
          if (r == P)
            CK(hipMemcpyAsync(cS, S, (size_t)48 * 128 * 128 * 4, hipMemcpyDeviceToDevice, g_str));
        }
      } else {
        // Keep producer and consumer close; S carries the exact recurrence
        // across windows and the last window alone can contain a padded chunk.
        int window = gdn_stream ? 64 * gdn_window_chunks : P;
        // QWENOX_GDN_PIPE (strip path): intra stays on g_str, strips run on a
        // side stream. ws is double-buffered by window parity: intra(w) waits
        // for strip(w-2) (its ws slot's previous reader), strip(w) waits for
        // intra(w) and stays stream-serial (the S recurrence). Event objects
        // are reused by parity; host enqueue order matches dependency order.
        const bool pipe = gdn_pipe && gdn_stream && !gdn_nostrip;
        static hipStream_t pipe_str = nullptr;
        static hipEvent_t ev_i[2] = {nullptr, nullptr}, ev_s[2] = {nullptr, nullptr};
        if (pipe && !pipe_str) {
          CK(hipStreamCreateWithFlags(&pipe_str, hipStreamNonBlocking));
          for (int i = 0; i < 2; i++) {
            CK(hipEventCreateWithFlags(&ev_i[i], hipEventDisableTiming));
            CK(hipEventCreateWithFlags(&ev_s[i], hipEventDisableTiming));
          }
        }
        const size_t ws_slot =
            (size_t)std::min<size_t>(gdn_window_chunks, (maxbatch_cap + 63) / 64) *
            48 * GDN_SPLIT_WS_FLOATS;
        int w = 0, lastw = -1;
        for (int first = 0; first < P; first += window, w++) {
          int count = std::min(window, P - first), nchunks = (count + 63) / 64;
          float* conv = convb + (size_t)first * 10240;
          float* ws = d_gdn_split_ws + (pipe ? (w & 1) * ws_slot : 0);
          if (pipe && w >= 2) CK(hipStreamWaitEvent(g_str, ev_s[w & 1], 0));
          if (gdn_pipe2)
            k_gdn_intra_p2<<<dim3((nchunks + GDN_PIPE2_NCPB - 1) / GDN_PIPE2_NCPB, 48),
                             1024, 0, g_str>>>(
                conv, g48b + first * 48, beta48b + first * 48,
                ws, count, 10240);
          else if (gdn_wave)
            if (gdn_ut5)
              k_gdn_intra<1024, true, true><<<dim3(nchunks, 48), 1024, 0, g_str>>>(
                  conv, g48b + first * 48, beta48b + first * 48,
                  ws, count, 10240);
            else
              k_gdn_intra<1024, true><<<dim3(nchunks, 48), 1024, 0, g_str>>>(
                  conv, g48b + first * 48, beta48b + first * 48,
                  ws, count, 10240);
          else
            k_gdn_intra<GDN_NT_INTRA><<<dim3(nchunks, 48), GDN_NT_INTRA, 0, g_str>>>(
                conv, g48b + first * 48, beta48b + first * 48,
                ws, count, 10240);
          if (!pipe) {
            if (gdn_nostrip)
              k_gdn_inter<GDN_NT_INTER><<<48, GDN_NT_INTER, 0, g_str>>>(
                  conv, S, conv, ws, count, 10240);
            else
              k_gdn_inter_strip<<<dim3(4, 48), GDN_NT_STRIP, 0, g_str>>>(
                  conv, S, conv, ws, count, 10240);
          } else {
            CK(hipEventRecord(ev_i[w & 1], g_str));
            CK(hipStreamWaitEvent(pipe_str, ev_i[w & 1], 0));
            k_gdn_inter_strip<<<dim3(4, 48), GDN_NT_STRIP, 0, pipe_str>>>(
                conv, S, conv, ws, count, 10240);
            CK(hipEventRecord(ev_s[w & 1], pipe_str));
            lastw = w;
          }
        }
        if (pipe) CK(hipStreamWaitEvent(g_str, ev_s[lastw & 1], 0));
      }
      KP_SUB_END();
#ifdef QWENOX_NPU
      npu_z_pend_finish(l, P, xb, zb);  // M3：conv+scan 已盖过 NPU，收尾 z
#endif
      KP_SUB("gdn:norm");
      k_gdn_gatednorm_b<true><<<dim3(48, (P + 31) / 32), 128, 0, g_str>>>(
          convb + 4096, zb, gnw, nullptr, g_cfg.norm_eps, 10240, 6144,
          xbf16b, P);
      KP_SUB_END();
      vgd_g("gs", convb, (size_t)P * 10240 * 4);
      // d_xbf16b rows are 48*128 = 6144 bf16 (gatednorm writes (t*48+h)*128).
      vgd_g("gz", xbf16b, (size_t)P * 6144 * 2);
      if (!state_only) gdn_proj_out_b(l, P, convb, yb, xbf16b);
      vgd_g("go", yb, (size_t)P * 2560 * 4);
      return;
    }
    if (state_only)
      throw std::runtime_error("gdn_b: state_only needs the batched (non-loop) path");
#ifdef QWENOX_NPU
    npu_z_pend_finish(l, P, xb, zb);  // M3：loop 路径消费 zb 前收尾
#endif
    for (int t = 0; t < P; t++) {
      float* qkv = qkvb + (size_t)t * 10240;
      k_gdn_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(qkv, cw, convst, qkv, 10240);
      k_l2norm_qk<<<16, 128, 0, g_str>>>(qkv, qkv + 2048, 128, qscale);
      k_gdn_gates<<<1, 64, 0, g_str>>>(g48b + t * 48, beta48b + t * 48, Alog, dtb,
                                       g48b + t * 48, beta48b + t * 48, 48, 48);
      k_gdn_step<<<48, 512, 0, g_str>>>(qkv, qkv + 2048, qkv + 4096, g48b + t * 48,
                                        beta48b + t * 48, S, qkv + 4096, 128, 128, 48,
                                        16);
      k_gdn_gatednorm<<<dim3(48, 1), 128, 0, g_str>>>(qkv + 4096,
                                                      zb + (size_t)t * 6144, gnw,
                                                      qkv + 4096, 128, g_cfg.norm_eps,
                                                      10240, 6144);
    }
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.out_proj.weight", l);
    gemm(wview(ckpt.at(buf)), qkvb + 4096, yb, P, 10240 /*row stride*/);
  }

  // Debug (QWENOX_ATTN_HASH): per prefill chunk and QSA layer, hashes of the
  // attention inputs (K / V / transposed V over valid rows [0, base+P), the
  // transposed-V tail slots [base+P, roundup4), q, selections) and the output.
  void dbg_attn_hash(int qi, int base, int P, const uint16_t* kc, const uint16_t* vc,
                     const uint16_t* vct) {
    static unsigned long long* d_h = nullptr;
    if (!d_h) CK(hipMalloc(&d_h, 8 * sizeof(unsigned long long)));
    CK(hipMemsetAsync(d_h, 0, 8 * sizeof(unsigned long long), g_str));
    const int r1 = base + P, r2 = (r1 + 3) & ~3, first = std::max(base, 2051);
    auto g = [](uint64_t n) { return (unsigned)((n + 255) / 256); };
    k_dbg_hash_rows<<<g((uint64_t)r1 * 512), 256, 0, g_str>>>(kc, nullptr, kvpt(), 0, r1, d_h + 0);
    k_dbg_hash_rows<<<g((uint64_t)r1 * 512), 256, 0, g_str>>>(vc, nullptr, kvpt(), 0, r1, d_h + 1);
    if (vct) {
      k_dbg_hash_rows<<<g((uint64_t)r1 * 512), 256, 0, g_str>>>(nullptr, vct, kvpt(), 0, r1, d_h + 2);
      if (r2 > r1)
        k_dbg_hash_rows<<<g((uint64_t)(r2 - r1) * 512), 256, 0, g_str>>>(nullptr, vct, kvpt(), r1, r2,
                                                                         d_h + 3);
    }
    k_dbg_hash_f32<<<g((uint64_t)P * 6144), 256, 0, g_str>>>(d_qsb, (uint64_t)P * 6144, d_h + 4);
    if (base + P > first)
      k_dbg_hash_i32<<<g((uint64_t)(base + P - first) * 512), 256, 0, g_str>>>(
          d_iselected + (size_t)(first - base) * 512, (uint64_t)(base + P - first) * 512, d_h + 5);
    k_dbg_hash_f32<<<g((uint64_t)P * 6144), 256, 0, g_str>>>(d_attnb, (uint64_t)P * 6144, d_h + 6);
    unsigned long long h[8];
    CK(hipStreamSynchronize(g_str));
    CK(hipMemcpy(h, d_h, sizeof h, hipMemcpyDeviceToHost));
    fprintf(stderr,
            "[ahash] base=%d P=%d qsa%d K=%016llx V=%016llx Vt=%016llx tail=%016llx "
            "q=%016llx sel=%016llx out=%016llx\n",
            base, P, qi, h[0], h[1], h[2], h[3], h[4], h[5], h[6]);
  }

  // QSA prefill flash over the KV cache: dense prefix plus the sparse tail.
  // base = absolute position of batch row 0 (0 for a fresh prefill; > 0 when
  // continuing from live KV/state). Dense rows are the batch-local prefix,
  // sparse queries are the absolute positions >= 2051 within the batch.
  template <typename KVT>
  void qsa_flash_b(const KVT* kc, const KVT* vc, int P, int dense_P, int base = 0,
                   const uint16_t* vc_transposed = nullptr,
                   const int* ptab = nullptr, const int* dvb = nullptr,
                   int row0 = 0, int spec_ovr = -1) {
    // D2-S1c: row0 shifts the batch-local row bases of the [BP] workspaces
    // (d_qsb/d_attnb/d_iselected) so one segment of a multi-segment verify
    // batch reads/writes only its own rows; spec_ovr >= 0 overrides the
    // spec_snap gate (per-slot flag). row0 == 0 && spec_ovr == -1 reproduces
    // the old single-segment behaviour exactly.
    float* const qsb = d_qsb + (size_t)row0 * 6144;
    float* const attnb = d_attnb + (size_t)row0 * 6144;
    const int* const iselb = d_iselected + (size_t)row0 * 512;
    const bool usnap = spec_ovr >= 0 ? spec_ovr != 0 : spec_snap;
    // dvb (P3-B graph capture): the kernels read base from *dvb and recompute
    // first_abs / ga / ge on device. Requires base >= 2051 (the verify graph
    // gate), so first_abs == base, nsparse == P and sel is the batch-local
    // row-0 pointer; grids are baked worst-case (edges: 3 tokens, union
    // groups: max over base mod 4) and masked in-kernel.
    // ptab (paged KV) is honoured by k_qsa_flash (all KV types) and, for the
    // BF16 cache, by k_qsa_wmma<*, true> (A1c). The union / wmma6 /
    // k_qsa_flash_bf16 paths are not paged: the constructor refuses paging
    // for those configurations.
    if (dense_P > 0) {
      // F1 (§11.1)：bf16 KV + qsa_wmma 时稠密前缀也走 k_qsa_wmma 的 Dense 变体
      // （grid=(dense_P,2)，first=kbase=base，selected 不用）。fp32 KV、
      // QWENOX_QSA_DENSE（dense_P=P 全稠密）保持原 fp32 标量路径。
      bool dense_launched = false;
      if constexpr (sizeof(KVT) == 2) {
        if (qsa_wmma && !qsa_dense) {
          dense_launched = true;
          if (vc_transposed) {
            if (ptab)
              k_qsa_wmma<true, true, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  qsb, kc, vc, vc_transposed, attnb, dense_P, 256, 2, 24,
                  nullptr, base, base, ptab, dvb);
            else
              k_qsa_wmma<true, false, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  qsb, kc, vc, vc_transposed, attnb, dense_P, 256, 2, 24,
                  nullptr, base, base, nullptr, dvb);
          } else {
            if (ptab)
              k_qsa_wmma<false, true, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  qsb, kc, vc, nullptr, attnb, dense_P, 256, 2, 24, nullptr,
                  base, base, ptab, dvb);
            else
              k_qsa_wmma<false, false, true><<<dim3(dense_P, 2), 256, 0, g_str>>>(
                  qsb, kc, vc, nullptr, attnb, dense_P, 256, 2, 24, nullptr,
                  base, base, nullptr, dvb);
          }
        }
      }
      if (!dense_launched)
        k_qsa_flash<false, false, KVT><<<dim3((dense_P + 15) / 16, 24), 128, 0, g_str>>>(
            qsb, kc, vc, attnb, dense_P, 256, 2, 24, nullptr, 0, nullptr, nullptr,
            nullptr, base, ptab, dvb);
    }
    int first_abs = std::max(base, 2051), nsparse = base + P - first_abs;
    if (dvb) nsparse = P;  // gate: base >= 2051 -> first_abs == base
    if (!qsa_dense && nsparse > 0) {
      // d_iselected rows are batch-local (row = abs - base; A1b)
      const int* sel = dvb ? iselb
                           : iselb + (size_t)(first_abs - base) * 512;
      if (qsa_global_v)
        k_qsa_flash<true, false, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            qsb, kc, vc, attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base, ptab, dvb);
      else if constexpr (sizeof(KVT) == 2) {
        // spec verify (spec_snap): force the per-token kernel. Union group
        // membership depends on the batch end (ge = (base+P) & ~3), so the
        // same row lands in a group at P=4 but on a ragged edge at P=3 —
        // and the union kernel is NOT bit-identical to the per-token one
        // (key order). Row invariance across gamma requires one kernel.
        // ngram 验证和拒绝后的前缀重放也必须保持逐 token 算术。
        if (qsa_union && vc_transposed && !usnap && !ngram_verify_active) {
          // 并集路径：[ga, ge) 内的 4 对齐完整组走 merge + k_qsa_wmma_u，
          // 两侧 ragged 边缘回退逐 token 的 k_qsa_wmma<true>
          const int ga = (first_abs + 3) & ~3;
          const int ge = (base + P) & ~3;
          int ngroups = (ge - ga) >> 2;
          if (dvb) {  // baked worst case over base mod 4; in-kernel ge guard
            ngroups = 0;
            for (int r = 0; r < 4; r++)
              ngroups = std::max(ngroups, (((r + P) & ~3) - ((r + 3) & ~3)) >> 2);
          }
          if (ngroups > 0) {
            k_qsa_union_merge<<<ngroups, 128, 0, g_str>>>(
                iselb, ga, d_ublk, d_umask, d_ucount, QSA_UCAP, base, P,
                dvb);
            if (ptab)
              k_qsa_wmma_u<true><<<dim3(ngroups, 2), 256, 0, g_str>>>(
                  qsb, kc, vc_transposed, attnb, 256, 2, 24, d_ublk, d_umask,
                  d_ucount, QSA_UCAP, ga, base, ptab, P, dvb);
            else
              k_qsa_wmma_u<<<dim3(ngroups, 2), 256, 0, g_str>>>(
                  qsb, kc, vc_transposed, attnb, 256, 2, 24, d_ublk, d_umask,
                  d_ucount, QSA_UCAP, ga, base, nullptr, P, dvb);
          }
          // ragged edges: per-token kernel (selected rows relative to its
          // own first token -> offset sel)
          auto edge = [&](int f, int n) {
            const int* es = sel + (size_t)(f - first_abs) * 512;
            if (ptab)
              k_qsa_wmma<true, true><<<dim3(n, 2), 256, 0, g_str>>>(
                  qsb, kc, vc, vc_transposed, attnb, P, 256, 2, 24, es, f, base, ptab);
            else
              k_qsa_wmma<true><<<dim3(n, 2), 256, 0, g_str>>>(
                  qsb, kc, vc, vc_transposed, attnb, P, 256, 2, 24, es, f, base);
          };
          if (dvb) {
            // baked 3-token grids for both edges; kernels mask by *dvb
            // (dvb_edge 2 = left [first_abs, ga), 1 = right [ge, base+P))
            for (int ek = 2; ek >= 1; ek--) {
              if (ptab)
                k_qsa_wmma<true, true><<<dim3(3, 2), 256, 0, g_str>>>(
                    qsb, kc, vc, vc_transposed, attnb, P, 256, 2, 24,
                    iselb, 0, 0, ptab, dvb, ek);
              else
                k_qsa_wmma<true><<<dim3(3, 2), 256, 0, g_str>>>(
                    qsb, kc, vc, vc_transposed, attnb, P, 256, 2, 24,
                    iselb, 0, 0, nullptr, dvb, ek);
            }
          } else {
            if (ga > first_abs) edge(first_abs, std::min(ga, base + P) - first_abs);
            if (base + P > ge && ge >= ga) edge(ge, base + P - ge);
          }
        }
        else if (qsa_wmma6 && vc_transposed)
          k_qsa_wmma6<<<dim3(nsparse, 2), 128, 0, g_str>>>(
              qsb, kc, vc, vc_transposed, attnb, P, 256, 2, 24, sel,
              first_abs, base);
        else if (qsa_wmma_btv && vc_transposed) {
          if (ptab)  // A1c paged instantiation (never taken unpaged)
            k_qsa_wmma<true, true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                qsb, kc, vc, vc_transposed, attnb, P, 256, 2, 24, sel, first_abs,
                base, ptab, dvb);
          else
            k_qsa_wmma<true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                qsb, kc, vc, vc_transposed, attnb, P, 256, 2, 24, sel, first_abs,
                base, nullptr, dvb);
        } else if (qsa_wmma) {
          // QWENOX_QSA_WMMA=1: WMMA bf16 sparse flash (256 thr), see k_qsa_wmma.
          if (ptab)
            k_qsa_wmma<false, true><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                qsb, kc, vc, nullptr, attnb, P, 256, 2, 24, sel, first_abs, base,
                ptab, dvb);
          else
            k_qsa_wmma<false><<<dim3(nsparse, 2), 256, 0, g_str>>>(
                qsb, kc, vc, nullptr, attnb, P, 256, 2, 24, sel, first_abs, base,
                nullptr, dvb);
        } else
          // BF16 KV mode (!qsa_kv_fp32): bf16 LDS tiles + fdot2 (-61% at 32K).
          k_qsa_flash_bf16<KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
              qsb, kc, vc, attnb, P, 256, 2, 24, sel, first_abs, base, dvb);
      }
      else
        k_qsa_flash<true, true, KVT><<<dim3(nsparse, 2), 128, 0, g_str>>>(
            qsb, kc, vc, attnb, P, 256, 2, 24, sel, first_abs, nullptr, nullptr,
            nullptr, base, ptab, dvb);
    }
  }

  // Copy P rows (512 floats) for absolute positions [base, base+P) from a
  // contiguous batch buffer into one layer of the FP32 KV cache. Unpaged: one
  // bulk copy (legacy behaviour). Paged: one copy per physically contiguous
  // run of pages (a single copy for the identity table).
  void kv_rows_store(float* layer, const float* src, int base, int P) {
    kv_rows_store_seq(kv_seq, layer, src, base, P);
  }
  // D2-S1c: sequence-explicit variant (multi-segment verify stores into a
  // non-bound slot's page table).
  void kv_rows_store_seq(const KvSeqTable& seq, float* layer, const float* src,
                         int base, int P) {
    kv_runs_seq(seq, base, base + P, KV_PAGE, [&](int lr, size_t pr, int n) {
      CK(hipMemcpyAsync(layer + pr * 512, src + (size_t)(lr - base) * 512,
                        (size_t)n * 512 * 4, hipMemcpyDeviceToDevice, g_str));
    });
  }

  // Visit logical rows [r0, r1) of one paged per-layer region as physically
  // contiguous runs: fn(first_logical_row, first_physical_row, nrows).
  // per_page = KV_PAGE for KV rows, KV_PAGE_BLOCKS for indexer key blocks.
  // Unpaged: a single identity run. Paged: requires r1 <= mapped pages *
  // per_page (A2: rows beyond the live mapping have no backing page).
  template <typename F>
  void kv_runs(int r0, int r1, int per_page, F&& fn) const {
    kv_runs_seq(kv_seq, r0, r1, per_page, fn);
  }
  // D2-S1c: sequence-explicit variants; the member wrappers above delegate
  // with (kv_seq, d_ptab), so the bound-slot path is byte-identical.
  template <typename F>
  void kv_runs_seq(const KvSeqTable& seq, int r0, int r1, int per_page, F&& fn) const {
    if (r1 <= r0) return;
    if (!kv_paged) {
      fn(r0, (size_t)r0, r1 - r0);
      return;
    }
    if ((size_t)r1 > (size_t)seq.mapped * per_page) {
      fprintf(stderr, "[kvpage] FATAL: kv_runs [%d, %d) x%d beyond the %d mapped pages\n",
              r0, r1, per_page, seq.mapped);
      abort();
    }
    const std::vector<int>& h_ptab = seq.tab;
    for (int t = r0; t < r1;) {
      const int lp = t / per_page, pp = h_ptab[lp];
      int q = lp + 1;
      while ((size_t)q * per_page < (size_t)r1 && h_ptab[q] == pp + (q - lp)) q++;
      const int re = (int)std::min<size_t>((size_t)r1, (size_t)q * per_page);
      fn(t, (size_t)pp * per_page + (size_t)(t % per_page), re - t);
      t = re;
    }
  }

  // ---- A2: on-demand page mapping of the live sequence ----
  // Every device write to the trunk QSA KV / indexer keys goes through
  // prefill_chunk (rows [base, base+P)) or forward (row pos), and both call
  // kv_prepare first; the only other writer is kvsnap::restore, which
  // reserves fresh rows itself. New entries were guard entries that no
  // in-flight kernel reads, so the table upload needs no extra sync.
  // A3: drop the LRU pinned checkpoint; false when there is none left.
  bool kv_evict_one() {
    return (kv_evict_fn && kv_evict_fn(*this)) || (kv_starve_fn && kv_starve_fn(*this));
  }
  [[noreturn]] void kv_fatal(const char* what, int end) {
    kv_fatal_seq(what, end, kv_seq.mapped);
  }
  [[noreturn]] void kv_fatal_seq(const char* what, int end, int mapped) {
    fprintf(stderr,
            "[kvpage] FATAL: %s (%d tokens; pool %d pages, %d free, table %d "
            "pages, mapped %d)\n",
            what, end, kv_pool_pages, kv_pool.nfree(), kv_lpages, mapped);
    crashdiag::report_fatal("fatal", what);
    exit(1);
  }
  void kv_reserve(int end) { kv_reserve_seq(kv_seq, d_ptab, end); }
  void kv_reserve_seq(KvSeqTable& seq, int* ptab, int end) {
    if (!kv_paged || end <= seq.rows()) return;
    if ((end + KV_PAGE - 1) / KV_PAGE > kv_lpages)
      kv_fatal_seq("beyond maxctx", end, seq.mapped);
    const int m0 = seq.mapped;
    // pool dry: evict pinned checkpoints (LRU first) until the rows fit
    while (!seq.reserve(kv_pool, end))
      if (!kv_evict_one()) kv_fatal_seq("pool exhausted", end, seq.mapped);
    CK(hipMemcpy(ptab + m0, seq.tab.data() + m0,
                 (size_t)(seq.mapped - m0) * 4, hipMemcpyHostToDevice));
    if (kv_zero_dbg)  // debug: fresh pages read as zeros, like a zeroed unpaged cache
      for (int lp = m0; lp < seq.mapped; lp++)
        kv_page_regions_k(seq.tab[(size_t)lp], [&](void* q, size_t n, char c) {
          if (kv_zero_has(c)) CK(hipMemset(q, 0, n));
        });
    if (kv_mode == 2 && m0 == 0)
      fprintf(stderr, "[kvpage] seq map: logical 0 -> phys %d (free %d/%d)\n",
              seq.tab[0], kv_pool.nfree(), kv_pool_pages);
  }
  // A3: rows [start, end) are about to be written. Pages there that a pinned
  // checkpoint shares are copied first (in practice only the partial page
  // holding `start`), then the tail is mapped. Cheap host check otherwise.
  void kv_prepare(int start, int end) { kv_prepare_seq(kv_seq, d_ptab, start, end); }
  void kv_prepare_seq(KvSeqTable& seq, int* ptab, int start, int end) {
    if (!kv_paged) return;
    const int lp1 = std::min(seq.mapped, (end + KV_PAGE - 1) / KV_PAGE);
    for (int lp = std::max(0, start) / KV_PAGE; lp < lp1; lp++)
      if (kv_pool.ref[(size_t)seq.tab[(size_t)lp]] > 1) kv_cow_seq(seq, ptab, lp);
    kv_reserve_seq(seq, ptab, end);
  }
  void kv_cow(int lp) { kv_cow_seq(kv_seq, d_ptab, lp); }
  void kv_cow_seq(KvSeqTable& seq, int* ptab, int lp) {
    const int old = seq.tab[(size_t)lp];
    int p;
    while ((p = kv_pool.alloc()) < 0) {
      if (!kv_evict_one())
        kv_fatal_seq("pool exhausted (copy-on-write)", lp * KV_PAGE, seq.mapped);
      if (kv_pool.ref[(size_t)old] == 1) return;  // the other holder was evicted
    }
    CK(hipDeviceSynchronize());  // no in-flight kernel may still write `old`
    std::vector<std::pair<void*, size_t>> src, dst;
    kv_page_regions(old, [&](void* q, size_t n) { src.emplace_back(q, n); });
    kv_page_regions(p, [&](void* q, size_t n) { dst.emplace_back(q, n); });
    for (size_t i = 0; i < src.size(); i++)
      CK(hipMemcpy(dst[i].first, src[i].first, src[i].second, hipMemcpyDeviceToDevice));
    seq.replace(kv_pool, lp, p);
    CK(hipMemcpy(ptab + lp, &p, 4, hipMemcpyHostToDevice));
    kv_cow_count++;
    if (kv_mode == 2)
      fprintf(stderr, "[kvpage] cow logical %d: phys %d -> %d (free %d/%d)\n", lp,
              old, p, kv_pool.nfree(), kv_pool_pages);
  }
  long long kv_cow_count = 0;
  // QWENOX_KV_ZERO=1 (debug, A5 hunt): no row a sequence has not written holds
  // another sequence's data -- unpaged: the whole trunk KV/ik cache is zeroed
  // on reset_state; paged: every freshly mapped page is zeroed. If outputs
  // then agree across modes, some kernel reads rows >= pos and uses them.
  // debug: QWENOX_KV_ZERO=1 zeroes all of K/V/Vt/ik; or a subset of letters
  // k v t i (e.g. QWENOX_KV_ZERO=i), for both the unpaged and the paged zeroing.
  bool kv_zero_dbg = getenv("QWENOX_KV_ZERO") != nullptr;
  bool kv_zero_has(char c) {
    const char* e = getenv("QWENOX_KV_ZERO");
    return e && (!strcmp(e, "1") || strchr(e, c));
  }
  void kv_zero_unpaged() {
    const size_t nqsa = (size_t)(g_cfg.layers / 4), rows = (size_t)kv_nl * kv_layer_rows;
    if (qsa_kv_fp32) {
      if (kv_zero_has('k')) CK(hipMemset(d_kc, 0, rows * 512 * 4));
      if (kv_zero_has('v')) CK(hipMemset(d_vc, 0, rows * 512 * 4));
    } else {
      if (kv_zero_has('k')) CK(hipMemset(d_kcb, 0, rows * 512 * 2));
      if (kv_zero_has('v')) CK(hipMemset(d_vcb, 0, rows * 512 * 2));
      if (qsa_wmma_btv && kv_zero_has('t'))
        CK(hipMemset(d_vcbt, 0, nqsa * qsa_vt_blocks * 2 * 256 * 4 * 2));
    }
    if (d_ik && kv_zero_has('i')) CK(hipMemset(d_ik, 0, (size_t)kv_nl * ik_layer_blocks * 128 * 4));
    if (d_ikh && kv_zero_has('i')) CK(hipMemset(d_ikh, 0, (size_t)kv_nl * ik_layer_blocks * 128 * 2));
  }
  // A3: pin / unpin / adopt the live prefix for RAM checkpoints.
  std::vector<int> kv_pin(int pos) {
    if (!kv_paged) return {};
    if ((pos + KV_PAGE - 1) / KV_PAGE > kv_seq.mapped) kv_fatal("pin beyond mapped", pos);
    std::vector<int> v = kv_seq.pin(kv_pool, pos);
    if (kv_mode == 2)
      fprintf(stderr, "[kvpage] pin %zu pages for %d tokens (free %d/%d)\n", v.size(),
              pos, kv_pool.nfree(), kv_pool_pages);
    return v;
  }
  void kv_unpin(const std::vector<int>& pages) {
    if (!kv_paged || pages.empty()) return;
    KvSeqTable::unpin(kv_pool, pages);
    if (kv_mode == 2)
      fprintf(stderr, "[kvpage] unpin %zu pages (free %d/%d)\n", pages.size(),
              kv_pool.nfree(), kv_pool_pages);
  }
  void kv_adopt(const std::vector<int>& pages) {
    if (!kv_paged) return;
    CK(hipDeviceSynchronize());  // entries about to change may still be read
    const int m1 = kv_seq.mapped;
    kv_seq.adopt(kv_pool, pages);
    const int n = std::max(m1, kv_seq.mapped);
    if (n > 0)
      CK(hipMemcpy(d_ptab, kv_seq.tab.data(), (size_t)n * 4, hipMemcpyHostToDevice));
    if (kv_mode == 2) {
      fprintf(stderr, "[kvpage] adopt %d pages (was %d mapped, free %d/%d)\n",
              kv_seq.mapped, m1, kv_pool.nfree(), kv_pool_pages);
      if (kv_guard_dirty())
        fprintf(stderr, "[kvpage] ERROR: GUARD PAGE WRITTEN — some kernel wrote "
                        "through an unmapped logical page\n");
    }
  }
  // Release the pages that hold no row < keep_pos (reset_state: 0; RAM/SSD
  // checkpoint restores: the restored length). Waits for the device first:
  // in-flight work may still read the entries being reset to the guard.
  void kv_trim(int keep_pos) {
    if (!kv_paged) return;
    const int m1 = kv_seq.mapped;
    if (kv_seq.trim(kv_pool, keep_pos) == 0) return;
    const int m0 = kv_seq.mapped;
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(d_ptab + m0, kv_seq.tab.data() + m0, (size_t)(m1 - m0) * 4,
                 hipMemcpyHostToDevice));
    if (kv_mode == 2) {
      fprintf(stderr, "[kvpage] released %d pages (mapped %d, free %d/%d)\n",
              m1 - m0, m0, kv_pool.nfree(), kv_pool_pages);
      if (kv_guard_dirty())
        fprintf(stderr, "[kvpage] ERROR: GUARD PAGE WRITTEN — some kernel wrote "
                        "through an unmapped logical page\n");
    }
  }
  // The guard page's slice of every paged buffer: fn(device ptr, bytes).
  template <typename F>
  void kv_guard_regions(F&& fn) {
    kv_page_regions(kv_pool.guard(), fn);
  }
  // Physical page `page`'s slice of every paged buffer, in a fixed order.
  template <typename F>
  void kv_page_regions(int page, F&& fn) {
    kv_page_regions_k(page, [&](void* q, size_t n, char) { fn(q, n); });
  }
  // Same, fn(device ptr, bytes, kind) with kind k / v / t (transposed V) / i.
  // Layers 0..kv_nl-1: the trunk QSA layers, then the MTP layer (no t).
  template <typename F>
  void kv_page_regions_k(int page, F&& fn) {
    const int nqsa = g_cfg.layers / 4;
    const size_t g = (size_t)page;
    for (int qi = 0; qi < kv_nl; qi++) {
      const size_t row0 = (size_t)qi * kv_layer_rows + g * KV_PAGE;
      if (qsa_kv_fp32) {
        fn(d_kc + row0 * 512, (size_t)KV_PAGE * 512 * 4, 'k');
        fn(d_vc + row0 * 512, (size_t)KV_PAGE * 512 * 4, 'v');
      } else {
        fn(d_kcb + row0 * 512, (size_t)KV_PAGE * 512 * 2, 'k');
        fn(d_vcb + row0 * 512, (size_t)KV_PAGE * 512 * 2, 'v');
        if (qsa_wmma_btv && qi < nqsa)
          fn(vct_layer(qi) + g * KV_PAGE_BLOCKS * 2 * 256 * 4,
             (size_t)KV_PAGE_BLOCKS * 2 * 256 * 4 * 2, 't');
      }
      if (d_ik)
        fn(d_ik + ((size_t)qi * ik_layer_blocks + g * KV_PAGE_BLOCKS) * 128,
           (size_t)KV_PAGE_BLOCKS * 128 * 4, 'i');
      if (d_ikh)  // C1 f16 twin: COW/zero/guard treat it like the fp32 page
        fn(d_ikh + ((size_t)qi * ik_layer_blocks + g * KV_PAGE_BLOCKS) * 128,
           (size_t)KV_PAGE_BLOCKS * 128 * 2, 'i');
    }
  }
  bool kv_guard_dirty() {
    std::vector<uint8_t> h;
    bool dirty = false;
    kv_guard_regions([&](void* p, size_t n) {
      if (dirty) return;
      h.resize(n);
      CK(hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost));
      for (uint8_t b : h)
        if (b) {
          dirty = true;
          break;
        }
    });
    return dirty;
  }

  // FP32 indexer projection y[640,P] = w[640,2560] . x[P,2560]^T (index_f32).
  // rocblas_sgemm's own pick for this shape is a MT32x32x8 kernel at ~2.7
  // TFLOPS; the rocBLAS solution list for the same fp32 problem (hipBLASLt-
  // backed, negative indices) holds kernels at 6.2-7.4 TFLOPS (x2.35-2.7 for
  // P >= 1024, slower at P=53). tools/sgemm_iproj_check.cu: deterministic
  // run-to-run, maxrel ~7e-6 vs sgemm — the same order as sgemm's own error vs
  // fp64 (~4.5e-6); still fp32 in/accumulate, only the K order differs. For
  // P >= 1024 the fastest listed solution is timed once (first such call,
  // one ~0.1 s sync stall) and reused for every P; it is only adopted if it
  // beats sgemm, and any non-success launch falls back to sgemm.
  // QWENOX_IPROJ_SGEMM=1 keeps plain rocblas_sgemm (A/B).
  int iproj_sol = INT_MIN;  // INT_MIN = not picked yet, INT_MAX = use sgemm
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // beta API
  void iproj_sgemm(const float* w, const float* x, float* y, int P) {
    const float one = 1.f, zero = 0.f;
    static const bool plain = getenv("QWENOX_IPROJ_SGEMM") != nullptr;
    // decode-P (<=8) multi-row fp32 GEMV in place of rocblas_sgemm's
    // MT32x32x8 (20 blocks, ~110 us at P=4 vs ~36 us streaming the 6.5 MB
    // weight once). NOT bit-exact vs sgemm (lane-split reorder); ktest
    // maxrel 1e-5 vs fp64 (same order as sgemm's own fp32 error) and
    // 8K/32K MTP+chain ids matched base exactly. QWENOX_IPROJ_GEMV=0 opt-out.
    static const bool use_gemv = !env_zero("QWENOX_IPROJ_GEMV");
    if (use_gemv && !plain && P >= 1 && P <= 8) {
      const unsigned blocks = (640 + 15) / 16;  // 16 warps (512 thr) per block
      switch (P) {
#define QWENOX_IPROJ_GV(PP)                                                \
  case PP:                                                               \
    k_f32_gemv_mr<PP><<<blocks, 512, 0, g_str>>>(w, x, y, 640, 2560);    \
    break;
        QWENOX_IPROJ_GV(1)
        QWENOX_IPROJ_GV(2)
        QWENOX_IPROJ_GV(3)
        QWENOX_IPROJ_GV(4)
        QWENOX_IPROJ_GV(5)
        QWENOX_IPROJ_GV(6)
        QWENOX_IPROJ_GV(7)
        QWENOX_IPROJ_GV(8)
#undef QWENOX_IPROJ_GV
      }
      return;
    }
    auto sol_run = [&](int idx) {
      return rocblas_gemm_ex(
          rbh, rocblas_operation_transpose, rocblas_operation_none, 640, P,
          2560, &one, w, rocblas_datatype_f32_r, 2560, x,
          rocblas_datatype_f32_r, 2560, &zero, y, rocblas_datatype_f32_r, 640,
          y, rocblas_datatype_f32_r, 640, rocblas_datatype_f32_r,
          rocblas_gemm_algo_solution_index, idx, 0);
    };
    auto sgemm_run = [&]() {
      return rocblas_sgemm(rbh, rocblas_operation_transpose,
                           rocblas_operation_none, 640, P, 2560, &one, w, 2560,
                           x, 2560, &zero, y, 640);
    };
    if (!plain && P >= 1024) {
      if (iproj_sol == INT_MIN) {
        iproj_sol = INT_MAX;
        rocblas_int n = 0;
        rocblas_gemm_ex_get_solutions(
            rbh, rocblas_operation_transpose, rocblas_operation_none, 640, P,
            2560, &one, w, rocblas_datatype_f32_r, 2560, x,
            rocblas_datatype_f32_r, 2560, &zero, y, rocblas_datatype_f32_r,
            640, y, rocblas_datatype_f32_r, 640, rocblas_datatype_f32_r,
            rocblas_gemm_algo_solution_index, 0, nullptr, &n);
        std::vector<rocblas_int> sols(std::max<rocblas_int>(n, 1));
        if (n > 0)
          rocblas_gemm_ex_get_solutions(
              rbh, rocblas_operation_transpose, rocblas_operation_none, 640, P,
              2560, &one, w, rocblas_datatype_f32_r, 2560, x,
              rocblas_datatype_f32_r, 2560, &zero, y, rocblas_datatype_f32_r,
              640, y, rocblas_datatype_f32_r, 640, rocblas_datatype_f32_r,
              rocblas_gemm_algo_solution_index, 0, sols.data(), &n);
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        auto time2 = [&](auto&& fn) {  // validate once, then time 2 runs
          if (fn() != rocblas_status_success ||
              hipStreamSynchronize(g_str) != hipSuccess) {
            (void)hipGetLastError();
            return 1e30f;
          }
          float ms = 0;
          CK(hipEventRecord(e0, g_str));
          fn();
          fn();
          CK(hipEventRecord(e1, g_str));
          CK(hipEventSynchronize(e1));
          CK(hipEventElapsedTime(&ms, e0, e1));
          return ms;
        };
        const float t_sgemm = time2(sgemm_run);
        float best = t_sgemm;
        for (int i = 0; i < (int)n; i++) {
          const float t = time2([&] { return sol_run(sols[i]); });
          if (t < best * 0.95f) {
            best = t;
            iproj_sol = sols[i];
          }
        }
        CK(hipEventDestroy(e0));
        CK(hipEventDestroy(e1));
        fprintf(stderr, "iproj: fp32 indexer projection P=%d sgemm %.2f ms, %s\n",
                P, t_sgemm / 2,
                iproj_sol == INT_MAX
                    ? "keeping sgemm"
                    : ("solution " + std::to_string(iproj_sol) + " " +
                       std::to_string(best / 2) + " ms")
                          .c_str());
      }
      if (iproj_sol != INT_MAX && sol_run(iproj_sol) == rocblas_status_success)
        return;
    }
    if (sgemm_run() != rocblas_status_success) {
      fprintf(stderr, "indexer fp32 projection failed (P=%d)\n", P);
      exit(1);
    }
  }
#pragma clang diagnostic pop

  // QSA: projections batched. Flash path: qsplit/norm/rope batched, KV rows
  // bulk-copied to cache slots 0..P-1, one causal flash kernel over all
  // positions. QWENOX_QSA_LOOP=1 falls back to the per-token loop (debug).
  // Indexer + score/select phase of qsa_b: host-driven, base-dependent grids,
  // so verify graph capture (P3-B) runs it eagerly between graph segments.
  void qsa_idx_b(int l, int qi, int P, int base, int slot = -1, int r0 = 0,
                 bool state_only = false) {
    if (qsa_dense) return;
    char buf[128];
    // D2-S1c: slot >= 0 runs the segment on that slot's state (iraw ring,
    // page table, snap rows) with explicit scalar base (no d_vbase), and r0
    // shifts the batch-row bases of the [BP] workspaces. slot == -1 &&
    // r0 == 0 is the old call, unchanged.
    const int bs = slot < 0 ? bound_ : slot;
    const bool vseg = slot >= 0;
    float* const xb = d_xb + (size_t)r0 * 2560;
    float* const iproj = d_iproj + (size_t)r0 * 640;
    float* const iq = d_iq + (size_t)r0 * 512;
    uint16_t* const iqh = d_iqh ? d_iqh + (size_t)r0 * 512 : nullptr;
    int* const iselb = d_iselected + (size_t)r0 * 512;
    float2* const ropecs = d_ropecs + (size_t)r0 * 32;
    float* const iraw = slot_d_iraw(bs);
    const int* const ptabc = slot_ptab_eff(bs, P);
    const bool usnap = slot_spec_snap(bs);
    const bool verify = usnap || slot_ngram_verify_active(bs);
    const int vrow0 = slot_vsnap_row0_(bs);
    const int cutr = slot_cut_r_(bs);
    const int* const vb = vseg ? nullptr : vbptr();
#ifdef QWENOX_NPU
    // M4：xb 此刻就绪；提交 q_proj 后整个 indexer + k/v GEMM 都是 NPU 窗口
    if (!state_only && slot < 0 && r0 == 0) npu_qg_submit(l, P, xb);
#endif
    {
      KP_SUB("qsa:idx");
      // rope cos/sin table for the indexer kernels below (idempotent; the QSA
      // prep path re-ensures the same (base, P) later and hits the cache)
      ensure_rope_tab(g_cfg.rope_theta, 64);
      if (vseg)  // per-segment rows [r0, r0+P); the bound slot's (base, P)
        // table cache stays untouched
        k_rope_cs<<<(P * 32 + 255) / 256, 256, 0, g_str>>>(ropecs, base, P,
                                                           nullptr);
      else
        ensure_rope_cs(base, P);
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      if (getenv("QWENOX_VGDUMP")) {  // P3-B: xf16 cache state at the idx gemm
        fprintf(stderr, "[xc%d] P=%d hit=%d xf16P=%d xf16K=%d\n", l, P,
                (int)(xf16_src == xb && xf16_xs == (size_t)g_cfg.d &&
                      xf16_P == P && xf16_K == g_cfg.d),
                xf16_P, xf16_K);
        char tag[12];
        snprintf(tag, sizeof tag, "xb%d", l);
        vg_dump_buf(tag, P, d_xbf16, (size_t)P * 2560 * 2);
      }
      if (state_only && !index_f32) {
        // projection hoisted (qsa_proj_in_b over all R rows)
      } else if (index_f32) {
        iproj_sgemm(d_iw + (size_t)qi * 640 * 2560, xb, iproj, P);
      } else {
        gemm(wview(ckpt.at(buf)), xb, iproj, P);
      }
      float* keys = d_ik + (size_t)qi * ik_layer_blocks * 128;
      uint16_t* keys16 =
          d_ikh ? d_ikh + (size_t)qi * ik_layer_blocks * 128 : nullptr;
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(iproj, d_inorm + qi * 256, iq,
                                              P, g_cfg.norm_eps, g_cfg.rope_theta,
                                              nullptr, base, ropecs, nullptr,
                                              vb, iqh);
      // absolute 4-token blocks [base/4, (base+P)/4); a left-straddling block
      // (base%4 != 0) is completed with history rows from the raw ring
      int nb = (base + P) / 4, b0 = base / 4;
      if (getenv("QWENOX_VGDUMP") && P >= 2 && P <= 16 && !vg_capturing_) {
        char tag[12];
        snprintf(tag, sizeof tag, "ir0_%d", l);
        vg_dump_buf(tag, P, iraw + (size_t)qi * 512, 512 * 4);
        int vbv = -1;
        if (vb) {
          CK(hipStreamSynchronize(g_str));
          CK(hipMemcpy(&vbv, d_vbase, 4, hipMemcpyDeviceToHost));
        }
        fprintf(stderr, "[vb%d] P=%d base=%d vbase=%d\n", l, P, base, vbv);
      }
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(iproj, d_inorm + qi * 256 + 128,
                                                 keys, g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base,
                                                 iraw + qi * 512, ropecs, ptabc,
                                                 vb, keys16);
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(iproj, iraw + qi * 512, P, base,
                                                 vb);
      if (cutr)  // in-pass cut: raw ring after the first r rows
        k_index_ring<<<min(cutr, 4), 128, 0, g_str>>>(iproj, d_ciraw + qi * 512,
                                                      cutr, base, vb);
      if (usnap)
        k_iproj_snap<<<P, 128, 0, g_str>>>(
            iproj, d_iproj_snap + ((size_t)qi * kVSnapRows + vrow0) * 128,
            P);
      // Paged: the score sgemm / tiled kernel read keys [0, nb) contiguously;
      // gather them into d_ikg first (same values, same sgemm shape -> the
      // scores and selections stay bit-identical to the unpaged layout).
      const float* skeys = keys;
      const uint16_t* skeys16 = keys16;
      if (kv_paged && ptabc && nb > 0 && std::max(2051, base) < base + P) {
        k_ik_gather<<<nb, 128, 0, g_str>>>(keys, ptabc, d_ikg, nb);
        skeys = d_ikg;
        if (keys16) {
          k_ik_gather16<<<nb, 128, 0, g_str>>>(keys16, ptabc, d_ikgh, nb);
          skeys16 = d_ikgh;
        }
      }
      const float one = 1.f, zero = 0.f;
      // NOTE: nb (m/ldc) is the absolute pooled-block count, matching the
      // shape a from-scratch prefill of the same total length would use —
      // the sgemm shape stays numerically locked across request splits.
      // Chunk starts are additionally aligned to the from-scratch grid
      // (2051 + k*INDEX_BATCH): with the same (m, n, k) every full chunk
      // replays bitwise-identical scores and selections, so a continuation
      // prefill can only diverge inside the leading ragged chunk [a0, grid).
      const int a0 = std::max(2051, base);
      const int grid =
          2051 + (a0 - 2051 + INDEX_BATCH - 1) / INDEX_BATCH * INDEX_BATCH;
      if (getenv("QWENOX_VGDUMP")) {  // P3-B bisect: selection inputs
        char tag[12];
        // split keys hash: prefix blocks vs last two blocks (freshly pooled)
        snprintf(tag, sizeof tag, "ikp%d", l);
        vg_dump_buf(tag, P, skeys, (size_t)(nb > 2 ? nb - 2 : 0) * 128 * 4);
        snprintf(tag, sizeof tag, "ikl%d", l);
        vg_dump_buf(tag, P, skeys + (size_t)(nb > 2 ? nb - 2 : 0) * 128,
                    (size_t)(nb > 2 ? 2 : nb) * 128 * 4);
        if (skeys16) {
          snprintf(tag, sizeof tag, "ih%d", l);
          vg_dump_buf(tag, P, skeys16, (size_t)nb * 128 * 2);
        }
        snprintf(tag, sizeof tag, "ip%d", l);
        vg_dump_buf(tag, P, iproj, (size_t)P * 640 * 4);
        if (P >= 2 && P <= 16) {  // q-half vs k-half split
          CK(hipStreamSynchronize(g_str));
          std::vector<float> ip((size_t)P * 640);
          CK(hipMemcpy(ip.data(), iproj, (size_t)P * 640 * 4,
                       hipMemcpyDeviceToHost));
          uint64_t hq = 1469598103934665603ULL, hk = 1469598103934665603ULL;
          for (int t = 0; t < P; t++)
            for (int d = 0; d < 640; d++) {
              uint32_t u;
              memcpy(&u, &ip[(size_t)t * 640 + d], 4);
              if (d < 512) { hq ^= u; hq *= 1099511628211ULL; }
              else { hk ^= u; hk *= 1099511628211ULL; }
            }
          fprintf(stderr, "[ips%d] P=%d q=%016llx k=%016llx\n", l, P,
                  (unsigned long long)hq, (unsigned long long)hk);
        }
        snprintf(tag, sizeof tag, "ir%d", l);
        vg_dump_buf(tag, P, iraw + (size_t)qi * 512, 512 * 4);
        snprintf(tag, sizeof tag, "pt%d", l);
        vg_dump_buf(tag, P, slot_d_ptab(bs), (size_t)kv_lpages * 4);
      }
      for (int first = a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        // 验证和重放不能随块大小切换 FP16 排名或 SGEMM 归约顺序。
        if ((verify || (index_fused2 && count >= 16)) &&
            getenv("QWENOX_INDEX_OLDSEL") == nullptr) {
          int* const selrow = iselb + (size_t)(first - base) * 512;  // batch-local
          if (index_f16() && !verify) {
            // C1: f16 WMMA 打分 + f16 score（复用 d_iscores 内存）+ f16 select
            uint16_t* const sc16 = (uint16_t*)d_iscores;
            k_index_scores_wmma16<false>
                <<<index_wmma16_grid(nb, count), 256, 0, g_str>>>(
                    iqh + (size_t)(first - base) * 512, skeys16, sc16, nb,
                    count, first);
            if ((first + count) / 4 <= 8192)
              k_index_select_rs<true, __half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            else
              k_index_select_2p<__half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            first = nxt;
            continue;
          }
          if (index_score64)
            k_index_scores_t64<<<index_t64_grid(nb, count), 256, 0, g_str>>>(
                iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          else
            k_index_scores_tiled<<<dim3((nb + 31) / 32, (count + 15) / 16), 256, 0, g_str>>>(
                iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          if ((first + count) / 4 <= 8192)
            k_index_select_rs<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else if (index_sel2p)
            k_index_select_2p<<<count, 256, 0, g_str>>>(d_iscores, nb, selrow, first);
          else if (index_stream_select)
            k_index_select_stream<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else
            k_index_select<16, true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          first = nxt;
          continue;
        }
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose, rocblas_operation_none,
                                    nb, count * 4, 128, &one, skeys, 128,
                                    iq + (size_t)(first - base) * 512, 128, &zero,
                                    d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, iselb + (size_t)(first - base) * 512, first,
                     count);
        first = nxt;
      }
      if (getenv("QWENOX_VGDUMP")) {
        char tag[12];
        snprintf(tag, sizeof tag, "is%d", l);
        vg_dump_buf(tag, P, iselb, (size_t)P * 512 * 4);
        snprintf(tag, sizeof tag, "iq%d", l);
        vg_dump_buf(tag, P, iq, (size_t)P * 512 * 4);
        snprintf(tag, sizeof tag, "rc%d", l);
        vg_dump_buf(tag, P, ropecs, (size_t)P * 32 * 8);
      }
      KP_SUB_END();
    }
  }

  void qsa_b(int l, int qi, int P, int base = 0, const int* dvb = nullptr,
             int slot = -1, int r0 = 0, bool state_only = false,
             bool defer_attn = false) {
    qsa_idx_b(l, qi, P, base, slot, r0, state_only);
    qsa_main_b(l, qi, P, base, dvb, slot, r0, state_only, defer_attn);
  }
  // Batched-verify attention: can every segment's qsa_flash_b be replaced by
  // the one-launch k_qsa_wmma_vseg? Only the production bf16-KV configuration
  // whose per-segment launches are k_qsa_wmma<true, true, Dense> (paged KV,
  // transposed V, WMMA, per-token sparse kernel because spec_snap is on).
  bool qsa_attn_vseg_ok(int qi, const int* sl, int nseg) {
    static const bool off = env_zero("QWENOX_VSEG_ATTN_BATCH");
    if (off || nseg < 2 || qsa_loop || qsa_kv_fp32 || qsa_dense || !qsa_wmma ||
        !qsa_wmma_btv || qsa_wmma6 || qsa_global_v || !kv_paged || !vct_layer(qi))
      return false;
    for (int s = 0; s < nseg; s++)
      if (!slot_spec_snap(sl[s])) return false;  // union path would differ
    return true;
  }
  // All segments' attention (dense prefix rows, then sparse rows) in at most
  // two launches, then the output gate over all R rows (elementwise). Same
  // kernels/arguments per row as qsa_flash_b + the vseg gate, so the output
  // bits match the per-segment loop.
  void qsa_attn_vseg(int qi, const int* Ps, const int* bases, const int* sl,
                     const int* voffs, int nseg, int R) {
    const size_t qsa_off = (size_t)qi * kv_layer_rows * 512;
    const uint16_t* kc = d_kcb + qsa_off;
    const uint16_t* vc = d_vcb + qsa_off;
    const uint16_t* vct = vct_layer(qi);
    QsaVSegs dn{}, sp{};
    int nd = 0, ns = 0;
    for (int s = 0; s < nseg; s++) {
      const int P = Ps[s], base = bases[s];
      const int* ptab = slot_d_ptab(sl[s]);
      const int* iselb = d_iselected + (size_t)voffs[s] * 512;
      const int dense_P = std::min(P, std::max(0, 2051 - base));
      const int first_abs = std::max(base, 2051), nsparse = base + P - first_abs;
      dn.s[s] = QsaVSeg{ptab, nullptr, base, base, dense_P, voffs[s]};
      for (int b = 0; b < dense_P; b++, nd++) {
        dn.rowseg[nd] = (signed char)s;
        dn.rowbx[nd] = (signed char)b;
      }
      sp.s[s] = QsaVSeg{ptab, iselb + (size_t)(first_abs - base) * 512, first_abs,
                        base, P, voffs[s]};
      for (int b = 0; b < nsparse; b++, ns++) {
        sp.rowseg[ns] = (signed char)s;
        sp.rowbx[ns] = (signed char)b;
      }
    }
    KP_SUB("qsa:flash");
    if (nd)
      k_qsa_wmma_vseg<true, true, true><<<dim3((unsigned)nd, 2), 256, 0, g_str>>>(
          d_qsb, kc, vc, vct, d_attnb, 256, 2, 24, dn);
    if (ns)
      k_qsa_wmma_vseg<true, true, false><<<dim3((unsigned)ns, 2), 256, 0, g_str>>>(
          d_qsb, kc, vc, vct, d_attnb, 256, 2, 24, sp);
    KP_SUB_END();
    KP_SUB("qsa:gate");
    k_sigmoid_gate<<<(R * 6144 + 255) / 256, 256, 0, g_str>>>(d_attnb, d_gsb,
                                                              d_attnb, R * 6144);
    KP_SUB_END();
  }
  // QSA q/k/v + indexer projections / o_proj over P batch rows (row-invariant
  // gemv: one R-row call == per-segment calls bit for bit). Batched verify
  // hoists these out of the per-segment loop (state_only below).
  void qsa_proj_in_b(int l, int qi, int P) {
    char buf[128];
    KP_SUB("qsa:proj");
    // Indexer projection only on the bf16/quantized gemm path: the fp32
    // iproj_sgemm switches kernel family at P > 8 (gemv -> rocBLAS, not
    // bit-exact), so under index_f32 it stays per segment (qsa_idx_b).
    if (!qsa_dense && !index_f32) {
      snprintf(buf, sizeof buf, "layers.%d.self_attn.indexer.index_qk_proj.weight", l);
      gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    KP_SUB_END();
  }
  void qsa_proj_out_b(int l, int P) {
    char buf[128];
    KP_SUB("qsa:oproj");
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_attnb, d_yb, P, 0, nullptr);
    KP_SUB_END();
  }

  // Graphable remainder of qsa_b (P3-B): dvb = device base mirror (graph
  // capture) or nullptr (direct execution, identical scalar path).
  // defer_attn (batched verify, with state_only): stop after the KV store —
  // the caller runs every segment's attention + output gate in one launch
  // (qsa_attn_vseg). Only the bf16-KV non-loop path supports it.
  void qsa_main_b(int l, int qi, int P, int base, const int* dvb, int slot = -1,
                  int r0 = 0, bool state_only = false, bool defer_attn = false) {
    char buf[128];
    // D2-S1c: slot >= 0 runs the segment on that slot's state (page table,
    // snap flag) with explicit scalar base (no d_vbase); r0 shifts the
    // batch-row bases of the [BP] workspaces. slot == -1 && r0 == 0 is the
    // old call, unchanged.
    const int bs = slot < 0 ? bound_ : slot;
    const bool vseg = slot >= 0;
    if (vseg) dvb = nullptr;  // graph capture is bound-slot only
    float* const xb = d_xb + (size_t)r0 * 2560;
    float* const qgb = d_qgb + (size_t)r0 * 12288;
    float* const kbb = d_kbb + (size_t)r0 * 512;
    float* const vbb = d_vbb + (size_t)r0 * 512;
    float* const qsb = d_qsb + (size_t)r0 * 6144;
    float* const gsb = d_gsb + (size_t)r0 * 6144;
    float* const attnb = d_attnb + (size_t)r0 * 6144;
    float* const yb = d_yb + (size_t)r0 * 2560;
    uint16_t* const xbf16b = d_xbf16b + (size_t)r0 * 6144;
    int* const iselb = d_iselected + (size_t)r0 * 512;
    float2* const ropecs = d_ropecs + (size_t)r0 * 32;
    KvSeqTable& seq = slot_kv_seq(bs);
    const int* const ptabc = slot_ptab_eff(bs, P);
    const int sovr = vseg ? (slot_spec_snap(bs) ? 1 : 0) : -1;
    const uint16_t* attn_bf = nullptr;  // gated o_proj input already in bf16
    if (!state_only) {
    KP_SUB("qsa:proj");
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
#ifdef QWENOX_NPU
    const int nqg = npu_qg_pend_rows(l);  // M4：NPU 已覆盖的行数
    if (nqg > 0) {
      if (P > nqg)  // 尾部行 GPU 补（GEMM 行独立），与 NPU 并行
        gemm(wview(ckpt.at(buf)), xb + (size_t)nqg * 2560,
             qgb + (size_t)nqg * 12288, P - nqg);
    } else
#endif
    gemm(wview(ckpt.at(buf)), xb, qgb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_proj.weight", l);
    gemm(wview(ckpt.at(buf)), xb, kbb, P);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.v_proj.weight", l);
    gemm(wview(ckpt.at(buf)), xb, vbb, P);
    KP_SUB_END();
    }
    if (getenv("QWENOX_VGDUMP")) {  // D2-S1b bisect: q/k/v proj outs
#ifdef QWENOX_NPU
      npu_qg_finish_all(l, xb);  // dump qgb 前强制收尾
#endif
      char tag[12];
      snprintf(tag, sizeof tag, "qg%d", l);
      vg_dump_buf(tag, P, qgb, (size_t)P * 12288 * 4);
      snprintf(tag, sizeof tag, "vk%d", l);
      vg_dump_buf(tag, P, kbb, (size_t)P * 512 * 4);
      snprintf(tag, sizeof tag, "vv%d", l);
      vg_dump_buf(tag, P, vbb, (size_t)P * 512 * 4);
    }
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_norm.weight", l);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "layers.%d.self_attn.k_norm.weight", l);
    float* knw = ensure_norm256(buf);
    // kv_layer_rows == maxctx unless paged (FP32 or BF16 cache, A1c)
    const size_t qsa_off = (size_t)qi * kv_layer_rows * 512;
    if (!qsa_loop) {
      ensure_rope_tab(g_cfg.rope_theta, 64);
      if (vseg)  // per-segment rows [r0, r0+P); the bound slot's (base, P)
        // table cache stays untouched
        k_rope_cs<<<(P * 32 + 255) / 256, 256, 0, g_str>>>(ropecs, base, P,
                                                           nullptr);
      else
        ensure_rope_cs(base, P);
      // fused: qsplit+rope (K1) and knorm+rope[+bf16 store] (K2), bit-exact
      // vs the previous qsplit / rmsnorm_zc / rope_b x2 / convert chain
      // P > 8: o_proj takes the dequant->bf16->GEMM path (P <= 8 would run
      // the multi-row GEMV on fp32 x), so the output gate + bf16 convert fuse
      // into k_sigmoid_gate_bf16_v4, which reads the gate half of d_qgb
      // directly (qsplit skips its gs copy). QWENOX_QSA_GATE_BF16=0 disables.
      static const bool gate_bf16_on = !env_zero("QWENOX_QSA_GATE_BF16");
      // 验证行的分块随 γ 和接受深度变化，不能按 P 提前舍入输入。
      const bool verify = slot_spec_snap(bs) || slot_ngram_verify_active(bs);
      const bool gate_bf16 = gate_bf16_on && P > 8 && !vseg && !verify;
#ifdef QWENOX_NPU
      npu_qg_finish_q(l);  // M4：qsplit 读 q 半前收尾（qsa:idx+k/v 已盖窗口）
#endif
      KP_SUB("qsa:prep");
      k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(
          qgb, qnw, qsb, gate_bf16 ? nullptr : gsb, 256, g_cfg.norm_eps, base,
          ropecs);
      if (getenv("QWENOX_VGDUMP")) {  // D2-S1b bisect: q post split+rope
        char tag[12];
        snprintf(tag, sizeof tag, "qs%d", l);
        vg_dump_buf(tag, P, qsb, (size_t)P * 6144 * 4);
      }
#ifdef QWENOX_NPU
      npu_qg_submit_g(l, xb);  // M4：gate 半趁 qsa:flash（~200ms）跑
#endif
      int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
      if (qsa_kv_fp32) {
        float* kc = d_kc + qsa_off;
        float* vc = d_vc + qsa_off;
        k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(kbb, knw, kbb,
                                                          nullptr, g_cfg.norm_eps,
                                                          base, ropecs);
        // absolute positions base..base+P-1 are logically contiguous: one bulk
        // copy (unpaged) or one copy per physically contiguous page run
        if (dvb) {  // P3-B graph capture: device-side base + page mapping
          k_kv_rows_store_pd<<<P, 128, 0, g_str>>>(kc, kbb, ptabc, P, dvb);
          k_kv_rows_store_pd<<<P, 128, 0, g_str>>>(vc, vbb, ptabc, P, dvb);
        } else {
          kv_rows_store_seq(seq, kc, kbb, base, P);
          kv_rows_store_seq(seq, vc, vbb, base, P);
        }
        KP_SUB_END();
        KP_SUB("qsa:flash");
        qsa_flash_b(kc, vc, P, dense_P, base, nullptr, ptabc, dvb, r0, sovr);
        KP_SUB_END();
      } else {
        uint16_t* kc = d_kcb + qsa_off;
        uint16_t* vc = d_vcb + qsa_off;
        uint16_t* vct = vct_layer(qi);
        if (vct && base > 0 && getenv("QWENOX_VCT_CHECK")) {  // debug only
          static unsigned* d_chk = nullptr;
          if (!d_chk) CK(hipMalloc(&d_chk, 3 * sizeof(unsigned)));
          const unsigned init[3] = {0u, ~0u, 0u};
          unsigned h[3];
          CK(hipMemcpy(d_chk, init, sizeof init, hipMemcpyHostToDevice));
          k_vct_check<<<(unsigned)(((size_t)base * 512 + 255) / 256), 256, 0, g_str>>>(
              vc, vct, ptabc, base, d_chk);
          CK(hipStreamSynchronize(g_str));
          CK(hipMemcpy(h, d_chk, sizeof h, hipMemcpyDeviceToHost));
          if (h[0] || qi == 0)
            fprintf(stderr, "[vctchk] qsa%d base=%d P=%d mismatch=%u first=%d last=%d\n",
                    qi, base, P, h[0], h[0] ? (int)h[1] : -1, h[0] ? (int)h[2] : -1);
          const uint64_t nraw = (uint64_t)qsa_vt_blocks * 2 * 256 * 4;
          CK(hipMemcpy(d_chk, init, sizeof init, hipMemcpyHostToDevice));
          k_vct_nonfinite<<<(unsigned)((nraw + 255) / 256), 256, 0, g_str>>>(vct, nraw,
                                                                            d_chk);
          CK(hipStreamSynchronize(g_str));
          CK(hipMemcpy(h, d_chk, sizeof h, hipMemcpyDeviceToHost));
          if (h[0] || qi == 0)
            fprintf(stderr, "[vctnan] qsa%d base=%d nonfinite=%u phys rows %d..%d\n", qi,
                    base, h[0], h[0] ? (int)h[1] : -1, h[0] ? (int)h[2] : -1);
        }
        if (!kv_paged) {
          k_qsa_kprep<true><<<dim3(2, P), 256, 0, g_str>>>(
              kbb, knw, kbb, dvb ? kc : kc + (size_t)base * 512,
              g_cfg.norm_eps, base, ropecs, dvb);
          // absolute positions base..base+P-1 are contiguous: one bulk convert
          k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * 128 + 255) / 256), 256, 0,
                             g_str>>>(vbb, dvb ? vc : vc + (size_t)base * 512,
                                      512, P, 512, dvb);
          if (vct) {
            // dvb: baked worst-case grid (base mod 4 unknown at capture)
            const int btb = dvb ? (P + 6) / 4 + 1 : (base + P + 3) / 4 - base / 4;
            k_f32_to_bf16_v4_bt<<<(unsigned)(((size_t)btb * 512 + 255) / 256),
                                 256, 0, g_str>>>(vbb, vct, 512, P, 512, base, dvb);
          }
        } else {
          // A1c paged: kprep writes the fp32 K rows only (kprep<true> stores
          // f2bf of exactly those values, so converting afterwards is the same
          // bits), then K / V / transposed V are converted per physically
          // contiguous page run. The transposed-V kernel is called with the
          // run's physical base: a page holds whole 4-token blocks, so the
          // slot (pos & 3) and the partial-edge-block masking are unchanged and
          // block pr/4 is the physical block ik_blk(lr/4).
          k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(
              kbb, knw, kbb, nullptr, g_cfg.norm_eps, base, ropecs);
          if (dvb) {  // P3-B graph capture: one launch per output, device-side
                      // base + page mapping (same bits as the per-run loop)
            const unsigned g = (unsigned)(((size_t)P * 128 + 255) / 256);
            k_f32_to_bf16_v4_pd<<<g, 256, 0, g_str>>>(kbb, kc, ptabc, P, 512,
                                                      dvb);
            k_f32_to_bf16_v4_pd<<<g, 256, 0, g_str>>>(vbb, vc, ptabc, P, 512,
                                                      dvb);
            if (vct)  // baked worst-case grid; the kernel masks by *dvb
              k_f32_to_bf16_v4_bt_pd<<<(unsigned)(((((size_t)P + 6) / 4 + 1) *
                                                   512 + 255) /
                                                  256),
                                    256, 0, g_str>>>(vbb, vct, ptabc, P, 512,
                                                     dvb);
          } else {
            kv_runs_seq(seq, base, base + P, KV_PAGE, [&](int lr, size_t prz, int n) {
              const int pr = (int)prz;
              const size_t so = (size_t)(lr - base) * 512;
              const unsigned g = (unsigned)(((size_t)n * 128 + 255) / 256);
              k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(kbb + so, kc + prz * 512,
                                                     512, n, 512);
              k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(vbb + so, vc + prz * 512,
                                                     512, n, 512);
              if (vct)
                k_f32_to_bf16_v4_bt<<<(unsigned)(((size_t)((pr + n + 3) / 4 - pr / 4) *
                                                  512 + 255) /
                                                     256),
                                     256, 0, g_str>>>(vbb + so, vct, 512, n, 512, pr);
            });
          }
        }
        if (vct && getenv("QWENOX_VCT_TAILZERO") && ((base + P) & 3)) {  // debug only
          const int rz = base + P, r1 = (rz + 3) & ~3;
          const char* tv = getenv("QWENOX_VCT_TAILVAL");  // hex bf16 poison (debug)
          k_vct_zero_slots<<<(unsigned)(((size_t)(r1 - rz) * 512 + 255) / 256), 256, 0,
                             g_str>>>(vct, ptabc, rz, r1,
                                      tv ? (uint16_t)strtoul(tv, nullptr, 16) : 0);
        }
        KP_SUB_END();
        if (getenv("QWENOX_VGDUMP")) {  // D2-S1b bisect: k/v post norm+rope
          char tag[12];
          snprintf(tag, sizeof tag, "kn%d", l);
          vg_dump_buf(tag, P, kbb, (size_t)P * 512 * 4);
          snprintf(tag, sizeof tag, "vn%d", l);
          vg_dump_buf(tag, P, vbb, (size_t)P * 512 * 4);
        }
        if (defer_attn) return;  // KV is stored; attention + gate batched later
        KP_SUB("qsa:flash");
        qsa_flash_b(kc, vc, P, dense_P, base, vct, ptabc, dvb, r0, sovr);
        KP_SUB_END();
        if (getenv("QWENOX_VGDUMP")) {  // D2-S1b bisect: raw attention out
          char tag[12];
          snprintf(tag, sizeof tag, "at%d", l);
          vg_dump_buf(tag, P, attnb, (size_t)P * 6144 * 4);
        }
        // D2-S1b: QWENOX_ATDUMP=<layer> dumps row P-1 of d_attnb as 12
        // per-512-float FNV hashes + first 8 floats (locate the differing
        // head/dim segment between P=3 and P=4 runs).
        if (getenv("QWENOX_ATDUMP") && atoi(getenv("QWENOX_ATDUMP")) == l) {
          const int dr = [] { const char* e = getenv("QWENOX_ATDROW");
                              return e ? atoi(e) : 2; }();
          if (P > dr) {
          CK(hipStreamSynchronize(g_str));
          std::vector<float> h(6144);
          CK(hipMemcpy(h.data(), attnb + (size_t)dr * 6144, 6144 * 4,
                       hipMemcpyDeviceToHost));
          fprintf(stderr, "[atd%d] P=%d row%d head", l, P, dr);
          for (int s = 0; s < 12; s++) {
            uint64_t fnv = 1469598103934665603ULL;
            for (int i = s * 512; i < s * 512 + 512; i++) {
              uint32_t u;
              memcpy(&u, &h[i], 4);
              fnv ^= u; fnv *= 1099511628211ULL;
            }
            fprintf(stderr, " %016llx", (unsigned long long)fnv);
          }
          fprintf(stderr, "\n[atd%d] first8", l);
          for (int i = 0; i < 8; i++) fprintf(stderr, " %a", (double)h[i]);
          fprintf(stderr, "\n");
          }
        }
        if (base > 0 && getenv("QWENOX_ATTN_HASH")) dbg_attn_hash(qi, base, P, kc, vc, vct);
      }
      KP_SUB("qsa:gate");
      if (gate_bf16) {  // see qsplit above
#ifdef QWENOX_NPU
        npu_qg_finish_g(l);  // M4：qsa:flash 已跑完，收尾 gate 半
#endif
        const uint64_t n4 = (uint64_t)P * 6144 / 4;
        k_sigmoid_gate_bf16_v4<<<(unsigned)((n4 + 255) / 256), 256, 0, g_str>>>(
            attnb, qgb, xbf16b, n4);
        attn_bf = xbf16b;
      } else {
        k_sigmoid_gate<<<(P * 6144 + 255) / 256, 256, 0, g_str>>>(attnb, gsb,
                                                                  attnb, P * 6144);
      }
      KP_SUB_END();
    } else {
#ifdef QWENOX_NPU
      npu_qg_finish_all(l, xb);  // M4：loop 逐 token 读 qg 全部
#endif
      for (int t = 0; t < P; t++) {
        const int* pabs = d_posarr + base + t;  // absolute position
        const float* qg = qgb + (size_t)t * 12288;
        float* qs = qsb + (size_t)t * 6144;
        float* gs = gsb + (size_t)t * 6144;
        float* kb = kbb + (size_t)t * 512;
        float* vb = vbb + (size_t)t * 512;
        float* att = attnb + (size_t)t * 6144;
        k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(qg, qnw, qs, gs, 256,
                                                     g_cfg.norm_eps);
        k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(kb, knw, kb, 256, g_cfg.norm_eps, 1);
        k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(qs, 24, 256, 64, pabs,
                                                         g_cfg.rope_theta);
        k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(kb, 2, 256, 64, pabs,
                                                        g_cfg.rope_theta);
        if (qsa_kv_fp32) {
          k_store_kv<<<2, 256, 0, g_str>>>(kb, vb, d_kc + qsa_off, d_vc + qsa_off,
                                           pabs, ptabc);
          k_qsa_step<float><<<24, 256, 0, g_str>>>(
              qs, d_kc + qsa_off, d_vc + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : iselb + (size_t)t * 512, false,
              ptabc);
        } else {
          k_store_kv_bf16<<<2, 256, 0, g_str>>>(kb, vb, d_kcb + qsa_off,
                                                d_vcb + qsa_off, pabs,
                                                vct_layer(qi), ptabc);
          k_qsa_step<uint16_t><<<24, 256, 0, g_str>>>(
              qs, d_kcb + qsa_off, d_vcb + qsa_off, att, pabs, 256, 2, 24,
              qsa_dense ? nullptr : iselb + (size_t)t * 512, false,
              ptabc);
        }
        k_sigmoid_gate<<<(6144 + 255) / 256, 256, 0, g_str>>>(att, gs, att, 6144);
      }
    }
    if (state_only) {
      if (qsa_loop || attn_bf)  // hoisted o_proj needs the fp32 gated attnb
        throw std::runtime_error("qsa_main_b: state_only needs fp32 gate path");
    } else {
    KP_SUB("qsa:oproj");
    snprintf(buf, sizeof buf, "layers.%d.self_attn.o_proj.weight", l);
    gemm(wview(ckpt.at(buf)), attnb, yb, P, 0, attn_bf);
    KP_SUB_END();
    }
    if (getenv("QWENOX_VGDUMP")) {  // D2-S1b bisect: o_proj out
      char tag[12];
      snprintf(tag, sizeof tag, "op%d", l);
      vg_dump_buf(tag, P, yb, (size_t)P * 2560 * 4);
    }
  }

  // MoE: router batched; expert GEMVs grouped over P*k slots (naive: full
  // weight traffic per token — correct but slow; optimized in a later phase).
  // Returns true when the HC write-back of the moe output was fused into the
  // reduce (B2-1; only on the LUT sg_fuse paths with try_hcw set): R is
  // updated, acc/d_yb are not written and the caller skips gr_write_b.
  // ---- in-pass snapshot cut (see cut_abs) ----
  bool cut_snap_supported() const {
    static const bool off = env_zero("QWENOX_CUT_INPASS");
    return !off && gdn_fused && !gdn_loop && !gdn_nosplit && gr_bf16 && !ple_loop;
  }
  void cut_alloc() {
    if (d_cS) return;
    const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
    CK(hipMalloc(&d_cS, (size_t)ngdn * 48 * 128 * 128 * 4));
    CK(hipMalloc(&d_cconvst, (size_t)ngdn * 3 * 10240 * 4));
    CK(hipMalloc(&d_cring, (size_t)9 * 10240 * 4));
    CK(hipMalloc(&d_ciraw, (size_t)nqsa * 4 * 128 * 4));
    CK(hipMalloc(&d_cR, (size_t)g_cfg.branches * g_cfg.d * 4));
    CK(hipMalloc(&d_cmiraw, (size_t)4 * 128 * 4));
  }
  // chunk start: r = chunk-local cut row in (0, P], else 0. Copies the
  // pre-chunk rings / conv state; the per-site replays below update them.
  void cut_begin_chunk(int base, int P) {
    cut_r_ = (cut_abs > base && cut_abs <= base + P) ? cut_abs - base : 0;
    if (!cut_r_) return;
    const int nqsa = g_cfg.layers / 4, ngdn = g_cfg.layers - nqsa;
    CK(hipMemcpyAsync(d_cconvst, d_convst, (size_t)ngdn * 3 * 10240 * 4,
                      hipMemcpyDeviceToDevice, g_str));
    CK(hipMemcpyAsync(d_cring, d_ring, (size_t)9 * 10240 * 4, hipMemcpyDeviceToDevice, g_str));
    if (d_iraw)
      CK(hipMemcpyAsync(d_ciraw, d_iraw, (size_t)nqsa * 4 * 128 * 4, hipMemcpyDeviceToDevice,
                        g_str));
  }
  // Run fn with the cut-point state swapped in (pos = cut_abs). mtp: the MTP
  // ring snapshot was taken, so the lockstep invariant mtp_pos == pos-1 holds.
  template <class F>
  void with_cut_state(bool mtp, F&& fn) {
    const int sv_pos = pos, sv_ring = ple_ringpos, sv_mpos = mtp_pos;
    auto swp = [&] {
      std::swap(d_S, d_cS);
      std::swap(d_convst, d_cconvst);
      std::swap(d_ring, d_cring);
      if (d_iraw) std::swap(d_iraw, d_ciraw);
      std::swap(d_R, d_cR);
      if (mtp && d_miraw) std::swap(d_miraw, d_cmiraw);
    };
    swp();
    pos = cut_abs;
    ple_ringpos = cut_abs % 9;
    if (mtp) mtp_pos = cut_abs - 1;
    fn();
    swp();
    pos = sv_pos;
    ple_ringpos = sv_ring;
    mtp_pos = sv_mpos;
  }

  // d_routerb [P][512] -> d_topidsb/d_topwb. wave-per-token kernel (bit-
  // identical, aggr 10-05); QWENOX_ROUTER_W32=0 restores the block-per-token one.
  void router_topk_b(int P, int k, VizTap vt = {}) {
    static const bool w32 = !env_zero("QWENOX_ROUTER_W32");
    if (w32 && g_cfg.experts == 512 && k == 10)
      k_router_topk_w32<10><<<(P + 7) / 8, 256, 0, g_str>>>(d_routerb, d_topidsb, d_topwb, P,
                                                           vt);
    else
      k_router_topk<<<P, 512, 0, g_str>>>(d_routerb, d_topidsb, d_topwb, g_cfg.experts, k,
                                          vt);
  }
  bool moe_b(int l, int P, bool try_hcw = false, bool vseg = false) {
    char buf[128];
    // D2-S1c: vseg = one segment of a multi-segment verify batch. Force the
    // per-row naive path so the batched rows stay bit-identical to each
    // segment's serial P<=8 run (the direct_output/q4w gates are P-dependent).
    const bool direct_output = pp_moe_out && P > 8 && !vseg;
    float* const acc = direct_output ? d_yb : d_moeaccb;
    snprintf(buf, sizeof buf, "layers.%d.mlp.gate.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_routerb, P);
    int k = g_cfg.topk;
    router_topk_b(P, k, vz_tap(l, true, false, P));  // vseg rows: per-row table
    if (l == g_cfg.layers - 1) vz_publish(P, true);
    // QWENOX_MOE_UNION_LOG=1: distinct experts across the P rows (P<=8, i.e.
    // verify); decides whether naive-MoE expert dedup is worth it.
    static const bool moe_union_log = getenv("QWENOX_MOE_UNION_LOG") != nullptr;
    if (moe_union_log && P <= 8) {
      struct UStat {
        long long n = 0, uni = 0, slots = 0;
        ~UStat() {
          if (n)
            fprintf(stderr,
                    "moe-union: layers=%lld avg_union=%.2f avg_slots=%.2f "
                    "union/slots=%.3f\n",
                    n, (double)uni / n, (double)slots / n,
                    (double)uni / slots);
        }
      };
      static UStat S;
      std::vector<int> hids((size_t)P * 16);
      CK(hipMemcpy(hids.data(), d_topidsb, (size_t)P * 16 * 4,
                   hipMemcpyDeviceToHost));
      std::vector<char> seen(g_cfg.experts, 0);
      int u = 0;
      for (int t = 0; t < P; t++)
        for (int s = 0; s < k; s++) {
          int e = hids[(size_t)t * 16 + s];
          if (!seen[e]) { seen[e] = 1; u++; }
        }
      S.n++; S.uni += u; S.slots += (long long)P * k;
    }
    // QWENOX_MOE_TOPIDS_DUMP=<file>: 把每次小 P 调用（2<=P<=32，即 verify/合批
    // 场景）的 topids 追加到文件，供 tools/c0_union.py 做跨序列专家并集分析。
    // 行格式：L <layer> B <base> P <P> 后接 P*k 个专家 id。默认零开销。
    static FILE* moe_dump = [] {
      const char* f = getenv("QWENOX_MOE_TOPIDS_DUMP");
      return f ? fopen(f, "w") : nullptr;
    }();
    if (moe_dump && P >= 2 && P <= 32) {
      std::vector<int> hids((size_t)P * 16);
      CK(hipMemcpy(hids.data(), d_topidsb, (size_t)P * 16 * 4,
                   hipMemcpyDeviceToHost));
      fprintf(moe_dump, "L %d B %d P %d", l, moe_dump_base_, P);
      for (int t = 0; t < P; t++)
        for (int s = 0; s < k; s++)
          fprintf(moe_dump, " %d", hids[(size_t)t * 16 + s]);
      fputc('\n', moe_dump);
      fflush(moe_dump);
    }
    if (gg_on()) {
      // P1: shared expert first (its d_guvb/d_hidb/d_xbf16b scratch is dead
      // after it returns), then fold its add into the routed reduce.
      // QWENOX_GG_SG_FUSE=0 restores the old order + k_axpy_sg.
      static const bool gg_sg_fuse = !env_zero("QWENOX_GG_SG_FUSE");
      if (gg_sg_fuse) {
        KP("moe:shared");
        moe_b_shared_ey(l, P);
        if (!moe_gguf(g_ggx.at(l), d_xb, d_topidsb, d_topwb, acc, P, d_eyb, d_sgb)) {
          // Small-P GEMV path: shared add still has to run separately.
          k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(acc, d_eyb, d_sgb,
                                                               g_cfg.d, P * g_cfg.d);
        }
        if (!direct_output)
          CK(hipMemcpyAsync(d_yb, acc, (size_t)P * g_cfg.d * 4, hipMemcpyDeviceToDevice,
                            g_str));
      } else {
        moe_gguf(g_ggx.at(l), d_xb, d_topidsb, d_topwb, acc, P);
        moe_b_shared(l, P, acc, direct_output);
      }
      return false;
    }
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.gate_up_proj.weight", l);
    WView gu = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "layers.%d.mlp.experts.down_proj.weight", l);
    WView dn = wview(ckpt.at(buf));
    if (gu.dtype == 23) {  // hgn v2 i4r: own paths (q4cp fast paths do not apply)
      return moe_b_v2(l, gu, dn, acc, P, direct_output, try_hcw ? d_w4b : nullptr,
                      try_hcw ? (uint16_t*)d_Rb : nullptr, vseg);
    }
    // Shared expert first (its scratch d_guvb/d_hidb/d_xbf16b is dead before
    // the routed experts reuse it), so its add folds into the routed reduce.
    // QWENOX_MOE_SG_FUSE=0: old order + k_axpy_sg.
    static const bool sg_fuse = !env_zero("QWENOX_MOE_SG_FUSE");
    if (!vseg && !moe_naive && P > moe_naive_max && sg_fuse && moe_q4w_ok(gu, dn, P)) {
      KP("moe:shared");
      moe_b_shared_ey(l, P);
      moe_q4w_run(gu, dn, d_xb, d_topidsb, d_topwb, acc, P, d_eyb, d_sgb,
                  try_hcw ? d_w4b : nullptr, try_hcw ? (uint16_t*)d_Rb : nullptr);
      if (try_hcw) return true;
      if (!direct_output)
        CK(hipMemcpyAsync(d_yb, acc, (size_t)P * g_cfg.d * 4, hipMemcpyDeviceToDevice,
                          g_str));
      return false;
    }
    if (!vseg && !moe_naive && P > moe_naive_max &&
        moe_q4w_run(gu, dn, d_xb, d_topidsb, d_topwb, acc, P)) {
      KP("moe:shared");
      moe_b_shared(l, P, acc, direct_output);
      return false;
    }
    // fused kernels require exact tiling (guards would spill accumulators)
    bool fused_ok = g_cfg.moe_mid % MOE_CT == 0 && g_cfg.d % MOE_CT == 0 &&
                    g_cfg.d % FD_CT_DN == 0 &&
                    gu.cols % MOE_KC == 0 && dn.cols % MOE_KC == 0;
    bool naive_path = vseg || moe_naive || !fused_ok || P <= moe_naive_max;
    bool direct_down = naive_path && moe_down_topk && k == 10 && dn.cols == 640;
    // Direct-down and deterministic reductions overwrite every output element.
    // Atomic/ordered accumulation still requires a zero destination.
    if (!direct_down && (!direct_output || naive_path || moe_ordered || !moe_deterministic))
      CK(hipMemsetAsync(acc, 0, (size_t)P * g_cfg.d * 4, g_str));
    const int* topidsb = d_topidsb;
    if (naive_path) {
      // 同专家槽相邻化（k_moe_sort_slots 排序 + gg 查表，纯 warp 调度排列，逐 bit
      // 不变）。09-28 HQ γ=4 实测：verify gg 351→338µs/层，减去 sort 7.3µs/层后
      // 净 +0.26 ms/轮（0.3%）——Cache 层级已接住绝大部分重复专家读（未排序 gg 等效
      // 262 GB/s，已超 DRAM 冷流率），单序列 P=5 收益低于门槛。
      // 默认关，QWENOX_MOE_SORT=1 启用（留待 D2 合批 R≥16、工作集远超 Cache 的场景）。
      // P=1（串行/draft）单 token topk 专家天然不重复，跳过。
      static const bool moe_sort =
          getenv("QWENOX_MOE_SORT") != nullptr && !env_zero("QWENOX_MOE_SORT");
      const int* slot_order = nullptr;
      if (moe_sort && P >= 2 && P * k <= 256) {
        k_moe_sort_slots<<<1, 128, 0, g_str>>>(topidsb, P, k, 16, d_slot_order);
        slot_order = d_slot_order;
      }
      // v3 kernels: 2 rows per warp -> halved warp counts
      uint64_t pairs = (uint64_t)P * k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<0><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
          d_xb, d_guvb, topidsb, (uint64_t)2 * g_cfg.moe_mid, gu.cols,
          gu.scale_stride, P * k, (uint64_t)g_cfg.d, 16, k, slot_order);
      k_silu_mul_g<<<(P * k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
          d_guvb, d_hidb, g_cfg.moe_mid, P * k);
      if (direct_down) {
        uint64_t row_pairs = (g_cfg.d + 1) / 2;
        constexpr uint64_t warps_per_block = 128 / 32;
        k_q4cp_gemv_gd_topk_h16<10, 0><<<
            (unsigned)(((uint64_t)P * row_pairs + warps_per_block - 1) /
                       warps_per_block),
            128, 0, g_str>>>(
            dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
            (const float*)dn.data, d_hidb, acc, topidsb, d_topwb,
            (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
            (uint64_t)g_cfg.moe_mid, P, 16, (uint64_t)g_cfg.d);
      } else {
        uint64_t pairs2 = (uint64_t)P * k * ((g_cfg.d + 1) / 2);
        k_q4cp_gemv_gd<0><<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
            dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
            (const float*)dn.data, d_hidb, acc, topidsb, d_topwb,
            (uint64_t)g_cfg.d, dn.cols, dn.scale_stride,
            (uint64_t)g_cfg.moe_mid, P * k, 16, (uint64_t)g_cfg.d, k);
      }
    } else {
      // Stable expert bucketing preserves token order and deterministic reduction.
      int E = g_cfg.experts, npairs = P * k;
      if (!moe_host_route && !moe_ordered) {
        moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff,
                        moe_deterministic ? d_pairids : nullptr, P, k, E);
      } else {
        std::vector<int> hids(P * 16);  // topids rows are 16 wide, first k valid
        std::vector<float> hws(P * 16);
        auto pt1 = std::chrono::steady_clock::now();
        CK(hipMemcpy(hids.data(), d_topidsb, (size_t)P * 16 * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(hws.data(), d_topwb, (size_t)P * 16 * 4, hipMemcpyDeviceToHost));
        auto pt2 = std::chrono::steady_clock::now();
        std::vector<int> off(E + 1, 0);
        for (int t = 0; t < P; t++)
          for (int s = 0; s < k; s++) off[hids[t * 16 + s] + 1]++;
        for (int e = 0; e < E; e++) off[e + 1] += off[e];
        std::vector<int> cur(off.begin(), off.end() - 1);
        std::vector<int> tidx(npairs);
        std::vector<float> pw(npairs);
        for (int t = 0; t < P; t++)
          for (int s = 0; s < k; s++) {
            int slot = cur[hids[t * 16 + s]]++;
            tidx[slot] = t;
            pw[slot] = hws[t * 16 + s];
          }
        CK(hipMemcpy(d_tokidx, tidx.data(), npairs * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_pweight, pw.data(), npairs * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(d_eoff, off.data(), (E + 1) * 4, hipMemcpyHostToDevice));
        if (moe_deterministic && !moe_ordered) {
          std::vector<int> pairids(npairs), next(P, 0);
          for (int p = 0; p < npairs; p++) {
            int t = tidx[p];
            pairids[t * k + next[t]++] = p;
          }
          CK(hipMemcpy(d_pairids, pairids.data(), npairs * 4, hipMemcpyHostToDevice));
        }
        if (prof) {
          auto pt3 = std::chrono::steady_clock::now();
          p_topk_wait +=
              std::chrono::duration<double, std::milli>(pt2 - pt1).count();
          p_bucket += std::chrono::duration<double, std::milli>(pt3 - pt2).count();
        }
      }
      bool tiled = !moe_untiled && !moe_ordered;
      // Packed bf16 x/hid no longer requires the tiled grid: the untiled
      // expert-major grid resolves rows from eoff and is bit-identical.
      bool packed = !moe_ordered && moe_deterministic && !moe_fp32_io;
      int tasks = (npairs + 63) / 64 + E;
      bool lt_ran = false;
      if (moe_lt && k == 10 && moe_deterministic && !moe_host_route && !moe_ordered &&
          P >= moe_lt_min) {
        lt_ran = true;
        // Dequant + per-expert hipBLASLt path (see kernels' banner comment).
        // One host readback per layer to get expert offsets for the launch
        // loop. Async into a pinned buffer + event, issued BEFORE the
        // convert+gather launches: the host waits only for the 2KB copy
        // while the GPU works through gather (~16 ms), instead of the GPU
        // draining idle before a blocking sync (~5.5 ms/layer of GPU gap).
        const int mid = g_cfg.moe_mid, dd = g_cfg.d;
        static int* hoff_pin = nullptr;
        static hipEvent_t ev_eoff = nullptr;
        if (!hoff_pin) {
          CK(hipHostMalloc((void**)&hoff_pin, (size_t)(E + 1) * 4,
                           hipHostMallocDefault));
          CK(hipEventCreateWithFlags(&ev_eoff, hipEventDisableTiming));
        }
        CK(hipMemcpyAsync(hoff_pin, d_eoff, (E + 1) * 4, hipMemcpyDeviceToHost,
                          g_str));
        CK(hipEventRecord(ev_eoff, g_str));
        if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
          k_f32_to_bf16_v4<<<((size_t)P * dd / 4 + 255) / 256, 256, 0, g_str>>>(
              d_xb, d_xbf16, dd, P, dd);
          xf16_src = d_xb;
          xf16_xs = xf16_K = dd;
          xf16_P = P;
        }
        k_moe_gather_bf16<<<npairs, 256, 0, g_str>>>(
            d_moexg, (const __hip_bfloat16*)d_xbf16, d_tokidx, dd);
        CK(hipEventSynchronize(ev_eoff));
        const int* hoff = hoff_pin;
        bool b16 = moe_lt_bf16;
        __hip_bfloat16* guv16 = (__hip_bfloat16*)d_guvb;
        // up phase: group dequant -> per-expert GEMM -> segment silu
        // QWENOX_MOE_LT_OVL pipeline: scratch is double-buffered (slots gi&1);
        // group gi+1's dequant runs on ovl_str while g_str runs group gi's
        // GEMMs. ovl_deq[b] = deq into slot b done; ovl_gemm[b] = GEMMs that
        // read slot b done. deq(gi+1) waits on the group gi-1 GEMMs (the
        // previous reader of its slot). g_str never touches slot b before
        // waiting ovl_deq[b], and by loop end has waited on every ovl_str
        // deq, so the next phase/layer (whose ovl_str work chains behind
        // these deqs and g_str's later GEMM event records) cannot race the
        // slots either. Prefill only runs outside graph capture.
        static hipStream_t ovl_str = nullptr;
        static hipEvent_t ovl_deq[2] = {nullptr, nullptr};
        static hipEvent_t ovl_gemm[2] = {nullptr, nullptr};
        if (moe_lt_ovl && !ovl_str) {
          // Non-blocking: g_str is the legacy default stream outside graph
          // capture, and a blocking side stream would implicitly serialize
          // with it (measured: -16% at 8K, no overlap at all).
          CK(hipStreamCreateWithFlags(&ovl_str, hipStreamNonBlocking));
          for (int i = 0; i < 2; i++) {
            CK(hipEventCreateWithFlags(&ovl_deq[i], hipEventDisableTiming));
            CK(hipEventCreateWithFlags(&ovl_gemm[i], hipEventDisableTiming));
          }
        }
        const int ngrp = (E + 3) / 4;
        auto deq_up = [&](int gi, hipStream_t st) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          int64_t total = (int64_t)g * 2 * mid * (dd / 32);
          k_moe_deq_bf16<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
              gu.data + 64 + (size_t)e0 * 2 * mid * (dd / 2),
              gu.data + 64 + gu.rows * gu.cols / 2 +
                  (size_t)e0 * 2 * mid * gu.scale_stride,
              (const float*)gu.data,
              d_moewup + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * 2 * mid * dd,
              g * 2 * mid, dd, gu.scale_stride);
        };
        auto deq_dn = [&](int gi, hipStream_t st) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          int64_t total = (int64_t)g * dd * (mid / 32);
          k_moe_deq_bf16<<<(unsigned)((total + 255) / 256), 256, 0, st>>>(
              dn.data + 64 + (size_t)e0 * dd * (mid / 2),
              dn.data + 64 + dn.rows * dn.cols / 2 +
                  (size_t)e0 * dd * dn.scale_stride,
              (const float*)dn.data,
              d_moewdn + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * dd * mid,
              g * dd, mid, dn.scale_stride);
        };
        if (moe_lt_ovl) {
          deq_up(0, ovl_str);
          CK(hipEventRecord(ovl_deq[0], ovl_str));
        }
        for (int gi = 0; gi < ngrp; gi++) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          __hip_bfloat16* wup =
              d_moewup + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * 2 * mid * dd;
          if (moe_lt_ovl) {
            if (gi + 1 < ngrp) {
              if (gi >= 1)
                CK(hipStreamWaitEvent(ovl_str, ovl_gemm[(gi + 1) & 1], 0));
              deq_up(gi + 1, ovl_str);
              CK(hipEventRecord(ovl_deq[(gi + 1) & 1], ovl_str));
            }
            CK(hipStreamWaitEvent(g_str, ovl_deq[gi & 1], 0));
          } else {
            deq_up(gi, g_str);
          }
          for (int e = e0; e < e0 + g; e++) {
            int n0 = hoff[e], m = hoff[e + 1] - n0;
            if (!m) continue;
            void* cup = b16 ? (void*)(guv16 + (size_t)n0 * 2 * mid)
                            : (void*)(d_guvb + (size_t)n0 * 2 * mid);
            gemm_bf16(wup + (size_t)(e - e0) * 2 * mid * dd,
                      d_moexg + (size_t)n0 * dd, cup, 2 * mid, dd, m, b16);
            int64_t seg = (int64_t)m * mid;
            if (b16)
              k_moe_silu_bf16<<<(unsigned)((seg + 255) / 256), 256, 0, g_str>>>(
                  guv16 + (size_t)n0 * 2 * mid,
                  (__hip_bfloat16*)d_hidb + (size_t)n0 * mid, mid, m);
            else
              k_moe_silu_f32_bf16<<<(unsigned)((seg + 255) / 256), 256, 0, g_str>>>(
                  d_guvb + (size_t)n0 * 2 * mid,
                  (__hip_bfloat16*)d_hidb + (size_t)n0 * mid, mid, m);
          }
          if (moe_lt_ovl) CK(hipEventRecord(ovl_gemm[gi & 1], g_str));
        }
        if (moe_lt_cmp && l == 0) {
          // Reference: fused packed up on identical inputs -> d_hidb2.
          k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
          k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0,
                                       g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2,
              (const float*)gu.data, d_xb, d_tokidx, d_eoff, nullptr, 2 * mid,
              gu.cols, mid, gu.scale_stride, d_moetiles, d_moentiles, d_xbf16,
              (uint16_t*)d_hidb2);
          CK(hipMemsetAsync(d_cmpi, 0, 8, g_str));
          int64_t nh = (int64_t)npairs * mid;
          k_cmp_absmax_bf16<<<(unsigned)std::min<int64_t>(1024, (nh + 255) / 256),
                              256, 0, g_str>>>(
              (const __hip_bfloat16*)d_hidb, (const __hip_bfloat16*)d_hidb2, nh,
              d_cmpi);
          int hc[2];
          CK(hipMemcpy(hc, d_cmpi, 8, hipMemcpyDeviceToHost));
          float fd, fr;
          memcpy(&fd, &hc[0], 4);
          memcpy(&fr, &hc[1], 4);
          fprintf(stderr, "moe_lt cmp up-hid: max|d|=%g max|ref|=%g\n", fd, fr);
        }
        // down phase: d_guvb is dead after silu; pair outputs alias it.
        if (moe_lt_ovl) {
          deq_dn(0, ovl_str);
          CK(hipEventRecord(ovl_deq[0], ovl_str));
        }
        for (int gi = 0; gi < ngrp; gi++) {
          int e0 = gi * 4, g = std::min(4, E - e0);
          __hip_bfloat16* wdn =
              d_moewdn + (size_t)(moe_lt_ovl ? (gi & 1) : 0) * 4 * dd * mid;
          if (moe_lt_ovl) {
            if (gi + 1 < ngrp) {
              if (gi >= 1)
                CK(hipStreamWaitEvent(ovl_str, ovl_gemm[(gi + 1) & 1], 0));
              deq_dn(gi + 1, ovl_str);
              CK(hipEventRecord(ovl_deq[(gi + 1) & 1], ovl_str));
            }
            CK(hipStreamWaitEvent(g_str, ovl_deq[gi & 1], 0));
          } else {
            deq_dn(gi, g_str);
          }
          for (int e = e0; e < e0 + g; e++) {
            int n0 = hoff[e], m = hoff[e + 1] - n0;
            if (!m) continue;
            void* cdn = b16 ? (void*)(guv16 + (size_t)n0 * dd)
                            : (void*)(d_guvb + (size_t)n0 * dd);
            gemm_bf16(wdn + (size_t)(e - e0) * dd * mid,
                      (const __hip_bfloat16*)d_hidb + (size_t)n0 * mid, cdn, dd,
                      mid, m, b16);
          }
          if (moe_lt_ovl) CK(hipEventRecord(ovl_gemm[gi & 1], g_str));
        }
        if (moe_lt_cmp && l == 0 && b16) {
          // Reference: fused packed down on LT's hid -> weighted fp32 pairs.
          k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks),
                                             256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
              (const float*)dn.data, nullptr, d_tokidx, d_pweight, d_eoff,
              d_pairs2, dd, dn.cols, dn.scale_stride, d_moetiles, d_moentiles,
              (const uint16_t*)d_hidb);
          CK(hipMemsetAsync(d_cmpi, 0, 8, g_str));
          int64_t npr = (int64_t)npairs * dd;
          k_cmp_absmax_pw<<<(unsigned)std::min<int64_t>(1024, (npr + 255) / 256),
                            256, 0, g_str>>>(
              d_pairs2, (const __hip_bfloat16*)d_guvb, d_pweight, npairs, dd,
              d_cmpi);
          int hc[2];
          CK(hipMemcpy(hc, d_cmpi, 8, hipMemcpyDeviceToHost));
          float fd, fr;
          memcpy(&fd, &hc[0], 4);
          memcpy(&fr, &hc[1], 4);
          fprintf(stderr, "moe_lt cmp down-pairs: max|d|=%g max|ref|=%g\n", fd,
                  fr);
        }
        if (b16)
          k_moe_reduce_pw_bf16<10><<<dim3((dd + 255) / 256, P), 256, 0, g_str>>>(
              (const __hip_bfloat16*)d_guvb, d_pairids, d_pweight, acc, P, dd);
        else
          k_moe_reduce_pw<10><<<dim3((dd + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, d_pweight, acc, P, dd);
      } else if (tiled) {
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
        if (packed) {
          if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
            k_f32_to_bf16_v4<<<((size_t)P * g_cfg.d / 4 + 255) / 256, 256, 0, g_str>>>(
                d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
            xf16_src = d_xb;
            xf16_xs = xf16_K = g_cfg.d;
            xf16_P = P;
          }
          // Hidden BF16 occupies the existing workspace until the shared expert rewrites it.
          if (moe_up_legacy)
            k_moe_w4_up<true, true, false, false><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
          else if (moe_up_no_table)
            k_moe_w4_up<true, true, true, false><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
          else
            k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
                gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
                d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
                gu.scale_stride, d_moetiles, d_moentiles, d_xbf16, (uint16_t*)d_hidb);
        } else {
          k_moe_w4_up<true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, d_hidb, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride, d_moetiles, d_moentiles);
        }
      } else {
        if (packed) {
          if (!(xf16_src == d_xb && xf16_xs == g_cfg.d && xf16_K == g_cfg.d && xf16_P == P)) {
            k_f32_to_bf16_v4<<<((size_t)P * g_cfg.d / 4 + 255) / 256, 256, 0, g_str>>>(
                d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
            xf16_src = d_xb;
            xf16_xs = xf16_K = g_cfg.d;
            xf16_P = P;
          }
          k_moe_w4_up<false, true><<<dim3(g_cfg.moe_mid / MOE_CT, E), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, nullptr, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride, nullptr, nullptr, d_xbf16, (uint16_t*)d_hidb);
        } else {
          k_moe_w4_up<<<dim3(g_cfg.moe_mid / MOE_CT, E), 256, 0, g_str>>>(
              gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
              d_xb, d_tokidx, d_eoff, d_hidb, 2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid,
              gu.scale_stride);
        }
      }
      if (lt_ran) {
        // LT branch already produced acc (down + weighted reduce).
      } else if (moe_ordered) {
        for (int e = 0; e < E; e++)
          k_moe_w4_down<<<dim3(g_cfg.d / FD_CT_DN, 1), 256, 0, g_str>>>(
              dn.data + 64 + (size_t)e * g_cfg.d * dn.cols / 2,
              dn.data + 64 + dn.rows * dn.cols / 2 + (size_t)e * g_cfg.d * dn.scale_stride,
              (const float*)dn.data, d_hidb, d_tokidx, d_pweight, d_eoff + e,
              acc, g_cfg.d, dn.cols, dn.scale_stride);
      } else if (moe_deterministic) {
        // QWENOX_MOE_PAIRS_BF16: down writes unweighted bf16 pair rows; the
        // router weight moves to the reduce (fp32 multiply). Requires the
        // fixed top-10 reduce below.
        bool pairs_bf16 = moe_pairs_bf16 && k == 10 && !moe_reduce_old;
        if (packed) {
          if (tiled) {
            if (moe_down_no_table) {
              if (pairs_bf16)
                k_moe_w4_down<false, true, true, false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
              else
                k_moe_w4_down<false, true, true, false><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
            } else {
              if (pairs_bf16)
                k_moe_w4_down<false, true, true, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
              else
                k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                    dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                    nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                    dn.scale_stride, d_moetiles, d_moentiles, (const uint16_t*)d_hidb);
            }
          } else {
            if (pairs_bf16)
              k_moe_w4_down<false, false, true, true, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                  dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                  nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                  dn.scale_stride, nullptr, nullptr, (const uint16_t*)d_hidb);
            else
              k_moe_w4_down<false, false, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                  dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                  nullptr, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                  dn.scale_stride, nullptr, nullptr, (const uint16_t*)d_hidb);
          }
        } else if (tiled) {
          if (pairs_bf16)
            k_moe_w4_down<false, true, false, false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride, d_moetiles, d_moentiles);
          else
            k_moe_w4_down<false, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride, d_moetiles, d_moentiles);
        } else {
          if (pairs_bf16)
            k_moe_w4_down<false, false, false, false, true><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride);
          else
            k_moe_w4_down<false><<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
                dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
                d_hidb, d_tokidx, d_pweight, d_eoff, d_guvb, g_cfg.d, dn.cols,
                dn.scale_stride);
        }
        if (pairs_bf16)
          k_moe_reduce_pw_bf16<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
              (const __hip_bfloat16*)d_guvb, d_pairids, d_pweight, acc, P, g_cfg.d);
        else if (k == 10 && !moe_reduce_old)
          k_moe_reduce_fast<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
              d_guvb, d_pairids, acc, P, g_cfg.d);
        else
          k_moe_reduce<<<(P * g_cfg.d + 255) / 256, 256, 0, g_str>>>(
              d_guvb, d_pairids, acc, P, k, g_cfg.d);
      } else {
        if (tiled) {
          k_moe_w4_down<true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
              d_hidb, d_tokidx, d_pweight, d_eoff, acc, g_cfg.d, dn.cols,
              dn.scale_stride, d_moetiles, d_moentiles);
        } else {
          k_moe_w4_down<<<dim3(g_cfg.d / FD_CT_DN, E), 256, 0, g_str>>>(
              dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2, (const float*)dn.data,
              d_hidb, d_tokidx, d_pweight, d_eoff, acc, g_cfg.d, dn.cols,
              dn.scale_stride);
        }
      }
    }
    moe_b_shared(l, P, acc, direct_output);
    return false;
  }

  // hgn v2 i4r MoE of trunk layer l for P rows of d_xb (topk in d_topidsb /
  // d_topwb): routed + shared expert -> acc f32 [P][d] (then d_yb unless
  // direct_output). Same dataflow as moe_v2_1; small P (verify) runs the
  // grouped GEMVs, larger P the kI4R LUT WMMA kernels on the moe_q4w_run
  // workspaces (moe_lut_up_mid writes the F16 down input straight to d_hidb;
  // d_guvb takes the down pairs). LUT path: the
  // shared expert runs first (its d_guvb / d_hidb scratch is dead by then) and
  // k_v2_reduce_rot folds reduce + rot_out + shared add into one pass
  // (QWENOX_MOE_SG_FUSE=0: shared expert afterwards via k_axpy_sg).
  // w4hc/Rhc: fuse the HC write-back into k_v2_reduce_rot_hc (B2-1) and
  // return true; only on the LUT sg_fuse path.
  bool moe_b_v2(int l, const WView& gu, const WView& dn, float* acc, int P,
                bool direct_output, const float* w4hc = nullptr, uint16_t* Rhc = nullptr,
                bool vseg = false) {
    const int k = g_cfg.topk, E = g_cfg.experts, d = g_cfg.d, mid = g_cfg.moe_mid;
    if (gu.cols != (uint64_t)d || gu.rows != (uint64_t)E * 2 * mid || dn.dtype != 23 ||
        dn.cols != (uint64_t)mid || dn.rows != (uint64_t)E * d || mid % 128 || d % 128 ||
        2 * mid > 2048)
      throw std::runtime_error("moe_b_v2: unexpected i4r expert shapes");
    const unsigned mid_threads = 2 * mid / 128 * 32;
    const moelut::LutW pgu = lut_i4r_view(gu.data, gu.rows, gu.cols, g_i4r_cb, 2 * mid, mid);
    const moelut::LutW pdn = lut_i4r_view(dn.data, dn.rows, dn.cols, g_i4r_cb, d, 0);
    // D2-S1c: vseg = batched multi-segment verify (R can exceed moe_naive_max
    // while each segment's serial run takes the per-row path) — force the
    // non-LUT per-row path so batched rows stay bit-identical to serial.
    // ngram 长验证块和短前缀重放须与 MTP 保持同一精度。
    const bool lut = !vseg && !spec_snap && !ngram_verify_active &&
                     moe_q4w && !moe_naive && P > moe_naive_max && k == 10 && moe_deterministic &&
                     !moe_host_route && !moe_ordered && !moe_untiled && P <= maxbatch_cap &&
                     mid % 64 == 0 &&
                     moelut::lut_ok(moelut::kI4R, pgu, true, d) &&
                     moelut::lut_ok(moelut::kI4R, pdn, false, mid);
    if (lut) {
      static bool logged = false;
      if (!logged) {
        logged = true;
        fprintf(stderr, "moe prefill: hgn v2 i4r LUT WMMA\n");
      }
      static const bool sg_fuse = !env_zero("QWENOX_MOE_SG_FUSE");
      KP("moe:shared");
#ifdef QWENOX_NPU
      // M2 重叠：共享专家 gate/up 提交给 NPU 后继续跑路由专家（占 MoE 时间
      // 大头），收尾压在 moe:reduce 前——NPU 的 ~1.5ms 被整个路由阶段盖住。
      // sg_fuse 关时共享专家在 moe_b_shared（路由之后）算，没有重叠点。
      bool npu_pend = false;
      if (sg_fuse) {
        char sg[128], su[128];
        snprintf(sg, sizeof sg, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
        snprintf(su, sizeof su, "layers.%d.mlp.shared_expert.up_proj.weight", l);
        npu_pend = npu_shexp_submit(l, P, sg, su);
      }
      if (sg_fuse && !npu_pend) moe_b_shared_ey(l, P);
#else
      if (sg_fuse) moe_b_shared_ey(l, P);
#endif
      KP("moe:route");
      moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff, d_pairids, P, k, E);
      // §11.5 多 tile 共享解码（mt）：tile 按 128 slot 合并（同专家两个 64-slot
      // tile 一个 CTA 解码一遍），逐 bit 不变；默认开，QWENOX_MOE_MT=0 回退。
      // microbench：P=2048 亏 5%、4096 +1.8%、8192 +7.5%、16384 +5.5%——按平均
      // 每专家 slot 数 >= 80 设门槛，小 P（tail chunk / 短 prompt）走原路径。
      // QWENOX_MOE_A4 —— 可选的 prefill 提速开关（hipfire IU4 路线），默认关。
      // 路由专家的激活按 128 元素量化成 int4（k_quant_a4），GEMM 走
      // v_wmma_i32_16x16x16_iu4（k_moe_iu4，gfx1151 上是 f16 WMMA 的 2 倍算力）。
      // 取值为位掩码：1 = gate/up，2 = down，3 = 两者。
      //   * 收益：64K 稳态 1609 → 1752（=3，+8.9%；=1 +6.8%，=2 +0.4%）。
      //   * 代价：不逐 bit。Windows KLD（bf16_c512，逐 token 走 prefill 打分）
      //     0.1075 → 0.1427（=3；=1 0.1280，=2 0.1226）。只影响 prompt 部分，
      //     生成的 token 走 decode，不经过这条路径；hipfire 在 27B 上的任务
      //     评测（HumanEval / GSM8K / MMLU-Pro）两条路径互有胜负，在噪声内。
      //   * 打开时 MoE 固定 BN64 tile（不走 mt/hyb）。
      // 并发：d_a4* 与 d_hidb / d_guvb 一样是 P 行 prefill 工作区，在一次 MoE
      // 调用内写完即读完；并发 slot 只在 chunk 之间或层边界（D1b 先排空 GPU）
      // 交接，交接时这些缓冲不含活数据，多路 prefill 共用一份即可。verify /
      // 合批 verify / D1b 借用的小 P 不进 LUT 路径。
      const int a4m = (!vseg && d_a4x && !env_zero("QWENOX_V2_PAIRS_F16")) ? moe_a4_env() : 0;
      const bool a4 = a4m != 0, a4_up = a4m & 1, a4_dn = a4m & 2;
      if (a4) {
        static bool la = false;
        if (!la) {
          la = true;
          fprintf(stderr,
                  "moe prefill: QWENOX_MOE_A4=%d W4A4 iu4 on (%s%s; faster prefill, "
                  "prompt KLD +0.015~0.035, decode unaffected)\n",
                  a4m, a4_up ? "up " : "", a4_dn ? "down" : "");
        }
      }
      static const bool moe_mt_env = !env_zero("QWENOX_MOE_MT");
      const bool mt = moe_mt_env && !a4 && (int64_t)P * k >= 80 * E;
      // Hybrid tiling (10-14, default on, QWENOX_MOE_HYB=0 = the mt/64 rule
      // above): full 128-slot super-tiles on the BN=128 kernels + one <=64-slot
      // remainder per expert on the BN=64 kernels. Bit-identical to both other
      // tilings (same per-(slot,row) K order). moe_var L20 harness, Windows:
      // up/down per token vs production P=2048 2.94/1.58 -> 2.93/1.53 us,
      // 4096 2.56/1.25 -> 2.47/1.26, 8192 2.30/1.14 -> 2.26/1.13.
      static const bool moe_hyb_env = !env_zero("QWENOX_MOE_HYB") && d_moetilesA;
      const bool hyb = moe_hyb_env && !vseg && !a4;
      int max_tiles;
      if (hyb) {
        k_moe_tiles_hyb<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, d_moetilesA,
                                              d_moentilesA, E);
        // B tiles hold > 64 slots each (or are full 128s): <= P*k/65 of them
        max_tiles = std::min((P * k) / 65 + 1, (P * k + 127) / 128 + E);
      } else {
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, mt ? 128 : 64);
        max_tiles = mt ? (P * k + 127) / 128 + E : (P * k + 63) / 64 + E;
      }
      const int max_tiles_a = std::min(E, P * k);
      __half* x16 = (__half*)d_xbf16b;
      __half* hid = (__half*)d_hidb;
      KP("moe:rot_in");
      v2_rot(d_xb, d, x16, d, gu.suh, nullptr, d, P, true);
      KP("moe:up");
      // up + mid fused (方向 3): hid = fwht(silu(g)*u*suh) 直接落盘，
      // 省掉 guv16 的写读往返和 k_i4r_mid 那次 launch，数学逐位一致。
      if (a4_up) {
        k_quant_a4<<<(unsigned)(((size_t)P * (d / 128) + 7) / 8), 256, 0, g_str>>>(x16, P, d, d_a4x,
                                                                                  d_a4xs);
        const Iu4W qgu{gu.data, gu.data + gu.rows * gu.cols / 2, 2 * mid, mid};
        k_moe_iu4<true><<<dim3(mid / 128, max_tiles), 256, 0, g_str>>>(
            qgu, d_a4x, d_a4xs, d_tokidx, d_moetiles, d_moentiles, hid, mid, d, gu.svh, dn.suh);
      } else if (a4) {
        moe_lut_up_mid(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, gu.svh, dn.suh,
                       mid, d, g_str);
      } else if (hyb) {
        moe_lut_up_mid_mt(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, gu.svh,
                          dn.suh, mid, d, g_str);
        moe_lut_up_mid(pgu, x16, d_tokidx, d_moetilesA, d_moentilesA, max_tiles_a, hid, gu.svh,
                       dn.suh, mid, d, g_str);
      } else if (mt)
        moe_lut_up_mid_mt(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, gu.svh,
                          dn.suh, mid, d, g_str);
      else
        moe_lut_up_mid(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, gu.svh, dn.suh,
                       mid, d, g_str);
      KP("moe:mid");
      KP("moe:down");
      // f16 pairs（down 写/reduce 读，v2-only，省一半 pairs 往返流量）。
      // 实测（Windows, 2026-10-02）：reduce −148 ms/16K，down 几乎不变，整体
      // +1.2~1.5%。终判（10-03，Linux BF16 基准 data/kld/bf16_c512.kld 64×512）：
      // mean_kld 0.106993(关) → 0.107705(开)，门槛 0.10699+0.002=0.10899 过线，
      // 默认开；QWENOX_V2_PAIRS_F16=0 回退 fp32 pairs。
      static const bool pairs_f16 = !env_zero("QWENOX_V2_PAIRS_F16");
      if (a4_dn) {
        k_quant_a4<<<(unsigned)(((size_t)P * k * (mid / 128) + 7) / 8), 256, 0, g_str>>>(
            hid, P * k, mid, d_a4h, d_a4hs);
        const Iu4W qdn{dn.data, dn.data + dn.rows * dn.cols / 2, d, 0};
        k_moe_iu4<false><<<dim3(d / 128, max_tiles), 256, 0, g_str>>>(
            qdn, d_a4h, d_a4hs, nullptr, d_moetiles, d_moentiles, (__half*)d_guvb, d, mid, nullptr,
            nullptr);
      } else if (a4) {
        moe_lut_down(moelut::kI4R, pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid,
                     g_str, true);
      } else if (hyb) {
        moe_lut_down_mt(pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid, g_str,
                        pairs_f16);
        moe_lut_down(moelut::kI4R, pdn, hid, d_moetilesA, d_moentilesA, max_tiles_a, d_guvb, d,
                     mid, g_str, pairs_f16);
      } else if (mt)
        moe_lut_down_mt(pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid, g_str,
                        pairs_f16);
      else
        moe_lut_down(moelut::kI4R, pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid,
                     g_str, pairs_f16);
      KP("moe:reduce");
#ifdef QWENOX_NPU
      // NPU 收尾压在 reduce 入队前：host 在此 WaitHost 时 GPU 仍在跑路由专家，
      // finish 的 down_proj+sgate 入队早于 reduce，与 M1 的 GPU 侧顺序一致。
      if (npu_pend) moe_b_shared_ey(l, P);
#endif
      const unsigned waves = (unsigned)P * (d / 128);
      static const int moe_rep = [] {  // diagnostic: QWENOX_MOE_REP=<layer>
        const char* e = getenv("QWENOX_MOE_REP");
        return e && *e ? atoi(e) : -1;
      }();
      if (l == moe_rep && !vseg) {
        auto tm = [&](const char* what, auto&& fn) {
          CK(hipStreamSynchronize(g_str));
          const auto t0 = std::chrono::steady_clock::now();
          for (int r = 0; r < 10; r++) fn();
          CK(hipStreamSynchronize(g_str));
          const double ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                  .count() / 10;
          fprintf(stderr, " %s %.3f ms (%.2f us/tok)", what, ms, ms * 1000 / P);
        };
        std::vector<int> eo(E + 1);
        int nt = 0;
        CK(hipMemcpy(eo.data(), d_eoff, (E + 1) * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(&nt, d_moentiles, 4, hipMemcpyDeviceToHost));
        int mx = 0, used = 0;
        for (int e = 0; e < E; e++) {
          mx = std::max(mx, eo[e + 1] - eo[e]);
          used += eo[e + 1] > eo[e];
        }
        fprintf(stderr, "moe_rep L%d P=%d mt=%d tiles=%d/%d experts_used=%d max_rows=%d:", l, P,
                (int)mt, nt, max_tiles, used, mx);
        tm("shared", [&] { moe_b_shared_ey(l, P); });
        tm("route", [&] {
          moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff, d_pairids, P, k, E);
          k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, mt ? 128 : 64);
        });
        tm("rot_in", [&] { v2_rot(d_xb, d, x16, d, gu.suh, nullptr, d, P, true); });
        tm("up", [&] {
          if (mt)
            moe_lut_up_mid_mt(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid,
                              gu.svh, dn.suh, mid, d, g_str);
          else
            moe_lut_up_mid(pgu, x16, d_tokidx, d_moetiles, d_moentiles, max_tiles, hid, gu.svh,
                           dn.suh, mid, d, g_str);
        });
        tm("down", [&] {
          if (mt)
            moe_lut_down_mt(pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d, mid, g_str,
                            pairs_f16);
          else
            moe_lut_down(moelut::kI4R, pdn, hid, d_moetiles, d_moentiles, max_tiles, d_guvb, d,
                         mid, g_str, pairs_f16);
        });
        fprintf(stderr, "\n");
        // tiling variants: V64 (cap 64, BN=64), V128 (cap 128, BN=128), HYB
        // (k_moe_tiles_hyb: full 128-tiles BN=128 + <=64 remainders BN=64).
        static MoeTile* tA = nullptr;
        static int *nA = nullptr, *nB = nullptr;
        static MoeTile* tB = nullptr;
        static __half *hid_ref = nullptr, *pr_ref = nullptr;
        static unsigned long long* d_cnt = nullptr;
        const size_t hid_n = (size_t)P * k * mid, pr_n = (size_t)P * k * d;
        if (!tA) {
          CK(hipMalloc(&tA, (size_t)E * sizeof(MoeTile)));
          CK(hipMalloc(&tB, ((size_t)maxbatch * k / 128 + E + 1) * sizeof(MoeTile)));
          CK(hipMalloc(&nA, 4));
          CK(hipMalloc(&nB, 4));
          CK(hipMalloc(&hid_ref, (size_t)maxbatch * k * mid * 2));
          CK(hipMalloc(&pr_ref, (size_t)maxbatch * k * d * 2));
          CK(hipMalloc(&d_cnt, 8));
        }
        const int mt64 = (P * k + 63) / 64 + E, mt128 = (P * k + 127) / 128 + E;
        auto run64 = [&](bool up, bool dn_) {
          if (up)
            moe_lut_up_mid(pgu, x16, d_tokidx, d_moetiles, d_moentiles, mt64, hid, gu.svh,
                           dn.suh, mid, d, g_str);
          if (dn_)
            moe_lut_down(moelut::kI4R, pdn, hid, d_moetiles, d_moentiles, mt64, d_guvb, d, mid,
                         g_str, true);
        };
        auto run128 = [&](bool up, bool dn_) {
          if (up)
            moe_lut_up_mid_mt(pgu, x16, d_tokidx, d_moetiles, d_moentiles, mt128, hid, gu.svh,
                              dn.suh, mid, d, g_str);
          if (dn_)
            moe_lut_down_mt(pdn, hid, d_moetiles, d_moentiles, mt128, d_guvb, d, mid, g_str, true);
        };
        auto runh = [&](bool up, bool dn_) {
          if (up) {
            moe_lut_up_mid_mt(pgu, x16, d_tokidx, tB, nB, mt128, hid, gu.svh, dn.suh, mid, d,
                              g_str);
            moe_lut_up_mid(pgu, x16, d_tokidx, tA, nA, E, hid, gu.svh, dn.suh, mid, d, g_str);
          }
          if (dn_) {
            moe_lut_down_mt(pdn, hid, tB, nB, mt128, d_guvb, d, mid, g_str, true);
            moe_lut_down(moelut::kI4R, pdn, hid, tA, nA, E, d_guvb, d, mid, g_str, true);
          }
        };
        auto mism = [&](const void* a, const void* b, size_t n) {
          CK(hipMemsetAsync(d_cnt, 0, 8, g_str));
          k_u32_mismatch<<<1024, 256, 0, g_str>>>((const uint32_t*)a, (const uint32_t*)b, n / 2,
                                                  d_cnt);
          unsigned long long h = 0;
          CK(hipMemcpy(&h, d_cnt, 8, hipMemcpyDeviceToHost));
          return h;
        };
        k_moe_tiles_hyb<<<1, 512, 0, g_str>>>(d_eoff, tB, nB, tA, nA, E);
        int hnA = 0, hnB = 0;
        CK(hipMemcpy(&hnA, nA, 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(&hnB, nB, 4, hipMemcpyDeviceToHost));
        // reference = V64 outputs
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
        run64(true, true);
        CK(hipMemcpyAsync(hid_ref, hid, hid_n * 2, hipMemcpyDeviceToDevice, g_str));
        CK(hipMemcpyAsync(pr_ref, d_guvb, pr_n * 2, hipMemcpyDeviceToDevice, g_str));
        fprintf(stderr, "moe_var L%d P=%d hyb tiles B=%d A=%d:", l, P, hnB, hnA);
        tm("v64.up", [&] { run64(true, false); });
        tm("v64.dn", [&] { run64(false, true); });
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 128);
        tm("v128.up", [&] { run128(true, false); });
        tm("v128.dn", [&] { run128(false, true); });
        const unsigned long long m128 = mism(hid, hid_ref, hid_n) + mism(d_guvb, pr_ref, pr_n);
        tm("hyb.up", [&] { runh(true, false); });
        tm("hyb.dn", [&] { runh(false, true); });
        const unsigned long long mh = mism(hid, hid_ref, hid_n) + mism(d_guvb, pr_ref, pr_n);
        fprintf(stderr, " | mismatch v128=%llu hyb=%llu\n", m128, mh);
        // restore the production tiling for the rest of this layer
        k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, mt ? 128 : 64);
        if (mt) run128(true, true); else run64(true, true);
      }
      if (sg_fuse && Rhc) {
        if (pairs_f16)
          k_v2_reduce_rot_hc<10, __half><<<(waves + 7) / 8, 256, 0, g_str>>>(
              (const __half*)d_guvb, d_pairids, d_pweight, dn.svh, d_eyb, d_sgb, w4hc,
              Rhc, P, d);
        else
          k_v2_reduce_rot_hc<10><<<(waves + 7) / 8, 256, 0, g_str>>>(
              d_guvb, d_pairids, d_pweight, dn.svh, d_eyb, d_sgb, w4hc, Rhc, P, d);
        KP("moe:tail");
        return true;
      }
      if (pairs_f16)
        k_v2_reduce_rot<10, __half><<<(waves + 7) / 8, 256, 0, g_str>>>(
            (const __half*)d_guvb, d_pairids, d_pweight, dn.svh, sg_fuse ? d_eyb : nullptr,
            d_sgb, acc, P, d);
      else
        k_v2_reduce_rot<10><<<(waves + 7) / 8, 256, 0, g_str>>>(
            d_guvb, d_pairids, d_pweight, dn.svh, sg_fuse ? d_eyb : nullptr, d_sgb, acc, P, d);
      KP("moe:tail");
      if (sg_fuse) {
        if (!direct_output)
          CK(hipMemcpyAsync(d_yb, acc, (size_t)P * d * 4, hipMemcpyDeviceToDevice, g_str));
      } else {
        moe_b_shared(l, P, acc, direct_output);
      }
      return false;
    } else {
      const XCodes cg = xcodes(gu), cd = xcodes(dn);
      const bool direct_down = moe_down_topk && k == 10;
      if (!direct_down) CK(hipMemsetAsync(acc, 0, (size_t)P * d * 4, g_str));
      v2_rot(d_xb, d, d_eyb, d, gu.suh, nullptr, d, P);  // d_eyb: free until the shared expert
      const uint64_t pairs = (uint64_t)P * k * mid;
      k_q4cp_gemv_gg<2><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          cg.codes, cg.scales, cg.cb, d_eyb, d_guvb, d_topidsb, (uint64_t)2 * mid, gu.cols,
          gu.scale_stride, P * k, (uint64_t)d, 16, k, nullptr);
      k_i4r_mid<float, false><<<P * k, mid_threads, 0, g_str>>>(d_guvb, 2 * mid, d_hidb, mid,
                                                                gu.svh, dn.suh, mid);
      if (direct_down) {
        const uint64_t row_pairs = (d + 1) / 2, warps_per_block = 128 / 32;
        k_q4cp_gemv_gd_topk_h16<10, 2><<<
            (unsigned)(((uint64_t)P * row_pairs + warps_per_block - 1) / warps_per_block), 128, 0,
            g_str>>>(cd.codes, cd.scales, cd.cb, d_hidb, acc, d_topidsb, d_topwb, (uint64_t)d,
                     dn.cols, dn.scale_stride, (uint64_t)mid, P, 16, (uint64_t)d);
      } else {
        const uint64_t pairs2 = (uint64_t)P * k * ((d + 1) / 2);
        k_q4cp_gemv_gd<2><<<(unsigned)((pairs2 + 15) / 16), 512, 0, g_str>>>(
            cd.codes, cd.scales, cd.cb, d_hidb, acc, d_topidsb, d_topwb, (uint64_t)d, dn.cols,
            dn.scale_stride, (uint64_t)mid, P * k, 16, (uint64_t)d, k);
      }
    }
    v2_rot(acc, d, acc, d, nullptr, dn.svh, d, P);
    moe_b_shared(l, P, acc, direct_output);
    return false;
  }

  // Shared expert of trunk layer l added into acc (routed sum), then acc -> d_yb.
  void moe_b_shared(int l, int P, float* acc, bool direct_output) {
    moe_b_shared_ey(l, P);
    k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(acc, d_eyb, d_sgb,
                                                         g_cfg.d, P * g_cfg.d);
    if (!direct_output)
      CK(hipMemcpyAsync(d_yb, acc, (size_t)P * g_cfg.d * 4, hipMemcpyDeviceToDevice,
                        g_str));
  }
  // Shared-expert gate + up projections: P rows of x (row stride K) -> yg / yu
  // ([P][rows] each). QWENOX_DENSE_WMMA: one k_ht_wmma pair launch on every
  // path (decode gemv, draft, verify -- the pair's numerics differ from two
  // separate launches, so all P must take it); otherwise, or when the pair
  // does not apply (non-ht weights), exactly the old two calls: gemv() at the
  // single-row decode sites (one_row), gemm() at the batched ones.
  void se_gate_up(const char* gname, const char* uname, const float* x, float* yg, float* yu,
                  int P, bool one_row) {
    const WView g = wview(ckpt.at(gname)), u = wview(ckpt.at(uname));
    static const bool no_mr = getenv("QWENOX_NO_GEMVMR") != nullptr;
    if (dense_wmma_on() && (one_row || (!no_mr && P <= gemvmr_maxp())) &&
        ht_wmma_pair(g, u, x, false, g.cols, yg, yu, g.rows, P))
      return;
    if (one_row && gemv2(g, u, x, yg, yu)) return;  // D4: same dtype/cols
    if (one_row) {
      gemv(g, x, yg);
      gemv(u, x, yu);
    } else {
      gemm(g, x, yg, P);
      gemm(u, x, yu, P);
    }
  }
#ifdef QWENOX_NPU
  // ---- NPU 卸载（29_npu.inc；运行时 QWENOX_NPU=1 启用，QWENOX_NPU_MINP 调
  // 行数门槛，默认 4097——NPU kernel 固定算满 4096 行，P 不超过它时纯亏）----
  std::unique_ptr<NpuGemm> npu_;
  bool npu_off_ = false;  // 初始化或运行失败过一次 -> 永久回退 GPU
  int npu_a_ = -1, npu_cg_ = -1, npu_cu_ = -1;
  std::vector<int> npu_bg_, npu_bu_;  // 每层 gate/up 的 B 缓冲
  std::vector<char> npu_wready_;      // 每层：B 已编码
  float* d_npu_se_ = nullptr;         // [maxbatch][1280] gate|up 两半（M2 收尾用：
                                      // d_guvb/d_hidb 那时被路由专家 pairs 占用）
  int npu_pend_l_ = -1, npu_pend_P_ = 0;  // 已提交未收尾的共享专家（M2 重叠）
  // M3：GDN z 门控（in_proj_z 6144x2560 = 6 片 N=1024）。gdn_proj_in_b 提交，
  // gdn_b 的 gated norm（z 消费点）前收尾：NPU 的 ~4.4ms 被 qkv GEMM +
  // conv + scan（~42ms）盖住。B 缓冲按 GDN 序号 gi = l-(l+1)/4 索引。
  // 实测稳态仅 +0.4%（gdn:proj -95ms/chunk，被带宽争抢抵消大半），且 NPU
  // 高频提交在多租户下触发过整机挂死——默认关，QWENOX_NPU_Z=1 开。
  std::vector<std::array<int, 6>> npu_bz_;  // [ngdn][6] z 的 B 切片
  std::vector<char> npu_zready_;            // [ngdn]
  int npu_cz_[6] = {-1, -1, -1, -1, -1, -1};
  int npu_pend_z_l_ = -1, npu_pend_z_P_ = 0;  // 已提交未收尾的 z
  static bool npu_z_on() {
    static const bool on = [] {
      const char* s = getenv("QWENOX_NPU_Z");
      return s && *s && strcmp(s, "0") != 0;
    }();
    return on;
  }
  // M4：QSA q/gate（q_proj 12288x2560 = q 6 片 + gate 6 片），分半提交：
  // 单段 NPU 连续负载压到 6 片 ~4.4ms（12 片 ~9ms 在独占下也硬死机过）。
  // q 半在 qsa_idx_b 入口提交（窗口 = qsa:idx ~5ms + k/v GEMM），qsplit 前
  // 收尾；gate 半在 qsa:flash 前提交（~200ms 窗口），sigmoid gate 前收尾。
  // 依赖 gate_bf16 路径（默认开；QWENOX_QSA_GATE_BF16=0 时 qsplit 提前消费
  // gate 半，与分半冲突 -> 不卸载）。B 缓冲按 QSA 序号 qi=l/4。
  // QWENOX_NPU_QG=1 开（默认关：分半提交版仍触发硬死机，方向已封，
  // 见 NPU-WIN-FEASIBILITY.md §6f；仅 slot<0 && r0==0 的正常 prefill 提交）。
  std::vector<std::array<int, 12>> npu_bqg_;  // [nqsa][12]
  std::vector<char> npu_qgready_;             // [nqsa]
  int npu_cqg_[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
  int npu_pend_qg_l_ = -1, npu_pend_qg_P_ = 0;  // 已提交未收尾的 qg
  int npu_pend_qg_stage_ = 0;  // 1=q 半在飞 2=q 半已收尾 3=gate 半在飞
  static bool npu_qg_on() {
    static const bool on = [] {
      const char* s = getenv("QWENOX_NPU_QG");
      return s && *s && strcmp(s, "0") != 0;
    }();
    return on;
  }

  static int npu_minp() {
    static const int v = [] {
      const char* s = getenv("QWENOX_NPU_MINP");
      return s && *s ? atoi(s) : 4097;
    }();
    return v;
  }

  bool npu_init() {
    std::string err;
    npu_.reset(NpuGemm::Create(err));
    if (!npu_) {
      fprintf(stderr, "%s NPU 卸载不可用：%s（回退 GPU）\n", log_ts(), err.c_str());
      npu_off_ = true;
      return false;
    }
    npu_a_ = npu_->NewBuffer(NpuGemm::kABytes);
    npu_cg_ = npu_->NewBuffer(NpuGemm::kCBytes);
    npu_cu_ = npu_->NewBuffer(NpuGemm::kCBytes);
    bool ok = npu_a_ >= 0 && npu_cg_ >= 0 && npu_cu_ >= 0;
    npu_bg_.resize(g_cfg.layers, -1);
    npu_bu_.resize(g_cfg.layers, -1);
    for (int l = 0; ok && l < g_cfg.layers; ++l) {
      npu_bg_[l] = npu_->NewBuffer(NpuGemm::kBBytes);
      npu_bu_[l] = npu_->NewBuffer(NpuGemm::kBBytes);
      ok = npu_bg_[l] >= 0 && npu_bu_[l] >= 0;
    }
    const int ngdn = g_cfg.layers - g_cfg.layers / 4;
    if (ok && npu_z_on()) {
      npu_bz_.resize(ngdn);
      npu_zready_.assign(ngdn, 0);
      for (int i = 0; ok && i < 6; ++i) {
        npu_cz_[i] = npu_->NewBuffer(NpuGemm::kCBytes);
        ok = npu_cz_[i] >= 0;
      }
      for (int gi = 0; ok && gi < ngdn; ++gi)
        for (int s = 0; s < 6; ++s) {
          npu_bz_[gi][s] = npu_->NewBuffer(NpuGemm::kBBytes);
          ok = npu_bz_[gi][s] >= 0;
        }
    }
    const int nqsa = g_cfg.layers / 4;
    if (ok && npu_qg_on()) {
      npu_bqg_.resize(nqsa);
      npu_qgready_.assign(nqsa, 0);
      for (int i = 0; ok && i < 12; ++i) {
        npu_cqg_[i] = npu_->NewBuffer(NpuGemm::kCBytes);
        ok = npu_cqg_[i] >= 0;
      }
      for (int qi = 0; ok && qi < nqsa; ++qi)
        for (int s = 0; s < 12; ++s) {
          npu_bqg_[qi][s] = npu_->NewBuffer(NpuGemm::kBBytes);
          ok = npu_bqg_[qi][s] >= 0;
        }
    }
    if (!ok) {
      fprintf(stderr, "%s NPU 卸载不可用：固定页分配失败（回退 GPU）\n", log_ts());
      npu_off_ = true;
      npu_.reset();
      return false;
    }
    // A 缓冲清一次零：NPU kernel 固定读满 4096 行，本 chunk 未编码的行得是零
    memset(npu_->Host(npu_a_), 0, NpuGemm::kABytes);
    npu_wready_.assign(g_cfg.layers, 0);
    CK(hipMalloc(&d_npu_se_, (size_t)maxbatch * 1280 * 4));
    const double mb =
        (g_cfg.layers * 2.0 * NpuGemm::kBBytes + NpuGemm::kABytes +
         2.0 * NpuGemm::kCBytes +
         (npu_z_on() ? (double)ngdn * 6 * NpuGemm::kBBytes + 6.0 * NpuGemm::kCBytes : 0) +
         (npu_qg_on() ? (double)nqsa * 12 * NpuGemm::kBBytes + 12.0 * NpuGemm::kCBytes : 0)) /
        1e6;
    fprintf(stderr,
            "%s NPU 卸载就绪：共享专家 gate/up（%d 层）%s%s（%.0f MB 固定页，P>=%d 时 "
            "前 4096 行给 NPU）\n",
            log_ts(), g_cfg.layers, npu_z_on() ? " + GDN z 门控（6 片/层）" : "",
            npu_qg_on() ? " + QSA q/gate（12 片/层）" : "", mb,
            npu_minp());
    return true;
  }

  // 层 l 的 gate/up 权重编码进 pinned B 缓冲（每层一次；gemm_wbf16 把任意
  // dtype 反量化成 bf16——数值与 GPU GEMM 的输入一致，之后才是 NPU 的
  // 块浮点量化，误差由 KLD 把关）。
  bool npu_enc_layer(int l, const char* gname, const char* uname) {
    const WView ws[2] = {wview(ckpt.at(gname)), wview(ckpt.at(uname))};
    if (ws[0].rows != 640 || ws[0].cols != 2560 || ws[1].rows != 640 ||
        ws[1].cols != 2560) {
      npu_off_ = true;  // 形状不符（不是本模型）：整体回退
      return false;
    }
    const int bids[2] = {npu_bg_[l], npu_bu_[l]};
    constexpr size_t groups = (size_t)2560 * 1024 / 8;
    for (int i = 0; i < 2; ++i) {
      const uint16_t* wbf = gemm_wbf16(ws[i]);
      k_npu_enc_w<<<(unsigned)((groups + 255) / 256), 256, 0, g_str>>>(
          wbf, (uint8_t*)npu_->Dev(bids[i]), 640);
    }
    CK(hipStreamSynchronize(g_str));
    npu_->FinalizeWeights(bids[0]);
    npu_->FinalizeWeights(bids[1]);
    npu_wready_[l] = 1;
    return true;
  }

  // M2 提交：编码 A（行 [0, min(P,4096))）后把 gate/up 两次 run 交给 worker，
  // 立即返回——NPU 与后续 GPU 路由专家阶段重叠。true = 已挂上
  // npu_pend_，调用方必须在消费 d_eyb/d_sgb 之前调 npu_shexp_finish()。
  bool npu_shexp_submit(int l, int P, const char* gname, const char* uname) {
    static const bool want = [] {
      const char* s = getenv("QWENOX_NPU");
      return s && *s && strcmp(s, "0") != 0;
    }();
    if (!want || npu_off_ || P < npu_minp() || npu_pend_l_ >= 0) return false;
    if (!npu_ && !npu_init()) return false;
    if (!npu_->Healthy()) {
      npu_off_ = true;
      return false;
    }
    if (!npu_wready_[l] && !npu_enc_layer(l, gname, uname)) return false;
    const int nr = (int)std::min<uint64_t>(P, NpuGemm::kM);
    const unsigned mt = (nr + kNpuTileM - 1) / kNpuTileM;
    k_npu_enc_a<<<mt * (2560 / kNpuTileK), 256, 0, g_str>>>(
        d_xb, (uint8_t*)npu_->Dev(npu_a_), nr, 2560);
    const int bs[2] = {npu_bg_[l], npu_bu_[l]}, cs[2] = {npu_cg_, npu_cu_};
    if (!npu_->Submit(g_str, npu_a_, bs, cs, 2)) {
      npu_off_ = true;  // host func 没排上：什么都没写，安全回退
      return false;
    }
    npu_pend_l_ = l;
    npu_pend_P_ = P;
    return true;
  }

  // M2 收尾（也供 M1 提交后立即调用）：GPU 算剩余行（GEMM 行独立）-> 等
  // NPU -> 解码到 d_npu_se_ 两半 -> silu 到 d_hidb。之后 d_eyb/d_sgb 由
  // moe_b_shared_ey 尾部公共代码产出。d_npu_se_ 而不是 d_guvb：收尾发生在
  // 路由专家阶段之后，d_guvb 正装着路由 pairs 等 reduce。
  void npu_shexp_finish(int l, int P) {
    npu_pend_l_ = -1;
    const int nr = (int)std::min<uint64_t>(P, NpuGemm::kM);
    char buf[128], ub[128];
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
    snprintf(ub, sizeof ub, "layers.%d.mlp.shared_expert.up_proj.weight", l);
    if (P > nr) {
      const WView g = wview(ckpt.at(buf)), u = wview(ckpt.at(ub));
      gemm(g, d_xb + (size_t)nr * 2560, d_npu_se_ + (size_t)nr * 640, P - nr);
      gemm(u, d_xb + (size_t)nr * 2560, d_npu_se_ + (size_t)P * 640 + (size_t)nr * 640,
           P - nr);
    }
    if (!npu_->Wait(g_str)) {  // host func 排不上：CPU 侧直接等，结果仍有效
      npu_->WaitHost();
      npu_off_ = true;
    }
    constexpr size_t groups = (size_t)NpuGemm::kM * NpuGemm::kN / 8;
    constexpr unsigned blocks = (unsigned)((groups + 255) / 256);
    k_npu_dec_c<<<blocks, 256, 0, g_str>>>((const uint8_t*)npu_->Dev(npu_cg_), d_npu_se_,
                                           nr, 640, 640);
    k_npu_dec_c<<<blocks, 256, 0, g_str>>>((const uint8_t*)npu_->Dev(npu_cu_),
                                           d_npu_se_ + (size_t)P * 640, nr, 640, 640);
    k_silu_mul<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(d_npu_se_, d_npu_se_ + (size_t)P * 640,
                                                         d_hidb, P * 640);
  }

  // ---- M3：GDN z 门控（in_proj_z 6144x2560 = 6 片 N=1024）------------------
  // 提交/收尾都在 gdn_proj_in_b 内：NPU 的 ~4.4ms 被同函数的 qkv GEMM
  // （~12ms）盖住，无需跨函数 pend。z 消费点（gated norm）在 conv/scan 之后，
  // 收尾只是等 NPU + 解码，不进关键路径。
  bool npu_enc_z(int gi, const WView& w) {
    const uint16_t* wbf = gemm_wbf16(w);  // [6144][2560] bf16（d_wbf16）
    constexpr size_t groups = (size_t)2560 * 1024 / 8;
    for (int s = 0; s < 6; ++s)
      k_npu_enc_w<<<(unsigned)((groups + 255) / 256), 256, 0, g_str>>>(
          wbf + (size_t)s * 1024 * 2560, (uint8_t*)npu_->Dev(npu_bz_[gi][s]), 1024);
    CK(hipStreamSynchronize(g_str));
    for (int s = 0; s < 6; ++s) npu_->FinalizeWeights(npu_bz_[gi][s]);
    npu_zready_[gi] = 1;
    return true;
  }

  // 返回 NPU 覆盖的行数 nr（0 = 未提交，调用方走全 GPU z GEMM）。
  int npu_z_submit(int l, int P, const float* xb) {
    static const bool want = [] {
      const char* s = getenv("QWENOX_NPU");
      return s && *s && strcmp(s, "0") != 0;
    }();
    if (!want || !npu_z_on() || npu_off_ || P < npu_minp()) return 0;
    if (!npu_ && !npu_init()) return 0;
    if (!npu_->Healthy()) {
      npu_off_ = true;
      return 0;
    }
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    const WView w = wview(ckpt.at(buf));
    if (w.rows != 6144 || w.cols != 2560) {
      npu_off_ = true;  // 形状不符（不是本模型）：整体回退
      return 0;
    }
    const int gi = l - (l + 1) / 4;
    if (!npu_zready_[gi] && !npu_enc_z(gi, w)) return 0;
    const int nr = (int)std::min<uint64_t>(P, NpuGemm::kM);
    const unsigned mt = (nr + kNpuTileM - 1) / kNpuTileM;
    k_npu_enc_a<<<mt * (2560 / kNpuTileK), 256, 0, g_str>>>(
        xb, (uint8_t*)npu_->Dev(npu_a_), nr, 2560);
    if (!npu_->Submit(g_str, npu_a_, npu_bz_[gi].data(), npu_cz_, 6)) {
      npu_off_ = true;  // host func 没排上：什么都没写，安全回退
      return 0;
    }
    return nr;
  }

  // 等 NPU + 把 6 条 C 流解码进 zb 的列区间 [s*1024, s*1024+1024)。
  void npu_z_finish(float* zb, int nr) {
    if (!npu_->Wait(g_str)) {  // host func 排不上：CPU 侧直接等，结果仍有效
      npu_->WaitHost();
      npu_off_ = true;
    }
    constexpr size_t groups = (size_t)NpuGemm::kM * NpuGemm::kN / 8;
    constexpr unsigned blocks = (unsigned)((groups + 255) / 256);
    for (int s = 0; s < 6; ++s)
      k_npu_dec_c<<<blocks, 256, 0, g_str>>>(
          (const uint8_t*)npu_->Dev(npu_cz_[s]), zb + s * 1024, nr, 1024, 6144);
  }

  // zb 首个消费点前的兜底收尾：提交与消费之间隔着 conv+scan，NPU 有充足窗口。
  // 层/P 不匹配（不该发生）时禁用 NPU 并用 GPU 全量重算 z 保正确性。
  void npu_z_pend_finish(int l, int P, const float* xb, float* zb) {
    if (npu_pend_z_l_ < 0) return;
    const int pl = npu_pend_z_l_, pP = npu_pend_z_P_;
    npu_pend_z_l_ = -1;
    if (pl == l && pP == P) {
      npu_z_finish(zb, (int)std::min<uint64_t>(P, NpuGemm::kM));
      return;
    }
    fprintf(stderr,
            "[npu] z pend mismatch: submitted l=%d P=%d, consumed l=%d P=%d; "
            "disabling NPU\n",
            pl, pP, l, P);
    npu_->WaitHost();
    npu_off_ = true;
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.linear_attn.in_proj_z.weight", l);
    gemm(wview(ckpt.at(buf)), xb, zb, P);
  }

  // ---- M4：QSA q/gate（q_proj [12288][2560]，q=片 0..5，gate=片 6..11）----
  bool npu_enc_qg(int qi, const WView& w) {
    const uint16_t* wbf = gemm_wbf16(w);  // [12288][2560] bf16（d_wbf16）
    constexpr size_t groups = (size_t)2560 * 1024 / 8;
    for (int s = 0; s < 12; ++s)
      k_npu_enc_w<<<(unsigned)((groups + 255) / 256), 256, 0, g_str>>>(
          wbf + (size_t)s * 1024 * 2560, (uint8_t*)npu_->Dev(npu_bqg_[qi][s]),
          1024);
    CK(hipStreamSynchronize(g_str));
    for (int s = 0; s < 12; ++s) npu_->FinalizeWeights(npu_bqg_[qi][s]);
    npu_qgready_[qi] = 1;
    return true;
  }

  // q 半提交（qsa_idx_b 入口，xb 此刻就绪；到 qsplit 消费还有整个
  // indexer + k/v GEMM 的窗口）。只交片 0..5，单段 ~4.4ms。
  int npu_qg_submit(int l, int P, const float* xb) {
    static const bool want = [] {
      const char* s = getenv("QWENOX_NPU");
      return s && *s && strcmp(s, "0") != 0;
    }();
    if (!want || !npu_qg_on() || npu_off_ || P < npu_minp()) return 0;
    if (npu_pend_qg_l_ >= 0) return 0;  // 上一个还没收尾（不该发生）：不叠提交
    static const bool gate_bf16_ok = !env_zero("QWENOX_QSA_GATE_BF16");
    if (!gate_bf16_ok) return 0;  // qsplit 会提前读 gate 半，与分半提交冲突
    if (!npu_ && !npu_init()) return 0;
    if (!npu_->Healthy()) {
      npu_off_ = true;
      return 0;
    }
    char buf[128];
    snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
    const WView w = wview(ckpt.at(buf));
    if (w.rows != 12288 || w.cols != 2560) {
      npu_off_ = true;  // 形状不符（不是本模型）：整体回退
      return 0;
    }
    const int qi = l / 4;
    if (!npu_qgready_[qi] && !npu_enc_qg(qi, w)) return 0;
    const int nr = (int)std::min<uint64_t>(P, NpuGemm::kM);
    const unsigned mt = (nr + kNpuTileM - 1) / kNpuTileM;
    k_npu_enc_a<<<mt * (2560 / kNpuTileK), 256, 0, g_str>>>(
        xb, (uint8_t*)npu_->Dev(npu_a_), nr, 2560);
    if (!npu_->Submit(g_str, npu_a_, npu_bqg_[qi].data(), npu_cqg_, 6)) {
      npu_off_ = true;  // host func 没排上：什么都没写，安全回退
      return 0;
    }
    npu_pend_qg_l_ = l;
    npu_pend_qg_P_ = P;
    npu_pend_qg_stage_ = 1;
    return nr;
  }

  // 本层有未收尾 qg 时返回 NPU 覆盖行数（qsa_main_b 据此跳过/补尾 GPU gemm）
  int npu_qg_pend_rows(int l) const {
    return npu_pend_qg_l_ == l
               ? (int)std::min<uint64_t>(npu_pend_qg_P_, NpuGemm::kM)
               : 0;
  }

  // q 半收尾（qsplit 前）：Wait + 解码片 0..5 到 d_qgb 列 [0,6144)。
  // 层不匹配（不该发生）：禁 NPU + GPU 全量重算 q_proj 保正确性。
  void npu_qg_finish_q(int l) {
    if (npu_pend_qg_l_ < 0) return;
    if (npu_pend_qg_l_ != l) {
      fprintf(stderr,
              "[npu] qg pend mismatch: submitted l=%d, consumed l=%d; "
              "disabling NPU\n",
              npu_pend_qg_l_, l);
      npu_->WaitHost();
      npu_off_ = true;
      char buf[128];
      snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight",
               npu_pend_qg_l_);
      gemm(wview(ckpt.at(buf)), d_xb, d_qgb, npu_pend_qg_P_);
      npu_pend_qg_l_ = -1;
      npu_pend_qg_stage_ = 0;
      return;
    }
    if (npu_pend_qg_stage_ != 1) return;
    if (!npu_->Wait(g_str)) {  // host func 排不上：CPU 侧直接等，结果仍有效
      npu_->WaitHost();
      npu_off_ = true;
    }
    const int nr = (int)std::min<uint64_t>(npu_pend_qg_P_, NpuGemm::kM);
    constexpr size_t groups = (size_t)NpuGemm::kM * NpuGemm::kN / 8;
    constexpr unsigned blocks = (unsigned)((groups + 255) / 256);
    for (int s = 0; s < 6; ++s)
      k_npu_dec_c<<<blocks, 256, 0, g_str>>>(
          (const uint8_t*)npu_->Dev(npu_cqg_[s]), d_qgb + s * 1024, nr, 1024,
          12288);
    npu_pend_qg_stage_ = 2;
  }

  // gate 半提交（qsa:flash 前，~200ms 窗口）。失败则 GPU 全量重算覆盖行
  // （q 半已被 qsplit 消费，重写无害；gate 列必须补上）。
  void npu_qg_submit_g(int l, const float* xb) {
    if (npu_pend_qg_l_ != l || npu_pend_qg_stage_ != 2) return;
    const int nr = (int)std::min<uint64_t>(npu_pend_qg_P_, NpuGemm::kM);
    const unsigned mt = (nr + kNpuTileM - 1) / kNpuTileM;
    k_npu_enc_a<<<mt * (2560 / kNpuTileK), 256, 0, g_str>>>(
        xb, (uint8_t*)npu_->Dev(npu_a_), nr, 2560);
    if (!npu_->Submit(g_str, npu_a_, npu_bqg_[l / 4].data() + 6, npu_cqg_ + 6,
                      6)) {
      fprintf(stderr, "[npu] qg gate submit 失败，GPU 重算（禁用 NPU）\n");
      npu_off_ = true;
      char buf[128];
      snprintf(buf, sizeof buf, "layers.%d.self_attn.q_proj.weight", l);
      gemm(wview(ckpt.at(buf)), d_xb, d_qgb, nr);
      npu_pend_qg_l_ = -1;
      npu_pend_qg_stage_ = 0;
      return;
    }
    npu_pend_qg_stage_ = 3;
  }

  // gate 半收尾（sigmoid gate 前，qsa:flash 已跑完）：解码片 6..11。
  void npu_qg_finish_g(int l) {
    if (npu_pend_qg_l_ != l || npu_pend_qg_stage_ != 3) return;
    if (!npu_->Wait(g_str)) {
      npu_->WaitHost();
      npu_off_ = true;
    }
    const int nr = (int)std::min<uint64_t>(npu_pend_qg_P_, NpuGemm::kM);
    constexpr size_t groups = (size_t)NpuGemm::kM * NpuGemm::kN / 8;
    constexpr unsigned blocks = (unsigned)((groups + 255) / 256);
    for (int s = 6; s < 12; ++s)
      k_npu_dec_c<<<blocks, 256, 0, g_str>>>(
          (const uint8_t*)npu_->Dev(npu_cqg_[s]), d_qgb + s * 1024, nr, 1024,
          12288);
    npu_pend_qg_l_ = -1;
    npu_pend_qg_stage_ = 0;
  }

  // 调试路径（vgdump/qsa_loop）用：q 收尾 -> gate 提交 -> gate 收尾。
  void npu_qg_finish_all(int l, const float* xb) {
    npu_qg_finish_q(l);
    npu_qg_submit_g(l, xb);
    npu_qg_finish_g(l);
  }
#endif  // QWENOX_NPU

  // Shared expert output d_eyb [P][d] and gate logit d_sgb [P].
  void moe_b_shared_ey(int l, int P) {
    char buf[128];
    // shared expert (gate/up scratch in d_guvb head, silu out in d_hidb head)
    {
      char ub[128];
      snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.gate_proj.weight", l);
      snprintf(ub, sizeof ub, "layers.%d.mlp.shared_expert.up_proj.weight", l);
#ifdef QWENOX_NPU
      bool npu_did = false;
      if (npu_pend_l_ >= 0) {
        if (npu_pend_l_ == l && npu_pend_P_ == P) {  // M2：moe_b_v2 入口已提交，收尾
          npu_shexp_finish(l, P);
          npu_did = true;
        } else {  // 挂着不匹配的提交——不该发生；丢弃在飞结果，永久回退
          fprintf(stderr, "%s NPU: pend L%d/P%d != L%d/P%d，丢弃并回退 GPU\n", log_ts(),
                  npu_pend_l_, npu_pend_P_, l, P);
          npu_->WaitHost();
          npu_pend_l_ = -1;
          npu_off_ = true;
        }
      } else if (npu_shexp_submit(l, P, buf, ub)) {
        npu_shexp_finish(l, P);  // 无重叠点的调用点：提交完立即收尾
        npu_did = true;
      }
      if (!npu_did) {
#endif
      if (se_gate_up_merged(wview(ckpt.at(buf)), wview(ckpt.at(ub)), P)) {
        k_silu_mul_g<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(d_guvb, d_hidb, 640, P);
      } else {
        se_gate_up(buf, ub, d_xb, d_guvb, d_guvb + (size_t)P * 640, P, false);
        k_silu_mul<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(d_guvb, d_guvb + (size_t)P * 640,
                                                             d_hidb, P * 640);
      }
#ifdef QWENOX_NPU
      }
#endif
    }
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert.down_proj.weight", l);
    gemm(wview(ckpt.at(buf)), d_hidb, d_eyb, P);
    snprintf(buf, sizeof buf, "layers.%d.mlp.shared_expert_gate.weight", l);
    gemm(wview(ckpt.at(buf)), d_xb, d_sgb, P);
  }

  // PLE device phase for one prefill token (embeddings already in d_eb).
  void ple_gpu_t(int t, int base = 0) {
    const float* e = d_eb + (size_t)t * 2560;
    gemv(wview(ckpt.at("layers.1.ple.key_proj.weight")), e, d_keys);
    gemv(wview(ckpt.at("layers.1.ple.value_proj.weight")), e, d_v);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_keys, d_ple_nk, d_kn, 2560,
                                                g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_Rb + (size_t)t * 10240,
                                                d_ple_nk + 10240, d_qn, 2560,
                                                g_cfg.norm_eps, 4);
    k_ple_gate<<<4, 1024, 0, g_str>>>(d_kn, d_qn, d_v, d_U, 2560, 4);
    k_rmsnorm_zc_ring<<<4, 1024, 0, g_str>>>(d_U, d_ple_nk + 20480, d_ring,
                                             d_ringarr + base + t, 2560,
                                             g_cfg.norm_eps);
    k_ple_conv<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_ring, d_ringarr + base + t,
                                                       d_ple_convw, d_convout, 10240);
    k_add2<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_Rb + (size_t)t * 10240, d_U,
                                                   d_convout, 10240);
  }

  // Batched PLE over all P positions (embeddings already in d_eb). Bitwise
  // matches the ple_gpu_t loop: same per-block reductions, ring semantics of
  // k_ple_conv (tap back-9 aliases the current slot) folded into k_ple_conv_b.
  void ple_gpu_b(int P, int base = 0) {
    gemm(wview(ckpt.at("layers.1.ple.key_proj.weight")), d_eb, d_keysb, P);
    gemm(wview(ckpt.at("layers.1.ple.value_proj.weight")), d_eb, d_plevb, P);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_keysb, d_ple_nk, d_keysb, 2560,
                                                    g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_Rb, d_ple_nk + 10240, d_qnb,
                                                    2560, g_cfg.norm_eps, 4);
    k_ple_gate_b<<<dim3(4, P), 1024, 0, g_str>>>(d_keysb, d_qnb, d_plevb, d_Ub, 2560,
                                                 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_Ub, d_ple_nk + 20480, d_Unb,
                                                    2560, g_cfg.norm_eps, 4);
    int64_t tot = (int64_t)P * 10240;
    k_ple_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
        d_Unb, d_ple_convw, d_convoutb, P, 10240, base, d_ring, vbptr());
    k_add2<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(d_Rb, d_Ub, d_convoutb,
                                                             (int)tot);
    k_ple_ring_wr<<<9, 256, 0, g_str>>>(d_Unb, d_ring, P, 10240, base, vbptr());
    if (cut_r_)  // in-pass cut: PLE ring after the first r rows
      k_ple_ring_wr<<<9, 256, 0, g_str>>>(d_Unb, d_cring, cut_r_, 10240, base, vbptr());
  }

  // D2-S1c: one segment of a multi-segment verify batch's PLE device phase.
  // The segment's rows live at [r0, r0+P) of the batch buffers; ring is the
  // segment slot's own; the base goes by scalar (vb=nullptr). Bit-identical
  // to ple_gpu_b(P, base) on those rows: every kernel is row-parallel and the
  // conv/ring kernels take the same base by scalar instead of *d_vbase.
  // d_Rb is read through the member at call time (the bf16 path swaps it to
  // the f32 d_Rhatb for the PLE phase, as in enq_ple_device_b).
  void ple_gpu_b_vseg(int P, int base, int r0, float* ring) {
    float* const eb = d_eb + (size_t)r0 * 2560;
    float* const Rb = d_Rb + (size_t)r0 * 10240;
    float* const keysb = d_keysb + (size_t)r0 * 10240;
    float* const plevb = d_plevb + (size_t)r0 * 10240;
    float* const qnb = d_qnb + (size_t)r0 * 10240;
    float* const Ub = d_Ub + (size_t)r0 * 10240;
    float* const Unb = d_Unb + (size_t)r0 * 10240;
    float* const convoutb = d_convoutb + (size_t)r0 * 10240;
    gemm(wview(ckpt.at("layers.1.ple.key_proj.weight")), eb, keysb, P);
    gemm(wview(ckpt.at("layers.1.ple.value_proj.weight")), eb, plevb, P);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(keysb, d_ple_nk, keysb, 2560,
                                                    g_cfg.norm_eps, 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(Rb, d_ple_nk + 10240, qnb,
                                                    2560, g_cfg.norm_eps, 4);
    k_ple_gate_b<<<dim3(4, P), 1024, 0, g_str>>>(keysb, qnb, plevb, Ub, 2560, 4);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(Ub, d_ple_nk + 20480, Unb,
                                                    2560, g_cfg.norm_eps, 4);
    int64_t tot = (int64_t)P * 10240;
    k_ple_conv_b<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(
        Unb, d_ple_convw, convoutb, P, 10240, base, ring, nullptr);
    k_add2<<<(unsigned)((tot + 255) / 256), 256, 0, g_str>>>(Rb, Ub, convoutb,
                                                             (int)tot);
    k_ple_ring_wr<<<9, 256, 0, g_str>>>(Unb, ring, P, 10240, base, nullptr);
  }

  // P3-B: the layer's HC read + attention norm as a standalone step. A
  // verify-graph segment boundary runs this eagerly ahead of qsa_idx_b,
  // because the indexer reads d_xb, which this norm writes (it also carries
  // layer l-1's deferred MLP write-back under gr_scatter_norm).
  void enq_layer_norm_b(int l, int P, bool fuse_prev_mlp) {
    char attn_prefix[128], mlp_prefix[128];
    snprintf(attn_prefix, sizeof attn_prefix,
             "layers.%d.attn_hyper_connection", l);
    KP("hc");
    if (fuse_prev_mlp) {
      snprintf(mlp_prefix, sizeof mlp_prefix,
               "layers.%d.mlp_hyper_connection", l - 1);
      gr_write_read_b(mlp_prefix, attn_prefix, P);
    } else {
      gr_read_b(attn_prefix, P);
    }
    if (getenv("QWENOX_VGDUMP")) {
      char tag[12];
      snprintf(tag, sizeof tag, "nx%d", l);
      vg_dump_buf(tag, P, d_xb, (size_t)P * 2560 * 4);
      snprintf(tag, sizeof tag, "xn%d", l);
      vg_dump_buf(tag, P, d_xbf16, (size_t)P * 2560 * 2);
      snprintf(tag, sizeof tag, "rn%d", l);
      vg_dump_buf(tag, P, d_Rhatbf16, (size_t)P * 10240 * 2);
      snprintf(tag, sizeof tag, "gb%d", l);
      vg_dump_buf(tag, P, d_Gb, (size_t)P * 10240 * 4);
    }
  }

  void enq_layer_b(int l, int P, bool fuse_prev_mlp = false,
                   bool flush_mlp = true, int base = 0,
                   const int* dvb = nullptr, bool skip_qsa_idx = false) {
    vz_base_ = base;  // QWENOX_VIZ: this batch's row positions
    vz_dvb_ = dvb;
    vz_rowtab_ = nullptr;
    vz_rows_ = true;
    char attn_prefix[128], mlp_prefix[128];
    snprintf(attn_prefix, sizeof attn_prefix,
             "layers.%d.attn_hyper_connection", l);
    // skip_qsa_idx (P3-B): norm + indexer ran eagerly between graph segments
    // (enq_layer_norm_b + qsa_idx_b); start at the attention projections.
    // Callers pass it only for QSA layers.
    // QWENOX_SEG_REP=<l1>,<l2>: diagnostic stage timing (re-runs each stage of
    // layer l 10x after a sync; corrupts the output -- timing only).
    static const std::string seg_rep = [] {
      const char* e = getenv("QWENOX_SEG_REP");
      return e ? "," + std::string(e) + "," : std::string();
    }();
    const bool srep = !seg_rep.empty() &&
                      seg_rep.find("," + std::to_string(l) + ",") != std::string::npos;
    auto stm = [&](const char* what, auto&& fn) {
      if (!srep) return;
      CK(hipStreamSynchronize(g_str));
      const auto t0 = std::chrono::steady_clock::now();
      const auto s_ic = injp_cur;
      const auto s_if = injp_for;
      const auto s_hw = hcw_site_;
      const auto s_x0 = xf16_src;
      const auto s_x1 = xf16_xs;
      const auto s_x2 = xf16_P;
      const auto s_x3 = xf16_K;
      for (int r = 0; r < 10; r++) {
        fn();
        injp_cur = s_ic;
        injp_for = s_if;
        hcw_site_ = s_hw;
        xf16_src = s_x0;
        xf16_xs = s_x1;
        xf16_P = s_x2;
        xf16_K = s_x3;
      }
      CK(hipStreamSynchronize(g_str));
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
          10;
      fprintf(stderr, " %s %.3f ms (%.3f us/tok)", what, ms, ms * 1000 / P);
    };
    if (srep) fprintf(stderr, "seg_rep L%d P=%d base=%d:", l, P, base);
    if (!skip_qsa_idx) enq_layer_norm_b(l, P, fuse_prev_mlp);
    if (!skip_qsa_idx) stm("hc", [&] { enq_layer_norm_b(l, P, fuse_prev_mlp); });
    if (srep && !skip_qsa_idx) {
      float* hwn = ensure_hcnorm(attn_prefix);
      stm("hc.mix", [&] { gr_mix_b(attn_prefix, P, hwn, false); });
      char dbuf[160];
      snprintf(dbuf, sizeof dbuf, "%s.input_mix_weight_down.weight", attn_prefix);
      const WView wdn = wview(ckpt.at(dbuf));
      stm("hc.down", [&] { gemm(wdn, d_Rhatb, d_t320b, P, 0, d_Rhatbf16); });
      snprintf(dbuf, sizeof dbuf, "%s.input_mix_weight_up.weight", attn_prefix);
      const WView wup = wview(ckpt.at(dbuf));
      stm("hc.up", [&] { hc_up_fused(wup, P); });
    }
    KP(is_qsa(l) ? "qsa" : "gdn");
    if (is_qsa(l)) stm("qsa", [&] { qsa_b(l, l / 4, P, base, dvb); });
    else stm("gdn", [&] { gdn_b(l, l - (l + 1) / 4, P); });
    if (is_qsa(l)) {
      if (abl_has("qsa")) {
      } else if (skip_qsa_idx)  // P3-B: indexer ran eagerly between graph segments
        qsa_main_b(l, l / 4, P, base, dvb);
      else
        qsa_b(l, l / 4, P, base, dvb);
    } else if (!abl_has("gdn"))
      gdn_b(l, l - (l + 1) / 4, P);
    if (dbg) {
      h_rms("gpu", l, "attn-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "attn-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    snprintf(mlp_prefix, sizeof mlp_prefix,
             "layers.%d.mlp_hyper_connection", l);
    KP("hc2");
    stm("hc2", [&] {
      if (gr_scatter_norm) gr_write_read_b(attn_prefix, mlp_prefix, P);
    });
    if (abl_has("hc2")) {
    } else if (gr_scatter_norm)
      gr_write_read_b(attn_prefix, mlp_prefix, P);
    else {
      gr_write_b(attn_prefix, P);
      gr_read_b(mlp_prefix, P);
    }
    {  // D2-S1b bisect: QWENOX_VGDUMP_GL=<layer> → attn-HC 后 / MoE 后的 d_Rb
      static const int vgl2 = [] {
        const char* e = getenv("QWENOX_VGDUMP_GL");
        return e ? atoi(e) : -1;
      }();
      if (l == vgl2) vg_dump_row("ra", P, base);
    }
    KP("moe");
    // B2-1 (QWENOX_HC_FUSE_WRITE, 默认关 — KPROF 实测回退，见 HANDOFF-PREFILL-ALL
    // §4 B2-1 状态段): fold the mlp site's HC write-back into the moe reduce
    // (saves the d_yb f32 write + re-read; R bit-identical). Requires the
    // gr_bf16 write path; the deferred (flush_mlp=false) consumer additionally
    // needs the inj-fused scatter's Write=false mode. moe_b fuses only on the
    // LUT sg_fuse paths and reports back; otherwise the precomputed d_w4b is
    // recomputed bit-identically by the normal write path.
    static const bool hcw_env =
        getenv("QWENOX_HC_FUSE_WRITE") != nullptr && !env_zero("QWENOX_HC_FUSE_WRITE");
    const bool can_hcw = hcw_env && !dbg && gr_bf16 &&
                         (flush_mlp || (gr_scatter_norm && inj_fuse_on()));
    bool hcw = false;
    stm("moe", [&] { moe_b(l, P); });
    if (srep) fprintf(stderr, "\n");
    if (abl_has("moe")) {
    } else if (can_hcw) {
      gr_write_prep_b(mlp_prefix, P);
      hcw = moe_b(l, P, true);
    } else {
      moe_b(l, P);
    }
    if (dbg) {
      h_rms("gpu", l, "mlp-x", d_xb + (size_t)(P - 1) * 2560, 2560);
      h_rms("gpu", l, "mlp-y", d_yb + (size_t)(P - 1) * 2560, 2560);
    }
    KP("hc3");
    {
      static const int vgl3 = [] {
        const char* e = getenv("QWENOX_VGDUMP_GL");
        return e ? atoi(e) : -1;
      }();
      if (l == vgl3) vg_dump_row("rm", P, base);  // D2-S1b: MoE 后的 d_Rb
    }
    if (flush_mlp) {
      if (!hcw) gr_write_b(mlp_prefix, P);
      else df2_hc_capture(mlp_prefix, P);  // moe_b already applied the write
      if (srep && flush_mlp && !hcw && !vg_capturing_) {
        stm("hc3.prep", [&] { gr_write_prep_b(mlp_prefix, P); });
        if (gr_bf16)
          stm("hc3.wr", [&] {
            k_gr_write_b_hc_bf16<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
                d_w4b, (uint16_t*)d_Rb, d_yb, g_cfg.d, g_cfg.branches, P);
          });
      }
    } else if (hcw) {
      hcw_site_ = mlp_prefix;
    }
    if (!dbg && !bench_quiet && isatty(2)) {  // per-layer \r is terminal-only; logs use prefill-live
      fprintf(stderr, "\rprefill L%02d/%d", l + 1, g_cfg.layers);
      fflush(stderr);
    }
  }

  // D2-S1c: one layer of a multi-segment verify batch. Row-parallel ops (HC
  // read/write, MoE) run once over the R = sum(Ps) concatenated rows; the
  // state operators (QSA indexer+attention, GDN recurrence) run per segment
  // with (slot, r0) so each touches only its own slot's state and rows.
  // Bit-identical to each segment's serial enq_layer_b(l, P_s, ...) on its
  // rows: gemm/norm are row-invariant (ktest ri_ anchors), moe_b is forced to
  // the row-invariant naive path (vseg=true, same path the serial P<=8 verify
  // takes), hcw fusion stays off (try_hcw=false).
  void enq_layer_vseg(int l, const int* Ps, const int* bases, const int* sl,
                      const int* voffs, int nseg, int R, bool fuse_prev_mlp,
                      bool flush_mlp) {
    if (l == 0 && viz::on()) {  // QWENOX_VIZ: rows of several sequences
      vz_rowtab_ = viz::rows_table(Ps, bases, sl, voffs, nseg, R);
      vz_rows_ = vz_rowtab_ != nullptr;
    }
    char attn_prefix[128], mlp_prefix[128];
    snprintf(attn_prefix, sizeof attn_prefix,
             "layers.%d.attn_hyper_connection", l);
    enq_layer_norm_b(l, R, fuse_prev_mlp);
    KP(is_qsa(l) ? "qsa" : "gdn");
    // Projections once over all R rows (one weight stream instead of one
    // per segment); only the state ops run per segment on their slot state.
    // QWENOX_VSEG_PROJ_SPLIT=0: old per-segment projections (A/B control).
    static const bool hoist = !env_zero("QWENOX_VSEG_PROJ_SPLIT");
    if (is_qsa(l)) {
      if (hoist && !qsa_loop) {
        qsa_proj_in_b(l, l / 4, R);
        const bool batt = qsa_attn_vseg_ok(l / 4, sl, nseg);
        for (int s = 0; s < nseg; s++)
          qsa_b(l, l / 4, Ps[s], bases[s], nullptr, sl[s], voffs[s],
                /*state_only=*/true, /*defer_attn=*/batt);
        if (batt) qsa_attn_vseg(l / 4, Ps, bases, sl, voffs, nseg, R);
        qsa_proj_out_b(l, R);
      } else {
        for (int s = 0; s < nseg; s++)
          qsa_b(l, l / 4, Ps[s], bases[s], nullptr, sl[s], voffs[s]);
      }
    } else {
      const int gi = l - (l + 1) / 4;
      if (hoist) {
        gdn_proj_in_b(l, R, d_xb, d_qkvb, d_g48b, d_beta48b, d_zb);
        if (!gdn_state_vseg(l, gi, Ps, sl, voffs, nseg, R))
          for (int s = 0; s < nseg; s++)
            gdn_b(l, gi, Ps[s], sl[s], voffs[s], /*state_only=*/true);
        gdn_proj_out_b(l, R, d_convb, d_yb, d_xbf16b);
      } else {
        for (int s = 0; s < nseg; s++) gdn_b(l, gi, Ps[s], sl[s], voffs[s]);
      }
    }
    snprintf(mlp_prefix, sizeof mlp_prefix,
             "layers.%d.mlp_hyper_connection", l);
    KP("hc2");
    if (gr_scatter_norm)
      gr_write_read_b(attn_prefix, mlp_prefix, R);
    else {
      gr_write_b(attn_prefix, R);
      gr_read_b(mlp_prefix, R);
    }
    KP("moe");
    moe_b(l, R, false, true);  // vseg: row-invariant naive path, no hcw
    KP("hc3");
    if (flush_mlp) gr_write_b(mlp_prefix, R);
  }

  // Batched prefill: returns first generated token (argmax at last position).
  // Batched prefill: returns first generated token (argmax at last position).
  // base = absolute position of tokens[0] (KV-continuation); state handoff:
  // d_R = last row, pos = base+P, d_pos/d_ringpos absolute, chist/ring/KV/GDN
  // state fully advanced.
  // Chunked prefill: prompts longer than maxbatch are processed in
  // maxbatch-token continuations (the same path KV reuse uses), so batch
  // workspace stays chunk-sized regardless of maxctx.
  // 1s-cadence prefill progress: record a completion probe after layer l's
  // enqueue, and once a second report the highest COMPLETED layer as an
  // estimated token count. hipEventQuery is non-blocking (hipErrorNotReady
  // is the expected answer for in-flight layers), so this never serializes
  // the pipeline the way the QWENOX_PHASE sync profiling does.
  void prog_print(int P) {
    auto now = std::chrono::steady_clock::now();
    double iv = std::chrono::duration<double>(now - prog_last).count();
    if (iv < 1.0) return;
    int done = -1;
    for (int k = g_cfg.layers - 1; k >= 0; --k)
      if (prog_ev[k] && hipEventQuery(prog_ev[k]) == hipSuccess) {
        done = k;
        break;
      }
    // done+1 of g_cfg.layers layers finished (L0 counts as one layer).
    double frac = double(done + 1) / g_cfg.layers;
    int64_t est = prog_base + (int64_t)(P * frac);
    double el = std::chrono::duration<double>(now - prog_t0).count();
    fprintf(stderr,
            "%s req %lld | prompt processing | n_tokens = %lld/%d (%.0f%%) "
            "| t = %.1f s | %.0f tok/s (avg %.0f)\n",
            log_ts(), (long long)prog_req, (long long)est, prog_total,
            100.0 * est / prog_total, el, (est - prog_last_est) / iv,
            el > 0 ? est / el : 0.0);
    perfsnap::write_prefill((uint64_t)prog_req, (est - prog_last_est) / iv,
                            (uint64_t)est, (uint64_t)prog_total);
    crashdiag::note("req %lld prefill %lld/%d", (long long)prog_req, (long long)est, prog_total);
    prog_last = now;
    prog_last_est = est;
  }
  void prog_probe(int l, int P) {
    if (l < 0 || l >= 64) return;
    if (!prog_ev[l])
      CK(hipEventCreateWithFlags(&prog_ev[l], hipEventDisableTiming));
    CK(hipEventRecord(prog_ev[l], g_str));
    prog_print(P);
  }
  // The layer enqueue is fully asynchronous: every probe above fires within
  // the first fraction of a second of host time, then the host blocks on the
  // final readback. Real progress is therefore reported while WAITING for
  // the graph to drain — poll the last layer's event here at 1 s cadence.
  void prog_wait(int P) {
    if (!prog_ev[g_cfg.layers - 1]) return;
    // 1 ms poll (was 20 ms: up to 20 ms of TTFT per served chunk end, i.e. at
    // the snapshot cut and again at the prompt end). prog_print self-limits
    // to its 1 s cadence.
    while (hipEventQuery(prog_ev[g_cfg.layers - 1]) == hipErrorNotReady) {
      prog_print(P);
      std::this_thread::sleep_for(std::chrono::microseconds(1000));
    }
  }

  // ---- PLE gather flights ---------------------------------------------------
  // cat() window for a flight: token at window position t-back, falling back
  // to the two history tokens carried with the flight (pad_id at sequence
  // start). Matches the chist[ple_base0 + t - back] reads of the old inline
  // gather exactly.
  static int64_t ple_cat(const PleFlight& fl, int t, int back) {
    int j = t - back;
    if (fl.nseg > 0) {
      // Multi-segment flight: the window of token t is its own segment, so a
      // look-back past the segment start reads that segment's history pair —
      // the same bytes a per-segment flight would gather.
      int s = 0;
      while (s + 1 < fl.nseg && t >= fl.seg_start[s + 1]) s++;
      const int r = j - fl.seg_start[s];
      return r >= 0 ? fl.toks[j] : (r == -1 ? fl.seg_pre0[s] : fl.seg_pre1[s]);
    }
    return j >= 0 ? fl.toks[j] : (j == -1 ? fl.pre0 : fl.pre1);
  }
  // per-token NOSORT fallback, window variant of ple_rows_fp8.
  void ple_rows_fp8_win(const PleFlight& fl, int t, uint8_t* dst) {
    const uint8_t* addrs[16];
    for (int h = 0; h < 16; h++) {
      int64_t mix = ple_cat(fl, t, 0) * ple_mult[0] ^ (ple_cat(fl, t, 1) * ple_mult[1]);
      if (h >= 8) mix ^= ple_cat(fl, t, 2) * ple_mult[2];
      uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
      const uint8_t* p = ple_table->data + row * g_ple_rb;
      addrs[h] = p;
      madvise((void*)(uintptr_t)((uintptr_t)p & ~4095ull), 4096, MADV_WILLNEED);
    }
#ifdef _WIN32
    if (g_ple_win.ok()) {
      uint64_t a64[16];
      for (int h = 0; h < 16; h++) a64[h] = (uint64_t)(uintptr_t)addrs[h];
      g_ple_win.run(a64, dst, (uint64_t)(uintptr_t)ple_table->data, 16);
      return;
    }
#endif
    for (int h = 0; h < 16; h++) memcpy(dst + h * g_ple_rb, addrs[h], g_ple_rb);
  }
  // pre0/pre1 = the two tokens preceding the window (pad_id at stream start).
  // D2-S1c: takes the history explicitly so a non-bound slot's chist works.
  void ple_pre01(const std::vector<int64_t>& ch, int64_t& pre0, int64_t& pre1) const {
    pre0 = ch.empty() ? pad_id : ch.back();
    pre1 = ch.size() < 2 ? pad_id : ch[ch.size() - 2];
  }
  // Spawn the disk gather for toks (global position base) into staging fp8.
  // Same three pipelines as the old inline gather (IOCP/io_uring, madvise
  // two-phase, NOSORT per-token); the flight owns all worker-visible state so
  // it can be joined from a later prefill_chunk call.
  void ple_spawn(PleFlight& fl, std::vector<int>&& toks, int base, int64_t pre0, int64_t pre1,
                 uint8_t* fp8, int buf, const PleSegs* segs = nullptr) {
    fl.toks = std::move(toks);
    fl.slot = bound_;
    fl.base = base;
    fl.P = (int)fl.toks.size();
    fl.pre0 = pre0;
    fl.pre1 = pre1;
    fl.nseg = segs ? segs->n : 0;
    if (segs) {
      for (int s = 0; s <= segs->n; s++) fl.seg_start[s] = segs->start[s];
      for (int s = 0; s < segs->n; s++) {
        fl.seg_pre0[s] = segs->pre0[s];
        fl.seg_pre1[s] = segs->pre1[s];
      }
    }
    fl.fp8 = fp8;
    fl.buf = buf;
    fl.start = std::chrono::steady_clock::now();
    if (ple_stg_ev_ok[buf]) {
      // Staging was last consumed by an earlier chunk's dequant; in steady
      // state that kernel finished two chunks ago, so this never blocks.
      CK(hipEventSynchronize(ple_stg_ev[buf]));
    }
    const int P = fl.P;
    PleFlight* f = &fl;
    // Address shards: one thread per token up to 16 (a 5-row verify used to
    // spawn 16 threads, 11 with no tokens; each token's rows are computed
    // independently, so the staging bytes do not depend on nt).
    const int nt = std::max(1, std::min(16, P));
    fl.workers.store(nt);
    fl.pool.clear();
    fl.pool.reserve(nt);
    if (!getenv("QWENOX_PLE_NOSORT")) {
      fl.barrier.store(0, std::memory_order_relaxed);
      if (ple_uring_ready()) {
#ifdef _WIN32
        // Small batch (decode / spec verify / batched verify, 16*P <= 512
        // rows): row addresses on this thread, every read posted to the IOCP
        // now, completions collected in ple_join -- no threads. Same rows, same
        // staging bytes as the threaded pump below. QWENOX_PLE_INLINE=0: off.
        static const bool ple_inline = !env_zero("QWENOX_PLE_INLINE");
        if (ple_inline && (size_t)P * 16 <= PleWin::DEPTH) {
          fl.addrs.resize((size_t)P * 16);
          for (int t = 0; t < P; t++)
            for (int h = 0; h < 16; h++) {
              int64_t mix = ple_cat(*f, t, 0) * ple_mult[0] ^ (ple_cat(*f, t, 1) * ple_mult[1]);
              if (h >= 8) mix ^= ple_cat(*f, t, 2) * ple_mult[2];
              uint64_t row = (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
              fl.addrs[(size_t)t * 16 + h] =
                  (uint64_t)(uintptr_t)(ple_table->data + row * g_ple_rb);
            }
          PLE_PUMP.issue(fl.addrs.data(), fl.fp8, (uint64_t)(uintptr_t)ple_table->data,
                         (size_t)P * 16);
          fl.inline_io = true;
          fl.workers.store(0);
          fl.valid = true;
          return;
        }
#endif
        // IOCP/io_uring batch-pread gather: phase A is the same 16-thread row
        // address computation; the pump threads then push all 16*P rows in
        // token order through the deep rings straight into pinned staging
        // (same multi-ring split as the inline gather; tiny P keeps one ring).
        fl.addrs.resize((size_t)P * 16);
        const int npump = P >= 64 ? std::min(nt, ple_pump_count()) : 1;
        fl.workers.store(npump);  // only the pump threads account for fl.done
        for (int w = 0; w < nt; w++)
          fl.pool.emplace_back([this, f, w, npump, nt] {
            const int t0 = (int)((size_t)w * f->P / nt),
                      t1 = (int)((size_t)(w + 1) * f->P / nt);
            for (int t = t0; t < t1; t++)
              for (int h = 0; h < 16; h++) {
                int64_t mix = ple_cat(*f, t, 0) * ple_mult[0] ^
                              (ple_cat(*f, t, 1) * ple_mult[1]);
                if (h >= 8) mix ^= ple_cat(*f, t, 2) * ple_mult[2];
                uint64_t row =
                    (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                f->addrs[(size_t)t * 16 + h] =
                    (uint64_t)(uintptr_t)(ple_table->data + row * g_ple_rb);
              }
            f->barrier.fetch_add(1, std::memory_order_acq_rel);
            if (w >= npump) return;
            while (f->barrier.load(std::memory_order_acquire) < nt)
              std::this_thread::yield();
            const size_t nrow = (size_t)f->P * 16;
            const size_t r0 = nrow * w / npump, r1 = nrow * (w + 1) / npump;
            // Diagnostic: emulate a slow disk (QWENOX_PLE_DELAY_MS) to verify
            // that the two-chunk lookahead hides multi-second gather stalls.
            static const int ple_delay_ms = [] {
              const char* e = getenv("QWENOX_PLE_DELAY_MS");
              return e ? std::max(0, atoi(e)) : 0;
            }();
            if (ple_delay_ms > 0)
              std::this_thread::sleep_for(std::chrono::milliseconds(ple_delay_ms));
            PLE_PUMP_I(w).run(f->addrs.data() + r0, f->fp8 + r0 * g_ple_rb,
                              (uint64_t)(uintptr_t)ple_table->data, r1 - r0);
            if (f->workers.fetch_sub(1) == 1)
              f->done = std::chrono::steady_clock::now();
          });
      } else {
        // Two-phase gather: phase A computes all 16*P row addresses, sorts
        // each shard by page and issues madvise in ascending order; phase B
        // copies rows in token order with pages already in flight. Output
        // staging bytes are identical to the per-token pipeline.
        fl.addrs.resize((size_t)P * 16);
        fl.keys.resize((size_t)P * 16);
        for (int w = 0; w < nt; w++)
          fl.pool.emplace_back([this, f, w, nt] {
            const int t0 = (int)((size_t)w * f->P / nt),
                      t1 = (int)((size_t)(w + 1) * f->P / nt);
            for (int t = t0; t < t1; t++)
              for (int h = 0; h < 16; h++) {
                int64_t mix = ple_cat(*f, t, 0) * ple_mult[0] ^
                              (ple_cat(*f, t, 1) * ple_mult[1]);
                if (h >= 8) mix ^= ple_cat(*f, t, 2) * ple_mult[2];
                uint64_t row =
                    (uint64_t)(ple_voff[h] + (int64_t)(mix % ple_vsz[h]));
                f->addrs[(size_t)t * 16 + h] =
                    (uint64_t)(uintptr_t)(ple_table->data + row * g_ple_rb);
              }
            uint64_t* kb = f->keys.data() + (size_t)t0 * 16;
            std::memcpy(kb, f->addrs.data() + (size_t)t0 * 16,
                        (size_t)(t1 - t0) * 16 * 8);
            std::sort(kb, kb + (size_t)(t1 - t0) * 16);
            uint64_t prev = ~0ull;
            for (int64_t i = 0; i < (int64_t)(t1 - t0) * 16; i++) {
              uint64_t pg = kb[i] & ~4095ull;
              if (pg != prev) {
                madvise((void*)(uintptr_t)pg, 4096, MADV_WILLNEED);
                prev = pg;
              }
            }
            f->barrier.fetch_add(1, std::memory_order_acq_rel);
            while (f->barrier.load(std::memory_order_acquire) < nt)
              std::this_thread::yield();
            for (int t = t0; t < t1; t++) {
              uint8_t* dst = f->fp8 + (size_t)t * 16 * g_ple_rb;
              const uint64_t* ap = f->addrs.data() + (size_t)t * 16;
              for (int h = 0; h < 16; h++)
                std::memcpy(dst + h * g_ple_rb, (const void*)(uintptr_t)ap[h],
                            g_ple_rb);
            }
            if (f->workers.fetch_sub(1) == 1)
              f->done = std::chrono::steady_clock::now();
          });
      }
    } else {
      for (int w = 0; w < nt; w++)
        fl.pool.emplace_back([this, f, w, nt] {
          for (int t = w; t < f->P; t += nt)
            ple_rows_fp8_win(*f, t, f->fp8 + (size_t)t * 16 * g_ple_rb);
          if (f->workers.fetch_sub(1) == 1)
            f->done = std::chrono::steady_clock::now();
        });
    }
    fl.valid = true;
  }
  // The in-flight gather is exactly this chunk's rows: same owner slot, window,
  // tokens and the two history tokens the cat() window reads before it.
  bool ple_flight_hit(const PleFlight& fl, const std::vector<int>& tokens, int base) const {
    if (!fl.valid || fl.nseg != 0 || fl.slot != bound_ || fl.base != base ||
        fl.P != (int)tokens.size() || fl.toks != tokens)
      return false;
    int64_t pre0, pre1;
    ple_pre01(chist, pre0, pre1);
    return fl.pre0 == pre0 && fl.pre1 == pre1;
  }
  void ple_join(PleFlight& fl) {
#ifdef _WIN32
    if (fl.inline_io) {
      PLE_PUMP.wait_all();
      fl.done = std::chrono::steady_clock::now();
      fl.inline_io = false;
    }
#endif
    for (auto& th : fl.pool) th.join();
    fl.pool.clear();
  }

  // QWENOX_VGDUMP (debug): sync + D2H an arbitrary device buffer, FNV-1a hash.
  // QWENOX_VGDUMP_ROWS=1: one hash per row (bytes/P each) so runs with
  // different P can be compared row by row.
  void vg_dump_buf(const char* tag, int P, const void* p, size_t bytes) {
    if (vg_capturing_ || P < 2 || P > 16 || !getenv("QWENOX_VGDUMP")) return;
    CK(hipStreamSynchronize(g_str));
    std::vector<uint8_t> h(bytes);
    CK(hipMemcpy(h.data(), p, bytes, hipMemcpyDeviceToHost));
    auto fnv1a = [](const uint8_t* d, size_t n) {
      uint64_t fnv = 1469598103934665603ULL;
      for (size_t i = 0; i < n; i++) { fnv ^= d[i]; fnv *= 1099511628211ULL; }
      return fnv;
    };
    if (getenv("QWENOX_VGDUMP_ROWS") && bytes % (size_t)P == 0) {
      const size_t rb = bytes / (size_t)P;
      fprintf(stderr, "[%s] P=%d base=%d rows", tag, P, moe_dump_base_);
      for (int r = 0; r < P; r++)
        fprintf(stderr, " %016llx", (unsigned long long)fnv1a(h.data() + (size_t)r * rb, rb));
      fprintf(stderr, "\n");
      return;
    }
    fprintf(stderr, "[%s] P=%d base=%d bufhash=%016llx\n", tag, P,
            moe_dump_base_, (unsigned long long)fnv1a(h.data(), bytes));
  }

  // QWENOX_VGDUMP (debug): sync + D2H d_Rb rows [0,P), print an FNV-1a hash of
  // the raw row bytes (bf16 or f32 depending on gr_bf16) — sensitive
  // round-state comparator for verify-graph bisection.
  void vg_dump_row(const char* tag, int P, int base) {
    if (vg_capturing_ || P > 16 || !getenv("QWENOX_VGDUMP")) return;
    CK(hipStreamSynchronize(g_str));
    const size_t row = (size_t)10240 * (gr_bf16 ? 2 : 4);
    std::vector<uint8_t> h((size_t)P * row);
    CK(hipMemcpy(h.data(), (const uint8_t*)d_Rb, (size_t)P * row,
                 hipMemcpyDeviceToHost));
    auto fnv1a = [](const uint8_t* d, size_t n) {
      uint64_t fnv = 1469598103934665603ULL;
      for (size_t i = 0; i < n; i++) { fnv ^= d[i]; fnv *= 1099511628211ULL; }
      return fnv;
    };
    if (getenv("QWENOX_VGDUMP_ROWS")) {
      fprintf(stderr, "[%s] P=%d base=%d rows", tag, P, base);
      for (int r = 0; r < P; r++)
        fprintf(stderr, " %016llx",
                (unsigned long long)fnv1a(h.data() + (size_t)r * row, row));
      fprintf(stderr, "\n");
      return;
    }
    fprintf(stderr, "[%s] P=%d base=%d rowhash=%016llx\n", tag, P, base,
            (unsigned long long)fnv1a(h.data(), h.size()));
  }

  // embed all tokens: gather rows then repeat to 4 branches (in-place safe).
  // src is the H2D source: pageable in direct prefill, the pinned staging
  // p_vgtok under verify graph capture (P3-B).
  void enq_embed_body(const int* src, int P) {
    CK(hipMemcpyAsync(d_tokarr, src, P * 4, hipMemcpyHostToDevice, g_str));
    WView et = wview(ckpt.at("embed_tokens.weight"));
    if (gr_bf16)
      embed_rows(et, d_tokarr, P, d_Rb, 0, true);
    else {
      embed_rows(et, d_tokarr, P, d_Rb, 10240);
      k_repeat4_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_Rb, d_Rb, g_cfg.d);
    }
    if (!vg_capturing_ && getenv("QWENOX_VGDUMP")) {
      CK(hipStreamSynchronize(g_str));
      int tk[16] = {};
      int nd = P < 16 ? P : 16;
      CK(hipMemcpy(tk, d_tokarr, (size_t)nd * 4, hipMemcpyDeviceToHost));
      fprintf(stderr, "[vgd] P=%d tok=%d,%d,%d,%d\n", P, tk[0],
              P > 1 ? tk[1] : -1, P > 2 ? tk[2] : -1, P > 3 ? tk[3] : -1);
      vg_dump_row("vge", P, -1);
    }
  }

  // PLE device phase on a gathered staging buffer: dequant -> conv/ring ->
  // add. Under verify graph capture (P3-B) the staging-free event is skipped:
  // the round's closing blocking D2H fences the staging reuse instead.
  void enq_ple_device_b(int P, int base, int buf) {
    uint8_t* stg_dev = ple_stgdev(buf);
    // dequant staging -> f32 d_eb on GPU (zero-copy read of pinned staging)
    if (ple_iq4) {
      size_t nb = (size_t)P * 2560 / 32;
      k_ple_iq4nl_dequant<<<(unsigned)((nb + 255) / 256), 256, 0, g_str>>>(
          stg_dev, d_eb, nb);
    } else if (ple_bf16) {
      size_t n16 = (size_t)P * 2560 / 16;
      k_ple_bf16_dequant<<<(unsigned)((n16 + 255) / 256), 256, 0, g_str>>>(
          stg_dev, d_eb, n16);
    } else {
      size_t n16 = (size_t)P * 2560 / 16;
      k_ple_fp8_dequant<<<(unsigned)((n16 + 255) / 256), 256, 0, g_str>>>(
          stg_dev, d_eb, ple_tscale, n16);
    }
    // staging is free once the dequant ran; gates the next gather into buf.
    if (!vg_capturing_) {
      CK(hipEventRecord(ple_stg_ev[buf], g_str));
      ple_stg_ev_ok[buf] = true;
    }
    float* hc_storage = d_Rb;
    if (gr_bf16) {
      uint64_t n = (uint64_t)P * 10240;
      k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
          (const uint16_t*)hc_storage, d_Rhatb, n);
      d_Rb = d_Rhatb;
    }
    if (ple_loop)
      for (int t = 0; t < P; t++) ple_gpu_t(t, base);
    else
      ple_gpu_b(P, base);
    if (gr_bf16) {
      uint64_t n = (uint64_t)P * 10240;
      k_f32_to_bf16_v4<<<(unsigned)((n / 4 + 255) / 256), 256, 0, g_str>>>(
          d_Rb, (uint16_t*)hc_storage, 10240, P, 10240);
      d_Rb = hc_storage;
    }
  }

  int prefill_batch(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false) {
    int N = (int)tokens.size();
    int id = -1;
    for (int off = 0; off < N;) {
      int rem = N - off;
      // Absorb a small tail into this chunk instead of running a tiny final
      // chunk whose per-chunk fixed cost (~0.9 s) dwarfs its token count.
      int n = ple_prefetch::chunk_size(rem, maxbatch, maxbatch_cap);
      // D1b: while the GPU is lent mid-chunk, only light rounds may start;
      // a heavy chunk requeues behind the owner and retries. 10-13: it also
      // requeues while a heavy waiter has less prompt left (heavy_gate_fn).
      if (n > conc_rmax_ && borrow_skip_fn) {
        const int rem = pf_end > 0 ? std::max(pf_end - (base + off), n) : N - off;
        while (g_borrow_light.load(std::memory_order_acquire) ||
               (heavy_gate_fn && heavy_gate_fn(*this, rem))) {
          heavy_rem_ = rem;
          borrow_skip_fn(*this);
        }
      }
      // Next chunks (same sizing rule) feed the two-flight PLE prefetch:
      // their gathers run on host threads under this chunk's layer compute.
      std::vector<int> nxt, nxt2;
      if (off + n < N) {
        int rem2 = N - (off + n);
        int n2 = ple_prefetch::chunk_size(rem2, maxbatch, maxbatch_cap);
        nxt.assign(tokens.begin() + off + n, tokens.begin() + off + n + n2);
        if (off + n + n2 < N) {
          int rem3 = N - (off + n + n2);
          int n3 = ple_prefetch::chunk_size(rem3, maxbatch, maxbatch_cap);
          nxt2.assign(tokens.begin() + off + n + n2,
                      tokens.begin() + off + n + n2 + n3);
        }
      }
      id = prefill_chunk(
          std::vector<int>(tokens.begin() + off, tokens.begin() + off + n),
          base + off, final, quiet, nxt.empty() ? nullptr : &nxt,
          nxt2.empty() ? nullptr : &nxt2);
      // DFlash2: build the draft context K/V for this chunk's (last <= 2048)
      // positions before the next chunk overwrites the tap ring.
      if (df2.on && df2.cap_mode == 1) df2_inject_chunk(base + off, n);
      if (mtp_tap_capture) {
        // Ingest this chunk's taps (chunk-local rows in d_mtap) into the MTP
        // layer before the next chunk overwrites them. MTP position i takes
        // token i+1, so the very last prompt row is left to the spec loop.
        int c0 = base + off;
        int count = std::min(n, base + N - 1 - c0);
        if (count > 0) {
          std::vector<int> shifted(tokens.begin() + off + 1,
                                   tokens.begin() + off + 1 + count);
          std::chrono::steady_clock::time_point mt0;
          if (prof) {
            CK(hipStreamSynchronize(g_str));
            mt0 = std::chrono::steady_clock::now();
          }
          mtp_ingest_b(d_mtap, shifted, count, c0);
          if (prof) {
            CK(hipStreamSynchronize(g_str));
            fprintf(stderr, "prof-mtp: P=%d ingest=%.1fms\n", count,
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - mt0).count());
          }
        }
      }
      off += n;
      // 客户端已断连/取消：不再为已断开的请求预填剩余分段（长 prompt 可能还有
      // 几万 token）。段间取消延迟 ≤ 一个分段；抛点状态语义同 yield_point。
      if (abort_flag && abort_flag->load(std::memory_order_relaxed))
        throw ReqCancelled("prefill aborted: client gone");
      // Concurrent serve: prompt prefill hands the GPU over between chunks
      // (only persistent slot state is live here; spec verify never yields).
      if (prog_total > 0 && off < N) yield_point();
    }
    return id;
  }

  int prefill_chunk(const std::vector<int>& tokens, int base = 0, bool final = true,
                    bool quiet = false, const std::vector<int>* nxt = nullptr,
                    const std::vector<int>* nxt2 = nullptr) {
    int P = (int)tokens.size();
    if (P < 1 || base + P > maxctx || base < 0 || P > maxbatch_cap) {
      fprintf(stderr, "prefill_chunk: base=%d P=%d out of range "
              "(maxctx=%d maxbatch_cap=%d)\n", base, P, maxctx, maxbatch_cap);
      exit(1);
    }
    moe_dump_base_ = base;
    {  // QWENOX_LT_FORGET=1 (diagnostic): every chunk re-pays the hipBLASLt
       // per-P algo miss, as a serving request with a new prompt length does.
      static const bool lt_forget = getenv("QWENOX_LT_FORGET") != nullptr;
      if (lt_forget) {
        lt_algos.clear();
        lt_buckets.clear();
      }
    }
    kv_prepare(base, base + P);  // A2/A3: back (and unshare) rows [base, base+P)
    set_vbase(base);  // P3-A: all base-reading batch kernels this chunk read d_vbase
    cut_begin_chunk(base, P);
    static const bool phase_prof = getenv("QWENOX_PHASE") != nullptr;
    auto phase = [&](const char* tag, std::chrono::steady_clock::time_point& tp) {
      if (!phase_prof) return;
      CK(hipStreamSynchronize(g_str));
      auto now = std::chrono::steady_clock::now();
      fprintf(stderr, "phase %-10s %8.1f ms\n", tag,
              std::chrono::duration<double, std::milli>(now - tp).count());
      tp = now;
    };
    xf16_src = nullptr;  // a new batch invalidates the bf16 conversion cache
#ifdef QWENOX_PHASE_PROF
    {
      static const unsigned long long zeros[16] = {};
      CK(hipMemcpyToSymbol(g_ph, zeros, sizeof(zeros)));
    }
#endif
    auto T0 = std::chrono::steady_clock::now();
    auto TP = T0;
    // PLE rows depend only on token history. This chunk's gather may already
    // be inflight — prefetched one or two chunks ahead so the disk pump
    // overlaps layer compute; otherwise gather now (first chunk, quiet
    // re-ingest, or a flight mismatch). Either way the gather fills a
    // rotating staging buffer and its GPU consumer stays ordered after L0.
    const bool ple_lookahead = ple_on && P >= 64 && !quiet;
    if (ple_on) {
      if (!ple_lookahead) ple_drop_one(ple_next());  // small/quiet: one flight only
      if (!ple_flight_hit(ple_cur(), tokens, base)) {
        if (ple_cur().valid) {  // stale flight (another chunk or slot): finish and drop
          static const bool miss_log = getenv("QWENOX_PLE_MISS_LOG") != nullptr;
          if (miss_log)
            fprintf(stderr, "%s ple: drop flight slot %d base %d P %d (chunk slot %d base %d P %d)\n",
                    log_ts(), ple_cur().slot, ple_cur().base, ple_cur().P, bound_, base, P);
        }
        ple_drop_all();
        int64_t pre0, pre1;
        ple_pre01(chist, pre0, pre1);
        const int b = ple_take_buf();
        ple_spawn(ple_cur(), std::vector<int>(tokens), base, pre0, pre1,
                  ple_stgbuf(b), b);
      }
      for (int t = 0; t < P; t++) chist.push_back(tokens[t]);
    }
    KP("embed_ple");
    enq_embed_body(tokens.data(), P);
    if (dbg && !gr_bf16)
      h_rms("gpu", -1, "emb", d_Rb + (size_t)(P - 1) * 10240, g_cfg.d);
    // Vision (stage 4): masked_scatter the ViT embeds over this chunk's image
    // placeholder rows (HF inputs_embeds semantics; bf16-quantized source).
    if (mrope_active_ && !img_inj.empty()) {
      h_img_inj.clear();
      for (const int2& e : img_inj)
        if (e.y >= base && e.y < base + P)
          h_img_inj.push_back(make_int2(e.x, e.y - base));
      if (!h_img_inj.empty()) {
        CK(hipMemcpyAsync(d_img_inj, h_img_inj.data(),
                          h_img_inj.size() * sizeof(int2),
                          hipMemcpyHostToDevice, g_str));
        if (gr_bf16)
          k_img_inject_bf16<<<(unsigned)h_img_inj.size(), 256, 0, g_str>>>(
              d_img_emb, d_img_inj, (uint16_t*)d_Rb);
        else
          k_img_inject_f32<<<(unsigned)h_img_inj.size(), 256, 0, g_str>>>(
              d_img_emb, d_img_inj, d_Rb);
      }
    }
    DpfScope dpf(*this, P);
    dpf.layer(0);
    enq_layer_b(0, P, false, true, base);
    dpf.pause();
    vg_dump_row("vg0", P, base);
    if (prog_total > 0) prog_probe(0, P);
    phase("embed+L0", TP);
    if (ple_on) {
      KP("ple_wait");  // 否则 join 的磁盘等待会落进上一个分段（hc3）
      auto wait_start = std::chrono::steady_clock::now();
      ple_join(ple_cur());
      auto wait_done = std::chrono::steady_clock::now();
      if (prof) {
        p_pleh += std::chrono::duration<double, std::milli>(ple_cur().done - ple_cur().start)
                      .count();
        p_ple_wait +=
            std::chrono::duration<double, std::milli>(wait_done - wait_start).count();
      }
      const int buf = ple_cur().buf;
      ple_cur().valid = false;
      KP("ple_dev");
      enq_ple_device_b(P, base, buf);
      vg_dump_row("vgp", P, base);
    }
    phase("ple", TP);
    if (ws_ok_) CK(hipEventRecord(conc_ev_a_, g_str));  // D1b quantum clock
    for (int l = 1; l < g_cfg.layers; l++) {
      bool fuse_prev = gr_scatter_norm && l > 1;
      bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
      dpf.layer(l);
      enq_layer_b(l, P, fuse_prev, flush_mlp, base);
      dpf.pause();
      {
        char tag[8];
        snprintf(tag, sizeof tag, "pcL%d", l);
        vg_dump_row(tag, P, base);
      }
      if (prog_total > 0) prog_probe(l, P);
      conc_borrow_yield(base, P, l);  // D1b: no-op unless armed and contended
      if (l == 24) phase("L01-24", TP);
      if (phase_prof && getenv("QWENOX_PHASE")[0] == '2') {
        char lbuf[16];
        snprintf(lbuf, sizeof lbuf, "L%02d", l);
        phase(lbuf, TP);
      }
    }
    phase("L25-47", TP);
    // PLE chunk-ahead prefetch: keep two flights in the air (N+1, N+2) while
    // the GPU drains this chunk's layer queue below (row addresses are pure
    // functions of token ids, all known now; staging rotates over 3 buffers).
    // QWENOX_PLE_NOPREFETCH disables for A/B.
    static const bool ple_noprefetch = getenv("QWENOX_PLE_NOPREFETCH") != nullptr;
    if (ple_on && !ple_noprefetch) {
      int64_t pre0, pre1;
      ple_pre01(chist, pre0, pre1);
      if (ple_lookahead) {
        // N+1: normally already in flight (spawned two chunks ago). Re-validate
        // it against the actual next chunk before trusting it (slot switch,
        // tail absorption or a cancelled request can leave a stale flight).
        if (nxt && !nxt->empty()) {
          if (ple_next().valid && !ple_flight_hit(ple_next(), *nxt, base + P))
            ple_drop_one(ple_next());
          if (!ple_next().valid) {
            const int b = ple_take_buf();
            ple_spawn(ple_next(), std::vector<int>(*nxt), base + P, pre0, pre1,
                      ple_stgbuf(b), b);
          }
        }
        // N+2 into the flight object just consumed.
        if (nxt2 && !nxt2->empty()) {
          if (ple_cur().valid) ple_drop_one(ple_cur());
          int64_t p0 = pre0, p1 = pre1;
          if (nxt && !nxt->empty()) {
            p0 = nxt->back();
            p1 = nxt->size() >= 2 ? (*nxt)[nxt->size() - 2] : pre0;
          }
          const int nb = base + P + (nxt ? (int)nxt->size() : 0);
          const int b = ple_take_buf();
          ple_spawn(ple_cur(), std::vector<int>(*nxt2), nb, p0, p1,
                    ple_stgbuf(b), b);
        }
        ple_cur_i ^= 1;  // next chunk consumes the object holding N+1
      } else if (nxt && !nxt->empty() && !ple_cur().valid) {
        const int b = ple_take_buf();
        ple_spawn(ple_cur(), std::vector<int>(*nxt), base + P, pre0, pre1,
                  ple_stgbuf(b), b);
      }
    }
    KP("tail");
    // MTP tap capture: d_Rb now holds the final 48-layer stream for every row.
    if (mtp_tap_capture) {
      // Taps are stored at chunk-local rows (0..P) regardless of the absolute
      // base; prefill_batch ingests each chunk right after it completes, so
      // prompts longer than maxbatch work.
      if (P > maxbatch_cap) {
        fprintf(stderr, "mtp tap capture: unsupported (P=%d maxbatch_cap=%d)\n", P,
                maxbatch_cap);
        exit(1);
      }
      if (gr_bf16) {  // d_Rb rows are bf16; d_mtap stays fp32 for ingest
        uint64_t n = (uint64_t)P * 10240;
        k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
            (const uint16_t*)d_Rb, d_mtap, n);
      } else {
        CK(hipMemcpyAsync(d_mtap, d_Rb, (size_t)P * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      }
    }
    if (cut_r_) {  // in-pass cut: d_R(cut) = final stream row cut-1
      if (gr_bf16)
        k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
            (const uint16_t*)d_Rb + (size_t)(cut_r_ - 1) * 10240, d_cR, 10240);
      else
        CK(hipMemcpyAsync(d_cR, d_Rb + (size_t)(cut_r_ - 1) * 10240, 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      cut_hit = true;
      cut_r_ = 0;
    }
    // hand off last row to the single-token decode state
    if (gr_bf16)
      k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
          (const uint16_t*)d_Rb + (size_t)(P - 1) * 10240, d_R, 10240);
    else
      CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(P - 1) * 10240, 10240 * 4,
                        hipMemcpyDeviceToDevice, g_str));
    pos = base + P;
    ple_ringpos = (base + P) % 9;
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    // Verify consumes batch logits; do not compute and synchronously read
    // the last row's lm_head a second time.
    if (final) enq_final();
    phase("final", TP);
    int id = -1;
    if (final && prog_total > 0) prog_wait(P);
    if (final) CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    phase("argmax_d2h", TP);
    KP("end");
    if (g_kprof_on) g_kprof_ring.flush(base, P, quiet);
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0).count();
    // quiet: spec-verify re-ingest chunks (<= gamma+1 tokens each round).
    // Serve-mode prompt chunks use the live progress and request summary logs;
    // keep this legacy line only for offline prefill benchmarks.
    if (!quiet && prog_total <= 0)
      fprintf(stderr, "\n%s prefill: %d tokens in %.2f s = %.1f tok/s\n", log_ts(),
              P, sec, P / sec);
    if (prog_total > 0) prog_base += P;  // live progress: chunk complete
    if (prof)
      fprintf(stderr,
              "prof: gemm_host=%.0fms topk_wait=%.0fms bucket+h2d=%.0fms "
              "ple_host=%.0fms ple_wait=%.0fms\n",
              p_gemm, p_topk_wait, p_bucket, p_pleh, p_ple_wait);
    if (prof && P >= 64) {  // per-chunk view: zero the accumulators (aggr 10-05)
      fprintf(stderr, "prof-chunk: P=%d wall=%.1fms\n", P, sec * 1e3);
      p_gemm = p_topk_wait = p_bucket = p_pleh = p_ple_wait = 0;
    }
#ifdef QWENOX_PHASE_PROF
    if (getenv("QWENOX_PHASE_PROF")) {
      CK(hipStreamSynchronize(g_str));
      unsigned long long h[16];
      CK(hipMemcpyFromSymbol(h, g_ph, sizeof(h)));
      hipDeviceProp_t dp;
      CK(hipGetDeviceProperties(&dp, 0));
      double cyc2ms = 1.0 / (dp.clockRate * 1e3);  // clockRate is kHz
      const char* names[11] = {"in_staging", "in_dots",   "in_solve", "in_vpkcd",
                               "in_attn2",   "st_Sstage", "st_vnew",  "st_out",
                               "st_Supd",    "in_TOTAL",  "st_TOTAL"};
      for (int s = 0; s < 11; ++s)
        fprintf(stderr, "ph %-10s %12llu cyc  %9.1f CU-ms\n", names[s], h[s],
                h[s] * cyc2ms);
    }
#endif
    return id;
  }

  // ---- D2 verify_segments --------------------------------------------------
  // Multi-segment spec verify. One segment = one slot's verify chunk: toks[0]
  // is the last committed token, the rest are the drafts; base = absolute
  // position of toks[0]. nseg==1 on the bound slot keeps the S1a path: exactly
  // prefill_batch(toks, base, final=false, quiet=true) + batch_argmax(P, vout)
  // (a verify chunk never exceeds maxbatch_cap, so prefill_batch degenerates
  // to one prefill_chunk with no tap capture, no PLE next-chunk prefetch and
  // no inter-chunk yield). nseg>1 (or a non-bound slot) goes through
  // prefill_chunk_vseg: one batched pass over the R = sum(P_s) concatenated
  // rows, bit-identical to running each segment through the serial path.
  // Returns false -> caller falls back to the per-segment serial path
  // (QWENOX_NO_VSEG=1 forces the fallback at the call site).
  struct VSeg {
    int slot;  // owner of the state this segment advances
    const std::vector<int>* toks;
    int base;
  };

  // D2-S3c: one worker's draft-round submission to the rendezvous pool:
  // slot drafts g tokens from (slot d_R, tok0); drafts receives g ints.
  struct DrSeg {
    int slot;
    int g;
    int tok0;
    int* drafts;  // worker stack
    // C7 (sampled drafts): called by the executor after step j with this
    // member's raw logits row (device); sets tok to the drawn draft, returns
    // false to stop after this step. nullptr = greedy (device argmax).
    const std::function<bool(int, const float*, int&)>* step = nullptr;
  };

  // D2-S1c: the multi-segment verify chunk. Guards (any failure -> false, the
  // caller falls back to serial): nseg in [1,8], P_s in [1,8], R <= kVSnapRows
  // rows,
  // distinct live slots, no mrope / in-pass cut / progress / MTP tap on any
  // segment's slot, the loop debug paths off, no HC inject state in flight.
  bool prefill_chunk_vseg(const VSeg* segs, int nseg) {
    // QWENOX_VSEG_DBG=1: say which guard refused (fallback stays silent by
    // default — a refusal is a legitimate route to the serial path).
    static const bool vdbg = getenv("QWENOX_VSEG_DBG") != nullptr;
    auto rej = [&](const char* why) {
      if (vdbg) fprintf(stderr, "vseg refuse: %s\n", why);
      return false;
    };
    if (!segs || nseg < 1 || nseg > 8) return rej("nseg");
    if (gdn_loop || qsa_loop || ple_loop || qsa_dense) return rej("loop/dense");
    if (vg_capturing_ || injp_cur >= 0 || !hcw_site_.empty())
      return rej("capture/hc-inject");
    int Ps[8], bases[8], sl[8], voffs[8], R = 0;
    for (int s = 0; s < nseg; s++) {
      if (!segs[s].toks) return rej("toks");
      const int P = (int)segs[s].toks->size(), b = segs[s].base, t = segs[s].slot;
      if (P < 1 || P > 8 || b < 0 || b + P > maxctx) return rej("P/base range");
      if (t < 0 || t >= nslots) return rej("slot id");
      for (int q = 0; q < s; q++) if (sl[q] == t) return rej("dup slot");
      if (slot_mrope_active_(t) || slot_cut_abs(t) >= 0 || slot_cut_r_(t) ||
          slot_prog_total(t) > 0 || slot_mtp_tap_capture(t) ||
          slot_ngram_verify_active(t))
        return rej("slot flags");
      if (slot_spec_snap(t) &&
          (slot_vsnap_row0_(t) < 0 || slot_vsnap_row0_(t) + P > kVSnapRows))
        return rej("vsnap window");  // snap rows [row0, row0+P) must fit
      Ps[s] = P; bases[s] = b; sl[s] = t; voffs[s] = R; R += P;
    }
    if (R > kVSnapRows || R > maxbatch_cap) return rej("R cap");
    moe_dump_base_ = bases[0];
    xf16_src = nullptr;  // a new batch invalidates the bf16 conversion cache
    for (int s = 0; s < nseg; s++)  // back (and unshare) each segment's rows
      kv_prepare_seq(slot_kv_seq(sl[s]), slot_d_ptab(sl[s]), bases[s],
                     bases[s] + Ps[s]);
    // PLE: ONE disk-gather flight for all R rows (segment s = staging rows
    // [voffs[s], voffs[s]+P_s), each token's n-gram window read from its own
    // slot's history — the same staging bytes as per-segment flights). Like
    // the serial chunk it is spawned before embed and joined after L0 is
    // enqueued, so the host IO overlaps embed+L0 instead of idling the GPU.
    // The dequant is elementwise per row: one launch over R rows writes the
    // same d_eb bytes as per-segment launches. The conv/gate device phase
    // runs after L0, in the serial chunk's order.
    g_vseg_rows += R;
    g_vseg_calls++;
    int ple_buf = 0;
    if (ple_on) {
      ple_drop_all();  // verify uses one flight; no two-chunk lookahead here
      PleSegs ps;
      ps.n = nseg;
      std::vector<int> cat;
      cat.reserve((size_t)R);
      for (int s = 0; s < nseg; s++) {
        ps.start[s] = voffs[s];
        ple_pre01(slot_chist(sl[s]), ps.pre0[s], ps.pre1[s]);
        cat.insert(cat.end(), segs[s].toks->begin(), segs[s].toks->end());
        for (int t = 0; t < Ps[s]; t++)
          slot_chist(sl[s]).push_back((*segs[s].toks)[(size_t)t]);
      }
      ps.start[nseg] = R;
      ple_buf = ple_take_buf();
      ple_spawn(ple_cur(), std::move(cat), bases[0], 0, 0,
                ple_stgbuf(ple_buf), ple_buf, &ps);
    }
    // embed: one gather pass over the concatenated rows (row-independent)
    int all[8 * 8];
    for (int s = 0; s < nseg; s++)
      memcpy(all + voffs[s], segs[s].toks->data(), (size_t)Ps[s] * 4);
    enq_embed_body(all, R);
    enq_layer_vseg(0, Ps, bases, sl, voffs, nseg, R, false, true);
    if (ple_on) {
      const auto ple_t0 = std::chrono::steady_clock::now();
      ple_join(ple_cur());
      g_vseg_ple_ms += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - ple_t0).count();
      g_vseg_pleh_ms +=
          std::chrono::duration<double, std::milli>(ple_cur().done - ple_cur().start).count();
      ple_cur().valid = false;
      ple_cur().nseg = 0;  // later single-window flights must not see segments
      uint8_t* stg_dev = ple_stgdev(ple_buf);
      if (ple_iq4) {
        size_t nb = (size_t)R * 2560 / 32;
        k_ple_iq4nl_dequant<<<(unsigned)((nb + 255) / 256), 256, 0, g_str>>>(
            stg_dev, d_eb, nb);
      } else if (ple_bf16) {
        size_t n16 = (size_t)R * 2560 / 16;
        k_ple_bf16_dequant<<<(unsigned)((n16 + 255) / 256), 256, 0, g_str>>>(
            stg_dev, d_eb, n16);
      } else {
        size_t n16 = (size_t)R * 2560 / 16;
        k_ple_fp8_dequant<<<(unsigned)((n16 + 255) / 256), 256, 0, g_str>>>(
            stg_dev, d_eb, ple_tscale, n16);
      }
      // staging is free once the dequant ran; gates the next gather into buf
      CK(hipEventRecord(ple_stg_ev[ple_buf], g_str));
      ple_stg_ev_ok[ple_buf] = true;
    }
    if (ple_on) {  // device phase, same shape as enq_ple_device_b over R rows
      float* hc_storage = d_Rb;
      if (gr_bf16) {
        uint64_t n = (uint64_t)R * 10240;
        k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
            (const uint16_t*)hc_storage, d_Rhatb, n);
        d_Rb = d_Rhatb;
      }
      for (int s = 0; s < nseg; s++)
        ple_gpu_b_vseg(Ps[s], bases[s], voffs[s], slot_d_ring(sl[s]));
      if (gr_bf16) {
        uint64_t n = (uint64_t)R * 10240;
        k_f32_to_bf16_v4<<<(unsigned)((n / 4 + 255) / 256), 256, 0, g_str>>>(
            d_Rb, (uint16_t*)hc_storage, 10240, R, 10240);
        d_Rb = hc_storage;
      }
    }
    for (int l = 1; l < g_cfg.layers; l++) {
      bool fuse_prev = gr_scatter_norm && l > 1;
      bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
      enq_layer_vseg(l, Ps, bases, sl, voffs, nseg, R, fuse_prev, flush_mlp);
    }
    // per-segment tail: hand the last row to that slot's decode state
    I32Set iset{};
    for (int s = 0; s < nseg; s++) {
      const int lastrow = voffs[s] + Ps[s] - 1;
      if (gr_bf16)
        k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
            (const uint16_t*)d_Rb + (size_t)lastrow * 10240, slot_d_R(sl[s]),
            10240);
      else
        CK(hipMemcpyAsync(slot_d_R(sl[s]), d_Rb + (size_t)lastrow * 10240,
                          10240 * 4, hipMemcpyDeviceToDevice, g_str));
      int& p = slot_pos(sl[s]);
      int& rp = slot_ple_ringpos(sl[s]);
      p = bases[s] + Ps[s];
      rp = (bases[s] + Ps[s]) % 9;
      // device mirrors: one stream-ordered kernel for all segments (no
      // pageable H2D round trips); same values the copies used to write
      iset.p[iset.n] = slot_d_pos(sl[s]);
      iset.v[iset.n++] = p;
      iset.p[iset.n] = slot_d_ringpos(sl[s]);
      iset.v[iset.n++] = rp;
    }
    k_set_i32<<<1, 32, 0, g_str>>>(iset);
    return true;
  }

  bool verify_segments(const VSeg* segs, int nseg, int* vout) {
    if (!segs || nseg < 1) return false;
    if (nseg == 1 && segs[0].toks && segs[0].slot == bound_) {
      // S1a: members are the bound slot's — the original path, unchanged
      const VSeg& sg = segs[0];
      const int P = (int)sg.toks->size();
      if (P < 1 || P > maxbatch_cap) return false;
      prefill_chunk(*sg.toks, sg.base, false, true, nullptr);
      batch_argmax(P, vout);
      return true;
    }
    // D2-S1c: multi-segment (or non-bound slot) batched pass. vout receives
    // all R rows concatenated in segment order; row voff_s+P_s-1 is segment
    // s's next token. Guards failed -> false -> caller runs serial per segment.
    // Pack each segment's rollback-snapshot window at its batch row offset so
    // the post-verify restore/ingest (vsnap_row0_-relative) finds its own rows.
    int voff = 0;
    for (int s = 0; s < nseg; s++) {
      slot_vsnap_row0_(segs[s].slot) = voff;
      voff += (int)segs[s].toks->size();
    }
    KP("vseg_begin");
    const auto te0 = std::chrono::steady_clock::now();
    if (!prefill_chunk_vseg(segs, nseg)) return false;
    int R = 0;
    for (int s = 0; s < nseg; s++) R += (int)segs[s].toks->size();
    KP("lm_head");
    batch_logits_device(R);
    // host time to enqueue the whole pass (layers + lm_head); compare with
    // the wall time: equal means the pass is launch-bound, not GPU-bound
    g_vseg_enq_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - te0).count();
    argmax_rows(d_logits_b, g_cfg.vocab, R, d_amx_cand, d_argmax_arr, g_str);
    CK(hipMemcpy(vout, d_argmax_arr, R * 4, hipMemcpyDeviceToHost));
    KP("end");
    if (g_kprof_on) g_kprof_ring.flush(segs[0].base, R, true);
    return true;
  }

  // ---- 10-13 并发分叉排查：会合 verify 之后的共享行窗口 ----
  // The batched verify leaves each segment's post-verify inputs in SHARED
  // scratch at its row window [row0, row0+P): trunk streams d_Rb (rollback
  // d_R, MTP ingest taps), GDN replay rows d_grow, conv snapshots d_csnap,
  // indexer proj rows d_iproj_snap. The executor then releases the FIFO
  // turn, so a non-member (prefill, ngram verify, a serial spec verify that
  // packs at row 0, ...) may run before the member re-acquires it and reads
  // its window (Windows v2 C4: 2-3 grants in between, outputs forked at
  // tokens 5-86). QWENOX_VR_RESTORE (default on, =0 off) saves the window
  // per slot in the executor and copies it back when the member resumes
  // (~50 MB per slot, ~0.3 ms per member round); QWENOX_VR_CHECK=1 (debug)
  // hashes the window on both sides and logs clobbers.
  static bool vr_restore_env() {
    static const bool v = !env_zero("QWENOX_VR_RESTORE");
    return v;
  }
  template <class F>
  void vr_rows_each(int row0, int P, F f) {
    const size_t ngdn = (size_t)(g_cfg.layers - g_cfg.layers / 4),
                 nqsa = (size_t)(g_cfg.layers / 4);
    const size_t esz = gr_bf16 ? 2 : 4;
    f(0, (char*)d_Rb + (size_t)row0 * 10240 * esz, (size_t)P * 10240 * esz);
    if (d_grow)
      f(1, (char*)(d_grow + (size_t)row0 * ngdn * kGdnRowF), (size_t)P * ngdn * kGdnRowF * 4);
    if (d_csnap)
      f(2, (char*)(d_csnap + (size_t)row0 * ngdn * 3 * 10240), (size_t)P * ngdn * 3 * 10240 * 4);
    if (d_iproj_snap)
      for (size_t qi = 0; qi < nqsa; qi++)
        f(3, (char*)(d_iproj_snap + (qi * kVSnapRows + row0) * 128), (size_t)P * 128 * 4);
  }
  // Per-buffer FNV of the window: [0] d_Rb [1] d_grow [2] d_csnap [3] d_iproj_snap.
  std::array<uint64_t, 4> vr_rows_hash(int row0, int P) {
    CK(hipStreamSynchronize(g_str));
    std::array<uint64_t, 4> h;
    h.fill(1469598103934665603ULL);
    std::vector<uint8_t> buf;
    vr_rows_each(row0, P, [&](int k, char* p, size_t n) {
      buf.resize(n);
      CK(hipMemcpy(buf.data(), p, n, hipMemcpyDeviceToHost));
      for (uint8_t b : buf) { h[k] ^= b; h[k] *= 1099511628211ULL; }
    });
    return h;
  }
  // Per slot, room for kVrSaveRows rows of every buffer (a spec segment is
  // γ+1 <= 9 rows). Allocated by slots_init (devarena_estimate counts it).
  static constexpr int kVrSaveRows = 9;
  std::vector<char*> vr_save_;
  size_t vr_save_bytes(int P) {
    size_t n = 0;
    vr_rows_each(0, P, [&](int, char*, size_t b) { n += b; });
    return n;
  }
  bool vr_rows_save(int slot, int row0, int P) {
    if (P > kVrSaveRows || slot < 0 || slot >= (int)vr_save_.size()) return false;
    char* s = vr_save_[(size_t)slot];
    if (!s) return false;
    size_t off = 0;
    vr_rows_each(row0, P, [&](int, char* p, size_t n) {
      CK(hipMemcpyAsync(s + off, p, n, hipMemcpyDeviceToDevice, g_str));
      off += n;
    });
    return true;
  }
  void vr_rows_load(int slot, int row0, int P) {
    char* s = slot < (int)vr_save_.size() ? vr_save_[(size_t)slot] : nullptr;
    if (!s) return;
    size_t off = 0;
    vr_rows_each(row0, P, [&](int, char* p, size_t n) {
      CK(hipMemcpyAsync(p, s + off, n, hipMemcpyDeviceToDevice, g_str));
      off += n;
    });
  }

  // Wiring: MTP stream position p takes (trunk h_p, emb(t_{p+1})) and predicts
  // t_{p+2}. Input: h = sum_b fc_hidden(norm_h(tap)[b]), e =
  // fc_embedding(norm_e(embed(tok))), stream = repeat4(h + e). Norm mode is
  // QWENOX_MTP_NORM=full (one 10240 RMSNorm, default) or =branch (4x2560).
  void mtp_input_1(const float* tap) {
    embed_rows(wview(ckpt.at("embed_tokens.weight")), d_mtok, 1, d_emb, 0);
    k_rmsnorm_zc_grouped<<<1, 1024, 0, g_str>>>(d_emb, d_mnorm_e, d_me, 2560,
                                                g_cfg.norm_eps, 1);
    gemv(wview(ckpt.at("mtp.fc_embedding.weight")), d_me, d_meo);
    if (mtp_norm_mode == 0)
      k_rmsnorm_zc_grouped<<<1, 1024, 0, g_str>>>(tap, d_mnorm_h, d_mRhat, 10240,
                                                  g_cfg.norm_eps, 1);
    else
      k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(tap, d_mnorm_h, d_mRhat, 2560,
                                                  g_cfg.norm_eps, 4);
    if (mtp_hagg_mode == 0) {
      // H-agg=sum: x = sum_b fc_hidden(norm_h[b]) + e, then repeat4.
      gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat, d_mh);
      for (int b = 1; b < 4; b++) {
        gemv(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat + (size_t)b * 2560,
             d_mt);
        k_acc<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_mt, 2560);
      }
      k_acc<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_meo, 2560);
      k_repeat4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_mh, d_mR, g_cfg.d);
    } else {
      // H-agg=branch: stream[b] = fc_hidden(norm_h[b]) + e (branch identity
      // kept). d_mRhat IS a contiguous [4][2560] batch of branch rows, so
      // one 4-row gemm streams fc_hidden once instead of once per branch.
      gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhat, d_mR, 4, 2560);
      k_acc_broadcast4<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_meo, d_mR, 2560);
    }
  }

  // GR read/write on the MTP stream (mirrors gr_read/gr_write with d_mR).
  void mtp_gr_read(const std::string& prefix, float* x_out) {
    char buf[128];
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4, 1024, 0, g_str>>>(d_mR, hw, d_mRhat, 2560,
                                                g_cfg.norm_eps, 4);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemv_silu(wview(ckpt.at(buf)), d_mRhat, d_t320);  // D3: silu*0.25 epilogue
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_t320, d_G);
    static const bool gr_sig = !env_zero("QWENOX_GR_COMBINE_SIG");
    if (gr_sig) {
      k_gr_combine<true><<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_mRhat, x_out,
                                                                g_cfg.d, g_cfg.branches);
    } else {
      k_sigmoid_inplace<<<(10240 + 255) / 256, 256, 0, g_str>>>(d_G, 10240);
      k_gr_combine<false><<<(2560 + 255) / 256, 256, 0, g_str>>>(d_G, d_mRhat, x_out,
                                                                 g_cfg.d, g_cfg.branches);
    }
  }
  void mtp_gr_write(const std::string& prefix, const float* y) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemv(wview(ckpt.at(buf)), d_mRhat, d_w4);
    k_gr_write<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_w4, d_mR, y, g_cfg.d,
                                                      g_cfg.branches);
  }

  template <typename KVT>
  void mtp_qsa_attn(const KVT* kc, const KVT* vc) {
    k_qsa_flash<true, false, KVT, true><<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
        d_qs, kc, vc, d_attn, 1, 256, 2, 24, qsa_dense ? nullptr : d_iselected, 0,
        d_mpos, d_pacc, d_pml, 0, kvpt());
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, d_attn, 12,
                                                       256, d_mpos);
  }

  // One QSA attention step for the MTP layer (mirrors qsa()).
  void mtp_qsa(const float* x, float* y) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    ensure_rope_tab(g_cfg.rope_theta, g_cfg.rotary_dim);
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, 1,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               d_mpos, 0, nullptr, d_rdelta);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_miraw,
                                           d_mik, d_mpos, g_cfg.norm_eps,
                                           g_cfg.rope_theta, d_rdelta, kvpt(),
                                           d_ikh ? d_ikh + (size_t)(g_cfg.layers / 4) *
                                                       ik_layer_blocks * 128
                                                 : nullptr);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(
          d_iq, d_mik, d_iscores, index_blocks, d_mpos, kvpt());
      index_select(d_iscores, index_blocks, d_iselected, 0, 1, d_mpos);
    }
    snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_qg);
    snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
    const WView mkw = wview(ckpt.at(buf));
    snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
    const WView mvw = wview(ckpt.at(buf));
    if (!gemv2(mkw, mvw, x, d_kb, d_vb)) {  // D4
      gemv(mkw, x, d_kb);
      gemv(mvw, x, d_vb);
    }
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(d_qg, qnw, d_qs, d_gs, 256,
                                                 g_cfg.norm_eps);
    k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(d_kb, knw, d_kb, 256,
                                               g_cfg.norm_eps, 1);
    k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(d_qs, 24, 256, 64, d_mpos,
                                                     g_cfg.rope_theta,
                                                     d_rdelta);
    k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(d_kb, 2, 256, 64, d_mpos,
                                                    g_cfg.rope_theta,
                                                    d_rdelta);
    if (qsa_kv_fp32) {
      k_store_kv<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcf, d_mvcf, d_mpos, kvpt());
      mtp_qsa_attn(d_mkcf, d_mvcf);
    } else {
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(d_kb, d_vb, d_mkcb, d_mvcb, d_mpos,
                                            nullptr, kvpt());
      mtp_qsa_attn(d_mkcb, d_mvcb);
    }
    k_sigmoid_gate<<<(24 * 256 + 255) / 256, 256, 0, g_str>>>(d_attn, d_gs,
                                                              d_attn, 24 * 256);
    snprintf(buf, sizeof buf, "%s.o_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), d_attn, y);
  }

  // D2-S3d: per-slot QSA mid-section for the batched draft executor. The
  // q/k/v projections run batched (gemv_multi rows, P=B) before the member
  // loop and o_proj batched after it; the q8g64 mr kernels are
  // row-invariant, so every row is bit-identical to the serial mtp_qsa gemv.
  // The indexer block must stay inside the per-slot section: d_iproj/d_iq/
  // d_iscores/d_iselected are single-row scratch consumed by the same slot's
  // flash before the next slot overwrites them. Identical launches to
  // mtp_qsa with d_mpos/d_miraw/d_rdelta/kvpt() swapped for the slot's own
  // state; d_mik/d_mkc{b,f}/d_mvc{b,f} are shared pool layer views, so they
  // stay.
  template <typename KVT>
  void mtp_qsa_attn_slot(int s, const float* q, float* out, const KVT* kc,
                         const KVT* vc) {
    auto* const mp = slot_d_mpos(s);
    const auto* const pt = kv_paged ? slot_d_ptab(s) : nullptr;
    k_qsa_flash<true, false, KVT, true>
        <<<dim3(QSA_DEC_NSPLIT, 2), 128, 0, g_str>>>(
            q, kc, vc, out, 1, 256, 2, 24,
            qsa_dense ? nullptr : d_iselected, 0, mp, d_pacc, d_pml, 0, pt);
    k_qsa_flash_combine<<<dim3(12, 2), 64, 0, g_str>>>(d_pml, d_pacc, out,
                                                       12, 256, mp);
  }
  void mtp_qsa_core_slot(int s, int r) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    auto* const mp = slot_d_mpos(s);
    auto* const ir = slot_d_miraw(s);
    const auto* const rd = slot_d_rdelta(s);
    const auto* const pt = kv_paged ? slot_d_ptab(s) : nullptr;
    const float* const x = d_xb + (size_t)r * 2560;
    float* const qg = d_qgb + (size_t)r * 12288;
    float* const kb = d_kbb + (size_t)r * 512;
    float* const vb = d_vbb + (size_t)r * 512;
    float* const qs = d_qsb + (size_t)r * 6144;
    float* const gs = d_gsb + (size_t)r * 6144;
    float* const at = d_attnb + (size_t)r * 6144;
    ensure_rope_tab(g_cfg.rope_theta, g_cfg.rotary_dim);
    if (!qsa_dense) {
      snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
      gemv(wview(ckpt.at(buf)), x, d_iproj);
      k_index_q<<<dim3(1, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, 1,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               mp, 0, nullptr, rd);
      k_index_append<<<1, 128, 0, g_str>>>(d_iproj, d_minorm + 128, ir,
                                           d_mik, mp, g_cfg.norm_eps,
                                           g_cfg.rope_theta, rd, pt,
                                           d_ikh ? d_ikh + (size_t)(g_cfg.layers / 4) *
                                                       ik_layer_blocks * 128
                                                 : nullptr);
      k_index_scores<<<(index_blocks + 3) / 4, 128, 0, g_str>>>(
          d_iq, d_mik, d_iscores, index_blocks, mp, pt);
      index_select(d_iscores, index_blocks, d_iselected, 0, 1, mp);
    }
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    k_qsa_qsplit<<<dim3(24, 1), 256, 0, g_str>>>(qg, qnw, qs, gs, 256,
                                                 g_cfg.norm_eps);
    k_rmsnorm_zc_grouped<<<2, 256, 0, g_str>>>(kb, knw, kb, 256,
                                               g_cfg.norm_eps, 1);
    k_rope<<<(24 * 32 + 127) / 128, 128, 0, g_str>>>(qs, 24, 256, 64, mp,
                                                     g_cfg.rope_theta, rd);
    k_rope<<<(2 * 32 + 127) / 128, 128, 0, g_str>>>(kb, 2, 256, 64, mp,
                                                    g_cfg.rope_theta, rd);
    if (qsa_kv_fp32) {
      k_store_kv<<<2, 256, 0, g_str>>>(kb, vb, d_mkcf, d_mvcf, mp, pt);
      mtp_qsa_attn_slot(s, qs, at, d_mkcf, d_mvcf);
    } else {
      k_store_kv_bf16<<<2, 256, 0, g_str>>>(kb, vb, d_mkcb, d_mvcb, mp,
                                            nullptr, pt);
      mtp_qsa_attn_slot(s, qs, at, d_mkcb, d_mvcb);
    }
    k_sigmoid_gate<<<(24 * 256 + 255) / 256, 256, 0, g_str>>>(at, gs, at,
                                                              24 * 256);
  }

  // MTP MoE step (mirrors moe()).
  void mtp_moe(const float* x, float* y) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    snprintf(buf, sizeof buf, "%s.gate.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_router);
    int k = g_cfg.topk;
    k_router_topk<<<1, 512, 0, g_str>>>(d_router, d_topids, d_topw, g_cfg.experts,
                                        k, VizTap{});
    if (gg_mtp_on()) {  // GGUF sidecar experts (Q8_0)
      moe_gguf(g_ggx_mtp, x, d_topids, d_topw, d_moeacc, 1);
    } else {
      snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
      WView gu = wview(ckpt.at(buf));
      snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
      WView dn = wview(ckpt.at(buf));
      if (!moe_down_direct(k, dn)) CK(hipMemsetAsync(d_moeacc, 0, g_cfg.d * 4, g_str));
      const int* topids = d_topids;  // 直读:ids 即专家号,免 D2H 同步
      uint64_t pairs = (uint64_t)k * g_cfg.moe_mid;
      k_q4cp_gemv_gg<0><<<(unsigned)((pairs + 15) / 16), 512, 0, g_str>>>(
          gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2, (const float*)gu.data,
          x, d_guv, topids, (uint64_t)2 * g_cfg.moe_mid, gu.cols, gu.scale_stride,
          k, 0 /*x_stride*/, 16 /*id_stride*/, k);
      k_silu_mul_g<<<(k * g_cfg.moe_mid + 255) / 256, 256, 0, g_str>>>(
          d_guv, d_hid, g_cfg.moe_mid, k);
      moe_down1(k, dn, topids);
    }
    {
      char ub[128];
      snprintf(buf, sizeof buf, "%s.shared_expert.gate_proj.weight", pfx);
      snprintf(ub, sizeof ub, "%s.shared_expert.up_proj.weight", pfx);
      se_gate_up(buf, ub, x, d_guv, d_guv + 640, 1, true);
    }
    k_silu_mul<<<(640 + 255) / 256, 256, 0, g_str>>>(d_guv, d_guv + 640, d_hid,
                                                     640);
    snprintf(buf, sizeof buf, "%s.shared_expert.down_proj.weight", pfx);
    gemv(wview(ckpt.at(buf)), d_hid, d_ey);
    snprintf(buf, sizeof buf, "%s.shared_expert_gate.weight", pfx);
    gemv(wview(ckpt.at(buf)), x, d_sg);
    static const bool axpy_out = !env_zero("QWENOX_AXPY_SG_OUT");
    if (axpy_out) {
      k_axpy_sg_out<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg, y,
                                                           g_cfg.d, g_cfg.d);
    } else {
      k_axpy_sg<<<(2560 + 255) / 256, 256, 0, g_str>>>(d_moeacc, d_ey, d_sg,
                                                       g_cfg.d, g_cfg.d);
      CK(hipMemcpyAsync(y, d_moeacc, g_cfg.d * 4, hipMemcpyDeviceToDevice, g_str));
    }
  }

  // One full MTP draft step: appends (tap, tok) at mtp_pos. Greedy callers get
  // the argmax; sampling callers request the full logits instead.
  // d_tok_in: read the input token from this device pointer instead of the
  // host `tok`. d_amx_out: write argmax here on-device and skip the D2H.
  int mtp_step(const float* tap, int tok,
               std::vector<float>* host_logits = nullptr,
               bool argmax_out = true, const int* d_tok_in = nullptr,
               int* d_amx_out = nullptr) {
    // B0: row mtp_pos is in the shared pool; drafts run ahead of the trunk
    // (rows pos-1 .. pos+gamma-2), so map / un-share the page first
    kv_prepare(mtp_pos, mtp_pos + 1);
    if (d_tok_in)
      CK(hipMemcpyAsync(d_mtok, d_tok_in, 4, hipMemcpyDeviceToDevice, g_str));
    else
      CK(hipMemcpyAsync(d_mtok, &tok, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
    mtp_input_1(tap);
    mtp_gr_read("mtp.layers.0.attn_hyper_connection", d_x);
    mtp_qsa(d_x, d_y);
    mtp_gr_write("mtp.layers.0.attn_hyper_connection", d_y);
    mtp_gr_read("mtp.layers.0.mlp_hyper_connection", d_x);
    mtp_moe(d_x, d_y);
    mtp_gr_write("mtp.layers.0.mlp_hyper_connection", d_y);
    mtp_gr_read("mtp.hyper_connection_mixer", d_x);
    WView lm = dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight"));
    const int dvn =
        (draft_vocab > 0 && !host_logits) ? std::min(draft_vocab, g_cfg.vocab)
                                          : 0;
    if (dvn)  // q4cp scales sit behind a codes region sized by lm.rows; nrows
      gemv(lm, d_x, d_logits, 0, (uint64_t)dvn);  // keeps that base intact
    else
      gemv(lm, d_x, d_logits);
    int id = -1;
    if (host_logits) {
      host_logits->resize(g_cfg.vocab);
      CK(hipMemcpy(host_logits->data(), d_logits, (size_t)g_cfg.vocab * 4,
                   hipMemcpyDeviceToHost));
    } else if (argmax_out) {
      int* dst = d_amx_out ? d_amx_out : d_argmax;
      argmax_rows(d_logits, dvn ? dvn : g_cfg.vocab, 1, d_amx_cand, dst,
                  g_str);
      if (!d_amx_out) CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    }
    mtp_pos++;
    return id;
  }

  // ---- batch ingest: build MTP KV over a token range (prompt/accepted) ----
  // Row j takes (tap[j] = trunk h_{base+j}, toks[j] = t_{base+j+1}) at MTP
  // position base+j. No mixer/lm_head: only the layer state (KV) is built.
  void mtp_input_b(const float* taps, int P) {
    embed_rows(wview(ckpt.at("embed_tokens.weight")), d_tokarr, P, d_eb, 2560);
    k_rmsnorm_zc_grouped<<<P, 1024, 0, g_str>>>(d_eb, d_mnorm_e, d_eb, 2560,
                                                g_cfg.norm_eps, 1);
    gemm(wview(ckpt.at("mtp.fc_embedding.weight")), d_eb, d_qnb, P);
    if (mtp_norm_mode == 0)
      k_rmsnorm_zc_grouped<<<P, 1024, 0, g_str>>>(taps, d_mnorm_h, d_mRhatb,
                                                  10240, g_cfg.norm_eps, 1);
    else
      k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(taps, d_mnorm_h, d_mRhatb,
                                                      2560, g_cfg.norm_eps, 4);
    if (mtp_hagg_mode == 0) {
      gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb, d_Unb, P,
           10240 /*xstride*/);
      for (int b = 1; b < 4; b++) {
        gemm(wview(ckpt.at("mtp.fc_hidden.weight")), d_mRhatb + (size_t)b * 2560,
             d_convoutb, P, 10240 /*xstride*/);
        k_acc<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
            d_Unb, d_convoutb, P * 2560);
      }
      k_acc<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
          d_Unb, d_qnb, P * 2560);
      k_row_head_copy<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                        g_str>>>(d_Unb, d_mRb, 2560);
      k_repeat4_b<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0, g_str>>>(
          d_mRb, d_mRb, g_cfg.d);
    } else {
      // H-agg=branch, branch-batched: d_mRhatb/d_mRb rows are [p*4+b], so
      // fc_hidden runs as one R=4P-row pass (chunked <=8-row direct reads for
      // spec-depth R, a single LT gemm past that) instead of 4 full passes +
      // 4 row copies.
      const WView& hw = wview(ckpt.at("mtp.fc_hidden.weight"));
      const int R = 4 * P;
      bool done = false;
      if (R <= 32) {
        done = true;
        for (int off = 0; off < R && done; off += 8)
          done = gemv_multi_q4cp(hw, d_mRhatb + (size_t)off * 2560,
                                 d_mRb + (size_t)off * 2560,
                                 std::min(8, R - off), 2560);
      }
      if (!done) {
        // d_xbf16b (BP*6144) overflows past P > 0.6*BP, so convert into the
        // 10240-wide d_xbf16 ourselves (mtp_gr_read_b pattern).
        uint64_t tot = (uint64_t)R * 2560;
        k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
            d_mRhatb, d_xbf16, 2560, R, 2560);
        gemm(hw, d_mRhatb, d_mRb, R, 2560, d_xbf16);
        if (xf16_src == d_xb) xf16_src = nullptr;  // d_xbf16 clobbered
      }
      k_acc_broadcast4<<<(unsigned)(((size_t)P * 2560 + 255) / 256), 256, 0,
                         g_str>>>(d_qnb, d_mRb, 2560);
    }
  }

  void mtp_gr_read_b(const std::string& prefix, int P) {
    char buf[128];
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_mRb, hw, d_mRhatb, 2560,
                                                    g_cfg.norm_eps, 4);
    // K=10240 gemms would overflow d_xbf16b (maxbatch*6144); convert into the
    // 10240-wide d_xbf16 ourselves and pass it as xbf. (d_xbf16 is clobbered
    // by later d_xb gemms, so gr_write_b reconverts.)
    uint64_t tot = (uint64_t)P * 10240;
    k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
        d_mRhatb, d_xbf16, 10240, P, 10240);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_t320b, P, 0, d_xbf16);
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f,
                                                           P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_t320b, d_Gb, P);
    k_sigmoid_inplace<<<(unsigned)(((size_t)P * 10240 + 255) / 256), 256, 0,
                        g_str>>>(d_Gb, P * 10240);
    k_gr_combine_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_Gb, d_mRhatb, d_xb, g_cfg.d, g_cfg.branches, P);
    if (xf16_src == d_xb) xf16_src = nullptr;
  }
  void mtp_gr_write_b(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    uint64_t tot = (uint64_t)P * 10240;
    k_f32_to_bf16_v4<<<(unsigned)((tot / 4 + 255) / 256), 256, 0, g_str>>>(
        d_mRhatb, d_xbf16, 10240, P, 10240);
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_w4b, P, 0, d_xbf16);
    k_gr_write_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_w4b, d_mRb, d_yb,
                                                            g_cfg.d,
                                                            g_cfg.branches, P);
  }

  // ---- D2-S3c: batched cross-slot draft (rendezvous executor) ----------
  // mtp_gr_read_d/write_d are mtp_step's GR read/write lifted to P rows.
  // Unlike mtp_gr_read_b/write_b (prefill ingest; bf16 gemm x), these stay
  // on the fp32 multi-row gemv (P <= 8: k_q4cp_gemv_mr -- per-row
  // bit-identical to the serial k_q4cp_gemv), so a batched draft round
  // matches the serial per-slot mtp_step bursts bit for bit.
  void mtp_gr_read_d(const std::string& prefix, int P) {
    char buf[128];
    float* hw = ensure_hcnorm(prefix);
    k_rmsnorm_zc_grouped<<<4 * P, 1024, 0, g_str>>>(d_mRb, hw, d_mRhatb, 2560,
                                                    g_cfg.norm_eps, 4);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_down.weight",
             prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_t320b, P);
    k_silu_scale<<<(P * 320 + 255) / 256, 256, 0, g_str>>>(d_t320b, 0.25f,
                                                           P * 320);
    snprintf(buf, sizeof buf, "%s.input_mix_weight_up.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_t320b, d_Gb, P);
    k_sigmoid_inplace<<<(unsigned)(((size_t)P * 10240 + 255) / 256), 256, 0,
                        g_str>>>(d_Gb, P * 10240);
    k_gr_combine_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(
        d_Gb, d_mRhatb, d_xb, g_cfg.d, g_cfg.branches, P);
    if (xf16_src == d_xb) xf16_src = nullptr;  // d_xb clobbered
  }
  void mtp_gr_write_d(const std::string& prefix, int P) {
    char buf[128];
    snprintf(buf, sizeof buf, "%s.block_inject_weight.weight", prefix.c_str());
    gemm(wview(ckpt.at(buf)), d_mRhatb, d_w4b, P);
    k_gr_write_b<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_w4b, d_mRb, d_yb,
                                                            g_cfg.d,
                                                            g_cfg.branches, P);
  }

  // One-time gate: every gemm on the batched path must land on the fp32
  // multi-row gemv, whose per-row results must be bit-identical to the
  // serial single-row gemv. Established for q4cp (k_q4cp_gemv_mr shares the
  // v3 fma order and reduction tree), q8g64 (same per-row math in
  // k_q8g64_gemv_mr) and ht/q6g64 (ktest 1h locks gemv_multi row invariance
  // for P=1..48 on the full production shape family). q8g32/bf16 multi-row
  // kernels are NOT proven against their serial gemv -> keep the serial
  // per-slot burst for those.
  bool mtp_draft_batch_ok() {
    if (dr_ok_ >= 0) return dr_ok_ > 0;
    bool ok = !qsa_dense && draft_vocab <= 0 && !getenv("QWENOX_NO_GEMVMR") &&
              gemvmr_maxp() >= 8;
    const char* wts[] = {
        "mtp.fc_embedding.weight",
        "mtp.fc_hidden.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.layers.0.self_attn.v_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
        "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
        "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
        "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    };
    auto dr_dtype_ok = [](uint32_t dt) {
      return dt == 5 || dt == 7 || dt == 16 || dt == 24;
    };
    for (const char* w : wts)
      if (ok && !dr_dtype_ok(wview(ckpt.at(w)).dtype)) ok = false;
    if (ok && !dr_dtype_ok((dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight"))).dtype))
      ok = false;
    dr_ok_ = ok ? 1 : 0;
    return ok;
  }
  int dr_ok_ = -1;
  // Would this worker join the draft rendezvous for this request? The
  // membership count (DrActive/DrLoop) and the submit checks must use this
  // one predicate, else the all-arrived trigger stalls to the timeout.
  bool dr_worker_ok() {
    return dr_pool_fn && mtp_draft_batch_ok() && !mrope_active_;
  }

  // Run one draft round for n slots as a per-step batch. segs[i]: slot s_i
  // drafts g_i tokens from (slot_d_R(s_i), tok0_i). Step j batches the
  // members {i : g_i > j} through the P-row buffers; the QSA q/k/v/o
  // projections run batched (row-invariant gemv_multi), the QSA mid-section
  // (indexer, rope, KV store, flash, gate) and MoE run serially per member
  // on its own slot state via the slot accessors (no bind).
  // Every kernel is per-row identical to the serial mtp_step chain, so the
  // drafts -- and therefore the accept stats -- match the serial burst bit
  // for bit. Guards run before any GPU work; a refusal (false) leaves every
  // slot untouched and the callers fall back to their serial bursts.
  bool mtp_draft_batch(const DrSeg* segs, int n) {
    if (!segs || n < 1 || n > 8 || n > maxbatch || ws_lent_) return false;
    if (!mtp_draft_batch_ok()) return false;
    int gmax = 0;
    for (int i = 0; i < n; i++) {
      const int s = segs[i].slot;
      if (s < 0 || s >= nslots || segs[i].g < 1 || segs[i].g > 8 ||
          !segs[i].drafts)
        return false;
      if (slot_mrope_active_(s) || slot_mtp_tap_capture(s) ||
          slot_prog_total(s) > 0)
        return false;
      gmax = std::max(gmax, segs[i].g);
    }
    if (!d_tokmat) DALLOC(&d_tokmat, 64 * sizeof(int));
    const int rb = g_cfg.branches * g_cfg.d;
    int lists[8][8];  // step j's member seg indices (submission order)
    int ln[8] = {};
    int gcur[8];      // steps each member still runs (a sampled member may stop early)
    int stok[8][8];   // sampled draws, host copy (lives until the final D2H)
    for (int i = 0; i < n; i++) gcur[i] = segs[i].g;
    for (int j = 0; j < gmax; j++) {
      int cur[8], prow[8], B = 0;
      if (j == 0) {
        for (int i = 0; i < n; i++) { cur[B] = i; prow[B] = -1; B++; }
      } else {
        for (int r = 0; r < ln[j - 1]; r++)
          if (gcur[lists[j - 1][r]] > j) {
            cur[B] = lists[j - 1][r];
            prow[B] = r;
            B++;
          }
      }
      if (B == 0) break;  // every member stopped
      for (int r = 0; r < B; r++) lists[j][r] = cur[r];
      ln[j] = B;
      // mtp_step's per-slot prologue: map the MTP KV row, upload the position
      for (int r = 0; r < B; r++) {
        const int s = segs[cur[r]].slot;
        const int mp = slot_mtp_pos(s);
        kv_prepare_seq(slot_kv_seq(s), slot_d_ptab(s), mp, mp + 1);
        CK(hipMemcpyAsync(slot_d_mpos(s), &mp, 4, hipMemcpyHostToDevice,
                          g_str));
      }
      // tokens: step 0 takes the committed token from the host; later steps
      // gather each member's own previous argmax out of the token matrix
      if (j == 0) {
        int t0[8];
        for (int r = 0; r < B; r++) t0[r] = segs[cur[r]].tok0;
        CK(hipMemcpyAsync(d_tokarr, t0, (size_t)B * 4, hipMemcpyHostToDevice,
                          g_str));
      } else {
        for (int r = 0; r < B; r++)
          CK(hipMemcpyAsync(d_tokarr + r, d_tokmat + (j - 1) * 8 + prow[r], 4,
                            hipMemcpyDeviceToDevice, g_str));
      }
      // taps: step 0 reads the trunk h (slot d_R), later steps the slot's
      // own draft R (scattered back to its home row at each step's end)
      for (int r = 0; r < B; r++) {
        const float* src = j == 0 ? slot_d_R(segs[cur[r]].slot)
                                  : slot_d_mR(segs[cur[r]].slot);
        CK(hipMemcpyAsync(d_mtap + (size_t)r * rb, src, (size_t)rb * 4,
                          hipMemcpyDeviceToDevice, g_str));
      }
      mtp_input_b(d_mtap, B);
      mtp_gr_read_d("mtp.layers.0.attn_hyper_connection", B);
      {
        const char* pfx = "mtp.layers.0.self_attn";
        char buf[128];
        snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_xb, d_qgb, B);
        snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_xb, d_kbb, B);
        snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_xb, d_vbb, B);
        for (int r = 0; r < B; r++)
          mtp_qsa_core_slot(segs[cur[r]].slot, r);
        snprintf(buf, sizeof buf, "%s.o_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_attnb, d_yb, B);
      }
      mtp_gr_write_d("mtp.layers.0.attn_hyper_connection", B);
      mtp_gr_read_d("mtp.layers.0.mlp_hyper_connection", B);
      for (int r = 0; r < B; r++)
        mtp_moe(d_xb + (size_t)r * 2560, d_yb + (size_t)r * 2560);
      mtp_gr_write_d("mtp.layers.0.mlp_hyper_connection", B);
      mtp_gr_read_d("mtp.hyper_connection_mixer", B);
      WView lm = dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight"));
      gemm(lm, d_xb, d_logits_b, B);
      argmax_rows(d_logits_b, g_cfg.vocab, B, d_amx_cand, d_tokmat + j * 8,
                  g_str);
      for (int r = 0; r < B; r++) {
        const int s = segs[cur[r]].slot;
        CK(hipMemcpyAsync(slot_d_mR(s), d_mRb + (size_t)r * rb, (size_t)rb * 4,
                          hipMemcpyDeviceToDevice, g_str));
        slot_mtp_pos(s)++;
        // draft 1's indexer ring rows are truth; later steps pollute them
        if (j == 0)
          CK(hipMemcpyAsync(slot_d_mring_ckpt(s), slot_d_miraw(s), 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
      }
      // C7: sampled members draw on the host from their own row (same logits
      // as the serial mtp_step: row-invariant kernels), then the draw replaces
      // the argmax in the token matrix that step j+1 gathers from.
      for (int r = 0; r < B; r++) {
        const DrSeg& sg = segs[cur[r]];
        if (!sg.step) continue;
        if (!(*sg.step)(j, d_logits_b + (size_t)r * g_cfg.vocab, stok[j][r]))
          gcur[cur[r]] = j + 1;
        CK(hipMemcpyAsync(d_tokmat + j * 8 + r, &stok[j][r], 4, hipMemcpyHostToDevice,
                          g_str));
      }
    }
    int host[64];
    CK(hipMemcpy(host, d_tokmat, sizeof host, hipMemcpyDeviceToHost));
    for (int j = 0; j < gmax; j++)
      for (int r = 0; r < ln[j]; r++)
        segs[lists[j][r]].drafts[j] = host[j * 8 + r];
    return true;
  }

  // Batch QSA for the MTP layer (mirrors qsa_b flash path).
  // kv_only (aggr 10-05): ingest needs only the persistent state this layer
  // leaves behind — indexer keys + raw ring and the K/V rows. The query path,
  // selection, flash, gate and o_proj outputs are never read by any ingest
  // caller, so they are skipped (identical KV -> identical drafts).
  void mtp_qsa_b(int P, int base, bool kv_only = false) {
    const char* pfx = "mtp.layers.0.self_attn";
    char buf[128];
    // Table-driven rope (M-RoPE requests carry their 3-row positions in the
    // table; text gets the same absolute-position table, bit-identical to
    // the inline fp64 path it replaces).
    ensure_rope_tab(g_cfg.rope_theta, 64);
    ensure_rope_cs(base, P);
    if (!qsa_dense) {
      if (index_f32) {
        iproj_sgemm(d_miw, d_xb, d_iproj, P);
      } else {
        snprintf(buf, sizeof buf, "%s.indexer.index_qk_proj.weight", pfx);
        gemm(wview(ckpt.at(buf)), d_xb, d_iproj, P);
      }
      if (!kv_only)
      k_index_q<<<dim3(P, 4), 128, 0, g_str>>>(d_iproj, d_minorm, d_iq, P,
                                               g_cfg.norm_eps, g_cfg.rope_theta,
                                               nullptr, base, d_ropecs, nullptr,
                                               nullptr, d_iqh);
      uint16_t* const mikh =
          d_ikh ? d_ikh + (size_t)(g_cfg.layers / 4) * ik_layer_blocks * 128
                : nullptr;
      int nb = (base + P) / 4, b0 = base / 4;
      if (nb > b0)
        k_index_pool<<<nb - b0, 128, 0, g_str>>>(d_iproj, d_minorm + 128, d_mik,
                                                 g_cfg.norm_eps,
                                                 g_cfg.rope_theta, base, d_miraw,
                                                 d_ropecs, kvpt(), nullptr, mikh);
      if (cut_abs > 0 && d_cmiraw && cut_abs - 1 >= base && cut_abs - 1 <= base + P) {
        // in-pass cut: MTP lockstep at the cut = rows [0, cut-1) ingested
        CK(hipMemcpyAsync(d_cmiraw, d_miraw, 4 * 128 * 4, hipMemcpyDeviceToDevice, g_str));
        const int rr = cut_abs - 1 - base;
        if (rr > 0)
          k_index_ring<<<min(rr, 4), 128, 0, g_str>>>(d_iproj, d_cmiraw, rr, base);
        cut_mtp_hit = true;
      }
      k_index_ring<<<min(P, 4), 128, 0, g_str>>>(d_iproj, d_miraw, P, base);
      // paged: gather the keys contiguous for the scores (as trunk qsa_b)
      const float* skeys = d_mik;
      const uint16_t* skeys16 = mikh;
      if (!kv_only && kv_paged && nb > 0 && std::max(2051, base) < base + P) {
        k_ik_gather<<<nb, 128, 0, g_str>>>(d_mik, d_ptab, d_ikg, nb);
        skeys = d_ikg;
        if (mikh) {
          k_ik_gather16<<<nb, 128, 0, g_str>>>(mikh, d_ptab, d_ikgh, nb);
          skeys16 = d_ikgh;
        }
      }
      const float one = 1.f, zero = 0.f;
      const int a0 = std::max(2051, base);
      const int grid =
          2051 + (a0 - 2051 + INDEX_BATCH - 1) / INDEX_BATCH * INDEX_BATCH;
      for (int first = kv_only ? base + P : a0; first < base + P;) {
        int nxt = first < grid ? min(grid, base + P)
                               : min(first + INDEX_BATCH, base + P);
        int count = nxt - first;
        // Same fused tiled scores + select as the trunk qsa_b path (the MTP
        // sgemm alone was ~0.23 s per 32K prompt, the tiled kernel ~0.03 s).
        // Only the MTP draft path sees these selections.
        // QWENOX_MTP_INDEX_SGEMM=1 keeps the old sgemm + index_select.
        static const bool mtp_sgemm = getenv("QWENOX_MTP_INDEX_SGEMM") != nullptr;
        if (index_fused2 && !mtp_sgemm && count >= 16 &&
            getenv("QWENOX_INDEX_OLDSEL") == nullptr) {
          int* const selrow = d_iselected + (size_t)(first - base) * 512;
          if (index_f16()) {
            uint16_t* const sc16 = (uint16_t*)d_iscores;
            k_index_scores_wmma16<false>
                <<<index_wmma16_grid(nb, count), 256, 0, g_str>>>(
                    d_iqh + (size_t)(first - base) * 512, skeys16, sc16, nb,
                    count, first);
            if ((first + count) / 4 <= 8192)
              k_index_select_rs<true, __half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            else
              k_index_select_2p<__half><<<count, 256, 0, g_str>>>(
                  (const __half*)sc16, nb, selrow, first);
            first = nxt;
            continue;
          }
          if (index_score64)
            k_index_scores_t64<<<index_t64_grid(nb, count), 256, 0, g_str>>>(
                d_iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          else
            k_index_scores_tiled<<<dim3((nb + 31) / 32, (count + 15) / 16), 256, 0, g_str>>>(
                d_iq + (size_t)(first - base) * 512, skeys, d_iscores, nb, count, first);
          if ((first + count) / 4 <= 8192)
            k_index_select_rs<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else if (index_sel2p)
            k_index_select_2p<<<count, 256, 0, g_str>>>(d_iscores, nb, selrow, first);
          else if (index_stream_select)
            k_index_select_stream<true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          else
            k_index_select<16, true><<<count, 256, 0, g_str>>>(
                d_iscores, nb, selrow, first);
          first = nxt;
          continue;
        }
        auto status = rocblas_sgemm(rbh, rocblas_operation_transpose,
                                    rocblas_operation_none, nb, count * 4, 128,
                                    &one, skeys, 128,
                                    d_iq + (size_t)(first - base) * 512, 128,
                                    &zero, d_iscores, nb);
        if (status != rocblas_status_success) {
          fprintf(stderr, "mtp indexer sgemm failed: %d\n", (int)status);
          exit(1);
        }
        index_select(d_iscores, nb, d_iselected + (size_t)(first - base) * 512,
                     first, count);
        first = nxt;
      }
    }
    if (!kv_only) {
      snprintf(buf, sizeof buf, "%s.q_proj.weight", pfx);
      gemm(wview(ckpt.at(buf)), d_xb, d_qgb, P);
    }
    snprintf(buf, sizeof buf, "%s.k_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_kbb, P);
    snprintf(buf, sizeof buf, "%s.v_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_vbb, P);
    snprintf(buf, sizeof buf, "%s.q_norm.weight", pfx);
    float* qnw = ensure_norm256(buf);
    snprintf(buf, sizeof buf, "%s.k_norm.weight", pfx);
    float* knw = ensure_norm256(buf);
    if (!kv_only)
      k_qsa_qsplit<true><<<dim3(24, P), 256, 0, g_str>>>(d_qgb, qnw, d_qsb, d_gsb,
                                                         256, g_cfg.norm_eps, base,
                                                         d_ropecs);
    k_qsa_kprep<false><<<dim3(2, P), 256, 0, g_str>>>(d_kbb, knw, d_kbb,
                                                      nullptr, g_cfg.norm_eps,
                                                      base, d_ropecs);
    int dense_P = qsa_dense ? P : min(P, std::max(0, 2051 - base));
    // rows [base, base+P) of pool layer nqsa: one copy / convert per
    // physically contiguous page run (a single one unpaged)
    if (qsa_kv_fp32) {
      kv_rows_store(d_mkcf, d_kbb, base, P);
      kv_rows_store(d_mvcf, d_vbb, base, P);
      if (!kv_only) qsa_flash_b(d_mkcf, d_mvcf, P, dense_P, base, nullptr, kvpt());
    } else {
      kv_runs(base, base + P, KV_PAGE, [&](int lr, size_t pr, int n) {
        const size_t so = (size_t)(lr - base) * 512;
        const unsigned g = (unsigned)(((size_t)n * 128 + 255) / 256);
        k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(d_kbb + so, d_mkcb + pr * 512, 512, n,
                                               512);
        k_f32_to_bf16_v4<<<g, 256, 0, g_str>>>(d_vbb + so, d_mvcb + pr * 512, 512, n,
                                               512);
      });
      if (!kv_only) qsa_flash_b(d_mkcb, d_mvcb, P, dense_P, base, nullptr, kvpt());
    }
    if (kv_only) return;
    k_sigmoid_gate<<<(P * 6144 + 255) / 256, 256, 0, g_str>>>(d_attnb, d_gsb,
                                                              d_attnb, P * 6144);
    snprintf(buf, sizeof buf, "%s.o_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_attnb, d_yb, P);
  }

  // Batch MoE for the MTP layer: default deterministic packed path only
  // (mtp_init refuses non-default MoE flags).
  void mtp_moe_b(int P) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    snprintf(buf, sizeof buf, "%s.gate.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_routerb, P);
    int k = g_cfg.topk, E = g_cfg.experts, npairs = P * k;
    router_topk_b(P, k);
    if (gg_mtp_on()) {  // GGUF sidecar experts; overwrites d_moeaccb
      moe_gguf(g_ggx_mtp, d_xb, d_topidsb, d_topwb, d_moeaccb, P);
      mtp_moe_b_shared(P);
      return;
    }
    snprintf(buf, sizeof buf, "%s.experts.gate_up_proj.weight", pfx);
    const Tensor& gu_t = ckpt.at(buf);
    WView gu = wview(gu_t);
    snprintf(buf, sizeof buf, "%s.experts.down_proj.weight", pfx);
    const Tensor& dn_t = ckpt.at(buf);
    WView dn = wview(dn_t);
    if (P > moe_naive_max && moe_q4w_run(gu, dn, d_xb, d_topidsb, d_topwb, d_moeaccb, P)) {
      mtp_moe_b_shared(P);
      return;
    }
    CK(hipMemsetAsync(d_moeaccb, 0, (size_t)P * g_cfg.d * 4, g_str));
    moe_routing.run(d_topidsb, d_topwb, d_tokidx, d_pweight, d_eoff, d_pairids,
                    P, k, E);
    int tasks = (npairs + 63) / 64 + E;
    k_moe_tiles<<<1, 512, 0, g_str>>>(d_eoff, d_moetiles, d_moentiles, E, 64);
    k_f32_to_bf16_v4<<<(unsigned)(((size_t)P * g_cfg.d / 4 + 255) / 256), 256, 0,
                       g_str>>>(d_xb, d_xbf16, g_cfg.d, P, g_cfg.d);
    k_moe_w4_up<true, true><<<dim3(g_cfg.moe_mid / MOE_CT, tasks), 256, 0,
                               g_str>>>(
        gu.data + 64, gu.data + 64 + gu.rows * gu.cols / 2,
        (const float*)gu.data, d_xb, d_tokidx, d_eoff, nullptr,
        2 * g_cfg.moe_mid, gu.cols, g_cfg.moe_mid, gu.scale_stride, d_moetiles,
        d_moentiles, d_xbf16, (uint16_t*)d_hidb);
    k_moe_w4_down<false, true, true><<<dim3(g_cfg.d / FD_CT_DN, tasks), 256, 0,
                                        g_str>>>(
        dn.data + 64, dn.data + 64 + dn.rows * dn.cols / 2,
        (const float*)dn.data, nullptr, d_tokidx, d_pweight, d_eoff, d_guvb,
        g_cfg.d, dn.cols, dn.scale_stride, d_moetiles, d_moentiles,
        (const uint16_t*)d_hidb);
    k_moe_reduce_fast<10><<<dim3((g_cfg.d + 255) / 256, P), 256, 0, g_str>>>(
        d_guvb, d_pairids, d_moeaccb, P, g_cfg.d);
    mtp_moe_b_shared(P);
  }
  // shared expert + gate into d_moeaccb, then d_yb = d_moeaccb
  void mtp_moe_b_shared(int P) {
    const char* pfx = "mtp.layers.0.mlp";
    char buf[128];
    {
      char ub[128];
      snprintf(buf, sizeof buf, "%s.shared_expert.gate_proj.weight", pfx);
      snprintf(ub, sizeof ub, "%s.shared_expert.up_proj.weight", pfx);
      se_gate_up(buf, ub, d_xb, d_guvb, d_guvb + (size_t)P * 640, P, false);
    }
    k_silu_mul<<<(P * 640 + 255) / 256, 256, 0, g_str>>>(
        d_guvb, d_guvb + (size_t)P * 640, d_hidb, P * 640);
    snprintf(buf, sizeof buf, "%s.shared_expert.down_proj.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_hidb, d_eyb, P);
    snprintf(buf, sizeof buf, "%s.shared_expert_gate.weight", pfx);
    gemm(wview(ckpt.at(buf)), d_xb, d_sgb, P);
    k_axpy_sg<<<(P * 2560 + 255) / 256, 256, 0, g_str>>>(d_moeaccb, d_eyb, d_sgb,
                                                         g_cfg.d, P * g_cfg.d);
    CK(hipMemcpyAsync(d_yb, d_moeaccb, (size_t)P * g_cfg.d * 4,
                      hipMemcpyDeviceToDevice, g_str));
  }

  void mtp_ingest_b(const float* taps, const std::vector<int>& toks, int P,
                    int base) {
    if (P < 1) return;
    kv_prepare(base, base + P);  // B0: MTP rows live in the shared pool
    CK(hipMemcpyAsync(d_tokarr, toks.data(), P * 4, hipMemcpyHostToDevice,
                      g_str));
    mtp_input_b(taps, P);
    xf16_src = nullptr;
    mtp_gr_read_b("mtp.layers.0.attn_hyper_connection", P);
    // aggr 10-05: ingest = KV/indexer build only (see mtp_qsa_b kv_only). The
    // attention output, the MLP-site HC and the MoE of ingested rows were
    // never consumed (~80% of a 16K prompt's 425 ms MTP ingest).
    // QWENOX_MTP_INGEST_FULL=1 runs the full layer as before.
    static const bool full = getenv("QWENOX_MTP_INGEST_FULL") != nullptr;
    static const bool kvo_qsa = !full && !env_zero("QWENOX_MTP_KVO_QSA");
    static const bool kvo_rest = !full && !env_zero("QWENOX_MTP_KVO_REST");
    mtp_qsa_b(P, base, kvo_qsa);
    static const int rest_lvl = [] {  // bisect: how many of the 4 rest steps run
      const char* e = getenv("QWENOX_MTP_REST_LVL");
      return e ? atoi(e) : -1;
    }();
    const int lvl = rest_lvl >= 0 ? rest_lvl : (kvo_rest ? 0 : 4);
    if (lvl >= 1) mtp_gr_write_b("mtp.layers.0.attn_hyper_connection", P);
    if (lvl >= 2) mtp_gr_read_b("mtp.layers.0.mlp_hyper_connection", P);
    if (lvl >= 3) mtp_moe_b(P);
    if (lvl >= 4) mtp_gr_write_b("mtp.layers.0.mlp_hyper_connection", P);
    if (getenv("QWENOX_MTP_HASH")) {  // debug: hash the MTP persistent state rows
      CK(hipStreamSynchronize(g_str));
      auto fnv = [](uint64_t h, const void* p, size_t n) {
        const uint8_t* b = (const uint8_t*)p;
        for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ULL;
        return h;
      };
      uint64_t hk = 1469598103934665603ULL, hv = hk, hi = hk, hr = hk;
      std::vector<uint8_t> tmp;
      kv_runs(base, base + P, KV_PAGE, [&](int, size_t pr, int n) {
        const size_t bytes = (size_t)n * 512 * (qsa_kv_fp32 ? 4 : 2);
        tmp.resize(bytes);
        CK(hipMemcpy(tmp.data(), qsa_kv_fp32 ? (void*)(d_mkcf + pr * 512) : (void*)(d_mkcb + pr * 512),
                     bytes, hipMemcpyDeviceToHost));
        hk = fnv(hk, tmp.data(), bytes);
        CK(hipMemcpy(tmp.data(), qsa_kv_fp32 ? (void*)(d_mvcf + pr * 512) : (void*)(d_mvcb + pr * 512),
                     bytes, hipMemcpyDeviceToHost));
        hv = fnv(hv, tmp.data(), bytes);
      });
      const int nb = (base + P) / 4, b0 = base / 4;
      if (!qsa_dense && nb > b0)
        kv_runs(b0, nb, KV_PAGE_BLOCKS, [&](int, size_t pb, int n) {
          tmp.resize((size_t)n * 128 * 4);
          CK(hipMemcpy(tmp.data(), d_mik + pb * 128, tmp.size(), hipMemcpyDeviceToHost));
          hi = fnv(hi, tmp.data(), tmp.size());
        });
      if (d_miraw) {
        tmp.resize(512 * 4);
        CK(hipMemcpy(tmp.data(), d_miraw, tmp.size(), hipMemcpyDeviceToHost));
        hr = fnv(hr, tmp.data(), tmp.size());
      }
      fprintf(stderr, "[mtp-hash] base=%d P=%d K=%016llx V=%016llx ik=%016llx ring=%016llx\n",
              base, P, (unsigned long long)hk, (unsigned long long)hv,
              (unsigned long long)hi, (unsigned long long)hr);
    }
    // The MTP layer's d_xb rows live in the trunk's shared batch buffers: never
    // leave a bf16-conversion cache entry for them behind (with the KV-only
    // ingest a later same-shape gemm(d_xb) reused the MTP input — bisected
    // 10-05: output diverged from plain greedy until this reset).
    xf16_src = nullptr;
    mtp_pos = base + P;
    CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
  }

  // Acceptance-rate harness: after prefill_batch(prompt) with tap capture,
  // ingest the prompt into the MTP layer, then walk K trunk steps; at each,
  // the draft (from the frozen pre-step d_R) must match the trunk's next
  // argmax. Prints top-1 acceptance.
  void mtp_accept_test(const std::vector<int>& prompt, int first_id, int K) {
    int P = (int)prompt.size();
    if (P > maxbatch) {
      fprintf(stderr, "mtp-test: prompt %d > maxbatch %d\n", P, maxbatch);
      exit(1);
    }
    std::vector<int> shifted(prompt.begin() + 1, prompt.end());
    mtp_ingest_b(d_mtap, shifted, P - 1, 0);
    int cur = first_id, hits = 0, n = 0;
    auto T0 = std::chrono::steady_clock::now();
    for (int g = 0; g < K; g++) {
      int draft = mtp_step(d_R, cur);  // predicts the token after next
      int nxt = forward(cur);          // trunk commits cur
      hits += (draft == nxt);
      n++;
      cur = nxt;
      if (g % 32 == 31 || g == K - 1) {
        double sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - T0)
                         .count();
        fprintf(stderr, "\rmtp-test %d/%d acc=%.4f (%.1f steps/s)", n, K,
                (double)hits / n, n / sec);
        fflush(stderr);
      }
    }
    fprintf(stderr, "\nmtp-test: top-1 acceptance %d/%d = %.4f (norm=%s hagg=%s)\n",
            hits, n, (double)hits / n, mtp_norm_mode ? "branch" : "full",
            mtp_hagg_mode ? "branch" : "sum");
  }

  // =================== MTP speculative loop (Phase B) ========================
  // gr_bf16 shims: d_Rb rows are bf16 under QWENOX_GR_BF16, but the MTP ingest
  // path and the decode state d_R are fp32. spec_taps stages rows
  // row0..row0+P-1 as fp32 in d_Rhatb (scratch outside prefill; consumed by
  // mtp_input_b before anything else touches it). row0 is the slot's row base
  // in d_Rb for the D2 rendezvous; 0 for B=1.
  const float* spec_taps(int P, int row0 = 0) {
    if (!gr_bf16) return d_Rb + (size_t)row0 * 10240;
    uint64_t n = (uint64_t)P * 10240;
    k_bf16_to_f32<<<(unsigned)((n + 255) / 256), 256, 0, g_str>>>(
        (const uint16_t*)d_Rb + (size_t)row0 * 10240, d_Rhatb, n);
    return d_Rhatb;
  }
  void restore_dR_row(int d, int row0 = 0) {
    if (gr_bf16)
      k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
          (const uint16_t*)d_Rb + (size_t)(row0 + d) * 10240, d_R, 10240);
    else
      CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(row0 + d) * 10240, 10240 * 4,
                        hipMemcpyDeviceToDevice, g_str));
  }
  // Spec checkpoints: only the position-mod rings (PLE ring + QSA indexer raw
  // ring) are needed — GDN S / convst are restored from per-row snapshots.
  void mtp_ckpt_save() {
    float* p = d_ckpt + (ckpt_S + ckpt_convst) / 4;
    CK(hipMemcpyAsync(p, d_ring, ckpt_ring, hipMemcpyDeviceToDevice, g_str));
    p += ckpt_ring / 4;
    if (ckpt_iraw)
      CK(hipMemcpyAsync(p, d_iraw, ckpt_iraw, hipMemcpyDeviceToDevice, g_str));
  }

  // Produce logits for every batch row's final stream. Rows remain live in
  // d_Rb after prefill_batch, and the multi-row lm_head streams weights once.
  void batch_logits_device(int P) {
    batch_logits_device_(P);
    if (viz::on()) {  // QWENOX_VIZ: top-5 of each row (row positions as tapped)
      const VizTap t = vz_tap(g_cfg.layers, true, true, P);
      if (t.ring) k_viz_top5<<<P, 256, 0, g_str>>>(d_logits_b, g_cfg.vocab, t);
    }
  }
  void batch_logits_device_(int P) {
    gr_read_b("hyper_connection_mixer", P);
    const WView& lw = wview(ckpt.at("lm_head.weight"));
    // QWENOX_DENSE_WMMA: k_ht_wmma takes all P <= 48 rows in one weight
    // stream (rows are independent, so == the 8-row groups bit for bit).
    if (((lw.dtype == 16 && dense_wmma_on()) || (lw.dtype == 8 && dense_wmma_q8_on())) &&
        gemv_multi_q4cp(lw, d_xb, d_logits_b, P, 2560))
      return;
    // The multi-row kernel is specialized for up to eight rows. Chunk larger
    // ngram proposals into groups of eight so lm_head weights are streamed
    // once per group instead of once per row.
    for (int off = 0; off < P; ) {
      const int q = std::min(8, P - off);
      const bool ok = gemv_multi_q4cp(
          lw, d_xb + (size_t)off * 2560,
          d_logits_b + (size_t)off * g_cfg.vocab, q, 2560);
      if (!ok) {
        for (int r = 0; r < q; r++) {
          gemv(lw, d_xb + (size_t)(off + r) * 2560,
               d_logits_b + (size_t)(off + r) * g_cfg.vocab);
        }
      }
      off += q;
    }
  }

  void batch_logits(int P, std::vector<float>& host) {
    batch_logits_device(P);
    host.resize((size_t)P * g_cfg.vocab);
    CK(hipMemcpy(host.data(), d_logits_b, host.size() * sizeof(float),
                 hipMemcpyDeviceToHost));
  }

  void batch_nll(int P, int first, int scored, const int* targets,
                 std::vector<float>& losses) {
    losses.resize(scored);
    if (!scored) return;
    gr_read_b("hyper_connection_mixer", P);
    const WView& lw = wview(ckpt.at("lm_head.weight"));
    float* d_losses = d_logits_b + (size_t)64 * g_cfg.vocab;
    for (int off = first; off < first + scored; off += 64) {
      int rows = std::min(64, first + scored - off);
      for (int group = 0; group < rows; group += 8) {
        int count = std::min(8, rows - group);
        if (!gemv_multi_q4cp(lw, d_xb + (size_t)(off + group) * 2560,
                              d_logits_b + (size_t)group * g_cfg.vocab,
                              count, 2560)) {
          for (int row = 0; row < count; row++)
            gemv(lw, d_xb + (size_t)(off + group + row) * 2560,
                 d_logits_b + (size_t)(group + row) * g_cfg.vocab);
        }
      }
      CK(hipMemcpyAsync(d_argmax_arr, targets + off - first, rows * sizeof(int),
                        hipMemcpyHostToDevice, g_str));
      k_batch_nll<<<rows, 256, 0, g_str>>>(d_logits_b, g_cfg.vocab,
                                           d_argmax_arr, d_losses);
      CK(hipMemcpy(losses.data() + off - first, d_losses, rows * sizeof(float),
                   hipMemcpyDeviceToHost));
    }
  }

  // Full logits of rows [first, first+n) of the last prefill chunk (P rows),
  // row-major [n][vocab] on the host; same 64-row lm_head passes as batch_nll.
  void batch_logits_rows(int P, int first, int n, std::vector<float>& host) {
    host.resize((size_t)n * g_cfg.vocab);
    if (!n) return;
    gr_read_b("hyper_connection_mixer", P);
    const WView& lw = wview(ckpt.at("lm_head.weight"));
    for (int off = first; off < first + n; off += 64) {
      int rows = std::min(64, first + n - off);
      for (int group = 0; group < rows; group += 8) {
        int count = std::min(8, rows - group);
        if (!gemv_multi_q4cp(lw, d_xb + (size_t)(off + group) * 2560,
                              d_logits_b + (size_t)group * g_cfg.vocab,
                              count, 2560)) {
          for (int row = 0; row < count; row++)
            gemv(lw, d_xb + (size_t)(off + group + row) * 2560,
                 d_logits_b + (size_t)(group + row) * g_cfg.vocab);
        }
      }
      CK(hipMemcpy(host.data() + (size_t)(off - first) * g_cfg.vocab, d_logits_b,
                   (size_t)rows * g_cfg.vocab * sizeof(float), hipMemcpyDeviceToHost));
    }
  }

  // Greedy reduction of the same batched logits.
  void batch_argmax(int P, int* host) {
    batch_logits_device(P);
    argmax_rows(d_logits_b, g_cfg.vocab, P, d_amx_cand, d_argmax_arr, g_str);
    CK(hipMemcpy(host, d_argmax_arr, P * 4, hipMemcpyDeviceToHost));
  }

  // ---- GPU sampling prep (spec sampling fast path) ----
  // Corrections (bias/penalties) and the top-k pre-filter of
  // HostSampler::prepare run on device; only k (id, logit) pairs per row come
  // back and the host replays the remaining filter chain on them. The
  // distribution is unchanged; candidate order is (value desc, id asc)
  // instead of nth_element's arbitrary order, so same-seed draws can differ
  // from the dense path (allowed: nth_element never promised an order).
  template <typename Sampler>
  bool sms_gpu_active(const Sampler& s) const {
    static const bool off =
        getenv("QWENOX_SMS_GPU") && atoi(getenv("QWENOX_SMS_GPU")) == 0;
    if (off) return false;
    if (s.top_k <= 0 || s.top_k > 64 || s.top_k >= g_cfg.vocab) return false;
    if ((int)s.bias.size() > SMS_BIAS_CAP) return false;
    if (s.bias.size() > 1) {
      // The GPU bias phase applies entries in parallel; duplicate ids would
      // race instead of adding sequentially, so those requests stay dense.
      std::vector<int> ids;
      ids.reserve(s.bias.size());
      for (auto& b : s.bias) ids.push_back(b.first);
      std::sort(ids.begin(), ids.end());
      for (size_t i = 1; i < ids.size(); i++)
        if (ids[i] == ids[i - 1]) return false;
    }
    return true;
  }

  // Upload the round-shared correction lists (bias, then hist penalties).
  // False when the hist overflows the buffer: caller uses the dense path.
  template <typename Sampler>
  bool sms_round_shared(const Sampler& s) {
    if ((int)sms_out_ids.size() < SMS_ROWS * s.top_k) {
      sms_out_ids.resize((size_t)SMS_ROWS * s.top_k);
      sms_out_vals.resize((size_t)SMS_ROWS * s.top_k);
    }
    sm_nb = (int)s.bias.size();
    sm_nh = 0;
    const bool pen = (s.presence != 0.0f || s.frequency != 0.0f);
    if (pen) {
      if ((int)s.hist.size() > SMS_HIST_CAP) return false;
      sm_nh = (int)s.hist.size();
    }
    if (!sm_nb && !sm_nh) return true;
    h_sm_ci.clear();
    h_sm_cv.clear();
    for (auto& b : s.bias) {
      h_sm_ci.push_back(b.first);
      h_sm_cv.push_back(b.second);
    }
    if (sm_nb) {
      CK(hipMemcpyAsync(d_sm_bids, h_sm_ci.data(), sm_nb * 4,
                        hipMemcpyHostToDevice, g_str));
      CK(hipMemcpyAsync(d_sm_bvals, h_sm_cv.data(), sm_nb * 4,
                        hipMemcpyHostToDevice, g_str));
    }
    if (sm_nh) {
      for (auto& h : s.hist) {
        h_sm_ci.push_back(h.first);
        h_sm_cv.push_back(s.presence + s.frequency * (float)h.second);
      }
      CK(hipMemcpyAsync(d_sm_hids, h_sm_ci.data() + sm_nb, sm_nh * 4,
                        hipMemcpyHostToDevice, g_str));
      CK(hipMemcpyAsync(d_sm_hvals, h_sm_cv.data() + sm_nb, sm_nh * 4,
                        hipMemcpyHostToDevice, g_str));
    }
    return true;
  }

  // Per-row draft-prefix penalties. Walk row a (flat draft index base_off+a)
  // is only prepared when drafts[0..base_off+a) were all drawn before it, so
  // its penalty hist counts include exactly those. An id already in the base
  // hist gains only frequency * extra (presence came with the shared list);
  // a new id costs presence + frequency * count. The shared+prefix split
  // rounds differently from prepare()'s single merged subtraction by ~1 ulp;
  // the sampling path is distribution-gated, not bit-gated.
  // Returns whether any prefix list is nonempty.
  template <typename Sampler>
  bool sms_prefix_upload(const Sampler& s, const int* drafts, int base_off,
                         int rows) {
    const bool pen = (s.presence != 0.0f || s.frequency != 0.0f);
    if (!pen) return false;
    std::vector<int> pn(rows, 0);
    h_sm_pi.assign((size_t)rows * SMS_PCAP, 0);
    h_sm_pv.assign((size_t)rows * SMS_PCAP, 0.0f);
    bool any = false;
    for (int a = 0; a < rows; a++) {
      const int nd = std::min(base_off + a, SMS_PCAP);
      int used = 0;
      int* rids = h_sm_pi.data() + (size_t)a * SMS_PCAP;
      float* rvals = h_sm_pv.data() + (size_t)a * SMS_PCAP;
      for (int j = 0; j < nd; j++) {
        const int id = drafts[j];
        int slot = -1;
        for (int u = 0; u < used; u++)
          if (rids[u] == id) { slot = u; break; }
        if (slot < 0) {
          slot = used++;
          rids[slot] = id;
          rvals[slot] = s.hist.count(id) ? 0.0f : s.presence;
        }
        rvals[slot] += s.frequency;
      }
      pn[a] = used;
      if (used) any = true;
    }
    if (!any) return false;
    CK(hipMemcpyAsync(d_sm_pids, h_sm_pi.data(), (size_t)rows * SMS_PCAP * 4,
                      hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_sm_pvals, h_sm_pv.data(), (size_t)rows * SMS_PCAP * 4,
                      hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_sm_pn, pn.data(), rows * 4, hipMemcpyHostToDevice,
                      g_str));
    return true;
  }

  // Corrections + two-pass radix top-k + collect for `rows` logit rows
  // (stride floats each), then D2H and an exact top-k truncation ordered by
  // (value desc, id asc). On success out_ids/out_vals hold rows*k pairs.
  // Returns false on candidate overflow: the caller re-runs the dense path
  // (logits may already carry corrections, so the dense path must regenerate
  // or read a pristine copy -- hence the scratch bounce when corrections
  // exist).
  bool sms_topk_rows(float* x, int stride, int rows, int k, bool use_pfx,
                     int* out_ids, float* out_vals) {
    float* src = x;
    if (sm_nb > 0 || sm_nh > 0 || use_pfx) {
      if (!d_sm_scratch)
        DALLOC(&d_sm_scratch, (size_t)SMS_ROWS * g_cfg.vocab * 4);
      CK(hipMemcpyAsync(d_sm_scratch, x, (size_t)rows * stride * 4,
                        hipMemcpyDeviceToDevice, g_str));
      k_corr_apply<<<dim3(1, rows), 256, 0, g_str>>>(
          d_sm_scratch, stride, d_sm_bids, d_sm_bvals, sm_nb, d_sm_hids,
          d_sm_hvals, sm_nh, d_sm_pids, d_sm_pvals,
          use_pfx ? d_sm_pn : nullptr, SMS_PCAP);
      src = d_sm_scratch;
    }
    k_tk_init<<<rows, 32, 0, g_str>>>(d_sm_ctrl, k);
    CK(hipMemsetAsync(d_sm_hist, 0, (size_t)rows * 4096 * sizeof(unsigned),
                      g_str));
    k_tk_hist<0><<<dim3(8, rows), 256, 0, g_str>>>(src, stride, stride,
                                                   d_sm_ctrl, d_sm_hist);
    k_tk_scan<0><<<dim3(1, rows), 256, 0, g_str>>>(d_sm_hist, d_sm_ctrl,
                                                   d_sm_cand_n);
    CK(hipMemsetAsync(d_sm_hist, 0, (size_t)rows * 4096 * sizeof(unsigned),
                      g_str));
    k_tk_hist<1><<<dim3(8, rows), 256, 0, g_str>>>(src, stride, stride,
                                                   d_sm_ctrl, d_sm_hist);
    k_tk_scan<1><<<dim3(1, rows), 256, 0, g_str>>>(d_sm_hist, d_sm_ctrl,
                                                   d_sm_cand_n);
    k_tk_collect<<<dim3(8, rows), 256, 0, g_str>>>(
        src, stride, stride, d_sm_ctrl, d_sm_cand_ids, d_sm_cand_vals,
        d_sm_cand_n, SMS_TCAP);
    std::vector<int> cn(rows);
    CK(hipMemcpy(cn.data(), d_sm_cand_n, rows * 4, hipMemcpyDeviceToHost));
    for (int r = 0; r < rows; r++)
      if (cn[r] < k || cn[r] > SMS_TCAP) return false;
    CK(hipMemcpy(h_sm_ids.data(), d_sm_cand_ids, (size_t)rows * SMS_TCAP * 4,
                 hipMemcpyDeviceToHost));
    CK(hipMemcpy(h_sm_vals.data(), d_sm_cand_vals,
                 (size_t)rows * SMS_TCAP * 4, hipMemcpyDeviceToHost));
    std::vector<int> ord;
    for (int r = 0; r < rows; r++) {
      const int m = cn[r];
      const int* ids = h_sm_ids.data() + (size_t)r * SMS_TCAP;
      const float* vals = h_sm_vals.data() + (size_t)r * SMS_TCAP;
      ord.resize(m);
      for (int j = 0; j < m; j++) ord[j] = j;
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (vals[a] != vals[b]) return vals[a] > vals[b];
        return ids[a] < ids[b];
      });
      for (int j = 0; j < k; j++) {
        out_ids[(size_t)r * k + j] = ids[ord[j]];
        out_vals[(size_t)r * k + j] = vals[ord[j]];
      }
    }
    return true;
  }

  // prepare_sparse on candidate row `row` (draft: row 0 of the d_logits
  // pass). QWENOX_SMS_CHECK=1 also replays the dense prepare() on the raw
  // logits row and compares the two distributions (support + probabilities).
  template <typename Sampler>
  void sms_prep(Sampler& s, int row, bool draft,
                typename Sampler::SDist& out) {
    const int k = s.top_k;
    s.prepare_sparse(sms_out_ids.data() + (size_t)row * k,
                     sms_out_vals.data() + (size_t)row * k, k, out);
    static const bool chk = getenv("QWENOX_SMS_CHECK") != nullptr;
    if (!chk) return;
    static long n = 0, bad = 0;
    static double maxd = 0, maxout = 0;
    std::vector<float> v(g_cfg.vocab);
    const float* src = draft ? d_logits : d_logits_b + (size_t)row * g_cfg.vocab;
    CK(hipMemcpy(v.data(), src, (size_t)g_cfg.vocab * 4, hipMemcpyDeviceToHost));
    Sampler c = s;
    c.prepare(v);
    double d = 0, in = 0;
    for (size_t j = 0; j < out.ids.size(); j++) {
      d = std::max(d, (double)fabsf(v[out.ids[j]] - out.pr[j]));
      in += v[out.ids[j]];
    }
    int dn = 0;
    for (int i = 0; i < g_cfg.vocab; i++) dn += v[i] > 0.0f;
    const double o = fabs(1.0 - in);
    n++;
    maxd = std::max(maxd, d);
    maxout = std::max(maxout, o);
    const bool b = d > 1e-5 || o > 1e-5 || dn != (int)out.ids.size();
    if (b) {
      bad++;
      fprintf(stderr, "[sms-check] MISMATCH %s row %d: maxdiff %.3g outside %.3g support dense %d sparse %zu\n",
              draft ? "draft" : "verify", row, d, o, dn, out.ids.size());
    }
    if (b || n % 32 == 0)
      fprintf(stderr, "[sms-check] n=%ld bad=%ld maxdiff=%.3g maxoutside=%.3g\n", n, bad, maxd, maxout);
  }

  std::vector<int> spec_loop_ngram(int cur, int gen, int gamma,
      const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, gamma), maxbatch - 1});
    int accepted = 0, fallback = 0;
    double verify_ms = 0, replay_ms = 0;
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto step = [&](int tok) {
      int nx = forward(tok);
      if (!ple_on) chist.push_back(tok);
      ple_ringpos = pos % 9;  // graph advances the device scalar, not the host mirror
      return nx;
    };
    auto verify = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    while ((int)out.size() < gen && pos < maxctx) {
      yield_point();  // round top: only persistent slot state is live
      if (chist.size() != (size_t)pos)
        throw std::runtime_error("ngram: token history and trunk position disagree");
      ngram.observe(chist);
      // Leave one slot for the target-model continuation. Never verify past
      // the generation budget or context boundary, including the final round.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      std::vector<int> draft = want > 0 ? ngram.draft(chist, cur, want) : std::vector<int>{};
      if (draft.empty()) {
        int next = step(cur);
        ++fallback;
        if (emit && !emit(next)) break;
        out.push_back(next);
        cur = next;
        continue;
      }
      const int base = pos;
      const size_t history_size = chist.size();
      ngram_ckpt_save();
      std::vector<int> tokens{cur};
      tokens.insert(tokens.end(), draft.begin(), draft.end());
      auto t = std::chrono::steady_clock::now();
      verify(tokens, base);
      std::vector<int> targets(tokens.size());
      batch_argmax((int)tokens.size(), targets.data());
      verify_ms += elapsed(t);
      int a = 0;
      while (a < (int)draft.size() && draft[a] == targets[a]) ++a;
      int keep = a;
      bool stop = false;
      for (int j = 0; j < a; ++j) {
        if (emit && !emit(draft[j])) { keep = j; stop = true; break; }
        out.push_back(draft[j]);
      }
      if (!stop) {
        if (emit && !emit(targets[a])) stop = true;
        else out.push_back(targets[a]);
      }
      if (keep != (int)draft.size()) {
        // Restore every overwritten position-mod state, then replay exactly
        // [cur, accepted drafts]. Causal KV/indexer entries are overwritten
        // before they are visible. Replaying a batch preserves verify math.
        t = std::chrono::steady_clock::now();
        ngram_ckpt_restore();
        chist.resize(history_size);
        pos = base;
        ple_ringpos = base % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
        tokens.resize(keep + 1);
        verify(tokens, base);
        CK(hipStreamSynchronize(g_str));
        replay_ms += elapsed(t);
        ++last_spec_rollbacks;
      }
      cur = targets[keep];
      accepted += keep;
      ++last_spec_rounds;
      last_spec_proposed += draft.size();
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = accepted + last_spec_rounds;
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    last_spec_commit = (int)out.size();
    fprintf(stderr, "%s ngram: tokens=%d rounds=%d accepted=%d proposed=%d "
                    "fallback=%d rollbacks=%d verify=%.1fms replay=%.1fms\n",
            log_ts(), last_spec_commit, last_spec_rounds, accepted,
            last_spec_proposed, fallback, last_spec_rollbacks, verify_ms,
            replay_ms);
    return out;
  }

  void ngram_ckpt_save() {
    CK(hipMemcpyAsync(d_ckpt,d_S,ckpt_S,hipMemcpyDeviceToDevice,g_str));
    CK(hipMemcpyAsync(d_ckpt+ckpt_S/4,d_convst,ckpt_convst,hipMemcpyDeviceToDevice,g_str));
    float* p=d_ckpt+(ckpt_S+ckpt_convst)/4; CK(hipMemcpyAsync(p,d_ring,ckpt_ring,hipMemcpyDeviceToDevice,g_str)); p+=ckpt_ring/4;
    if(ckpt_iraw) CK(hipMemcpyAsync(p,d_iraw,ckpt_iraw,hipMemcpyDeviceToDevice,g_str));
  }
  void ngram_ckpt_restore() {
    CK(hipMemcpyAsync(d_S,d_ckpt,ckpt_S,hipMemcpyDeviceToDevice,g_str));
    CK(hipMemcpyAsync(d_convst,d_ckpt+ckpt_S/4,ckpt_convst,hipMemcpyDeviceToDevice,g_str));
    const float* p=d_ckpt+(ckpt_S+ckpt_convst)/4; CK(hipMemcpyAsync(d_ring,p,ckpt_ring,hipMemcpyDeviceToDevice,g_str)); p+=ckpt_ring/4;
    if(ckpt_iraw) CK(hipMemcpyAsync(d_iraw,p,ckpt_iraw,hipMemcpyDeviceToDevice,g_str));
  }

  // Adaptive MTP draft length, active when a spec_loop* is entered with
  // gamma == 0 (API: sampling with QWENOX_SPEC_GAMMA unset; offline: --gamma 0).
  // Tracks an EMA of the per-depth conditional acceptance p_j = P(draft j
  // accepted | drafts 0..j-1 accepted) and after each round picks γ in
  // [kmin, gmax] maximizing E[commit]/cost, cost(γ) = 1 + r·γ.
  // Two states: p is kept separately for rounds that follow a full accept (F)
  // and rounds that follow a reject (R) — 09-29 γ7 traces, sampling real
  // text: F .93/.86/.83 vs R .77/.71/.73. A deep γ makes full accepts rare,
  // so the effective acceptance falls with γ; a single-state estimate misses
  // that and drifts deep. With A_s = Π_{j<γ} p_s,j and E_s the per-state
  // expected commit, the stationary P(F) = A_R / (1 − A_F + A_R) and
  // E(γ) = π_F·E_F + (1 − π_F)·E_R (predicts fixed γ3 commit/round 3.03 vs
  // measured 2.98; single-state 2.82). Round wall ms ≈ 42 + 10.7·γ → r ≈ 0.25;
  // default 0.3 absorbs the model's remaining optimism for deep γ. kmin = 3:
  // on-GPU sampling runs stuck at γ2 were ~20% slower than γ3 (the replay
  // can't see that). tools/gamma_sim.py MODEL=markov: real text = fixed γ3,
  // code/repetitive text ≈ best fixed. On GPU (tools/gamma_sample_check.sh,
  // 18 samples): real text 1.004× γ3 (single-state 0.940×, fixed γ4 0.953×),
  // code/repetitive 1.040× γ3 (fixed γ4 1.032×). Depths >= the current γ
  // are unobserved and extrapolated from the deepest probed one. gmax = 7:
  // γ=8 means P=9 verify rows, past the P<=8 fast paths (cliff). Sampling
  // stays exact: γ is fixed before the round draws anything. Greedy ids can
  // differ from a fixed γ only through fp near-tie flips (as any γ change).
  // A gamma > 0 is used as-is (clamped to 8); adapt stays off.
  // Spec verify commit (10-12): install the bound slot's GDN S after the first
  // n rows of the last verify round (n = commit depth + 1, or all P rows on a
  // full accept) by replaying the saved rows (d_grow at vsnap_row0_) through
  // gdn_rec_body. The verify left S at the round start; one launch covers all
  // GDN layers.
  void gdn_commit(int n) {
    if (n < 1) return;
    const int ngdn = g_cfg.layers - g_cfg.layers / 4;
    k_gdn_commit<<<dim3(48, (unsigned)ngdn), 512, 0, g_str>>>(
        d_grow + (size_t)vsnap_row0_ * ngdn * kGdnRowF, (size_t)ngdn * kGdnRowF, d_S, n);
  }

  // QWENOX_SPEC_CONF=θ (sampled MTP rounds; default 0.9, 0 = GammaCtl):
  // per-round draft length from the draft's own confidence. Draft step j is
  // always verified; step j+1 is drafted only while the top-1 probability of
  // step j's prepared draft distribution is >= θ, up to kSpecConfMax steps
  // (OccGamma still caps rows under concurrency). The rule reads only q, so a
  // seeded request stays reproducible and the target distribution is exact.
  // Windows 10-16 (round cost T = 35 + 7.9·γ ms, early stops cost nothing):
  // offline replay of a forced-γ7 SPEC_DUMP picked θ 0.9 (stopping before
  // verifying a low-confidence draft and reshaping the draft temperature
  // were worse); serve A/B vs adaptive γ, seeds 1-3, 512 tok — code 0K
  // 57.5 -> 63.2, 64K 51.3 -> 63.4; prose 0K 40.2 -> 42.2, 64K 36.4 -> 39.3.
  static constexpr int kSpecConfMax = 7;
  static float spec_conf() {
    static const float v = [] {
      const char* e = getenv("QWENOX_SPEC_CONF");
      return e ? std::max(0.f, std::min(1.f, (float)atof(e))) : 0.9f;
    }();
    return v;
  }
  // True when drafting should stop after step j (of g planned). sparse = the
  // prepared sparse probabilities, else dense = prepared full-vocab probs.
  static bool spec_conf_stop(int j, int g, const std::vector<float>* sparse,
                             const std::vector<float>& dense) {
    const float th = spec_conf();
    if (th <= 0.f || j + 1 >= g) return false;
    float m = 0.f;
    if (sparse)
      for (float v : *sparse) m = std::max(m, v);
    else
      for (float v : dense) m = std::max(m, v);
    return m < th;
  }

  // QWENOX_SPEC_DUMP=path (observation only): one JSON line per sampled MTP
  // round of chain-sample, for offline acceptance studies (draft-length stop
  // rules, draft temperature). q = q position of the round, g drafts, a
  // accepted; qt = top-1 prob of each draft distribution; per evaluated row
  // (0..min(a, g-1)): x = draft token, qx/px = its draft/target probability,
  // ov = sum min(p, q) (the row's expected acceptance), q/p = the prepared
  // sparse distributions as [id, prob] pairs.
  template <typename SD>
  void spec_dump_round(int q, int g, int a, const std::vector<int>& drafts,
                       const std::vector<SD>& q_sd, const std::vector<SD>& p_sd) {
    static FILE* f = [] {
      const char* e = getenv("QWENOX_SPEC_DUMP");
      return e && *e ? fopen(e, "a") : nullptr;
    }();
    if (!f || g < 1 || (int)q_sd.size() < g) return;
    auto prob = [](const SD& d, int id) {
      for (size_t j = 0; j < d.ids.size(); j++)
        if (d.ids[j] == id) return d.pr[j];
      return 0.0f;
    };
    auto pairs = [&](const SD& d) {
      fputc('[', f);
      for (size_t j = 0; j < d.ids.size(); j++)
        fprintf(f, "%s[%d,%.5g]", j ? "," : "", d.ids[j], d.pr[j]);
      fputc(']', f);
    };
    fprintf(f, "{\"q\":%d,\"g\":%d,\"a\":%d,\"qt\":[", q, g, a);
    for (int j = 0; j < g; j++) {
      float m = 0.0f;
      for (float v : q_sd[j].pr) m = std::max(m, v);
      fprintf(f, "%s%.5g", j ? "," : "", m);
    }
    fprintf(f, "],\"rows\":[");
    const int nr = std::min(a + 1, g);
    for (int j = 0; j < nr && j < (int)p_sd.size(); j++) {
      const SD& qd = q_sd[j];
      const SD& pd = p_sd[j];
      double ov = 0;
      for (size_t u = 0; u < qd.ids.size(); u++)
        ov += std::min(qd.pr[u], prob(pd, qd.ids[u]));
      fprintf(f, "%s{\"x\":%d,\"qx\":%.5g,\"px\":%.5g,\"ov\":%.5g,\"q\":", j ? "," : "",
              drafts[j], prob(qd, drafts[j]), prob(pd, drafts[j]), ov);
      pairs(qd);
      fprintf(f, ",\"p\":");
      pairs(pd);
      fputc('}', f);
    }
    fprintf(f, "]}\n");
    fflush(f);
  }

  struct GammaCtl {
    static constexpr float kPrior = 0.8f, kPriorN = 4.f;
    bool adapt = false;
    int cur = 4, kmin = 3, gmax = 7, switches = 0, full = 0;
    float r = 0.3f, hyst = 0.02f, alpha = 0.05f;
    float p[2][8], n[2][8];  // [0] after a reject, [1] after a full accept
    int hist[9] = {0};
    static float env_f(const char* k, float def) {
      const char* e = getenv(k);
      return e ? (float)atof(e) : def;
    }
    GammaCtl(int gamma, bool sample) {
      for (int s = 0; s < 2; s++)
        for (int j = 0; j < 8; j++) p[s][j] = kPrior, n[s][j] = kPriorN;
      if (gamma > 0) { cur = std::min(8, gamma); return; }
      adapt = true;
      cur = sample ? 3 : 4;  // previous fixed auto defaults
      static const float env_r = env_f("QWENOX_SPEC_ADAPT_R", 0.3f);
      static const float env_alpha = env_f("QWENOX_SPEC_ADAPT_ALPHA", 0.05f);
      static const float env_hyst = env_f("QWENOX_SPEC_ADAPT_HYST", 0.02f);
      static const int env_min = (int)env_f("QWENOX_SPEC_ADAPT_MIN", 3.f);
      static const int env_max = (int)env_f("QWENOX_SPEC_ADAPT_MAX", 7.f);
      r = std::max(0.01f, env_r);
      alpha = std::max(0.005f, std::min(1.f, env_alpha));
      hyst = std::max(0.f, env_hyst);
      kmin = std::max(1, std::min(8, env_min));
      gmax = std::max(kmin, std::min(8, env_max));
      cur = std::max(kmin, std::min(cur, gmax));
    }
    double score(int G) const {
      double e[2], a[2];
      for (int s = 0; s < 2; s++) {
        double pr = 1.0;
        e[s] = 1.0;
        for (int k = 0; k < G; k++) {
          pr *= p[s][std::min(k, cur - 1)];
          e[s] += pr;
        }
        a[s] = pr;
      }
      double pf = a[0] / std::max(1e-9, 1.0 - a[1] + a[0]);
      return (pf * e[1] + (1.0 - pf) * e[0]) / (1.0 + r * G);
    }
    // QWENOX_SPEC_TRACE=1: record every round (γ, accepted, wall ms since the
    // previous update) and print it in report() — offline controller replay
    // (tools/gamma_sim.py). Observation only; fixed and adaptive γ alike.
    static bool trace_on() {
      static const bool on = [] {
        const char* e = getenv("QWENOX_SPEC_TRACE");
        return e && e[0] == '1';
      }();
      return on;
    }
    std::vector<std::array<float, 3>> trace;
    std::chrono::steady_clock::time_point t_last = std::chrono::steady_clock::now();
    // g = drafts proposed this round, a = accepted prefix length.
    void update(int g, int a) {
      if (trace_on()) {
        auto now = std::chrono::steady_clock::now();
        trace.push_back({(float)g, (float)a,
                         (float)std::chrono::duration<double, std::milli>(now - t_last).count()});
        t_last = now;
      }
      hist[std::min(8, g)]++;
      const int s = full;
      full = a >= g;
      if (!adapt || g < cur) return;  // tail round cut short by gen/ctx
      for (int j = 0; j <= a && j < g; j++) {
        n[s][j] += 1.f;
        p[s][j] += std::max(alpha, 1.f / n[s][j]) * ((j < a ? 1.f : 0.f) - p[s][j]);
      }
      int best = cur;
      double sb = score(cur);
      for (int G = kmin; G <= gmax; G++) {
        double s = score(G);
        if (s > sb * (1.0 + hyst) || (best != cur && s > sb)) best = G, sb = s;
      }
      if (best == cur) return;
      for (int t = 0; t < 2; t++)
        for (int j = cur; j < best; j++) p[t][j] = p[t][cur - 1], n[t][j] = kPriorN;
      cur = best;
      switches++;
    }
    void report(const char* tag) const {
      if (trace_on()) {
        fprintf(stderr, "%s gamma-trace:", tag);
        for (auto& t : trace) fprintf(stderr, " %d,%d,%.1f", (int)t[0], (int)t[1], t[2]);
        fprintf(stderr, "\n");
      }
      if (!adapt) return;
      if (strstr(tag, "sample") && spec_conf() > 0) {
        // QWENOX_SPEC_CONF drove the draft lengths: hist is per-round γ.
        fprintf(stderr, "%s spec-conf θ=%.2f: rounds per γ", tag, spec_conf());
        for (int G = 1; G <= 8; G++)
          if (hist[G]) fprintf(stderr, " %d:%d", G, hist[G]);
        fprintf(stderr, "\n");
        return;
      }
      fprintf(stderr, "%s gamma-adapt: rounds per γ", tag);
      for (int G = 1; G <= 8; G++)
        if (hist[G]) fprintf(stderr, " %d:%d", G, hist[G]);
      fprintf(stderr, " | switches=%d final γ=%d r=%.2f p(after full):", switches, cur, r);
      for (int j = 0; j < cur; j++) fprintf(stderr, " %.2f", p[1][j]);
      fprintf(stderr, " p(after reject):");
      for (int j = 0; j < cur; j++) fprintf(stderr, " %.2f", p[0][j]);
      fprintf(stderr, "\n");
    }
  };

  // D2 阶段 3（10-12）：按会合路数选 γ。B 路同时 verify 时一轮的成本近似
  //   T(B, γ) = A + B·(SEQ + ROW·(γ+1)) + DRAFT(B)·max(γ, 1)
  // 常数按服务端实测拟合（conc_bench 代码题，强制 γ=0/1/2 与旧逻辑 γ=4，
  // 每轮墙钟 = B·E(γ)/吞吐：C8 ≈ 135 + 81γ ms，C4 ≈ 75 + 43γ ms；合批 draft
  // 每步的增量测不出来，DRAFT 取 0），
  // 所以每多投机一步的边际成本随 B 增大，而 GammaCtl 的单序列最优是按 B = 1
  // 算的。实测 C8：γ0 59.3 / γ1 69.9 / γ2 71.1 / γ4 65.7 tok/s——纯合批 AR
  // 最差（每路私有状态成本合不掉），γ1/γ2 近乎打平。这里另记一份逐深度接受率（GammaCtl 只在自适应模式、且只在未被截断
  // 的轮次上学习），在 [0, base] 里选 E(γ)/T(B, γ) 最大的 γ。B ≤ 1 时原样
  // 返回 base：单用户行为不变。
  // γ = 0 仍跑第 1 步 MTP draft（它写入位置 q 的 MTP KV，即轮首不变量），
  // 但 verify 只送 cur 一行，等于一次合批 AR。
  // greedy 输出与 γ 无关（d2_s1b_gamma），用实时会合人数。采样请求的 γ 会改变
  // RNG 消耗：客户端显式给了 seed（协议 SEEDED）的请求用确定的 B = nslots，
  // 只依赖本请求自己的接受历史，同 seed 在并发下仍可复现（代价：单用户时也按
  // 满并发选 γ，实测 C1 慢约 11%）；未给 seed 的请求输出本来就随机，用实时人数。
  // QWENOX_OCC_SAMPLE：auto【默认，按 SEEDED】/ fixed / live / off。
  // 开关：QWENOX_OCC=0 关闭（退回旧的行预算截断）；QWENOX_OCC_FORCE=G 校准 /
  // 对拍用，任何 B 下都把 γ 钉在 min(G, 预算)；成本常数 QWENOX_OCC_A / _SEQ /
  // _ROW / _DRAFT（合批 draft 每步）/ _DRAFT1（串行 draft 每路每步），单位 ms。
  struct OccGamma {
    static float env_f(const char* k, float def) {
      const char* e = getenv(k);
      return e && *e ? (float)atof(e) : def;
    }
    static int mode_sample() {  // 0 off, 1 fixed (nslots), 2 live, 3 auto
      static const int m = [] {
        const char* e = getenv("QWENOX_OCC_SAMPLE");
        if (!e || !*e || !strcmp(e, "auto")) return 3;
        if (!strcmp(e, "fixed")) return 1;
        if (!strcmp(e, "live")) return 2;
        return 0;
      }();
      return m;
    }
    bool on = true, sample = false, draft_batched = true, fixed = false;
    int force = -1, fixed_b = 1, last = -1;
    float A = 10.f, SEQ = 5.5f, ROW = 10.f, DRAFT = 0.f, DRAFT1 = 3.f, hyst = 0.01f;
    float p[8], n[8];
    long rounds = 0, hist[9] = {0};
    OccGamma(bool sample_, int nslots_, bool draft_batched_, bool seeded = false)
        : sample(sample_), draft_batched(draft_batched_) {
      static const bool off = env_zero("QWENOX_OCC");
      static const int env_force = [] {
        const char* e = getenv("QWENOX_OCC_FORCE");
        return e && *e ? std::max(0, std::min(8, atoi(e))) : -1;
      }();
      // QWENOX_DENSE_WMMA (10-12 refit): a verify row costs ~5 ms instead of ~10
      // (VSEG: T ≈ 26 + 5.1·R ms for R = 6..40) while the per-round fixed part
      // (host enqueue, worker, PLE) stays, so the optimum γ moves deeper.
      // Serving sweep, forced γ1/2/3/4 (tok/s): C2 greedy 62.4/67.8/70.2/68.6,
      // C2 sampled 57.6/63.7/63.9/62.0, C8 greedy 92.8/96.6/96.8/92.0 — the
      // fp32-era constants picked γ≈2 at C2. Batched draft ≈ 3 ms per step
      // (TURN_PROF draft-exec 8 ms per C2 round).
      static const bool dw = dense_wmma_on();
      static const float eA = env_f("QWENOX_OCC_A", dw ? 40.f : 10.f),
                         eS = env_f("QWENOX_OCC_SEQ", dw ? 2.f : 5.5f),
                         eR = env_f("QWENOX_OCC_ROW", dw ? 5.f : 10.f),
                         eD = env_f("QWENOX_OCC_DRAFT", dw ? 3.f : 0.f),
                         eD1 = env_f("QWENOX_OCC_DRAFT1", 3.f),
                         eH = env_f("QWENOX_OCC_HYST", 0.01f);
      on = !off && (!sample || mode_sample() != 0);
      fixed = sample && (mode_sample() == 1 || (mode_sample() == 3 && seeded));
      force = env_force;
      fixed_b = std::max(1, nslots_);
      A = eA, SEQ = eS, ROW = std::max(0.01f, eR), DRAFT = eD, DRAFT1 = eD1;
      hyst = std::max(0.f, eH);
      // Prior 0.85 (measured conditional acceptance on code ≈ 0.90 / 0.82 /
      // 0.82 / 0.81 by depth): slightly optimistic so a deeper γ gets tried —
      // a γ1 round never observes depth 2, a low prior would lock γ1 in.
      for (int j = 0; j < 8; j++) p[j] = 0.85f, n[j] = 4.f;
    }
    // B the policy sees: live members, or nslots for a fixed (seeded sampled)
    // request (QWENOX_OCC=0 / QWENOX_OCC_SAMPLE=off: the old live-member clamp).
    int eff_b(int live) const {
      if (!on || !fixed) return live;
      return live > 0 ? fixed_b : 1;  // live 0 = not in the rendezvous at all
    }
    double cost(int B, int G) const {
      const double dr = draft_batched ? DRAFT : DRAFT1 * B;
      return A + B * (SEQ + ROW * (G + 1)) + dr * std::max(G, 1);
    }
    double gain(int G) const {
      double e = 1.0, pr = 1.0;
      for (int k = 0; k < G; k++) e += (pr *= p[k]);
      return e;
    }
    // g: γ after the gen/ctx bounds; live: rendezvous members (1 = alone);
    // rows: the batched verify's row budget (kVSnapRows). Returns γ ≥ 0.
    int pick(int g, int live, int rows) {
      const int B = eff_b(live);
      // Shared row budget of the batched verify (the executor re-checks the
      // hard cap per segment and drops overflow to the serial path).
      if (B > 1) g = std::min(g, std::max(1, rows / B - 1));
      if (force >= 0) return std::min(g, force);
      if (!on || B <= 1) return g;
      int best = g;
      double sb = gain(g) / cost(B, g);
      for (int G = 0; G < g; G++) {
        const double s = gain(G) / cost(B, G);
        if (s > sb) best = G, sb = s;
      }
      // Hysteresis: keep the previous choice unless the new one wins clearly.
      if (last >= 0 && last <= g && best != last &&
          sb <= (1.0 + hyst) * gain(last) / cost(B, last))
        best = last;
      last = best;
      rounds++;
      // Probe one step deeper every 8th capped round: depth best+1 is never
      // observed otherwise, and a noisy low estimate there would lock the
      // policy in. Depends only on this request's own round count.
      const int out = (best < g && (rounds & 7) == 0) ? best + 1 : best;
      hist[out]++;
      return out;
    }
    // g = drafts verified this round, a = accepted prefix length.
    void update(int g, int a) {
      for (int j = 0; j < g && j <= a; j++) {
        n[j] += 1.f;
        p[j] += std::max(0.02f, 1.f / n[j]) * ((j < a ? 1.f : 0.f) - p[j]);
      }
    }
    void report(const char* tag) const {
      if (!rounds) return;
      fprintf(stderr, "%s occ-gamma: %ld rounds at B>1, γ", tag, rounds);
      for (int G = 0; G <= 8; G++)
        if (hist[G]) fprintf(stderr, " %d:%ld", G, hist[G]);
      fprintf(stderr, " | p:");
      for (int j = 0; j < 6; j++) fprintf(stderr, " %.2f", p[j]);
      fprintf(stderr, "\n");
    }
  };

  // Speculative decode driver. cur = emitted token for position pos (trunk
  // state processed through pos-1, MTP KV covers 0..pos-2). Greedy accept:
  // draft j committed iff it equals the trunk argmax at its verify row; the
  // correction (or full-accept continuation) comes from the trunk.
  // emit (optional): called with each committed token as it lands; returning
  // false aborts the loop early (serve: eos/cancel/streaming).
  std::vector<int> spec_loop(int cur, int gen, int gamma,
                             const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(gen + 1);
    GammaCtl gctl(gamma, false);
    std::vector<int> drafts(8), v(9);
    // D2-S3c: submit this loop's draft bursts to the rendezvous pool when
    // wired and batch-safe (the serve call site counts membership with the
    // same predicate -- see dr_worker_ok).
    const bool dr_join = dr_worker_ok();
    OccGamma occ(false, nslots, dr_join);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      yield_point();  // round top: mtp_pos == pos-1, d_R == h_{pos-1}
      int q = pos - 1;  // trunk processed through q
      if (d_iraw) vg_dump_buf("irT", 2, d_iraw, 512 * 4);  // P3-B bisect
      int g = std::min(gctl.cur, gen - (int)out.size());
      // D2-S2a row budget + D2 阶段 3 occupancy γ (round-top snapshot of the
      // rendezvous membership; may return 0 = batched AR round).
      g = occ.pick(g, vseg_active_fn ? vseg_active_fn() : 1, (int)kVSnapRows);
      const int gd = std::max(1, g);  // draft steps: γ = 0 still writes MTP KV at q
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      if (d_iraw) vg_dump_buf("irC", 2, d_iraw, 512 * 4);  // P3-B bisect
      auto t0 = std::chrono::steady_clock::now();
      // D2-S3c: hand the whole draft burst to the rendezvous pool (one
      // batched cross-slot pass per step, bit-identical to the serial burst
      // below); false = this round keeps the serial burst.
      const bool dr_done =
          dr_join && dr_pool_fn(*this, DrSeg{bound_, gd, cur, drafts.data()});
      if (!dr_done) {
      // draft 1 used the committed (h_q, cur): its indexer ring rows are
      // truth. Later drafts pollute the ring; save/restore around them.
      if (draft_desync) {
        mtp_step(d_R, cur, nullptr, true, nullptr, d_argmax_arr);
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        for (int j = 1; j < gd; j++)
          mtp_step(d_mR, 0, nullptr, true, d_argmax_arr + j - 1,
                   d_argmax_arr + j);
        CK(hipMemcpy(drafts.data(), d_argmax_arr, (size_t)gd * sizeof(int),
                     hipMemcpyDeviceToHost));
      } else {
        drafts[0] = mtp_step(d_R, cur);
        CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        for (int j = 1; j < gd; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
      }
      }
      auto t1 = std::chrono::steady_clock::now();
      if (d_iraw) vg_dump_buf("irD", 2, d_iraw, 512 * 4);  // P3-B bisect
      std::vector<int> vt;
      vt.reserve(g + 1);
      vt.push_back(cur);
      for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
      spec_snap = true;  // capture per-row rollback snapshots during verify
      // D2-S2a: the pool's batched verify repacks this slot's snap window at
      // its segment row offset; serial paths (and slot reuse) start at row 0.
      vsnap_row0_ = 0;
      if (!verify_graph_round_try(vt, q + 1, v.data())) {
        // D2-S1a: single-segment verify_segments (bit-identical); the env
        // opt-out and any misfit (multi-slot, oversize) keep the old path.
        static const bool no_vseg =
            getenv("QWENOX_NO_VSEG") && !env_zero("QWENOX_NO_VSEG");
        const VSeg vseg{bound_, &vt, q + 1};
        // D2-S2a: rendezvous verify — the pool's executor already ran this
        // segment (rollback snapshots included) when it returns true; false
        // (dropped / pool idle) -> run the serial verify here.
        if (vseg_pool_fn && vseg_pool_fn(*this, vseg, v.data(), nullptr))
          spec_snap = false;
        else if (no_vseg || !verify_segments(&vseg, 1, v.data())) {
          prefill_batch(vt, q + 1, false, true);
          batch_argmax(g + 1, v.data());
        }
        spec_snap = false;
      } else
        spec_snap = false;
      if (getenv("QWENOX_VGDUMP")) {  // P3-B bisect: whole-logits round hash
        CK(hipStreamSynchronize(g_str));
        const size_t nb = (size_t)(g + 1) * g_cfg.vocab * 4;
        std::vector<uint8_t> lh(nb);
        CK(hipMemcpy(lh.data(), d_logits_b, nb, hipMemcpyDeviceToHost));
        uint64_t fnv = 1469598103934665603ULL;
        for (uint8_t b : lh) { fnv ^= b; fnv *= 1099511628211ULL; }
        fprintf(stderr, "[vgl] base=%d logithash=%016llx v=%d,%d,%d,%d\n",
                q + 1, (unsigned long long)fnv, v[0], v[1], v[2], v[3]);
      }
      if (!spec_force.empty())  // v[j] predicts out[out.size() + j]
        for (int j = 0; j <= g; j++)
          if (out.size() + j < spec_force.size()) v[j] = spec_force[out.size() + j];
      if (fork_dump) {
        for (int j = 0; j <= g; j++) {
          int apos = q + 1 + j;
          if (apos < fork_dump - 4 || apos > fork_dump + 4) continue;
          std::vector<float> L(g_cfg.vocab);
          CK(hipMemcpy(L.data(), d_logits_b + (size_t)j * g_cfg.vocab,
                       (size_t)g_cfg.vocab * 4, hipMemcpyDeviceToHost));
          int t1 = 0, t2 = 1;
          for (int i = 1; i < g_cfg.vocab; i++)
            if (L[i] > L[t1]) { t2 = t1; t1 = i; }
            else if (t2 == t1 || L[i] > L[t2]) t2 = i;
          fprintf(stderr,
                  "[fork] pos=%d row=%d top1=%d %.4f top2=%d %.4f margin=%.6f\n",
                  apos, j, t1, L[t1], t2, L[t2], L[t1] - L[t2]);
        }
      }
      auto t2 = std::chrono::steady_clock::now();
      int a = 0;
      while (a < g && drafts[a] == v[a]) a++;
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
      if (g > 0) gctl.update(g, a);  // γ = 0 rounds carry no draft evidence
      occ.update(g, a);
      rounds++;
      live_spec_rounds = rounds;
      live_spec_proposed += g;
      live_spec_commits = (int)out.size();  // pre-emit; one round of lag is fine
      // emit() streams each committed token as it lands; returning false
      // stops the round and leaves THIS token out of the trunk state (eos,
      // or cancel after it was streamed — the next turn re-ingests it via
      // ordinary KV-reuse continuation). d = commit depth: keep
      // drafts[0..d-1]; keep_v = also keep the correction/continuation v[a].
      int d = a;
      bool stop = false, keep_v = true;
      if (emit) {
        for (int j = 0; j < a; j++)
          if (!emit(drafts[j])) { d = j; stop = true; keep_v = false; break; }
        if (!stop && !emit(v[a])) { stop = true; keep_v = false; }
      }
      for (int j = 0; j < d; j++) out.push_back(drafts[j]);
      if (keep_v) out.push_back(v[a]);
      auto t3 = std::chrono::steady_clock::now();
      if (!stop && d == g) {
        gdn_commit(g + 1);  // GDN S: replay all rows (verify left it read-only)
        cur = v[g];  // full accept; state at q+1+g
      } else {
        rollbacks++;
        // Install the state at commit depth d from the verify snapshots —
        // no re-prefill. KV pool entries and pooled indexer keys for the
        // accepted positions were written causally by the verify pass (and
        // rejected positions are rewritten before they can be read), so only
        // the position-mod ring slots polluted by rejected rows are restored
        // from the checkpoint. d_Rb rows 0..d hold the accepted tokens'
        // streams for the MTP ingest below.
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        gdn_commit(d + 1);  // S after the committed rows (replay)
        CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)(vsnap_row0_ + d) * ngdn * 3 * 10240,
                          (size_t)ngdn * 3 * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (ckpt_iraw)
          k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
              d_iproj_snap + (size_t)vsnap_row0_ * 128,
              d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
              d_iraw, q + 1, q + d + 1, kVSnapRows);
        const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
        for (int p = q + d + 2; p <= q + g + 1; p++)
          CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                            ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        chist.resize(chist_sz + d + 1);
        pos = q + d + 2;
        ple_ringpos = pos % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                          g_str));
        // vsnap_row0_ = this slot's d_Rb row base (D2-S2a rendezvous packing;
        // 0 on the serial path).
        restore_dR_row(d, vsnap_row0_);
        cur = v[d];
      }
      auto t4 = std::chrono::steady_clock::now();
      // Rebuild MTP KV over committed positions q+1..q+d with trunk taps
      // (d_Rb rows of the state-advancing pass) and committed tokens.
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4, hipMemcpyDeviceToDevice,
                        g_str));
      if (d > 0) {
        std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
        mtp_ingest_b(spec_taps(d, vsnap_row0_), mtoks, d, q + 1);
      } else {
        mtp_pos = q + 1;
        CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      }
      if (stop) break;
      auto t5 = std::chrono::steady_clock::now();
      t_draft += ms(t0, t1);
      t_verify += ms(t1, t2);
      t_roll += ms(t3, t4);
      t_ingest += ms(t4, t5);
      if (d_iraw) vg_dump_buf("irE", 2, d_iraw, 512 * 4);  // P3-B bisect
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    if (!bench_quiet) {
      fprintf(stderr,
              "\nspec: %d tokens in %.2f s = %.1f tok/s | rounds=%d commit/round=%.2f "
              "rollbacks=%d\n",
              (int)out.size(), sec, out.size() / sec, rounds,
              (double)out.size() / rounds, rollbacks);
      fprintf(stderr, "spec depth acc:");
      for (int j = 0; j < 8 && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr,
              "\nspec time: draft=%.0fms verify=%.0fms rollback+reprefill=%.0fms "
              "ingest=%.0fms\n",
              t_draft, t_verify, t_roll, t_ingest);
    }
    gctl.report("spec");
    occ.report("spec");
    last_spec_rounds = rounds;
    last_spec_commit = (int)out.size();
    last_spec_proposed = live_spec_proposed;
    last_spec_rollbacks = rollbacks;
    return out;
  }

  // Chain speculative driver (llama.cpp common/speculative.cpp style): every
  // round asks the ngram table first; an empty draft hands the round to MTP.
  // Both drafters stay live — ngram observes every round, and ngram rounds
  // patch the MTP KV over the committed positions — so either can draft the
  // next round. Round-start invariant (same as spec_loop): mtp_pos = pos-1,
  // MTP KV covers 0..pos-2, d_R = h_{pos-1} pending pairing with cur.
  std::vector<int> spec_loop_chain(int cur, int gen, int gamma,
      const std::function<bool(int)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks =
        last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, ngram.n_max), maxbatch - 1});
    GammaCtl gctl(std::max(0, gamma), false);  // 0 = adaptive MTP rounds
    std::vector<int> drafts(8), v(9);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int accepted = 0, ng_rounds = 0, mtp_rounds = 0, ng_roll = 0, mtp_roll = 0;
    double verify_ms = 0, replay_ms = 0;
    // D2-S2c 瓶颈分析：MTP 轮三段耗时（verify 段含会合等待）。
    double mtp_draft_ms = 0, mtp_vrfy_ms = 0, mtp_commit_ms = 0;
    // ngram verify chunking: QWENOX_NGRAM_CHUNK=0 keeps the legacy single-pass
    // 65-row verify; N pins the chunk size; unset adapts within [8,64]
    // (init 16), doubling after fully-accepted rounds, halving after rejects.
    // The level is a Model member shared with the sampling chain and kept
    // across requests (workload property, not per-request).
    ng_chunk_init();
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto verify_ng = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    auto T0 = std::chrono::steady_clock::now();
    // D2-S2c: count this worker for the WHOLE chain loop (same shape as
    // spec_loop's VrActive). Per-MTP-round counting was dead: the round body
    // runs only while holding the GPU turn, so the count never exceeded 1 and
    // chain rounds never joined the rendezvous. ngram rounds excuse
    // themselves per round (VrExcuse) — they never submit, so counting them
    // would stall the all-arrived trigger until every timeout.
    struct VrLoop {
      GpuModel& m;
      VrLoop(GpuModel& m) : m(m) { m.vseg_round_enter(); }
      ~VrLoop() { m.vseg_round_exit(); }
    } vr_loop(*this);
    struct VrExcuse {
      GpuModel& m;
      VrExcuse(GpuModel& m) : m(m) { m.vseg_round_exit(); }
      ~VrExcuse() { m.vseg_round_enter(); }
    };
    // D2-S3c: draft rendezvous membership mirrors the verify one -- ngram
    // rounds never submit an MTP draft, so they excuse themselves per round.
    const bool dr_on = dr_round_enter_fn && dr_worker_ok();
    OccGamma occ(false, nslots, dr_on);
    struct DrLoop {
      GpuModel& m;
      bool on;
      DrLoop(GpuModel& m, bool on) : m(m), on(on) {
        if (on) m.dr_round_enter();
      }
      ~DrLoop() { if (on) m.dr_round_exit(); }
    } dr_loop(*this, dr_on);
    struct DrExcuse {
      GpuModel& m;
      bool on;
      DrExcuse(GpuModel& m, bool on) : m(m), on(on) {
        if (on) m.dr_round_exit();
      }
      ~DrExcuse() { if (on) m.dr_round_enter(); }
    };
    while ((int)out.size() < gen && pos < maxctx) {
      yield_point();  // round top: only persistent slot state is live
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the pure-ngram loop: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      const bool ng_open = ng_gate.open();
      if (ng_open && ng_gate.probing())
        want = std::min(want, std::max(8, ngram.n_min));
      std::vector<int> draft =
          (want > 0 && ng_open)
              ? ngram.draft(chist, cur, want)
              : std::vector<int>{};
      bool stop = false;
      if (!draft.empty() && ng_legacy) {
        // ---- ngram round: long draft, whole-state checkpoint + replay ----
        // Never submits to the rendezvous: excuse this worker for the round.
        VrExcuse vr_x(*this);
        DrExcuse dr_x(*this, dr_on);
        const int base = pos;
        const size_t history_size = chist.size();
        // Stash h_{base-1} (d_R) for the MTP patch below: prefill_batch
        // overwrites d_R with its last row. d_mtap is idle outside prefill.
        CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        ngram_ckpt_save();
        std::vector<int> tokens{cur};
        tokens.insert(tokens.end(), draft.begin(), draft.end());
        auto t = std::chrono::steady_clock::now();
        verify_ng(tokens, base);
        std::vector<int> targets(tokens.size());
        batch_argmax((int)tokens.size(), targets.data());
        verify_ms += elapsed(t);
        int a = 0;
        while (a < (int)draft.size() && draft[a] == targets[a]) ++a;
        int keep = a;
        for (int j = 0; j < a; ++j) {
          if (emit && !emit(draft[j])) { keep = j; stop = true; break; }
          out.push_back(draft[j]);
        }
        if (!stop) {
          if (emit && !emit(targets[a])) stop = true;
          else out.push_back(targets[a]);
        }
        if (keep != (int)draft.size()) {
          // Restore every overwritten position-mod state, then replay exactly
          // [cur, accepted drafts] (same as spec_loop_ngram).
          t = std::chrono::steady_clock::now();
          ngram_ckpt_restore();
          chist.resize(history_size);
          pos = base;
          ple_ringpos = base % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          tokens.resize(keep + 1);
          verify_ng(tokens, base);
          CK(hipStreamSynchronize(g_str));
          replay_ms += elapsed(t);
          ++ng_roll;
        }
        cur = targets[keep];
        accepted += keep;
        ++ng_rounds;
        ng_gate.update(keep, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        // Patch the MTP KV over committed positions base-1..base+keep-1: tap
        // row 0 = stashed h_{base-1} paired with cur; rows 1..keep = d_Rb
        // rows 0..keep-1 (the committed tokens' trunk streams — the replay
        // re-generated them on partial accept). This restores the round-start
        // invariant (mtp_pos = pos-1), keeping MTP drafts live for later
        // rounds. d_R itself already holds h_{pos-1} (verify/replay leave the
        // last processed row's stream there; mtp_ingest_b does not touch it).
        if (keep > 0)
          CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                            (size_t)keep * 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        mtp_ingest_b(d_mtap,
                     std::vector<int>(tokens.begin(),
                                      tokens.begin() + keep + 1),
                     keep + 1, base - 1);
      } else if (!draft.empty()) {
        // ---- ngram round: chunked verify, tokens emit as each chunk ----
        // accepts. A rejection restores only this chunk's checkpoint and
        // replays its committed prefix (bounded by ng_cs), replacing the
        // legacy 65-row verify + keep-sized replay. ng_cs doubles after a
        // fully-accepted round (hot tables return to single-pass 65-row
        // throughput) and halves after a rejection. MTP KV is patched per
        // chunk while its d_Rb rows are still live; d_mtap row 0 carries the
        // h_{base-1} stash for the chunk about to verify (verify/replay never
        // touch d_mtap — mtp_tap_capture is off outside prefill).
        // Never submits to the rendezvous: excuse this worker for the round.
        VrExcuse vr_x(*this);
        DrExcuse dr_x(*this, dr_on);
        CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                          g_str));
        int cur_c = cur, keep_round = 0;
        bool round_full = true;
        int di = 0;
        while (di < (int)draft.size()) {
          const int cs = std::min(ng_cs, (int)draft.size() - di);
          const int base = pos;
          const size_t history_size = chist.size();
          ngram_ckpt_save();
          std::vector<int> tokens{cur_c};
          tokens.insert(tokens.end(), draft.begin() + di,
                        draft.begin() + di + cs);
          auto t = std::chrono::steady_clock::now();
          verify_ng(tokens, base);
          std::vector<int> targets(tokens.size());
          batch_argmax((int)tokens.size(), targets.data());
          verify_ms += elapsed(t);
          int a = 0;
          while (a < cs && draft[di + a] == targets[a]) ++a;
          int keep = a;
          for (int j = 0; j < a; ++j) {
            if (emit && !emit(draft[di + j])) { keep = j; stop = true; break; }
            out.push_back(draft[di + j]);
          }
          if (!stop) {
            if (emit && !emit(targets[a])) stop = true;
            else out.push_back(targets[a]);
          }
          if (keep != cs) {
            // Mid-chunk rejection (or consumer stop): rows keep+1..cs
            // polluted the state — restore this chunk's checkpoint and
            // replay exactly its committed prefix. The round ends here.
            t = std::chrono::steady_clock::now();
            ngram_ckpt_restore();
            chist.resize(history_size);
            pos = base;
            ple_ringpos = base % 9;
            CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
            CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4,
                              hipMemcpyHostToDevice, g_str));
            tokens.resize(keep + 1);
            verify_ng(tokens, base);
            CK(hipStreamSynchronize(g_str));
            replay_ms += elapsed(t);
            ++ng_roll;
            round_full = false;
          }
          // MTP patch for this chunk's committed positions (same tap layout
          // as the legacy round: row 0 = stash, rows 1..keep = d_Rb rows).
          if (keep > 0)
            CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                              (size_t)keep * 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          mtp_ingest_b(d_mtap,
                       std::vector<int>(tokens.begin(),
                                        tokens.begin() + keep + 1),
                       keep + 1, base - 1);
          keep_round += keep;
          cur_c = targets[keep];
          if (!round_full || stop) break;
          // Chunk fully accepted: the continuation targets[cs] fills the
          // position the next draft token proposes. Chain into the next
          // chunk only when it equals that draft token (a mismatch is a
          // boundary rejection: the correction is already committed, the
          // state is clean, and the round ends without a replay).
          if (di + cs >= (int)draft.size()) break;  // bonus row: draft done
          if (targets[cs] != draft[di + cs]) { round_full = false; break; }
          ++keep_round;  // the splice committed draft[di+cs] as a draft
          di += cs + 1;
          // Stash h_{pos-1} (d_R = the accepted chunk's last row) as the next
          // chunk's tap pair before its verify overwrites d_R.
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        }
        cur = cur_c;
        accepted += keep_round;
        ++ng_rounds;
        ng_gate.update(keep_round, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        if (!ng_cs_fixed) {
          if (round_full && (int)draft.size() > ng_cs)
            ng_cs = std::min(64, ng_cs * 2);
          else if (!round_full)
            ng_cs = std::max(8, ng_cs / 2);
        }
      } else {
        // ---- MTP round (ngram table missed): short draft, snapshot rollback
        // ---- (same machinery as spec_loop). This worker is counted by the
        // ---- loop-scope VrLoop; the rendezvous happens at the verify below.
        int q = pos - 1;
        int g = std::min({gctl.cur, gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        // Shared row budget + occupancy γ, as spec_loop (may return 0: the
        // round still drafts one step to keep the MTP KV, verifies cur only).
        g = occ.pick(g, vseg_active_fn ? vseg_active_fn() : 1, (int)kVSnapRows);
        const int gd = std::max(1, g);
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        auto t0 = std::chrono::steady_clock::now();
        // D2-S3c: rendezvous draft (see spec_loop); false -> serial burst.
        const bool dr_done =
            dr_on && dr_pool_fn(*this, DrSeg{bound_, gd, cur, drafts.data()});
        if (!dr_done) {
        // draft 1's indexer ring rows are truth; later drafts pollute them.
        if (draft_desync) {
          mtp_step(d_R, cur, nullptr, true, nullptr, d_argmax_arr);
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          for (int j = 1; j < gd; j++)
            mtp_step(d_mR, 0, nullptr, true, d_argmax_arr + j - 1,
                     d_argmax_arr + j);
          CK(hipMemcpy(drafts.data(), d_argmax_arr, (size_t)gd * sizeof(int),
                       hipMemcpyDeviceToHost));
        } else {
          drafts[0] = mtp_step(d_R, cur);
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          for (int j = 1; j < gd; j++) drafts[j] = mtp_step(d_mR, drafts[j - 1]);
        }
        }
        std::vector<int> vt;
        vt.reserve(g + 1);
        vt.push_back(cur);
        for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
        auto t1 = std::chrono::steady_clock::now();
        mtp_draft_ms += elapsed(t0);
        spec_snap = true;  // capture per-row rollback snapshots during verify
        vsnap_row0_ = 0;  // D2-S2a: serial verify always packs at row 0
        if (!verify_graph_round_try(vt, q + 1, v.data())) {
          // D2-S2b: rendezvous verify (same shape as spec_loop); false ->
          // this slot runs the serial verify itself.
          const VSeg vseg{bound_, &vt, q + 1};
          if (vseg_pool_fn && vseg_pool_fn(*this, vseg, v.data(), nullptr))
            spec_snap = false;
          else {
            static const bool no_vseg =
                getenv("QWENOX_NO_VSEG") && !env_zero("QWENOX_NO_VSEG");
            if (no_vseg || !verify_segments(&vseg, 1, v.data())) {
              prefill_batch(vt, q + 1, false, true);
              batch_argmax(g + 1, v.data());
            }
            spec_snap = false;
          }
        } else
          spec_snap = false;
        mtp_vrfy_ms += elapsed(t1);  // 含会合等待（池内等同伴到齐/超时）
        auto t2 = std::chrono::steady_clock::now();
        int a = 0;
        while (a < g && drafts[a] == v[a]) a++;
        for (int j = 0; j < g; j++) {
          depth_tot[j]++;
          if (j < a) depth_hits[j]++;
        }
        if (g > 0) gctl.update(g, a);  // γ = 0 rounds carry no draft evidence
        occ.update(g, a);
        ++mtp_rounds;
        last_spec_proposed += g;
        int d = a;
        bool keep_v = true;
        if (emit) {
          for (int j = 0; j < a; j++)
            if (!emit(drafts[j])) { d = j; stop = true; keep_v = false; break; }
          if (!stop && !emit(v[a])) { stop = true; keep_v = false; }
        }
        for (int j = 0; j < d; j++) out.push_back(drafts[j]);
        if (keep_v) out.push_back(v[a]);
        accepted += d;
        if (!stop && d == g) {
          gdn_commit(g + 1);  // GDN S: replay all rows (verify left it read-only)
        cur = v[g];  // full accept; state at q+1+g
        } else {
          ++mtp_roll;
          // Install the state at commit depth d from the verify snapshots —
          // identical to spec_loop's rollback.
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          gdn_commit(d + 1);  // S after the committed rows (replay)
          CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)(vsnap_row0_ + d) * ngdn * 3 * 10240,
                            (size_t)ngdn * 3 * 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          if (ckpt_iraw)
            k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
                d_iproj_snap + (size_t)vsnap_row0_ * 128,
                d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
                d_iraw, q + 1, q + d + 1, kVSnapRows);
          const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
          for (int p = q + d + 2; p <= q + g + 1; p++)
            CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                              ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          pos = q + d + 2;
          ple_ringpos = pos % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          // vsnap_row0_ = this slot's d_Rb row base (D2-S2 rendezvous
          // packing; 0 on the serial path).
          restore_dR_row(d, vsnap_row0_);
          cur = v[d];
        }
        // Committed-token history: rebuild explicitly so it is right whether
        // or not ple_on made the verify append (spec_loop relies on ple_on).
        chist.resize(chist_sz);
        chist.insert(chist.end(), vt.begin(), vt.begin() + d + 1);
        // MTP self-repair over committed positions q+1..q+d (as spec_loop).
        CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (d > 0) {
          std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
          mtp_ingest_b(spec_taps(d, vsnap_row0_), mtoks, d, q + 1);
        } else {
          mtp_pos = q + 1;
          CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
        }
        mtp_commit_ms += elapsed(t2);
      }
      ++last_spec_rounds;
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = (int)out.size();
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    last_spec_commit = (int)out.size();
    last_spec_rollbacks = ng_roll + mtp_roll;
    fprintf(stderr,
            "%s chain: %d tokens in %.2f s = %.1f tok/s | rounds=%d (ngram=%d "
            "mtp=%d) commit/round=%.2f accepted=%d proposed=%d rollbacks=%d\n",
            log_ts(), (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0,
            last_spec_rounds, ng_rounds, mtp_rounds,
            last_spec_rounds ? (double)out.size() / last_spec_rounds : 0.0,
            accepted, last_spec_proposed, last_spec_rollbacks);
    if (mtp_rounds) {
      fprintf(stderr, "chain mtp depth acc:");
      for (int j = 0; j < 8 && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1, (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
      // D2-S2c: per-phase ms/round (verify 段含会合等待；残差 = 轮转等待 +
      // ngram 轮 + emit)。
      fprintf(stderr,
              "chain mtp ms/round: draft=%.1f verify=%.1f commit=%.1f "
              "other=%.1f\n",
              mtp_draft_ms / mtp_rounds, mtp_vrfy_ms / mtp_rounds,
              mtp_commit_ms / mtp_rounds,
              (sec * 1000.0 - mtp_draft_ms - mtp_vrfy_ms - mtp_commit_ms -
               verify_ms - replay_ms) /
                  last_spec_rounds);
      gctl.report("chain");
      occ.report("chain");
    }
    return out;
  }

  // Sampling speculative decode. The sampler supplies normalized filtered
  // distributions, independent draft/accept RNG streams, and residual draws.
  // Accepted draft tokens use p/q rejection sampling; a rejection draws from
  // normalized max(p-q, 0), while full acceptance draws the verify continuation.
  template <typename Sampler>
  std::vector<int> spec_loop_sample(
      int cur, int gen, int gamma, Sampler& sampler,
      const std::function<bool(int, float)>& emit = nullptr) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(gen + 1);
    GammaCtl gctl(gamma, true);
    std::vector<int> drafts(8);
    std::vector<float> draft_lps(8), verify_logits, p_row(g_cfg.vocab);
    std::vector<std::vector<float>> q_rows(8);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int rounds = 0, rollbacks = 0;
    double t_draft = 0, t_verify = 0, t_roll = 0, t_ingest = 0;
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto T0 = std::chrono::steady_clock::now();
    while ((int)out.size() < gen) {
      yield_point();  // round top: mtp_pos == pos-1, d_R == h_{pos-1}
      int q = pos - 1;
      int g = std::min(spec_conf() > 0 ? kSpecConfMax : gctl.cur,
                       gen - (int)out.size());
      size_t chist_sz = chist.size();
      mtp_ckpt_save();
      Sampler draft_sampler = sampler.fork_stream(
          0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
      Sampler accept_sampler = sampler.fork_stream(
          0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
      auto t0 = std::chrono::steady_clock::now();
      // GPU sampling prep (direction A): corrections + top-k on device, host
      // replays the filter chain on <= top_k candidates per row. The dense
      // path remains as fallback (QWENOX_SMS_GPU=0, top_k unset/>64, overflow).
      const bool smg = sms_gpu_active(sampler) && sms_round_shared(sampler);
      std::vector<typename Sampler::SDist> q_sd(smg ? g : 0);
      std::vector<char> q_sparse(g, 0);
      for (int j = 0; j < g; j++) {
        bool dense = true;
        if (smg) {
          mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, nullptr, false);
          if (sms_topk_rows(d_logits, g_cfg.vocab, 1, sampler.top_k, false,
                            sms_out_ids.data(), sms_out_vals.data())) {
            dense = false;
          } else {
            // Candidate overflow (never expected): corrections touched the
            // logits in place, so regenerate the raw row before going dense.
            gemv(dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight")), d_x,
                 d_logits);
            q_rows[j].resize(g_cfg.vocab);
            CK(hipMemcpy(q_rows[j].data(), d_logits, (size_t)g_cfg.vocab * 4,
                         hipMemcpyDeviceToHost));
          }
        } else {
          mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
        }
        float qlp = 0.0f;
        // Penalties are defined over committed/emitted tokens only. Provisional
        // draft tokens therefore do not enter the proposal sampler's history;
        // q_rows[j]/q_sd[j] records the distribution that produced drafts[j].
        if (dense) {
          draft_sampler.prepare(q_rows[j]);
          drafts[j] = draft_sampler.draw_prepared(q_rows[j], &qlp, false);
        } else {
          sms_prep(draft_sampler, 0, true, q_sd[j]);
          drafts[j] = draft_sampler.draw_sparse(q_sd[j], &qlp, false);
          q_sparse[j] = 1;
        }
        if (j == 0) {
          // MTP indexer state after consuming cur is causal truth. Later
          // speculative draft steps are discarded before committed ingest.
          CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        }
        if (spec_conf_stop(j, g, q_sparse[j] ? &q_sd[j].pr : nullptr, q_rows[j])) {
          g = j + 1;
          break;
        }
      }
      auto t1 = std::chrono::steady_clock::now();
      std::vector<int> vt;
      vt.reserve(g + 1);
      vt.push_back(cur);
      for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
      spec_snap = true;
      vsnap_row0_ = 0;  // D2-S2a: serial verify always packs at row 0
      prefill_batch(vt, q + 1, true, true);
      spec_snap = false;
      bool smg_v = false;
      if (smg) {
        const bool pfx = sms_prefix_upload(sampler, drafts.data(), 0, g + 1);
        batch_logits_device(g + 1);
        smg_v = sms_topk_rows(d_logits_b, g_cfg.vocab, g + 1, sampler.top_k,
                              pfx, sms_out_ids.data(), sms_out_vals.data());
      }
      if (!smg_v) batch_logits(g + 1, verify_logits);
      const int smk = sampler.top_k;

      Sampler target_view = sampler;
      std::vector<typename Sampler::SDist> p_sd(smg_v ? g + 1 : 0);
      int a = 0;
      // q_dense[a]: q_rows[a] holds a real dense distribution. The sparse
      // draft path leaves q_rows[a] stale, so scatter q_sd[a] when the two
      // sides disagree (rare per-row fallback mixing).
      auto q_dense = [&](int r) -> std::vector<float>& {
        if (q_sparse[r]) {
          q_sparse[r] = 0;
          q_rows[r].assign(g_cfg.vocab, 0.0f);
          for (size_t u = 0; u < q_sd[r].ids.size(); u++)
            q_rows[r][q_sd[r].ids[u]] = q_sd[r].pr[u];
        }
        return q_rows[r];
      };
      for (; a < g; a++) {
        float px, qx;
        const bool q_sp = q_sparse[a];
        if (smg_v && q_sp) {
          sms_prep(target_view, a, false, p_sd[a]);
          px = Sampler::sdist_prob(p_sd[a], drafts[a]);
          qx = Sampler::sdist_prob(q_sd[a], drafts[a]);
        } else {
          if (smg_v) {
            // Sparse verify but dense draft row: materialize p densely.
            sms_prep(target_view, a, false, p_sd[a]);
            p_row.assign(g_cfg.vocab, 0.0f);
            for (size_t u = 0; u < p_sd[a].ids.size(); u++)
              p_row[p_sd[a].ids[u]] = p_sd[a].pr[u];
          } else {
            std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            target_view.prepare(p_row);
          }
          px = p_row[drafts[a]];
          qx = q_dense(a)[drafts[a]];
        }
        draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
        double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
        if (accept_sampler.uniform() >= threshold) break;
        target_view.remember(drafts[a]);
      }

      int next;
      float next_lp = 0.0f;
      target_view.rng = sampler.rng;
      if (a < g) {
        next = (smg_v && q_sparse[a])
                   ? target_view.sample_residual_sparse(p_sd[a], q_sd[a],
                                                        &next_lp)
                   : target_view.sample_residual(p_row, q_dense(a),
                                                 &next_lp);
      } else if (smg_v) {
        sms_prep(target_view, g, false, p_sd[g]);
        next = target_view.draw_sparse(p_sd[g], &next_lp, false);
      } else {
        std::copy_n(verify_logits.data() + (size_t)g * g_cfg.vocab,
                    g_cfg.vocab, p_row.data());
        target_view.prepare(p_row);
        next = target_view.draw_prepared(p_row, &next_lp, false);
      }
      sampler.rng = target_view.rng;
      auto t2 = std::chrono::steady_clock::now();
      for (int j = 0; j < g; j++) {
        depth_tot[j]++;
        if (j < a) depth_hits[j]++;
      }
      gctl.update(g, a);
      rounds++;
      live_spec_rounds = rounds;
      live_spec_proposed += g;
      live_spec_commits = (int)out.size();  // pre-emit; one round of lag is fine

      int d = a, stopped_token = -1;
      bool stop = false, keep_next = true;
      if (emit) {
        for (int j = 0; j < a; j++)
          if (!emit(drafts[j], draft_lps[j])) {
            d = j;
            stopped_token = drafts[j];
            stop = true;
            keep_next = false;
            break;
          }
        if (!stop && !emit(next, next_lp)) {
          stop = true;
          keep_next = false;
        }
      }
      for (int j = 0; j < d; j++) {
        out.push_back(drafts[j]);
        sampler.remember(drafts[j]);
      }
      if (keep_next) {
        out.push_back(next);
        sampler.remember(next);
      }

      auto t3 = std::chrono::steady_clock::now();
      if (!stop && d == g) {
        gdn_commit(g + 1);  // GDN S: replay all rows
        cur = next;
      } else {
        rollbacks++;
        int ngdn = g_cfg.layers - g_cfg.layers / 4;
        gdn_commit(d + 1);  // S after the committed rows (replay)
        CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)(vsnap_row0_ + d) * ngdn * 3 * 10240,
                          (size_t)ngdn * 3 * 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (ckpt_iraw)
          k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
              d_iproj_snap + (size_t)vsnap_row0_ * 128,
              d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
              d_iraw, q + 1, q + d + 1, kVSnapRows);
        const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
        for (int p = q + d + 2; p <= q + g + 1; p++)
          CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                            ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
        chist.resize(chist_sz + d + 1);
        pos = q + d + 2;
        ple_ringpos = pos % 9;
        CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
        CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                          g_str));
        restore_dR_row(d);
        cur = stopped_token >= 0 ? stopped_token : next;
      }
      auto t4 = std::chrono::steady_clock::now();
      CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                        hipMemcpyDeviceToDevice, g_str));
      if (d > 0) {
        std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
        mtp_ingest_b(spec_taps(d), mtoks, d, q + 1);
      } else {
        mtp_pos = q + 1;
        CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
      }
      if (stop) break;
      auto t5 = std::chrono::steady_clock::now();
      t_draft += ms(t0, t1);
      t_verify += ms(t1, t2);
      t_roll += ms(t3, t4);
      t_ingest += ms(t4, t5);
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    fprintf(stderr,
            "\nspec-sample: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "commit/round=%.2f rollbacks=%d\n",
            (int)out.size(), sec, out.size() / sec, rounds,
            rounds ? (double)out.size() / rounds : 0.0, rollbacks);
    fprintf(stderr, "spec-sample depth acc:");
    for (int j = 0; j < 8 && depth_tot[j]; j++)
      fprintf(stderr, " %d:%.3f", j + 1,
              (double)depth_hits[j] / depth_tot[j]);
    fprintf(stderr,
            "\nspec-sample time: draft=%.0fms verify=%.0fms "
            "rollback=%.0fms ingest=%.0fms\n",
            t_draft, t_verify, t_roll, t_ingest);
    gctl.report("spec-sample");
    last_spec_rounds = rounds;
    last_spec_commit = (int)out.size();
    last_spec_proposed = live_spec_proposed;
    last_spec_rollbacks = rollbacks;
    return out;
  }

  // Sampling chain (llama.cpp ngram semantics + rejection-sampled MTP leg).
  // ngram round: a draft token is committed only when it equals a fresh draw
  // from the target distribution at its verify row; the first mismatching
  // draw becomes the correction token. Every emitted token is therefore an
  // exact target-distribution sample — the draft only decides how many
  // verify rows stay usable. Rollback/replay is the greedy ngram round's
  // checkpoint machinery; the MTP patch over committed positions is the
  // greedy chain's. The MTP round is spec_loop_sample's round body with the
  // greedy chain's explicit chist rebuild. mtp_ok=false degrades empty-draft
  // rounds to plain serial sampling steps (MTP KV unavailable).
  template <typename Sampler>
  std::vector<int> spec_loop_chain_sample(
      int cur, int gen, int gamma, Sampler& sampler, bool mtp_ok,
      const std::function<bool(int, float)>& emit = nullptr, bool seeded = false) {
    last_spec_rounds = last_spec_commit = last_spec_rollbacks =
        last_spec_proposed = 0;
    live_spec_rounds = live_spec_commits = live_spec_proposed = 0;
    std::vector<int> out;
    out.reserve(std::max(0, gen));
    const int maxg = std::min({64, std::max(1, ngram.n_max), maxbatch - 1});
    GammaCtl gctl(std::max(0, gamma), true);  // 0 = adaptive MTP rounds
    std::vector<int> drafts(8);
    std::vector<float> draft_lps(8), verify_logits, p_row(g_cfg.vocab);
    std::vector<std::vector<float>> q_rows(8);
    int depth_hits[8] = {0}, depth_tot[8] = {0};
    int accepted = 0, ng_rounds = 0, mtp_rounds = 0, ng_roll = 0,
        mtp_roll = 0, fallback = 0, ng_keep = 0, ng_prop = 0;
    double verify_ms = 0, replay_ms = 0, ng_ms = 0;
    // ngram verify chunking: same env semantics as the greedy chain; the
    // level lives in the shared Model member (see ng_chunk_init).
    ng_chunk_init();
    auto elapsed = [](auto t) {
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t).count();
    };
    auto verify_ng = [&](const std::vector<int>& tokens, int base) {
      ngram_verify_active = true;
      prefill_batch(tokens, base, ngram_legacy_final, true);
      ngram_verify_active = false;
      if (!ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
    };
    auto T0 = std::chrono::steady_clock::now();
    // Verify rendezvous membership (as spec_loop_chain): count this worker for
    // the whole loop when its MTP rounds can submit; ngram rounds and plain
    // serial steps never submit, so they excuse themselves per round.
    // QWENOX_VR_SAMPLE=0 keeps sampled requests out of the rendezvous.
    static const bool vr_sample = !env_zero("QWENOX_VR_SAMPLE");
    const bool vr_on = vr_sample && mtp_ok && vseg_pool_fn != nullptr;
    // C7: MTP draft bursts go to the draft rendezvous pool too (one batched
    // MTP layer + lm_head pass per step; each member draws on the host from
    // its own row, see DrSeg::step). Same membership discipline as the
    // verify pool: the whole loop counts, ngram rounds and serial steps
    // excuse themselves. QWENOX_DR_SAMPLE=0 keeps the serial per-slot burst.
    static const bool dr_sample = !env_zero("QWENOX_DR_SAMPLE");
    const bool dr_on = dr_sample && mtp_ok && dr_round_enter_fn && dr_worker_ok();
    OccGamma occ(true, nslots, dr_on, seeded);
    struct DrLoopS {
      GpuModel& m;
      bool on;
      DrLoopS(GpuModel& m, bool on) : m(m), on(on) { if (on) m.dr_round_enter(); }
      ~DrLoopS() { if (on) m.dr_round_exit(); }
    } dr_loop(*this, dr_on);
    struct DrExcuseS {
      GpuModel& m;
      bool on;
      DrExcuseS(GpuModel& m, bool on) : m(m), on(on) { if (on) m.dr_round_exit(); }
      ~DrExcuseS() { if (on) m.dr_round_enter(); }
    };
    struct VrLoopS {
      GpuModel& m;
      bool on;
      VrLoopS(GpuModel& m, bool on) : m(m), on(on) { if (on) m.vseg_round_enter(); }
      ~VrLoopS() { if (on) m.vseg_round_exit(); }
    } vr_loop(*this, vr_on);
    struct VrExcuseS {
      GpuModel& m;
      bool on;
      VrExcuseS(GpuModel& m, bool on) : m(m), on(on) { if (on) m.vseg_round_exit(); }
      ~VrExcuseS() { if (on) m.vseg_round_enter(); }
    };
    std::vector<int> pool_ids;  // post-callback top-k rows (worker-local)
    std::vector<float> pool_vals;
    // QWENOX_SYNC_PROF (T1): per-MTP-round wall split, host-side clock only.
    // GPU work is async, so a segment holds its enqueue time plus whatever it
    // blocked on (syncprof attributes the blocking to call sites).
    enum { RP_YIELD, RP_TOP, RP_PRE, RP_DRAFT, RP_VERIFY, RP_VPOOL, RP_JUDGE, RP_EMIT,
           RP_COMMIT, RP_WALL, RP_N };
    double rp[RP_N] = {};
    using sclk = std::chrono::steady_clock;
    sclk::time_point rp_t;
    auto rp_lap = [&](int k) {  // add time since the last lap to segment k
      if (!g_sp_on) return;
      const auto n = sclk::now();
      rp[k] += std::chrono::duration<double, std::milli>(n - rp_t).count();
      rp_t = n;
    };
    while ((int)out.size() < gen && pos < maxctx) {
      const sclk::time_point rp_top = g_sp_on ? sclk::now() : sclk::time_point{};
      rp_t = rp_top;
      yield_point();  // round top: only persistent slot state is live
      rp_lap(RP_YIELD);
      if (chist.size() != (size_t)pos)
        throw std::runtime_error(
            "chain-sample: token history and trunk position disagree");
      ngram.observe(chist);
      // Same budget math as the greedy chain: one slot stays for the
      // target-model continuation, never verify past gen or maxctx.
      int want = std::min({maxg, gen - (int)out.size() - 1, maxctx - pos - 1});
      const bool ng_open = ng_gate.open();
      if (ng_open && ng_gate.probing())
        want = std::min(want, std::max(8, ngram.n_min));
      if (ng_open && NgGate::sample_cap() > 0 && !ng_gate.last_full)
        want = std::min(want, std::max(NgGate::sample_cap(), ngram.n_min));
      std::vector<int> draft =
          (want > 0 && ng_open)
              ? ngram.draft(chist, cur, want)
              : std::vector<int>{};
      bool stop = false;
      const auto t_round = std::chrono::steady_clock::now();
      // GPU sampling prep (direction A) applies to all three legs below.
      const bool smg = sms_gpu_active(sampler) && sms_round_shared(sampler);
      if (!draft.empty() && ng_legacy) {
        // ---- ngram round: long draft, whole-state checkpoint + replay ----
        VrExcuseS vr_x(*this, vr_on);
        DrExcuseS dr_x(*this, dr_on);
        const int base = pos;
        const size_t history_size = chist.size();
        if (mtp_ok)
          // Stash h_{base-1} (d_R) for the MTP patch below (as spec_loop_chain).
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        ngram_ckpt_save();
        std::vector<int> tokens{cur};
        tokens.insert(tokens.end(), draft.begin(), draft.end());
        auto t = std::chrono::steady_clock::now();
        verify_ng(tokens, base);
        const int ngP = (int)tokens.size();
        bool smg_v = false;
        if (smg) {
          const bool pfx = sms_prefix_upload(sampler, draft.data(), 0, ngP);
          batch_logits_device(ngP);
          smg_v = sms_topk_rows(d_logits_b, g_cfg.vocab, ngP, sampler.top_k,
                                pfx, sms_out_ids.data(), sms_out_vals.data());
        }
        if (!smg_v) batch_logits(ngP, verify_logits);
        const int smk = sampler.top_k;
        verify_ms += elapsed(t);
        // Walk the draft rows with a scratch sampler view: accept draft[a]
        // iff it equals this round's draw from row a, else the draw is the
        // correction. keep = committed draft count (emit may stop early).
        Sampler view = sampler;
        const size_t committed0 = out.size();
        std::vector<typename Sampler::SDist> p_sd(smg_v ? ngP : 0);
        int a = 0, next = -1;
        float next_lp = 0.0f;
        for (; a < (int)draft.size(); a++) {
          int s;
          float lp_a = 0.0f;
          if (smg_v) {
            sms_prep(view, a, false, p_sd[a]);
            s = view.draw_sparse(p_sd[a], &lp_a, true);
          } else {
            std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                        g_cfg.vocab, p_row.data());
            s = view.sample(p_row, &lp_a);
          }
          if (s != draft[a]) {
            next = s;
            next_lp = lp_a;
            break;
          }
          if (emit && !emit(draft[a], lp_a)) { stop = true; break; }
          out.push_back(draft[a]);
        }
        const int keep = a;
        if (!stop) {
          if (next < 0) {
            // Full accept: the bonus row continues the sequence.
            if (smg_v) {
              sms_prep(view, draft.size(), false, p_sd[draft.size()]);
              next = view.draw_sparse(p_sd[draft.size()], &next_lp, true);
            } else {
              std::copy_n(verify_logits.data() +
                              (size_t)draft.size() * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              next = view.sample(p_row, &next_lp);
            }
          }
          if (emit && !emit(next, next_lp)) stop = true;
          else out.push_back(next);
        }
        sampler.rng = view.rng;
        for (size_t j = committed0; j < out.size(); j++)
          sampler.remember(out[j]);
        if (keep != (int)draft.size()) {
          // Restore every overwritten position-mod state, then replay exactly
          // [cur, accepted drafts] (same as the greedy ngram round).
          t = std::chrono::steady_clock::now();
          ngram_ckpt_restore();
          chist.resize(history_size);
          pos = base;
          ple_ringpos = base % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          tokens.resize(keep + 1);
          verify_ng(tokens, base);
          CK(hipStreamSynchronize(g_str));
          replay_ms += elapsed(t);
          ++ng_roll;
        }
        if (next >= 0) cur = next;
        accepted += keep;
        ++ng_rounds;
        ng_keep += keep;
        ng_prop += (int)draft.size();
        ng_gate.update(keep, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        if (mtp_ok) {
          // Patch the MTP KV over committed positions (as spec_loop_chain):
          // tap row 0 = stashed h_{base-1} paired with cur; rows 1..keep the
          // committed tokens' trunk streams. Restores mtp_pos = pos-1.
          if (keep > 0)
            CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                              (size_t)keep * 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          mtp_ingest_b(d_mtap,
                       std::vector<int>(tokens.begin(),
                                        tokens.begin() + keep + 1),
                       keep + 1, base - 1);
        }
        ng_ms += elapsed(t_round);
      } else if (!draft.empty()) {
        // ---- chunked ngram round, sampling semantics: the Sampler view ----
        // walks rows in draft order across chunks, so the draw stream matches
        // the single-pass form; tokens emit per chunk (see the greedy chain).
        VrExcuseS vr_x(*this, vr_on);
        DrExcuseS dr_x(*this, dr_on);
        if (mtp_ok)
          // Stash h_{base-1} (d_R) for the per-chunk MTP patch below.
          CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                            g_str));
        Sampler view = sampler;
        const size_t committed0 = out.size();
        int cur_c = cur, keep_round = 0;
        bool round_full = true;
        int cont = -1;
        float cont_lp = 0.0f;
        int di = 0;
        while (di < (int)draft.size()) {
          const int cs = std::min(ng_cs, (int)draft.size() - di);
          const int base = pos;
          const size_t history_size = chist.size();
          ngram_ckpt_save();
          std::vector<int> tokens{cur_c};
          tokens.insert(tokens.end(), draft.begin() + di,
                        draft.begin() + di + cs);
          auto t = std::chrono::steady_clock::now();
          verify_ng(tokens, base);
          const int ngP = (int)tokens.size();
          bool smg_v = false;
          if (smg) {
            // Row a of this chunk is prepared with draft[0..di+a) in the
            // penalty history (the walk only reaches it via matching draws).
            const bool pfx = sms_prefix_upload(sampler, draft.data(), di, ngP);
            batch_logits_device(ngP);
            smg_v =
                sms_topk_rows(d_logits_b, g_cfg.vocab, ngP, sampler.top_k,
                              pfx, sms_out_ids.data(), sms_out_vals.data());
          }
          if (!smg_v) batch_logits(ngP, verify_logits);
          const int smk = sampler.top_k;
          verify_ms += elapsed(t);
          std::vector<typename Sampler::SDist> p_sd(smg_v ? ngP : 0);
          cont = -1;
          int a = 0;
          for (; a < cs; a++) {
            int s;
            float lp_a = 0.0f;
            if (smg_v) {
              sms_prep(view, a, false, p_sd[a]);
              s = view.draw_sparse(p_sd[a], &lp_a, true);
            } else {
              std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              s = view.sample(p_row, &lp_a);
            }
            if (s != draft[di + a]) { cont = s; cont_lp = lp_a; break; }
            if (emit && !emit(draft[di + a], lp_a)) { stop = true; break; }
            out.push_back(draft[di + a]);
          }
          const int keep = a;
          if (!stop) {
            if (cont < 0) {
              // Full chunk accept: the bonus row continues the sequence.
              if (smg_v) {
                sms_prep(view, cs, false, p_sd[cs]);
                cont = view.draw_sparse(p_sd[cs], &cont_lp, true);
              } else {
                std::copy_n(verify_logits.data() + (size_t)cs * g_cfg.vocab,
                            g_cfg.vocab, p_row.data());
                cont = view.sample(p_row, &cont_lp);
              }
            }
            if (emit && !emit(cont, cont_lp)) stop = true;
            else out.push_back(cont);
          }
          if (keep != cs) {
            // Mid-chunk rejection: restore + replay only this chunk's
            // committed prefix (the greedy block has the full commentary).
            t = std::chrono::steady_clock::now();
            ngram_ckpt_restore();
            chist.resize(history_size);
            pos = base;
            ple_ringpos = base % 9;
            CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
            CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4,
                              hipMemcpyHostToDevice, g_str));
            tokens.resize(keep + 1);
            verify_ng(tokens, base);
            CK(hipStreamSynchronize(g_str));
            replay_ms += elapsed(t);
            ++ng_roll;
            round_full = false;
          }
          if (mtp_ok) {
            if (keep > 0)
              CK(hipMemcpyAsync(d_mtap + 10240, spec_taps(keep),
                                (size_t)keep * 10240 * 4,
                                hipMemcpyDeviceToDevice, g_str));
            mtp_ingest_b(d_mtap,
                         std::vector<int>(tokens.begin(),
                                          tokens.begin() + keep + 1),
                         keep + 1, base - 1);
          }
          keep_round += keep;
          if (cont >= 0) cur_c = cont;
          if (!round_full || stop) break;
          // Full chunk accept: chain into the next chunk only when the drawn
          // continuation equals the next draft token (see the greedy block);
          // a mismatch is a clean boundary rejection — no replay needed.
          if (di + cs >= (int)draft.size()) break;  // bonus row: draft done
          if (cont != draft[di + cs]) { round_full = false; break; }
          ++keep_round;  // the splice committed draft[di+cs] as a draft
          di += cs + 1;
          if (mtp_ok)
            CK(hipMemcpyAsync(d_mtap, d_R, 10240 * 4, hipMemcpyDeviceToDevice,
                              g_str));
        }
        sampler.rng = view.rng;
        for (size_t j = committed0; j < out.size(); j++)
          sampler.remember(out[j]);
        if (cont >= 0) cur = cur_c;
        accepted += keep_round;
        ++ng_rounds;
        ng_keep += keep_round;
        ng_prop += (int)draft.size();
        ng_gate.update(keep_round, (int)draft.size());
        last_spec_proposed += (int)draft.size();
        if (!ng_cs_fixed) {
          if (round_full && (int)draft.size() > ng_cs)
            ng_cs = std::min(64, ng_cs * 2);
          else if (!round_full)
            ng_cs = std::max(8, ng_cs / 2);
        }
        ng_ms += elapsed(t_round);
      } else if (mtp_ok) {
        // ---- MTP round (ngram table missed): spec_loop_sample's round ----
        rp_lap(RP_TOP);
        int q = pos - 1;
        int g = std::min({spec_conf() > 0 ? kSpecConfMax : gctl.cur,
                          gen - (int)out.size(), maxctx - pos - 1});
        if (g < 1) break;  // context edge: no room for even one draft
        // Row budget + occupancy γ. A sampled round's γ shapes its RNG draws:
        // for a client-seeded request OccGamma sees a deterministic B (nslots)
        // so the output does not depend on peers (the old live-member clamp
        // here let C7/C8 peers change it: 48/8 - 1 = 5 < adaptive γ max 7);
        // unseeded requests use the live B. γ = 0 still drafts one step (MTP
        // KV at q; the draft stream is forked per position) and verifies cur.
        g = occ.pick(g, vr_on ? (vseg_active_fn ? vseg_active_fn() : 1) : 0,
                     (int)kVSnapRows);
        const int gd = std::max(1, g);
        size_t chist_sz = chist.size();
        mtp_ckpt_save();
        Sampler draft_sampler = sampler.fork_stream(
            0xd1b54a32d192ed03ull ^ (uint64_t)(q + 1));
        Sampler accept_sampler = sampler.fork_stream(
            0x3c6ef372fe94f82aull ^ (uint64_t)(q + 1));
        auto t = std::chrono::steady_clock::now();
        // GPU sampling prep (direction A), as in spec_loop_sample.
        std::vector<typename Sampler::SDist> q_sd(smg ? gd : 0);
        std::vector<char> q_sparse(gd, 0);
        rp_lap(RP_PRE);
        // Draw draft j from its logits row. src: d_logits after the serial
        // mtp_step, a batched step's row of d_logits_b (C7), or nullptr when
        // the serial dense mtp_step already filled q_rows[j]. False = stop
        // drafting after this step (QWENOX_SPEC_CONF), g = j + 1.
        auto draw = [&](int j, const float* src) -> bool {
          bool dense = true;
          if (smg) {
            // A batched step runs between peers' rounds: re-upload this
            // request's correction lists first (no-op without bias/penalties).
            if (src != d_logits) sms_round_shared(sampler);
            if (sms_topk_rows(const_cast<float*>(src), g_cfg.vocab, 1, sampler.top_k, false,
                              sms_out_ids.data(), sms_out_vals.data())) {
              dense = false;
            } else {
              // Candidate overflow: go dense on the raw row (sms_topk_rows
              // only reads it; the serial path regenerates it as before).
              if (src == d_logits)
                gemv(dlm_on ? w_dlm : wview(ckpt.at("lm_head.weight")), d_x, d_logits);
              q_rows[j].resize(g_cfg.vocab);
              CK(hipMemcpy(q_rows[j].data(), src, (size_t)g_cfg.vocab * 4,
                           hipMemcpyDeviceToHost));
            }
          } else if (src) {
            q_rows[j].resize(g_cfg.vocab);
            CK(hipMemcpy(q_rows[j].data(), src, (size_t)g_cfg.vocab * 4,
                         hipMemcpyDeviceToHost));
          }
          // Provisional drafts stay out of the penalty history (as
          // spec_loop_sample): q_rows[j]/q_sd[j] records the distribution.
          // (QWENOX_SMS_CHECK compares against d_logits: serial path only.)
          if (dense) {
            draft_sampler.prepare(q_rows[j]);
            drafts[j] =
                draft_sampler.draw_prepared(q_rows[j], &draft_lps[j], false);
          } else {
            sms_prep(draft_sampler, 0, true, q_sd[j]);
            drafts[j] =
                draft_sampler.draw_sparse(q_sd[j], &draft_lps[j], false);
            q_sparse[j] = 1;
          }
          if (spec_conf_stop(j, g, q_sparse[j] ? &q_sd[j].pr : nullptr, q_rows[j])) {
            g = j + 1;
            return false;
          }
          return true;
        };
        // C7: one batched pass per step across the rendezvous members (the
        // executor calls draw on our row; logits are the serial ones bit for
        // bit, so seeded output does not depend on batching); false = this
        // round keeps the serial burst below.
        bool dr_done = false;
        if (dr_on) {
          const std::function<bool(int, const float*, int&)> step =
              [&](int j, const float* row, int& tok) {
                const bool more = draw(j, row);
                tok = drafts[j];
                return more;
              };
          dr_done = dr_pool_fn(*this, DrSeg{bound_, gd, cur, drafts.data(), &step});
          // peers ran while we waited: restore our lists for the verify below
          if (dr_done && smg) sms_round_shared(sampler);
        }
        if (!dr_done)
        for (int j = 0; j < gd; j++) {
          if (smg)
            mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, nullptr, false);
          else
            mtp_step(j ? d_mR : d_R, j ? drafts[j - 1] : cur, &q_rows[j]);
          const bool more = draw(j, smg ? d_logits : nullptr);
          if (j == 0) {
            // Draft 1's indexer ring rows are truth; later drafts pollute.
            CK(hipMemcpyAsync(d_mring_ckpt, d_miraw, 512 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          }
          if (!more) break;
        }
        rp_lap(RP_DRAFT);
        std::vector<int> vt;
        vt.reserve(g + 1);
        vt.push_back(cur);
        for (int j = 0; j < g; j++) vt.push_back(drafts[j]);
        spec_snap = true;  // capture per-row rollback snapshots during verify
        vsnap_row0_ = 0;  // D2-S2a: serial verify always packs at row 0
        bool smg_v = false, pooled = false;
        if (vr_on) {
          // Rendezvous verify: the executor runs the batched pass, then this
          // callback on our rows (bit-identical logits to the serial verify),
          // so the sampling inputs below are the serial path's exactly. The
          // shared correction lists are re-uploaded first: peers' rounds ran
          // in between and overwrote them.
          bool cb_smg = false;
          const std::function<void(int)> post = [&](int r0) {
            float* rows = d_logits_b + (size_t)r0 * g_cfg.vocab;
            cb_smg = false;
            if (smg && sms_round_shared(sampler)) {
              const bool pfx = sms_prefix_upload(sampler, drafts.data(), 0, g + 1);
              pool_ids.resize((size_t)(g + 1) * sampler.top_k);
              pool_vals.resize((size_t)(g + 1) * sampler.top_k);
              cb_smg = sms_topk_rows(rows, g_cfg.vocab, g + 1, sampler.top_k, pfx,
                                     pool_ids.data(), pool_vals.data());
            }
            if (!cb_smg) {
              verify_logits.resize((size_t)(g + 1) * g_cfg.vocab);
              CK(hipMemcpy(verify_logits.data(), rows,
                           verify_logits.size() * sizeof(float),
                           hipMemcpyDeviceToHost));
            }
          };
          std::vector<int> vtmp((size_t)g + 1);
          const VSeg vseg{bound_, &vt, q + 1};
          rp_lap(RP_VERIFY);
          const bool vp_ok = vseg_pool_fn(*this, vseg, vtmp.data(), &post);
          rp_lap(RP_VPOOL);  // rendezvous wait + batched pass (+ peers' turns)
          if (vp_ok) {
            pooled = true;
            smg_v = cb_smg;
            if (smg_v) {
              std::copy(pool_ids.begin(), pool_ids.end(), sms_out_ids.begin());
              std::copy(pool_vals.begin(), pool_vals.end(), sms_out_vals.begin());
            }
          }
        }
        if (!pooled) {
          vsnap_row0_ = 0;
          prefill_batch(vt, q + 1, true, true);
          if (smg) {
            const bool pfx = sms_prefix_upload(sampler, drafts.data(), 0, g + 1);
            batch_logits_device(g + 1);
            smg_v = sms_topk_rows(d_logits_b, g_cfg.vocab, g + 1, sampler.top_k,
                                  pfx, sms_out_ids.data(), sms_out_vals.data());
          }
          if (!smg_v) batch_logits(g + 1, verify_logits);
        }
        spec_snap = false;
        const int smk = sampler.top_k;
        verify_ms += elapsed(t);
        rp_lap(RP_VERIFY);
        Sampler target_view = sampler;
        std::vector<typename Sampler::SDist> p_sd(smg_v ? g + 1 : 0);
        int a = 0;
        auto q_dense = [&](int r) -> std::vector<float>& {
          if (q_sparse[r]) {
            q_sparse[r] = 0;
            q_rows[r].assign(g_cfg.vocab, 0.0f);
            for (size_t u = 0; u < q_sd[r].ids.size(); u++)
              q_rows[r][q_sd[r].ids[u]] = q_sd[r].pr[u];
          }
          return q_rows[r];
        };
        for (; a < g; a++) {
          float px, qx;
          const bool q_sp = q_sparse[a];
          if (smg_v && q_sp) {
            sms_prep(target_view, a, false, p_sd[a]);
            px = Sampler::sdist_prob(p_sd[a], drafts[a]);
            qx = Sampler::sdist_prob(q_sd[a], drafts[a]);
          } else {
            if (smg_v) {
              // Sparse verify but dense draft row: materialize p densely.
              sms_prep(target_view, a, false, p_sd[a]);
              p_row.assign(g_cfg.vocab, 0.0f);
              for (size_t u = 0; u < p_sd[a].ids.size(); u++)
                p_row[p_sd[a].ids[u]] = p_sd[a].pr[u];
            } else {
              std::copy_n(verify_logits.data() + (size_t)a * g_cfg.vocab,
                          g_cfg.vocab, p_row.data());
              target_view.prepare(p_row);
            }
            px = p_row[drafts[a]];
            qx = q_dense(a)[drafts[a]];
          }
          draft_lps[a] = px > 0.0f ? logf(px) : -INFINITY;
          double threshold = qx > 0.0f ? std::min(1.0, (double)px / qx) : 0.0;
          if (accept_sampler.uniform() >= threshold) break;
          target_view.remember(drafts[a]);
        }
        if (smg_v) spec_dump_round(q, g, a, drafts, q_sd, p_sd);
        int next;
        float next_lp = 0.0f;
        target_view.rng = sampler.rng;
        if (a < g) {
          next = (smg_v && q_sparse[a])
                     ? target_view.sample_residual_sparse(p_sd[a], q_sd[a],
                                                          &next_lp)
                     : target_view.sample_residual(p_row, q_dense(a),
                                                   &next_lp);
        } else if (smg_v) {
          sms_prep(target_view, g, false, p_sd[g]);
          next = target_view.draw_sparse(p_sd[g], &next_lp, false);
        } else {
          std::copy_n(verify_logits.data() + (size_t)g * g_cfg.vocab,
                      g_cfg.vocab, p_row.data());
          target_view.prepare(p_row);
          next = target_view.draw_prepared(p_row, &next_lp, false);
        }
        sampler.rng = target_view.rng;
        for (int j = 0; j < g; j++) {
          depth_tot[j]++;
          if (j < a) depth_hits[j]++;
        }
        if (g > 0) gctl.update(g, a);  // γ = 0 rounds carry no draft evidence
        occ.update(g, a);
        ++mtp_rounds;
        last_spec_proposed += g;
        int d = a, stopped_token = -1;
        bool keep_next = true;
        rp_lap(RP_JUDGE);
        if (emit) {
          for (int j = 0; j < a; j++)
            if (!emit(drafts[j], draft_lps[j])) {
              d = j;
              stopped_token = drafts[j];
              stop = true;
              keep_next = false;
              break;
            }
          if (!stop && !emit(next, next_lp)) { stop = true; keep_next = false; }
        }
        rp_lap(RP_EMIT);
        for (int j = 0; j < d; j++) {
          out.push_back(drafts[j]);
          sampler.remember(drafts[j]);
        }
        if (keep_next) {
          out.push_back(next);
          sampler.remember(next);
        }
        accepted += d;
        if (!stop && d == g) {
          gdn_commit(g + 1);  // GDN S: replay all rows
          cur = next;  // full accept; state at q+1+g
        } else {
          ++mtp_roll;
          // Install the state at commit depth d from the verify snapshots —
          // identical to spec_loop_sample's rollback.
          int ngdn = g_cfg.layers - g_cfg.layers / 4;
          gdn_commit(d + 1);  // S after the committed rows (replay)
          CK(hipMemcpyAsync(d_convst, d_csnap + (size_t)(vsnap_row0_ + d) * ngdn * 3 * 10240,
                            (size_t)ngdn * 3 * 10240 * 4,
                            hipMemcpyDeviceToDevice, g_str));
          if (ckpt_iraw)
            k_iraw_restore<<<g_cfg.layers / 4, 128, 0, g_str>>>(
                d_iproj_snap + (size_t)vsnap_row0_ * 128,
                d_ckpt + (ckpt_S + ckpt_convst + ckpt_ring) / 4,
                d_iraw, q + 1, q + d + 1, kVSnapRows);
          const float* ckring = d_ckpt + (ckpt_S + ckpt_convst) / 4;
          for (int p = q + d + 2; p <= q + g + 1; p++)
            CK(hipMemcpyAsync(d_ring + (size_t)(p % 9) * 10240,
                              ckring + (size_t)(p % 9) * 10240, 10240 * 4,
                              hipMemcpyDeviceToDevice, g_str));
          pos = q + d + 2;
          ple_ringpos = pos % 9;
          CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
          CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice,
                            g_str));
          restore_dR_row(d, vsnap_row0_);  // row base 0 unless pooled
          cur = stopped_token >= 0 ? stopped_token : next;
        }
        // Committed-token history: rebuild explicitly so it is right whether
        // or not ple_on made the verify append (as spec_loop_chain).
        chist.resize(chist_sz);
        chist.insert(chist.end(), vt.begin(), vt.begin() + d + 1);
        // MTP self-repair over committed positions q+1..q+d.
        CK(hipMemcpyAsync(d_miraw, d_mring_ckpt, 512 * 4,
                          hipMemcpyDeviceToDevice, g_str));
        if (d > 0) {
          std::vector<int> mtoks(drafts.begin(), drafts.begin() + d);
          mtp_ingest_b(spec_taps(d, vsnap_row0_), mtoks, d, q + 1);
        } else {
          mtp_pos = q + 1;
          CK(hipMemcpyAsync(d_mpos, &mtp_pos, 4, hipMemcpyHostToDevice, g_str));
        }
        vsnap_row0_ = 0;
        rp_lap(RP_COMMIT);
        if (g_sp_on)
          rp[RP_WALL] += std::chrono::duration<double, std::milli>(sclk::now() - rp_top).count();
      } else {
        // ---- no draft, no MTP: plain serial sampling step ----
        VrExcuseS vr_x(*this, vr_on);
        DrExcuseS dr_x(*this, dr_on);
        ++fallback;
        forward(cur);
        if (!ple_on) chist.push_back(cur);
        ple_ringpos = pos % 9;  // graph advances the device scalar
        get_logits(p_row);
        float lp_s = 0.0f;
        int nx = sampler.sample(p_row, &lp_s);
        if (emit && !emit(nx, lp_s)) break;
        out.push_back(nx);
        cur = nx;
        continue;  // serial steps are not speculation rounds
      }
      ++last_spec_rounds;
      live_spec_rounds = last_spec_rounds;
      live_spec_commits = (int)out.size();
      live_spec_proposed = last_spec_proposed;
      if (stop) break;
    }
    double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - T0)
            .count();
    last_spec_commit = (int)out.size();
    last_spec_rollbacks = ng_roll + mtp_roll;
    fprintf(stderr,
            "%s chain-sample: %d tokens in %.2f s = %.1f tok/s | rounds=%d "
            "(ngram=%d mtp=%d serial=%d) commit/round=%.2f accepted=%d "
            "proposed=%d rollbacks=%d\n",
            log_ts(), (int)out.size(), sec, sec > 0 ? out.size() / sec : 0.0,
            last_spec_rounds, ng_rounds, mtp_rounds, fallback,
            last_spec_rounds ? (double)out.size() / last_spec_rounds : 0.0,
            accepted, last_spec_proposed, last_spec_rollbacks);
    if (ng_rounds)
      fprintf(stderr,
              "chain-sample ngram: rounds=%d accepted=%d proposed=%d "
              "time=%.0f ms (%.1f ms/round)\n",
              ng_rounds, ng_keep, ng_prop, ng_ms, ng_ms / ng_rounds);
    if (mtp_rounds) {
      fprintf(stderr, "chain-sample mtp depth acc:");
      for (int j = 0; j < 8 && depth_tot[j]; j++)
        fprintf(stderr, " %d:%.3f", j + 1,
                (double)depth_hits[j] / depth_tot[j]);
      fprintf(stderr, "\n");
      gctl.report("chain-sample");
      occ.report("chain-sample");
    }
    if (g_sp_on && mtp_rounds) {
      static const char* const nm[RP_N] = {"yield", "top", "pre", "draft", "verify", "vpool",
                                           "judge", "emit", "commit", "wall"};
      double seg = 0;
      for (int k = 0; k < RP_WALL; k++) seg += rp[k];
      fprintf(stderr, "[roundprof] chain-sample mtp %d rounds, ms/round:", mtp_rounds);
      for (int k = 0; k < RP_N; k++) fprintf(stderr, " %s %.2f", nm[k], rp[k] / mtp_rounds);
      fprintf(stderr, " | other %.2f | γ avg %.2f, commit/round %.2f\n",
              (rp[RP_WALL] - seg) / mtp_rounds, (double)last_spec_proposed / std::max(1, last_spec_rounds),
              last_spec_rounds ? (double)out.size() / last_spec_rounds : 0.0);
      g_sp.report("chain-sample", last_spec_rounds);
    }
    return out;
  }

  // --- decode graphs: G1 = token upload + embed + L0; G2 = [PLE] + L1..end --
  // (g1_exec/g2_exec are per-slot, declared with the slot state above)
  bool graphs_ok = false;
  void build_graphs() {
    viz::ensure(bound_);  // QWENOX_VIZ: the ring pointer is baked into the graphs
    if (dbg || perf || getenv("QWENOX_NOGRAPH")) return;
    // pre-populate lazy caches (hipMalloc/hipMemcpy are illegal during capture)
    for (int l = 0; l < g_cfg.layers; l++) {
      char p[128];
      snprintf(p, sizeof p, "layers.%d.attn_hyper_connection", l);
      ensure_hcnorm(p);
      snprintf(p, sizeof p, "layers.%d.mlp_hyper_connection", l);
      ensure_hcnorm(p);
      if (is_qsa(l)) {
        snprintf(p, sizeof p, "layers.%d.self_attn.q_norm.weight", l);
        ensure_norm256(p);
        snprintf(p, sizeof p, "layers.%d.self_attn.k_norm.weight", l);
        ensure_norm256(p);
      }
    }
    ensure_hcnorm("hyper_connection_mixer");
    hipStream_t cap;
    CK(hipStreamCreate(&cap));
    g_str = cap;
    g_kprof_mute = true;  // hipEventRecord 进捕获流会把 event 录进图里
    hipGraph_t g1, g2;
    CK(hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal));
    CK(hipMemcpyAsync(d_token, p_token, 4, hipMemcpyHostToDevice, cap));
    enq_embed();
    enq_layer(0);
    CK(hipStreamEndCapture(cap, &g1));
    CK(hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal));
    if (ple_on) ple_gpu();
    for (int l = 1; l < g_cfg.layers; l++) enq_layer(l);
    enq_final();
    k_step_incr<<<1, 1, 0, cap>>>(d_pos, d_ringpos);
    CK(hipStreamEndCapture(cap, &g2));
    g_str = nullptr;
    g_kprof_mute = false;
    CK(hipGraphInstantiate(&g1_exec, g1, nullptr, nullptr, 0));
    CK(hipGraphInstantiate(&g2_exec, g2, nullptr, nullptr, 0));
    vb_forget_stream(cap);
    CK(hipStreamDestroy(cap));
    graphs_ok = true;
    fprintf(stderr, "decode graphs captured (ple=%d)\n", (int)ple_on);
    crashdiag::note("decode graphs captured");
  }

  // ==================== P3-B: spec-verify segment graphs ====================
  // One graph set per batch size P (= gamma+1): seg 0 = pinned token upload +
  // embed + L0 + PLE device phase + L1..L2; then per QSA layer lq an eager
  // indexer (qsa_idx_b: base-dependent host grids) followed by a captured
  // segment with lq's layer body and the dense layers up to the next QSA; the
  // last segment also carries the d_R handoff, the HC mixer read, the batched
  // lm_head and argmax_rows. Every base-dependent write inside a captured
  // segment reads *d_vbase (P3-A/P3-B device base), so a replay with a new
  // base is bit-identical to the eager path: same kernels, same inputs, same
  // order — only the base scalar moved to device memory. The PLE gather, KV
  // page backing, indexer and the pos/argmax readback stay eager per round.
  static constexpr int kVGMaxP = 16;   // graphs for P in [2, 16]
  static constexpr int kVGMaxSeg = 16; // 1 + layers/4 QSA segments
  struct VGraph {
    hipGraphExec_t seg[kVGMaxSeg] = {};
    int nseg = 0;  // 0 = not built, -1 = capture failed (permanent fallback)
    // Host-side HC inject-partials pipeline state as the captured sequence
    // left it after each segment (graph replay runs no host code, so
    // vgraph_run restores this after each launch; otherwise the eager
    // inter-segment norm would misjudge have_p and recompute the inject
    // GEMM — same math, different summation order, different bits).
    int bnd_injp_cur[kVGMaxSeg] = {};
    std::string bnd_injp_for[kVGMaxSeg];
  };
  VGraph vg_[kVGMaxP - 1];
  int vg_warm_[kVGMaxP - 1] = {};  // eager rounds per P (descriptor warmup)
  bool vg_capturing_ = false;      // suppresses capture-illegal side ops
  int* p_vgtok = nullptr;          // pinned token staging for seg 0's H2D node
  int* p_vgbase = nullptr;         // pinned base staging for the d_vbase write

  bool vgraph_ok(int P, int base) {
    static const bool off = env_zero("QWENOX_VGRAPH");
    if (off || !vbase_on()) return false;
    if ((dbg && !getenv("QWENOX_VGRAPH_DBG")) || perf || g_kprof_on)
      return false;
    if (P < 2 || P > kVGMaxP || base < 2051 || base + P > maxctx) return false;
    if (mrope_active_ || cut_abs >= 0 || qsa_dense || qsa_loop || qsa_wmma6)
      return false;
    if (ple_on && ple_loop) return false;
    if (ngram_verify_active || ws_lent_ || prog_total > 0) return false;
    // graphs bake this slot's pointers (d_ptab, d_R, ...): single-slot only
    if (nslots > 1) return false;
    if (injp_cur >= 0 || !hcw_site_.empty()) return false;
    if (getenv("QWENOX_VCT_CHECK") || getenv("QWENOX_ATTN_HASH") ||
        getenv("QWENOX_VCT_TAILZERO"))
      return false;
    // D1b mid-chunk GPU lending is not part of the graphs (production has
    // moe_lt on, which already makes the yield a no-op).
    if (ws_ok_ && borrow_query_fn && !moe_lt && P > conc_rmax_) return false;
    return true;
  }

  // Capture the segments for P (kernels do not execute while capturing; the
  // interleaved eager indexer runs harmlessly on scratch). kv_prepare runs
  // first so the eager indexer sees a backed page table. On any capture error
  // the set is discarded and the caller falls back to the eager round.
  bool vgraph_capture(int P, int base) {
    VGraph& G = vg_[P - 2];
    if (!p_vgtok) {
      if (hipHostMalloc((void**)&p_vgtok, kVGMaxP * 4, hipHostMallocDefault) !=
              hipSuccess ||
          hipHostMalloc((void**)&p_vgbase, 4, hipHostMallocDefault) != hipSuccess)
        return false;
    }
    kv_prepare(base, base + P);
    // The eager boundary ops below (norm + indexer) execute for real during
    // capture, on stale inputs. Give them the same device base the replay
    // will use so their pooled-block/ring write footprint matches the
    // replay's exactly (the replay then rewrites all of it)...
    set_vbase(base);
    // ...except the raw indexer ring: its 4 rows are fully overwritten by
    // the capture-phase indexer, while the replay's pooling reads a history
    // row (base%4 != 0 straddle) written before this round. Shield it.
    static float* d_iraw_sav = nullptr;
    static size_t iraw_sav_sz = 0;
    const size_t iraw_sz = (size_t)(g_cfg.layers / 4) * 4 * 128 * 4;
    if (d_iraw) {
      if (iraw_sav_sz < iraw_sz) {
        if (d_iraw_sav) (void)hipFree(d_iraw_sav);
        if (hipMalloc(&d_iraw_sav, iraw_sz) != hipSuccess) return false;
        iraw_sav_sz = iraw_sz;
      }
      CK(hipMemcpyAsync(d_iraw_sav, d_iraw, iraw_sz, hipMemcpyDeviceToDevice,
                        g_str));
    }
    hipStream_t cap = nullptr;
    if (hipStreamCreate(&cap) != hipSuccess) return false;
    bool ok = true;
    hipStream_t save = g_str;
    g_str = cap;
    g_kprof_mute = true;
    vg_capturing_ = true;
    auto end_seg = [&]() {
      hipGraph_t gr = nullptr;
      if (hipStreamEndCapture(cap, &gr) != hipSuccess || !gr) return false;
      hipGraphExec_t ex = nullptr;
      const bool iok =
          hipGraphInstantiate(&ex, gr, nullptr, nullptr, 0) == hipSuccess;
      (void)hipGraphDestroy(gr);
      if (!iok) return false;
      G.seg[G.nseg++] = ex;
      return true;
    };
    if (hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal) != hipSuccess)
      ok = false;
    if (ok) {  // seg 0: token upload + embed + L0 + PLE + L1..first QSA-1
      // QWENOX_VGRAPH_NOPLE / QWENOX_VGRAPH_NOGDN / QWENOX_VGRAPH_NOEMB (debug
      // bisect): run that part eagerly in vgraph_run instead of capturing it.
      static const bool s0_ple = !getenv("QWENOX_VGRAPH_NOPLE");
      static const bool s0_gdn = !getenv("QWENOX_VGRAPH_NOGDN");
      static const bool s0_emb = !getenv("QWENOX_VGRAPH_NOEMB");
      if (s0_emb)
        enq_embed_body(p_vgtok, P);
      else
        CK(hipMemcpyAsync(d_tokarr, p_vgtok, P * 4, hipMemcpyHostToDevice,
                          g_str));
      if (s0_gdn) enq_layer_b(0, P, false, true, base, d_vbase);
      if (ple_on && s0_ple) enq_ple_device_b(P, base, 0);
      if (s0_gdn)
        for (int l = 1; l < 3 && l < g_cfg.layers; l++) {
          const bool fuse_prev = gr_scatter_norm && l > 1;
          const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
          enq_layer_b(l, P, fuse_prev, flush_mlp, base, d_vbase);
        }
      ok = end_seg();
      if (ok) {  // host-side injp pipeline state this segment's capture left
        G.bnd_injp_cur[G.nseg - 1] = injp_cur;
        G.bnd_injp_for[G.nseg - 1] = injp_for;
      }
    }
    // QWENOX_VGRAPH_SEGS=k (debug bisect): capture only the first k segments;
    // vgraph_run finishes the remaining layers eagerly (host base).
    static const int seg_limit =
        getenv("QWENOX_VGRAPH_SEGS") ? atoi(getenv("QWENOX_VGRAPH_SEGS")) : 999;
    for (int lq = 3; ok && lq < g_cfg.layers && G.nseg < seg_limit; lq += 4) {
      // The indexer reads d_xb, written by the layer's attention norm: run
      // the norm eagerly too, ahead of the indexer, outside the graph.
      enq_layer_norm_b(lq, P, gr_scatter_norm && lq > 1);
      qsa_idx_b(lq, lq / 4, P, base);
      if (hipStreamBeginCapture(cap, hipStreamCaptureModeGlobal) != hipSuccess) {
        ok = false;
        break;
      }
      const int lend = std::min(lq + 4, (int)g_cfg.layers);
      for (int l = lq; l < lend; l++) {
        const bool fuse_prev = gr_scatter_norm && l > 1;
        const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
        enq_layer_b(l, P, fuse_prev, flush_mlp, base, d_vbase, l == lq);
      }
      if (lend == g_cfg.layers) {  // final segment: handoff + logits + argmax
        if (gr_bf16)
          k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, cap>>>(
              (const uint16_t*)d_Rb + (size_t)(P - 1) * 10240, d_R, 10240);
        else
          CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(P - 1) * 10240, 10240 * 4,
                            hipMemcpyDeviceToDevice, cap));
        batch_logits_device(P);
        argmax_rows(d_logits_b, g_cfg.vocab, P, d_amx_cand, d_argmax_arr, cap);
      }
      ok = end_seg();
      if (ok) {  // host-side injp pipeline state this segment's capture left
        G.bnd_injp_cur[G.nseg - 1] = injp_cur;
        G.bnd_injp_for[G.nseg - 1] = injp_for;
      }
    }
    g_str = save;
    g_kprof_mute = false;
    vg_capturing_ = false;
    // capture-phase eager boundary writes landed on cap; order the restore
    (void)hipStreamSynchronize(cap);
    vb_forget_stream(cap);
    (void)hipStreamDestroy(cap);
    if (d_iraw && iraw_sav_sz)
      CK(hipMemcpyAsync(d_iraw, d_iraw_sav, iraw_sav_sz, hipMemcpyDeviceToDevice,
                        g_str));
    if (!ok || G.nseg == 0) {
      for (int s = 0; s < G.nseg; s++) (void)hipGraphExecDestroy(G.seg[s]);
      G.nseg = 0;
      (void)hipGetLastError();  // clear any capture error
      return false;
    }
    return true;
  }

  // One spec-verify round on the graphs: eager head (page backing, device
  // base, PLE gather, token staging), segment replays interleaved with the
  // eager per-QSA-layer indexer, eager tail (position mirrors + blocking
  // argmax readback). Matches prefill_batch(final=false) + batch_argmax.
  void vgraph_run(const std::vector<int>& tokens, int base, int* vout) {
    const int P = (int)tokens.size();
    VGraph& G = vg_[P - 2];
    if (d_iraw) vg_dump_buf("irH0", 2, d_iraw, 512 * 4);  // P3-B bisect
    kv_prepare(base, base + P);
    *p_vgbase = base;
    CK(hipMemcpyAsync(d_vbase, p_vgbase, 4, hipMemcpyHostToDevice, g_str));
    vz_base_ = base;  // QWENOX_VIZ: eager parts below tag rows like the replays
    vz_dvb_ = nullptr;
    vz_rowtab_ = nullptr;
    vz_rows_ = true;
    xf16_src = nullptr;  // a new batch invalidates the bf16 conversion cache
    if (ple_on) {
      ple_drop_all();  // graph verify: one flight, fixed staging buffer 0
      int64_t pre0, pre1;
      ple_pre01(chist, pre0, pre1);
      // fixed staging buffer 0: seg 0's dequant reads ple_eb_dev
      ple_spawn(ple_cur(), std::vector<int>(tokens), base, pre0, pre1,
                (uint8_t*)ple_eb, 0);
      for (int t : tokens) chist.push_back(t);
      ple_join(ple_cur());
      ple_cur().valid = false;
      // replay consumed staging buffer 0, the next gather writes 1
      ple_par = 1;
    }
    memcpy(p_vgtok, tokens.data(), P * 4);
    if (d_iraw) vg_dump_buf("irH1", 2, d_iraw, 512 * 4);  // P3-B bisect
    CK(hipGraphLaunch(G.seg[0], g_str));
    // Graph replay runs no host code: restore the HC inject-partials pipeline
    // state the capture left after this segment, so the eager code below
    // branches exactly as the fully-eager reference would.
    injp_cur = G.bnd_injp_cur[0];
    injp_for = G.bnd_injp_for[0];
    if (d_iraw) vg_dump_buf("irH2", 2, d_iraw, 512 * 4);  // P3-B bisect
    if (getenv("QWENOX_VGDUMP")) {
      CK(hipStreamSynchronize(g_str));
      int tk[16] = {};
      int nd = P < 16 ? P : 16;
      CK(hipMemcpy(tk, d_tokarr, (size_t)nd * 4, hipMemcpyDeviceToHost));
      fprintf(stderr, "[vgr] P=%d base=%d tok=%d,%d,%d,%d\n", P, base, tk[0],
              P > 1 ? tk[1] : -1, P > 2 ? tk[2] : -1, P > 3 ? tk[3] : -1);
      vg_dump_row("vgs", P, base);
    }
    {  // eager fallback for parts excluded from seg 0 (bisect envs)
      static const bool s0_ple = !getenv("QWENOX_VGRAPH_NOPLE");
      static const bool s0_gdn = !getenv("QWENOX_VGRAPH_NOGDN");
      static const bool s0_emb = !getenv("QWENOX_VGRAPH_NOEMB");
      if (!s0_emb) enq_embed_body(tokens.data(), P);  // H2D repeats, same value
      if (!s0_gdn) enq_layer_b(0, P, false, true, base);
      if (ple_on && !s0_ple) enq_ple_device_b(P, base, 0);
      if (!s0_gdn)
        for (int l = 1; l < 3 && l < g_cfg.layers; l++) {
          const bool fuse_prev = gr_scatter_norm && l > 1;
          const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
          enq_layer_b(l, P, fuse_prev, flush_mlp, base);
        }
    }
    int s = 1, lq = 3;
    for (; lq < g_cfg.layers && s < G.nseg; lq += 4) {
      enq_layer_norm_b(lq, P, gr_scatter_norm && lq > 1);
      qsa_idx_b(lq, lq / 4, P, base);
      CK(hipGraphLaunch(G.seg[s], g_str));
      injp_cur = G.bnd_injp_cur[s];  // see the restore after seg[0]
      injp_for = G.bnd_injp_for[s];
      s++;
      if (getenv("QWENOX_VGDUMP")) {
        char tag[8];
        snprintf(tag, sizeof tag, "vgS%d", s - 1);
        vg_dump_row(tag, P, base);
      }
    }
    if (lq < g_cfg.layers) {  // QWENOX_VGRAPH_SEGS bisect: eager remainder
      for (int l = lq; l < g_cfg.layers; l++) {
        if (is_qsa(l)) {
          enq_layer_norm_b(l, P, gr_scatter_norm && l > 1);
          qsa_idx_b(l, l / 4, P, base);
        }
        const bool fuse_prev = gr_scatter_norm && l > 1;
        const bool flush_mlp = !gr_scatter_norm || l == g_cfg.layers - 1;
        enq_layer_b(l, P, fuse_prev, flush_mlp, base, nullptr, is_qsa(l));
        {
          char tag[8];
          snprintf(tag, sizeof tag, "rL%d", l);
          vg_dump_row(tag, P, base);
        }
      }
      if (gr_bf16)
        k_bf16_to_f32<<<(10240 + 255) / 256, 256, 0, g_str>>>(
            (const uint16_t*)d_Rb + (size_t)(P - 1) * 10240, d_R, 10240);
      else
        CK(hipMemcpyAsync(d_R, d_Rb + (size_t)(P - 1) * 10240, 10240 * 4,
                          hipMemcpyDeviceToDevice, g_str));
      batch_logits_device(P);
      argmax_rows(d_logits_b, g_cfg.vocab, P, d_amx_cand, d_argmax_arr, g_str);
    }
    pos = base + P;
    ple_ringpos = (base + P) % 9;
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpy(vout, d_argmax_arr, (size_t)P * 4, hipMemcpyDeviceToHost));
  }

  // verify_graph_round_try: graph fast path for MTP verify; false -> caller
  // runs the legacy eager sequence. The first qualifying round per P stays
  // eager (warms the P-shaped GEMM/LT descriptors), the second captures.
  bool verify_graph_round_try(const std::vector<int>& tokens, int base,
                              int* vout) {
    const int P = (int)tokens.size();
    if (!vgraph_ok(P, base)) return false;
    VGraph& G = vg_[P - 2];
    if (G.nseg < 0) return false;
    if (!G.nseg) {
      if (vg_warm_[P - 2]++ < 1) return false;
      if (!vgraph_capture(P, base)) {
        G.nseg = -1;
        fprintf(stderr, "verify graphs: capture failed for P=%d, staying eager\n",
                P);
        return false;
      }
      fprintf(stderr, "verify graphs captured (P=%d, %d segments)\n", P, G.nseg);
      crashdiag::note("verify graphs captured P=%d base=%d", P, base);
      // QWENOX_VGRAPH_NOLAUNCH (debug): capture once, then stay eager forever —
      // isolates capture-time side effects from replay effects.
      if (getenv("QWENOX_VGRAPH_NOLAUNCH")) {
        for (int s = 0; s < G.nseg; s++) (void)hipGraphExecDestroy(G.seg[s]);
        G.nseg = -1;
        return false;
      }
    }
    vgraph_run(tokens, base, vout);
    return true;
  }

  // ---- D1b borrow workspace ----------------------------------------------
  // Second, small copy of every maxbatch_cap-sized row buffer a borrower can
  // reach: spec verify runs the full prefill_chunk path at P <= conc_rmax_
  // and mtp_step touches row 0 of d_mRb/d_xbf16/d_eb/d_xb/d_Gb/d_t320b/
  // d_qnb/d_Unb/d_convoutb/d_tokarr; serial decode stays on the P=1 buffers
  // and its graphs bake only those (never BP-sized ones). Buffers provably
  // dead at a layer boundary stay shared: d_wbf16, d_gdn_ws, d_gdn_split_ws,
  // d_logits_b, d_eoff, d_gg_uq, d_iscores tiles are row-windowed... (see
  // HANDOFF-CONCURRENCY §5.2 audit), d_posarr/d_ringarr (constant).
  // d_moetiles/d_moentiles/moe_routing/moe_lt/qsa_union paths are never taken
  // at P <= conc_rmax_; conc_borrow_yield refuses to lend when they are on.
#define QWENOX_WS_FIELDS(X)                                                     \
  X(d_Rb) X(d_Rhatb) X(d_Rhatbf16) X(d_hcsc) X(d_xb) X(d_yb) X(d_Gb)          \
  X(d_t320b)                                                                  \
  X(d_w4b) X(d_qkvb) X(d_zb) X(d_g48b) X(d_beta48b) X(d_qgb) X(d_qsb)        \
  X(d_gsb) X(d_kbb) X(d_vbb) X(d_ropecs) X(d_attnb) X(d_routerb)             \
  X(d_topidsb) X(d_topwb) X(d_slot_order) X(d_guvb) X(d_hidb) X(d_moeaccb)   \
  X(d_eyb) X(d_sgb) X(d_eb) X(d_xbf16) X(d_xbf16b) X(d_tokidx) X(d_pweight)  \
  X(d_pairids) X(d_convb) X(d_keysb) X(d_plevb) X(d_qnb) X(d_Ub) X(d_Unb)    \
  X(d_convoutb) X(d_tokarr) X(d_img_inj) X(d_iproj) X(d_iq) X(d_iqh) X(d_iselected)   \
  X(d_iscores) X(d_mRb) X(d_mRhatb) X(d_mtap)
  struct WsPtrs {
#define QWENOX_WS_DECL(f) decltype(GpuModel::f) f = nullptr;
    QWENOX_WS_FIELDS(QWENOX_WS_DECL)
#undef QWENOX_WS_DECL
    float *d_injp0 = nullptr, *d_injp1 = nullptr;
    float* ple_eb = nullptr;
    uint8_t* ple_eb_dev = nullptr;
  };
  WsPtrs ws_small_, ws_saved_;
  bool ws_lent_ = false, ws_ok_ = false;
  hipEvent_t conc_ev_a_ = nullptr, conc_ev_b_ = nullptr;
  long conc_lends_ = 0;  // D1b diagnostics: total lends since init

  // Mirror of the constructor's BP-sized block at R rows. Each field is
  // allocated only when the big counterpart exists (same feature flags), so
  // "small == nullptr" always implies "big == nullptr" and the lend/takeback
  // guards below never drop a live buffer.
  void ws_alloc_small(int R) {
    if (ws_ok_ || R < 1) return;
    const size_t B = (size_t)R;
#define QWENOX_WS_DUP(p, bytes) \
    if (p) DALLOC(&ws_small_.p, (bytes))
    QWENOX_WS_DUP(d_Rb, B * 10240 * 4);
    QWENOX_WS_DUP(d_Rhatb, B * 10240 * 4);
    QWENOX_WS_DUP(d_Rhatbf16, B * 10240 * 2);
    QWENOX_WS_DUP(d_hcsc, B * 4 * 4);
    QWENOX_WS_DUP(d_xb, B * 2560 * 4);
    QWENOX_WS_DUP(d_yb, B * 2560 * 4);
    QWENOX_WS_DUP(d_Gb, B * 10240 * 4);
    QWENOX_WS_DUP(d_t320b, B * 320 * 4);
    QWENOX_WS_DUP(d_w4b, B * 4 * 4);
    QWENOX_WS_DUP(d_qkvb, B * 10240 * 4);
    QWENOX_WS_DUP(d_zb, B * 6144 * 4);
    QWENOX_WS_DUP(d_g48b, B * 48 * 4);
    QWENOX_WS_DUP(d_beta48b, B * 48 * 4);
    QWENOX_WS_DUP(d_qgb, B * 12288 * 4);
    QWENOX_WS_DUP(d_qsb, B * 6144 * 4);
    QWENOX_WS_DUP(d_gsb, B * 6144 * 4);
    QWENOX_WS_DUP(d_kbb, B * 512 * 4);
    QWENOX_WS_DUP(d_vbb, B * 512 * 4);
    QWENOX_WS_DUP(d_ropecs, B * 32 * sizeof(float2));
    QWENOX_WS_DUP(d_attnb, B * 6144 * 4);
    QWENOX_WS_DUP(d_routerb, B * 512 * 4);
    QWENOX_WS_DUP(d_topidsb, B * 16 * 4);
    QWENOX_WS_DUP(d_topwb, B * 16 * 4);
    QWENOX_WS_DUP(d_slot_order, 256 * 4);
    QWENOX_WS_DUP(d_guvb, B * g_cfg.topk * (moe_deterministic ? 2560 : 1280) * 4);
    QWENOX_WS_DUP(d_hidb, B * g_cfg.topk * 640 * 4);
    QWENOX_WS_DUP(d_moeaccb, B * 2560 * 4);
    QWENOX_WS_DUP(d_eyb, B * 2560 * 4);
    QWENOX_WS_DUP(d_sgb, B * 4);
    QWENOX_WS_DUP(d_eb, B * 2560 * 4);
    QWENOX_WS_DUP(d_xbf16, B * 10240 * 2);
    QWENOX_WS_DUP(d_xbf16b, B * 6144 * 2);
    QWENOX_WS_DUP(d_tokidx, B * g_cfg.topk * 4);
    QWENOX_WS_DUP(d_pweight, B * g_cfg.topk * 4);
    QWENOX_WS_DUP(d_pairids, B * g_cfg.topk * 4);
    QWENOX_WS_DUP(d_convb, B * 10240 * 4);
    QWENOX_WS_DUP(d_keysb, B * 10240 * 4);
    QWENOX_WS_DUP(d_plevb, B * 2560 * 4);
    QWENOX_WS_DUP(d_qnb, B * 10240 * 4);
    QWENOX_WS_DUP(d_Ub, B * 10240 * 4);
    QWENOX_WS_DUP(d_Unb, B * 10240 * 4);
    QWENOX_WS_DUP(d_convoutb, B * 10240 * 4);
    QWENOX_WS_DUP(d_tokarr, B * 4);
    QWENOX_WS_DUP(d_img_inj, B * sizeof(int2));
    if (d_iproj) {  // !qsa_dense indexer block
      DALLOC(&ws_small_.d_iproj, B * 640 * 4);
      DALLOC(&ws_small_.d_iq, B * 512 * 4);
      if (d_iqh) DALLOC(&ws_small_.d_iqh, B * 512 * 2);
      DALLOC(&ws_small_.d_iselected, B * INDEX_BUDGET * 4);
      DALLOC(&ws_small_.d_iscores,
             (size_t)std::min(R, INDEX_BATCH) * 4 * index_blocks * 4);
    }
    QWENOX_WS_DUP(d_mRb, B * 10240 * 4);
    QWENOX_WS_DUP(d_mRhatb, B * 10240 * 4);
    QWENOX_WS_DUP(d_mtap, B * 10240 * 4);
#undef QWENOX_WS_DUP
    if (d_injp[0]) {
      DALLOC(&ws_small_.d_injp0, B * 16 * 4);
      DALLOC(&ws_small_.d_injp1, B * 16 * 4);
    }
    if (ple_eb) {
      CK(hipHostMalloc((void**)&ws_small_.ple_eb, B * 2560 * 4,
                       hipHostMallocDefault));
      CK(hipHostGetDevicePointer((void**)&ws_small_.ple_eb_dev,
                                 ws_small_.ple_eb, 0));
    }
    CK(hipEventCreate(&conc_ev_a_));  // timing events: quantum clock
    CK(hipEventCreate(&conc_ev_b_));
    ws_ok_ = true;
  }

  void ws_lend() {  // members -> small set; big pointers saved in ws_saved_
#define QWENOX_WS_SAVE(f) \
    ws_saved_.f = f;    \
    if (ws_small_.f) f = ws_small_.f;
    QWENOX_WS_FIELDS(QWENOX_WS_SAVE)
#undef QWENOX_WS_SAVE
    ws_saved_.d_injp0 = d_injp[0];
    ws_saved_.d_injp1 = d_injp[1];
    if (ws_small_.d_injp0) {
      d_injp[0] = ws_small_.d_injp0;
      d_injp[1] = ws_small_.d_injp1;
    }
    ws_saved_.ple_eb = ple_eb;
    ws_saved_.ple_eb_dev = ple_eb_dev;
    if (ws_small_.ple_eb) {
      ple_eb = ws_small_.ple_eb;
      ple_eb_dev = ws_small_.ple_eb_dev;
    }
    ws_lent_ = true;
  }
  void ws_takeback() {  // members -> big set (exact inverse of ws_lend)
#define QWENOX_WS_REST(f) \
    if (ws_small_.f) f = ws_saved_.f;
    QWENOX_WS_FIELDS(QWENOX_WS_REST)
#undef QWENOX_WS_REST
    if (ws_small_.d_injp0) {
      d_injp[0] = ws_saved_.d_injp0;
      d_injp[1] = ws_saved_.d_injp1;
    }
    if (ws_small_.ple_eb) {
      ple_eb = ws_saved_.ple_eb;
      ple_eb_dev = ws_saved_.ple_eb_dev;
    }
    ws_lent_ = false;
  }

  // Layer-boundary yield inside a big prefill chunk. Nobody queued: one
  // atomic read. Contended: drain the layers enqueued so far (event sync),
  // and if the GPU-busy quantum elapsed, lend the small workspace and block
  // in borrow_fn until the turnstile comes back. The chunk's live state
  // (residual stream in the big set, per-slot fields, KV pages) is untouched
  // by the borrower; chunk-level member state is restored explicitly.
  void conc_borrow_yield(int base, int P, int layer) {
    if (!ws_ok_ || !borrow_query_fn || ws_lent_ || P <= conc_rmax_) return;
    // moe_lt: unduplicated scratch live across layers. qsa_union's
    // d_ublk/d_umask/d_ucount are written and consumed inside one qsa_flash_b
    // call -> dead at this layer boundary, so a borrower may reuse them
    // (aggr 10-05: union is default-on; refusing here would disable D1b).
    if (moe_lt) return;
    if (!borrow_query_fn()) return;
    CK(hipEventRecord(conc_ev_b_, g_str));
    CK(hipEventSynchronize(conc_ev_b_));  // GPU drained through this layer
    float gpu_ms = 0;
    CK(hipEventElapsedTime(&gpu_ms, conc_ev_a_, conc_ev_b_));
    if (gpu_ms < conc_quantum_ms_) return;
    fprintf(stderr, "%s [conc] prefill lends GPU at L%d (base=%d, %.0f ms since last)\n",
            log_ts(), layer, base, (double)gpu_ms);
    conc_lends_++;
    // HC inject partials pipeline (injp_cur/injp_for -> big d_injp[]) is live
    // across this layer boundary: park it, hand the borrower an empty
    // pipeline, restore on return. The partial sums themselves sit in the
    // big d_injp buffers, which the borrower never touches (small set).
    const int saved_injp_cur = injp_cur;
    const std::string saved_injp_for = injp_for;
    injp_cur = -1;
    injp_for.clear();
    ws_lend();
    // queue back as heavy (rest of this prompt) so the lend's borrowers see it
    heavy_rem_ = pf_end > 0 ? std::max(pf_end - base, P) : P;
    try {
      borrow_fn(*this);  // flag up; release; light borrowers run; reacquire; bind
    } catch (...) {
      ws_takeback();  // aborted mid-chunk: never leave members on the small set
      injp_cur = saved_injp_cur;
      injp_for = saved_injp_for;
      throw;
    }
    ws_takeback();
    injp_cur = saved_injp_cur;
    injp_for = saved_injp_for;
    set_vbase(base);    // the borrower's set_vbase rotated the ring
    ropecs_base_ = -1;  // rope table and bf16 cache below are the borrower's
    xf16_src = nullptr;
    CK(hipEventRecord(conc_ev_a_, g_str));  // restart the quantum clock
  }

  // Allocate slots 1..n-1 (slot 0 is the constructor's state) and capture
  // their decode graphs. Paged KV only: all slots map pages from one pool.
  void slots_init(int n) {
    if (n <= 1) return;
    if (!kv_paged) {
      fprintf(stderr, "qwenox: QWENOX_PARALLEL=%d needs paged KV (QWENOX_KV_PAGED); "
                      "running 1 slot\n", n);
      return;
    }
    const size_t rb = (size_t)g_cfg.branches * g_cfg.d;
    const size_t ngdn = g_cfg.layers - g_cfg.layers / 4, nqsa = g_cfg.layers / 4;
    const size_t sS = ngdn * 48 * 128 * 128 * 4, sC = ngdn * 3 * 10240 * 4,
                 sRing = (size_t)9 * 10240 * 4, sIraw = nqsa * 4 * 128 * 4;
    slots_.resize((size_t)n);
    if (mtp_avail) {
      // S3c Stage 1: per-slot MTP 流（d_mR 加入 SLOT_FIELDS，bind 时换指针）。
      // 连续池 [n][rb] 而非分散分配：Stage 2 的跨槽合批 draft 按行直读本池。
      // 单槽路径（slots_init 不运行）仍用构造器的独立 d_mR。
      DALLOC(&d_mRpool, (size_t)n * rb * 4);
      d_mR = d_mRpool;  // 绑定槽 0 用第 0 行；构造器那份（rb*4）弃置
    }
    for (int i = 1; i < n; i++) {
      SeqState& t = slots_[(size_t)i];
      DALLOC(&t.d_R, rb * 4);
      DALLOC(&t.d_S, sS);
      DALLOC(&t.d_convst, sC);
      DALLOC(&t.d_ring, sRing);
      DALLOC(&t.d_pos, 4);
      DALLOC(&t.d_ringpos, 4);
      DALLOC(&t.d_rdelta, 4);
      CK(hipMemset(t.d_R, 0, rb * 4));
      CK(hipMemset(t.d_S, 0, sS));
      CK(hipMemset(t.d_convst, 0, sC));
      CK(hipMemset(t.d_ring, 0, sRing));
      CK(hipMemset(t.d_pos, 0, 4));
      CK(hipMemset(t.d_ringpos, 0, 4));
      CK(hipMemset(t.d_rdelta, 0, 4));
      if (d_iraw) {
        DALLOC(&t.d_iraw, sIraw);
        CK(hipMemset(t.d_iraw, 0, sIraw));
      }
      // Verify rollback checkpoints: write-before-read scratch, no memset
      // (same as slot 0's d_ckpt / d_mring_ckpt in the constructor).
      DALLOC(&t.d_ckpt, sS + sC + sRing + (ckpt_iraw ? sIraw : 0));
      t.kv_seq.init(kv_lpages, kv_pool.guard());
      DALLOC(&t.d_ptab, (size_t)kv_lpages * 4);
      CK(hipMemcpy(t.d_ptab, t.kv_seq.tab.data(), (size_t)kv_lpages * 4,
                   hipMemcpyHostToDevice));
      if (mtp_avail) {
        DALLOC(&t.d_mpos, 4);
        CK(hipMemset(t.d_mpos, 0, 4));
        t.d_mR = d_mRpool + (size_t)i * rb;  // S3c: 第 i 行
        // the MTP KV / keys live in the shared pool (layer nqsa, B0)
        if (d_miraw) {
          DALLOC(&t.d_miraw, 4 * 128 * 4);
          CK(hipMemset(t.d_miraw, 0, 4 * 128 * 4));
        }
        if (d_mring_ckpt) DALLOC(&t.d_mring_ckpt, 512 * 4);
      }
      t.prog_req = -1;
      t.cut_abs = -1;  // SeqState{} value-init loses the member's -1 default
      if (graphs_ok) {  // per-slot decode graphs (they bake the slot pointers)
        bind(i);
        build_graphs();
        bind(0);
      }
    }
    nslots = n;
    // 10-13: per-slot backup of the rendezvous verify's shared row window
    // (vr_rows_save/load; the verify pool only runs with MTP and without
    // QWENOX_NO_VR).
    const bool no_vr = getenv("QWENOX_NO_VR") && !env_zero("QWENOX_NO_VR");
    if (vr_restore_env() && !no_vr && mtp_avail && d_grow && d_csnap && d_iproj_snap) {
      const size_t b = vr_save_bytes(kVrSaveRows);
      vr_save_.assign((size_t)n, nullptr);
      for (int i = 0; i < n; i++) DALLOC(&vr_save_[(size_t)i], b);
      fprintf(stderr, "qwenox: verify rendezvous row backup on (%d slots x %.1f MB, "
                      "QWENOX_VR_RESTORE=0 off)\n", n, b / 1048576.0);
    }
    if (!env_zero("QWENOX_CONC_PREFILL")) {  // D1b: arm the borrow workspace
      if (const char* v = getenv("QWENOX_CONC_PREFILL_QUANTUM_MS"))
        conc_quantum_ms_ = (float)atof(v);
      if (const char* v = getenv("QWENOX_CONC_RMAX")) conc_rmax_ = atoi(v);
      // 小工作区行数下限 kVSnapRows：会合 verify 批（<=48 行）作为 borrower
      // 借用本工作区且不经 conc_rmax_ 闸门，env 调小也不能窄过它。
      ws_alloc_small(conc_rmax_ < (int)kVSnapRows ? (int)kVSnapRows : conc_rmax_);
      if (ws_ok_)
        fprintf(stderr, "qwenox: prefill borrow armed (R<=%d rows, quantum %.0f ms)\n",
                conc_rmax_, (double)conc_quantum_ms_);
    }
    fprintf(stderr, "qwenox: %d sequence slots (shared KV pool %d pages = %d tokens)\n",
            n, kv_pool_pages, kv_pool_pages * KV_PAGE);
  }
  // Pages the slot's live sequence maps (bound slot: the members).
  int slot_pages(int s) const {
    return s == bound_ ? kv_seq.mapped : slots_[(size_t)s].kv_seq.mapped;
  }
  // Drop slot s's sequence (returns its pages to the pool) while `back`
  // stays the slot the caller runs on.
  void slot_reset(int s, int back) {
    bind(s);
    reset_state();
    mtp_tap_capture = spec_snap = ngram_verify_active = false;
    prog_total = 0;
    bind(back);
  }

  // Zero all cross-request state so the next request behaves like a fresh
  // process (the constructor memsets d_S/d_convst; hipMalloc's fresh zero
  // pages cover d_ring — reproduce both here).
  void reset_state() {
    ple_drop_all();  // prefetches belong to the sequence being reset
    pos = 0;
    ple_ringpos = 0;
    chist.clear();
    xf16_src = nullptr;  // invalidate the prefill bf16 conversion cache
    clear_mrope();       // drop any M-RoPE request state (d_rdelta <- 0)
    int ngdn = g_cfg.layers - g_cfg.layers / 4;
    CK(hipMemsetAsync(d_pos, 0, 4, 0));
    CK(hipMemsetAsync(d_ringpos, 0, 4, 0));
    CK(hipMemsetAsync(d_S, 0, (size_t)ngdn * 48 * 128 * 128 * 4, 0));
    CK(hipMemsetAsync(d_convst, 0, (size_t)ngdn * 3 * 10240 * 4, 0));
    CK(hipMemsetAsync(d_ring, 0, (size_t)9 * 10240 * 4, 0));
    CK(hipDeviceSynchronize());
    kv_trim(0);  // A2: the next sequence maps its pages from scratch
    if (kv_zero_dbg && !kv_paged) kv_zero_unpaged();
    // DFlash2: no target taps / draft context survive a reset.
    df2.tap_hi = 0;
    df2.cap_mode = 0;
  }

  // ---- D2-S1c vseg selftest ----
  // QWENOX_VSEG_SELFTEST="3,4,5": prefill nseg fresh slots (prompt lengths
  // 2048/4097/6146/... — verify bases with %4 = 0/1/2/3 coverage, all >= 2048
  // so the sparse-only QSA path and split paged page runs are exercised),
  // then verify the segments twice: once as nseg serial single-segment
  // verify_segments calls (the production path), once as one multi-segment
  // call. The verify ids and every piece of per-slot state (d_S / d_convst /
  // d_ring / d_iraw / d_R / pos mirrors / chist / kv_seq.mapped, plus the
  // shared indexer snapshot rows) must match bit for bit. Needs the MTP build
  // (snap buffers) and paged KV (slots). Process exit: 0 = PASS, 1 = FAIL.
  bool vseg_selftest(const char* spec) {
    int pv[8] = {0}, nseg = 0, R = 0;
    for (const char* p = spec; *p && nseg < 8;) {
      pv[nseg++] = atoi(p);
      while (*p && *p != ',') p++;
      if (*p == ',') p++;
    }
    if (nseg < 2) {
      fprintf(stderr, "vseg-selftest: need >= 2 segment sizes (e.g. 3,4,5)\n");
      return false;
    }
    for (int s = 0; s < nseg; s++) {
      if (pv[s] < 1 || pv[s] > 8) {
        fprintf(stderr, "vseg-selftest: segment %d size %d out of [1,8]\n", s,
                pv[s]);
        return false;
      }
      R += pv[s];
    }
    if (R > kVSnapRows || R > maxbatch_cap) {
      fprintf(stderr, "vseg-selftest: R=%d rows exceed the verify cap\n", R);
      return false;
    }
    if (!d_grow || !d_iraw || !d_iproj_snap) {
      fprintf(stderr, "vseg-selftest: needs the MTP build (snap/iraw buffers)\n");
      return false;
    }
    slots_init(nseg);
    if (nslots < nseg) {
      fprintf(stderr, "vseg-selftest: slots_init(%d) refused (needs "
              "QWENOX_KV_PAGED=1)\n", nseg);
      return false;
    }
    // QWENOX_VSEG_SHORT=1: short prompts (bases < 2051 -> the dense attention
    // path that short chat prompts take; mixed base %4).
    static const int plens_long[8] = {2048, 4097, 6146, 2055, 3076, 4101, 6152, 2049};
    static const int plens_short[8] = {97, 130, 163, 99, 140, 171, 103, 152};
    const int* plens = getenv("QWENOX_VSEG_SHORT") ? plens_short : plens_long;
    for (int s = 0; s < nseg; s++)
      if (plens[s] + pv[s] > maxctx) {
        fprintf(stderr, "vseg-selftest: slot %d needs base %d + P %d > maxctx "
                "%d — rerun with --maxctx 16384\n", s, plens[s], pv[s], maxctx);
        return false;
      }
    auto fnv = [](const void* d, size_t n) {
      uint64_t f = 1469598103934665603ULL;
      const uint8_t* b = (const uint8_t*)d;
      for (size_t i = 0; i < n; i++) { f ^= b[i]; f *= 1099511628211ULL; }
      return f;
    };
    auto fnv_dev = [&](const void* p, size_t n) {
      std::vector<uint8_t> h(n);
      CK(hipMemcpy(h.data(), p, n, hipMemcpyDeviceToHost));
      return fnv(h.data(), n);
    };
    auto gen_tokens = [this](int n, uint32_t seed) {
      std::vector<int> v((size_t)n);
      uint32_t x = seed;
      for (int i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        v[(size_t)i] = (int)((x >> 8) % (uint32_t)g_cfg.vocab);
      }
      return v;
    };
    const size_t ngdn = (size_t)(g_cfg.layers - g_cfg.layers / 4),
                 nqsa = (size_t)(g_cfg.layers / 4);
    struct Snap {
      uint64_t dS, dconv, dring, diraw, dR, dpos, dringpos, chist;
      uint64_t logits = 0;  // this segment's verify rows of d_logits_b
      int pos, ringpos, mapped;
      std::vector<int> v;
    };
    auto capture = [&](Snap& o) {  // the slot must be bound
      o.dS = fnv_dev(d_S, ngdn * 48 * 128 * 128 * 4);
      o.dconv = fnv_dev(d_convst, ngdn * 3 * 10240 * 4);
      o.dring = fnv_dev(d_ring, (size_t)9 * 10240 * 4);
      o.diraw = fnv_dev(d_iraw, nqsa * 4 * 128 * 4);
      o.dR = fnv_dev(d_R, (size_t)10240 * 4);
      o.dpos = fnv_dev(d_pos, 4);
      o.dringpos = fnv_dev(d_ringpos, 4);
      o.chist = chist.empty() ? 0 : fnv(chist.data(), chist.size() * 8);
      o.pos = pos;
      o.ringpos = ple_ringpos;
      o.mapped = kv_seq.mapped;
    };
    // prompts + draft segments (deterministic); toks[0] = last committed token
    std::vector<std::vector<int>> prompts((size_t)nseg), segt((size_t)nseg);
    for (int s = 0; s < nseg; s++) {
      prompts[(size_t)s] = gen_tokens(plens[s], 0x1234u + (uint32_t)s * 999u);
      segt[(size_t)s].push_back(prompts[(size_t)s].back());
      std::vector<int> dr = gen_tokens(pv[s] - 1, 0xabcd0u + (uint32_t)s * 555u);
      for (int t : dr) segt[(size_t)s].push_back(t);
    }
    // ---- phase A: serial single-segment verifies (production path) ----
    // Snap windows pack at the batch row offsets (R <= kVSnapRows), not
    // 8*s — 8*s overflows the snap buffers for nseg >= 4 (UB, and it
    // corrupts adjacent slot state).
    std::vector<Snap> A((size_t)nseg);
    int voff = 0;
    for (int s = 0; s < nseg; s++) {
      slot_reset(s, s);
      prefill_batch(prompts[(size_t)s], 0, false, true);
      spec_snap = true;        // bound == s: per-row rollback snapshots on,
      vsnap_row0_ = voff;      // this slot's snap window (same in both phases)
      voff += pv[s];
      VSeg one{s, &segt[(size_t)s], plens[s]};
      A[(size_t)s].v.resize((size_t)pv[s]);
      if (!verify_segments(&one, 1, A[(size_t)s].v.data())) {
        fprintf(stderr, "vseg-selftest: serial verify of segment %d refused\n",
                s);
        return false;
      }
      spec_snap = false;
      gdn_commit(pv[s]);  // full-accept commit: S via the replay path
      capture(A[(size_t)s]);
      // full logits rows: sampling consumes every value, not just the argmax
      A[(size_t)s].logits =
          fnv_dev(d_logits_b, (size_t)pv[s] * g_cfg.vocab * 4);
    }
    const uint64_t ipsA = fnv_dev(d_iproj_snap, nqsa * kVSnapRows * 128 * 4);
    const uint64_t growA = fnv_dev(d_grow, (size_t)R * ngdn * kGdnRowF * 4);
    // ---- phase B: one multi-segment verify ----
    voff = 0;
    for (int s = 0; s < nseg; s++) {
      slot_reset(s, s);
      prefill_batch(prompts[(size_t)s], 0, false, true);
      spec_snap = true;
      vsnap_row0_ = voff;
      voff += pv[s];
    }
    bind(0);
    VSeg segs[8];
    for (int s = 0; s < nseg; s++)
      segs[s] = VSeg{s, &segt[(size_t)s], plens[s]};
    std::vector<int> vout((size_t)R, -1);
    if (!verify_segments(segs, nseg, vout.data())) {
      fprintf(stderr, "vseg-selftest: multi-segment call refused (a guard fell "
              "through)\n");
      return false;
    }
    const uint64_t ipsB = fnv_dev(d_iproj_snap, nqsa * kVSnapRows * 128 * 4);
    const uint64_t growB = fnv_dev(d_grow, (size_t)R * ngdn * kGdnRowF * 4);
    std::vector<uint64_t> logB((size_t)nseg);
    {
      int vo = 0;
      for (int s = 0; s < nseg; s++) {
        logB[(size_t)s] = fnv_dev(d_logits_b + (size_t)vo * g_cfg.vocab,
                                  (size_t)pv[s] * g_cfg.vocab * 4);
        vo += pv[s];
      }
    }
    // ---- compare ----
    bool pass = true;
    voff = 0;
    for (int s = 0; s < nseg; s++) {
      for (int i = 0; i < pv[s]; i++)
        if (vout[(size_t)voff + i] != A[(size_t)s].v[(size_t)i]) {
          fprintf(stderr, "vseg-selftest: seg %d row %d id %d != serial %d\n",
                  s, i, vout[(size_t)voff + i], A[(size_t)s].v[(size_t)i]);
          pass = false;
        }
      voff += pv[s];
      bind(s);
      gdn_commit(pv[s]);  // same commit as phase A, from the batched rows
      Snap B;
      capture(B);
      const Snap& a = A[(size_t)s];
      auto cmp = [&](const char* what, uint64_t x, uint64_t y) {
        if (x != y) {
          fprintf(stderr, "vseg-selftest: seg %d %s %016llx != serial %016llx\n",
                  s, what, (unsigned long long)x, (unsigned long long)y);
          pass = false;
        }
      };
      cmp("d_S", B.dS, a.dS);
      cmp("d_convst", B.dconv, a.dconv);
      cmp("d_ring", B.dring, a.dring);
      cmp("d_iraw", B.diraw, a.diraw);
      cmp("d_R", B.dR, a.dR);
      cmp("d_pos", B.dpos, a.dpos);
      cmp("d_ringpos", B.dringpos, a.dringpos);
      cmp("chist", B.chist, a.chist);
      cmp("logits", logB[(size_t)s], a.logits);
      if (B.pos != a.pos || B.ringpos != a.ringpos || B.mapped != a.mapped) {
        fprintf(stderr, "vseg-selftest: seg %d pos/ringpos/mapped (%d,%d,%d) "
                "!= serial (%d,%d,%d)\n", s, B.pos, B.ringpos, B.mapped, a.pos,
                a.ringpos, a.mapped);
        pass = false;
      }
    }
    bind(0);
    if (growA != growB) {
      fprintf(stderr, "vseg-selftest: d_grow %016llx != serial %016llx\n",
              (unsigned long long)growB, (unsigned long long)growA);
      pass = false;
    }
    if (ipsA != ipsB) {
      fprintf(stderr, "vseg-selftest: d_iproj_snap %016llx != serial %016llx\n",
              (unsigned long long)ipsB, (unsigned long long)ipsA);
      pass = false;
    }
    fprintf(stderr, "VSEG SELFTEST (nseg=%d R=%d): %s\n", nseg, R,
            pass ? "PASS" : "FAIL");
    // QWENOX_VSEG_BENCH=N: after the check, time N batched calls vs N rounds
    // of the serial per-segment calls (same segments, same bases; slot state
    // is overwritten in place, so only timing is meaningful here). With
    // QWENOX_KPROF=1 QWENOX_KPROF_V=1 each call also prints its kprof line.
    if (const char* be = getenv("QWENOX_VSEG_BENCH")) {
      const int N = std::max(1, atoi(be));
      auto now = [] { return std::chrono::steady_clock::now(); };
      auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
      };
      std::vector<double> tb, ts, te;
      for (int it = 0; it < N + 1; it++) {  // first iteration = warm-up
        // production state for the batched call: every slot verifies with
        // rollback snapshots on (the serial loop below clears spec_snap, and
        // a snap-off batch takes different kernels -- measured the wrong path)
        {
          int vo = 0;
          for (int s = 0; s < nseg; s++) {
            slot_spec_snap(s) = true;
            slot_vsnap_row0_(s) = vo;
            vo += pv[s];
          }
        }
        bind(0);
        CK(hipStreamSynchronize(g_str));
        g_vseg_enq_ms = 0;
        auto t0 = now();
        verify_segments(segs, nseg, vout.data());
        CK(hipStreamSynchronize(g_str));
        auto t1 = now();
        if (it) te.push_back(g_vseg_enq_ms);
        int vo = 0;
        for (int s = 0; s < nseg; s++) {
          bind(s);
          spec_snap = true;
          vsnap_row0_ = vo;
          vo += pv[s];
          VSeg one{s, &segt[(size_t)s], plens[s]};
          verify_segments(&one, 1, A[(size_t)s].v.data());
          spec_snap = false;
        }
        CK(hipStreamSynchronize(g_str));
        auto t2 = now();
        if (it) {
          tb.push_back(ms(t0, t1));
          ts.push_back(ms(t1, t2));
        }
      }
      std::sort(tb.begin(), tb.end());
      std::sort(ts.begin(), ts.end());
      // Reference: ONE segment of R rows on slot 0 (the single-sequence verify
      // path at the same total row count), when R fits a verify chunk.
      double t1med = -1;
      if (R <= 8) {
        std::vector<int> one_t((size_t)R);
        for (int i = 0; i < R; i++) one_t[(size_t)i] = segt[0][(size_t)i % pv[0]];
        std::vector<int> o((size_t)R);
        std::vector<double> t1;
        for (int it = 0; it < N + 1; it++) {
          bind(0);
          spec_snap = true;
          vsnap_row0_ = 0;
          VSeg one{0, &one_t, plens[0]};
          CK(hipStreamSynchronize(g_str));
          auto a = now();
          verify_segments(&one, 1, o.data());
          CK(hipStreamSynchronize(g_str));
          if (it) t1.push_back(ms(a, now()));
          spec_snap = false;
        }
        std::sort(t1.begin(), t1.end());
        t1med = t1[t1.size() / 2];
      }
      std::sort(te.begin(), te.end());
      fprintf(stderr,
              "VSEG BENCH nseg=%d R=%d: batched median %.1f ms (min %.1f, host "
              "enqueue %.1f) | serial %d segs median %.1f ms (min %.1f) | ratio "
              "%.2f | single-seg P=%d %.1f ms\n",
              nseg, R, tb[tb.size() / 2], tb[0], te[te.size() / 2], nseg,
              ts[ts.size() / 2], ts[0], ts[ts.size() / 2] / tb[tb.size() / 2], R,
              t1med);
      bind(0);
    }
    return pass;
  }

  int forward(int token) {
    kv_prepare(pos, pos + 1);  // A2/A3: back (and unshare) row pos, both paths
    if (graphs_ok) {
      *p_token = token;
      auto g0 = std::chrono::steady_clock::now();
      CK(hipGraphLaunch(g1_exec, 0));
      dump_decode_step(pos);  // debug only (QWENOX_MROPE_DUMP); syncs the stream
      double g1_ms;
      if (ple_on) {
        ple_host(token, ple_e);  // host hash+read overlaps G1 on GPU
        g1_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - g0)
                    .count();
      } else {
        g1_ms = 0;
      }
      CK(hipGraphLaunch(g2_exec, 0));
      pos++;
      int id;
      auto sync0 = std::chrono::steady_clock::now();
      CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
      double sync_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync0)
              .count();
      if (getenv("QWENOX_TIME")) {
        t_enq += g1_ms;
        t_sync += sync_ms;
        if (t_n % 16 == 15)
          fprintf(stderr, "TIME: ple_host=%.2f ms/tok g2_sync=%.2f ms/tok\n",
                  t_enq / (t_n + 1), t_sync / (t_n + 1));
      }
      t_n++;
      return id;
    }
    // direct path (debug/perf): upload scalars, enqueue everything
    CK(hipMemcpyAsync(d_token, &token, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_pos, &pos, 4, hipMemcpyHostToDevice, g_str));
    CK(hipMemcpyAsync(d_ringpos, &ple_ringpos, 4, hipMemcpyHostToDevice, g_str));
    auto T0 = tick();
    enq_embed();
    tock(0, T0);
    // layer 0 enqueued first so PLE host reads overlap with GPU work
    enq_layer(0);
    dump_decode_step(pos);  // debug only (QWENOX_MROPE_DUMP); syncs the stream
    if (ple_on) {
      auto t1 = tick();
      ple_host(token, ple_e);
      tock(1, t1);
      ple_gpu();
      ple_ringpos = (ple_ringpos + 1) % 9;
    }
    for (int l = 1; l < g_cfg.layers; l++) enq_layer(l);
    auto t4 = tick();
    auto enq0 = std::chrono::steady_clock::now();
    enq_final();
    pos++;
    int id;
    double enq_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - enq0)
            .count();
    auto sync0 = std::chrono::steady_clock::now();
    CK(hipMemcpy(&id, d_argmax, 4, hipMemcpyDeviceToHost));
    double sync_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sync0)
            .count();
    if (getenv("QWENOX_TIME")) {
      t_enq += enq_ms;
      t_sync += sync_ms;
      if (t_n % 16 == 15)
        fprintf(stderr, "TIME: enq=%.2f ms/tok sync=%.2f ms/tok\n", t_enq / t_n,
                t_sync / t_n);
    }
    tock(4, t4);
    t_n++;
    return id;
  }

  void perf_report() {
    if (!perf || !t_n) return;
    fprintf(stderr,
            "PERF ms/token: embed=%.2f ple=%.2f attn=%.2f moe=%.2f head=%.2f total=%.2f "
            "(%d tokens)\n",
            t_acc[0] / t_n, t_acc[1] / t_n, t_acc[2] / t_n, t_acc[3] / t_n, t_acc[4] / t_n,
            (t_acc[0] + t_acc[1] + t_acc[2] + t_acc[3] + t_acc[4]) / t_n, t_n);
  }

  void get_logits(std::vector<float>& out) {
    out.resize(g_cfg.vocab);
    CK(hipMemcpy(out.data(), d_logits, (size_t)g_cfg.vocab * 4,
                 hipMemcpyDeviceToHost));
  }
};

// --- QSA q/gate split + per-head zc norm: qg (24, 512) -> qs (24,256), gs (24,256)
// Batched: grid.y = token row (qg row stride 24*2*dh, qs/gs row stride 24*dh).
// Rope=true additionally applies the rotary (64 dims, pairs i/i+32) in-kernel,
// bit-exact vs the k_qsa_qsplit + k_rope_b sequence: the normed value is staged
// in LDS (store/reload of a float is exact) and the cos/sin come from the
// k_rope_cs table (bit-identical to k_rope_b's inline fp64 math, but computed
// once per (token, i) instead of once per (head, token, i) — gfx1151 fp64 is
// too weak to pay 24x redundant transcendentals inside this kernel).
// Requires dh == 256, rotary == 64 (all QSA callers).
template <bool Rope>
__global__ void k_qsa_qsplit(const float* __restrict__ qg, const float* __restrict__ qnw,
                             float* __restrict__ qs, float* __restrict__ gs, int dh,
                             float eps, int rope_base, const float2* __restrict__ ropecs) {
  int h = blockIdx.x;
  int t = threadIdx.x;
  qg += (size_t)blockIdx.y * 48 * dh;
  qs += (size_t)blockIdx.y * 24 * dh;
  gs += (size_t)blockIdx.y * 24 * dh;
  __shared__ float sm[256];
  float v = (t < dh) ? qg[h * 2 * dh + t] : 0.f;
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / dh + eps);
  if (gs && t < dh) gs[h * dh + t] = qg[h * 2 * dh + dh + t];  // null: gate read from qg
  if (!Rope) {
    if (t < dh) qs[h * dh + t] = v * inv * (1.f + qnw[t]);
    return;
  }
  float* q = qs + h * dh;
  if (t < 32) {
    // recompute the pair's normed values inline (same expressions, same bits;
    // the extra qg load is L2-hot) instead of staging them through LDS —
    // saves two __syncthreads and the sm round-trip per block.
    float2 cs = ropecs[(size_t)blockIdx.y * 32 + t];
    if (g_rope_mscale != 1.f) cs.x *= g_rope_mscale, cs.y *= g_rope_mscale;
    float v1 = qg[h * 2 * dh + t + 32];
    float x0 = v * inv * (1.f + qnw[t]);
    float x1 = v1 * inv * (1.f + qnw[t + 32]);
    // contraction pinned to k_rope_b's compiled forms (verified bit-identical
    // at -O2 and -O3): r0 = fma(x0,cs,-(x1*sn)), r1 = fma(x1,cs,x0*sn)
    q[t] = fmaf(x0, cs.x, -(x1 * cs.y));
    q[t + 32] = fmaf(x1, cs.x, x0 * cs.y);
  } else if (t >= 64 && t < dh) {
    q[t] = v * inv * (1.f + qnw[t]);
  }
  (void)rope_base;
}

// --- Fused QSA K prepare, one block per (kv head, token): zc rmsnorm over the
// 256-dim head (same tree, weight per head == k_rmsnorm_zc_grouped with
// ngroups=2 addressing) + rotary (cos/sin from the k_rope_cs table, bit-
// identical to k_rope_b) + optional bf16 store into the compact KV cache
// (same f2bf as k_f32_to_bf16_v4). Bit-exact vs the rmsnorm + k_rope_b
// [+ k_f32_to_bf16_v4] sequence.
// grid = dim3(2, P); kc (when Bf16) already includes the base*512 offset —
// or, with dvb set (P3-B graph capture), kc is the layer base and the kernel
// adds (*dvb)*512 itself.
template <bool Bf16>
__global__ void k_qsa_kprep(const float* __restrict__ kb, const float* __restrict__ knw,
                            float* __restrict__ kout, uint16_t* __restrict__ kc,
                            float eps, int rope_base,
                            const float2* __restrict__ ropecs,
                            const int* __restrict__ dvb) {
  if (Bf16 && dvb) kc += (size_t)(*dvb) * 512;
  int h = blockIdx.x;
  int t = threadIdx.x;
  const float* xg = kb + ((size_t)blockIdx.y * 2 + h) * 256;
  const float* wg = knw;  // both K heads share the single 256-weight (knfix)
  float* og = kout + ((size_t)blockIdx.y * 2 + h) * 256;
  __shared__ float sm[256];
  float v = xg[t];
  sm[t] = v * v;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (t < off) sm[t] += sm[t + off];
    __syncthreads();
  }
  float inv = rsqrtf(sm[0] / 256 + eps);
  if (t < 32) {
    // inline recompute instead of LDS staging (see k_qsa_qsplit)
    float2 cs = ropecs[(size_t)blockIdx.y * 32 + t];
    if (g_rope_mscale != 1.f) cs.x *= g_rope_mscale, cs.y *= g_rope_mscale;
    float v1 = xg[t + 32];
    float x0 = v * inv * (1.f + wg[t]);
    float x1 = v1 * inv * (1.f + wg[t + 32]);
    // contraction pinned to k_rope_b's compiled forms (see k_qsa_qsplit)
    float r0 = fmaf(x0, cs.x, -(x1 * cs.y)), r1 = fmaf(x1, cs.x, x0 * cs.y);
    og[t] = r0;
    og[t + 32] = r1;
    if (Bf16) {
      uint16_t* dst = kc + (size_t)blockIdx.y * 512 + h * 256;
      dst[t] = f2bf(r0);
      dst[t + 32] = f2bf(r1);
    }
  } else if (t >= 64) {
    float o = v * inv * (1.f + wg[t]);
    og[t] = o;
    if (Bf16) kc[(size_t)blockIdx.y * 512 + h * 256 + t] = f2bf(o);
  }
  (void)rope_base;
}

// --- acc[r] += sigmoid(sg[i/n]) * y[i]; flat over total = P*n
__global__ void k_axpy_sg(float* __restrict__ acc, const float* __restrict__ y,
                          const float* __restrict__ sg, int n, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    float s = 1.f / (1.f + expf(-sg[i / n]));
    acc[i] += s * y[i];
  }
}
// --- out[r] = acc[r] + sigmoid(sg[i/n]) * y[i]; same FMA as k_axpy_sg, so the
// values are identical to `k_axpy_sg(acc,...)` followed by a copy acc -> out.
__global__ void k_axpy_sg_out(const float* __restrict__ acc, const float* __restrict__ y,
                              const float* __restrict__ sg, float* __restrict__ out,
                              int n, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    float s = 1.f / (1.f + expf(-sg[i / n]));
    float v = acc[i];
    v += s * y[i];
    out[i] = v;
  }
}
