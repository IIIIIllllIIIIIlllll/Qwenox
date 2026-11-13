// ============================ host side =====================================

// --- Windows 单 arena 分配器 ------------------------------------------------
// Windows/ROCm 路由规则（tools/winprobe 实测）：进程累计中小额分配 >32 GiB 后
// 新分配会被路由到共享内存（主机 RAM，越界即灾难）；而单次大额 hipMalloc
// （实测 40-95 GiB）整体落 VRAM。且 postarena 探针证明：大额 arena 之后的
// 小额 hipMalloc 仍然安全。所以 Windows 下全部持久设备内存来自一次大额
// hipMalloc（g_devarena），bump 分配；溢出时回退 hipMalloc（小额、安全）。
// Linux 行为不变：DALLOC 就是 hipMalloc。
#ifdef _WIN32
static void* g_devarena = nullptr;
static size_t g_devarena_sz = 0, g_devarena_off = 0;
static hipError_t dalloc_arena(void** p, size_t bytes, const char* name) {
  static const bool dbg = getenv("QWENOX_ARENA_DEBUG") != nullptr;
  size_t off = (g_devarena_off + 255) & ~(size_t)255;
  if (!g_devarena || off + bytes > g_devarena_sz) {
    if (g_devarena) {
      static bool warned = false;
      if (!warned) {
        warned = true;
        fprintf(stderr,
                "arena: capacity %.2f GiB exhausted at +%zu B; further allocations "
                "fall back to hipMalloc\n",
                g_devarena_sz / 1073741824.0, bytes);
      }
    }
    if (dbg)
      fprintf(stderr, "arena-debug: %-16s %10.2f MiB @ n/a (hipMalloc fallback)\n",
              name, bytes / 1048576.0);
    return hipMalloc(p, bytes);
  }
  if (dbg)
    fprintf(stderr, "arena-debug: %-16s %10.2f MiB @ %.2f GiB\n", name,
            bytes / 1048576.0, off / 1073741824.0);
  *p = (char*)g_devarena + off;
  g_devarena_off = off + bytes;
  return hipSuccess;
}
#define DALLOC(p, s) CK(dalloc_arena((void**)(p), (size_t)(s), #p))
#else
#define DALLOC(p, s) CK(hipMalloc(p, s))
#endif

// 与 devarena_estimate 共享的 maxbatch 决策。32K 是 Linux 上实测能与 68 GiB
// 模型及长上下文 KV 共存的最大批（QWENOX_PREFILL_CHUNK 可压低）；Windows 受
// 96 GiB VRAM 硬门槛限制，默认 6144：64K 稳态 PP 1598 vs 8192 的 1606（−0.5%），
// 工作区少约 1.7 GiB，且不触发 P>=8192 才开的 B3 split-K（见 PORTING-WINDOWS.md）。
// 并发 serve 的 prefill 分段覆盖（D1a）：main 的 serve 分支在 eff_parallel()>1 且
// QWENOX_CONC_PREFILL_CHUNK>0 时设置为 min(它, 单路分段)，之后取代单路值。prefill 只在层间把 GPU
// 让给别的会话（D1b），分段越小每层越短、别的会话卡得越短。0 = 不覆盖。
static int g_maxbatch_override = 0;

static int eff_maxbatch(int maxctx) {
  if (g_maxbatch_override > 0) return std::min(maxctx, g_maxbatch_override);
  if (const char* e = getenv("QWENOX_PREFILL_CHUNK")) {
    int v = atoi(e);
    if (v > 0) return std::min(maxctx, v);
    return maxctx;
  }
#ifdef _WIN32
  return std::min(maxctx, 6144);
#else
  return std::min(maxctx, 32768);
#endif
}

// 尾包合并 slack：prompt 切成 maxbatch chunk 后若最后只剩 ≤ slack 个
// token，则并入前一个 chunk（该 chunk 最大 maxbatch+slack）。workspace
// 按 maxbatch+slack 分配。0 禁用。Windows arena 有 95 GiB 硬顶，默认关。
static int eff_tail_slack() {
  if (const char* e = getenv("QWENOX_PREFILL_TAIL_SLACK"))
    return std::max(0, atoi(e));
#ifdef _WIN32
  return 0;
#else
  return 1024;
#endif
}

// 并发序列槽数（QWENOX_PARALLEL，serve 模式；1..8）。槽共享同一个 KV 页池，
// 每多一个槽只多 GDN/卷积/环形状态、verify 回滚检查点、页表和 MTP KV
// （见 GpuModel::slots_init）。
static int eff_parallel() {
  const char* e = getenv("QWENOX_PARALLEL");
  return e ? std::max(1, std::min(8, atoi(e))) : 1;
}
// Multi-sequence verify rendezvous possible in this process (serve with
// several slots, or the vseg selftest/bench). Sizes the rollback snapshot
// rows and the multi-row gemv cap: a single sequence never batches more
// than gamma+1 <= 9 verify rows, so it keeps the pre-D2 footprint.
static bool vseg_multi() {
  static const bool v = eff_parallel() > 1 || getenv("QWENOX_VSEG_SELFTEST");
  return v;
}
static int vsnap_rows_cfg() { return vseg_multi() ? 48 : 9; }

// hgn q4cp 路由专家 prefill 走 LUT 解码 WMMA kernel（27_kernels_moe_lut.inc）。
// 默认开；QWENOX_MOE_Q4W=0 恢复旧路径（QWENOX_MOE_LT / tiled k_moe_w4）。开启时
// QWENOX_MOE_LT 不生效，其 dequant/gather 缓冲也不分配。
static bool moe_q4w_env() {
  const char* s = getenv("QWENOX_MOE_Q4W");
  return !(s && *s == '0');
}

// B1 w4b f16 pairs（k_moe_lut<kQ4CP,...,kF16Pairs> 写 + k_moe_reduce_pw*<K,
// __half> 读）。默认开（KLD 过线后翻转），QWENOX_W4B_PAIRS_F16=0 回退 f32。
// 与 40_model.inc moe_q4w_run 的 pairs_f16 必须保持同一语义。
static bool w4b_pairs_f16_env() {
  const char* s = getenv("QWENOX_W4B_PAIRS_F16");
  return !s || !(s[0] == '0' && s[1] == 0);
}
// d_guvb 可按 f16 pairs 尺寸分配的条件（40_model.inc 构造函数同款）：v2+w4b
// 两条 LUT 路径的 pairs 全 f16 后，剩余 d_guvb 读者要么被这些 env 排除
// （GGUF 专家 / moe_lt / tiled w4 / v2 xcodes 回退），要么每 token-slot 字节数
// 与 f16 pairs 相同（naive gemv 部分积 / 共享专家 scratch / decode 小 P），
// 没有全尺寸 f32 读者。
static bool guvb_f16_env() {
  const char* v2 = getenv("QWENOX_V2_PAIRS_F16");
  const char* nm = getenv("QWENOX_MOE_NAIVE_MAX");
  return w4b_pairs_f16_env() && !(v2 && v2[0] == '0' && v2[1] == 0) &&
         moe_q4w_env() && !getenv("QWENOX_MOE_NAIVE") &&
         getenv("QWENOX_MOE_ATOMIC") == nullptr &&
         !getenv("QWENOX_MOE_HOST_ROUTE") && !getenv("QWENOX_MOE_ORDERED") &&
         !getenv("QWENOX_MOE_UNTILED") && getenv("QWENOX_GGUF") == nullptr &&
         (!nm || atoi(nm) <= 16) && g_cfg.topk == 10 && g_cfg.moe_mid % 64 == 0;
}

// QWENOX_MOE_A4（可选，默认关）：hgn v2 路由专家 prefill 走 W4A4 iu4 WMMA，
// 位掩码 1 = gate/up，2 = down，3 = 两者。见 40_model.inc moe_b_v2。
static int moe_a4_env() {
  const char* e = getenv("QWENOX_MOE_A4");
  return e && *e ? atoi(e) & 3 : 0;
}

// QWENOX_GG_A4（已废弃，禁止使用，默认关）：GGUF 原生 W4A4，位掩码同 MOE_A4。
// 废弃结论与证据见 29_kernels_moe_gg_iu4.inc 头部。与 QWENOX_MOE_A4 分开，
// 避免 service.conf 里的 hgn 开关悄悄改变 GGUF 行为。
static int gg_a4_env() {
  const char* e = getenv("QWENOX_GG_A4");
  return e && *e ? atoi(e) & 3 : 0;
}

// The PLE n-gram table never enters the device arena (read from page cache /
// disk at runtime): dtype 10 (hgn fp8), 11 (GGUF IQ4_NL) — or dtype 0 when the
// table ships as raw bf16 under the same name (320 B rows). dtype 4 = the tiny
// u64 hash config tensors, also host-side only.
static inline bool is_ple_offarena(const Tensor& t) {
  return t.dtype == 10 || t.dtype == 11 || t.dtype == 4 ||
         t.name == "layers.1.ple.ngram_embedding.weight";
}

// 镜像 GpuModel 构造函数的分配公式（含相同 env 分支），用于 arena 精确 sizing。
// 每项 256B 对齐后求和；估少了一律有 dalloc_arena 的 hipMalloc 回退兜底。
// GGUF routed experts that load_arena will copy into the arena (Windows),
// and the hgn expert bytes they replace (defined after gguf_experts_open).
[[maybe_unused]] static size_t gguf_experts_arena_bytes(const Checkpoint& ck, size_t* hgn_replaced);
static size_t devarena_estimate(const Checkpoint& ck, int maxctx) {
  size_t total = 0;
  auto add = [&](size_t b) { total += (b + 255) & ~(size_t)255; };
  // 权重：Windows 下专家也进 arena（无 hipHostRegister 直读；PLE 表/配置留盘）
  for (auto& kv : ck.tensors()) {
    const Tensor& t = kv.second;
    if (is_ple_offarena(t)) continue;
    add(t.data_size);
  }
#ifdef _WIN32
  {  // GGUF 专家（QWENOX_GGUF）同样拷进 arena；被它替换的 hgn 专家不进
    size_t hgn_rep = 0;
    total += gguf_experts_arena_bytes(ck, &hgn_rep);
    total -= std::min(total, hgn_rep);
  }
#endif
  const bool qsa_dense = getenv("QWENOX_QSA_DENSE") != nullptr;
  const bool kv_fp32 = getenv("QWENOX_QSA_KV_BF16") == nullptr;
  const bool index_f32 = getenv("QWENOX_INDEX_BF16") == nullptr;
  const bool btv = getenv("QWENOX_QSA_WMMA") && !kv_fp32 &&
                   getenv("QWENOX_QSA_WMMA_BTV");
  const bool fused = getenv("QWENOX_PREFILL_UNFUSED") == nullptr;
  const char* cut_off = getenv("QWENOX_CUT_INPASS");
  const bool cut_inpass = !(cut_off && cut_off[0] == '0' && cut_off[1] == 0) &&
                          getenv("QWENOX_GDN_FUSED") && !getenv("QWENOX_GDN_LOOP") &&
                          !getenv("QWENOX_GDN_NOSPLIT") && getenv("QWENOX_GR_BF16") && fused &&
                          !getenv("QWENOX_PLE_LOOP");
  const bool moe_det = getenv("QWENOX_MOE_ATOMIC") == nullptr;  // 与 40_model.inc 同步
  const bool moe_lt_ok = getenv("QWENOX_MOE_LT") && !moe_q4w_env() && g_cfg.topk == 10 && moe_det &&
                         !getenv("QWENOX_MOE_HOST_ROUTE") &&
                         !getenv("QWENOX_MOE_ORDERED");
  const bool lt_cmp = getenv("QWENOX_MOE_LT_CMP") != nullptr;
  const bool lt_ovl = getenv("QWENOX_MOE_LT_OVL") != nullptr;
  const bool gdn_split =
      !getenv("QWENOX_GDN_NOSPLIT") && !getenv("QWENOX_GDN_LOOP") &&
      !getenv("QWENOX_GDN_FUSED");
  const bool gdn_str = getenv("QWENOX_GDN_STREAM") != nullptr;
  const bool gdn_pipe = getenv("QWENOX_GDN_PIPE") != nullptr;
  const int gdn_window_chunks = [] {
    const char* e = getenv("QWENOX_GDN_WINDOW_CHUNKS");
    return e ? std::max(1, atoi(e)) : 4;
  }();
  const size_t mc = (size_t)maxctx,
               BP = (size_t)(eff_maxbatch(maxctx) + eff_tail_slack());
  const size_t ngdn = g_cfg.layers - g_cfg.layers / 4, nqsa = g_cfg.layers / 4;
  const size_t ib = (mc + 3) / 4, rb = (size_t)g_cfg.branches * g_cfg.d;
  // QWENOX_KV_PAGED：KV/ik（BF16 时含 d_vcbt）按整页（256 token）取整，另加页表与
  // ik gather 缓冲。BF16 KV 的可分页条件与构造函数一致（A1c）
  const char* kvpe = getenv("QWENOX_KV_PAGED");
  const bool kv_bf16_pg_ok =
      qsa_dense || (getenv("QWENOX_QSA_WMMA") && !(btv && getenv("QWENOX_QSA_WMMA6")) &&
                    !(btv && getenv("QWENOX_QSA_UNION")));
  const bool kv_paged = (kv_fp32 || kv_bf16_pg_ok) && kvpe && atoi(kvpe) != 0;
  const size_t kvpg = (mc + KV_PAGE - 1) / KV_PAGE;
  // A2/A3：物理页池 = kv_pool_pages_for(mc) 页（默认 kvpg，QWENOX_KV_POOL_TOKENS 可加大）
  // + 1 个 guard 页；页表仍是每序列 kvpg 项
  const size_t kvpool = (size_t)kv_pool_pages_for((int)mc);
  const size_t kv_rows = kv_paged ? (kvpool + 1) * KV_PAGE : mc;
  const size_t ik_rows = kv_paged ? (kvpool + 1) * KV_PAGE_BLOCKS : ib;
  // MTP draft head（构造函数同款开启条件）
  const bool moe_default = moe_det && !getenv("QWENOX_MOE_ORDERED") &&
                           !getenv("QWENOX_MOE_HOST_ROUTE") &&
                           !getenv("QWENOX_MOE_UNTILED") &&
                           !getenv("QWENOX_MOE_FP32_IO");
  const bool mtp_on = ck.find("mtp.fc_hidden.weight") && moe_default;
  // B0：MTP 层是 KV/ik 数组的第 nqsa 层（与 trunk 共用页池，无 d_vcbt）
  const size_t nkv = nqsa + (mtp_on ? 1 : 0);
  if (kv_paged) add(kvpg * 4);  // d_ptab
  add((size_t)64 << 20);  // decode 单行小缓冲合计（d_R..d_logits 等）
  add(ngdn * 48 * 128 * 128 * 4);           // d_S
  add(ngdn * 3 * 10240 * 4);                // d_convst
  add(ngdn * 10240 * 4 * 4);                // d_convw
  add((size_t)g_cfg.vocab * 4 * 65);        // d_logits_b
  add(ngdn * 48 * 128 * 128 * 4 + ngdn * 3 * 10240 * 4 + 9 * 10240 * 4 +
      (qsa_dense ? 0 : nqsa * 4 * 128 * 4));  // d_ckpt
  if (cut_inpass) {
    add(rb * 4);                              // d_cR（大页复用 d_ckpt）
    if (!qsa_dense) add(4 * 128 * 4);         // d_cmiraw
  }
  add((size_t)48 * 2 * 64 * 128 * 4);       // d_gdn_ws
  if (gdn_split)
    add((gdn_str ? std::min<size_t>(gdn_window_chunks, (BP + 63) / 64)
                 : (BP + 63) / 64) * 48 *
        (size_t)GDN_SPLIT_WS_FLOATS * 4 * (gdn_pipe ? 2 : 1));
  add((size_t)64 << 20);                    // d_wbf16
  add((size_t)64 << 20);                    // d_ltws
  if (!qsa_dense) {
    const size_t bc = std::min(mc, BP);     // maxbatch_cap（A1b：按 chunk 定尺寸）
    add(bc * 640 * 4);                      // d_iproj
    add(bc * 512 * 4);                      // d_iq
    add(nkv * ik_rows * 128 * 4);           // d_ik（含 MTP 层）
    if (kv_paged) add(ib * 128 * 4);        // d_ikg
    // C1 f16 twins（与 40_model.inc 的分配门控同步）
    if (!(getenv("QWENOX_INDEX_F16") && getenv("QWENOX_INDEX_F16")[0] == '0' &&
          getenv("QWENOX_INDEX_F16")[1] == 0) &&
        !(getenv("QWENOX_INDEX_SCORE64") &&
          getenv("QWENOX_INDEX_SCORE64")[0] == '0' &&
          getenv("QWENOX_INDEX_SCORE64")[1] == 0) &&
        !(getenv("QWENOX_INDEX_SEL2P") && getenv("QWENOX_INDEX_SEL2P")[0] == '0' &&
          getenv("QWENOX_INDEX_SEL2P")[1] == 0) &&
        getenv("QWENOX_INDEX_OLDSEL") == nullptr) {
      add(bc * 512 * 2);                // d_iqh
      add(nkv * ik_rows * 128 * 2);     // d_ikh
      if (kv_paged) add(ib * 128 * 2);  // d_ikgh
    }
    add(nqsa * 4 * 128 * 4);                // d_iraw
    add(nqsa * 256 * 4);                    // d_inorm
    if (index_f32) add(nqsa * 640 * 2560 * 4);  // d_iw
    add((size_t)std::min(mc, (size_t)INDEX_BATCH) * 4 * ib * 4);  // d_iscores
    add(bc * INDEX_BUDGET * 4);             // d_iselected
  }
  if (kv_fp32) {
    add(2 * nkv * kv_rows * 512 * 4);       // d_kc + d_vc（含 MTP 层）
  } else {
    add(2 * nkv * kv_rows * 512 * 2);       // d_kcb + d_vcb（含 MTP 层）
    if (btv) add(nqsa * ((kv_rows + 3) / 4) * 2 * 256 * 4 * 2);  // d_vcbt
    // QWENOX_QSA_UNION: d_ublk + d_umask + d_ucount（按单 chunk 组数）
    if (btv && getenv("QWENOX_QSA_UNION")) {
      const size_t ng = (size_t)BP / 4 + 1;
      add(ng * QSA_UCAP * 4 + ng * 4);
    }
  }
  add(2 * mc * 4);                          // d_posarr + d_ringarr
  // prefill 批缓冲（BP 维）
  add(10 * BP * 10240 * 4);   // Rb Rhatb Gb qkvb convb keysb qnb Ub Unb convoutb
  add(6 * BP * 2560 * 4);     // xb yb moeaccb eyb eb plevb
  add(BP * 320 * 4);          // t320b
  add(BP * 6144 * 4 * 2);     // zb attnb
  add(BP * 48 * 4 * 2);       // g48b beta48b
  add(BP * 12288 * 4);        // qgb
  add(BP * 6144 * 4 * 2);     // qsb gsb
  add(BP * 512 * 4 * 3);      // kbb vbb routerb
  add(BP * 32 * sizeof(float2));  // ropecs
  add(BP * 4 * 4 + BP * 16 * 4 * 2 + BP * 4);  // w4b topidsb topwb sgb
  add(BP * 16 * 4 * 2);       // injp[2] (HC inject fusion partials)
  add(BP * g_cfg.topk * (moe_det ? 2560 : 1280) * (guvb_f16_env() ? 2 : 4));  // guvb
  add(BP * g_cfg.topk * 640 * 4);                    // hidb
  add(BP * 10240 * 2 + BP * 6144 * 2);               // xbf16 xbf16b
  // d_skx（B3 split-K，与 GpuModel::gemm_splitk_on 同步）
  if (getenv("QWENOX_GEMM_WMMA") && !getenv("QWENOX_GEMM_NO_K64") &&
      !getenv("QWENOX_GEMM_RB") &&
      !(getenv("QWENOX_GEMM_SPLITK") && getenv("QWENOX_GEMM_SPLITK")[0] == '0' &&
        getenv("QWENOX_GEMM_SPLITK")[1] == 0))
    add(BP * 6144 * 2);
  if (fused) add(BP * 10240 * 2);                    // Rhatbf16
  add(BP * g_cfg.topk * 4 * 2);                      // tokidx pweight
  add((g_cfg.experts + 1) * 4);                      // eoff
  if (moe_det) add(BP * g_cfg.topk * 4);             // pairids
  add(4 * BP * g_cfg.topk * 4 + ((size_t)8 << 20));  // moe_routing keys + scratch
  add(((BP * g_cfg.topk + 63) / 64 + g_cfg.experts) * sizeof(MoeTile) + 4);  // moetiles
  if (moe_lt_ok) {
    add(BP * g_cfg.topk * g_cfg.d * 2);         // moexg
    add((size_t)4 * 2 * g_cfg.moe_mid * g_cfg.d * 2 * (lt_ovl ? 2 : 1));  // moewup
    add((size_t)4 * g_cfg.d * g_cfg.moe_mid * 2 * (lt_ovl ? 2 : 1));      // moewdn
    if (lt_cmp) {
      add(BP * g_cfg.topk * g_cfg.moe_mid * 2);  // hidb2
      add(BP * g_cfg.topk * g_cfg.d * 4);        // pairs2
      add(8);                                    // cmpi
    }
  }
  add(BP * 4 + BP * sizeof(int2));            // tokarr img_inj
  add((size_t)g_cfg.experts * sizeof(MoeTile) + 4);  // moetilesA + count (hybrid MoE tiling)
  {  // dpf: 2 slots of the largest layer's dense weights as bf16 (GpuModel::dpf_plan)
    const char* e = getenv("QWENOX_DEQ_AHEAD");  // opt-in (GpuModel::dpf_env)
    if (e && !(e[0] == '0' && e[1] == 0)) {
      std::vector<size_t> per(g_cfg.layers, 0);
      for (auto& kv : ck.tensors()) {
        const Tensor& t = kv.second;
        if (!(t.dtype == 5 || t.dtype == 7 || t.dtype == 8 || t.dtype == 16 || t.dtype == 24 ||
              (t.dtype >= 100 && ggd::type_ok((int)t.dtype - ggd::kDtypeBase))))
          continue;
        if (kv.first.rfind("layers.", 0) != 0) continue;
        const int l = atoi(kv.first.c_str() + 7);
        if (l < 0 || l >= g_cfg.layers) continue;
        const size_t b = (size_t)t.numel() * 2;
        if (b > ((size_t)64 << 20)) continue;
        per[l] += (b + 255) & ~(size_t)255;
      }
      size_t mx = 0;
      for (size_t v : per) mx = std::max(mx, v);
      add(mx);
      add(mx);
    }
  }
  // QWENOX_MOE_A4 激活量化缓冲（GpuModel d_a4*；构造器另要求 v2 权重，此处
  // 早于 v2_init 只看 env，非 v2 时多估几十 MiB）
  if (moe_a4_env() || gg_a4_env()) {
    add(BP * g_cfg.d / 2);
    add(BP * (g_cfg.d / 128) * 4);
    add(BP * g_cfg.topk * g_cfg.moe_mid / 2);
    add(BP * g_cfg.topk * (g_cfg.moe_mid / 128) * 4);
  }
  // 额外序列槽（GpuModel::slots_init，仅分页 KV）：每槽一份序列状态
  if (kv_paged)
    for (int i = 1; i < eff_parallel(); i++) {
      add(rb * 4);                                     // d_R
      add(ngdn * 48 * 128 * 128 * 4);                  // d_S
      add(ngdn * 3 * 10240 * 4);                       // d_convst
      add(9 * 10240 * 4);                              // d_ring
      add(4); add(4); add(4);                          // d_pos d_ringpos d_rdelta
      if (!qsa_dense) add(nqsa * 4 * 128 * 4);         // d_iraw
      add(ngdn * 48 * 128 * 128 * 4 + ngdn * 3 * 10240 * 4 + 9 * 10240 * 4 +
          (qsa_dense ? 0 : nqsa * 4 * 128 * 4));       // d_ckpt（D2 起 per-slot）
      add(kvpg * 4);                                   // d_ptab
      if (mtp_on) {
        add(4);                                        // d_mpos
        if (!qsa_dense) add(4 * 128 * 4);              // d_miraw（MTP KV/ik 在页池里）
        add(512 * 4);                                  // d_mring_ckpt
      }
    }
  if (mtp_on) {
    add(rb * 4 * 3 + (size_t)g_cfg.d * 4 * 5 + 8);  // mR mRhat mh me meo mt mnorm_*
    add(3 * BP * 10240 * 4);                        // mRb mRhatb mtap
    add(512 * 4);                                   // mring_ckpt
    // kVSnapRows = vsnap_rows_cfg()：多槽 48（D2 会合），单槽 9（γ+1 ≤ 9）
    const size_t vr = (size_t)vsnap_rows_cfg();
    add(vr * ngdn * 8288 * 4);                      // grow (kGdnRowF)
    add(vr * ngdn * 3 * 10240 * 4);                 // csnap
    add(nqsa * vr * 128 * 4);                       // iproj_snap
    // 10-13 会合 verify 行窗口每槽备份（GpuModel::vr_save_，9 行，d_Rb 按 fp32 上界）
    const char* vre = getenv("QWENOX_VR_RESTORE");
    if (kv_paged && eff_parallel() > 1 && !(vre && vre[0] == '0' && vre[1] == 0))
      add((size_t)eff_parallel() * 9 *
          (10240 * 4 + ngdn * 8288 * 4 + ngdn * 3 * 10240 * 4 + nqsa * 128 * 4));
    if (!qsa_dense) {
      add(4 * 128 * 4 + 256 * 4);                   // miraw minorm
      if (index_f32) add((size_t)640 * 2560 * 4);   // miw
    }
  }
  // DFlash2 draft head (GpuModel::df2_init): fused draft matrices (q8g32 by
  // default, QWENOX_DF2_WQ=bf16; not Checkpoint tensors), tap/vtap
  // rings, context K/V rings, injection + block-forward scratch.
  if (getenv("QWENOX_DFLASH2")) {
    const size_t welems = (size_t)12800 * 2560 + (size_t)10240 * 2560 +
                          5 * ((size_t)6144 * 2560 + (size_t)2560 * 4096 +
                               (size_t)15360 * 2560 + (size_t)2560 * 7680 +
                               2 * (size_t)640 * 2560);
    const char* wq = getenv("QWENOX_DF2_WQ");
    add(wq && !strcmp(wq, "bf16") ? welems * 2 : welems + welems / 16);  // weights
    add((size_t)256 * 2560 * 2);                           // selector_hidden
    add((size_t)2048 * 5 * 2560 * 4);                      // d_tap
    add((size_t)9 * 5 * 2560 * 4);                         // d_vtap
    add(2 * (size_t)5 * 2048 * 8 * 128 * 4);               // d_kctx + d_vctx
    add((size_t)48 * (12800 + 2 * 2560 + 10240) * 4);      // injection scratch
    add((size_t)8 * (5 * 2560 + 6144 + 4096 + 640 + 3 * 7680 + 256 + 33) * 4);
    add((size_t)8 * 17 * 32 * 130 * 4);                    // attention partials
  }
  add((size_t)1536 << 20);  // vision tower 权重（dup_dev 进 arena）约 1.5 GiB
  return total;
}

// Windows：在 load_arena 之前一次性 hipMalloc 整个 arena（大额单次才进 VRAM）。
static void devarena_init(const Checkpoint& ck, int maxctx) {
#ifdef _WIN32
  size_t need = devarena_estimate(ck, maxctx);
  double slack_gb = 4.0, cap_gb = 95.0;  // bigalloc 实测 95 GiB 成功 / 96 失败
  if (const char* e = getenv("QWENOX_ARENA_SLACK_GB")) slack_gb = atof(e);
  if (const char* e = getenv("QWENOX_ARENA_CAP_GB")) cap_gb = atof(e);
  size_t target = need + (size_t)(slack_gb * 1073741824.0);
  size_t cap = (size_t)(cap_gb * 1073741824.0);
  if (target > cap) {
    fprintf(stderr,
            "arena: estimate %.2f + slack %.2f GiB exceeds cap %.2f GiB; "
            "clamping (overflow falls back to hipMalloc)\n",
            need / 1073741824.0, slack_gb, cap_gb);
    target = cap;
  }
  fprintf(stderr, "devarena: requesting %.2f GiB (estimate %.2f + slack %.2f GiB)\n",
          target / 1073741824.0, need / 1073741824.0, slack_gb);
  hipError_t aerr = hipMalloc(&g_devarena, target);
  if (aerr != hipSuccess) {
    fprintf(stderr,
            "devarena: hipMalloc(%.2f GiB) failed: %s\n"
            "  this is the process's first large allocation and must land entirely\n"
            "  in VRAM as one block. Check:\n"
            "  1) BIOS VRAM carve-out (256K config needs ~95 GiB; assign 96 GiB),\n"
            "     or shrink MAX_CONTEXT / QWENOX_ARENA_CAP_GB to fit a smaller carve-out\n"
            "  2) no other process holding VRAM (single engine; close browsers/games)\n"
            "  3) TheRock DLLs from build/ sit next to the exe — stale HIP SDK DLLs\n"
            "     in System32 cap one allocation at 41 GiB and fail here\n",
            target / 1073741824.0, hipGetErrorString(aerr));
  }
  CK(aerr);
  g_devarena_sz = target;
  if (const char* f = getenv("QWENOX_ARENA_FILL")) {  // diagnostic: poison before any use
    const int byte = (int)strtol(f, nullptr, 0) & 0xff;
    CK(hipMemset(g_devarena, byte, target));
    CK(hipDeviceSynchronize());
    fprintf(stderr, "devarena: filled with 0x%02x (QWENOX_ARENA_FILL)\n", byte);
  }
  fprintf(stderr, "devarena: %.2f GiB reserved (estimate %.2f GiB)\n",
          target / 1073741824.0, need / 1073741824.0);
#else
  (void)ck;
  (void)maxctx;
#endif
}

// --- GGUF routed experts (hgn -> GGUF transition, QWENOX_GGUF=<first shard>) --
// Trunk routed experts come straight from a llama.cpp GGUF (e.g. Unsloth
// UD-Q4_K_XL: gate/up Q4_K|Q5_K, down Q5_1|Q8_0) and run on the WMMA kernels
// of 26_kernels_moe_gguf.inc; everything else still comes from hgn. The
// tensors stay in their mmap'd shards; like the hgn experts they are
// hipHostRegister'd (mapped) so kernels read the host pointers directly.
// Adjacent tensors can share a page, and a page may be registered only once,
// so page-rounded ranges are merged first. The hgn trunk experts are then
// never registered or touched (their page cache ages out). MTP experts
// (mtp.layers.0.mlp.experts.*) are not in the main GGUF: with
// QWENOX_GGUF_MTP=<sidecar> they come from its blk.<layers>.ffn_*_exps (Q8_0,
// same kernels; QWENOX_GGUF_MTP_EXPERTS=0 keeps the hgn ones).
struct GgExperts {
  const uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;
  int tgu = 0, tdn = 0;  // gguf::Type of gate/up and of down
  // source tensors (gate, up, down). Linux: gate/up/down are their
  // hipHostRegister'd mmap pointers. Windows: load_arena copies them into the
  // device arena and re-points gate/up/down there.
  const gguf::Tensor* src[3] = {nullptr, nullptr, nullptr};
};
static gguf::File* g_gguf = nullptr;
static gguf::File* g_gguf_mtp = nullptr;  // QWENOX_GGUF_MTP sidecar
static std::vector<GgExperts> g_ggx;  // per trunk layer; empty = hgn experts
static GgExperts g_ggx_mtp;           // gate == nullptr: MTP experts from hgn
static bool gg_on() { return !g_ggx.empty(); }
static bool gg_mtp_on() { return g_ggx_mtp.gate != nullptr; }
static bool is_trunk_experts(const std::string& n) {
  return n.compare(0, 7, "layers.") == 0 && n.find(".mlp.experts.") != std::string::npos;
}
static bool is_gguf_experts(const std::string& n) {
  return (gg_on() && is_trunk_experts(n)) ||
         (gg_mtp_on() && n.compare(0, 25, "mtp.layers.0.mlp.experts.") == 0);
}

static void gguf_experts_open() {
  const char* path = getenv("QWENOX_GGUF");
  if (!path || !*path || gg_on()) return;
  // QWENOX_GGUF_EXPERTS=0: keep the hgn routed experts (trunk + MTP) and take
  // only the dense tensors from the GGUF (QWENOX_GGUF_DENSE=1). Quality probe.
  if (const char* e = getenv("QWENOX_GGUF_EXPERTS"); e && !strcmp(e, "0")) {
    fprintf(stderr, "gguf experts: off (QWENOX_GGUF_EXPERTS=0), hgn experts\n");
    return;
  }
  auto t0 = std::chrono::steady_clock::now();
  if (!g_gguf) g_gguf = new gguf::File(path);  // may be open already (gguf_dense_apply)
  struct Range { uintptr_t a, b; };
  std::vector<Range> rs;
  size_t bytes = 0;
  int n_q5k = 0, n_q80 = 0;
  // blk.<l>.ffn_{gate,up,down}_exps of file f -> GgExperts (+ page ranges)
  auto take = [&](const gguf::File& f, int l, bool mtp) {
    char n[96];
    const gguf::Tensor* t[3];
    const char* nm[3] = {"gate", "up", "down"};
    for (int i = 0; i < 3; i++) {
      snprintf(n, sizeof n, "blk.%d.ffn_%s_exps.weight", l, nm[i]);
      t[i] = &f.at(n);
    }
    const uint64_t D = g_cfg.d, M = g_cfg.moe_mid, E = g_cfg.experts;
    if (t[0]->ne[0] != D || t[0]->ne[1] != M || t[0]->ne[2] != E || t[1]->type != t[0]->type ||
        t[1]->ne[0] != D || t[1]->ne[1] != M || t[1]->ne[2] != E || t[2]->ne[0] != M ||
        t[2]->ne[1] != D || t[2]->ne[2] != E)
      throw std::runtime_error("QWENOX_GGUF: unexpected expert shapes/types at layer " +
                               std::to_string(l));
    // UD-Q4_K_XL: Q4_K/Q5_K gate+up, Q5_1/Q8_0 down; UD-IQ4_XS: mixed IQ
    // gate+up, Q5_0/IQ4_NL/Q8_0/Q2_0 down (per-layer mix); MTP sidecar: Q8_0.
    (void)mtp;
    const uint32_t tg = t[0]->type, td = t[2]->type;
    if (tg != gguf::Q4_K && tg != gguf::Q5_K && tg != gguf::Q8_0 && tg != gguf::IQ4_XS &&
        tg != gguf::IQ3_S && tg != gguf::Q2_0 && tg != gguf::IQ2_XXS && tg != gguf::IQ2_XS &&
        tg != gguf::IQ3_XXS && tg != gguf::IQ2_S)
      throw std::runtime_error(std::string("QWENOX_GGUF: gate/up type ") +
                               gguf::type_name(t[0]->type) + " unsupported");
    if (td != gguf::Q5_0 && td != gguf::Q5_1 && td != gguf::Q8_0 && td != gguf::IQ4_NL && td != gguf::Q2_0)
      throw std::runtime_error(std::string("QWENOX_GGUF: down type ") +
                               gguf::type_name(t[2]->type) + " unsupported");
    for (auto* x : t) {
      const uintptr_t a = (uintptr_t)x->data & ~(uintptr_t)4095;
      const uintptr_t b = ((uintptr_t)x->data + x->nbytes + 4095) & ~(uintptr_t)4095;
      rs.push_back({a, b});
      bytes += x->nbytes;
    }
    GgExperts g{t[0]->data, t[1]->data, t[2]->data, (int)t[0]->type, (int)t[2]->type};
    for (int i = 0; i < 3; i++) g.src[i] = t[i];
    return g;
  };
  std::vector<GgExperts> trunk;
  for (int l = 0; l < g_cfg.layers; l++) {
    trunk.push_back(take(*g_gguf, l, false));
    n_q5k += trunk.back().tgu == gguf::Q5_K;
    n_q80 += trunk.back().tdn == gguf::Q8_0;
  }
  GgExperts mtpx;
  const char* mp = getenv("QWENOX_GGUF_MTP");
  const char* mx = getenv("QWENOX_GGUF_MTP_EXPERTS");
  if (mp && *mp && !(mx && !strcmp(mx, "0"))) {
    if (!g_gguf_mtp) g_gguf_mtp = new gguf::File(mp);
    mtpx = take(*g_gguf_mtp, g_cfg.layers, true);  // the MTP block follows the trunk
  }
  std::sort(rs.begin(), rs.end(), [](const Range& x, const Range& y) { return x.a < y.a; });
  std::vector<Range> merged;
  for (const Range& r : rs) {
    // only overlapping ranges merge (a shared page => same mapping); merely
    // touching ranges may belong to different shard mappings
    if (!merged.empty() && r.a < merged.back().b)
      merged.back().b = std::max(merged.back().b, r.b);
    else
      merged.push_back(r);
  }
#ifndef _WIN32
  for (const Range& r : merged) {
    hipError_t re = hipHostRegister((void*)r.a, r.b - r.a, hipHostRegisterMapped);
    if (re != hipSuccess)
      throw std::runtime_error(std::string("QWENOX_GGUF: hipHostRegister: ") +
                               hipGetErrorString(re));
  }
#endif
  g_ggx = std::move(trunk);
  g_ggx_mtp = mtpx;
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::string mix;  // per-layer type mix, e.g. "IQ3_S/IQ4_NL x42, IQ4_XS/Q8_0 x1, ..."
  {
    std::vector<std::pair<std::string, int>> kinds;
    for (const auto& g : g_ggx) {
      const std::string k = std::string(gguf::type_name(g.tgu)) + "/" + gguf::type_name(g.tdn);
      auto it = std::find_if(kinds.begin(), kinds.end(), [&](auto& p) { return p.first == k; });
      if (it == kinds.end()) kinds.push_back({k, 1});
      else it->second++;
    }
    for (auto& k : kinds) mix += (mix.empty() ? "" : ", ") + k.first + " x" + std::to_string(k.second);
  }
  (void)n_q5k;
  (void)n_q80;
#ifdef _WIN32
  const char* how = "copied into the device arena";
#else
  const char* how = "registered";
#endif
  fprintf(stderr, "gguf experts: %zu layers from %s%s (%.1f GiB, %zu ranges %s, %.1f s; gate+up/down: %s)\n",
          g_ggx.size(), path, gg_mtp_on() ? " + MTP sidecar" : "", bytes / 1073741824.0,
          merged.size(), how, s, mix.c_str());
}

[[maybe_unused]] static size_t gguf_experts_arena_bytes(const Checkpoint& ck, size_t* hgn_replaced) {
  *hgn_replaced = 0;
#ifdef _WIN32
  gguf_experts_open();  // Windows: tensor lookup only (no registration), idempotent
#endif
  if (!gg_on()) return 0;
  size_t b = 0;
  for (const auto& g : g_ggx)
    for (const gguf::Tensor* t : g.src) b += (t->nbytes + 255) & ~(size_t)255;
  if (gg_mtp_on())
    for (const gguf::Tensor* t : g_ggx_mtp.src) b += (t->nbytes + 255) & ~(size_t)255;
  for (const auto& kv : ck.tensors())
    if (is_gguf_experts(kv.first)) *hgn_replaced += (kv.second.data_size + 255) & ~(size_t)255;
  return b;
}

// --- GGUF non-expert weights (G3.1, QWENOX_GGUF + QWENOX_GGUF_DENSE=1) ----------
// Dense/norm/router/HC/PLE-projection tensors are rebuilt from the GGUF under
// their hgn names (src/gguf_map.h: Q8_0 -> dtype 8 q8g32, F32 vectors ->
// dtype 1) and override the hgn records, so everything downstream (arena,
// gemv/gemm dispatch, small fp32 uploads) is unchanged. QWENOX_GGUF_MTP=<mtp
// sidecar gguf> also replaces the MTP head's non-expert tensors. The PLE
// n-gram table becomes a view of the GGUF IQ4_NL table when there is no hgn
// base (pure GGUF start) or QWENOX_GGUF_PLE=1. The synthetic host copies of
// large tensors are dropped after upload (gguf_dense_release).
static bool g_gguf_dense = false;
// 8-bit dense weights (GGUF dense, or an hgn overlay with q8g32 dense tensors,
// tools/hgn_hq.py): few-row bf16 matrices (HC inject 4x10240, in_proj_a/b
// 48x2560) take the one-block-per-row gemv kernels. Old 4-bit hgn files have
// no dtype 8 and keep the original kernels (bit-exact vs earlier builds).
static bool g_dense_rowfast = false;
static void gguf_dense_apply(Checkpoint& ck) {
  if (!g_dense_rowfast)
    for (auto& kv : ck.tensors())
      if (kv.second.dtype == 8 && kv.first.find(".mlp.experts.") == std::string::npos) {
        g_dense_rowfast = true;
        fprintf(stderr, "dense: 8-bit (q8g32) weights present, few-row bf16 gemv on\n");
        break;
      }
  const char* on = getenv("QWENOX_GGUF_DENSE");
  if (!on || !*on || !strcmp(on, "0")) return;
  const char* path = getenv("QWENOX_GGUF");
  if (!path || !*path) throw std::runtime_error("QWENOX_GGUF_DENSE needs QWENOX_GGUF=<first shard>");
  auto t0 = std::chrono::steady_clock::now();
  if (!g_gguf) g_gguf = new gguf::File(path);
  const char* mp = getenv("QWENOX_GGUF_MTP");
  if (mp && *mp && !g_gguf_mtp) g_gguf_mtp = new gguf::File(mp);
  // Debug bisect: QWENOX_GGUF_DENSE_FILTER=sub1,sub2,...  only hgn names
  // containing one of the substrings come from GGUF ("!" prefix = invert).
  if (const char* f = getenv("QWENOX_GGUF_DENSE_FILTER"); f && *f) {
    bool inv = *f == '!';
    std::vector<std::string> subs;
    for (std::string s(f + inv); !s.empty();) {
      size_t c = s.find(',');
      if (c) subs.push_back(s.substr(0, c));
      s = c == std::string::npos ? "" : s.substr(c + 1);
    }
    gguf_map::g_keep = [subs, inv](const std::string& nm) {
      for (auto& s : subs)
        if (nm.find(s) != std::string::npos) return !inv;
      return inv;
    };
    fprintf(stderr, "gguf dense: filter %s\n", f);
  }
  // QWENOX_GGUF_NATIVE=0: every non-Q8_0 dense matrix is requantized to q8g32
  // (the earlier layout). Default: K-quant / IQ4 matrices stay native.
  {
    const char* ne = getenv("QWENOX_GGUF_NATIVE");
    gguf_map::g_native_dense = !(ne && ne[0] == '0' && ne[1] == 0);
  }
  size_t n = gguf_map::build(ck, *g_gguf, g_gguf_mtp);
  // PLE n-gram table: a hgn base keeps its fp8 table by default (KLD 0.0453
  // vs 0.0511 with the GGUF's IQ4_NL one); pure GGUF / QWENOX_GGUF_PLE=1 use IQ4_NL.
  const char* pe = getenv("QWENOX_GGUF_PLE");
  const bool ple = (pe && !strcmp(pe, "1")) || !ck.find("layers.1.ple.ngram_embedding.weight");
  if (ple) gguf_map::add_ple_table(ck, *g_gguf);
  size_t bytes = 0, nbytes = 0;
  int n8 = 0, nn = 0;
  for (auto& kv : ck.tensors()) {
    const uint32_t dt = kv.second.dtype;
    if (dt == 8 || dt == 1) {
      bytes += kv.second.data_size;
      n8 += dt == 8;
    } else if (dt >= 100 && ggd::type_ok((int)dt - ggd::kDtypeBase)) {
      bytes += kv.second.data_size;
      nbytes += kv.second.data_size;
      nn++;
    }
  }
  g_gguf_dense = true;
  g_dense_rowfast = true;
  double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  fprintf(stderr,
          "gguf dense: %zu tensors from GGUF%s (%d q8g32, %d native %.2f GiB, %.2f GiB synthetic, "
          "%.1f s)%s\n",
          n, g_gguf_mtp ? " + MTP sidecar" : "", n8, nn, nbytes / 1073741824.0,
          bytes / 1073741824.0, s, ple ? "; PLE table IQ4_NL" : "");
}
// After GpuModel construction: free host copies of large synthetic tensors
// (the device arena has them). Small ones stay: ensure_hcnorm/ensure_norm256
// upload norms lazily from the host record.
static void gguf_dense_release(Checkpoint& ck) {
  if (!g_gguf_dense) return;
  size_t f = ck.release_owned([](const Tensor& t) { return t.data_size < ((size_t)1 << 20); });
  fprintf(stderr, "gguf dense: released %.2f GiB of host copies\n", f / 1073741824.0);
#ifdef _WIN32
  // the converted tensors' source pages (read through the shard mappings)
  // are dead too: take them out of the working set (gguf::File::trim_resident)
  size_t t = 0;
  if (g_gguf) t += g_gguf->trim_resident();
  if (g_gguf_mtp) t += g_gguf_mtp->trim_resident();
  if (t) fprintf(stderr, "gguf dense: trimmed mapped shard pages from the working set (%.1f GiB mapped)\n",
                 t / 1073741824.0);
#endif
}

// --- weight arena: copy trunk tensors (except the 47.7 GiB PLE table and the
// tiny u64 config arrays) into one big device allocation at startup ----------
static std::unordered_map<std::string, const uint8_t*> g_devtensor;

// parallel loader: N worker threads memcpy page-cache -> pinned staging -> H2D
static void load_arena(const Checkpoint& ck) {
  // Expert weights (~65 GiB) never enter the arena: they are hipHostRegister'd
  // per tensor and kernels read the host pointers directly (gfx1151 unified
  // memory, 220 GB/s measured, same DRAM as GTT, see tools/tgv2_probe*.cu).
  // Registration is mandatory: direct reads of non-resident pages fault.
  // Per-tensor registration of 65 GiB takes ~2.6 s with no GTT shadow.
  // MoE sparsity (10/512 experts per token) makes the sparse reads cheap, and
  // TG/PP measured at parity with the old all-in-arena form, which is gone.
  // Windows: hipHostRegister 这条直读路不存在（65 GiB pin 进 32 GiB RAM 物理
  // 不可能）；专家与普通张量一样拷贝进 devarena，VRAM 直读（同一片 DRAM）。
  auto is_experts = [](const std::string& n) {
    return n.find(".mlp.experts.") != std::string::npos;
  };
  gguf_experts_open();
  size_t total = 0, xbytes = 0;
  for (auto& kv : ck.tensors()) {
    const Tensor& t = kv.second;
    if (is_ple_offarena(t)) continue;  // PLE table / u64 config
    if (t.dtype == 2) continue;  // hgn v2 f16 sidecars: v2_init uploads them as f32
    if (is_gguf_experts(t.name)) continue;  // served from GGUF
#ifndef _WIN32
    if (is_experts(t.name)) {
      hipError_t re =
          hipHostRegister((void*)t.data, t.data_size, hipHostRegisterMapped);
      if (re != hipSuccess)
        throw std::runtime_error(std::string("hipHostRegister ") + t.name +
                                 ": " + hipGetErrorString(re));
      g_devtensor[t.name] = t.data;  // 注册后 host 指针即设备指针,kernel 直读
      xbytes += t.data_size;
      continue;
    }
#endif
    total += (t.data_size + 255) & ~255ull;
  }
  // Windows: GGUF routed experts (trunk + MTP sidecar) are copied into the
  // same arena (no hipHostRegister direct-read path, see above); their
  // GgExperts pointers are re-pointed below.
  std::vector<GgExperts*> ggx_arena;
#ifdef _WIN32
  for (auto& g : g_ggx) ggx_arena.push_back(&g);
  if (gg_mtp_on()) ggx_arena.push_back(&g_ggx_mtp);
  size_t ggbytes = 0;
  for (GgExperts* g : ggx_arena)
    for (const gguf::Tensor* t : g->src) ggbytes += (t->nbytes + 255) & ~255ull;
  total += ggbytes;
  if (ggbytes)
    fprintf(stderr, "weight arena: GGUF experts %.1f GiB copied in\n", ggbytes / 1073741824.0);
#endif
  fprintf(stderr, "weight arena: %.1f GiB (experts direct-read: %.1f GiB)\n",
          total / 1073741824.0, xbytes / 1073741824.0);
  uint8_t* arena = nullptr;
  DALLOC(&arena, total);

  // src = host view (mmap / synthetic). Windows: os_handle + foff = the file
  // range behind src for the unbuffered ReadFile path; nullptr = host memory
  // (gguf_map synthetic tensors), copied with memcpy.
  struct Job {
    std::string name;
    const uint8_t* src;
    uint8_t* dst;
    size_t size;
    void* os_handle = nullptr;
    uint64_t foff = 0;
  };
  std::vector<Job> jobs;
  size_t off = 0;
  for (auto& kv : ck.tensors()) {
    const Tensor& t = kv.second;
    if (is_ple_offarena(t)) continue;
    if (t.dtype == 2) continue;
    if (is_gguf_experts(t.name)) continue;
#ifndef _WIN32
    if (is_experts(t.name)) continue;  // 已在上面登记 host 指针
#endif
    Job j{t.name, t.data, arena + off, t.data_size};
    for (auto& mp : ck.mappings())
      if (t.data >= mp.base && t.data < mp.base + mp.len) {
        j.os_handle = mp.os_handle;
        j.foff = (uint64_t)(t.data - mp.base);
        break;
      }
    jobs.push_back(j);
    g_devtensor[t.name] = arena + off;
    off += (t.data_size + 255) & ~255ull;
  }
  for (GgExperts* g : ggx_arena) {
    const uint8_t** ptrs[3] = {&g->gate, &g->up, &g->down};
    for (int i = 0; i < 3; i++) {
      const gguf::Tensor* t = g->src[i];
      const gguf::File* f = (g == &g_ggx_mtp) ? g_gguf_mtp : g_gguf;
      Job j{t->name, t->data, arena + off, (size_t)t->nbytes};
      j.os_handle = f->mappings()[t->shard].os_handle;
      j.foff = t->file_off;
      jobs.push_back(j);
      *ptrs[i] = arena + off;
      off += (t->nbytes + 255) & ~255ull;
    }
  }
  // big tensors first so stragglers don't serialize at the end
  std::sort(jobs.begin(), jobs.end(),
            [](const Job& a, const Job& b) { return a.size > b.size; });

  const int NW = 4;
  std::atomic<size_t> next{0}, done_bytes{0};
  std::atomic<long long> pload_last{0};
  // 0.5 s cadence: \r on a terminal, one timestamped line per tick in log
  // files (unthrottled \r fragments turn logs into a single blob).
  auto prog_tick = [&] {
    auto nowms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now().time_since_epoch())
                     .count();
    long long prev = pload_last.load();
    if (nowms - prev >= 500 && pload_last.compare_exchange_strong(prev, nowms)) {
      if (isatty(2))
        fprintf(stderr, "\rpload %.1f/%.1f GiB", done_bytes / 1073741824.0,
                total / 1073741824.0);
      else
        fprintf(stderr, "%s pload %.1f/%.1f GiB\n", log_ts(),
                done_bytes / 1073741824.0, total / 1073741824.0);
      fflush(stderr);
    }
  };
#ifdef _WIN32
  // Windows: memcpy 缺页读在主机 RAM 紧张时因 page cache 逐出抖动退化到
  // ~30 MB/s。改用 hgn.h 保持打开的 OVERLAPPED|NO_BUFFERING 句柄直接
  // ReadFile 进 pinned slot（4 线程 x 4 槽 x 16 MiB），再 H2D —— 完全不进
  // page cache，无逐出抖动。扇区对齐：offset/length/buffer 均按 4096。
  constexpr size_t WCHUNK = 16ull << 20;
  constexpr int WSLOTS = 4;
  struct WSlot {
    OVERLAPPED ov{};
    uint8_t* alloc = nullptr;
    uint8_t* buf = nullptr;  // 4096-aligned inside alloc
    bool busy = false;
    uint64_t c0 = 0;         // 该槽本次读的文件起始偏移
  };
  WSlot wslots[NW][WSLOTS];
  hipStream_t st[NW];
  for (int i = 0; i < NW; i++) {
    CK(hipStreamCreate(&st[i]));
    for (auto& s : wslots[i]) {
      s.ov.hEvent = CreateEventA(nullptr, TRUE, FALSE, nullptr);
      CK(hipHostMalloc((void**)&s.alloc, WCHUNK + 4096, hipHostMallocDefault));
      s.buf = (uint8_t*)(((uintptr_t)s.alloc + 4095) & ~(uintptr_t)4095);
    }
  }
  auto wworker = [&](int wid) {
    hip_use_selected_device();
    WSlot* slots = wslots[wid];
    for (;;) {
      size_t ji = next.fetch_add(1);
      if (ji >= jobs.size()) break;
      const Job& j = jobs[ji];
      // 所属映射 -> （句柄， 文件偏移）：预先解析在 Job 里
      HANDLE h = (HANDLE)j.os_handle;
      const uint64_t foff = j.foff;
      if (!h) {
        // 无文件句柄 = 主机内存里的合成张量（gguf_map）：memcpy 进 pinned 槽再 H2D
        WSlot& s = slots[0];
        for (size_t c = 0; c < j.size; c += WCHUNK) {
          const size_t n = std::min(WCHUNK, j.size - c);
          memcpy(s.buf, j.src + c, n);
          CK(hipMemcpyAsync(j.dst + c, s.buf, n, hipMemcpyHostToDevice, st[wid]));
          CK(hipStreamSynchronize(st[wid]));
          done_bytes += n;
          prog_tick();
        }
        continue;
      }
      const uint64_t A = foff & ~4095ull, E = (foff + j.size + 4095) & ~4095ull;
      int si = 0;
      auto drain = [&](WSlot& s) {  // 等完成 + H2D 落在张量区间内的部分
        DWORD got = 0;
        if (!GetOverlappedResult(h, &s.ov, &got, TRUE)) {
          fprintf(stderr, "pload: overlapped read failed on %s: %lu\n",
                  j.name.c_str(), GetLastError());
          abort();
        }
        s.busy = false;
        uint64_t lo = std::max(s.c0, foff),
                 hi = std::min(s.c0 + got, foff + j.size);
        if (hi > lo) {
          CK(hipMemcpyAsync(j.dst + (lo - foff), s.buf + (lo - s.c0), hi - lo,
                            hipMemcpyHostToDevice, st[wid]));
          CK(hipStreamSynchronize(st[wid]));
          done_bytes += hi - lo;
          prog_tick();
        }
      };
      for (uint64_t c = A; c < E; c += WCHUNK) {
        WSlot& s = slots[si];
        si = (si + 1) % WSLOTS;
        if (s.busy) drain(s);
        DWORD len = (DWORD)std::min<uint64_t>(WCHUNK, E - c);
        ResetEvent(s.ov.hEvent);
        s.ov.Offset = (DWORD)(c & 0xffffffffu);
        s.ov.OffsetHigh = (DWORD)(c >> 32);
        s.c0 = c;
        BOOL r = ReadFile(h, s.buf, len, nullptr, &s.ov);
        if (!r) {
          DWORD e = GetLastError();
          if (e == ERROR_HANDLE_EOF) continue;  // 整块在 EOF 之外（未排队，勿 drain）
          if (e != ERROR_IO_PENDING) {
            fprintf(stderr, "pload: ReadFile %s failed: %lu\n", j.name.c_str(), e);
            abort();
          }
        }
        s.busy = true;
      }
      for (int k = 0; k < WSLOTS; k++)
        if (slots[k].busy) drain(slots[k]);
    }
  };
  std::vector<std::thread> ths;
  for (int i = 0; i < NW; i++) ths.emplace_back(wworker, i);
  for (auto& t : ths) t.join();
  CK(hipDeviceSynchronize());
  for (int i = 0; i < NW; i++) {
    for (auto& s : wslots[i]) {
      CK(hipHostFree(s.alloc));
      CloseHandle(s.ov.hEvent);
    }
    CK(hipStreamDestroy(st[i]));
  }
#else
  const size_t STAGE = 64ull << 20;
  uint8_t* stage[NW];
  hipStream_t st[NW];
  for (int i = 0; i < NW; i++) {
    CK(hipHostMalloc(&stage[i], STAGE, hipHostMallocDefault));
    CK(hipStreamCreate(&st[i]));
  }
  auto worker = [&](int wid) {
    hip_use_selected_device();
    for (;;) {
      size_t ji = next.fetch_add(1);
      if (ji >= jobs.size()) break;
      const Job& j = jobs[ji];
      size_t copied = 0;
      while (copied < j.size) {
        size_t n = std::min(STAGE, j.size - copied);
        memcpy(stage[wid], j.src + copied, n);  // faults page cache in
        CK(hipMemcpyAsync(j.dst + copied, stage[wid], n, hipMemcpyHostToDevice,
                          st[wid]));
        CK(hipStreamSynchronize(st[wid]));
        copied += n;
        done_bytes += n;
        prog_tick();
      }
    }
  };
  std::vector<std::thread> ths;
  for (int i = 0; i < NW; i++) ths.emplace_back(worker, i);
  for (auto& t : ths) t.join();
  CK(hipDeviceSynchronize());
  for (int i = 0; i < NW; i++) {
    CK(hipHostFree(stage[i]));
    CK(hipStreamDestroy(st[i]));
  }
#endif
  if (isatty(2)) fprintf(stderr, "\n");
  fprintf(stderr, "%s pload done: %.1f GiB\n", log_ts(), total / 1073741824.0);
}

// device pointer for a tensor's data
static const uint8_t* dev_ptr(const Tensor& t) {
  auto it = g_devtensor.find(t.name);
  if (it == g_devtensor.end())
    throw std::runtime_error("tensor not in arena: " + t.name);
  return it->second;
}

// --- device weight view for GEMV dispatch ------------------------------------
struct WView {
  int dtype;
  const uint8_t* data;  // device pointer
  uint64_t rows, cols, scale_stride;
  // hgn v2 rotated dtypes (16 ht, 23 i4r): input / output sign-scale planes
  // (f32 device copies of the f16 .suh / .svh sidecars), else nullptr.
  const float* suh = nullptr;
  const float* svh = nullptr;
};

// --- hgn v2 host side ---------------------------------------------------------
// v1 / v2 is decided per tensor by its own dtype (16 / 23 / 24 exist only in
// v2), so one engine binary serves both and mixed files need no flag.
struct V2Rot {
  const float* suh = nullptr;
  const float* svh = nullptr;
};
static std::unordered_map<std::string, V2Rot> g_v2rot;  // weight name -> planes
static float* g_i4r_cb = nullptr;   // linear codebook -8..7 for the q4cp-style kernels
static float* g_ht_zs = nullptr;    // split-K partial z, zero between launches
static int* g_ht_cnt = nullptr;     // split-K per-tile-row arrival counters
static constexpr size_t kHtZs = 16ull << 20;  // split-K slots: ks * P * O floats
static bool g_hgn_v2 = false;

static float f16_to_f32_host(uint16_t h) {
  const uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
  uint32_t u;
  if (e == 0) {
    if (!m) {
      u = s;
    } else {  // subnormal
      int k = -1;
      uint32_t mm = m;
      do { k++; mm <<= 1; } while (!(mm & 1024));
      u = s | ((uint32_t)(127 - 15 - k) << 23) | ((mm & 1023) << 13);
    }
  } else if (e == 31) {
    u = s | 0x7f800000u | (m << 13);
  } else {
    u = s | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// Upload every rotated weight's planes once (f16 -> f32), build the i4r
// codebook and split-K scratch. No-op on v1 checkpoints. Call after load_arena.
float* g_q6g_part = nullptr;  // q6g64 split-K partials [16][8][2048]
extern float* g_q8w_part;
static constexpr size_t kQ8wPartAlloc = (size_t)4 << 20;  // floats (== kQ8wPartFloats)
static void v2_init(const Checkpoint& ck) {
  if (!g_q6g_part) CK(hipMalloc(&g_q6g_part, (size_t)16 * 8 * 2048 * 4));
  // q8g32 WMMA split-K partials (k_q8g32_wmma); before any graph capture
  if (!g_q8w_part) CK(hipMalloc(&g_q8w_part, kQ8wPartAlloc * 4));
  std::unordered_map<std::string, const float*> up;  // sidecar name -> f32 copy
  auto sidecar = [&](const std::string& n, uint64_t want) -> const float* {
    auto it = up.find(n);
    if (it != up.end()) return it->second;
    const Tensor* t = ck.find(n);
    if (!t) return nullptr;
    if (t->dtype != 2 || t->numel() != want)
      throw std::runtime_error("hgn v2: bad sidecar " + n);
    std::vector<float> h(want);
    const uint16_t* src = (const uint16_t*)t->data;
    for (uint64_t i = 0; i < want; i++) h[i] = f16_to_f32_host(src[i]);
    float* d = nullptr;
    CK(hipMalloc(&d, want * 4));
    CK(hipMemcpy(d, h.data(), want * 4, hipMemcpyHostToDevice));
    return up[n] = d;
  };
  int n16 = 0, n23 = 0, n24 = 0;
  for (auto& kv : ck.tensors()) {
    const Tensor& t = kv.second;
    if (t.dtype == 24) n24++;
    if (t.dtype != 16 && t.dtype != 23) continue;
    const uint64_t K = t.dims[t.ndims - 1], O = t.dims[t.ndims - 2];
    if (K % 128 || O % 128)
      throw std::runtime_error("hgn v2: " + t.name + " off the 128 grid");
    std::string base = t.name;
    const std::string sfx = ".weight";
    if (base.size() > sfx.size() && base.compare(base.size() - sfx.size(), sfx.size(), sfx) == 0)
      base.resize(base.size() - sfx.size());
    V2Rot r;
    r.suh = sidecar(base + ".suh", K);
    if (!r.suh) {  // experts.gate_up_proj: input plane shared as experts.su
      const size_t dot = base.rfind('.');
      if (dot != std::string::npos) r.suh = sidecar(base.substr(0, dot) + ".su", K);
    }
    r.svh = sidecar(base + ".svh", O);
    if (!r.suh || !r.svh) throw std::runtime_error("hgn v2: missing .suh/.svh for " + t.name);
    g_v2rot[t.name] = r;
    (t.dtype == 16 ? n16 : n23)++;
  }
  if (!n16 && !n23 && !n24) return;
  g_hgn_v2 = true;
  const float cb[16] = {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
  CK(hipMalloc(&g_i4r_cb, sizeof cb));
  CK(hipMemcpy(g_i4r_cb, cb, sizeof cb, hipMemcpyHostToDevice));
  CK(hipMalloc(&g_ht_zs, kHtZs * 4));
  CK(hipMemset(g_ht_zs, 0, kHtZs * 4));
  CK(hipMalloc(&g_ht_cnt, 4096 * 4));
  CK(hipMemset(g_ht_cnt, 0, 4096 * 4));
  fprintf(stderr, "hgn v2: %d ht dense, %d i4r experts, %d q6g64 HC, %zu sidecars (f32)\n",
          n16, n23, n24, up.size());
}

static WView wview(const Tensor& t) {
  WView w;
  w.dtype = t.dtype;
  w.data = dev_ptr(t);
  w.cols = t.dims[t.ndims - 1];
  w.rows = t.numel() / w.cols;
  if (t.dtype == 5) {
    uint64_t codes_bytes = w.rows * w.cols / 2;
    w.scale_stride = ((w.cols / 32 * 2) + 15) & ~15ull;
    (void)codes_bytes;
  } else if (t.dtype == 23) {
    w.scale_stride = w.cols / 128 * 2;  // unpadded f16 per 128 cols
  } else if (t.dtype == 24) {
    w.scale_stride = q6g64_stride(w.cols);  // full interleaved row stride
  } else {
    w.scale_stride = 0;
  }
  if (t.dtype == 16 || t.dtype == 23) {
    auto it = g_v2rot.find(t.name);
    if (it == g_v2rot.end()) throw std::runtime_error("hgn v2: planes not loaded for " + t.name);
    w.suh = it->second.suh;
    w.svh = it->second.svh;
  }
  return w;
}

// ht GEMV for P rows: x row p at x + p*xs (f32 or bf16), y row p at y + p*ys.
// row_off must sit on the 128 grid (codes are tiled by 128 rows).
template <int P>
static void ht_gemv_p(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                      uint64_t ys, uint64_t row_off, uint64_t nrows) {
  if (row_off % 128) throw std::runtime_error("ht gemv: row_off off the 128 grid");
  const int K = (int)w.cols, ntx = K / 128;
  const int nty = (int)((nrows + 127) / 128);
  const uint8_t* codes = w.data + (row_off / 128) * (uint64_t)ntx * 8192;
  const float* svh = w.svh + row_off;
  // Split K only up to ~80 blocks (2 per CU): more splits cost more in atomics
  // than they win (microbench 10240x2048 / 2048x4096). LDS holds tpc tiles
  // of rotated x per pass (<= 24 tile-rows) or the 8 wave partials.
  int ks = std::max(1, std::min((80 + nty - 1) / nty, ntx));
  // deterministic split-K needs ks separate slots: shrink ks to fit
  while (ks > 1 && (size_t)ks * P * nty * 128 > kHtZs) ks = (ks + 1) / 2;
  const int tpb = (ntx + ks - 1) / ks;
  ks = (ntx + tpb - 1) / tpb;
  const int tpc = std::min(tpb, std::max(1, 24 / P));
  if (ks > 1 && ((size_t)ks * P * nty * 128 > kHtZs || nty > 4096))
    throw std::runtime_error("ht gemv: split-K scratch too small");
  const size_t lds = (size_t)std::max(P * tpc, 8 * P) * 128 * 4;
  const dim3 grid((unsigned)nty, (unsigned)ks);
  if (xbf16)
    k_ht_gemv<P, uint16_t><<<grid, 256, lds, g_str>>>(codes, w.suh, svh, (const uint16_t*)x, xs,
                                                       y, ys, nty * 128, K, (int)nrows, tpb,
                                                       tpc, g_ht_zs, g_ht_cnt);
  else
    k_ht_gemv<P, float><<<grid, 256, lds, g_str>>>(codes, w.suh, svh, (const float*)x, xs, y,
                                                    ys, nty * 128, K, (int)nrows, tpb, tpc,
                                                    g_ht_zs, g_ht_cnt);
}
static bool ht_gemv(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                    uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  switch (P) {
#define QWENOX_HTG(PP) \
  case PP: ht_gemv_p<PP>(w, x, xbf16, xs, y, ys, row_off, nrows); return true;
    QWENOX_HTG(1) QWENOX_HTG(2) QWENOX_HTG(3) QWENOX_HTG(4)
    QWENOX_HTG(5) QWENOX_HTG(6) QWENOX_HTG(7) QWENOX_HTG(8)
#undef QWENOX_HTG
    default:
      return false;
  }
}

// --- dense WMMA ht GEMV (k_ht_wmma), row-invariant for P = 1..48 ------------
// Split-K count from the shape only (never from P): the K slicing and the kz
// summation order are part of every row's numerics, so they must not change
// with the batch. Scratch is sized for the largest P (48).
static constexpr int kHtWmmaMaxP = 48;
static int ht_wmma_blocks() {
  static const int v = [] {
    const char* e = getenv("QWENOX_HT_WMMA_BLOCKS");
    return e && *e ? std::max(1, atoi(e)) : 80;
  }();
  return v;
}
static void ht_wmma_split(int ntx, int nty, int& ks, int& tpb) {
  ks = std::max(1, std::min((ht_wmma_blocks() + nty - 1) / nty, ntx));
  while (ks > 1 && (size_t)ks * kHtWmmaMaxP * nty * 128 > kHtZs) ks = (ks + 1) / 2;
  tpb = (ntx + ks - 1) / ks;
  ks = (ntx + tpb - 1) / tpb;
}
template <int NPT>
static void ht_wmma_p(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                      uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  if (row_off % 128) throw std::runtime_error("ht wmma: row_off off the 128 grid");
  const int K = (int)w.cols, ntx = K / 128;
  const int nty = (int)((nrows + 127) / 128);
  const uint8_t* codes = w.data + (row_off / 128) * (uint64_t)ntx * 8192;
  const float* svh = w.svh + row_off;
  int ks, tpb;
  ht_wmma_split(ntx, nty, ks, tpb);
  if (ks > 1 && ((size_t)ks * P * nty * 128 > kHtZs || nty > 4096))
    throw std::runtime_error("ht wmma: split-K scratch too small");
  // x^ chunk: <= ~QWENOX_HT_WMMA_LDS_KB of f16 for the P live rows per pass
  // (staging only, no effect on numerics): P=1 holds the whole K slice, one
  // rotate + barrier per block like k_ht_gemv.
  static const int ldskb = [] {
    const char* e = getenv("QWENOX_HT_WMMA_LDS_KB");
    return e && *e ? std::max(4, atoi(e)) : 24;
  }();
  const int tpc = std::max(1, std::min(tpb, ldskb * 1024 / (P * 128 * 2)));
  const size_t lds = std::max((size_t)P * (tpc * 128 + 8) * 2, (size_t)P * 128 * 4);
  const dim3 grid((unsigned)nty, (unsigned)ks);
  if (xbf16)
    k_ht_wmma<NPT, uint16_t><<<grid, 256, lds, g_str>>>(codes, w.suh, svh, (const uint16_t*)x,
                                                        xs, y, ys, nty * 128, K, (int)nrows,
                                                        tpb, tpc, P, g_ht_zs, g_ht_cnt);
  else
    k_ht_wmma<NPT, float><<<grid, 256, lds, g_str>>>(codes, w.suh, svh, (const float*)x, xs, y,
                                                     ys, nty * 128, K, (int)nrows, tpb, tpc, P,
                                                     g_ht_zs, g_ht_cnt);
}
// QWENOX_DENSE_WMMA (default on; =0 restores the fp32 k_ht_gemv path): every
// v2 ht GEMV of P <= 48 rows (decode gemv, draft, spec verify, batched verify,
// small prefill chunks) runs k_ht_wmma instead of the fp32 k_ht_gemv. One
// row-invariant kernel for all P keeps concurrent == single-run bit for bit;
// numerics differ from the fp32 path (x^ -> f16). KLD 10-12 (Linux, v2,
// BF16 ref ctx512x64, QWENOX_PREFILL_CHUNK=24): off 0.107830 -> on 0.106627,
// top-1 88.96 -> 89.03 %.
static bool dense_wmma_on() {
  static const bool v = [] {
    const char* e = getenv("QWENOX_DENSE_WMMA");
    const bool on = !(e && e[0] == '0' && e[1] == 0);
    if (on) fprintf(stderr, "dense wmma: v2 ht GEMV P<=%d on k_ht_wmma (QWENOX_DENSE_WMMA)\n", 48);
    return on;
  }();
  return v;
}
// QWENOX_DENSE_WMMA_HC=0: keep the q6g64 HC projections on the fp32 path while
// dense WMMA is on (A/B; default follows QWENOX_DENSE_WMMA).
static bool dense_wmma_hc_on() {
  static const bool v = [] {
    const char* e = getenv("QWENOX_DENSE_WMMA_HC");
    const bool on = dense_wmma_on() && !(e && e[0] == '0' && e[1] == 0);
    if (on) fprintf(stderr, "dense wmma: q6g64 HC GEMV P<=48 on k_q6g64_wmma\n");
    return on;
  }();
  return v;
}
// Pair launch: two ht weights with the same K reading the same x (shared
// expert gate + up) in one k_ht_wmma grid; split-K from the combined shape.
// Row-invariant like ht_wmma (shape-only config), but NOT bit-identical to two
// separate ht_wmma launches (ks differs) -- callers use it for every P.
static bool ht_wmma_pair(const WView& a, const WView& b, const void* x, bool xbf16, uint64_t xs,
                         float* ya, float* yb, uint64_t ys, int P) {
  if (P < 1 || P > kHtWmmaMaxP || a.dtype != 16 || b.dtype != 16 || a.cols != b.cols)
    return false;
  if (a.rows % 128 || b.rows % 128) return false;
  if (((uintptr_t)x | (uintptr_t)ya | (uintptr_t)yb) & 15) return false;
  const int K = (int)a.cols, ntx = K / 128;
  const int na = (int)(a.rows / 128), nb = (int)(b.rows / 128), nty = na + nb;
  int ks, tpb;
  ht_wmma_split(ntx, nty, ks, tpb);
  if (ks > 1 && ((size_t)ks * P * nty * 128 > kHtZs || nty > 4096)) return false;
  static const int ldskb = [] {
    const char* e = getenv("QWENOX_HT_WMMA_LDS_KB");
    return e && *e ? std::max(4, atoi(e)) : 24;
  }();
  const int tpc = std::max(1, std::min(tpb, ldskb * 1024 / (P * 128 * 2)));
  const size_t lds = std::max((size_t)P * (tpc * 128 + 8) * 2, (size_t)P * 128 * 4);
  const dim3 grid((unsigned)nty, (unsigned)ks);
#define QWENOX_HTP(NPT)                                                                       \
  do {                                                                                        \
    if (xbf16)                                                                                \
      k_ht_wmma<NPT, uint16_t><<<grid, 256, lds, g_str>>>(                                    \
          a.data, a.suh, a.svh, (const uint16_t*)x, xs, ya, ys, nty * 128, K, (int)a.rows,     \
          tpb, tpc, P, g_ht_zs, g_ht_cnt, na, b.data, b.suh, b.svh, yb, (int)b.rows);         \
    else                                                                                      \
      k_ht_wmma<NPT, float><<<grid, 256, lds, g_str>>>(                                       \
          a.data, a.suh, a.svh, (const float*)x, xs, ya, ys, nty * 128, K, (int)a.rows, tpb,   \
          tpc, P, g_ht_zs, g_ht_cnt, na, b.data, b.suh, b.svh, yb, (int)b.rows);              \
  } while (0)
  if (P <= 16) QWENOX_HTP(1);
  else if (P <= 32) QWENOX_HTP(2);
  else QWENOX_HTP(3);
#undef QWENOX_HTP
  return true;
}
static bool ht_wmma(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                    uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  if (P < 1 || P > kHtWmmaMaxP) return false;
  if (P <= 16) ht_wmma_p<1>(w, x, xbf16, xs, y, ys, P, row_off, nrows);
  else if (P <= 32) ht_wmma_p<2>(w, x, xbf16, xs, y, ys, P, row_off, nrows);
  else ht_wmma_p<3>(w, x, xbf16, xs, y, ys, P, row_off, nrows);
  return true;
}

// q6g64 GEMV for P rows (same x / y conventions); x 16 B aligned.
template <int P>
static void q6g64_gemv_p(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                         uint64_t ys, uint64_t row_off, uint64_t nrows) {
  const uint8_t* wq = w.data + row_off * w.scale_stride;
  const int R = (int)nrows, C = (int)w.cols, st = (int)w.scale_stride;
  if (C < 1024) {
    const unsigned blocks = (unsigned)((R + 31) / 32);
    if (xbf16)
      k_q6g64_gemv<P, uint16_t, 8><<<blocks, 256, 0, g_str>>>(wq, (const uint16_t*)x, xs, y, ys,
                                                              R, C, st);
    else
      k_q6g64_gemv<P, float, 8><<<blocks, 256, 0, g_str>>>(wq, (const float*)x, xs, y, ys, R,
                                                           C, st);
  } else if (C >= 4096 && R <= 2048 && [] {
               const char* e = getenv("QWENOX_Q6G_SPLITK");
               return !(e && e[0] == '0' && e[1] == 0);
             }()) {
    // aggr 10-05: split-K for long-K small-R (HC mix-down 320x10240).
    // D2-S1b: 门控不再带 P>=2——LPR=32 与 split-K 的归约树不同，P=1 走
    // LPR=32 会让同一行在 decode(P=1) 与 verify(P>=2) 间换 bits（ktest
    // ri_q6g64_320x10240/4x10240 实测 P1 vs P2..24 全部分叉）。行不变性
    // 要求 kernel 选择只是形状的函数；代价是 P=1 放弃 LPR=32（历史实测
    // decode -1.7% 的由来），split-K 在 P=1 略慢但逐位一致。
    // fixed-size scratch, allocated on first use before any graph capture
    // (q6g64_splitk_init from v2_init); C <= 32768 -> ns <= 16.
    const int ng = C >> 6, gps = 32;
    const int ns = (ng + gps - 1) / gps;
    if (!g_q6g_part || ns > 16) {
      fprintf(stderr, "q6g64 split-K scratch missing (ns=%d)\n", ns);
      abort();
    }
    float* const part = g_q6g_part;
    const dim3 grid((unsigned)((R + 7) / 8), (unsigned)ns);
    if (xbf16)
      k_q6g64_gemv_sk<P, uint16_t><<<grid, 256, 0, g_str>>>(wq, (const uint16_t*)x, xs, part, R,
                                                           C, st, gps);
    else
      k_q6g64_gemv_sk<P, float><<<grid, 256, 0, g_str>>>(wq, (const float*)x, xs, part, R, C,
                                                        st, gps);
    k_q6g64_skred<P><<<(unsigned)((P * R + 255) / 256), 256, 0, g_str>>>(part, y, ys, R, ns);
  } else if (C >= 8192 && [] {
               // opt-in only: measured slower (P=7 21.4 -> 28.3 ms, 10-05)
               const char* e = getenv("QWENOX_Q6G_ROW256");
               return e && !(e[0] == '0' && e[1] == 0);
             }()) {
    // aggr 10-05: one row per 256-lane block (see k_q6g64_gemv LPR=256)
    const unsigned blocks = (unsigned)R;
    if (xbf16)
      k_q6g64_gemv<P, uint16_t, 256><<<blocks, 256, 0, g_str>>>(wq, (const uint16_t*)x, xs, y,
                                                                ys, R, C, st);
    else
      k_q6g64_gemv<P, float, 256><<<blocks, 256, 0, g_str>>>(wq, (const float*)x, xs, y, ys, R,
                                                             C, st);
  } else {
    const unsigned blocks = (unsigned)((R + 7) / 8);
    if (xbf16)
      k_q6g64_gemv<P, uint16_t, 32><<<blocks, 256, 0, g_str>>>(wq, (const uint16_t*)x, xs, y,
                                                               ys, R, C, st);
    else
      k_q6g64_gemv<P, float, 32><<<blocks, 256, 0, g_str>>>(wq, (const float*)x, xs, y, ys, R,
                                                            C, st);
  }
}
static bool q6g64_gemv(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                       uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  if (w.cols % 64) return false;
  switch (P) {
#define QWENOX_Q6G(PP) \
  case PP: q6g64_gemv_p<PP>(w, x, xbf16, xs, y, ys, row_off, nrows); return true;
    QWENOX_Q6G(1) QWENOX_Q6G(2) QWENOX_Q6G(3) QWENOX_Q6G(4)
    QWENOX_Q6G(5) QWENOX_Q6G(6) QWENOX_Q6G(7) QWENOX_Q6G(8)
#undef QWENOX_Q6G
    default:
      return false;
  }
}

// --- dense WMMA q6g64 GEMV (k_q6g64_wmma), row-invariant for P = 1..48 ------
// Config from the shape only:
// - short K (C <= 1024, HC mix-up 10240x320): RW row tiles x KW group lanes
//   per block (QWENOX_Q6W_SK, default 4x2), no split;
// - long K (HC mix-down 320x10240): 8 waves across groups, ns K slices for
//   ~QWENOX_Q6W_BLOCKS (40) blocks.
// Sweep (logs/dw_q6_sweep.log, 10-12): 4x2 / 40 blocks best at P=5 and P=40.
// Shapes off the grid (R % 16, e.g. the 4-row inject) return false -> the
// caller's old path (still a function of the shape only).
static bool q6g64_wmma(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                       uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  if (P < 1 || P > 48) return false;
  const int R = (int)nrows, C = (int)w.cols, st = (int)w.scale_stride;
  if (C % 64 || R % 16) return false;
  const int ng = C / 64;
  const bool shortk = C <= 1024;
  // short-K wave layout RW x KW (QWENOX_Q6W_SK = 8 -> 8x1, 4 -> 4x2, 2 -> 2x4)
  static const int skrw = [] {
    const char* e = getenv("QWENOX_Q6W_SK");
    const int v = e && *e ? atoi(e) : 4;
    return (v == 8 || v == 4 || v == 2) ? v : 4;
  }();
  const int rw = shortk ? skrw : 1;
  if (R % (16 * rw)) return false;
  static const int nblk = [] {
    const char* e = getenv("QWENOX_Q6W_BLOCKS");
    return e && *e ? std::max(1, atoi(e)) : 40;
  }();
  int ns = 1;
  if (!shortk) {
    ns = std::max(1, std::min((nblk + R / 16 - 1) / (R / 16), ng));
    ns = std::max(ns, (ng + 63) / 64);  // gps <= 64 (LDS group sums)
  }
  int gps = (ng + ns - 1) / ns;
  ns = (ng + gps - 1) / gps;
  if (gps > 64) return false;
  if (ns > 1 && (size_t)ns * 48 * R > (size_t)16 * 8 * 2048) return false;
  if (ns > 1 && !g_q6g_part) return false;
  const uint8_t* wq = w.data + row_off * st;
  float* out = ns > 1 ? g_q6g_part : y;
  const dim3 grid((unsigned)(R / (16 * rw)), (unsigned)ns);
#define QWENOX_Q6W(NPT, RW, KW)                                                                \
  do {                                                                                         \
    if (xbf16)                                                                                 \
      k_q6g64_wmma<NPT, RW, KW, uint16_t><<<grid, 256, 0, g_str>>>(                            \
          wq, (const uint16_t*)x, xs, out, ys, R, C, st, gps, P);                              \
    else                                                                                       \
      k_q6g64_wmma<NPT, RW, KW, float><<<grid, 256, 0, g_str>>>(wq, (const float*)x, xs, out, \
                                                                ys, R, C, st, gps, P);         \
  } while (0)
  if (shortk && rw == 8) {
    if (P <= 16) QWENOX_Q6W(1, 8, 1);
    else if (P <= 32) QWENOX_Q6W(2, 8, 1);
    else QWENOX_Q6W(3, 8, 1);
  } else if (shortk && rw == 4) {
    if (P <= 16) QWENOX_Q6W(1, 4, 2);
    else if (P <= 32) QWENOX_Q6W(2, 4, 2);
    else QWENOX_Q6W(3, 4, 2);
  } else if (shortk) {
    if (P <= 16) QWENOX_Q6W(1, 2, 4);
    else if (P <= 32) QWENOX_Q6W(2, 2, 4);
    else QWENOX_Q6W(3, 2, 4);
  } else {
    if (P <= 16) QWENOX_Q6W(1, 1, 8);
    else if (P <= 32) QWENOX_Q6W(2, 1, 8);
    else QWENOX_Q6W(3, 1, 8);
  }
#undef QWENOX_Q6W
  if (ns > 1)
    k_q6g64_wred<<<(unsigned)((P * R + 255) / 256), 256, 0, g_str>>>(g_q6g_part, y, ys, R, ns, P);
  return true;
}

// --- dense WMMA q8g32 GEMV (k_q8g32_wmma), row-invariant for P = 1..48 ------
// GGUF Q8_0 dense (and q8g32 hgn overlays). QWENOX_DENSE_WMMA_Q8 (default
// follows QWENOX_DENSE_WMMA): every q8g32 GEMV of P <= 48 rows -- decode gemv,
// lm_head, spec verify, batched verify / drafts, small prefill chunks -- runs
// this one kernel, so concurrent == single-run bit for bit. The wave layout
// (KW) and the K slicing (gps) are functions of the FULL tensor shape
// (w.rows, w.cols) only: row slices (gemv row_off / nrows) keep the same
// per-element arithmetic. ns > 1 partials go to g_q8w_part (v2_init).
static constexpr size_t kQ8wPartFloats = (size_t)4 << 20;  // 16 MiB
float* g_q8w_part = nullptr;
static bool dense_wmma_q8_on() {
  static const bool v = [] {
    const char* e = getenv("QWENOX_DENSE_WMMA_Q8");
    const bool on = dense_wmma_on() && !(e && e[0] == '0' && e[1] == 0);
    if (on) fprintf(stderr, "dense wmma: q8g32 GEMV P<=48 on k_q8g32_wmma (QWENOX_DENSE_WMMA_Q8)\n");
    return on;
  }();
  return v;
}
// shape-only config: KW group lanes (RW = 8 / KW row tiles) and gps groups/slice.
// Knobs (speed only; every value is row-invariant because it depends on the
// shape alone): QWENOX_Q8W_KW=<short>,<mid>,<long> for cols <= 1024 / mid /
// >= 4096 (default 2,4,8), QWENOX_Q8W_BLOCKS target blocks per launch (split-K
// fills it), QWENOX_Q8W_U loads in flight per lane (4 or 8).
static int q8w_env(const char* n, int d) {
  const char* e = getenv(n);
  return e && *e ? atoi(e) : d;
}
static void q8w_config(uint64_t rows, uint64_t cols, int& kw, int& gps, int& ns) {
  static const std::array<int, 3> kwv = [] {
    int v[3] = {2, 4, 8};
    if (const char* e = getenv("QWENOX_Q8W_KW"); e && *e) sscanf(e, "%d,%d,%d", &v[0], &v[1], &v[2]);
    for (int& x : v) x = (x == 1 || x == 2 || x == 4 || x == 8) ? x : 4;
    return std::array<int, 3>{v[0], v[1], v[2]};
  }();
  static const int nblk = std::max(1, q8w_env("QWENOX_Q8W_BLOCKS", 80));
  const int ng = (int)(cols / 32);
  kw = cols <= 1024 ? kwv[0] : cols >= 4096 ? kwv[2] : kwv[1];
  const int rw = 8 / kw;
  const int nrt = (int)((rows + 16 * rw - 1) / (16 * rw));  // row tiles of the full tensor
  ns = std::max(1, std::min((nblk + nrt - 1) / nrt, std::max(1, ng / (2 * kw))));
  gps = (ng + ns - 1) / ns;
  ns = (ng + gps - 1) / gps;
}
// QWENOX_Q8W_SKIP=RxC,RxC,... (debug): these full-tensor shapes keep the old
// q8g32 kernels (shape-only, so still consistent across P).
static bool q8w_skip(uint64_t rows, uint64_t cols) {
  static const std::vector<std::pair<uint64_t, uint64_t>> sk = [] {
    std::vector<std::pair<uint64_t, uint64_t>> v;
    if (const char* e = getenv("QWENOX_Q8W_SKIP"))
      for (const char* p = e; *p;) {
        unsigned long long r = 0, c = 0;
        if (sscanf(p, "%llux%llu", &r, &c) == 2) v.push_back({r, c});
        while (*p && *p != ',') p++;
        if (*p) p++;
      }
    return v;
  }();
  for (auto& [r, c] : sk)
    if (r == rows && c == cols) return true;
  return false;
}
static bool q8g32_wmma(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                       uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  if (P < 1 || P > 48 || w.dtype != 8 || w.cols % 32 || !nrows) return false;
  if (q8w_skip(w.rows, w.cols)) return false;
  // QWENOX_Q8W_XSTAT=N (debug, syncs): magnitude stats of x for the first N
  // calls per shape -- f16 operand range check (|x| < 2^-14 subnormal, > 65504 inf).
  static const int xstat = q8w_env("QWENOX_Q8W_XSTAT", 0);
  if (xstat > 0) {
    static std::map<std::pair<uint64_t, uint64_t>, int> seen;
    int& c = seen[{w.rows, w.cols}];
    hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
    (void)hipStreamIsCapturing(g_str, &cap);
    if (cap == hipStreamCaptureStatusNone && c++ < xstat) {
      CK(hipStreamSynchronize(g_str));
      std::vector<float> h((size_t)P * w.cols);
      for (int p = 0; p < P; p++) {
        if (xbf16) {
          std::vector<uint16_t> b(w.cols);
          CK(hipMemcpy(b.data(), (const uint16_t*)x + (size_t)p * xs, w.cols * 2, hipMemcpyDeviceToHost));
          for (uint64_t i = 0; i < w.cols; i++) {
            uint32_t u = (uint32_t)b[i] << 16;
            memcpy(&h[p * w.cols + i], &u, 4);
          }
        } else {
          CK(hipMemcpy(&h[p * w.cols], (const float*)x + (size_t)p * xs, w.cols * 4, hipMemcpyDeviceToHost));
        }
      }
      double mx = 0, ss = 0;
      size_t sub = 0, ovf = 0, nz = 0;
      for (float v : h) {
        const double a = fabs((double)v);
        mx = std::max(mx, a);
        ss += a * a;
        nz += a > 0;
        sub += a > 0 && a < 6.103515625e-05;
        ovf += a > 65504.0;
      }
      fprintf(stderr, "q8w xstat %llux%llu P=%d xbf=%d: max %.3g rms %.3g subnormal %.2f%% overflow %zu\n",
              (unsigned long long)w.rows, (unsigned long long)w.cols, P, (int)xbf16, mx,
              sqrt(ss / h.size()), 100.0 * sub / std::max<size_t>(1, nz), ovf);
    }
  }
  int kw, gps, ns;
  q8w_config(w.rows, w.cols, kw, gps, ns);
  const int R = (int)nrows, C = (int)w.cols;
  // capacity from the FULL row count (shape-only: a slice never takes a
  // different path than its full tensor)
  if (ns > 1 && (!g_q8w_part || (size_t)ns * 48 * w.rows > kQ8wPartFloats)) return false;
  const int8_t* q = (const int8_t*)w.data + row_off * w.cols;
  const __half* sc = (const __half*)(w.data + w.rows * w.cols) + row_off * (w.cols / 32);
  float* out = ns > 1 ? g_q8w_part : y;
  const int rw = 8 / kw;
  const dim3 grid((unsigned)((R + 16 * rw - 1) / (16 * rw)), (unsigned)ns);
  static const int uq = q8w_env("QWENOX_Q8W_U", 4) == 8 ? 8 : 4;
#define QWENOX_Q8W_(NPT, KW, UU)                                                               \
  do {                                                                                         \
    if (xbf16)                                                                                 \
      k_q8g32_wmma<NPT, KW, uint16_t, UU><<<grid, 256, 0, g_str>>>(                            \
          q, sc, C / 32, (const uint16_t*)x, xs, out, ys, R, C, gps, P);                       \
    else                                                                                       \
      k_q8g32_wmma<NPT, KW, float, UU><<<grid, 256, 0, g_str>>>(                               \
          q, sc, C / 32, (const float*)x, xs, out, ys, R, C, gps, P);                          \
  } while (0)
#define QWENOX_Q8W(NPT, KW)            \
  do {                                 \
    if (uq == 8) QWENOX_Q8W_(NPT, KW, 8); \
    else QWENOX_Q8W_(NPT, KW, 4);       \
  } while (0)
#define QWENOX_Q8W_KW(KW)                  \
  do {                                     \
    if (P <= 16) QWENOX_Q8W(1, KW);        \
    else if (P <= 32) QWENOX_Q8W(2, KW);   \
    else QWENOX_Q8W(3, KW);                \
  } while (0)
  if (kw == 1) QWENOX_Q8W_KW(1);
  else if (kw == 2) QWENOX_Q8W_KW(2);
  else if (kw == 4) QWENOX_Q8W_KW(4);
  else QWENOX_Q8W_KW(8);
#undef QWENOX_Q8W_KW
#undef QWENOX_Q8W
#undef QWENOX_Q8W_
  if (ns > 1)
    k_q6g64_wred<<<(unsigned)((P * R + 255) / 256), 256, 0, g_str>>>(g_q8w_part, y, ys, R, ns, P);
  return true;
}

// --- GGUF native dense (dtype 100 + GGML type, 26_kernels_dense_gguf.inc) -----
static inline bool ggd_dtype(int dt) { return dt >= 100 && ggd::type_ok(dt - ggd::kDtypeBase); }
template <class F>
static inline void ggd_with_type(int t, F&& f) {
  switch (t) {
    case ggd::kQ4K: f(std::integral_constant<int, ggd::kQ4K>{}); break;
    case ggd::kQ5K: f(std::integral_constant<int, ggd::kQ5K>{}); break;
    case ggd::kQ6K: f(std::integral_constant<int, ggd::kQ6K>{}); break;
    case ggd::kIQ4NL: f(std::integral_constant<int, ggd::kIQ4NL>{}); break;
    default: f(std::integral_constant<int, ggd::kIQ4XS>{}); break;
  }
}
// y row p at y + p*ys (row r of the slice at y[p*ys + r]); x row p at x + p*xs
// (f32 or bf16). Any P: passes of <= 8 tokens, each token bit-identical to P = 1.
static bool ggd_gemv(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                     uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  if (!ggd_dtype(w.dtype) || P < 1 || !nrows || w.cols % 32) return false;
  const int t = w.dtype - ggd::kDtypeBase;
  if (t != ggd::kIQ4NL && w.cols % 256) return false;
  if (xs % (xbf16 ? 8 : 4) || ((uintptr_t)x & 15)) return false;
  const uint64_t rb = ggd::row_bytes(t, w.cols);
  const uint8_t* wd = w.data + row_off * rb;
  if (rb % 4 || ((uintptr_t)wd & 3)) return false;  // dword-staged rows
  const unsigned blocks = (unsigned)((nrows + 3) / 4);
  for (int off = 0; off < P; off += 8) {
    const int pn = std::min(8, P - off);
    const void* xp = xbf16 ? (const void*)((const uint16_t*)x + (size_t)off * xs)
                           : (const void*)((const float*)x + (size_t)off * xs);
    float* yp = y + (size_t)off * ys;
    ggd_with_type(t, [&](auto tc_) {
      constexpr int T = decltype(tc_)::value;
#define QWENOX_GGD(NP)                                                                  \
  case NP:                                                                              \
    if (xbf16)                                                                          \
      k_ggd_gemv<T, NP, true><<<blocks, 128, 0, g_str>>>(wd, rb, xp, xs, yp, ys,      \
                                                         (int)nrows, (int)w.cols);      \
    else                                                                                \
      k_ggd_gemv<T, NP, false><<<blocks, 128, 0, g_str>>>(wd, rb, xp, xs, yp, ys,     \
                                                          (int)nrows, (int)w.cols);     \
    break;
      switch (pn) {
        QWENOX_GGD(1)
        QWENOX_GGD(2)
        QWENOX_GGD(3)
        QWENOX_GGD(4)
        QWENOX_GGD(5)
        QWENOX_GGD(6)
        QWENOX_GGD(7)
        QWENOX_GGD(8)
      }
#undef QWENOX_GGD
    });
  }
  return true;
}
// D4: two same-dtype/same-cols native-GGUF GEMVs in one launch (P = 1 decode).
// Per-row K order is k_ggd_gemv's, so this is bit-identical to two launches.
static bool ggd_gemv2(const WView& a, const WView& b, const float* x, float* ya, float* yb) {
  if (a.dtype != b.dtype || a.cols != b.cols || !ggd_dtype(a.dtype)) return false;
  if (!a.rows || !b.rows || a.cols % 32 || ((uintptr_t)x & 15)) return false;
  const int t = a.dtype - ggd::kDtypeBase;
  if (t != ggd::kIQ4NL && a.cols % 256) return false;
  const uint64_t rb = ggd::row_bytes(t, a.cols);
  if (rb % 4 || ((uintptr_t)a.data & 3) || ((uintptr_t)b.data & 3)) return false;
  const int RA = (int)a.rows, RB = (int)b.rows;
  const unsigned blocks = (unsigned)((RA + RB + 3) / 4);
  ggd_with_type(t, [&](auto tc_) {
    constexpr int T = decltype(tc_)::value;
    k_ggd_gemv2<T, false><<<blocks, 128, 0, g_str>>>(a.data, b.data, RA, RB, rb, x, ya, yb,
                                                     (int)a.cols);
  });
  return true;
}

// --- GGUF native dense WMMA (k_ggd_wmma), row-invariant for P = 1..48 -------
// One weight pass for any P <= 48 (the chunked <= 8-row GEMV re-reads the
// weight ceil(P/8) times: verify gamma=8 twice, a 48-row rendezvous 6x). Same
// contract as the hgn dense WMMA kernels: the kernel family serves decode
// (P = 1), verify and rendezvous alike, so concurrent output matches
// single-run bit for bit. QWENOX_GG_WMMA=0 keeps the chunked GEMV.
float* g_ggw_part = nullptr;  // split-K partials, ns * 48 * rows floats
static constexpr size_t kGgwPartFloats = (size_t)4 << 20;  // 16 MiB
static bool dense_wmma_gg_on() {
  static const bool v = [] {
    const char* e = getenv("QWENOX_GG_WMMA");
    const int mode = e && *e ? atoi(e) : 1;  // 1 = all P (default), 0 = off
    const bool on = mode != 0;
    if (on)
      fprintf(stderr, "dense wmma: GGUF native GEMV P<=48 on k_ggd_wmma (QWENOX_GG_WMMA)\n");
    return on;
  }();
  return v;
}
// Wave layout (RW x KW, 8 waves) and K slicing (gps quant blocks per slice)
// from the FULL tensor shape only. RW is the largest of 8/4/2 keeping
// R % (16*RW) == 0 with >= 32 row tiles (else 1); split-K then fills toward
// QWENOX_GGW_BLOCKS (default 80) blocks. The Q4_K/Q5_K x group-sum LDS budget
// (48 x 192 f32 max) caps the slice at 192 32-col groups.
static bool ggw_config(int dtype, uint64_t rows, uint64_t cols, int& rw, int& gps, int& ns) {
  const int t = dtype - ggd::kDtypeBase;
  const int bpk = t == ggd::kIQ4NL ? 1 : 8;  // 32-col groups per quant block
  if (t == ggd::kIQ4NL) {
    if (cols % 32) return false;
  } else if (cols % 256)
    return false;
  if (rows % 16) return false;
  const int nblk = (int)(cols / (32 * bpk));
  rw = 1;
  for (int cand : {8, 4, 2})
    if (rows % (16 * cand) == 0 && rows / (16 * cand) >= 32) {
      rw = cand;
      break;
    }
  const int tiles = (int)(rows / (16 * rw));
  static const int target = [] {
    const char* e = getenv("QWENOX_GGW_BLOCKS");
    return e && *e ? std::max(1, atoi(e)) : 80;
  }();
  ns = 1;
  if (tiles < target) ns = std::min((target + tiles - 1) / tiles, nblk);
  gps = (nblk + ns - 1) / ns;
  const int gcap = 192 / bpk;
  if (gps > gcap) {
    gps = gcap;
    ns = (nblk + gps - 1) / gps;
  }
  if (ns > 1 && (size_t)ns * 48 * rows > kGgwPartFloats) return false;
  return true;
}
static bool ggd_wmma(const WView& w, const void* x, bool xbf16, uint64_t xs, float* y,
                     uint64_t ys, int P, uint64_t row_off, uint64_t nrows) {
  static const bool dbg = getenv("QWENOX_GGW_DEBUG") != nullptr;
  auto rej = [&](const char* why) {
    if (dbg)
      fprintf(stderr, "ggw refuse: %s dtype=%d R=%llu C=%llu P=%d row_off=%llu nrows=%llu\n",
              why, w.dtype, (unsigned long long)w.rows, (unsigned long long)w.cols, P,
              (unsigned long long)row_off, (unsigned long long)nrows);
    return false;
  };
  if (P < 1 || P > 48 || !nrows || !ggd_dtype(w.dtype)) return rej("P/dtype");
  int rw = 1, gps = 1, ns = 1;
  if (!ggw_config(w.dtype, w.rows, w.cols, rw, gps, ns)) return rej("config");
  // row slices keep the full-tensor per-element arithmetic only when the
  // slice boundary falls on a wave-tile boundary
  if (row_off % (uint64_t)(16 * rw) || nrows % (uint64_t)(16 * rw)) return rej("rowslice");
  const int t = w.dtype - ggd::kDtypeBase;
  const uint64_t rb = ggd::row_bytes(t, w.cols);
  if (rb % 4 || !w.data) return rej("rb/data");
  const uint8_t* wd = w.data + row_off * rb;
  if (xs % (xbf16 ? 8 : 4) || (((uintptr_t)x | (uintptr_t)y) & 15)) return rej("align");
  const int bpk = t == ggd::kIQ4NL ? 1 : 8;
  const bool bias = t == ggd::kQ4K || t == ggd::kQ5K;
  const size_t lds = bias ? (size_t)P * (gps * bpk) * 4 : 0;
  if (lds > 48 * 192 * 4) return rej("lds");
  if (ns > 1 && !g_ggw_part) CK(hipMalloc(&g_ggw_part, kGgwPartFloats * 4));
  if (dbg) {
    static int nlog = 0;
    if (nlog++ < 12)
      fprintf(stderr, "ggw accept: dtype=%d R=%llu C=%llu P=%d rw=%d gps=%d ns=%d\n",
              w.dtype, (unsigned long long)w.rows, (unsigned long long)w.cols, P, rw, gps,
              ns);
  }
  float* out = ns > 1 ? g_ggw_part : y;
  const dim3 grid((unsigned)(nrows / (16 * rw)), (unsigned)ns);
  const int R = (int)nrows, C = (int)w.cols;
  ggd_with_type(t, [&](auto tc_) {
    constexpr int T = decltype(tc_)::value;
#define QWENOX_GGW(NPT, RW, KW)                                                             \
  do {                                                                                      \
    if (xbf16)                                                                              \
      k_ggd_wmma<T, NPT, RW, KW, uint16_t><<<grid, 256, lds, g_str>>>(                      \
          wd, rb, (const uint16_t*)x, xs, out, ys, R, C, gps, P);                           \
    else                                                                                    \
      k_ggd_wmma<T, NPT, RW, KW, float><<<grid, 256, lds, g_str>>>(                         \
          wd, rb, (const float*)x, xs, out, ys, R, C, gps, P);                              \
  } while (0)
#define QWENOX_GGW_P(RW, KW)                                                                \
  do {                                                                                      \
    if (P <= 16)                                                                            \
      QWENOX_GGW(1, RW, KW);                                                                \
    else if (P <= 32)                                                                       \
      QWENOX_GGW(2, RW, KW);                                                                \
    else                                                                                    \
      QWENOX_GGW(3, RW, KW);                                                                \
  } while (0)
    switch (rw) {
      case 8: QWENOX_GGW_P(8, 1); break;
      case 4: QWENOX_GGW_P(4, 2); break;
      case 2: QWENOX_GGW_P(2, 4); break;
      default: QWENOX_GGW_P(1, 8); break;
    }
#undef QWENOX_GGW
#undef QWENOX_GGW_P
  });
  if (ns > 1)
    k_ggd_wred<<<(unsigned)(((size_t)P * R + 255) / 256), 256, 0, g_str>>>(out, y, ys, R, ns,
                                                                          P);
  return true;
}
// D4: merge two independent GEMVs that share x and dtype/cols. Only the
// native-GGUF path is fused today; everything else falls back (caller runs
// the old two launches). QWENOX_GEMV_MERGE=0 disables.
static bool gemv2(const WView& a, const WView& b, const float* x, float* ya, float* yb) {
  static const bool on = [] {
    const char* s = getenv("QWENOX_GEMV_MERGE");
    return !(s && s[0] == '0' && s[1] == 0);
  }();
  return on && ggd_gemv2(a, b, x, ya, yb);
}
static void ggd_deq_bf16(const WView& w, uint16_t* out, hipStream_t st) {
  const int t = w.dtype - ggd::kDtypeBase;
  const uint64_t rb = ggd::row_bytes(t, w.cols), n = w.rows * (w.cols / 32);
  ggd_with_type(t, [&](auto tc_) {
    constexpr int T = decltype(tc_)::value;
    k_ggd_deq_bf16<T><<<(unsigned)((n + 255) / 256), 256, 0, st>>>(w.data, rb, out, w.rows,
                                                                   (int)w.cols);
  });
}

// ht / q6g64 -> bf16 W[rows][cols] (prefill GEMM operand).
// fast (bit-identical variants): ht -> k_ht_deq_bf16_h (half LDS, 2x resident
// blocks), q6g64 -> k_q6g64_deq_bf16_c8 (coalesced 8-column threads).
// fast < 0 = default from QWENOX_DEQ_FAST (unset = on); =0 keeps the originals.
static void v2_deq_bf16(const WView& w, uint16_t* out, hipStream_t st, int fast = -1) {
  static const bool fast_env = [] {
    const char* e = getenv("QWENOX_DEQ_FAST");
    return !(e && e[0] == '0' && e[1] == 0);
  }();
  const bool f = fast < 0 ? fast_env : fast != 0;
  if (w.dtype == 16) {
    const unsigned tiles = (unsigned)((w.rows / 128) * (w.cols / 128));
    if (f)
      k_ht_deq_bf16_h<<<tiles, 256, 0, st>>>(w.data, w.suh, w.svh, out, (int)w.cols);
    else
      k_ht_deq_bf16<<<tiles, 256, 0, st>>>(w.data, w.suh, w.svh, out, (int)w.cols);
  } else if (f) {
    const uint64_t n8 = w.rows * (w.cols / 8);
    k_q6g64_deq_bf16_c8<<<(unsigned)((n8 + 255) / 256), 256, 0, st>>>(
        w.data, out, (int)w.rows, (int)w.cols, (int)w.scale_stride);
  } else {
    const uint64_t ng = w.rows * (w.cols / 64);
    k_q6g64_deq_bf16<<<(unsigned)((ng + 255) / 256), 256, 0, st>>>(
        w.data, out, (int)w.rows, (int)w.cols, (int)w.scale_stride);
  }
}

static void gemv(const WView& w, const float* x, float* y, uint64_t row_off = 0,
                 uint64_t nrows = 0) {
  uint64_t rows = nrows ? nrows : w.rows;
  int warps_per_block = 8;
  int threads = warps_per_block * 64;
  uint64_t blocks = (rows + warps_per_block - 1) / warps_per_block;
  if (w.dtype == 5) {
    const uint8_t* codes = w.data + 64 + row_off * (w.cols / 2);
    const uint8_t* scales =
        w.data + 64 + w.rows * w.cols / 2 + row_off * w.scale_stride;
    // v3 kernel: 2 rows per warp, blockDim/warpSize warps per block
    uint64_t wpb = threads / 32;
    uint64_t blocks_v3 = (rows + wpb * 2 - 1) / (wpb * 2);
    k_q4cp_gemv<<<(unsigned)blocks_v3, threads, 0, g_str>>>(codes, scales, (const float*)w.data, x,
                                               y, rows, w.cols, w.scale_stride);
  } else if (w.dtype == 7) {
    uint64_t stride = w.cols + w.cols / 64 * 4;
    k_q8g64_gemv<<<(unsigned)blocks, threads, 0, g_str>>>(w.data + row_off * stride, x, y, rows,
                                                w.cols);
  } else if (g_dense_rowfast && w.dtype == 0 && rows < 64 && w.cols % 8 == 0 && w.cols >= 2048) {
    // 8-bit dense only: keeps the 4-bit hgn path bit-exact (g3_verify 3)
    k_bf16_gemv_row<<<(unsigned)rows, 256, 0, g_str>>>(
        (const uint16_t*)w.data + row_off * w.cols, x, y, w.cols);
  } else if (w.dtype == 0) {
    k_bf16_gemv<<<(unsigned)blocks, threads, 0, g_str>>>(
        (const uint16_t*)w.data + row_off * w.cols, x, y, rows, w.cols);
  } else if (w.dtype == 8) {
    // P = 1 decode forward / MTP draft step keeps the bandwidth-tuned GEMV
    // (k_q8g32_wmma at P = 1 is 15-20% slower). Nothing compares this path
    // bit for bit with a batched one: verify rows (any P, incl. γ = 0) go
    // through gemm -> gemv_multi_q4cp -> k_q8g32_wmma, and the AR decode
    // forward differs from verify rows anyway (decode-graph GDN / attention).
    const int8_t* q = (const int8_t*)w.data + row_off * w.cols;
    const __half* s = (const __half*)(w.data + w.rows * w.cols) + row_off * (w.cols / 32);
    // tools/q8g32_gemv_bench.cu: full-wave rows in 256-thread blocks beat the
    // 2-rows-per-wave kernel on every decode shape (2560x6144 141 -> 210 GB/s,
    // big ones +4-7%); 8 lanes per row for the 320/640-col matrices.
    if (w.cols < 1024)
      q8g32_gemv_small<8, 256>(q, s, x, y, rows, w.cols, g_str);
    else
      q8g32_gemv_small<32, 256>(q, s, x, y, rows, w.cols, g_str);
  } else if (w.dtype == 16) {
    if (dense_wmma_on())
      ht_wmma(w, x, false, w.cols, y, 0, 1, row_off, rows);
    else
      ht_gemv(w, x, false, 0, y, 0, 1, row_off, rows);
  } else if (w.dtype == 24) {
    if (!(dense_wmma_hc_on() && q6g64_wmma(w, x, false, w.cols, y, 0, 1, row_off, rows)))
      q6g64_gemv(w, x, false, 0, y, 0, 1, row_off, rows);
  } else if (ggd_dtype(w.dtype)) {
    // P = 1 decode/draft stays on the bandwidth-tuned GEMV (k_ggd_wmma at
    // P = 1 is ~30% slower on the Q6_K lm_head; same split as q8g32). Verify
    // rows (P >= 2) go through gemm -> gemv_multi_q4cp -> k_ggd_wmma.
    if (!ggd_gemv(w, x, false, w.cols, y, 0, 1, row_off, rows))
      throw std::runtime_error("gemv: GGUF native dense shape/alignment");
  } else {
    throw std::runtime_error("gemv: unsupported dtype");
  }
}

// D3: GEMV with k_silu_scale's (v*0.25 -> silu) epilogue folded in. The GGUF
// q8g32 path fuses it into the row store; every other dtype keeps the old two
// launches (same result). QWENOX_GEMV_SILU=0 restores the old path.
static void gemv_silu(const WView& w, const float* x, float* y) {
  static const bool on = [] {
    const char* s = getenv("QWENOX_GEMV_SILU");
    return !(s && s[0] == '0' && s[1] == 0);
  }();
  if (on && w.dtype == 8) {
    const int8_t* q = (const int8_t*)w.data;
    const __half* s = (const __half*)(w.data + w.rows * w.cols);
    if (w.cols < 1024)
      q8g32_gemv_small<8, 256, true>(q, s, x, y, w.rows, w.cols, g_str);
    else
      q8g32_gemv_small<32, 256, true>(q, s, x, y, w.rows, w.cols, g_str);
    return;
  }
  gemv(w, x, y);
  k_silu_scale<<<(unsigned)((w.rows + 255) / 256), 256, 0, g_str>>>(y, 0.25f, (int)w.rows);
}

// Embedding gather for P tokens (d_tok[0..P)): fp32 rows at out + p*outstride,
// or (hc4) four bf16 branch copies per token at out + p*4*cols. dtype 5 q4cp
// (hgn) or 8 q8g32 (GGUF Q8_0).
static void embed_rows(const WView& et, const int* d_tok, int P, void* out,
                       uint64_t outstride, bool hc4 = false) {
  if (et.dtype == 8) {
    const int8_t* q = (const int8_t*)et.data;
    const __half* s = (const __half*)(et.data + et.rows * et.cols);
    if (hc4)
      k_q8g32_row_b<true><<<P, 128, 0, g_str>>>(q, s, d_tok, et.cols, out, 0);
    else
      k_q8g32_row_b<false><<<P, 128, 0, g_str>>>(q, s, d_tok, et.cols, out, outstride);
    return;
  }
  if (et.dtype != 5) throw std::runtime_error("embed: unsupported dtype");
  const uint8_t* codes = et.data + 64;
  const uint8_t* scales = et.data + 64 + et.rows * et.cols / 2;
  if (hc4)
    k_q4cp_row_b_hc_bf16<<<P, 128, 0, g_str>>>(codes, scales, (const float*)et.data, d_tok,
                                               et.cols, et.scale_stride, (uint16_t*)out);
  else
    k_q4cp_row_b<<<P, 128, 0, g_str>>>(codes, scales, (const float*)et.data, d_tok, et.cols,
                                       et.scale_stride, (float*)out, outstride);
}

// Multi-row GEMV dispatch: P ≤ 8 streams the weight once for all P rows;
// 8 < P ≤ QWENOX_GEMVMR_MAXP runs chunked ≤8-row passes (weights re-read
// per pass). Covers q4cp (dtype 5, fp32/bf16 x), bf16 (dtype 0), q8g64
// (dtype 7), q8g32 (dtype 8) and v2 ht/q6g64 (dtype 16/24). Returns false
// when the shape/layout is unsupported (caller falls back to the
// dequant->bf16->hipBLASLt path).
static int gemvmr_maxp() {
  // 48: kVSnapRows（40_model.inc，S3a）。会合合批 R≤48 必须与串行单段
  // （P≤9）走同一 gemvmr 门，否则 P 门分叉破坏逐位一致。chunked 按行分块
  // （每块独立 ≤8 行 pass），行结果与分块无关 → 任意 P 逐位 ≡ 串行。
  // gfx1151 A/B crossover（logs/gemvmr_ps*.log, exact-P psweep vs MAXP=8）:
  // P=16 -21~-24% wall, P=24 -6.4%, P=32 +8% (chunked re-reads scale P/8).
  // P>24 的 +8% 是相对 deq+WMMA 单遍；对会合场景真正照是串行 N 段
  // （同样 N 遍权重读取），chunked 无掉崖。S3a VBENCH 复测 P=32/48 见
  // logs/s3a_vbench_*.log。
  // Measured on v2 ht/q6g64; dtype 0/5/7/8 share the default pending their
  // own sweep (per-pass weight cost is the same 0.5-0.75 B/elem order).
  // Single sequence (no rendezvous): 24, the measured single-pass optimum
  // (P=32 chunked is +8% vs deq+WMMA); 48 only where batched verify needs
  // the same gemv family as its per-segment calls.
  static const int v = [] {
    const char* e = getenv("QWENOX_GEMVMR_MAXP");
    return e && *e ? atoi(e) : (vseg_multi() ? 48 : 24);
  }();
  return v;
}
// P in 1..8 core: one pass over the weights for all P rows. xstride must be
// non-zero and x/y alignment already checked by the caller.
static bool gemv_multi_q4cp_le8(const WView& w, const void* x, float* y, int P,
                                uint64_t xstride, bool xbf16) {
  if (P < 1 || P > 8) return false;
  if (w.dtype == 16) return ht_gemv(w, x, xbf16, xstride, y, w.rows, P, 0, w.rows);
  if (w.dtype == 24) return q6g64_gemv(w, x, xbf16, xstride, y, w.rows, P, 0, w.rows);
  int threads = 512;
  uint64_t wpb = threads / 32;
  uint64_t blocks = (w.rows + wpb * 2 - 1) / (wpb * 2);
  if (g_dense_rowfast && w.dtype == 0 && w.rows < 64 && w.cols % 8 == 0 && w.cols >= 2048 &&
      xstride % 8 == 0) {
    // few-row bf16 (HC inject 4x10240 with bf16 x in spec verify, in_proj_a/b
    // 48x2560): one block per row. 8-bit dense only (4-bit hgn unchanged).
#define QWENOX_BROW(PP)                                                          \
  case PP:                                                                     \
    if (xbf16)                                                                 \
      k_bf16_gemv_row_mp<PP, true><<<(unsigned)w.rows, 256, 0, g_str>>>(      \
          (const uint16_t*)w.data, x, y, w.rows, w.cols, xstride);             \
    else                                                                       \
      k_bf16_gemv_row_mp<PP, false><<<(unsigned)w.rows, 256, 0, g_str>>>(     \
          (const uint16_t*)w.data, x, y, w.rows, w.cols, xstride);             \
    return true;
    switch (P) {
      QWENOX_BROW(1)
      QWENOX_BROW(2)
      QWENOX_BROW(3)
      QWENOX_BROW(4)
      QWENOX_BROW(5)
      QWENOX_BROW(6)
      QWENOX_BROW(7)
      QWENOX_BROW(8)
      default:
        break;
    }
#undef QWENOX_BROW
  }
  if (w.dtype == 0 && !xbf16 && w.cols % 8 == 0) {
    switch (P) {
#define QWENOX_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_bf16_gemv_mr<PP><<<(unsigned)blocks, threads, 0, g_str>>>(           \
        (const uint16_t*)w.data, (const float*)x, y, w.rows, w.cols,       \
        xstride);                                                          \
    break;
      QWENOX_GEMV_MR_CASE(1)
      QWENOX_GEMV_MR_CASE(2)
      QWENOX_GEMV_MR_CASE(3)
      QWENOX_GEMV_MR_CASE(4)
      QWENOX_GEMV_MR_CASE(5)
      QWENOX_GEMV_MR_CASE(6)
      QWENOX_GEMV_MR_CASE(7)
      QWENOX_GEMV_MR_CASE(8)
#undef QWENOX_GEMV_MR_CASE
      default:
        return false;
    }
    return true;
  }
  if (w.dtype == 7 && w.cols % 64 == 0) {
    if (xbf16) switch (P) {
#define QWENOX_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_q8g64_gemv_mr<PP, true><<<(unsigned)blocks, threads, 0, g_str>>>(    \
        w.data, x, y, w.rows, w.cols, xstride);                            \
    break;
      QWENOX_GEMV_MR_CASE(1)
      QWENOX_GEMV_MR_CASE(2)
      QWENOX_GEMV_MR_CASE(3)
      QWENOX_GEMV_MR_CASE(4)
      QWENOX_GEMV_MR_CASE(5)
      QWENOX_GEMV_MR_CASE(6)
      QWENOX_GEMV_MR_CASE(7)
      QWENOX_GEMV_MR_CASE(8)
#undef QWENOX_GEMV_MR_CASE
      default:
        return false;
    }
    else switch (P) {
#define QWENOX_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_q8g64_gemv_mr<PP><<<(unsigned)blocks, threads, 0, g_str>>>(          \
        w.data, (const float*)x, y, w.rows, w.cols, xstride);              \
    break;
      QWENOX_GEMV_MR_CASE(1)
      QWENOX_GEMV_MR_CASE(2)
      QWENOX_GEMV_MR_CASE(3)
      QWENOX_GEMV_MR_CASE(4)
      QWENOX_GEMV_MR_CASE(5)
      QWENOX_GEMV_MR_CASE(6)
      QWENOX_GEMV_MR_CASE(7)
      QWENOX_GEMV_MR_CASE(8)
#undef QWENOX_GEMV_MR_CASE
      default:
        return false;
    }
    return true;
  }
  if (w.dtype == 8 && w.cols % 32 == 0) {
    // LPR lanes per row (tools/q8g32_gemv_bench.cu, P=4): short rows (HC up
    // 320, shexp down 640 cols) want 8 lanes, long rows (out_proj 6144,
    // HC down 10240) a full wave, the 2560-col majority 16.
    const int8_t* q = (const int8_t*)w.data;
    const __half* s = (const __half*)(w.data + w.rows * w.cols);
    const int lpr = w.cols < 1024 ? 8 : w.cols >= 4096 ? 32 : 16;
    // bf16 x, few long rows (HC down 320x10240, out_proj 2560x6144 in spec
    // verify): one 128-thread block per row (bench P=4: 39->23 / 113->93 us).
    if (xbf16 && w.cols >= 4096 && w.rows <= 4096) {
      // Row-tiled (RPB=4): x fetched once per (group, block) instead of once
      // per row; bit-exact vs brow (ktest brow_r_bitexact). QWENOX_BROW_R=0
      // opt-out. Only for kBrowRMinP <= P <= kBrowRMaxP: it keeps x[P][32]
      // bf16 and RPB x P accumulators in registers (VGPR 221 at P=4, 256 +
      // spill from P=5; P=6/7/8 spill 98/181/240 regs), so at P >= 6 (γ >= 5
      // verify) it ran 1.5-4.8x slower than brow and the 8K γ=7 MTP round
      // went 110 -> 139 ms. At P=1/2 its lower occupancy loses to brow too.
      // tools/brow_bench.cu (cold weights, brow_r/brow): hc_down 320x10240
      // P1..8 = 1.09 1.03 .98 .87 .94 1.47 3.07 4.45, out_proj 2560x6144
      // 1.18 1.10 1.04 .90 .92 1.59 3.31 4.77. Other P fall through to brow
      // below (bit-identical results, so the range only changes speed).
      constexpr int kBrowRMinP = 3, kBrowRMaxP = 5;
      static const bool brow_r = [] {
        const char* e = getenv("QWENOX_BROW_R");
        return !e || e[0] != '0';
      }();
      if (brow_r && P >= kBrowRMinP && P <= kBrowRMaxP) {
        switch (P) {
#define QWENOX_Q8BR(PP)                                                          \
  case PP:                                                                     \
    k_q8g32_gemv_brow_r<PP, 128, 4>                                            \
        <<<(unsigned)((w.rows + 3) / 4), 128, 0, g_str>>>(                     \
            q, s, x, y, w.rows, w.cols, xstride);                              \
    return true;
          QWENOX_Q8BR(1)
          QWENOX_Q8BR(2)
          QWENOX_Q8BR(3)
          QWENOX_Q8BR(4)
          QWENOX_Q8BR(5)
          QWENOX_Q8BR(6)
          QWENOX_Q8BR(7)
          QWENOX_Q8BR(8)
          default:
            return false;
        }
#undef QWENOX_Q8BR
      }
      switch (P) {
#define QWENOX_Q8B(PP)                                                           \
  case PP:                                                                     \
    k_q8g32_gemv_brow<PP, true, 128><<<(unsigned)w.rows, 128, 0, g_str>>>(    \
        q, s, x, y, w.rows, w.cols, xstride);                                  \
    return true;
        QWENOX_Q8B(1)
        QWENOX_Q8B(2)
        QWENOX_Q8B(3)
        QWENOX_Q8B(4)
        QWENOX_Q8B(5)
        QWENOX_Q8B(6)
        QWENOX_Q8B(7)
        QWENOX_Q8B(8)
#undef QWENOX_Q8B
        default:
          return false;
      }
    }
#define QWENOX_Q8M(PP, XB, L) \
  q8g32_gemv_multi<PP, XB, L, 256>(q, s, x, y, w.rows, w.cols, xstride, g_str)
#define QWENOX_Q8M_P(PP, XB)                  \
  case PP:                                  \
    if (lpr == 8) QWENOX_Q8M(PP, XB, 8);      \
    else if (lpr == 16) QWENOX_Q8M(PP, XB, 16); \
    else QWENOX_Q8M(PP, XB, 32);              \
    break;
    if (xbf16) switch (P) {
      QWENOX_Q8M_P(1, true)
      QWENOX_Q8M_P(2, true)
      QWENOX_Q8M_P(3, true)
      QWENOX_Q8M_P(4, true)
      QWENOX_Q8M_P(5, true)
      QWENOX_Q8M_P(6, true)
      QWENOX_Q8M_P(7, true)
      QWENOX_Q8M_P(8, true)
      default:
        return false;
    }
    else switch (P) {
      QWENOX_Q8M_P(1, false)
      QWENOX_Q8M_P(2, false)
      QWENOX_Q8M_P(3, false)
      QWENOX_Q8M_P(4, false)
      QWENOX_Q8M_P(5, false)
      QWENOX_Q8M_P(6, false)
      QWENOX_Q8M_P(7, false)
      QWENOX_Q8M_P(8, false)
      default:
        return false;
    }
#undef QWENOX_Q8M_P
#undef QWENOX_Q8M
    return true;
  }
  if (w.dtype != 5) return false;
  const uint8_t* codes = w.data + 64;
  const uint8_t* scales = w.data + 64 + w.rows * w.cols / 2;
  if (xbf16) switch (P) {
#define QWENOX_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_q4cp_gemv_mr<PP, true><<<(unsigned)blocks, threads, 0, g_str>>>(     \
        codes, scales, (const float*)w.data, x, y, w.rows, w.cols,         \
        w.scale_stride, xstride);                                          \
    break;
    QWENOX_GEMV_MR_CASE(1)
    QWENOX_GEMV_MR_CASE(2)
    QWENOX_GEMV_MR_CASE(3)
    QWENOX_GEMV_MR_CASE(4)
    QWENOX_GEMV_MR_CASE(5)
    QWENOX_GEMV_MR_CASE(6)
    QWENOX_GEMV_MR_CASE(7)
    QWENOX_GEMV_MR_CASE(8)
#undef QWENOX_GEMV_MR_CASE
    default:
      return false;
  }
  else switch (P) {
#define QWENOX_GEMV_MR_CASE(PP)                                              \
  case PP:                                                                 \
    k_q4cp_gemv_mr<PP, false><<<(unsigned)blocks, threads, 0, g_str>>>(    \
        codes, scales, (const float*)w.data, x, y, w.rows, w.cols,         \
        w.scale_stride, xstride);                                          \
    break;
    QWENOX_GEMV_MR_CASE(1)
    QWENOX_GEMV_MR_CASE(2)
    QWENOX_GEMV_MR_CASE(3)
    QWENOX_GEMV_MR_CASE(4)
    QWENOX_GEMV_MR_CASE(5)
    QWENOX_GEMV_MR_CASE(6)
    QWENOX_GEMV_MR_CASE(7)
    QWENOX_GEMV_MR_CASE(8)
#undef QWENOX_GEMV_MR_CASE
    default:
      return false;
  }
  return true;
}

// QWENOX_GEMV_STAT=1 (diagnostic): histogram of multi-row gemv calls with
// P > 8 by (dtype, rows, cols, xbf16), printed at process exit.
struct GemvStat {
  std::map<std::tuple<int, uint64_t, uint64_t, int>, std::pair<long, long>> m;
  ~GemvStat() {
    if (m.empty()) return;
    fprintf(stderr, "gemv-stat (P>8 calls): dtype rows cols xbf16 -> calls, sumP\n");
    for (auto& [k, v] : m)
      fprintf(stderr, "  %d %llu %llu %d -> %ld %ld\n", std::get<0>(k),
              (unsigned long long)std::get<1>(k), (unsigned long long)std::get<2>(k),
              std::get<3>(k), v.first, v.second);
  }
};
static GemvStat g_gemv_stat;
static bool gemv_multi_q4cp(const WView& w, const void* x, float* y, int P,
                            uint64_t xstride, bool xbf16 = false) {
  if (getenv("QWENOX_NO_GEMVMR")) return false;
  if (P < 1 || P > gemvmr_maxp()) return false;
  static const bool gstat = getenv("QWENOX_GEMV_STAT") != nullptr;
  if (gstat && P > 8) {
    auto& e = g_gemv_stat.m[{w.dtype, w.rows, w.cols, xbf16 ? 1 : 0}];
    e.first++;
    e.second += P;
  }
  if (!xstride) xstride = w.cols;
  if (xbf16) {
    if (xstride % 8 || (((uintptr_t)x | (uintptr_t)y) & 15)) return false;
  } else if (xstride % 4 || (((uintptr_t)x | (uintptr_t)y) & 15))
    return false;
  if (w.dtype == 16 && dense_wmma_on())
    return ht_wmma(w, x, xbf16, xstride, y, w.rows, P, 0, w.rows);
  if (w.dtype == 24 && dense_wmma_hc_on() &&
      q6g64_wmma(w, x, xbf16, xstride, y, w.rows, P, 0, w.rows))
    return true;
  if (w.dtype == 8 && dense_wmma_q8_on() &&
      q8g32_wmma(w, x, xbf16, xstride, y, w.rows, P, 0, w.rows))
    return true;
  // GGUF native dense: P >= 2 -> k_ggd_wmma (one weight pass, row-invariant
  // across P and rendezvous widths); P = 1 keeps the P = 1 GEMV (faster there;
  // same decode-vs-verify family split as q8g32). The GEMV fallback below is
  // reachable only when the WMMA config refuses the shape outright (a
  // shape-static decision, so the per-shape family choice stays consistent).
  if (ggd_dtype(w.dtype)) {
    if (P >= 2 && dense_wmma_gg_on() && ggd_wmma(w, x, xbf16, xstride, y, w.rows, P, 0, w.rows))
      return true;
    return ggd_gemv(w, x, xbf16, xstride, y, w.rows, P, 0, w.rows);
  }
  if (P <= 8) return gemv_multi_q4cp_le8(w, x, y, P, xstride, xbf16);
  // P > 8: chunked passes of ≤ 8 rows re-read the weight per pass
  // (0.5-0.75 B/elem/pass vs ~4.5 B/elem for deq+WMMA, so the crossover is
  // empirical; v2 A/B in logs/gemvmr_sweep_*.log: P=16 −21~−24% wall,
  // P=24 −6.4%, P=32 +8%). y rows per pass stay contiguous (w.rows is a
  // multiple of 16 for every shape on the decode path), so the 16-byte
  // alignment of y carries over to each chunk.
  static bool mrlog = false;
  if (!mrlog) {
    mrlog = true;
    fprintf(stderr, "gemvmr: chunked multi-row gemv P<=%d (QWENOX_GEMVMR_MAXP)\n",
            gemvmr_maxp());
  }
  for (int off = 0; off < P; off += 8) {
    const int pn = std::min(8, P - off);
    const void* xp = xbf16 ? (const void*)((const uint16_t*)x + (size_t)off * xstride)
                           : (const void*)((const float*)x + (size_t)off * xstride);
    float* yp = y + (size_t)off * w.rows;
    if (!gemv_multi_q4cp_le8(w, xp, yp, pn, xstride, xbf16)) return false;
  }
  return true;
}

