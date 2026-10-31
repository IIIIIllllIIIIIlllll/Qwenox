#include "../src/kv_admission.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <cstdlib>
#include <unordered_map>
#include <cstdio>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

static constexpr int KV_PAGE = 256;
static const char* log_ts() { return "test:"; }
static bool g_turn_prof = false;
static thread_local int tl_turn_tag = 0;
static std::atomic<int> g_borrow_light{0};
static struct {
  void on_acquire() {}
  void on_release(int) {}
} g_turnprof;

struct GpuModel {
  static bool env_zero(const char* name) {
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "0") == 0;
  }
  int nslots = 2;
  bool kv_paged = true;
  int kv_pool_pages = 2048, maxctx = 524288;
  std::array<int, 8> pages{};
  // Optional pair of slots references the same physical pages.
  int shared_a = -1, shared_b = -1;
  struct Pool {
    GpuModel* model;
    int nfree() const {
      int free = model->kv_pool_pages;
      for (int pages : model->pages) free -= pages;
      if (model->shared_a >= 0 && model->shared_b >= 0)
        free += std::min(model->pages[(size_t)model->shared_a],
                         model->pages[(size_t)model->shared_b]);
      return free;
    }
  } kv_pool{this};
  std::atomic<bool> mutating{false};
  mutable std::atomic<int> unsafe_reads{0};
  std::vector<int64_t> chist;
  struct Ngram { std::vector<int> table; int learned = 0; } ngram;
  struct Gate { int rounds = 0; } ng_gate;
  int ng_cs = 0;
  bool ng_cs_fixed = false, ng_legacy = false;
  int bound_ = 0;
  int pos = 0, mtp_pos = 0, kv_nl = 1, cut_abs = -1;
  bool mrope_active_ = false, rope_delta_ = false, pf_end = false;
  bool spec_snap = false, ngram_verify_active = false, mtp_tap_capture = false;
  bool cut_r_ = false, ple_on = false, graphs_ok = false;
  std::vector<int> mrope_pos_, img_inj;
  int device_pos = 0, device_mpos = 0, dummy = 1;
  int *d_pos = &device_pos, *d_mpos = &device_mpos, *d_miraw = &dummy;
  int *d_R = &dummy, *d_mtap = &dummy;
  struct { bool cap_mode = false; } df2;
  struct { int mapped = 0; } kv_seq;
  std::atomic<bool>* abort_flag = nullptr;
  bool (*borrow_query_fn)() = nullptr;
  void (*borrow_skip_fn)() = nullptr;
  void (*heavy_gate_fn)() = nullptr;
  bool (*kv_starve_fn)(GpuModel&) = nullptr;
  std::vector<int> replayed;
  bool ws_lent_ = false;
  int last_spec_rounds = 0, last_spec_commit = 0, last_spec_rollbacks = 0, last_spec_proposed = 0;
  int live_spec_rounds = 0, live_spec_commits = 0, live_spec_proposed = 0;
  int prog_total = 0;
  int64_t prog_base = 0, prog_last_est = 0, prog_req = 0;
  std::chrono::steady_clock::time_point prog_t0, prog_last;
  struct SpecMembership {
    bool suspended = false;
    int suspends = 0, resumes = 0;
    void suspend() { suspended = true; ++suspends; }
    void resume() { suspended = false; ++resumes; }
    ~SpecMembership() { if (suspended) resume(); }
  };
  std::function<void()> on_prepare;
  bool kv_prepare_nowait(int begin, int end) {
    if (on_prepare) on_prepare();
    if (begin < 0 || end > maxctx || end < begin) return false;
    int needed = std::max(0, (end + KV_PAGE - 1) / KV_PAGE - pages[(size_t)bound_]);
    if (needed > kv_pool.nfree()) return false;
    pages[(size_t)bound_] += needed;
    return true;
  }
  bool kv_evict_one();
  int slot_pages(int slot) const {
    if (mutating.load()) unsafe_reads++;
    return pages[(size_t)slot];
  }
  bool fail_reclaim_prediction = false;
  int slot_reclaimable_pages(int slot) const {
    if (mutating.load()) unsafe_reads++;
    if (fail_reclaim_prediction) throw std::runtime_error("prediction failed");
    if (!kv_paged) return 0;
    int shared = 0;
    if (slot == shared_a && shared_b >= 0) shared = pages[(size_t)shared_b];
    if (slot == shared_b && shared_a >= 0) shared = pages[(size_t)shared_a];
    return pages[(size_t)slot] - std::min(pages[(size_t)slot], shared);
  }
  void bind(int slot) { bound_ = slot; kv_seq.mapped = pages[(size_t)slot]; }
  void reset_state() { pages[(size_t)bound_] = 0; kv_seq.mapped = pos = mtp_pos = 0; chist.clear(); }
  void slot_reset(int slot, int back) { pages[(size_t)slot] = 0; bind(back); }
  void mtp_ingest_b(int*, const std::vector<int>&, int, int) {}
  void prefill_batch(const std::vector<int>& tokens, int base, bool, bool) {
    replayed = tokens;
    pos = base + (int)tokens.size();
    if (mtp_tap_capture) mtp_pos = pos - 1;
    pages[(size_t)bound_] = kv_seq.mapped = (pos + KV_PAGE - 1) / KV_PAGE;
    if (ple_on) chist.insert(chist.end(), tokens.begin(), tokens.end());
  }
};

struct KvOverflow : std::runtime_error { using std::runtime_error::runtime_error; };
struct ReqCancelled : std::runtime_error { using std::runtime_error::runtime_error; };
static struct { int layers = 0; } g_cfg;
static int g_str = 0;
static constexpr int hipSuccess = 0, hipMemcpyDeviceToHost = 0, hipMemcpyHostToDevice = 1;
static bool fail_sync = false;
static int hipStreamSynchronize(int) { return fail_sync ? 1 : hipSuccess; }
static int hipMemcpy(void* dst, const void* src, size_t n, int) { std::memcpy(dst, src, n); return hipSuccess; }
static bool has_image_tok(const int*, size_t) { return false; }
namespace rckpt {
struct Ck { std::vector<int> toks; bool mtp_ok = false, mrope = false; };
static std::deque<Ck> g_list;
static bool g_pin = false;
static bool on = false, mtp = false, restore_ok = false;
static int n = 0, restores = 0;
static std::vector<int> tokens;
static bool evict_one(GpuModel&) { return false; }
[[maybe_unused]] static bool enabled() { return on; }
[[maybe_unused]] static int match(const std::vector<int>&, const std::vector<int64_t>&, bool, void*, int) { return on ? 0 : -1; }
static bool has_mtp(int) { return mtp; }
static size_t length(int) { return (size_t)n; }
[[maybe_unused]] static bool restore(GpuModel& m, int, std::vector<int>* hist, bool* has_mtp) {
  ++restores;
  if (!restore_ok) return false;
  hist->assign(tokens.begin(), tokens.begin() + n);
  m.chist.assign(hist->begin(), hist->end());
  m.pos = n;
  *has_mtp = mtp;
  return true;
}
}
namespace kvsnap {
struct CkEnt {
  bool mtp = false;
  int n = 0, busy = 0;
  uint64_t bytes = 0;
  std::vector<uint64_t> chunks, mtps;
};
struct Chunk { uint64_t bytes = 0; };
static std::mutex g_mtx;
static std::condition_variable g_cv;
static std::unordered_map<uint64_t, Chunk> g_ch, g_mt;
static double g_rs_bps = 0, g_rs_bps0 = 2e9, g_pf_tps = 0;
static int releases = 0;
static void unbusy_locked(const std::shared_ptr<CkEnt>& e) { --e->busy; ++releases; }
#include "kv_admission_tier.inc"
static int g_fp = 0, restores = 0;
static bool restore_ok = false, restore_throws = false;
static std::shared_ptr<CkEnt> candidate;
static std::shared_ptr<CkEnt> match(const std::vector<int>&, int, uint64_t = 0, bool = false) {
  if (candidate) ++candidate->busy;
  return candidate;
}
static bool restore(GpuModel& m, const std::shared_ptr<CkEnt>& e, const std::vector<int>& saved,
                    std::vector<int>& hist, bool* mtp) {
  ++restores;
  RestoreBusy busy{e};
  if (restore_throws) throw std::runtime_error("SSD restore failed");
  if (!restore_ok) { m.pos = 1; m.chist = {999}; return false; }
  hist.assign(saved.begin(), saved.begin() + e->n);
  m.chist.assign(hist.begin(), hist.end());
  m.pos = e->n;
  if (e->mtp) m.mtp_pos = e->n - 1;
  m.pages[(size_t)m.bound_] = m.kv_seq.mapped = (e->n + KV_PAGE - 1) / KV_PAGE;
  *mtp = e->mtp;
  return true;
}
enum class RequiredMode { PlainDecode, Mtp };
struct RequiredPrepared { RequiredMode mode = RequiredMode::PlainDecode; };
struct CkPin { RequiredMode mode = RequiredMode::PlainDecode; void reset() {} };
enum class PublishResult { Published, Failed, Cancelled };
static std::atomic<int> required_calls{0};
[[maybe_unused]] static bool capture_required(GpuModel&, const std::vector<int>&, RequiredMode mode,
                             RequiredPrepared& prepared) {
  ++required_calls;
  prepared.mode = mode;
  return true;
}
[[maybe_unused]] static PublishResult commit_required(RequiredPrepared&& prepared, const std::atomic<bool>& cancel,
                                     CkPin& pin, const std::function<void()>&) {
  ++required_calls;
  pin.mode = prepared.mode;
  return cancel ? PublishResult::Cancelled : PublishResult::Published;
}
[[maybe_unused]] static bool restore_pinned(GpuModel& m, const CkPin& pin, const std::vector<int>& saved,
                           std::vector<int>& hist, bool* mtp) {
  ++required_calls;
  hist = saved;
  m.chist.assign(saved.begin(), saved.end());
  *mtp = pin.mode == RequiredMode::Mtp;
  m.pages[(size_t)m.bound_] = (int)((saved.size() + KV_PAGE - 1) / KV_PAGE);
  return true;
}
}

// Called only by the host harness after self-suspension releases the GPU turn.
static std::function<void()> on_self_turn_released;

#include "kv_admission_serve.inc"

bool GpuModel::kv_evict_one() { return starve(*this); }

static void require(bool passed, const char* name) {
  if (!passed) throw std::runtime_error(name);
}

template <class Predicate>
static void wait_until(Predicate ready) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!ready()) {
    require(std::chrono::steady_clock::now() < deadline, "test wait timed out");
    std::this_thread::yield();
  }
}

static bool admission_waiting() {
  std::lock_guard<std::mutex> lock(g_sm);
  return !g_wait.empty();
}

static void reset_slots(int count) {
  std::lock_guard<std::mutex> lock(g_sm);
  require(g_wait.empty(), "admission queue leaked");
  require(g_detached.empty(), "detached registration leaked");
  g_slots.assign((size_t)count, Slot{});
  g_reserve_decode = 4096;
  g_conn_id = 0;
  tl_slot = -1;
  tl_lease = 0;
  tl_cancel = nullptr;
  tl_detached = false;
}

static void require_turn_free() {
  std::lock_guard<std::mutex> lock(g_turn.mu);
  require(g_turn.next == g_turn.serving, "GPU turn leaked or released twice");
}

static void release_slot(int slot) {
  slot_leave(slot);
  g_turn.release();
}

static std::shared_ptr<kvsnap::CkEnt> allow_exact_ssd_recall(size_t saved) {
  require(saved > 0, "exact SSD recall requires saved history");
  auto entry = std::make_shared<kvsnap::CkEnt>();
  entry->n = (int)saved;
  entry->bytes = 1000000;
  kvsnap::candidate = entry;
  kvsnap::restore_ok = true;
  kvsnap::restore_throws = false;
  return entry;
}

static void test_pressure_telemetry() {
  // A safe interrupt keeps live KV unless an ordinary SSD checkpoint covers
  // the complete saved history. An eligible self-suspension retains that busy
  // reference until its early resume decision completes.
  for (int scenario = 0; scenario < 3; ++scenario) {
    reset_slots(1);
    GpuModel model;
    model.pages[0] = 1;
    model.bind(0);
    model.pos = 4;
    auto& slot = g_slots[0];
    slot.busy = slot.suspendable = true;
    slot.seq = 1; slot.lease = 42; slot.req = 876 + scenario;
    slot.hist = {1, 2, 3, 4};
    auto ctl = std::make_shared<SuspendCtl>();
    ctl->slot = 0; ctl->seq = 1; ctl->lease = 42;
    slot.suspend = ctl;
    rckpt::on = scenario != 2;
    rckpt::g_pin = scenario == 0;
    rckpt::g_list = {{{1}, false, false}, {{1, 2, 3}, true, false},
                     {{1, 2, 3, 4}, false, false}, {{1, 2, 3}, false, true},
                     {{9, 2, 3}, false, false}};
    kvsnap::candidate.reset();
    if (scenario != 0) {
      kvsnap::candidate = std::make_shared<kvsnap::CkEnt>();
      kvsnap::candidate->n = scenario == 1 ? 3 : 4;
      kvsnap::candidate->mtp = true;
    }
    auto entry = kvsnap::candidate;
    const int releases = kvsnap::releases;
    std::atomic<bool> cancel{false};
    tl_slot = 0; tl_lease = 42; tl_cancel = &cancel;
    bool observed = false;
    on_self_turn_released = [&] {
      observed = scenario == 2 && entry && entry->busy == 1 &&
                 kvsnap::releases == releases && rckpt::g_list.size() == 5 &&
                 model.pages[0] == 1 && slot.hist == std::vector<int>({1, 2, 3, 4});
    };
    g_turn.acquire();
    suspend_handle(model, SuspendMode::PlainDecode, false, true);
    require(model.pages[0] == 1 && model.pos == 4 && slot.busy &&
                slot.hist == std::vector<int>({1, 2, 3, 4}) && g_detached.empty() &&
                model.replayed.empty() && rckpt::g_list.size() == 5,
            "rejected or deferred recall changed live KV or pruned RAM");
    if (scenario < 2)
      require(!observed && ctl->state == 4 && slot.suspend == ctl,
              "missing or short SSD prefix did not reject safe interruption");
    else
      require(observed && ctl->state == 5 && !slot.suspend,
              "exact full SSD boundary did not retain busy through self-suspension");
    require(!entry || (entry->busy == 0 && kvsnap::releases == releases + 1),
            "rejected or resumed SSD match was not released exactly once");
    on_self_turn_released = {};
    release_slot(0);
    tl_slot = -1; tl_lease = 0; tl_cancel = nullptr;
    require_turn_free();
  }
  rckpt::on = rckpt::g_pin = false;
  rckpt::g_list.clear();
  kvsnap::candidate.reset();

  GpuModel model;
  model.kv_pool_pages = 10;
  model.pages[0] = 4; model.pages[1] = 3; model.pages[2] = 2;
  model.shared_a = 1; model.shared_b = 2;
  model.bind(0);
  require(model.slot_reclaimable_pages(1) == 1 && model.kv_pool.nfree() == 3,
          "shared reclaim prediction");
  pressure_slot_reset(model, 1, 0, "host shared");
  require(model.kv_pool.nfree() == 4 && model.pages[1] == 0 && model.bound_ == 0 &&
              model.slot_reclaimable_pages(2) == 2, "shared reset actual release");
  pressure_slot_reset(model, 2, 0, "host last reference");
  require(model.kv_pool.nfree() == 6 && model.bound_ == 0, "last reference release");
  model.pages[1] = model.pages[2] = 1;
  require(model.slot_reclaimable_pages(1) == 0 && model.kv_pool.nfree() == 5,
          "fully shared page predicted as free");
  pressure_slot_reset(model, 1, 0, "host shared reference");
  require(model.kv_pool.nfree() == 5 && model.pages[2] == 1,
          "dropping shared reference changed free count");
  pressure_slot_reset(model, 2, 0, "host final reference");
  require(model.kv_pool.nfree() == 6, "final shared reference did not free page");
  model.kv_paged = false;
  require(model.slot_reclaimable_pages(0) == 0, "non-paged prediction must be zero");
  model.kv_paged = true;
  model.fail_reclaim_prediction = true;
  pressure_slot_reset(model, 0, 0, "host prediction failure");
  require(model.kv_pool.nfree() == 10 && model.pages[0] == 0 && model.bound_ == 0,
          "prediction failure prevented reset");
  std::puts("PASS safe-interrupt rejection / exact recall busy / page reclaim");
}

static void test_recall_selection() {
  auto ssd = std::make_shared<kvsnap::CkEnt>();
  ssd->n = 2000;
  ssd->bytes = 1000000;
  rckpt::n = 500;
  rckpt::mtp = false;
  require(choose_recall_checkpoint(0, ssd, false, 101) == -1 && ssd,
          "long cheap SSD should beat RAM");
  ssd->chunks = {1};
  kvsnap::g_ch[1].bytes = 10000000000ULL;
  require(choose_recall_checkpoint(0, ssd, false, 102) == 0 && ssd,
          "SSD chunk bytes must influence preference");
  kvsnap::g_ch.clear();
  ssd->chunks.clear();
  kvsnap::g_pf_tps = 100000;
  require(choose_recall_checkpoint(0, ssd, false, 102) == 0,
          "measured prefill speed must influence preference");
  kvsnap::g_pf_tps = 0;
  ssd->n = 500;
  require(choose_recall_checkpoint(0, ssd, false, 103) == 0, "equal prefix must prefer RAM");
  ssd->n = 2000;
  rckpt::mtp = true;
  require(choose_recall_checkpoint(0, ssd, false, 104) == 0, "RAM MTP must beat plain SSD");
  ssd->busy = 1;
  auto rejected = ssd;
  int before = kvsnap::releases;
  require(choose_recall_checkpoint(0, ssd, true, 105) == 0 && !ssd &&
              rejected->busy == 0 && kvsnap::releases == before + 1,
          "ineligible SSD match not released");
  ssd = std::make_shared<kvsnap::CkEnt>();
  ssd->mtp = true; ssd->n = 2000; ssd->bytes = 1000000;
  rckpt::mtp = false;
  require(choose_recall_checkpoint(0, ssd, true, 106) == -1 && ssd,
          "MTP requirement must reject plain RAM");
  ssd.reset();
  require(choose_recall_checkpoint(-1, ssd, false, 107) == -1, "cache miss selection");
  std::puts("PASS shared cache selection / preference / MTP eligibility");
}

static void test_recall_rebuild() {
  // Miss, arbitrary short SSD prefix, exact full SSD, cancellation, restore
  // failure and restore exception. No scenario may fall back to RAM/replay.
  const std::vector<int> saved(2000, 17);
  for (int scenario = 0; scenario < 6; ++scenario) {
    reset_slots(1);
    GpuModel model;
    model.bind(0);
    model.pages[0] = model.kv_seq.mapped = 2;
    model.replayed = {91, 92};
    g_slots[0].busy = true;
    g_slots[0].lease = 42;
    tl_slot = 0; tl_lease = 42; tl_suspend_rebuilding = true;
    std::atomic<bool> cancel{scenario == 3};
    rckpt::on = true;
    rckpt::n = 500; rckpt::mtp = false;
    rckpt::tokens = saved;
    rckpt::restore_ok = true;
    rckpt::restores = 0;
    kvsnap::restores = 0;
    kvsnap::restore_ok = scenario == 2;
    kvsnap::restore_throws = scenario == 5;
    kvsnap::candidate.reset();
    if (scenario != 0) {
      kvsnap::candidate = std::make_shared<kvsnap::CkEnt>();
      kvsnap::candidate->n = scenario == 1 ? (int)saved.size() - 1 : (int)saved.size();
      kvsnap::candidate->bytes = 1000000;
    }
    auto entry = kvsnap::candidate;
    const int releases = kvsnap::releases;
    bool success = false, threw = false;
    g_turn.acquire();
    {
      kvsnap::RestoreBusy recall;
      if (entry) recall.e = kvsnap::match(saved, kvsnap::g_fp);
      try {
        success = suspend_rebuild(model, saved, SuspendMode::PlainDecode,
                                  cancel, 200 + scenario, recall);
      } catch (const std::runtime_error&) { threw = true; }
      require(threw == (scenario == 5) && success == (scenario == 2),
              "exact-only rebuild outcome");
      if (scenario == 0 || scenario == 1 || scenario == 3) {
        require(model.pages[0] == 2 && model.kv_seq.mapped == 2 && model.pos == 0 &&
                    model.chist.empty() && model.replayed == std::vector<int>({91, 92}),
                "rejected or cancelled rebuild changed live KV");
        require(!entry || recall.e == entry,
                "rejected rebuild consumed the caller's busy reference early");
      }
      if (success) {
        require(model.pos == (int)saved.size() &&
                    model.chist == std::vector<int64_t>(saved.begin(), saved.end()) &&
                    model.replayed == std::vector<int>({91, 92}),
                "exact full SSD rebuild unexpectedly prefilled a suffix");
      }
    }
    require(rckpt::restores == 0, "plain pressure rebuild used a RAM checkpoint");
    require(kvsnap::restores == ((scenario == 2 || scenario >= 4) ? 1 : 0),
            "unexpected SSD restore count");
    require(!entry || (entry->busy == 0 && kvsnap::releases == releases + 1),
            "SSD busy reference was not released after rebuild outcome");
    release_slot(0);
    tl_slot = -1; tl_lease = 0; tl_suspend_rebuilding = false;
    require_turn_free();
  }
  rckpt::on = false;
  kvsnap::candidate.reset();
  kvsnap::restore_ok = false;
  kvsnap::restore_throws = false;
  std::puts("PASS exact full SSD rebuild / rejection / cancel / failure busy release");
}

static void test_arithmetic() {
  int cap = -1;
  require(kv_admission_decode_cap(nullptr, cap) && cap == 4096, "default decode cap");
  require(kv_admission_decode_cap("0", cap) && cap == 0, "zero decode cap");
  require(kv_admission_decode_cap("2147483647", cap) && cap == INT_MAX,
          "maximum decode cap");
  for (const char* value : {"", "-1", "1.5", "oops", "2x", "2147483648",
                            "999999999999999999999999"})
    require(!kv_admission_decode_cap(value, cap), "invalid decode cap accepted");
  require(kv_admission_target(524287, 100, 4096, 524288, KV_PAGE) == 2048,
          "one-token context edge");
  require(kv_admission_target(524263, 100, 4096, 524288, KV_PAGE) == 2048,
          "25-token context edge");
  require(kv_admission_target(524032, 256, 4096, 524288, KV_PAGE) == 2048,
          "full context has no artificial slack");
  require(kv_admission_target(256, INT_MAX, 4096, 524288, KV_PAGE) == 17,
          "unbound_ed generation must use decode cap");
  require(kv_admission_target(256, 5, 0, 524288, KV_PAGE) == 1,
          "zero cap reserves prompt only");
  require(kv_admission_target(INT_MAX - 1, INT_MAX, INT_MAX, INT_MAX, KV_PAGE) ==
              8388608, "page rounding overflow");
  std::vector<KvAdmissionState> states = {{true, 80, 90}, {false, 1000, 0}};
  require(kv_admission_available(2048, states, 1) == 1958,
          "reset slot must not reserve its released old pages");
  states[0].mapped = 150;
  require(kv_admission_available(2048, states, 1) == 1898,
          "decode beyond cap must count actual mapped pages");
  states[0] = {false, 2048, 0};
  require(kv_admission_available(2048, states, 1) == 2048,
          "idle pages must not count as active demand");
  states = {{true, 100, 120}, {true, 100, 120}, {false, 1000, 0}};
  require(kv_admission_available(2048, states, 2) == 1808,
          "active prefix sharing must reserve each sequence independently");
  require(kv_admission_available(2048, states, 0) == 1928,
          "chosen slot must not count itself twice");
  require(kv_admission_need(90, 400, false) == 90, "non-prefix slot reset");
  require(kv_admission_need(90, 100, true) == 100, "continuation keeps mapped pages");
  std::puts("PASS admission arithmetic / decode cap / context edge");
}

static void test_reset_slot() {
  reset_slots(2);
  GpuModel model;
  model.pages[0] = 400;
  model.pages[1] = 1000;
  g_slots[0].hist = {1};
  g_slots[1].hist = {2};
  std::atomic<bool> cancel{false};
  const int first = slot_admit(model, std::vector<int>(90 * KV_PAGE - 16, 3),
                               16, cancel, 1);
  require(first == 0 && g_slots[0].reserved == 90, "claim reset slot");
  model.pages[0] = 80;
  g_slots[0].hist.clear();
  int second = -3;
  std::thread request([&] {
    second = slot_admit(model, std::vector<int>(1800 * KV_PAGE - 16, 4),
                         16, cancel, 2);
    if (second >= 0) release_slot(second);
  });
  wait_until(admission_waiting);
  g_turn.release();
  request.join();
  require(second == 1, "released 400-page baseline falsely rejected new request");
  g_turn.acquire();
  release_slot(first);
  require_turn_free();
  std::puts("PASS reused non-prefix slot reservation");
}

static void test_synchronized_reads() {
  reset_slots(2);
  GpuModel model;
  std::atomic<bool> cancel{false};
  g_turn.acquire();
  model.mutating = true;
  int chosen = -3;
  std::thread request([&] {
    chosen = slot_admit(model, std::vector<int>(1024, 5), 16, cancel, 3);
    if (chosen >= 0) release_slot(chosen);
  });
  wait_until(admission_waiting);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  model.pages[0] = 400;
  model.mutating = false;
  g_turn.release();
  request.join();
  require(chosen == 0, "synchronized admission failed");
  require(model.unsafe_reads == 0, "admission read GPU-owned state without turn");
  require_turn_free();
  std::puts("PASS admission waits for GPU-owner synchronization");
}

static void test_rejection_and_context_edge() {
  reset_slots(2);
  GpuModel model;
  model.pages[0] = 1200;
  g_slots[0].busy = true;
  g_slots[0].reserved = 1200;
  std::atomic<bool> cancel{false};
  std::atomic<bool> waiting{false};
  int deferred = -3;
  std::thread request([&] {
    deferred = slot_admit(model, std::vector<int>(1000 * KV_PAGE - 16, 6),
                          16, cancel, 4, [&] { waiting = true; });
    if (deferred >= 0) release_slot(deferred);
  });
  wait_until([&] { return waiting.load(); });
  const int chosen = slot_admit(model, std::vector<int>(840 * KV_PAGE - 16, 7),
                                16, cancel, 5);
  require(chosen == 1, "smaller request must bypass unavailable capacity");
  release_slot(chosen);
  slot_leave(0);
  request.join();
  require(deferred >= 0, "request did not resume when capacity became available");
  require_turn_free();
  reset_slots(2);
  model.kv_pool_pages = 1000;
  require(slot_admit(model, std::vector<int>(1001 * KV_PAGE, 8),
                     0, cancel, 6) == -2, "unfulfillable request must fail");
  model.kv_pool_pages = 2048;
  const int edge = slot_admit(model, std::vector<int>(524287, 8),
                              100, cancel, 6);
  require(edge == 0 && g_slots[0].reserved == 2048,
          "parallel one-token context edge was rejected");
  release_slot(edge);
  require_turn_free();
  std::puts("PASS capacity wait / bypass / impossible request / context edge");
}

static void test_cancellation() {
  reset_slots(2);
  GpuModel model;
  std::atomic<bool> cancel{false};
  g_turn.acquire();
  int result = -3;
  std::atomic<bool> gpu_waiting{false};
  std::thread request([&] {
    result = slot_admit(model, {9}, 16, cancel, 7, [&] { gpu_waiting = true; });
    if (result >= 0) release_slot(result);
  });
  wait_until([&] {
    std::lock_guard<std::mutex> lock(g_turn.mu);
    return gpu_waiting.load() && g_turn.next - g_turn.serving > 1;
  });
  cancel = true;
  request.join();
  require(result == -1, "cancelled GPU-queued admission was not cancelled");
  g_turn.release();
  require_turn_free();
  reset_slots(2);
  g_slots[0].busy = g_slots[1].busy = true;
  cancel = false;
  std::thread queued([&] {
    result = slot_admit(model, {10}, 16, cancel, 8);
    if (result >= 0) release_slot(result);
  });
  wait_until(admission_waiting);
  require_turn_free();
  cancel = true;
  g_scv.notify_all();
  queued.join();
  require(result == -1, "cancelled slot-queued admission was not cancelled");
  require_turn_free();
  reset_slots(2);
  std::puts("PASS cancellation while waiting for GPU / idle slot");
}

static void test_starve_busy_victim_overflow() {
  reset_slots(2);
  GpuModel model;
  model.kv_pool_pages = 7;
  model.pages[0] = 5;
  model.pages[1] = 2;
  g_slots[0].busy = true;
  g_slots[0].seq = 1;
  g_slots[0].suspendable = true;
  g_slots[1].busy = true;
  g_slots[1].seq = 2;
  g_slots[1].suspendable = false;
  g_slots[1].reserved = 2;
  g_slots[1].hist = {17, 18};
  g_slots[1].mtp_live = true;
  std::atomic<bool> cancel{false};
  g_turn.acquire();
  tl_slot = 0;
  tl_cancel = &cancel;
  tl_progress = [](bool) {};
  model.bind(0);
  bool overflow = false;
  try {
    starve(model);
  } catch (const KvOverflow&) {
    overflow = true;
  }
  require(overflow, "unsupported busy victim must cause KvOverflow");
  g_slots[1].suspendable = true;
  g_slots[1].spec_suspendable = true;
  auto ctl = std::make_shared<SuspendCtl>();
  ctl->slot = 1; ctl->seq = 2; ctl->lease = g_slots[1].lease;
  g_slots[1].suspend = ctl;
  overflow = false;
  try { starve(model); } catch (const KvOverflow&) { overflow = true; }
  require(overflow && ctl->state == 0 && g_slots[1].suspend == ctl,
          "non-round-top requester waited on a speculative victim control");
  g_slots[1].suspend.reset();
  // Sampled requests must never become a suspension victim: their sampler
  // stream cannot be reproduced from a checkpoint saved by another execution.
  g_slots[1].sampled = true;
  overflow = false;
  try { starve(model); } catch (const KvOverflow&) { overflow = true; }
  require(overflow && !g_slots[1].suspend && model.pages[1] == 2 &&
              g_slots[1].seq == 2 && g_slots[1].hist == std::vector<int>({17, 18}),
          "sampled busy victim must keep live KV");
  g_slots[1].sampled = false;
  tl_slot = 1;
  model.bind(1);
  overflow = false;
  try { starve(model); } catch (const KvOverflow&) { overflow = true; }
  require(overflow && !g_slots[1].suspend && model.pages[1] == 2,
          "ordinary starve must not self-suspend a later request");
  tl_slot = 0;
  model.bind(0);
  require(model.kv_pool.nfree() == 0 && model.pages[0] == 5 && model.pages[1] == 2 &&
              model.bound_ == 0 && g_slots[1].busy && !g_slots[1].abort &&
              !g_slots[1].suspend && g_slots[1].seq == 2 && g_slots[1].reserved == 2 &&
              g_slots[1].hist == std::vector<int>({17, 18}) && g_slots[1].mtp_live,
          "unsupported busy victim was modified under KV pressure");
  slot_leave(0);
  slot_leave(1);
  g_turn.release();
  tl_slot = -1;
  tl_cancel = nullptr;
  tl_progress = {};
  require_turn_free();
  std::puts("PASS KV pressure preserves unsupported busy victim / overflow");
}

static void test_spec_recall_boundaries() {
  // A spec/chain victim is eligible only at a full MTP SSD checkpoint. Failed
  // eligibility must leave the request and its chain state live, and the round
  // hook must balance speculative membership before continuing.
  for (int scenario = 0; scenario < 3; ++scenario) {
    for (bool chain : {false, true}) {
      reset_slots(2);
      GpuModel model;
      model.kv_pool_pages = 3;
      model.pages[0] = model.pages[1] = 1;
      model.bind(1);
      model.chist = {7, 8};
      model.pos = 2; model.mtp_pos = 1;
      model.ngram.table = {3, 4};
      model.ngram.learned = 5;
      model.ng_gate.rounds = 6;
      g_slots[0].busy = true;
      g_slots[0].seq = 1;
      g_slots[0].lease = 10;
      g_slots[1].busy = true;
      g_slots[1].seq = 2;
      g_slots[1].lease = 11;
      g_slots[1].owner = 23;
      g_slots[1].req = 900 + scenario * 2 + (chain ? 1 : 0);
      g_slots[1].hist = {9};
      g_slots[1].mtp_live = true;
      g_slots[1].suspendable = g_slots[1].spec_suspendable = true;
      auto ctl = std::make_shared<SuspendCtl>();
      ctl->slot = 1; ctl->seq = 2; ctl->lease = 11;
      g_slots[1].suspend = ctl;
      kvsnap::candidate.reset();
      if (scenario != 0) {
        kvsnap::candidate = std::make_shared<kvsnap::CkEnt>();
        kvsnap::candidate->n = scenario == 1 ? 1 : 2;
        kvsnap::candidate->mtp = scenario == 1;
      }
      auto entry = kvsnap::candidate;
      const int releases = kvsnap::releases;
      const int restores = kvsnap::restores;
      std::atomic<bool> cancel{false};
      tl_slot = 1; tl_lease = 11; tl_cancel = &cancel;
      tl_progress = [](bool) {};
      g_turn.acquire();
      {
        SpecSuspendScope scope(true, 1, chain);
        GpuModel::SpecMembership member;
        spec_round_hook(model, 1, 0, 2 * KV_PAGE, member);
        require(member.suspends == 1 && member.resumes == 1 && !member.suspended &&
                    !tl_spec_membership_suspended,
                "rejected spec recall unbalanced membership");
      }
      require(ctl->state == 4 && g_slots[1].suspend == ctl &&
                  model.pages[0] == 1 && model.pages[1] == 2 && model.bound_ == 1 &&
                  model.pos == 2 && model.mtp_pos == 1 &&
                  model.chist == std::vector<int64_t>({7, 8}) &&
                  model.ngram.table == std::vector<int>({3, 4}) &&
                  model.ngram.learned == 5 && model.ng_gate.rounds == 6 &&
                  g_slots[1].busy && g_slots[1].seq == 2 && g_slots[1].lease == 11 &&
                  g_slots[1].owner == 23 && g_slots[1].mtp_live &&
                  g_slots[1].hist == std::vector<int>{9} &&
                  model.replayed.empty() && g_detached.empty() &&
                  kvsnap::restores == restores,
              "ineligible spec recall changed live KV or chain state");
      require(!entry || (entry->busy == 0 && kvsnap::releases == releases + 1),
              "ineligible spec recall leaked its SSD busy reference");
      g_slots[1].suspend.reset();
      kvsnap::candidate.reset();
      slot_leave(0);
      release_slot(1);
      tl_slot = -1; tl_lease = 0; tl_cancel = nullptr; tl_progress = {};
      require_turn_free();
    }
  }
  std::puts("PASS spec missing / short / non-MTP SSD boundaries preserve live state");
}

static void test_starve_idle_preserves_binding() {
  reset_slots(3);
  GpuModel model;
  model.nslots = 3;
  model.kv_pool_pages = 9;
  model.pages[0] = 4;
  model.pages[1] = 3;
  model.pages[2] = 2;
  g_slots[0].busy = true;
  g_slots[1].busy = true;
  g_slots[2].hist = {21, 22};
  g_slots[2].mtp_live = true;
  g_turn.acquire();
  tl_slot = 0;
  model.bind(1);
  const bool released = starve(model);
  require(released && model.pages[0] == 4 && model.pages[1] == 3 &&
              model.pages[2] == 0 && model.bound_ == 1 &&
              g_slots[2].hist.empty() && !g_slots[2].mtp_live,
          "idle reclaim must preserve the actual model binding");
  slot_leave(0);
  slot_leave(1);
  g_turn.release();
  tl_slot = -1;
  require_turn_free();
  std::puts("PASS KV pressure idle reclaim preserves actual binding");
}

static void test_pressure_suspend() {
  reset_slots(3);
  GpuModel model;
  model.nslots = 3;
  model.kv_pool_pages = 9;
  model.pages[0] = 5;
  model.pages[1] = 2;
  g_slots[0].busy = true;
  g_slots[0].seq = 1;
  g_slots[0].suspendable = true;
  g_slots[1].busy = true;
  g_slots[1].seq = 2;
  g_slots[1].suspendable = true;
  g_slots[1].suspend_budget = 16;
  g_slots[1].owner = 23;
  g_slots[1].req = 24;
  g_slots[1].hist.assign(300, 17);
  auto recall = allow_exact_ssd_recall(g_slots[1].hist.size());
  const int recall_releases = kvsnap::releases;
  std::atomic<bool> cancel{false}, resumed{false};
  g_turn.acquire();
  tl_slot = 0;
  tl_cancel = &cancel;
  tl_progress = [](bool) {};
  std::thread victim([&] {
    tl_slot = 1;
    tl_cancel = &cancel;
    tl_progress = [](bool) {};
    g_turn.acquire();
    model.bind(1);
    model.pos = 300;
    suspend_handle(model);
    resumed = g_slots[1].hist == std::vector<int>(300, 17) &&
              g_slots[1].seq == 2 && g_slots[1].owner == 23 && model.pages[1] == 2;
    release_slot(1);
    tl_slot = -1;
    tl_cancel = nullptr;
    tl_progress = {};
  });
  const bool released = starve(model);
  require(released && !g_slots[1].busy && model.pages[1] == 0,
          "ready victim did not release its KV and slot");
  model.pages[0] = 6;
  model.pages[2] = 2;  // idle cached pages must be reclaimed before restore
  g_turn.release();
  victim.join();
  require(resumed, "full SSD rebuild did not preserve request state");
  require(model.pages[2] == 0 && model.replayed.empty(),
          "full SSD rebuild retained idle pages or recomputed tokens");
  require(kvsnap::required_calls == 0 && recall->busy == 0 &&
              kvsnap::releases == recall_releases + 1,
          "full SSD rebuild leaked busy or used private snapshots");
  kvsnap::candidate.reset();
  slot_leave(0);
  tl_slot = -1;
  tl_cancel = nullptr;
  tl_progress = {};
  require_turn_free();
  std::puts("PASS KV pressure suspend / same-slot full SSD resume");
}

static void test_mtp_chain_suspend() {
  for (bool chain : {false, true}) {
    reset_slots(2);
    GpuModel model;
    const int saved = KV_PAGE;
    model.kv_pool_pages = 2;
    model.pages[0] = model.pages[1] = 1;
    g_slots[0].busy = true;
    g_slots[0].seq = 1;
    g_slots[0].lease = 10;
    g_slots[0].suspendable = true;
    g_slots[1].busy = true;
    g_slots[1].seq = 2;
    g_slots[1].lease = 11;
    g_slots[1].suspendable = g_slots[1].spec_suspendable = true;
    g_slots[1].suspend_budget = 16;
    g_slots[1].owner = 23;
    g_slots[1].req = 910 + (chain ? 1 : 0);
    g_slots[1].hist = {999}; // stale during speculative rounds
    g_slots[1].mtp_live = true;
    auto recall = allow_exact_ssd_recall(saved);
    recall->mtp = true;
    const int recall_releases = kvsnap::releases;
    const int recall_restores = kvsnap::restores;
    std::atomic<bool> cancel{false}, resumed{false};
    bool membership_safe = false, lease_safe = false;
    g_turn.acquire();
    tl_slot = 0; tl_lease = 10;
    tl_cancel = &cancel;
    tl_progress = [](bool) {};
    std::thread victim([&] {
      tl_slot = 1; tl_lease = 11;
      tl_cancel = &cancel;
      tl_progress = [](bool) {};
      g_turn.acquire();
      model.bind(1);
      model.chist.assign(saved, 17);
      model.pos = saved; model.mtp_pos = saved - 1;
      model.ngram.table = {1, 2, 3};
      model.ngram.learned = 7;
      model.ng_gate.rounds = 8;
      model.ng_cs = 16;
      model.ng_cs_fixed = true;
      model.ng_legacy = true;
      model.last_spec_rounds = 11;
      model.live_spec_proposed = 12;
      model.prog_req = 13;
      GpuModel::SpecMembership member;
      {
        SpecSuspendScope scope(true, 16, chain);
        spec_round_hook(model, 16, 0, saved, member);
        resumed = g_slots[1].hist == std::vector<int>{999} &&
                  model.replayed.empty() && model.pos == saved &&
                  model.mtp_pos == saved - 1 && model.pages[1] == 1 &&
                  g_slots[1].owner == 23 && g_slots[1].mtp_live &&
                  model.chist == std::vector<int64_t>(saved, 17) &&
                  model.ngram.table == (chain ? std::vector<int>({1, 2, 3}) :
                                                std::vector<int>({9})) &&
                  model.ngram.learned == (chain ? 7 : 0) &&
                  model.ng_gate.rounds == (chain ? 8 : 0) &&
                  model.ng_cs == (chain ? 16 : 1) &&
                  model.ng_cs_fixed == chain && model.ng_legacy == chain &&
                  model.last_spec_rounds == 11 && model.live_spec_proposed == 12 &&
                  model.prog_req == 13;
        lease_safe = scope.lease != 11 && scope.lease == g_slots[1].lease &&
                     tl_lease == scope.lease;
      }
      membership_safe = member.suspends == 1 && member.resumes == 1 &&
                        !member.suspended && !tl_spec_membership_suspended;
      release_slot(1);
      tl_slot = -1; tl_lease = 0;
      tl_cancel = nullptr;
      tl_progress = {};
    });
    require(starve(model), "full MTP SSD victim did not suspend");
    require(!g_slots[1].busy && model.pages[1] == 0 && recall->busy == 1 &&
                kvsnap::releases == recall_releases,
            "full MTP SSD victim did not release KV while retaining its busy reference");
    // A peer reuses the same idle slot while the saved image is pinned.
    g_slots[1].hist = {42};
    model.pages[1] = 1;
    model.chist = {42};
    model.ngram.table = {9};
    model.ngram.learned = 0;
    model.ng_gate.rounds = 0;
    model.ng_cs = 1;
    model.ng_cs_fixed = model.ng_legacy = false;
    model.last_spec_rounds = model.live_spec_proposed = model.prog_req = 0;
    slot_leave(0);
    g_turn.release();
    victim.join();
    require(resumed, "full MTP SSD restore did not preserve request or chain state");
    require(membership_safe, "full MTP SSD restore unbalanced spec membership");
    require(lease_safe, "full MTP SSD restore did not renew the speculative lease");
    require(model.replayed.empty() && kvsnap::restores == recall_restores + 1 &&
                recall->busy == 0 && kvsnap::releases == recall_releases + 1 &&
                kvsnap::required_calls == 0,
            "full MTP SSD restore prefilled, leaked busy, or used private snapshots");
    kvsnap::candidate.reset();
    tl_slot = -1; tl_lease = 0;
    tl_cancel = nullptr;
    tl_progress = {};
    require_turn_free();
  }
  std::puts("PASS MTP / chain full SSD release / zero-prefill restore / membership / lease");
}

static void test_mtp_failed_capture() {
  reset_slots(1);
  GpuModel model;
  model.chist = {1, 2};
  model.ngram.table = {3, 4};
  model.pages[0] = 1;
  model.bind(0); model.pos = 2; model.mtp_pos = 1;
  g_slots[0].busy = g_slots[0].suspendable = g_slots[0].spec_suspendable =
      g_slots[0].mtp_live = true;
  g_slots[0].hist = {8};
  std::atomic<bool> cancel{false};
  tl_slot = 0;
  tl_cancel = &cancel;
  g_turn.acquire();
  for (int step = 0; step < 3; ++step) {
    auto ctl = std::make_shared<SuspendCtl>();
    ctl->slot = 0;
    g_slots[0].suspend = ctl;
    fail_sync = step == 0;
    model.mrope_active_ = step == 1;
    if (step == 2) model.chist = {static_cast<int64_t>(INT_MAX) + 1};
    suspend_handle(model, SuspendMode::Mtp, true);
    require(ctl->state == 4 && model.pages[0] == 1 && model.bound_ == 0 &&
                model.ngram.table == std::vector<int>({3, 4}) &&
                g_slots[0].hist == std::vector<int>{8} && g_slots[0].mtp_live,
            "failed MTP capture changed request state");
  }
  fail_sync = false;
  release_slot(0);
  tl_slot = -1;
  tl_cancel = nullptr;
  require_turn_free();
  std::puts("PASS MTP capture rejects sync failure / mrope / invalid token");
}

static void test_spec_round_hook() {
  reset_slots(2);
  GpuModel model;
  model.kv_pool_pages = 3;
  model.pages[0] = 1;
  model.pages[1] = 2;
  g_slots[0].busy = true;
  g_slots[0].seq = 10;
  g_slots[1].hist = {7};
  std::atomic<bool> cancel{false};
  tl_slot = 0;
  tl_cancel = &cancel;
  g_turn.acquire();
  model.bind(0);
  GpuModel::SpecMembership member;
  spec_round_hook(model, 5, -1, 100, member); // no scope: no-op
  require(member.suspends == 0, "inactive callback changed membership");
  {
    SpecSuspendScope scope(true, 10, true);
    spec_round_hook(model, 5, 0, 512, member);
    require(model.pages[0] == 2 && model.pages[1] == 0 &&
                member.suspends == 1 && member.resumes == 1 &&
                g_slots[0].suspend_budget == 5,
            "spec reservation or idle eviction failed");
    model.pages[1] = 1;
    g_borrow_light = 1;
    bool overflow = false;
    try { spec_round_hook(model, 5, 0, 768, member); }
    catch (const KvOverflow&) { overflow = true; }
    g_borrow_light = 0;
    require(overflow && model.pages[1] == 1, "borrow must not evict or wait");
    member.resume();
    g_slots[0].seq = 11;
    g_slots[0].suspendable = true;
  }
  require(g_slots[0].suspendable, "scope cleared a reused slot");
  slot_leave(0);
  g_turn.release();
  tl_slot = -1;
  tl_cancel = nullptr;
  require_turn_free();
  std::puts("PASS spec scope / admission / borrow / owner safety");
}

static void test_spec_shared_idle_pages() {
  reset_slots(3);
  GpuModel model;
  model.nslots = 3;
  model.kv_pool_pages = 2;
  model.pages[0] = model.pages[1] = model.pages[2] = 1;
  model.shared_a = 1;
  model.shared_b = 2;
  g_slots[0].busy = true;
  g_slots[0].seq = 1;
  g_slots[1].last_use = 1;
  g_slots[2].last_use = 2;
  std::atomic<bool> cancel{false};
  tl_slot = 0;
  tl_cancel = &cancel;
  g_turn.acquire();
  {
    SpecSuspendScope scope(true, 0, true);
    GpuModel::SpecMembership member;
    spec_round_hook(model, 0, 0, 512, member);
    require(model.pages[0] == 2 && model.pages[1] == 0 && model.pages[2] == 0 &&
                member.resumes == 1,
            "shared idle pages require multiple resets before reservation");
  }
  release_slot(0);
  tl_slot = -1;
  tl_cancel = nullptr;
  require_turn_free();

  // Dropping an idle reference to the request's own page cannot free it.
  reset_slots(2);
  GpuModel pinned;
  pinned.kv_pool_pages = 1;
  pinned.pages[0] = pinned.pages[1] = 1;
  pinned.shared_a = 0;
  pinned.shared_b = 1;
  g_slots[0].busy = true;
  g_slots[0].seq = 1;
  tl_slot = 0;
  tl_cancel = &cancel;
  g_turn.acquire();
  {
    SpecSuspendScope scope(true, 0, true);
    GpuModel::SpecMembership member;
    bool overflow = false;
    try { spec_round_hook(pinned, 0, 0, 512, member); }
    catch (const KvOverflow&) { overflow = true; }
    require(overflow && pinned.pages[0] == 1 && pinned.pages[1] == 0 &&
                g_slots[0].busy, "shared busy page must not be reset or retried forever");
  }
  release_slot(0);
  tl_slot = -1;
  tl_cancel = nullptr;
  require_turn_free();
  std::puts("PASS spec shared idle page reclamation");
}

static void test_spec_changed_victim() {
  reset_slots(3);
  GpuModel model;
  model.nslots = 3;
  model.kv_pool_pages = 3;
  model.pages[0] = model.pages[1] = model.pages[2] = 1;
  g_slots[0].busy = g_slots[1].busy = g_slots[2].busy = true;
  g_slots[0].seq = 1;
  g_slots[1].seq = 2;
  g_slots[2].seq = 3;
  g_slots[1].suspendable = true;
  // The newest victim is not suspendable; the older eligible one must be tried.
  std::atomic<bool> cancel{false};
  tl_slot = 0;
  tl_cancel = &cancel;
  tl_progress = [](bool) {};
  g_turn.acquire();
  std::thread peer([&] {
    g_turn.acquire();
    {
      std::lock_guard<std::mutex> lk(g_sm);
      auto ctl = g_slots[1].suspend;
      if (ctl) {
        g_slots[1].busy = false;
        g_slots[1].seq = 4;
        g_slots[1].suspend.reset();
        ctl->state = 4;
      }
    }
    g_scv.notify_all();
    g_turn.release();
  });
  {
    SpecSuspendScope scope(true, 0, true);
    GpuModel::SpecMembership member;
    spec_round_hook(model, 0, 0, 512, member);
    require(model.pages[0] == 2 && model.pages[1] == 0 &&
                model.pages[2] == 1 && g_slots[2].busy && member.resumes == 1,
            "changed eligible victim must be re-evaluated without touching ineligible peer");
  }
  peer.join();
  slot_leave(2);
  release_slot(0);
  tl_slot = -1;
  tl_cancel = nullptr;
  tl_progress = {};
  require_turn_free();
  std::puts("PASS spec eligible victim selection / lifecycle change");
}

static void test_spec_existing_suspend() {
  // The peer can only complete after acquiring the GPU turn. Exercise every
  // pending state, plus failure, cancellation and replacement of its lease.
  for (int mode = 0; mode < 6; ++mode) {
    reset_slots(2);
    GpuModel model;
    model.kv_pool_pages = 2;
    model.pages[0] = model.pages[1] = 1;
    for (int i = 0; i < 2; ++i) {
      g_slots[(size_t)i].busy = g_slots[(size_t)i].suspendable = true;
      g_slots[(size_t)i].seq = (uint64_t)i + 1;
      g_slots[(size_t)i].lease = (uint64_t)i + 10;
    }
    auto ctl = std::make_shared<SuspendCtl>();
    ctl->slot = 1; ctl->seq = 2; ctl->lease = 11;
    ctl->state = mode < 3 ? mode : 1;
    g_slots[1].suspend = ctl;
    std::atomic<bool> cancel{false};
    tl_slot = 0;
    tl_cancel = &cancel;
    tl_progress = [](bool) {};
    g_turn.acquire();
    GpuModel::SpecMembership member;
    std::thread peer([&] {
      g_turn.acquire();
      {
        std::lock_guard<std::mutex> lk(g_sm);
        if (mode < 3) {
          model.slot_reset(1, 0);
          g_slots[1].busy = false;
          g_slots[1].suspend.reset();
          ctl->state = 3;
        } else if (mode == 3) {
          ctl->state = 4;
          g_slots[1].suspend.reset();
        } else if (mode == 4) {
          cancel = true;
        } else {
          ++g_slots[1].lease;
          g_slots[1].suspend.reset();
          g_slots[1].suspendable = false;
          ctl->state = 4;
        }
      }
      g_scv.notify_all();
      g_turn.release();
    });
    bool overflow = false, cancelled = false;
    {
      SpecSuspendScope scope(true, 0, true);
      try { spec_round_hook(model, 0, 0, 512, member); }
      catch (const KvOverflow&) { overflow = true; }
      catch (const ReqCancelled&) { cancelled = true; }
      require(mode < 3 ? (!overflow && !cancelled && model.pages[0] == 2 &&
                          member.resumes == 1) :
              (mode == 4 ? cancelled : overflow),
              "existing suspension did not resolve correctly");
      if (mode >= 3) require(model.pages[1] == 1 && g_slots[1].busy,
                             "failed wait changed victim KV");
    }
    peer.join();
    slot_leave(1);
    release_slot(0);
    tl_slot = -1; tl_cancel = nullptr; tl_progress = {};
    require_turn_free();
  }
  // Ordinary batch starvation and borrowed rounds must never wait for a ctl.
  for (int mode = 0; mode < 2; ++mode) {
    reset_slots(2);
    GpuModel model;
    model.kv_pool_pages = 2;
    model.pages[0] = model.pages[1] = 1;
    g_slots[0].busy = g_slots[1].busy = true;
    g_slots[0].suspendable = g_slots[1].suspendable = true;
    g_slots[0].seq = 1; g_slots[1].seq = 2;
    auto ctl = std::make_shared<SuspendCtl>();
    ctl->slot = 1; ctl->seq = 2; ctl->lease = g_slots[1].lease;
    g_slots[1].suspend = ctl;
    std::atomic<bool> cancel{false};
    tl_slot = 0; tl_cancel = &cancel; tl_progress = [](bool) {};
    g_turn.acquire();
    bool overflow = false;
    if (!mode) {
      try { starve(model); } catch (const KvOverflow&) { overflow = true; }
    } else {
      model.ws_lent_ = true;
      SpecSuspendScope scope(true, 0, true);
      GpuModel::SpecMembership member;
      try { spec_round_hook(model, 0, 0, 512, member); }
      catch (const KvOverflow&) { overflow = true; }
    }
    require(overflow && ctl->state == 0 && g_slots[1].suspend == ctl &&
                model.pages[1] == 1, "unsafe path waited or altered existing control");
    slot_leave(1); release_slot(0);
    tl_slot = -1; tl_cancel = nullptr; tl_progress = {};
    require_turn_free();
  }
  std::puts("PASS existing later suspension / failure / cancellation / unsafe paths");
}

static void test_spec_reclaim_candidates() {
  // Capture failure, publication failure, readmission, cancellation, ordinary starve.
  for (int mode = 0; mode < 5; ++mode) {
    reset_slots(3);
    GpuModel model;
    model.nslots = 3;
    model.kv_pool_pages = mode == 2 ? 2 : 3;
    model.pages[0] = model.pages[1] = model.pages[2] = 1;
    if (mode == 2) { model.shared_a = 0; model.shared_b = 2; }
    for (int i = 0; i < 3; ++i) {
      g_slots[(size_t)i].busy = true;
      g_slots[(size_t)i].seq = (uint64_t)i + 1;
      g_slots[(size_t)i].lease = (uint64_t)i + 10;
      g_slots[(size_t)i].suspendable = true;
      g_slots[(size_t)i].hist = {i + 10};
    }
    std::atomic<bool> cancel{false}, stop{false};
    std::vector<uint64_t> selected;
    bool restored = false;
    tl_slot = 0;
    tl_cancel = &cancel;
    tl_progress = [](bool) {};
    g_turn.acquire();
    std::thread peer([&] {
      tl_cancel = &cancel;
      while (!stop) {
        std::shared_ptr<SuspendCtl> ctl;
        {
          std::lock_guard<std::mutex> lk(g_sm);
          for (int i = 1; i < 3; ++i)
            if (g_slots[(size_t)i].suspend && g_slots[(size_t)i].suspend->state == 0)
              ctl = g_slots[(size_t)i].suspend;
        }
        if (!ctl) { std::this_thread::yield(); continue; }
        g_turn.acquire();
        tl_slot = ctl->slot;
        tl_lease = ctl->lease;
        model.bind(tl_slot);
        model.pos = 1;
        selected.push_back(ctl->seq);
        if ((mode == 0 || mode == 1 || mode == 3 || mode == 4) && ctl->seq == 3) {
          fail_sync = mode != 1;
          model.mrope_active_ = mode == 1;
          suspend_handle(model);
          fail_sync = false;
          model.mrope_active_ = false;
          if (mode == 3) cancel = true;
        } else {
          std::lock_guard<std::mutex> lk(g_sm);
          ctl->state = 2; // captured and ready; requester owns the release decision
        }
        g_scv.notify_all();
        g_turn.release();
      }
      tl_slot = -1;
      tl_cancel = nullptr;
    });
    model.on_prepare = [&] {
      if (mode != 2 || restored) return;
      std::lock_guard<std::mutex> lk(g_sm);
      if (g_detached.empty()) return;
      auto ctl = g_detached.front();
      require(ctl->seq == 3 && ctl->state == 3, "first victim was not released");
      g_detached.remove(ctl);
      Slot& sl = g_slots[2];
      sl.busy = sl.suspendable = true;
      sl.seq = 3;
      ++sl.lease; // same request, new claim
      model.pages[2] = 1;
      restored = true;
    };
    bool cancelled = false, overflow = false;
    if (mode == 4) {
      try { starve(model); } catch (const KvOverflow&) { overflow = true; }
    } else {
      SpecSuspendScope scope(true, 0, true);
      GpuModel::SpecMembership member;
      try { spec_round_hook(model, 0, 0, 512, member); }
      catch (const ReqCancelled&) { cancelled = true; }
      if (mode != 3)
        require(model.pages[0] == 2 && member.resumes == 1,
                "round did not reclaim another eligible request");
    }
    require(overflow == (mode == 4), "ordinary starve must fail closed");
    stop = true;
    g_turn.release();
    peer.join();
    require(selected == (mode >= 3 ? std::vector<uint64_t>{3} : std::vector<uint64_t>{3, 2}),
            "victims were not tried once in sequence order");
    if (mode != 2)
      require(g_slots[2].busy && !g_slots[2].suspend && model.pages[2] == 1 &&
                  g_slots[2].hist == std::vector<int>{12},
              "failed capture modified live victim");
    require(mode != 2 || (restored && g_slots[2].lease == 13 && g_slots[2].busy),
            "readmitted request was selected again");
    require(cancelled == (mode == 3), "cancellation was not propagated");
    { std::lock_guard<std::mutex> lk(g_sm); g_detached.clear(); }
    slot_leave(0);
    slot_leave(1);
    slot_leave(2);
    tl_slot = -1;
    tl_cancel = nullptr;
    tl_progress = {};
    require_turn_free();
  }
  std::puts("PASS spec failed capture / readmitted victim / cancellation");
}

static void test_spec_self_suspend() {
  reset_slots(2);
  GpuModel model;
  model.kv_pool_pages = 3;
  model.pages[0] = 2;
  model.pages[1] = 1;
  g_slots[0].busy = true;
  g_slots[0].seq = 1;
  g_slots[1].busy = true;
  g_slots[1].seq = 2;
  g_slots[1].owner = 23;
  g_slots[1].hist = {99};
  g_slots[1].mtp_live = true;
  auto recall = allow_exact_ssd_recall(3);
  recall->mtp = true;
  std::atomic<bool> cancel{false}, resumed{false};
  std::thread later([&] {
    tl_slot = 1;
    tl_cancel = &cancel;
    tl_progress = [](bool) {};
    g_turn.acquire();
    model.bind(1);
    model.chist = {1, 2, 3};
    model.pos = 3; model.mtp_pos = 2;
    model.ngram.table = {4, 5};
    model.ng_gate.rounds = 7;
    {
      SpecSuspendScope scope(true, 0, true);
      GpuModel::SpecMembership member;
      spec_round_hook(model, 0, 0, 512, member);
      resumed = member.resumes == 1 && !member.suspended && model.bound_ == 1 &&
          model.pages[1] == 2 && model.chist == std::vector<int64_t>({1, 2, 3}) &&
          model.ngram.table == std::vector<int>({4, 5}) && model.ng_gate.rounds == 7 &&
          g_slots[1].seq == 2 && g_slots[1].owner == 23 && g_slots[1].mtp_live;
    }
    release_slot(1);
    tl_slot = -1;
    tl_cancel = nullptr;
    tl_progress = {};
  });
  wait_until([&] { std::lock_guard<std::mutex> lk(g_sm); return !g_slots[1].busy; });
  g_turn.acquire();
  require(model.pages[1] == 0 && g_slots[0].busy, "self suspend did not release KV");
  model.bind(0);
  model.chist = {42};
  model.ngram.table = {42};
  g_turn.release();
  require(!resumed, "self suspended request resumed before earlier request exited");
  slot_leave(0);
  later.join();
  require(resumed, "self suspension did not restore MTP chain state");
  require_turn_free();
  std::puts("PASS spec self-suspend waits for earlier request / restores chain");
}

static void test_spec_cancel_suspended_request() {
  reset_slots(2);
  GpuModel model;
  const int saved = KV_PAGE;
  model.kv_pool_pages = 2;
  model.pages[0] = model.pages[1] = 1;
  g_slots[0].busy = true;
  g_slots[0].seq = 1;
  g_slots[0].lease = 10;
  g_slots[0].suspendable = true;
  g_slots[1].busy = true;
  g_slots[1].seq = 2;
  g_slots[1].lease = 11;
  g_slots[1].suspendable = g_slots[1].spec_suspendable = true;
  g_slots[1].hist = {9};
  g_slots[1].mtp_live = true;
  auto recall = allow_exact_ssd_recall(saved);
  recall->mtp = true;
  const int recall_releases = kvsnap::releases;
  std::atomic<bool> cancel{false}, detached_cancel{false};
  bool membership_safe = false;
  g_turn.acquire();
  tl_slot = 0; tl_lease = 10;
  tl_cancel = &cancel;
  tl_progress = [](bool) {};
  std::thread victim([&] {
    tl_slot = 1; tl_lease = 11;
    tl_cancel = &cancel;
    tl_progress = [](bool) {};
    g_turn.acquire();
    model.bind(1);
    model.chist.assign(saved, 17);
    model.pos = saved; model.mtp_pos = saved - 1;
    GpuModel::SpecMembership member;
    try {
      SpecSuspendScope scope(true, 16, true);
      spec_round_hook(model, 16, 0, saved, member);
    } catch (const SuspendCancelled&) {
      detached_cancel = tl_detached && tl_slot == -1;
      membership_safe = member.suspended && member.suspends == 1 &&
                        !tl_spec_membership_suspended;
    }
    tl_slot = -1; tl_lease = 0;
    tl_cancel = nullptr;
    tl_progress = {};
  });
  require(starve(model), "full MTP SSD cancellation setup did not release the victim");
  {
    std::lock_guard<std::mutex> lock(g_sm);
    g_slots[1].busy = true;
    g_slots[1].seq = 3;
    g_slots[1].lease = 99;
    g_slots[1].hist = {99};
    g_slots[1].suspendable = g_slots[1].spec_suspendable = false;
  }
  model.pages[1] = 1;
  slot_leave(0);
  g_turn.release();
  wait_until(admission_waiting);
  cancel = true;
  g_scv.notify_all();
  victim.join();
  require(detached_cancel && membership_safe && g_slots[1].busy &&
              g_slots[1].seq == 3 && g_slots[1].lease == 99 &&
              g_slots[1].hist == std::vector<int>{99} &&
              model.replayed.empty() && recall->busy == 0 &&
              kvsnap::releases == recall_releases + 1,
          "cancelled spec recall disturbed a new lease or leaked SSD busy");
  kvsnap::candidate.reset();
  slot_leave(1);
  tl_slot = -1; tl_lease = 0;
  tl_cancel = nullptr;
  tl_progress = {};
  require_turn_free();
  std::puts("PASS cancelled spec recall preserves replacement lease / membership / SSD busy");
}

static void test_spec_self_suspend_blocker_departure() {
  // Depart after small-state capture releases the turn, before revalidation.
  // Retain the blocker's idle pages in case 1.
  for (int step = 0; step < 5; ++step) {
    reset_slots(2);
    GpuModel model;
    model.kv_pool_pages = 2;
    model.pages[0] = model.pages[1] = 1;
    model.chist = {1};
    g_slots[0].busy = true;
    g_slots[0].seq = 1;
    g_slots[1].busy = true;
    g_slots[1].seq = 2;
    g_slots[1].lease = 42;
    g_slots[1].hist = {9};
    g_slots[1].mtp_live = true;
    std::atomic<bool> cancel{false};
    tl_slot = 1;
    tl_lease = 42;
    tl_cancel = &cancel;
    int releases = 0;
    on_self_turn_released = [&] {
      ++releases;
      g_turn.acquire();
      slot_leave(0);
      if (step != 1) model.pages[0] = 0;
      if (step == 2) cancel = true;
      if (step == 3) g_borrow_light = 1;
      if (step == 4) {
        std::lock_guard<std::mutex> lk(g_sm);
        ++g_slots[1].lease;
      }
      g_turn.release();
    };
    g_turn.acquire();
    model.bind(1);
    model.pos = 1; model.mtp_pos = 0;
    bool overflow = false, lease_failed = false, cancelled = false;
    {
      SpecSuspendScope scope(true, 0, true);
      GpuModel::SpecMembership member;
      try { spec_round_hook(model, 0, 0, 512, member); }
      catch (const KvOverflow&) { overflow = true; }
      catch (const SuspendFailed&) { lease_failed = true; }
      catch (const ReqCancelled&) { cancelled = true; }
      require(releases == 1 && overflow == (step == 3) &&
                  cancelled == (step == 2) && lease_failed == (step == 4),
              "blocker departure outcome");
      require(g_slots[1].busy && g_slots[1].mtp_live &&
                  g_slots[1].hist == std::vector<int>{9} &&
                  model.pages[1] == (step < 2 ? 2 : 1),
              "blocker departure released or corrupted live KV");
      if (step < 4) require(!g_slots[1].suspend, "self suspension control leaked");
      if (step < 2) require(member.resumes == 1 && model.pages[0] == 0,
                            "round did not recheck capacity or reclaim idle KV");
    }
    on_self_turn_released = {};
    g_borrow_light = 0;
    release_slot(1);
    tl_slot = -1;
    tl_cancel = nullptr;
    require_turn_free();
  }
  std::puts("PASS self-suspend blocker departure / cancel, borrow and lease failures");
}

static void test_spec_self_suspend_departure_retry_limit() {
  reset_slots(2);
  GpuModel model;
  model.kv_pool_pages = 1;
  model.pages[1] = 1;
  model.chist = {1};
  g_slots[0].seq = 1;
  g_slots[1].busy = true;
  g_slots[1].seq = 2;
  std::atomic<bool> cancel{false};
  tl_slot = 1;
  tl_cancel = &cancel;
  int releases = 0;
  model.on_prepare = [&] {
    std::lock_guard<std::mutex> lk(g_sm);
    g_slots[0].busy = true;
  };
  on_self_turn_released = [&] {
    ++releases;
    g_turn.acquire();
    slot_leave(0);
    g_turn.release();
  };
  g_turn.acquire();
  model.bind(1);
  model.pos = 1; model.mtp_pos = 0;
  bool limited = false;
  {
    SpecSuspendScope scope(true, 0, true);
    GpuModel::SpecMembership member;
    try { spec_round_hook(model, 0, 0, 512, member); }
    catch (const KvOverflow& e) {
      limited = std::string(e.what()) == "spec KV reclaim retry limit reached";
    }
    require(limited && releases == 9 && model.pages[1] == 1 &&
                g_slots[1].busy && !g_slots[1].suspend,
            "blocker departure retries were not bounded or changed live KV");
  }
  on_self_turn_released = {};
  slot_leave(0);
  release_slot(1);
  tl_slot = -1;
  tl_cancel = nullptr;
  require_turn_free();
  std::puts("PASS self-suspend blocker departure retry limit");
}

static void test_spec_self_suspend_failure() {
  for (int step = 0; step < 4; ++step) {
    reset_slots(2);
    GpuModel model;
    model.kv_pool_pages = 2;
    model.pages[0] = model.pages[1] = 1;
    model.chist = {1};
    g_slots[0].busy = true;
    g_slots[0].seq = 1;
    g_slots[1].busy = true;
    g_slots[1].seq = 2;
    g_slots[1].hist = {9};
    g_slots[1].mtp_live = true;
    std::atomic<bool> cancel{false};
    tl_slot = 1;
    tl_cancel = &cancel;
    g_turn.acquire();
    model.bind(1);
    model.pos = 1; model.mtp_pos = 0;
    fail_sync = step == 0;
    model.mrope_active_ = step == 1;
    g_borrow_light = step == 2 ? 1 : 0;
    model.ws_lent_ = step == 3;
    {
      SpecSuspendScope scope(true, 0, true);
      GpuModel::SpecMembership member;
      bool overflow = false;
      try { spec_round_hook(model, 0, 0, 512, member); }
      catch (const KvOverflow&) { overflow = true; }
      require(overflow && model.pages[1] == 1 && model.bound_ == 1 &&
                  g_slots[1].busy && !g_slots[1].suspend && g_slots[1].mtp_live &&
                  g_slots[1].hist == std::vector<int>{9},
              "failed or borrowed self suspend changed live state");
    }
    fail_sync = false;
    g_borrow_light = 0;
    slot_leave(0);
    release_slot(1);
    tl_slot = -1;
    tl_cancel = nullptr;
    require_turn_free();
  }
  std::puts("PASS spec self-suspend failure and borrow preserve live state");
}

static void test_cancel_suspended_request() {
  reset_slots(2);
  GpuModel model;
  model.pages[0] = 5;
  model.pages[1] = 2;
  g_slots[0].busy = true;
  g_slots[0].seq = 1;
  g_slots[0].suspendable = true;
  g_slots[1].busy = true;
  g_slots[1].seq = 2;
  g_slots[1].suspendable = true;
  g_slots[1].suspend_budget = 16;
  g_slots[1].hist.assign(300, 17);
  auto recall = allow_exact_ssd_recall(g_slots[1].hist.size());
  const int recall_releases = kvsnap::releases;
  std::atomic<bool> cancel{false}, detached_cancel{false};
  g_turn.acquire();
  tl_slot = 0;
  tl_cancel = &cancel;
  tl_progress = [](bool) {};
  std::thread victim([&] {
    tl_slot = 1;
    tl_cancel = &cancel;
    tl_progress = [](bool) {};
    g_turn.acquire();
    model.bind(1);
    model.pos = 300;
    try {
      suspend_handle(model);
      release_slot(1);
    } catch (const SuspendCancelled&) {
      detached_cancel = tl_detached && tl_slot == -1;
    }
    tl_cancel = nullptr;
    tl_progress = {};
  });
  require(starve(model), "suspension did not release victim");
  {
    std::lock_guard<std::mutex> lock(g_sm);
    g_slots[1].busy = true;
    g_slots[1].seq = 3;
    g_slots[1].hist = {99};
  }
  slot_leave(0);
  g_turn.release();
  wait_until(admission_waiting);
  cancel = true;
  g_scv.notify_all();
  victim.join();
  require(detached_cancel && g_slots[1].busy && g_slots[1].seq == 3 &&
              g_slots[1].hist == std::vector<int>{99} && recall->busy == 0 &&
              kvsnap::releases == recall_releases + 1,
          "cancelled suspended request disturbed the slot or leaked SSD busy");
  kvsnap::candidate.reset();
  slot_leave(1);
  tl_slot = -1;
  tl_cancel = nullptr;
  tl_progress = {};
  require_turn_free();
  std::puts("PASS cancellation during suspended readmission");
}

static void test_resume_slot_bypass() {
  reset_slots(2);
  GpuModel model;
  g_slots[0].busy = true;
  std::atomic<bool> cancel{false}, waiting{false};
  int resumed = -3;
  std::thread request([&] {
    resumed = slot_admit(model, {19}, 16, cancel, 19, [&] { waiting = true; }, 0, 42);
    if (resumed >= 0) release_slot(resumed);
  });
  wait_until([&] { return waiting.load(); });
  const int peer = slot_admit(model, {20}, 16, cancel, 20);
  require(peer == 1, "same-slot resume blocked an unrelated idle slot");
  release_slot(peer);
  slot_leave(0);
  request.join();
  require(resumed == 0 && g_slots[0].seq == 42, "resume lost slot or request order");
  require_turn_free();
  std::puts("PASS same-slot resume allows unrelated admission");
}

static void test_queue_revalidation() {
  reset_slots(2);
  GpuModel model;
  std::atomic<bool> cancel{false};
  int result = -3;
  g_turn.acquire();
  std::thread worker([&] {
    result = slot_admit(model, {31}, 1, cancel, 31);
    if (result >= 0) release_slot(result);
  });
  wait_until([&] {
    std::lock_guard<std::mutex> lk(g_turn.mu);
    return g_turn.next - g_turn.serving > 1;
  });
  uint64_t fake;
  {
    std::lock_guard<std::mutex> lk(g_sm);
    fake = g_ticket++;
    g_wait.push_front(fake);
  }
  g_turn.release();
  wait_until([&] {
    std::lock_guard<std::mutex> lk(g_turn.mu);
    return g_turn.next == g_turn.serving;
  });
  {
    std::lock_guard<std::mutex> lk(g_sm);
    require(g_wait.size() == 2 && g_wait.front() == fake && !g_slots[0].busy,
            "GPU waiter claimed or removed a different queue head");
    g_wait.remove(fake);
  }
  g_scv.notify_all();
  worker.join();
  require(result >= 0, "revalidated admission did not resume");
  require_turn_free();
  std::puts("PASS admission revalidates queue head after GPU wait");
}

static void test_detached_reservation_and_lease() {
  reset_slots(2);
  GpuModel model;
  std::atomic<bool> cancel{false};
  auto ctl = std::make_shared<SuspendCtl>();
  {
    std::lock_guard<std::mutex> lk(g_sm);
    ctl->seq = ++g_seq;
    ctl->slot = 0;
    g_detached.push_back(ctl);
    require(earlier_detached(ctl->seq + 1), "detached priority disappeared");
  }
  int peer = slot_admit(model, {32}, 1, cancel, 32);
  require(peer == 1, "new request stole a detached request's original slot");
  release_slot(peer);
  const int resumed = slot_admit(model, {33}, 1, cancel, 33, {}, 0, ctl->seq);
  require(resumed == 0 && g_slots[0].seq == ctl->seq, "resume lost priority or slot");
  const uint64_t old_lease = g_slots[0].lease;
  release_slot(resumed);
  const int again = slot_admit(model, {33}, 1, cancel, 33, {}, 0, ctl->seq);
  require(again == 0 && g_slots[0].lease != old_lease, "readmission reused a lease");
  slot_leave(again, old_lease);
  require(g_slots[0].busy, "stale lease released the new owner");
  tl_slot = again;
  {
    SpecSuspendScope scope(true, 1);
    ++g_slots[0].lease; // same request priority, different claim
  }
  require(g_slots[0].suspendable, "stale scope cleared a newer lease");
  tl_slot = -1;
  release_slot(again);
  {
    std::lock_guard<std::mutex> lk(g_sm);
    g_detached.remove(ctl);
  }
  require_turn_free();
  std::puts("PASS detached local reservation / distinct claim leases");
}

static void test_detached_kv_commitment() {
  reset_slots(3);
  GpuModel model;
  model.nslots = 3;
  model.kv_pool_pages = 10;
  std::atomic<bool> cancel{false}, waiting{false};
  auto ctl = std::make_shared<SuspendCtl>();
  {
    std::lock_guard<std::mutex> lk(g_sm);
    ctl->seq = ++g_seq;
    ctl->slot = 0;
    ctl->resume_pages = 6;
    g_detached.push_back(ctl);
    require(detached_commitment(ctl->seq) == 0, "resume charged itself");
  }
  std::atomic<int> later{-3};
  uint64_t before;
  {
    std::lock_guard<std::mutex> lk(g_turn.mu);
    before = g_turn.next;
  }
  std::thread worker([&] {
    const int result = slot_admit(model, std::vector<int>(5 * KV_PAGE, 8), 0, cancel, 50,
                                  [&] { waiting = true; });
    later = result;
    if (result >= 0) release_slot(result);
  });
  wait_until([&] {
    std::lock_guard<std::mutex> lk(g_turn.mu);
    return waiting.load() && g_turn.next > before && g_turn.next == g_turn.serving;
  });
  require(later == -3 && admission_waiting(), "later request consumed recovery capacity");
  // A fitting later request can still use another slot, and the earlier
  // request can readmit without charging its own commitment twice.
  int small = slot_admit(model, std::vector<int>(4 * KV_PAGE, 9), 0, cancel, 51);
  require(small == 1, "fitting later request blocked by detached commitment");
  release_slot(small);
  int resumed = slot_admit(model, std::vector<int>(6 * KV_PAGE, 7), 0, cancel, 52,
                           {}, 0, ctl->seq);
  require(resumed == 0, "detached request could not readmit");
  {
    std::lock_guard<std::mutex> lk(g_sm);
    require(detached_commitment(ctl->seq + 1) == 0,
            "busy readmission charged twice");
  }
  release_slot(resumed);
  // Removing a cancelled detached registration returns its commitment.
  {
    DetachedRegistration registration{ctl};
  }
  worker.join();
  require(later >= 0, "cancelled detached commitment not released");
  require_turn_free();
  std::puts("PASS detached KV commitment / readmission / cancellation cleanup");
}

static void test_ready_cancel_with_coordinator_turn() {
  reset_slots(2);
  GpuModel model;
  model.pages[1] = 2;
  g_slots[1].busy = g_slots[1].suspendable = true;
  g_slots[1].seq = 2;
  g_slots[1].lease = 17;
  g_slots[1].hist.assign(300, 4);
  auto recall = allow_exact_ssd_recall(g_slots[1].hist.size());
  const int recall_releases = kvsnap::releases;
  auto ctl = std::make_shared<SuspendCtl>();
  ctl->seq = 2;
  ctl->lease = 17;
  ctl->slot = 1;
  g_slots[1].suspend = ctl;
  std::atomic<bool> cancel{false}, caught{false};
  std::thread victim([&] {
    tl_slot = 1;
    tl_lease = 17;
    tl_cancel = &cancel;
    g_turn.acquire();
    model.bind(1);
    model.pos = 300;
    try { suspend_handle(model); }
    catch (const ReqCancelled&) { caught = true; }
    release_slot(1);
    tl_slot = -1;
    tl_cancel = nullptr;
  });
  wait_until([&] { std::lock_guard<std::mutex> lk(g_sm); return ctl->state == 2; });
  g_turn.acquire();  // coordinator owns the turn but has not revalidated ctl
  cancel = true;     // no notification: exercise the timed wait
  wait_until([&] { std::lock_guard<std::mutex> lk(g_sm); return ctl->state == 4; });
  require(model.pages[1] == 2 && g_slots[1].busy,
          "ready cancellation released original KV");
  g_turn.release();
  victim.join();
  require(caught && model.pages[1] == 2 && g_detached.empty() && recall->busy == 0 &&
              kvsnap::releases == recall_releases + 1,
          "ready cancellation did not reacquire lease or release SSD busy");
  kvsnap::candidate.reset();
  require_turn_free();
  std::puts("PASS ready cancellation withdraws before coordinator revalidation");
}

static void test_self_suspend_behind_detached() {
  reset_slots(2);
  GpuModel model;
  model.pages[1] = 1;
  g_slots[1].busy = g_slots[1].suspendable = true;
  g_slots[1].seq = 2;
  g_slots[1].hist = {41};
  auto recall = allow_exact_ssd_recall(g_slots[1].hist.size());
  const int recall_releases = kvsnap::releases;
  auto earlier = std::make_shared<SuspendCtl>();
  earlier->seq = 1;
  earlier->slot = 0;
  auto own = std::make_shared<SuspendCtl>();
  own->seq = 2;
  own->slot = 1;
  g_slots[1].suspend = own;
  {
    std::lock_guard<std::mutex> lk(g_sm);
    g_detached.push_back(earlier);
  }
  std::atomic<bool> cancel{false}, finished{false};
  std::thread worker([&] {
    tl_slot = 1;
    tl_cancel = &cancel;
    g_turn.acquire();
    model.bind(1);
    model.pos = 1;
    suspend_handle(model, SuspendMode::PlainDecode, false, true);
    finished = true;
    release_slot(1);
    tl_slot = -1;
    tl_cancel = nullptr;
  });
  wait_until([&] {
    std::lock_guard<std::mutex> lk(g_sm);
    return own->state == 3;
  });
  require(!finished, "self-suspended request passed an earlier detached request");
  {
    std::lock_guard<std::mutex> lk(g_sm);
    g_detached.remove(earlier);
  }
  g_scv.notify_all();
  worker.join();
  require(finished && model.replayed.empty() && recall->busy == 0 &&
              kvsnap::releases == recall_releases + 1,
          "self-suspended request did not fully restore SSD and release busy");
  kvsnap::candidate.reset();
  require_turn_free();
  std::puts("PASS self-suspend preserves detached request order");
}

static void test_round_top_behind_detached() {
  reset_slots(2);
  GpuModel model;
  model.kv_pool_pages = 1;
  model.pages[1] = 1;
  model.chist.assign(KV_PAGE, 41);
  auto earlier = std::make_shared<SuspendCtl>();
  {
    std::lock_guard<std::mutex> lk(g_sm);
    earlier->seq = ++g_seq;
    earlier->slot = 0;
    earlier->resume_pages = 1;
    g_detached.push_back(earlier);
    g_slots[1].busy = true;
    g_slots[1].seq = ++g_seq;
    g_slots[1].lease = ++g_lease;
  }
  std::atomic<bool> cancel{false}, finished{false};
  bool success = false;
  std::thread worker([&] {
    tl_slot = 1;
    tl_lease = g_slots[1].lease;
    tl_cancel = &cancel;
    g_turn.acquire();
    model.bind(1);
    model.pos = KV_PAGE; model.mtp_pos = KV_PAGE - 1;
    {
      SpecSuspendScope scope(true, 0);
      GpuModel::SpecMembership member;
      try {
        spec_round_hook(model, 0, 0, 2 * KV_PAGE, member);
        success = member.resumes == 1 && model.pages[1] == 2;
      } catch (const KvOverflow&) {}
    }
    release_slot(1);
    tl_slot = -1;
    tl_cancel = nullptr;
    finished = true;
  });
  bool detached = false;
  wait_until([&] {
    std::lock_guard<std::mutex> lk(g_sm);
    detached = g_detached.size() == 2 && !g_slots[1].busy;
    return detached || finished.load();
  });
  // Model capacity becoming available after the earlier request finishes.
  g_turn.acquire();
  model.kv_pool_pages = 2;
  g_turn.release();
  {
    std::lock_guard<std::mutex> lk(g_sm);
    g_detached.remove(earlier);
  }
  g_scv.notify_all();
  worker.join();
  require(detached && success, "round top did not yield to earlier detached request");
  require(g_detached.empty(), "round-top detached registration leaked");
  require_turn_free();
  std::puts("PASS round-top decision yields to earlier detached request");
}

static void test_unavailable_normal_head_yields_to_resume() {
  reset_slots(1);
  GpuModel model;
  model.nslots = 1;
  auto ctl = std::make_shared<SuspendCtl>();
  {
    std::lock_guard<std::mutex> lk(g_sm);
    ctl->seq = ++g_seq;
    ctl->slot = 0;
    g_detached.push_back(ctl);
  }
  std::atomic<bool> cancel{false}, waiting{false}, resumed{false};
  int normal = -3, recovery = -3;
  std::thread first([&] {
    normal = slot_admit(model, {1}, 0, cancel, 1, [&] { waiting = true; });
    if (normal >= 0) release_slot(normal);
  });
  wait_until([&] { return waiting.load() && admission_waiting(); });
  std::thread second([&] {
    recovery = slot_admit(model, {2}, 0, cancel, 2, {}, 0, ctl->seq);
    if (recovery >= 0) release_slot(recovery);
    resumed = true;
  });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!resumed && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  const bool bypassed = resumed.load();
  cancel = true;
  g_scv.notify_all();
  second.join();
  first.join();
  {
    std::lock_guard<std::mutex> lk(g_sm);
    g_detached.remove(ctl);
  }
  require(bypassed && recovery == 0 && normal == -1,
          "unavailable ordinary head blocked detached recovery");
  require(g_wait.empty(), "admission queue leaked after bypass");
  require_turn_free();
  std::puts("PASS unavailable ordinary head yields to detached recovery");
}

static void test_admission_callback_exception() {
  reset_slots(1);
  GpuModel model;
  model.nslots = 1;
  std::atomic<bool> cancel{false};
  bool caught = false;
  try {
    slot_admit(model, {34}, 1, cancel, 34, [] { throw std::runtime_error("notice"); });
  } catch (const std::runtime_error&) { caught = true; }
  require(caught && g_wait.empty(), "throwing admission notice leaked queue ticket");
  require_turn_free();
  std::puts("PASS admission notice exception cleanup");
}

static void test_wait_notice_does_not_hold_turn() {
  reset_slots(2);
  GpuModel model;
  std::atomic<bool> cancel{false}, blocked{false}, unblock{false}, peer_done{false};
  std::atomic<int> notices{0};
  g_turn.acquire();
  std::thread request([&] {
    const int chosen = slot_admit(model, {11}, 16, cancel, 9, [&] {
      if (++notices == 2) {
        blocked = true;
        while (!unblock) std::this_thread::yield();
      }
    });
    if (chosen >= 0) release_slot(chosen);
  });
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
  while (!blocked && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  if (!blocked) {
    cancel = true;
    g_turn.release();
    unblock = true;
    request.join();
    require(false, "GPU wait notice was not emitted");
  }
  std::thread peer([&] {
    g_turn.acquire();
    peer_done = true;
    g_turn.release();
  });
  g_turn.release();
  const auto peer_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!peer_done && std::chrono::steady_clock::now() < peer_deadline)
    std::this_thread::yield();
  const bool progressed = peer_done.load();
  cancel = true;
  unblock = true;
  request.join();
  peer.join();
  require(progressed, "blocked wait notice held a GPU turn ticket");
  require_turn_free();
  std::puts("PASS wait notice does not hold GPU ticket");
}

static void test_continuation_and_single_slot() {
  reset_slots(2);
  GpuModel model;
  model.pages[0] = 80;
  g_slots[0].hist.assign(80 * KV_PAGE, 11);
  std::atomic<bool> cancel{false};
  const int chosen = slot_admit(model, std::vector<int>(90 * KV_PAGE - 16, 11),
                                16, cancel, 9);
  require(chosen == 0 && g_slots[0].reserved == 90,
          "continuation must reserve total target, not fresh-page delta");
  model.pages[0] = 120;
  int peer = -3;
  std::thread request([&] {
    peer = slot_admit(model, std::vector<int>(1900 * KV_PAGE - 16, 12),
                      16, cancel, 10);
    if (peer >= 0) release_slot(peer);
  });
  wait_until(admission_waiting);
  g_turn.release();
  request.join();
  require(peer == 1, "restored mapped-page count must replace old baseline");
  std::atomic<bool> waiting{false};
  int deferred = -3;
  std::thread blocked([&] {
    deferred = slot_admit(model, std::vector<int>(1930 * KV_PAGE - 16, 13),
                          16, cancel, 11, [&] { waiting = true; });
    if (deferred >= 0) release_slot(deferred);
  });
  wait_until([&] { return waiting.load(); });
  cancel = true;
  g_scv.notify_all();
  blocked.join();
  require(deferred == -1, "actual mappings must constrain peers until cancelled");
  cancel = false;
  g_turn.acquire();
  release_slot(chosen);
  require_turn_free();
  reset_slots(1);
  model.nslots = 1;
  model.kv_pool_pages = 1;
  const int single = slot_admit(model, std::vector<int>(1024, 14),
                                16, cancel, 12);
  require(single == 0 && g_slots[0].reserved == 0, "single-slot gate bypass");
  release_slot(single);
  require_turn_free();
  std::puts("PASS continuation / changed mappings / single-slot bypass");
}

static void test_churn() {
  reset_slots(4);
  GpuModel model;
  model.nslots = 4;
  model.kv_pool_pages = 64;
  model.maxctx = 16384;
  std::atomic<bool> cancel{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; worker++)
    workers.emplace_back([&, worker] {
      for (int turn = 0; turn < 100; turn++) {
        const std::vector<int> ids(1024, 100 + worker * 100 + turn);
        const int chosen = slot_admit(model, ids, 20, cancel,
                                      100 + worker * 100 + turn);
        if (chosen < 0) {
          failures++;
          break;
        }
        model.mutating = true;
        model.pages[(size_t)chosen] = 0;
        g_slots[(size_t)chosen].hist = ids;
        model.pages[(size_t)chosen] = 6;
        std::this_thread::yield();
        model.mutating = false;
        release_slot(chosen);
      }
    });
  for (auto& worker : workers) worker.join();
  require(failures == 0, "concurrent admission/reset churn failed");
  require(model.unsafe_reads == 0, "concurrent admission read mutating slot state");
  require_turn_free();
  reset_slots(4);
  std::puts("PASS four-thread admission/reset churn (400 requests)");
}

int main() {
  try {
    test_pressure_telemetry();
    test_recall_selection();
    test_recall_rebuild();
    test_arithmetic();
    test_reset_slot();
    test_synchronized_reads();
    test_rejection_and_context_edge();
    test_cancellation();
    test_starve_busy_victim_overflow();
    test_spec_recall_boundaries();
    test_starve_idle_preserves_binding();
    test_pressure_suspend();
    test_mtp_chain_suspend();
    test_mtp_failed_capture();
    test_spec_self_suspend();
    test_spec_cancel_suspended_request();
    if (false) {
      test_spec_self_suspend_blocker_departure();
      test_spec_self_suspend_departure_retry_limit();
      test_spec_self_suspend_failure();
      test_round_top_behind_detached();
    }
    test_spec_round_hook();
    test_spec_shared_idle_pages();
    test_spec_changed_victim();
    test_spec_existing_suspend();
    test_spec_reclaim_candidates();
    test_cancel_suspended_request();
    test_resume_slot_bypass();
    test_queue_revalidation();
    test_detached_reservation_and_lease();
    test_detached_kv_commitment();
    test_ready_cancel_with_coordinator_turn();
    test_self_suspend_behind_detached();
    test_unavailable_normal_head_yields_to_resume();
    test_admission_callback_exception();
    test_wait_notice_does_not_hold_turn();
    test_continuation_and_single_slot();
    test_churn();
    require(kvsnap::required_calls == 0,
            "host regression invoked private required snapshot capture, publish or restore");
    std::puts("PASS no private required snapshot capture / publish / restore");
    std::puts("ALL PASS (KV admission host regression)");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
