#include "../src/kv_admission.h"
#include "../src/vision_memory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

static constexpr int KV_PAGE = 256;
static const char* log_ts() { return "test:"; }
static bool g_turn_prof = false;
static thread_local int tl_turn_tag = 0;
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
  std::atomic<bool> mutating{false};
  mutable std::atomic<int> unsafe_reads{0};
  int slot_pages(int slot) const {
    if (mutating.load()) unsafe_reads++;
    return pages[(size_t)slot];
  }
  size_t test_vision_need = 0;
  mutable std::atomic<int> vision_checks{0};
  size_t vision_memory_need(int, const VisionMemDemand&) const {
    vision_checks++;
    return test_vision_need;
  }
};

struct EngineMemStats {
  uint64_t hip_free = 0, hip_total = 0;
};
static std::atomic<uint64_t> g_test_hip_free{0}, g_test_hip_total{0};
static EngineMemStats memstats_snapshot() {
  return {g_test_hip_free.load(), g_test_hip_total.load()};
}

namespace rckpt {
static bool enabled() { return false; }
static size_t lcp(const std::vector<int>&) { return 0; }
}
namespace kvsnap {
static uint64_t g_fp = 0;
static size_t shared_tokens(const std::vector<int>&, uint64_t) { return 0; }
}

#include "kv_admission_serve.inc"

static void borrow_abort_check() {}
static void borrow_turn(GpuModel&) {}

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
  g_slots.assign((size_t)count, Slot{});
  g_reserve_decode = 4096;
  g_conn_id = 0;
  tl_slot = -1;
  g_test_hip_free = g_test_hip_total = (uint64_t)1 << 40;
}

static void require_turn_free() {
  std::lock_guard<std::mutex> lock(g_turn.mu);
  require(g_turn.next == g_turn.serving, "GPU turn leaked or released twice");
}

static void release_slot(int slot) {
  slot_leave(slot);
  g_turn.release();
}

static void test_arithmetic() {
  (void)&borrow_abort_check;
  (void)&borrow_turn;
  require(lend_light_only() == !GpuModel::env_zero("QWENOX_CONC_LEND_LIGHT_ONLY"),
          "light-only lending configuration");
  require(prefill_srpt() == !GpuModel::env_zero("QWENOX_PREFILL_SRPT"),
          "prefill SRPT configuration");
  const size_t expected_fork = GpuModel::env_zero("QWENOX_FORK_CKPT") ? 0 : 1984;
  require(fork_cut(std::vector<int>(4096, 3), 2048, 0, 0, 1) == expected_fork,
          "fork cut checkpoint selection");
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
          "unbounded generation must use decode cap");
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
  const VisionMemDemand image{14400, 3600};
  require(vision_workspace_bytes(14400) == 1092556800,
          "ViT workspace byte formula");
  const size_t fresh_embedding = 4096 * kVisionEmbeddingRowBytes;
  require(vision_memory_budget(0, 0, image) ==
              fresh_embedding + vision_workspace_bytes(14400) + kVisionMemorySafetyBytes,
          "fresh vision request budget");
  require(vision_memory_budget(14400, 4096, image) == 0,
          "vision high-water marks charged twice");
  const VisionMemDemand embedding_only{14400, 5000};
  require(vision_memory_budget(14400, 4096, embedding_only) ==
              5000 * kVisionEmbeddingRowBytes + kVisionMemorySafetyBytes,
          "embedding old+new allocation peak");
  const VisionMemDemand huge{1, std::numeric_limits<size_t>::max()};
  require(vision_memory_budget(0, 0, huge) == std::numeric_limits<size_t>::max(),
          "vision budget overflow not saturated");
  std::puts("PASS admission arithmetic / decode cap / context edge / vision budget");
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
  require(slot_admit(model, std::vector<int>(1000 * KV_PAGE - 16, 6),
                     16, cancel, 4) == -2, "over-subscription not rejected");
  require_turn_free();
  const int chosen = slot_admit(model, std::vector<int>(840 * KV_PAGE - 16, 7),
                                16, cancel, 5);
  require(chosen == 1, "rejection left the admission queue blocked");
  release_slot(chosen);
  slot_leave(0);
  reset_slots(2);
  const int edge = slot_admit(model, std::vector<int>(524287, 8),
                              100, cancel, 6);
  require(edge == 0 && g_slots[0].reserved == 2048,
          "parallel one-token context edge was rejected");
  release_slot(edge);
  require_turn_free();
  std::puts("PASS rejection releases turn / parallel full-pool budget");
}

static void test_cancellation() {
  reset_slots(2);
  GpuModel model;
  std::atomic<bool> cancel{false};
  g_turn.acquire();
  int result = -3;
  std::thread request([&] {
    result = slot_admit(model, {9}, 16, cancel, 7);
    if (result >= 0) release_slot(result);
  });
  wait_until(admission_waiting);
  cancel = true;
  g_turn.release();
  request.join();
  require(result == -1, "cancelled GPU-queued admission was not cancelled");
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

static void test_vision_memory_queue() {
  const VisionMemDemand image{1024, 256};
  reset_slots(2);
  GpuModel model;
  model.test_vision_need = 100;
  g_test_hip_total = 1000;
  g_test_hip_free = 50;
  std::atomic<bool> cancel{false};
  require(slot_admit(model, {20}, 16, cancel, 13, image) == -3,
          "permanent vision memory shortage did not fail");
  require_turn_free();
  require(!admission_waiting(), "failed vision admission left a FIFO ticket");

  reset_slots(2);
  g_slots[0].busy = true;
  model.test_vision_need = 100;
  model.vision_checks = 0;
  g_test_hip_free = 50;
  int result = -3;
  std::thread queued([&] {
    result = slot_admit(model, {21}, 16, cancel, 14, image);
    if (result >= 0) release_slot(result);
  });
  wait_until([&] { return model.vision_checks.load() >= 1; });
  // Acquiring the turn here proves the memory waiter released it. Hold the
  // turn until memory becomes available so resume cannot race this assertion.
  g_turn.acquire();
  g_test_hip_free = 200;
  g_turn.release();
  queued.join();
  require(result == 1, "vision request did not resume when memory became available");
  require_turn_free();
  slot_leave(0);

  reset_slots(2);
  g_slots[0].busy = true;
  model.test_vision_need = 100;
  model.vision_checks = 0;
  g_test_hip_free = 50;
  cancel = false;
  result = -3;
  std::thread cancelled([&] {
    result = slot_admit(model, {22}, 16, cancel, 15, image);
    if (result >= 0) release_slot(result);
  });
  wait_until([&] { return model.vision_checks.load() >= 1; });
  cancel = true;
  g_scv.notify_all();
  cancelled.join();
  require(result == -1, "cancelled vision memory wait was not cancelled");
  require_turn_free();
  require(!admission_waiting(), "cancelled vision wait left a FIFO ticket");
  slot_leave(0);
  reset_slots(2);
  std::puts("PASS vision memory queue / resume / cancellation / permanent failure");
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
  require(slot_admit(model, std::vector<int>(1930 * KV_PAGE - 16, 13),
                     16, cancel, 11) == -2,
          "actual mappings above target must still constrain peers");
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
    test_arithmetic();
    test_reset_slot();
    test_synchronized_reads();
    test_rejection_and_context_edge();
    test_cancellation();
    test_vision_memory_queue();
    test_continuation_and_single_slot();
    test_churn();
    std::puts("ALL PASS (KV admission host regression)");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
