// The Python driver extracts prefill_batch from production, without HIP.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <vector>

#define CK(x) (void)(x)
static int g_str;
static int hipStreamSynchronize(int) { return 0; }
namespace ple_prefetch {
static int chunk_size(int remaining, int batch, int cap) {
  return std::min(remaining, std::min(batch, cap));
}
}
static std::atomic<bool> g_borrow_light{false};
struct ReqCancelled : std::runtime_error { using std::runtime_error::runtime_error; };
struct Model {
  int maxbatch = 2, maxbatch_cap = 2, conc_rmax_ = 8, pf_end = 0, heavy_rem_ = 0;
  int prog_total = 6, chunks = 0;
  bool prof = false, mtp_tap_capture = false;
  void* d_mtap = nullptr;
  struct { bool on = false; int cap_mode = 0; } df2;
  void (*borrow_skip_fn)(Model&) = nullptr;
  bool (*heavy_gate_fn)(Model&, int) = nullptr;
  std::function<void()> on_chunk, on_yield;
  // Present only in the old source. The test driver models gen_one's old
  // arm/disarm operations to reproduce another slot mutating the shared model.
#if !CALL_LOCAL_CANCEL
  std::atomic<bool>* abort_flag = nullptr;
#endif
  void yield_point() { if (on_yield) on_yield(); }
  int prefill_chunk(const std::vector<int>&, int, bool, bool,
                    const std::vector<int>*, const std::vector<int>*) {
    ++chunks; if (on_chunk) on_chunk(); return 42;
  }
  void df2_inject_chunk(int, int) {}
  void mtp_ingest_b(void*, const std::vector<int>&, int, int) {}
#include "prefill_batch_under_test.inc"
  int ingest(const std::atomic<bool>& flag) {
#if CALL_LOCAL_CANCEL
    return prefill_batch({1, 2, 3, 4, 5, 6}, 0, true, false, &flag);
#else
    abort_flag = const_cast<std::atomic<bool>*>(&flag);
    return prefill_batch({1, 2, 3, 4, 5, 6});
#endif
  }
};
static int failures = 0;
static void check(bool ok, const char* label) {
  printf("%s: %s\n", ok ? "PASS" : "FAIL", label);
  failures += !ok;
}
static bool cancelled(const std::function<void()>& f) {
  try { f(); return false; } catch (const ReqCancelled&) { return true; }
}
int main() {
  {
    Model m; std::atomic<bool> a{false};
    check(!cancelled([&] { m.ingest(a); }) && m.chunks == 3, "serial ingest");
  }
  {
    Model m; std::atomic<bool> a{true};
    check(cancelled([&] { m.ingest(a); }) && m.chunks == 0, "pre-cancel skips all chunks");
  }
  {
    Model m; std::atomic<bool> a{false}; m.on_chunk = [&] { a = true; };
    check(cancelled([&] { m.ingest(a); }) && m.chunks == 1, "cancel at completed chunk boundary");
  }
  {
    Model m; std::atomic<bool> a{false}; [[maybe_unused]] std::atomic<bool> b{true};
    m.on_yield = [&] {
#if !CALL_LOCAL_CANCEL
      m.abort_flag = &b;
#endif
    };
    check(!cancelled([&] { m.ingest(a); }) && m.chunks == 3,
          "cancelled peer cannot abort this request");
  }
  {
    Model m; std::atomic<bool> a{false};
    m.on_yield = [&] {
      a = true;
#if !CALL_LOCAL_CANCEL
      m.abort_flag = nullptr;
#endif
    };
    check(cancelled([&] { m.ingest(a); }) && m.chunks == 1,
          "peer disarm cannot hide this request's cancellation after yield");
  }
  {
    Model m; [[maybe_unused]] std::atomic<bool> peer{true};
#if !CALL_LOCAL_CANCEL
    m.abort_flag = &peer;
#endif
    check(!cancelled([&] { m.prefill_batch({1, 2, 3, 4}, 0, false, true); }) && m.chunks == 2,
          "default-null verifier cannot inherit peer cancellation");
  }
  {
    Model m; std::atomic<bool> a{false};
    m.on_yield = [&] { m.prog_total = 6; };
    check(!cancelled([&] { m.ingest(a); }) && m.chunks == 3, "uncancelled slot handoff");
  }
  return failures ? 1 : 0;
}
