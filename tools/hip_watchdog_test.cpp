#include "../src/hip_watchdog.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>

// Compile the production copy/synchronization wrappers against a deliberately
// blocking fake HIP runtime. No model, GPU driver or heavy load is required.
using hipError_t = int;
using hipStream_t = void*;
using hipEvent_t = void*;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice };
static std::atomic<bool> stuck{false};
static int fake_call() {
  while (stuck.load()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
  return 0;
}
static int hipMemcpy(void*, const void*, size_t, hipMemcpyKind) { return fake_call(); }
static int hipMemcpyAsync(void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return fake_call(); }
static int hipStreamSynchronize(hipStream_t) { return fake_call(); }
static int hipDeviceSynchronize() { return fake_call(); }
static int hipEventSynchronize(hipEvent_t) { return fake_call(); }
#include "../src/gpu/parts/01_syncprof.inc"

static std::atomic<int> reports{0};
static const char* expected = nullptr;
static void timeout_report(const hipwatch::Call& call) {
  if (!call.file || !call.kind || call.line <= 0 ||
      (expected && strcmp(expected, call.kind))) std::_Exit(3);
  reports++;
  if (expected) {
    fprintf(stderr, "detected %s at %s:%d\n", call.kind, call.file, call.line);
    std::_Exit(124);
  }
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  const std::string mode = argv[1];
  if (mode == "sync") expected = "stream sync";
  else if (mode == "device") expected = "device sync";
  else if (mode == "event") expected = "event sync";
  else if (mode == "copy") expected = "memcpy D2H";
  else if (mode == "async") expected = "async D2H";
  else if (mode == "nested") expected = "outer";
  hipwatch::instance().start(std::chrono::milliseconds(mode == "disabled" ? 0 : 200), timeout_report);
  if (expected) {
    stuck = true;
    // Activity on other threads must not reset the blocked thread's deadline.
    std::thread([] {
      for (;;) {
        { hipwatch::Scope s(__FILE__, __LINE__, "healthy"); }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }).detach();
    if (mode == "sync") hipStreamSynchronize(nullptr);
    if (mode == "device") hipDeviceSynchronize();
    if (mode == "event") hipEventSynchronize(nullptr);
    if (mode == "copy") hipMemcpy(nullptr, nullptr, 0, hipMemcpyDeviceToHost);
    if (mode == "async") hipMemcpyAsync(nullptr, nullptr, 0, hipMemcpyDeviceToHost, nullptr);
    if (mode == "nested") {
      hipwatch::Scope outer(__FILE__, __LINE__, "outer");
      for (;;) {
        hipwatch::Scope inner(__FILE__, __LINE__, "inner");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
    return 4;
  }
  if (mode == "disabled") {
    hipwatch::Scope s(__FILE__, __LINE__, "disabled");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } else if (mode == "healthy") {
    for (int i = 0; i < 100; ++i) {
      std::thread([] {
        hipStreamSynchronize(nullptr);
        hipDeviceSynchronize();
        hipEventSynchronize(nullptr);
        hipMemcpy(nullptr, nullptr, 0, hipMemcpyHostToDevice);
        hipMemcpyAsync(nullptr, nullptr, 0, hipMemcpyDeviceToDevice, nullptr);
        try { hipwatch::Scope s(__FILE__, __LINE__, "exception"); throw 1; }
        catch (int) {}
      }).join();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } else if (mode == "idle") {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  } else return 2;
  return reports.load() ? 5 : 0;
}
