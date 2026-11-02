// Host-only tests of the actual PleWin definition extracted by the Python driver.
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <crtdbg.h>

static uint64_t g_ple_foff = 4093;
static DWORD g_ple_rb = 90;
static bool g_ple_off_ok = true;
#include "ple_win_under_test.inc"

static constexpr size_t ROWS = 8193, GUARD = 256;
static constexpr uint64_t TABLE = 0x10000000;
static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
static uint8_t value(size_t row, size_t col) {
  return uint8_t((row * 73 + col * 19 + (row >> 5)) & 255);
}
static void fixture(const char* path, bool short_file = false) {
  FILE* f = fopen(path, "wb");
  require(f != nullptr, "open fixture");
  std::vector<uint8_t> bytes(g_ple_foff + (short_file ? g_ple_rb / 2 : ROWS * g_ple_rb), 0x37);
  for (size_t i = g_ple_foff; i < bytes.size(); ++i) {
    size_t off = i - g_ple_foff;
    bytes[i] = value(off / g_ple_rb, off % g_ple_rb);
  }
  require(fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size(), "write fixture");
  require(fclose(f) == 0, "close fixture");
}
struct Batch {
  std::vector<uint64_t> addrs;
  std::vector<uint8_t> bytes;
  Batch(size_t n, unsigned seed) : addrs(n), bytes(n * g_ple_rb + 2 * GUARD, 0xa5) {
    std::mt19937 rng(seed);
    for (auto& a : addrs) a = TABLE + (rng() % ROWS) * g_ple_rb;
  }
  uint8_t* data() { return bytes.data() + GUARD; }
  void check() {
    for (size_t i = 0; i < GUARD; ++i)
      require(bytes[i] == 0xa5 && bytes[bytes.size() - 1 - i] == 0xa5, "buffer guard changed");
    for (size_t i = 0; i < addrs.size(); ++i)
      for (size_t j = 0; j < g_ple_rb; ++j)
        require(data()[i * g_ple_rb + j] == value((addrs[i] - TABLE) / g_ple_rb, j), "row mismatch");
  }
  void run(PleWin& r, bool split = false) {
    if (split) { r.issue(addrs.data(), data(), TABLE, addrs.size()); r.wait_all(); }
    else r.run(addrs.data(), data(), TABLE, addrs.size());
    check();
  }
};
static void positive(const char* path) {
  DWORD before = 0, after = 0;
  require(GetProcessHandleCount(GetCurrentProcess(), &before), "handle count");
  for (unsigned width : {90u, 160u}) {
    g_ple_rb = width;
    fixture(path);
    {
      PleWin reader;
      require(reader.setup(path) && reader.ok(), "reader setup");
      for (size_t n : {0u, 1u, 16u, 511u, 512u, 513u, 1025u, 4096u, 4097u, 8193u})
        Batch(n, unsigned(n)).run(reader);
      for (size_t n : {0u, 1u, 16u, 511u, 512u}) Batch(n, unsigned(n + 19)).run(reader, true);
      std::vector<std::future<void>> jobs;
      for (unsigned i = 0; i < 8; ++i)
        jobs.push_back(std::async(std::launch::async, [&reader, i] { Batch(1025, i).run(reader); }));
      for (auto& job : jobs) job.get();
      jobs.clear();
      for (unsigned i = 0; i < 8; ++i)
        jobs.push_back(std::async(std::launch::async, [path, i] {
          PleWin independent; require(independent.setup(path), "independent setup");
          Batch(4097, i + 100).run(independent);
        }));
      for (auto& job : jobs) job.get();
      Batch pending(512, 918);
      reader.issue(pending.addrs.data(), pending.data(), TABLE, pending.addrs.size());
      std::atomic<bool> started{false};
      auto blocked = std::async(std::launch::async, [&] { started = true; Batch(16, 722).run(reader); });
      while (!started.load()) std::this_thread::yield();
      require(blocked.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
              "run did not wait for split batch");
      PleWin independent; require(independent.setup(path), "split independent setup");
      Batch(16, 713).run(independent);
      reader.wait_all(); pending.check(); blocked.get();
    }
    for (unsigned i = 0; i < 30; ++i) {
      PleWin reader; require(reader.setup(path), "repeated setup"); Batch(16, i).run(reader);
    }
  }
  require(GetProcessHandleCount(GetCurrentProcess(), &after), "final handle count");
  require(after <= before, "reader leaked handles");
  printf("PASS: 90/160-byte rows; depth boundaries; split; shared/8-pump concurrency; locks; handles\n");
}
int main(int argc, char** argv) {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
  try {
    require(argc == 3, "usage: test MODE FIXTURE");
    std::string mode = argv[1]; const char* path = argv[2];
    if (mode == "positive") { positive(path); return 0; }
    bool split = mode.find("split") != std::string::npos;
    bool short_file = mode.find("short") != std::string::npos;
    fixture(path, short_file);
    PleWin reader; require(reader.setup(path), "negative setup");
    Batch batch(1, 0); batch.addrs[0] = TABLE;
    OVERLAPPED foreign{};
    if (!short_file) {
      OVERLAPPED* p = &foreign;
      if (mode.find("null") != std::string::npos) p = nullptr;
      if (mode.find("unaligned") != std::string::npos)
        p = reinterpret_cast<OVERLAPPED*>(reinterpret_cast<uintptr_t>(&foreign) + 1);
      require(PostQueuedCompletionStatus(reader.iocp, g_ple_rb, 0, p), "post packet");
    }
    if (split) { reader.issue(batch.addrs.data(), batch.data(), TABLE, 1); reader.wait_all(); }
    else reader.run(batch.addrs.data(), batch.data(), TABLE, 1);
    // An injected packet must never be accepted as a completed file read.
    fprintf(stderr, "unexpectedly accepted invalid completion\n");
    return 0;
  } catch (const std::exception& e) { fprintf(stderr, "%s\n", e.what()); return 1; }
}
