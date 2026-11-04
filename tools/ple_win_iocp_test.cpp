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

static std::atomic<int> g_read_calls{0}, g_completion_calls{0};
static std::atomic<int> g_fail_read{-1}, g_fail_completion{-1};
static BOOL WINAPI hook_ReadFile(HANDLE file, LPVOID buffer, DWORD bytes,
                                 LPDWORD read, LPOVERLAPPED ov) {
  const int call = g_read_calls.fetch_add(1);
  if (call == g_fail_read.load()) {
    SetLastError(ERROR_ACCESS_DENIED);
    return FALSE;
  }
  return ::ReadFile(file, buffer, bytes, read, ov);
}
static BOOL WINAPI hook_GetQueuedCompletionStatus(HANDLE port, LPDWORD got,
                                                   PULONG_PTR key, LPOVERLAPPED* ov,
                                                   DWORD timeout) {
  const BOOL ok = ::GetQueuedCompletionStatus(port, got, key, ov, timeout);
  const int call = g_completion_calls.fetch_add(1);
  if (call == g_fail_completion.load() && ok && *ov) {
    SetLastError(ERROR_IO_DEVICE);
    return FALSE;
  }
  return ok;
}

static uint64_t g_ple_foff = 4093;
static DWORD g_ple_rb = 90;
static bool g_ple_off_ok = true;
#define ReadFile hook_ReadFile
#define GetQueuedCompletionStatus hook_GetQueuedCompletionStatus
#include "ple_win_under_test.inc"
#undef ReadFile
#undef GetQueuedCompletionStatus

static constexpr size_t ROWS = 8193, GUARD = 256;
static std::vector<uint8_t> g_table;
static uint64_t table_va() { return (uint64_t)(uintptr_t)g_table.data(); }
static void injection_reset(int read = -1, int completion = -1) {
  g_read_calls = g_completion_calls = 0;
  g_fail_read = read;
  g_fail_completion = completion;
}
static void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
static uint8_t value(size_t row, size_t col) {
  return uint8_t((row * 73 + col * 19 + (row >> 5)) & 255);
}
static void fixture(const char* path, bool short_file = false) {
  g_table.resize(ROWS * g_ple_rb);
  for (size_t i = 0; i < g_table.size(); ++i)
    g_table[i] = value(i / g_ple_rb, i % g_ple_rb);
  const size_t body = short_file ? g_ple_rb / 2 : g_table.size();
  std::vector<uint8_t> bytes(g_ple_foff + body, 0x37);
  memcpy(bytes.data() + g_ple_foff, g_table.data(), body);
  FILE* f = fopen(path, "wb");
  require(f != nullptr, "open fixture");
  require(fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size(), "write fixture");
  require(fclose(f) == 0, "close fixture");
}
struct Batch {
  std::vector<uint64_t> addrs;
  std::vector<uint8_t> bytes;
  Batch(size_t n, unsigned seed) : addrs(n), bytes(n * g_ple_rb + 2 * GUARD, 0xa5) {
    std::mt19937 rng(seed);
    for (auto& a : addrs) a = table_va() + (rng() % ROWS) * g_ple_rb;
  }
  uint8_t* data() { return bytes.data() + GUARD; }
  void check() {
    for (size_t i = 0; i < GUARD; ++i)
      require(bytes[i] == 0xa5 && bytes[bytes.size() - 1 - i] == 0xa5, "buffer guard changed");
    for (size_t i = 0; i < addrs.size(); ++i)
      for (size_t j = 0; j < g_ple_rb; ++j)
        require(data()[i * g_ple_rb + j] ==
                    value((addrs[i] - table_va()) / g_ple_rb, j),
                "row mismatch");
  }
  void run(PleWin& r, bool split = false) {
    if (split) {
      r.issue(addrs.data(), data(), table_va(), addrs.size());
      r.wait_all();
    } else {
      r.run(addrs.data(), data(), table_va(), addrs.size());
    }
    check();
  }
};
static void positive(const char* path) {
  DWORD before = 0, after = 0;
  require(GetProcessHandleCount(GetCurrentProcess(), &before), "handle count");
  for (unsigned width : {90u, 160u, 320u}) {
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
          PleWin independent;
          require(independent.setup(path), "independent setup");
          Batch(4097, i + 100).run(independent);
        }));
      for (auto& job : jobs) job.get();
      Batch pending(512, 918);
      reader.issue(pending.addrs.data(), pending.data(), table_va(), pending.addrs.size());
      std::atomic<bool> started{false};
      auto blocked = std::async(std::launch::async, [&] { started = true; Batch(16, 722).run(reader); });
      while (!started.load()) std::this_thread::yield();
      require(blocked.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
              "run did not wait for split batch");
      PleWin independent;
      require(independent.setup(path), "split independent setup");
      Batch(16, 713).run(independent);
      reader.wait_all();
      pending.check();
      blocked.get();
    }
    for (unsigned i = 0; i < 30; ++i) {
      PleWin reader;
      require(reader.setup(path), "repeated setup");
      Batch(16, i).run(reader);
    }
  }
  require(GetProcessHandleCount(GetCurrentProcess(), &after), "final handle count");
  require(after <= before, "reader leaked handles");
  printf("PASS: 90/160/320-byte rows; depth boundaries; split; shared/8-pump concurrency; locks; handles\n");
}
static void recovered(const char* path, const std::string& mode) {
  const bool split = mode.find("split") != std::string::npos;
  const bool short_file = mode.find("short") != std::string::npos;
  const bool submit = mode.find("submit") != std::string::npos;
  const bool completion = mode.find("completion") != std::string::npos;
  fixture(path, short_file);
  PleWin reader;
  require(reader.setup(path) && reader.ok(), "recovery setup");
  Batch batch(short_file ? 1 : split ? 16 : 1025, 77);
  if (short_file) batch.addrs[0] = table_va();
  const int read = submit ? (mode.find("first") != std::string::npos ? 0 : 3) : -1;
  injection_reset(read, completion ? 0 : -1);
  batch.run(reader, split);
  if (short_file) require(g_completion_calls.load() == 1, "short read bypassed IOCP completion");
  injection_reset();
  Batch(16, 99).run(reader, split);
  printf("PASS: recovered %s\n", mode.c_str());
}
int main(int argc, char** argv) {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
  try {
    require(argc == 3, "usage: test MODE FIXTURE");
    const std::string mode = argv[1];
    const char* path = argv[2];
    if (mode == "positive") {
      positive(path);
      return 0;
    }
    if (mode.find("short") != std::string::npos ||
        mode.find("submit") != std::string::npos ||
        mode.find("completion") != std::string::npos) {
      recovered(path, mode);
      return 0;
    }
    fixture(path);
    PleWin reader;
    require(reader.setup(path), "negative setup");
    Batch batch(1, 0);
    batch.addrs[0] = table_va();
    OVERLAPPED foreign{};
    OVERLAPPED* p = &foreign;
    if (mode.find("null") != std::string::npos) p = nullptr;
    if (mode.find("unaligned") != std::string::npos)
      p = reinterpret_cast<OVERLAPPED*>(reinterpret_cast<uintptr_t>(&foreign) + 1);
    require(PostQueuedCompletionStatus(reader.iocp, g_ple_rb, 0, p), "post packet");
    if (mode.find("split") != std::string::npos) {
      reader.issue(batch.addrs.data(), batch.data(), table_va(), 1);
      reader.wait_all();
    } else {
      reader.run(batch.addrs.data(), batch.data(), table_va(), 1);
    }
    fprintf(stderr, "unexpectedly accepted invalid completion\n");
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
