// ---- QWENOX_SYNC_PROF=1: host-blocking HIP call accounting (T1, diagnostic) ----
// Every hipMemcpy / hipMemcpyAsync / hip*Synchronize in
// the parts included after this file goes through a wrapper that, when the
// switch is on, times the call and accumulates (count, ms) per call site
// (file:line + kind). A call that returns slowly is the host waiting for the
// GPU (or the driver); summed against a round's wall time this splits the
// fixed per-round cost into "host busy" and "host blocked" parts.
// Profiling off: no profiling clock reads. The independent runtime watchdog
// still tracks these calls when QWENOX_HIP_TIMEOUT is enabled.
// Note: D2H hipMemcpyAsync into pageable memory is synchronous; it shows up
// here with the stream-drain time, which is exactly what we want to see.
static const bool g_sp_on = [] {
  const char* e = getenv("QWENOX_SYNC_PROF");
  return e && *e && strcmp(e, "0") != 0;
}();

enum SpKind {
  SP_MEMCPY_H2D, SP_MEMCPY_D2H, SP_MEMCPY_D2D, SP_ASYNC_H2D, SP_ASYNC_D2H,
  SP_ASYNC_D2D, SP_STREAM_SYNC, SP_DEVICE_SYNC, SP_EVENT_SYNC, SP_NKIND
};
static const char* const g_sp_kind_name[SP_NKIND] = {
    "memcpy H2D", "memcpy D2H", "memcpy D2D", "async H2D", "async D2H",
    "async D2D",  "stream sync", "device sync", "event sync"};

struct SpSite {
  const char* file;
  int line, kind;
  long n = 0;
  double ms = 0, max_ms = 0;
};
struct SyncProf {
  std::mutex mu;
  std::map<std::tuple<const char*, int, int>, SpSite> sites;
  std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
  void add(const char* file, int line, int kind, double ms) {
    std::lock_guard<std::mutex> lk(mu);
    auto& s = sites[std::make_tuple(file, line, kind)];
    s.file = file, s.line = line, s.kind = kind;
    s.n++;
    s.ms += ms;
    s.max_ms = std::max(s.max_ms, ms);
  }
  // Prints the top call sites since the previous report and resets. rounds =
  // spec rounds the caller ran in that window (0: totals only).
  void report(const char* tag, long rounds) {
    std::vector<SpSite> v;
    double span;
    {
      std::lock_guard<std::mutex> lk(mu);
      for (auto& kv : sites) v.push_back(kv.second);
      sites.clear();
      const auto now = std::chrono::steady_clock::now();
      span = std::chrono::duration<double, std::milli>(now - since).count();
      since = now;
    }
    long n = 0;
    double ms = 0;
    double kms[SP_NKIND] = {};
    long kn[SP_NKIND] = {};
    for (auto& s : v) n += s.n, ms += s.ms, kms[s.kind] += s.ms, kn[s.kind] += s.n;
    const double r = rounds > 0 ? (double)rounds : 1.0;
    fprintf(stderr, "[syncprof] %s: %.0f ms window, %ld rounds | %ld calls (%.1f/round), "
                    "blocked %.0f ms (%.1f%%, %.2f ms/round)\n",
            tag, span, rounds, n, n / r, ms, span > 0 ? 100 * ms / span : 0.0, ms / r);
    fprintf(stderr, "[syncprof]   by kind:");
    for (int k = 0; k < SP_NKIND; k++)
      if (kn[k]) fprintf(stderr, " %s %ld/%.1fms", g_sp_kind_name[k], kn[k], kms[k]);
    fprintf(stderr, "\n");
    std::sort(v.begin(), v.end(), [](const SpSite& a, const SpSite& b) { return a.ms > b.ms; });
    for (size_t i = 0; i < v.size() && i < 16; i++) {
      const char* f = v[i].file;
      for (const char* c = f; *c; c++)
        if (*c == '/' || *c == '\\') f = c + 1;
      fprintf(stderr, "[syncprof]   %-22s:%-5d %-11s n=%-6ld (%.2f/round) %8.1f ms "
                      "(%.3f ms/call, max %.2f)\n",
              f, v[i].line, g_sp_kind_name[v[i].kind], v[i].n, v[i].n / r, v[i].ms,
              v[i].ms / v[i].n, v[i].max_ms);
    }
  }
};
static SyncProf g_sp;

struct SpTimer {
  hipwatch::Scope watchdog;
  const char* file;
  int line, kind;
  std::chrono::steady_clock::time_point t0;
  SpTimer(const char* f, int l, int k)
      : watchdog(f, l, g_sp_kind_name[k]), file(f), line(l), kind(k) {
    if (g_sp_on) t0 = std::chrono::steady_clock::now();
  }
  ~SpTimer() {
    if (g_sp_on)
      g_sp.add(file, line, kind,
               std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0).count());
  }
};
static inline int sp_kind(hipMemcpyKind k, bool async) {
  switch (k) {
    case hipMemcpyHostToDevice: return async ? SP_ASYNC_H2D : SP_MEMCPY_H2D;
    case hipMemcpyDeviceToHost: return async ? SP_ASYNC_D2H : SP_MEMCPY_D2H;
    default: return async ? SP_ASYNC_D2D : SP_MEMCPY_D2D;
  }
}
static inline hipError_t sp_hipMemcpy(const char* f, int l, void* dst, const void* src,
                                      size_t n, hipMemcpyKind k) {
  SpTimer t(f, l, sp_kind(k, false));
  return hipMemcpy(dst, src, n, k);
}
static inline hipError_t sp_hipMemcpyAsync(const char* f, int l, void* dst, const void* src,
                                           size_t n, hipMemcpyKind k, hipStream_t s = nullptr) {
  SpTimer t(f, l, sp_kind(k, true));
  return hipMemcpyAsync(dst, src, n, k, s);
}
static inline hipError_t sp_hipStreamSynchronize(const char* f, int l, hipStream_t s) {
  SpTimer t(f, l, SP_STREAM_SYNC);
  return hipStreamSynchronize(s);
}
static inline hipError_t sp_hipDeviceSynchronize(const char* f, int l) {
  SpTimer t(f, l, SP_DEVICE_SYNC);
  return hipDeviceSynchronize();
}
static inline hipError_t sp_hipEventSynchronize(const char* f, int l, hipEvent_t e) {
  SpTimer t(f, l, SP_EVENT_SYNC);
  return hipEventSynchronize(e);
}
#define hipMemcpy(...) sp_hipMemcpy(__FILE__, __LINE__, __VA_ARGS__)
#define hipMemcpyAsync(...) sp_hipMemcpyAsync(__FILE__, __LINE__, __VA_ARGS__)
#define hipStreamSynchronize(...) sp_hipStreamSynchronize(__FILE__, __LINE__, __VA_ARGS__)
#define hipDeviceSynchronize() sp_hipDeviceSynchronize(__FILE__, __LINE__)
#define hipEventSynchronize(...) sp_hipEventSynchronize(__FILE__, __LINE__, __VA_ARGS__)
// hipMalloc / hipFree are already macro-wrapped by 06_memstats.inc (not timed).
