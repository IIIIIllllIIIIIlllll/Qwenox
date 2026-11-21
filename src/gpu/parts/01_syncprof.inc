// ---- QWENOX_SYNC_PROF=1: host-blocking HIP call accounting (T1, diagnostic) ----
// Every hipMemcpy / hipMemcpyAsync / hip*Synchronize in
// the parts included after this file goes through a wrapper that, when the
// switch is on, times the call and accumulates (count, ms) per call site
// (file:line + kind). A call that returns slowly is the host waiting for the
// GPU (or the driver); summed against a round's wall time this splits the
// fixed per-round cost into "host busy" and "host blocked" parts.
// Off (default): one predictable branch per call, no clock reads.
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
  const char* file;
  int line, kind;
  std::chrono::steady_clock::time_point t0;
  SpTimer(const char* f, int l, int k) : file(f), line(l), kind(k) {
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

#ifdef _WIN32
// ---- Windows small host<->device transfers (QWENOX_WIN_SMALLCOPY=0: off) ----
// Under WDDM a small hipMemcpy D2H or pageable hipMemcpyAsync H2D goes through
// the runtime's copy path and leaves the GPU idle ~0.1-0.3 ms per call around
// it; the sampled draft loop does ~16 of them per round. Small reads become
// one copy kernel into a persistent coherent pinned buffer plus one stream
// sync; small async writes become kernel arguments. Same bytes, same stream
// order. Large transfers, capture and non-device sources keep the runtime path.
static const bool g_wsc_on = [] {
  const char* e = getenv("QWENOX_WIN_SMALLCOPY");
  return !(e && !strcmp(e, "0"));
}();
__global__ void k_wsc_copy(uint32_t* dst, const uint32_t* src, unsigned n4) {
  for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x)
    dst[i] = src[i];
  __threadfence_system();
}
struct WscVals {
  int v[16];
  int n;
};
__global__ void k_wsc_set(int* dst, WscVals s) {
  if ((int)threadIdx.x < s.n) dst[threadIdx.x] = s.v[threadIdx.x];
}
struct WscSeg {
  void* dst;
  const void* src;
  size_t n;
};
struct WinSmallCopy {
  static constexpr size_t kCap = (size_t)512 << 10;
  std::mutex mu;
  uint8_t* h = nullptr;
  uint8_t* hd = nullptr;
  bool failed = false;
  bool ready() {
    if (h) return true;
    if (failed) return false;
    if (hipHostMalloc((void**)&h, kCap, hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess ||
        hipHostGetDevicePointer((void**)&hd, h, 0) != hipSuccess) {
      failed = true;
      h = nullptr;
      fprintf(stderr, "win-smallcopy: pinned staging unavailable, using hipMemcpy\n");
      return false;
    }
    return true;
  }
};
static WinSmallCopy g_wsc;
static inline bool wsc_capturing(hipStream_t s) {
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  return hipStreamIsCapturing(s, &st) != hipSuccess || st != hipStreamCaptureStatusNone;
}
static inline bool wsc_is_device(const void* p) {
  hipPointerAttribute_t a{};
  if (hipPointerGetAttributes(&a, p) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return a.type == hipMemoryTypeDevice;
}
// Pageable host memory: the runtime stages it synchronously anyway. Pinned
// destinations keep the real async path.
static inline bool wsc_is_pageable(const void* p) {
  hipPointerAttribute_t a{};
  if (hipPointerGetAttributes(&a, p) != hipSuccess) {
    (void)hipGetLastError();
    return true;
  }
  return a.type != hipMemoryTypeHost && a.type != hipMemoryTypeDevice &&
         a.type != hipMemoryTypeManaged;
}
// Reads every segment (device -> host) with one sync. False: nothing was
// enqueued and the caller must use hipMemcpy.
static inline bool wsc_d2hv(const WscSeg* sg, int nseg, hipStream_t s) {
  if (!g_wsc_on) return false;
  size_t tot = 0;
  for (int i = 0; i < nseg; i++) {
    if (sg[i].n == 0 || (sg[i].n & 3) || ((uintptr_t)sg[i].src & 3)) return false;
    tot += (sg[i].n + 15) & ~(size_t)15;
  }
  if (tot > WinSmallCopy::kCap) return false;
  for (int i = 0; i < nseg; i++)
    if (!wsc_is_device(sg[i].src)) return false;
  std::lock_guard<std::mutex> lk(g_wsc.mu);
  if (!g_wsc.ready() || wsc_capturing(s)) return false;
  size_t off = 0;
  for (int i = 0; i < nseg; i++) {
    const unsigned n4 = (unsigned)(sg[i].n / 4);
    const unsigned grid = std::min(64u, (n4 + 255) / 256);
    hipLaunchKernelGGL(k_wsc_copy, dim3(grid), dim3(256), 0, s, (uint32_t*)(g_wsc.hd + off),
                       (const uint32_t*)sg[i].src, n4);
    off += (sg[i].n + 15) & ~(size_t)15;
  }
  const hipError_t e = hipStreamSynchronize(s);
  if (e != hipSuccess) {
    fprintf(stderr, "win-smallcopy: sync failed: %s\n", hipGetErrorString(e));
    abort();
  }
  off = 0;
  for (int i = 0; i < nseg; i++) {
    memcpy(sg[i].dst, g_wsc.h + off, sg[i].n);
    off += (sg[i].n + 15) & ~(size_t)15;
  }
  return true;
}
// Small async host -> device write as kernel arguments (values read now, as
// the runtime does for pageable sources).
static inline bool wsc_h2d(void* dst, const void* src, size_t n, hipStream_t s) {
  if (!g_wsc_on || n == 0 || n > 64 || (n & 3) || ((uintptr_t)dst & 3)) return false;
  if (wsc_capturing(s) || !wsc_is_device(dst)) return false;
  WscVals v{};
  v.n = (int)(n / 4);
  memcpy(v.v, src, n);
  hipLaunchKernelGGL(k_wsc_set, dim3(1), dim3(32), 0, s, (int*)dst, v);
  return true;
}
#endif
static inline hipError_t sp_hipMemcpy(const char* f, int l, void* dst, const void* src,
                                      size_t n, hipMemcpyKind k) {
  SpTimer t(f, l, sp_kind(k, false));
#ifdef _WIN32
  if (k == hipMemcpyDeviceToHost) {
    const WscSeg sg{dst, src, n};
    if (wsc_d2hv(&sg, 1, nullptr)) return hipSuccess;
  }
#endif
  return hipMemcpy(dst, src, n, k);
}
static inline hipError_t sp_hipMemcpyAsync(const char* f, int l, void* dst, const void* src,
                                           size_t n, hipMemcpyKind k, hipStream_t s = nullptr) {
  SpTimer t(f, l, sp_kind(k, true));
#ifdef _WIN32
  if (k == hipMemcpyHostToDevice && wsc_h2d(dst, src, n, s)) return hipSuccess;
  if (k == hipMemcpyDeviceToHost && wsc_is_pageable(dst)) {
    const WscSeg sg{dst, src, n};
    if (wsc_d2hv(&sg, 1, s)) return hipSuccess;
  }
#endif
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
