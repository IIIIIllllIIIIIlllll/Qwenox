struct Cfg {
  std::string dir;
  uint64_t cap_bytes = 20ull << 30;
  size_t min_tokens = 4096;
  uint64_t queue_bytes = 1024ull << 20;  // captured but not yet written
  bool direct = true;  // direct IO (O_DIRECT / NO_BUFFERING), buffered fallback
  // files in flight: restore reads; writer threads (0 = auto: 4 on a network
  // file system, 1 locally -- parallel writes slowed a local NVMe 0.8 -> 2.2 s
  // per 1.8 GB and buy nothing there)
  int readers = 4, writers = 0;
  // Two-segment eviction: a checkpoint restored hot_hits times moves to the
  // hot segment, evicted only after every probation checkpoint; the hot
  // segment holds at most hot_frac of cap (its LRU entries are demoted past
  // that). hot_hits = 0: plain LRU.
  int hot_hits = 2;
  double hot_frac = 0.5;
  // Also store this turn's <tool_call> / </think> pause checkpoint (taken
  // from its RAM copy at the end of the turn).
  bool pause_ckpt = true;
};

// Model identity for the cache fingerprint (set by main before serve::run).
static std::vector<std::string> g_model_files;  // base + overlays
static std::string g_vision_file;

static const int FMT = 1;  // A4 layout version (part of the fingerprint)

// ---- hashing ----------------------------------------------------------------
static inline uint64_t mix64(uint64_t z) {  // splitmix64 finalizer
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static uint64_t hash_toks(uint64_t h, const int* p, size_t n) {
  for (size_t i = 0; i < n; i++) {
    h ^= (uint32_t)p[i];
    h *= 1099511628211ULL;
  }
  return mix64(h);
}
static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
// Content checksum: four independent lanes (xxhash-style rounds), so verifying
// a restore runs near memory speed.
static uint64_t sum64(const void* data, size_t n) {
  const char* p = (const char*)data;
  const uint64_t P1 = 0x9E3779B185EBCA87ULL, P2 = 0xC2B2AE3D27D4EB4FULL;
  uint64_t a[4] = {P1, P2, ~P1, ~P2};
  size_t i = 0;
  for (; i + 32 <= n; i += 32)
    for (int k = 0; k < 4; k++) {
      uint64_t w;
      memcpy(&w, p + i + 8 * k, 8);
      a[k] = rotl64(a[k] + w * P2, 31) * P1;
    }
  uint64_t h = mix64(n);
  for (int k = 0; k < 4; k++) h = mix64(h ^ a[k]);
  for (; i < n; i++) h = mix64(h ^ (uint8_t)p[i]);
  return h;
}
// c[i] = key of the prefix of i full pages; c[0] = seed. A page's key covers
// the whole prefix, so equal keys mean equal KV for that page.
static std::vector<uint64_t> chain(uint64_t seed, const int* t, int npages) {
  std::vector<uint64_t> c((size_t)npages + 1);
  c[0] = seed;
  for (int i = 0; i < npages; i++)
    c[(size_t)i + 1] =
        hash_toks(mix64(c[(size_t)i] + 0x6b76636bULL), t + (size_t)i * KV_PAGE, KV_PAGE);
  return c;
}
// MTP row r pairs h_r with token r+1: a page's MTP rows also depend on the
// first token of the next page.
static uint64_t mtp_key(uint64_t page_key, int look) {
  return mix64(page_key ^ mix64(0x6d747073ULL + (uint32_t)look));
}
static uint64_t ck_id(uint64_t seed, const int* t, size_t n) {
  return hash_toks(mix64(seed ^ 0x636b6964ULL), t, n);
}

static uint64_t g_fp = 0, g_vis_id = 0;
static uint64_t file_id(const std::string& path) {
  std::error_code ec;
  uint64_t h = mix64((uint64_t)std::filesystem::file_size(path, ec));
  for (unsigned char ch : std::filesystem::path(path).filename().string()) h = mix64(h ^ ch);
  return h;
}
// Plain requests: the fingerprint. M-RoPE requests: placeholder ids are
// pixel-agnostic, so the vision tower, the patch-bytes hash and the grids are
// folded into the seed (a different image never shares a page key).
static uint64_t seed_for(bool mrope, uint64_t img_hash,
                         const std::vector<std::array<int, 3>>& grids) {
  if (!mrope) return g_fp;
  uint64_t h = mix64(g_fp ^ 0x6d726f7065ULL);
  h = mix64(h ^ g_vis_id);
  h = mix64(h ^ img_hash);
  for (const auto& g : grids)
    for (int v : g) h = mix64(h ^ (uint32_t)v);
  return h;
}
// Seed of a live-state prefix: one without image tokens is text KV (see
// has_image_tok) and is keyed like a text request's, so it is shared.
static uint64_t seed_of(const GpuModel& m, const int* toks, size_t n) {
  if (!m.mrope_active_ || !has_image_tok(toks, n)) return g_fp;
  return seed_for(true, m.img_hash_, m.mrope_grids_);
}

// ---- geometry and file layouts ----------------------------------------------
struct Geo {
  int nqsa, ngdn, kvb, has_ik;
  size_t row_b, s_b, conv_b, ring_b, iraw_b;
};
static Geo geo(const GpuModel& m) {
  Geo G;
  G.nqsa = g_cfg.layers / 4;
  G.ngdn = g_cfg.layers - G.nqsa;
  G.kvb = m.qsa_kv_fp32 ? 4 : 2;
  G.has_ik = m.d_ik ? 1 : 0;
  G.row_b = (size_t)512 * G.kvb;
  G.s_b = (size_t)G.ngdn * 48 * 128 * 128 * 4;
  G.conv_b = (size_t)G.ngdn * 3 * 10240 * 4;
  G.ring_b = 9 * 10240 * 4;
  G.iraw_b = m.d_iraw ? (size_t)G.nqsa * 512 * 4 : 0;
  return G;
}
// one layer of one page: K rows, V rows, indexer key blocks
static size_t page_layer_b(int kvb, int has_ik) {
  return 2 * (size_t)KV_PAGE * 512 * kvb + (has_ik ? (size_t)KV_PAGE_BLOCKS * 512 : 0);
}
static const char MAG_C[8] = "GDKVC1", MAG_M[8] = "GDKVM1", MAG_K[8] = "GDKCK1";
struct ChHdr {  // c_<key>.kvc (trunk page) / m_<key>.kvm (MTP section)
  char magic[8];
  uint64_t key, parent, sum;  // sum: sum64 of the body
  int32_t nl, kvb, has_ik, look;  // look: MTP lookahead token (-1 trunk)
  int32_t toks[KV_PAGE];
};
struct CkHdr {  // k_<n>_<id>.kck
  char magic[8];
  uint64_t seed, base, id, sum;
  int32_t n, nc, nl, kvb, ringpos, mtp, mtp_pos, nik, has_ik, pad;
  uint64_t s_b, conv_b, ring_b, iraw_b, mR_b, miraw_b;
};
static_assert(sizeof(ChHdr) % 8 == 0 && sizeof(CkHdr) % 8 == 0, "header alignment");
static size_t ck_body_b(const CkHdr& h) {
  const size_t nt = (size_t)h.n - (size_t)h.nc * KV_PAGE;
  return (size_t)h.nc * 16 + nt * 4 + h.s_b + h.conv_b + h.ring_b + h.iraw_b + h.mR_b +
         h.miraw_b + (size_t)h.nl * (2 * nt * 512 * h.kvb + (size_t)h.nik * 512);
}
static bool ck_sane(const CkHdr& h) {
  const uint64_t big = 1ull << 36;
  return !memcmp(h.magic, MAG_K, 8) && h.n > 0 && h.n < (1 << 24) &&
         h.nc == (h.n - 1) / KV_PAGE && h.nl > 0 && h.nl <= 64 &&
         (h.kvb == 2 || h.kvb == 4) && h.nik >= 0 && h.nik <= 2 * KV_PAGE_BLOCKS &&
         h.s_b < big && h.conv_b < big && h.ring_b < big && h.iraw_b < big &&
         h.mR_b < big && h.miraw_b < big;
}
// Indexer key blocks past the full pages that a checkpoint carries. Clamped
// to the mapped pages (paged kv_runs aborts past them) and the capacity.
static int tail_ik(const GpuModel& m, int n, int nc) {
  if (!m.d_ik) return 0;
  const int hi = std::min({n / 4 + 2, KV_PAGE_BLOCKS * ((n + KV_PAGE - 1) / KV_PAGE),
                           m.index_blocks});
  return std::max(0, hi - KV_PAGE_BLOCKS * nc);
}

// Device slices (layer nqsa = the MTP layer, B0).
static char* kv_layer(GpuModel& m, int half, int l) {
  char* b = m.qsa_kv_fp32 ? (char*)(half ? m.d_vc : m.d_kc)
                          : (char*)(half ? m.d_vcb : m.d_kcb);
  return b + (size_t)l * m.kv_layer_rows * 512 * (m.qsa_kv_fp32 ? 4 : 2);
}
static char* ik_layer(GpuModel& m, int l) {
  return (char*)(m.d_ik + (size_t)l * m.ik_layer_blocks * 128);
}
// Copy logical rows [r0, r1) of a paged per-layer region to/from a
// row-contiguous host buffer (files hold logical order, layout-agnostic).
static bool xfer(GpuModel& m, bool h2d, char* dev, char* host, int r0, int r1,
                 int per_page, size_t row_b) {
  bool ok = true;
  m.kv_runs(r0, r1, per_page, [&](int lr, size_t pr, int n) {
    if (!ok) return;
    char* d = dev + pr * row_b;
    char* h = host + (size_t)(lr - r0) * row_b;
    ok = hipMemcpy(h2d ? (void*)d : (void*)h, h2d ? (void*)h : (void*)d,
                   (size_t)n * row_b,
                   h2d ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost) == hipSuccess;
  });
  return ok;
}
static bool d2h(void* h, const void* d, size_t n) {
  return !n || hipMemcpy(h, d, n, hipMemcpyDeviceToHost) == hipSuccess;
}
static bool h2d(void* d, const void* h, size_t n) {
  return !n || hipMemcpy(d, h, n, hipMemcpyHostToDevice) == hipSuccess;
}

// ---- host buffers ------------------------------------------------------------
// Pooled (page images are all one size): a fresh 100+ MB malloc per save is
// mmap/munmap churn whose first-touch faults stalled D2H for seconds under
// memory pressure. NOTE: never pinned -- hipHostMalloc'd memory makes D2H
// collapse to ~7 MB/s on gfx1151. OS_DIO_ALIGN-aligned, capacity rounded up
// to it: direct IO writes/reads the padded length in place.
struct HBuf {
  std::unique_ptr<char[]> p;  // raw storage; `a` is its aligned start
  char* a = nullptr;
  size_t cap = 0, n = 0;
  char* data() { return a; }
};
static std::mutex g_pool_mtx;
static std::vector<HBuf> g_pool;
static size_t g_pool_bytes = 0;
static const size_t g_pool_max = 512ull << 20;
static HBuf buf_get(size_t n) {
  {
    std::lock_guard<std::mutex> lk(g_pool_mtx);
    size_t best = g_pool.size();
    for (size_t i = 0; i < g_pool.size(); i++)
      if (g_pool[i].cap >= n && (best == g_pool.size() || g_pool[i].cap < g_pool[best].cap))
        best = i;
    if (best < g_pool.size()) {
      HBuf b = std::move(g_pool[best]);
      g_pool.erase(g_pool.begin() + (long)best);
      g_pool_bytes -= b.cap;
      b.n = n;
      return b;
    }
  }
  HBuf b;
  const size_t cap = os_dio_round(n ? n : 1);
  b.p.reset(new (std::nothrow) char[cap + OS_DIO_ALIGN]);
  if (!b.p) return b;
  b.a = (char*)(((uintptr_t)b.p.get() + OS_DIO_ALIGN - 1) & ~(uintptr_t)(OS_DIO_ALIGN - 1));
  b.cap = cap;
  b.n = n;
  return b;
}
static void buf_put(HBuf&& b) {
  if (!b.p) return;
  std::lock_guard<std::mutex> lk(g_pool_mtx);
  if (g_pool_bytes + b.cap > g_pool_max) return;  // freed on scope exit
  g_pool_bytes += b.cap;
  g_pool.push_back(std::move(b));
}

// ---- files -------------------------------------------------------------------
static std::string g_dir;
static uint64_t g_cap = 0, g_qcap = 0;
static bool g_on = false;
static std::string ch_name(uint64_t k) {
  char b[32];
  snprintf(b, sizeof b, "c_%016llx.kvc", (unsigned long long)k);
  return b;
}
static std::string mt_name(uint64_t k) {
  char b[32];
  snprintf(b, sizeof b, "m_%016llx.kvm", (unsigned long long)k);
  return b;
}
static std::string ck_name(int n, uint64_t id) {
  char b[48];
  snprintf(b, sizeof b, "k_%d_%016llx.kck", n, (unsigned long long)id);
  return b;
}
static std::string path_of(const std::string& name) { return g_dir + "/" + name; }
static int64_t now_t() {
  return (int64_t)std::filesystem::file_time_type::clock::now().time_since_epoch().count();
}
// Log lines produced while g_mtx is held are buffered per thread and printed
// by flush_log() after the lock is released: a stderr write can block (a
// paused Windows console behind the launcher's pipe), and blocking while
// holding g_mtx froze the GPU thread at its next match/enqueue/Turn.
static thread_local std::string tl_log;
__attribute__((format(printf, 1, 2))) static void logf_locked(const char* fmt, ...) {
  char b[512];
  va_list ap;
  va_start(ap, fmt);
  const int k = vsnprintf(b, sizeof b, fmt, ap);
  va_end(ap);
  if (k > 0) tl_log.append(b, std::min((size_t)k, sizeof b - 1));
}
static void flush_log() {  // call with g_mtx NOT held
  if (tl_log.empty()) return;
  fputs(tl_log.c_str(), stderr);
  tl_log.clear();
}
static bool wr_all(int fd, const char* p, size_t n) {
  while (n) {
    ssize_t w = os_write(fd, p, std::min(n, (size_t)1 << 30));
    if (w <= 0) return false;
    p += w;
    n -= (size_t)w;
  }
  return true;
}
static bool rd_all(int fd, char* p, size_t n) {
  while (n) {
    ssize_t r = os_read(fd, p, std::min(n, (size_t)1 << 30));
    if (r <= 0) return false;
    p += r;
    n -= (size_t)r;
  }
  return true;
}
// Direct IO: cache files are written once and read once per restore, so the
// OS file cache gains nothing from them, and on this unified-memory APU every
// byte it holds is taken from device memory and the PLE table's hot pages
// (Windows has no DONTNEED: buffered kvsnap IO grows its standby list by the
// whole file). A file system refusing it (tmpfs, some network/FUSE mounts)
// switches the process to buffered IO once.
static std::atomic<bool> g_direct{false};
// Files in flight: restore reads (QWENOX_KVSNAP_READERS) and writer threads
// (QWENOX_KVSNAP_WRITERS). One at a time is enough for a local NVMe; a
// network share pays a round trip per file (open, flush, rename).
static int g_readers = 4, g_writers = 4;
static void direct_unsupported(const char* op) {
  if (g_direct.exchange(false))
    fprintf(stderr, "kvsnap: direct IO not supported for %s in %s; using buffered IO\n", op,
            g_dir.c_str());
}
// tmp_<name>, fdatasync, rename: a crash never publishes a torn file.
// `p` is an HBuf's data (aligned, padded capacity).
static bool write_file(const std::string& name, const char* p, size_t n) {
  const std::string tmp = path_of("tmp_" + name);
  bool ok = false, done = false;
  if (g_direct.load()) {
    const int rc = os_write_direct(tmp.c_str(), p, n);
    if (rc < 0) direct_unsupported("writes");
    else ok = rc == 1, done = true;
  }
  if (!done) {
    int fd = os_open_wr(tmp.c_str());
    if (fd < 0) return false;
    ok = wr_all(fd, p, n) && os_sync(fd) == 0;
    if (ok) os_drop_cache(fd);  // clean now: don't let the page cache keep it
    if (os_close(fd)) ok = false;
  }
  std::error_code ec;
  if (ok) {
    std::filesystem::rename(tmp, path_of(name), ec);
    ok = !ec;
  }
  if (!ok) std::filesystem::remove(tmp, ec);
  return ok;
}
// Whole file into a pooled buffer; `want` = exact expected size (0 = any,
// bounded by `max`).
static bool read_file(const std::string& path, HBuf& b, size_t want, size_t max = 0) {
  if (g_direct.load()) {
    std::error_code ec;
    const uint64_t fsz = std::filesystem::file_size(path, ec);
    if (ec || !fsz || (want ? fsz != want : fsz > max)) return false;
    b = buf_get((size_t)fsz);
    if (!b.p) return false;
    const int rc = os_read_direct(path.c_str(), b.data(), (size_t)fsz);
    if (rc >= 0) return rc == 1;
    direct_unsupported("reads");  // b is reused by the buffered read below
  }
  int fd = os_open_rd(path.c_str());
  if (fd < 0) return false;
  const int64_t fsz = os_fsize(fd);
  bool ok = fsz > 0 && (want ? (uint64_t)fsz == want : (uint64_t)fsz <= max);
  if (ok) {
    b = buf_get((size_t)fsz);
    ok = b.p && rd_all(fd, b.data(), (size_t)fsz);
  }
  os_drop_cache(fd);
  os_close(fd);
  return ok;
}

// ---- in-memory index ---------------------------------------------------------
// Pages and MTP sections are shared by every checkpoint whose prefix covers
// them (refcounted); checkpoints are the LRU unit. All under g_mtx.
enum { ST_PEND, ST_PUB, ST_FAIL };
struct Blk {
  uint64_t bytes = 0;
  int refs = 0;
  int st = ST_PEND;
};
struct CkEnt {
  std::string name;
  uint64_t seed = 0, base = 0, id = 0, bytes = 0;
  int n = 0, nc = 0, nl = 0;
  bool mtp = false, pub = false;
  std::vector<uint64_t> chunks, mtps;  // mtps[i] == 0: not referenced
  std::vector<int> tail;               // tokens [256*nc, n)
  int64_t t = 0;                       // LRU time (file_time clock)
  int busy = 0;                        // restores in progress: not evictable
  int hits = 0;                        // successful restores (kvsnap_hits.txt)
  bool hot = false;                    // hot segment (two-segment eviction)
};
static std::mutex g_mtx;
static std::condition_variable g_cv;
static std::unordered_map<uint64_t, Blk> g_ch, g_mt;
static std::vector<std::shared_ptr<CkEnt>> g_ck;
static uint64_t g_total = 0;  // bytes of published files

// ---- janitor: file deletes and LRU mtime bumps, off the lock ------------------
// Each is a round trip on a share (2-8 ms), and evicting one checkpoint can
// drop hundreds of pages: done under g_mtx that froze the GPU thread at its
// next match/enqueue/Turn. The index only queues the operation (the index
// itself changes at once); one background thread performs it. File names
// are content-addressed and can come back (a page evicted, then saved
// again): a queued delete is skipped when the name is in the index again by
// the time it runs, and a writer about to publish a name waits while that
// name is being deleted.
// J_HITS rewrites kvsnap_hits.txt (restore counts, read back by init): at
// most one is queued, it snapshots the index when it runs.
enum { J_DEL, J_TOUCH, J_HITS };
struct JOp {
  std::string name;
  int op;  // J_TOUCH: bump the mtime (LRU order across restarts)
};
static std::deque<JOp> g_jq;
static std::string g_jcur;    // name being deleted right now
static bool g_jbusy = false;  // the janitor holds an operation
static bool g_hits_q = false; // a J_HITS is queued
static const char* const HITS_FILE = "kvsnap_hits.txt";
static void del_locked(const std::string& name) {
  g_jq.push_back({name, J_DEL});
  g_cv.notify_all();
}
static void hits_dirty_locked() {
  if (g_hits_q) return;
  g_hits_q = true;
  g_jq.push_back({std::string(), J_HITS});
  g_cv.notify_all();
}
// The index holds this name (published or pending): its file is wanted.
static bool live_locked(const std::string& n) {
  if (n.size() == 22 && n[1] == '_' && (n[0] == 'c' || n[0] == 'm')) {
    const uint64_t k = strtoull(n.substr(2, 16).c_str(), nullptr, 16);
    return (n[0] == 'c' ? g_ch : g_mt).count(k) > 0;
  }
  if (n[0] == 'k')
    for (const auto& e : g_ck)
      if (e->name == n) return true;
  return false;
}

static void unref_locked(std::unordered_map<uint64_t, Blk>& mp, uint64_t key, bool mtp) {
  auto it = mp.find(key);
  if (it == mp.end() || --it->second.refs > 0) return;
  if (it->second.st == ST_PEND) return;  // the writer deletes it on publish
  if (it->second.st == ST_PUB) {
    del_locked(mtp ? mt_name(key) : ch_name(key));
    g_total -= it->second.bytes;
  }
  mp.erase(it);
}
static void drop_ck_locked(std::shared_ptr<CkEnt> e, bool del_file) {
  auto it = std::find(g_ck.begin(), g_ck.end(), e);
  if (it == g_ck.end()) return;
  g_ck.erase(it);
  if (e->pub) {
    g_total -= e->bytes;
    if (del_file) del_locked(e->name);
  }
  // its line must leave kvsnap_hits.txt: the name may be saved again later
  if (e->hits > 0) hits_dirty_locked();
  for (uint64_t k : e->chunks) unref_locked(g_ch, k, false);
  for (uint64_t k : e->mtps)
    if (k) unref_locked(g_mt, k, true);
}
// Two-segment eviction (2Q-style). Checkpoints enter on probation; one
// restored g_hot_hits times becomes hot. Eviction takes the LRU probation
// checkpoint first, a hot one only when none is left, so the few prefixes
// that are read again and again (a client's fixed system prompt) are not
// flushed by a run of one-off long prompts. The hot segment is bounded by
// checkpoint file bytes (pages are shared, not counted): past g_hot_frac of
// the cap its LRU entries go back to probation and must earn their place
// again. Only a restore counts as a hit: commit_seed also re-meets known
// states (the next turn's cut at the previous turn end), which would make
// every checkpoint hot. g_hot_hits == 0: plain LRU.
static int g_hot_hits = 2;
static double g_hot_frac = 0.5;
static void demote_locked() {
  uint64_t hot_b = 0;
  for (auto& e : g_ck)
    if (e->pub && e->hot) hot_b += e->bytes;
  const uint64_t lim = (uint64_t)(g_cap * g_hot_frac);
  while (hot_b > lim) {
    std::shared_ptr<CkEnt> v;
    for (auto& e : g_ck)
      if (e->pub && e->hot && (!v || e->t < v->t)) v = e;
    if (!v) break;
    v->hot = false;
    v->hits = 0;
    hot_b -= v->bytes;
    logf_locked("kvsnap: %s back to probation (hot segment over %.1f GB)\n", v->name.c_str(),
                lim / 1073741824.0);
    hits_dirty_locked();
  }
}
static void evict_locked(const CkEnt* keep) {
  demote_locked();
  while (g_total > g_cap) {
    std::shared_ptr<CkEnt> v;  // probation before hot, LRU within a segment
    for (auto& e : g_ck)
      if (e->pub && !e->busy && e.get() != keep &&
          (!v || (v->hot && !e->hot) || (v->hot == e->hot && e->t < v->t)))
        v = e;
    if (!v) break;
    logf_locked("kvsnap: evicted %s (%s LRU, %d hit%s, cap %.1f GB)\n", v->name.c_str(),
                v->hot ? "hot" : "probation", v->hits, v->hits == 1 ? "" : "s",
                g_cap / 1073741824.0);
    drop_ck_locked(v, true);
  }
}
static void touch_locked(const std::shared_ptr<CkEnt>& e) {
  e->t = now_t();
  g_jq.push_back({e->name, J_TOUCH});  // the file mtime carries it across restarts
  g_cv.notify_all();
}
// A successful restore of `e`.
static void hit_locked(const std::shared_ptr<CkEnt>& e) {
  touch_locked(e);
  e->hits++;
  if (g_hot_hits > 0 && !e->hot && e->hits >= g_hot_hits) {
    e->hot = true;
    logf_locked("kvsnap: %s is hot (%d restores)\n", e->name.c_str(), e->hits);
    demote_locked();
  }
  hits_dirty_locked();
}
// kvsnap_hits.txt: "<checkpoint file name> <restores>" per line, checkpoints
// with at least one restore. Written whole (tmp + rename) by the janitor.
static std::string hits_text_locked() {
  std::string s;
  for (auto& e : g_ck)
    if (e->pub && e->hits > 0) s += e->name + " " + std::to_string(e->hits) + "\n";
  return s;
}
static void write_hits(const std::string& text) {
  const std::string tmp = path_of(std::string("tmp_") + HITS_FILE);
  std::error_code ec;
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(text.data(), (std::streamsize)text.size());
    if (!f.good()) {
      f.close();
      std::filesystem::remove(tmp, ec);
      return;
    }
  }
  std::filesystem::rename(tmp, path_of(HITS_FILE), ec);
  if (ec) std::filesystem::remove(tmp, ec);
}
static std::unordered_map<std::string, int> read_hits() {
  std::unordered_map<std::string, int> mp;
  std::ifstream f(path_of(HITS_FILE), std::ios::binary);
  std::string name;
  int n;
  while (f >> name >> n)
    if (n > 0) mp[name] = n;
  return mp;
}

static void janitor() {
  int nd = 0;  // deletes of the current burst (logged when the queue drains)
  double ds = 0;
  for (;;) {
    JOp op;
    std::string hits;  // J_HITS: the file text, snapshotted under the lock
    {
      std::unique_lock<std::mutex> lk(g_mtx);
      g_cv.wait(lk, [] { return !g_jq.empty(); });
      op = std::move(g_jq.front());
      g_jq.pop_front();
      if (op.op == J_HITS) {
        g_hits_q = false;
        hits = hits_text_locked();
      } else if ((op.op == J_DEL) == live_locked(op.name)) {  // re-created / already gone
        lk.unlock();
        g_cv.notify_all();
        continue;
      }
      g_jbusy = true;
      if (op.op == J_DEL) g_jcur = op.name;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::error_code ec;
    if (op.op == J_DEL)
      std::filesystem::remove(path_of(op.name), ec);
    else if (op.op == J_TOUCH)
      std::filesystem::last_write_time(path_of(op.name),
                                       std::filesystem::file_time_type::clock::now(), ec);
    else
      write_hits(hits);
    bool drained;
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      g_jbusy = false;
      g_jcur.clear();
      drained = g_jq.empty();
    }
    g_cv.notify_all();
    if (op.op != J_DEL) continue;
    nd++;
    ds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Windows: a sharing violation (antivirus scan, indexer, an open handle)
    // leaves the file on disk although the index already dropped its bytes
    if (ec)
      fprintf(stderr, "kvsnap: cannot delete %s (error %d); the file stays on disk\n",
              op.name.c_str(), ec.value());
    if (drained) {
      if (nd >= 16 || ds >= 0.5)
        fprintf(stderr, "%s kvsnap: deleted %d files in %.2fs (background)\n", log_ts(), nd,
                ds);
      nd = 0;
      ds = 0;
    }
  }
}

// ---- background writers ------------------------------------------------------
// g_writers detached threads take captured images in FIFO order: a commit
// enqueues its pages, then its MTP sections, then its checkpoints. Pages are
// written in parallel; a writer taking a checkpoint first waits until no page
// it names is still pending (those were taken earlier, so they are being
// written), then requires them all published: a checkpoint never names a
// page that is not on disk.
struct Item {
  int kind = 0;  // 0 trunk page, 1 MTP section, 2 checkpoint
  uint64_t key = 0;
  HBuf buf;
  std::shared_ptr<CkEnt> ck;
  double cap_s = 0;
  int new_ch = 0, new_mt = 0;
  uint64_t new_b = 0;
};
static std::deque<Item> g_q;
static uint64_t g_q_bytes = 0;  // queued + being written
static int g_wbusy = 0;         // writers holding an item
// Wall time of page/MTP-section writes since the last checkpoint log: from
// the first write start to the last write end (writers overlap).
static bool g_pw_on = false;
static std::chrono::steady_clock::time_point g_pw_t0, g_pw_t1;

static void writer() {
  for (;;) {
    Item it;
    {
      std::unique_lock<std::mutex> lk(g_mtx);
      g_cv.wait(lk, [] { return !g_q.empty(); });
      it = std::move(g_q.front());
      g_q.pop_front();
      g_wbusy++;
    }
    const size_t len = it.buf.n;
    char* p = it.buf.data();
    std::string name;
    bool go = true;
    if (it.kind == 2) {
      std::unique_lock<std::mutex> lk(g_mtx);
      g_cv.wait_for(lk, std::chrono::seconds(60), [&] {
        for (uint64_t k : it.ck->chunks) {
          auto f = g_ch.find(k);
          if (f != g_ch.end() && f->second.st == ST_PEND) return false;
        }
        return true;
      });
      for (uint64_t k : it.ck->chunks) {
        auto f = g_ch.find(k);
        if (f == g_ch.end() || f->second.st != ST_PUB) go = false;
      }
      name = it.ck->name;
    } else {
      name = it.kind ? mt_name(it.key) : ch_name(it.key);
    }
    {  // an earlier copy of this name is being deleted: publish after it
      std::unique_lock<std::mutex> lk(g_mtx);
      g_cv.wait(lk, [&] { return g_jcur != name; });
    }
    const size_t hb = it.kind == 2 ? sizeof(CkHdr) : sizeof(ChHdr);
    const uint64_t sum = sum64(p + hb, len - hb);
    memcpy(p + (it.kind == 2 ? offsetof(CkHdr, sum) : offsetof(ChHdr, sum)), &sum, 8);
    const auto w0 = std::chrono::steady_clock::now();
    if (it.kind < 2) {
      std::lock_guard<std::mutex> lk(g_mtx);
      if (!g_pw_on) g_pw_on = true, g_pw_t0 = w0;
    }
    const bool ok = go && write_file(name, p, len);
    const auto w1 = std::chrono::steady_clock::now();
    const double ws = std::chrono::duration<double>(w1 - w0).count();
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      g_q_bytes -= len;
      if (it.kind < 2) {
        if (w1 > g_pw_t1) g_pw_t1 = w1;
        auto& mp = it.kind ? g_mt : g_ch;
        auto f = mp.find(it.key);
        if (f == mp.end() || f->second.refs == 0) {
          if (ok) del_locked(name);  // every referencing checkpoint is gone
          if (f != mp.end()) mp.erase(f);
        } else if (ok) {
          f->second.st = ST_PUB;
          f->second.bytes = len;
          g_total += len;
        } else {
          f->second.st = ST_FAIL;  // erased with its last reference
          logf_locked("kvsnap: write of %s failed (non-fatal)\n", name.c_str());
        }
      } else {
        const auto& e = it.ck;
        if (!ok) {
          logf_locked("kvsnap: save of %d tokens failed (%s, non-fatal)\n", e->n,
                      go ? "write error" : "a page write failed");
          drop_ck_locked(e, false);
        } else {
          // same state saved before without MTP state: this file replaced it
          // (it inherits the restore count and segment)
          for (auto& o : g_ck)
            if (o != e && o->name == e->name) {
              e->hits = o->hits;
              e->hot = o->hot;
              drop_ck_locked(o, false);
              break;
            }
          e->pub = true;
          e->bytes = len;
          e->t = now_t();
          g_total += len;
          // "pages write": wall time of the page files written since the last
          // checkpoint log (this commit's new pages, unless commits overlap)
          const double pw =
              g_pw_on ? std::chrono::duration<double>(g_pw_t1 - g_pw_t0).count() : 0.0;
          logf_locked("%s kvsnap: saved %d tokens -> %s (ckpt %.1f MB + %d new pages%s "
                      "%.1f MB; capture %.2fs, %s IO pages write %.2fs, write %.2fs%s; "
                      "total %.2f GB)\n",
                      log_ts(), e->n, e->name.c_str(), len / 1048576.0, it.new_ch,
                      it.new_mt ? " + mtp" : "", it.new_b / 1048576.0, it.cap_s,
                      g_direct.load() ? "direct" : "buffered", pw > 0 ? pw : 0.0, ws,
                      e->mtp ? ", mtp" : "", g_total / 1073741824.0);
          crashdiag::note("kvsnap saved %d tokens (%.1f MB)", e->n, len / 1048576.0);
          g_pw_on = false;
          evict_locked(e.get());
        }
      }
      g_wbusy--;
    }
    g_cv.notify_all();
    flush_log();  // g_mtx released: a blocked stderr no longer stalls the GPU thread
    buf_put(std::move(it.buf));
  }
}
// Returns the seconds the caller (the GPU thread) waited for queue space.
static double enqueue(Item&& it) {
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_lock<std::mutex> lk(g_mtx);
  // bounded: the GPU thread waits for the disk rather than hoard host memory
  g_cv.wait(lk, [&] { return g_q_bytes == 0 || g_q_bytes + it.buf.n <= g_qcap; });
  g_q_bytes += it.buf.n;
  g_q.push_back(std::move(it));
  lk.unlock();
  g_cv.notify_all();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Requests in flight: a turn commits after its reply went out, so KVSYNC
// also waits for those (gen_one holds a Turn for its whole run).
static int g_turns = 0;
struct Turn {
  Turn() {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_turns++;
  }
  ~Turn() {
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      g_turns--;
    }
    g_cv.notify_all();
  }
};

// KVSYNC: wait for in-flight turns, the writer and the janitor to drain;
// "K <ckpts> <pages> <mtp> <bytes>".
static std::string sync_line() {
  std::unique_lock<std::mutex> lk(g_mtx);
  g_cv.wait(lk, [] {
    return g_q.empty() && !g_wbusy && !g_turns && g_jq.empty() && !g_jbusy;
  });
  size_t nck = 0;
  for (auto& e : g_ck) nck += e->pub;
  char b[128];
  snprintf(b, sizeof b, "K %zu %zu %zu %llu\n", nck, g_ch.size(), g_mt.size(),
           (unsigned long long)g_total);
  return b;
}

// ---- save --------------------------------------------------------------------
// A checkpoint captured at a cut or at the end of a turn; its trunk pages are
// captured by commit(), once, and only those not already on disk.
struct Pending {
  std::shared_ptr<CkEnt> ck;
  HBuf buf;  // checkpoint file image (the writer fills in the checksum)
  double cap_s = 0;
};

// GPU-owner thread. Without `rc`: `toks` must be exactly the live state
// (m.pos); `mtp`: the MTP layer is in lockstep (mtp_pos == pos-1, rows
// [0, pos-1) ingested), the checkpoint then carries d_R/d_miraw/mtp_pos and
// names MTP sections.
//
// With `rc` (capture_ram): a RAM checkpoint of `toks`, a strict prefix of
// the live state -- a <tool_call> / </think> pause of this turn, captured at
// its end. The GDN/PLE/indexer-raw/MTP-stream state comes from the
// checkpoint's pinned host copies; the partial page rows [256*nc, n) of
// every layer from the live KV: rows below the live position are
// append-only, so they still hold the values at n. Later writes do reach a
// few slots past them -- the straddling indexer key block, MTP row n-1 --
// exactly as in a cut capture taken before the next segment; a restore
// re-pools that block from the raw ring and re-ingests MTP row n-1 at the
// continuation boundary.
static void capture_impl(GpuModel& m, const std::vector<int>& toks, bool mtp,
                         const rckpt::Ck* rc, std::vector<Pending>& out) {
  if (!g_on) return;
  crashdiag::note("kvsnap capture %zu tokens (mtp %d, from rckpt %d)", toks.size(), (int)mtp,
                  (int)(rc != nullptr));
  const int n = (int)toks.size();
  if (n <= 0 || (rc ? (n >= m.pos || rc->pos != n) : n != m.pos)) {
    fprintf(stderr, "kvsnap: save skipped: %d tokens but live pos %d\n", n, m.pos);
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const Geo G = geo(m);
  if (rc) {
    if (rc->s.size() * 4 != G.s_b || rc->convst.size() * 4 != G.conv_b ||
        rc->ring.size() * 4 != G.ring_b || rc->iraw.size() * 4 != G.iraw_b) {
      fprintf(stderr, "kvsnap: pause checkpoint of %d tokens skipped (RAM copy layout)\n", n);
      return;
    }
    mtp = rc->mtp_ok && m.kv_nl > G.nqsa && rc->mtp_pos == n - 1 &&
          rc->mR.size() == (size_t)g_cfg.branches * g_cfg.d &&
          rc->miraw.size() == (m.d_miraw ? (size_t)4 * 128 : 0);
  } else {
    mtp = mtp && m.kv_nl > G.nqsa && m.mtp_pos == m.pos - 1;
  }
  const int nc = (n - 1) / KV_PAGE, nt = n - nc * KV_PAGE;
  auto e = std::make_shared<CkEnt>();
  e->seed = seed_of(m, toks.data(), (size_t)n);
  std::vector<uint64_t> ch = chain(e->seed, toks.data(), nc);
  e->base = ch[(size_t)nc];
  e->id = ck_id(e->seed, toks.data(), (size_t)n);
  e->n = n;
  e->nc = nc;
  e->nl = mtp ? G.nqsa + 1 : G.nqsa;
  e->mtp = mtp;
  e->chunks.assign(ch.begin() + 1, ch.end());
  e->mtps.assign((size_t)nc, 0);
  if (mtp)
    for (int i = 0; i < nc; i++)
      e->mtps[(size_t)i] = mtp_key(ch[(size_t)i + 1], toks[(size_t)(i + 1) * KV_PAGE]);
  e->tail.assign(toks.begin() + (size_t)nc * KV_PAGE, toks.end());
  e->name = ck_name(n, e->id);
  CkHdr h{};
  memcpy(h.magic, MAG_K, 8);
  h.seed = e->seed;
  h.base = e->base;
  h.id = e->id;
  h.n = n;
  h.nc = nc;
  h.nl = e->nl;
  h.kvb = G.kvb;
  h.ringpos = rc ? rc->ringpos : m.ple_ringpos;
  h.mtp = mtp;
  h.mtp_pos = mtp ? (rc ? rc->mtp_pos : m.mtp_pos) : 0;
  h.nik = tail_ik(m, n, nc);
  h.has_ik = G.has_ik;
  h.s_b = G.s_b;
  h.conv_b = G.conv_b;
  h.ring_b = G.ring_b;
  h.iraw_b = G.iraw_b;
  h.mR_b = mtp ? (uint64_t)g_cfg.branches * g_cfg.d * 4 : 0;
  h.miraw_b = mtp && m.d_miraw ? 4 * 128 * 4 : 0;
  const size_t len = sizeof h + ck_body_b(h);
  HBuf b = buf_get(len);
  if (!b.p) return;
  char* p = b.data();
  memcpy(p, &h, sizeof h);
  p += sizeof h;
  memcpy(p, e->chunks.data(), (size_t)nc * 8);
  p += (size_t)nc * 8;
  memcpy(p, e->mtps.data(), (size_t)nc * 8);
  p += (size_t)nc * 8;
  memcpy(p, e->tail.data(), (size_t)nt * 4);
  p += (size_t)nt * 4;
  CK(hipStreamSynchronize(g_str));  // also completes rckpt's async D2H into rc
  bool ok = true;
  // device state (rc == nullptr) or its RAM checkpoint copy
  auto blob = [&](const void* d, const void* host, size_t sz) {
    if (ok) {
      if (rc) {
        if (sz) memcpy(p, host, sz);
      } else {
        ok = d2h(p, d, sz);
      }
    }
    p += sz;
  };
  blob(m.d_S, rc ? rc->s.data() : nullptr, h.s_b);
  blob(m.d_convst, rc ? rc->convst.data() : nullptr, h.conv_b);
  blob(m.d_ring, rc ? rc->ring.data() : nullptr, h.ring_b);
  blob(m.d_iraw, rc ? rc->iraw.data() : nullptr, h.iraw_b);
  blob(m.d_R, rc ? rc->mR.data() : nullptr, h.mR_b);
  blob(m.d_miraw, rc ? rc->miraw.data() : nullptr, h.miraw_b);
  const int r0 = nc * KV_PAGE, b0 = nc * KV_PAGE_BLOCKS;
  for (int l = 0; l < h.nl && ok; l++) {
    ok = xfer(m, false, kv_layer(m, 0, l), p, r0, n, KV_PAGE, G.row_b) &&
         xfer(m, false, kv_layer(m, 1, l), p + (size_t)nt * G.row_b, r0, n, KV_PAGE,
              G.row_b) &&
         xfer(m, false, G.has_ik ? ik_layer(m, l) : nullptr,
              p + 2 * (size_t)nt * G.row_b, b0, b0 + h.nik, KV_PAGE_BLOCKS, 512);
    p += 2 * (size_t)nt * G.row_b + (size_t)h.nik * 512;
  }
  if (!ok) {
    fprintf(stderr, "kvsnap: capture of %d tokens failed (non-fatal)\n", n);
    return;
  }
  Pending pd;
  pd.ck = std::move(e);
  pd.buf = std::move(b);
  pd.cap_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  out.push_back(std::move(pd));
}
static void capture(GpuModel& m, const std::vector<int>& toks, bool mtp,
                    std::vector<Pending>& out) {
  capture_impl(m, toks, mtp, nullptr, out);
}
// End of turn, before commit(): the RAM checkpoint of hist[0, n) (a pause of
// this turn) for this tier too. Without one -- evicted for pool space, not
// saved -- nothing is captured. The MTP state goes with it when the RAM copy
// has it.
static void capture_ram(GpuModel& m, const std::vector<int>& hist, size_t n,
                        std::vector<Pending>& out) {
  if (!g_on || n == 0 || n >= hist.size() || (int)hist.size() != m.pos) return;
  const rckpt::Ck* rc = rckpt::find(hist, n);
  if (!rc) {
    fprintf(stderr, "kvsnap: pause checkpoint of %zu tokens not saved (no RAM copy)\n", n);
    return;
  }
  capture_impl(m, std::vector<int>(hist.begin(), hist.begin() + (std::ptrdiff_t)n), true, rc,
               out);
}

// One page's (or MTP section's) file image, read from the live sequence.
static bool page_image(GpuModel& m, const Geo& G, bool mtp, int i, uint64_t key,
                       uint64_t parent, int look, const int* toks, HBuf& b) {
  const int l0 = mtp ? G.nqsa : 0, nl = mtp ? 1 : G.nqsa;
  const size_t lb = page_layer_b(G.kvb, G.has_ik);
  b = buf_get(sizeof(ChHdr) + (size_t)nl * lb);
  if (!b.p) return false;
  ChHdr h{};
  memcpy(h.magic, mtp ? MAG_M : MAG_C, 8);
  h.key = key;
  h.parent = parent;
  h.nl = nl;
  h.kvb = G.kvb;
  h.has_ik = G.has_ik;
  h.look = look;
  memcpy(h.toks, toks, sizeof h.toks);
  memcpy(b.data(), &h, sizeof h);
  char* p = b.data() + sizeof h;
  const int r0 = i * KV_PAGE, k0 = i * KV_PAGE_BLOCKS;
  bool ok = true;
  for (int l = l0; l < l0 + nl && ok; l++, p += lb)
    ok = xfer(m, false, kv_layer(m, 0, l), p, r0, r0 + KV_PAGE, KV_PAGE, G.row_b) &&
         xfer(m, false, kv_layer(m, 1, l), p + (size_t)KV_PAGE * G.row_b, r0,
              r0 + KV_PAGE, KV_PAGE, G.row_b) &&
         (!G.has_ik || xfer(m, false, ik_layer(m, l), p + 2 * (size_t)KV_PAGE * G.row_b,
                            k0, k0 + KV_PAGE_BLOCKS, KV_PAGE_BLOCKS, 512));
  return ok;
}

// End of turn (GPU-owner thread, `hist` == the live state): register the
// pending checkpoints, capture the pages/MTP sections none of the cached
// files hold yet, and hand everything to the writer. Every pending state is
// a prefix of `hist` (cut states of this prompt), so the live KV still holds
// their pages: KV rows and MTP rows below each checkpoint are append-only.
// All of `pend` shares `seed` (commit() groups them).
static void commit_seed(GpuModel& m, const std::vector<int>& hist,
                        std::vector<Pending>& pend, uint64_t seed) {
  if (!g_on || pend.empty() || (int)hist.size() != m.pos) {
    pend.clear();
    return;
  }
  // The writer is still a full queue behind (a slow share, back-to-back
  // turns): drop this turn's saves instead of stalling the GPU thread in
  // enqueue -- the cache is an accelerator, the next request must not wait.
  uint64_t behind = 0;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_q_bytes >= g_qcap) behind = g_q_bytes;
  }
  if (behind) {
    fprintf(stderr, "%s kvsnap: writer behind (%.0f MB queued): not saving %zu state%s of "
            "%zu tokens\n", log_ts(), behind / 1048576.0, pend.size(),
            pend.size() == 1 ? "" : "s", hist.size());
    for (auto& pd : pend) buf_put(std::move(pd.buf));
    pend.clear();
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  double qwait = 0;  // GPU thread blocked on a full queue
  const Geo G = geo(m);
  std::sort(pend.begin(), pend.end(),
            [](const Pending& a, const Pending& b) { return a.ck->n < b.ck->n; });
  const int np = (int)((hist.size() - 1) / KV_PAGE);
  const std::vector<uint64_t> ch = chain(seed, hist.data(), np);
  std::vector<char> need_ch((size_t)np, 0), need_mt((size_t)np, 0);
  std::vector<int> owner_ch((size_t)np, -1), owner_mt((size_t)np, -1);
  std::vector<size_t> keep;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (size_t pi = 0; pi < pend.size(); pi++) {
      auto& e = pend[pi].ck;
      if (e->seed != seed || e->nc > np || ch[(size_t)e->nc] != e->base ||
          !std::equal(e->tail.begin(), e->tail.end(),
                      hist.begin() + (size_t)e->nc * KV_PAGE))
        continue;  // not a prefix of the final state (should not happen)
      std::shared_ptr<CkEnt> old;
      for (auto& o : g_ck)
        if (o->name == e->name) old = o;
      if (old && (old->mtp || !e->mtp)) {  // already cached (or queued)
        if (old->pub) touch_locked(old);
        continue;
      }
      bool bad = false;  // a failed page write whose references linger
      for (uint64_t k : e->chunks) {
        auto f = g_ch.find(k);
        if (f != g_ch.end() && f->second.st == ST_FAIL) bad = true;
      }
      if (bad) continue;
      for (int i = 0; i < e->nc; i++) {
        auto r = g_ch.try_emplace(e->chunks[(size_t)i]);
        r.first->second.refs++;
        if (r.second) need_ch[(size_t)i] = 1, owner_ch[(size_t)i] = (int)pi;
      }
      for (int i = 0; i < e->nc; i++) {
        uint64_t& k = e->mtps[(size_t)i];
        if (!k) continue;
        auto f = g_mt.find(k);
        if (f != g_mt.end() && f->second.st == ST_FAIL) {
          k = 0;  // restores then find the section missing: no MTP state
          continue;
        }
        auto r = g_mt.try_emplace(k);
        r.first->second.refs++;
        if (r.second) need_mt[(size_t)i] = 1, owner_mt[(size_t)i] = (int)pi;
      }
      g_ck.push_back(e);  // pending (pub = false): dedupes the next commit
      keep.push_back(pi);
    }
  }
  std::vector<int> nch(pend.size(), 0), nmt(pend.size(), 0);
  std::vector<uint64_t> nb(pend.size(), 0);
  auto fail_blk = [&](std::unordered_map<uint64_t, Blk>& mp, uint64_t k) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto f = mp.find(k);
    if (f != mp.end()) f->second.st = ST_FAIL;
  };
  CK(hipStreamSynchronize(g_str));
  for (int pass = 0; pass < 2; pass++)
    for (int i = 0; i < np; i++) {
      if (!(pass ? need_mt : need_ch)[(size_t)i]) continue;
      const int o = (pass ? owner_mt : owner_ch)[(size_t)i];
      Item it;
      it.kind = pass;
      it.key = pass ? pend[(size_t)o].ck->mtps[(size_t)i] : ch[(size_t)i + 1];
      const uint64_t parent = pass ? ch[(size_t)i + 1] : ch[(size_t)i];
      const int look = pass ? hist[(size_t)(i + 1) * KV_PAGE] : -1;
      if (!page_image(m, G, pass == 1, i, it.key, parent, look,
                      hist.data() + (size_t)i * KV_PAGE, it.buf)) {
        fprintf(stderr, "kvsnap: capture of page %d failed (non-fatal)\n", i);
        fail_blk(pass ? g_mt : g_ch, it.key);
        continue;
      }
      (pass ? nmt : nch)[(size_t)o]++;
      nb[(size_t)o] += it.buf.n;
      qwait += enqueue(std::move(it));
    }
  const double pages_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  for (size_t k = 0; k < keep.size(); k++) {
    Pending& pd = pend[keep[k]];
    Item it;
    it.kind = 2;
    it.ck = pd.ck;
    it.buf = std::move(pd.buf);
    it.cap_s = pd.cap_s + (k + 1 == keep.size() ? pages_s : 0.0);
    // log the new pages with the checkpoint that first needed them
    it.new_ch = nch[keep[k]];
    it.new_mt = nmt[keep[k]];
    it.new_b = nb[keep[k]];
    qwait += enqueue(std::move(it));
  }
  for (auto& pd : pend) buf_put(std::move(pd.buf));  // deduped images
  pend.clear();
  if (qwait >= 0.05)
    fprintf(stderr, "%s kvsnap: GPU thread waited %.2fs for the writer (queue cap %.0f MB)\n",
            log_ts(), qwait, g_qcap / 1048576.0);
}

// A vision request's state before its first image is keyed as text
// (seed_of), later states with its image seed: one pass per seed, since the
// page keys chain from the seed (pages below the first image may then be
// stored under both).
static void commit(GpuModel& m, const std::vector<int>& hist, std::vector<Pending>& pend) {
  std::vector<Pending> img;
  for (size_t i = 0; i < pend.size();) {
    if (pend[i].ck->seed != g_fp) {
      img.push_back(std::move(pend[i]));
      pend.erase(pend.begin() + (std::ptrdiff_t)i);
    } else {
      i++;
    }
  }
  commit_seed(m, hist, pend, g_fp);
  commit_seed(m, hist, img, seed_for(m.mrope_active_, m.img_hash_, m.mrope_grids_));
}

// ---- tier choice -------------------------------------------------------------
// serve tries the RAM tier (rckpt) before this one. A RAM hit used to win
// outright even when this tier held a far longer prefix: subagents keep
// refreshing the shared system-prompt checkpoint in RAM (8 entries) while the
// main conversation's long checkpoint survives only here, and switching back
// re-prefilled the difference (tens of seconds) instead of a restore of about
// a second. prefer_over_ram() decides from two running estimates, both under
// g_mtx: restore throughput (bytes/s of past restores; until one is seen,
// 2 GB/s on a local disk, 0.15 GB/s on a share) and prefill speed (tokens/s
// of past prompt evals of >= 1024 new tokens; 1000 until one is seen).
// QWENOX_KVSNAP_OVER_RAM=0: a RAM hit always wins.
static double g_rs_bps = 0;     // restore bytes/s (0: none measured yet)
static double g_rs_bps0 = 2e9;  // initial guess (init: 0.15e9 on a share)
static double g_pf_tps = 0;     // prefill tokens/s (0: none measured yet)

// A prompt eval of `n` new tokens took `ms` (serve, after the prefill).
static void note_prefill(size_t n, double ms) {
  if (!g_on || n < 1024 || ms <= 0) return;
  std::lock_guard<std::mutex> lk(g_mtx);
  const double tps = n / (ms / 1000.0);
  g_pf_tps = g_pf_tps > 0 ? 0.7 * g_pf_tps + 0.3 * tps : tps;
}
// Drop a match() that is not restored (its busy mark).
static void release(const std::shared_ptr<CkEnt>& e) {
  std::lock_guard<std::mutex> lk(g_mtx);
  e->busy--;
}
// Restore `e` rather than a RAM checkpoint of `ram_n` tokens? A RAM
// checkpoint that carries the MTP state wins over one here that does not
// (the turn would decode without speculation).
static bool prefer_over_ram(const CkEnt& e, size_t ram_n, bool ram_mtp, long long req) {
  static const bool on = !GpuModel::env_zero("QWENOX_KVSNAP_OVER_RAM");
  if (!on || (size_t)e.n <= ram_n || (ram_mtp && !e.mtp)) return false;
  double bytes = 0, bps, tps;
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    bytes = (double)e.bytes;
    for (uint64_t k : e.chunks) {
      auto f = g_ch.find(k);
      if (f != g_ch.end()) bytes += (double)f->second.bytes;
    }
    if (e.mtp)
      for (uint64_t k : e.mtps) {
        auto f = k ? g_mt.find(k) : g_mt.end();
        if (f != g_mt.end()) bytes += (double)f->second.bytes;
      }
    bps = g_rs_bps > 0 ? g_rs_bps : g_rs_bps0;
    tps = g_pf_tps > 0 ? g_pf_tps : 1000.0;
  }
  const double rs = bytes / bps, pf = (double)((size_t)e.n - ram_n) / tps;
  const bool take = rs + 0.25 < pf;  // 0.25 s: restore set-up, estimate slack
  fprintf(stderr, "%s kvsnap: req %lld: SSD checkpoint %d tokens vs RAM %zu: restore ~%.1fs "
          "(%.0f MB) vs prefill ~%.1fs -> %s\n", log_ts(), req, e.n, ram_n, rs,
          bytes / 1048576.0, pf, take ? "SSD" : "RAM");
  return take;
}

// ---- restore -----------------------------------------------------------------
// Longest published checkpoint whose tokens are a strict prefix of `prompt`
// (strictly shorter: a full-length match has no logits to resume from).
// `seed2` (optional): a second seed to search, e.g. g_fp for the text
// prefix of a vision prompt. The returned entry is marked busy until
// restore() releases it.
static std::shared_ptr<CkEnt> match(const std::vector<int>& prompt, uint64_t seed,
                                    uint64_t seed2 = 0) {
  if (!g_on || prompt.size() < 2) return nullptr;
  if (seed2 == seed) seed2 = 0;
  const int np = (int)((prompt.size() - 1) / KV_PAGE);
  const std::vector<uint64_t> ch = chain(seed, prompt.data(), np);
  const std::vector<uint64_t> ch2 =
      seed2 ? chain(seed2, prompt.data(), np) : std::vector<uint64_t>();
  std::lock_guard<std::mutex> lk(g_mtx);
  std::shared_ptr<CkEnt> best;
  for (auto& e : g_ck) {
    if (!e->pub || (e->seed != seed && (!seed2 || e->seed != seed2))) continue;
    const std::vector<uint64_t>& c = e->seed == seed ? ch : ch2;
    if ((size_t)e->n >= prompt.size() ||
        (best && e->n <= best->n) || e->nc > np || c[(size_t)e->nc] != e->base ||
        !std::equal(e->tail.begin(), e->tail.end(),
                    prompt.begin() + (size_t)e->nc * KV_PAGE))
      continue;
    best = e;
  }
  if (best) best->busy++;
  return best;
}

// Tokens of `prompt` in leading full pages that some cached sequence also
// holds (page-key chain): a page-granular lower bound on how far an earlier
// sequence shared this prompt. Fork detection after a restart, when no live
// slot or RAM checkpoint remembers that sequence.
static size_t shared_tokens(const std::vector<int>& prompt, uint64_t seed) {
  if (!g_on || prompt.size() < 2) return 0;
  const int np = (int)((prompt.size() - 1) / KV_PAGE);
  const std::vector<uint64_t> ch = chain(seed, prompt.data(), np);
  std::lock_guard<std::mutex> lk(g_mtx);
  int i = 0;
  while (i < np && g_ch.count(ch[(size_t)i + 1])) i++;
  return (size_t)i * KV_PAGE;
}

static double secs_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}
struct Rd {
  HBuf b;
  bool ok = false;
  double rd_s = 0, sum_s = 0;  // file read, checksum
};
// Read + verify one page / MTP section file (runs on a prefetch thread).
static Rd read_page(const std::string& name, bool mtp, uint64_t key, uint64_t parent,
                    int nl, int kvb, int has_ik, const int* toks, int look) {
  Rd r;
  const size_t len = sizeof(ChHdr) + (size_t)nl * page_layer_b(kvb, has_ik);
  auto t = std::chrono::steady_clock::now();
  const bool got = read_file(path_of(name), r.b, len);
  r.rd_s = secs_since(t);
  if (!got) return r;
  ChHdr h;
  memcpy(&h, r.b.data(), sizeof h);
  t = std::chrono::steady_clock::now();
  r.ok = !memcmp(h.magic, mtp ? MAG_M : MAG_C, 8) && h.key == key && h.parent == parent &&
         h.nl == nl && h.kvb == kvb && h.has_ik == has_ik && h.look == look &&
         !memcmp(h.toks, toks, sizeof h.toks) &&
         h.sum == sum64(r.b.data() + sizeof h, len - sizeof h);
  r.sum_s = secs_since(t);
  return r;
}
// Where a restore's time went. Page reads + checksums run on the prefetch
// thread, overlapped with the device copies; io_wait is the GPU thread's
// share of them (waiting for the next page).
struct RsTime {
  double ck_s = 0, rd_s = 0, sum_s = 0, wait_s = 0, h2d_s = 0;
  uint64_t bytes = 0;
};

enum { RS_OK, RS_CKPT, RS_PAGE, RS_OTHER };
// Pages stream in on a prefetch thread (read + checksum) while the previous
// one is copied to the device. *bad_page: the trunk page that failed.
static int restore_body(GpuModel& m, const CkEnt& e, const std::vector<int>& prompt,
                        bool* mtp_out, uint64_t* bad_page, int* npages, RsTime& tm) {
  const Geo G = geo(m);
  HBuf cb;
  const auto tc = std::chrono::steady_clock::now();
  if (!read_file(path_of(e.name), cb, 0, (size_t)1 << 40)) return RS_CKPT;
  struct Ret {  // pooled buffer back on every exit
    HBuf& b;
    ~Ret() { buf_put(std::move(b)); }
  } ret{cb};
  CkHdr h;
  if (cb.n < sizeof h) return RS_CKPT;
  memcpy(&h, cb.data(), sizeof h);
  if (!ck_sane(h) || cb.n != sizeof h + ck_body_b(h) || h.id != e.id || h.n != e.n ||
      h.seed != e.seed || h.base != e.base ||
      h.sum != sum64(cb.data() + sizeof h, cb.n - sizeof h))
    return RS_CKPT;
  tm.ck_s = secs_since(tc);
  tm.bytes = cb.n;
  const int n = h.n, nc = h.nc, nt = n - nc * KV_PAGE;
  if (n > m.maxctx || h.kvb != G.kvb || h.has_ik != G.has_ik || h.s_b != G.s_b ||
      h.conv_b != G.conv_b || h.ring_b != G.ring_b || h.iraw_b != G.iraw_b ||
      h.nl < G.nqsa || h.nl > G.nqsa + 1 || (h.nl > G.nqsa && !h.mtp))
    return RS_OTHER;
  const char* p = cb.data() + sizeof h;
  std::vector<uint64_t> chunks((size_t)nc), mtps((size_t)nc);
  memcpy(chunks.data(), p, (size_t)nc * 8);
  memcpy(mtps.data(), p + (size_t)nc * 8, (size_t)nc * 8);
  p += (size_t)nc * 16;
  if (memcmp(p, prompt.data() + (size_t)nc * KV_PAGE, (size_t)nt * 4)) return RS_CKPT;
  p += (size_t)nt * 4;
  bool want_mtp = h.mtp && h.nl == G.nqsa + 1 && m.kv_nl > G.nqsa &&
                  h.mtp_pos == n - 1 &&
                  h.mR_b == (uint64_t)g_cfg.branches * g_cfg.d * 4 &&
                  h.miraw_b == (m.d_miraw ? 4 * 128 * 4u : 0u);
  if (want_mtp) {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (uint64_t k : mtps) {
      auto f = g_mt.find(k);
      if (!k || f == g_mt.end() || f->second.st != ST_PUB) want_mtp = false;
    }
  }
  // the restored state replaces the live sequence: map fresh rows [0, n)
  m.kv_trim(0);
  m.kv_reserve(n);
  const std::vector<uint64_t> ch = chain(e.seed, prompt.data(), nc);
  const int njobs = nc + (want_mtp ? nc : 0);
  auto job = [&](int j) {
    const bool mt = j >= nc;
    const int i = mt ? j - nc : j;
    const int* toks = prompt.data() + (size_t)i * KV_PAGE;
    return std::async(std::launch::async, read_page,
                      mt ? mt_name(mtps[(size_t)i]) : ch_name(chunks[(size_t)i]), mt,
                      mt ? mtps[(size_t)i] : chunks[(size_t)i],
                      mt ? chunks[(size_t)i] : ch[(size_t)i], mt ? 1 : G.nqsa, G.kvb,
                      G.has_ik, toks, mt ? prompt[(size_t)(i + 1) * KV_PAGE] : -1);
  };
  // g_readers pages in flight: a network share is latency bound (one file
  // at a time used a fraction of the link); pages are consumed in order.
  std::deque<std::future<Rd>> win;
  int next = 0;
  auto refill = [&] {
    while (next < njobs && (int)win.size() < g_readers) win.push_back(job(next++));
  };
  auto drain = [&] {  // early exit: let in-flight reads finish (their buffers die)
    for (auto& f : win)
      if (f.valid()) f.wait();
    win.clear();
  };
  refill();
  const size_t lb = page_layer_b(G.kvb, G.has_ik);
  for (int j = 0; j < njobs; j++) {
    const auto tw = std::chrono::steady_clock::now();
    Rd r = win.front().get();
    win.pop_front();
    tm.wait_s += secs_since(tw);
    tm.rd_s += r.rd_s;
    tm.sum_s += r.sum_s;
    if (r.ok) tm.bytes += r.b.n;
    refill();
    const bool mt = j >= nc;
    const int i = mt ? j - nc : j;
    if (!mt && chunks[(size_t)i] != ch[(size_t)i + 1]) r.ok = false;
    if (!r.ok) {
      buf_put(std::move(r.b));
      drain();
      if (!mt) {
        *bad_page = chunks[(size_t)i];
        return RS_PAGE;
      }
      want_mtp = false;  // the trunk is complete; only speculation is lost
      break;             // (win was drained above)
    }
    char* q = r.b.data() + sizeof(ChHdr);
    const int l0 = mt ? G.nqsa : 0, nl = mt ? 1 : G.nqsa;
    const int r0 = i * KV_PAGE, k0 = i * KV_PAGE_BLOCKS;
    bool ok = true;
    const auto th = std::chrono::steady_clock::now();
    for (int l = l0; l < l0 + nl && ok; l++, q += lb)
      ok = xfer(m, true, kv_layer(m, 0, l), q, r0, r0 + KV_PAGE, KV_PAGE, G.row_b) &&
           xfer(m, true, kv_layer(m, 1, l), q + (size_t)KV_PAGE * G.row_b, r0,
                r0 + KV_PAGE, KV_PAGE, G.row_b) &&
           (!G.has_ik || xfer(m, true, ik_layer(m, l), q + 2 * (size_t)KV_PAGE * G.row_b,
                              k0, k0 + KV_PAGE_BLOCKS, KV_PAGE_BLOCKS, 512));
    tm.h2d_s += secs_since(th);
    buf_put(std::move(r.b));
    if (!ok) {
      drain();
      return RS_OTHER;
    }
  }
  *npages = nc;
  // checkpoint: GDN/PLE/indexer state, then the partial page of every layer
  char* src = (char*)p;
  bool ok = true;
  const auto th = std::chrono::steady_clock::now();
  struct Lap {  // checkpoint H2D, every exit below
    RsTime& t;
    std::chrono::steady_clock::time_point t0;
    ~Lap() { t.h2d_s += secs_since(t0); }
  } lap{tm, th};
  auto blob = [&](void* d, size_t sz, bool use) {
    if (ok && use) ok = h2d(d, src, sz);
    src += sz;
  };
  blob(m.d_S, h.s_b, true);
  blob(m.d_convst, h.conv_b, true);
  blob(m.d_ring, h.ring_b, true);
  blob(m.d_iraw, h.iraw_b, true);
  blob(m.d_R, h.mR_b, want_mtp);
  blob(m.d_miraw, h.miraw_b, want_mtp);
  const int r0 = nc * KV_PAGE, k0 = nc * KV_PAGE_BLOCKS;
  const int nik = std::min(
      h.nik, std::max(0, std::min(KV_PAGE_BLOCKS * ((n + KV_PAGE - 1) / KV_PAGE),
                                  m.index_blocks) - k0));
  const int nl = want_mtp ? h.nl : G.nqsa;
  for (int l = 0; l < nl && ok; l++) {
    ok = xfer(m, true, kv_layer(m, 0, l), src, r0, n, KV_PAGE, G.row_b) &&
         xfer(m, true, kv_layer(m, 1, l), src + (size_t)nt * G.row_b, r0, n, KV_PAGE,
              G.row_b) &&
         (!G.has_ik || xfer(m, true, ik_layer(m, l), src + 2 * (size_t)nt * G.row_b, k0,
                            k0 + nik, KV_PAGE_BLOCKS, 512));
    src += 2 * (size_t)nt * G.row_b + (size_t)h.nik * 512;
  }
  if (!ok) return RS_OTHER;
  if (want_mtp) {
    m.mtp_pos = h.mtp_pos;
    CK(hipMemcpy(m.d_mpos, &m.mtp_pos, 4, hipMemcpyHostToDevice));
  }
  // QWENOX_QSA_WMMA_BTV: the transposed prefill V cache is derived state (not
  // saved); rebuild it for [0, n) from the restored row-major V, or sparse
  // prefill after the restore would read stale V for the prefix. Paged:
  // per physically contiguous run; a page holds whole 4-token blocks, so
  // physical row block pr/4 is exactly the physical vct block of that run.
  if (m.qsa_wmma_btv) {
    for (int qi = 0; qi < G.nqsa; qi++)
      m.kv_runs(0, n, KV_PAGE, [&](int, size_t prz, int cnt) {
        const int pr = (int)prz;
        const unsigned grid =
            (unsigned)((((size_t)(pr + cnt + 3) / 4 - (size_t)pr / 4) * 512 + 255) / 256);
        k_bf16_rows_to_bt<<<grid, 256, 0, g_str>>>(
            m.d_vcb + (size_t)qi * m.kv_layer_rows * 512, m.vct_layer(qi), pr, pr + cnt);
      });
    CK(hipGetLastError());
    CK(hipStreamSynchronize(g_str));
  }
  // C1 f16 twins: derived state (not saved); rebuild the complete blocks
  // [0, n/4) from the restored fp32 indexer keys, or f16 scoring in a later
  // prefill would read stale twins for the prefix. The straddling block is
  // re-pooled from the raw ring by the continuation prefill.
  if (m.d_ikh && G.has_ik && n / 4 > 0) {
    const int nb = n / 4;
    for (int qi = 0; qi < m.kv_nl; qi++)
      k_ik_twin_build<<<nb, 128, 0, g_str>>>(
          m.d_ik + (size_t)qi * m.ik_layer_blocks * 128,
          m.kv_paged ? m.d_ptab : nullptr,
          m.d_ikh + (size_t)qi * m.ik_layer_blocks * 128, nb);
    CK(hipGetLastError());
    CK(hipStreamSynchronize(g_str));
  }
  // host-side and scalar state: pos/ringpos (both mirrors), PLE token history
  m.pos = n;
  m.ple_ringpos = h.ringpos;
  CK(hipMemcpy(m.d_pos, &m.pos, 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(m.d_ringpos, &m.ple_ringpos, 4, hipMemcpyHostToDevice));
  m.chist.assign(prompt.begin(), prompt.begin() + n);
  *mtp_out = want_mtp;
  return RS_OK;
}

// Restore `e` (from match()) for `prompt`: hist := its tokens. *mtp_out: the
// MTP layer is back in lockstep (speculation may continue). On failure the
// live state is undefined: the caller resets it.
static bool restore(GpuModel& m, const std::shared_ptr<CkEnt>& e,
                    const std::vector<int>& prompt, std::vector<int>& hist,
                    bool* mtp_out) {
  *mtp_out = false;
  crashdiag::note("kvsnap restore %d tokens from %.60s", e->n, e->name.c_str());
  const auto t0 = std::chrono::steady_clock::now();
  uint64_t bad = 0;
  int npages = 0;
  RsTime tm;
  const int rc = restore_body(m, *e, prompt, mtp_out, &bad, &npages, tm);
  {
    std::lock_guard<std::mutex> lk(g_mtx);
    e->busy--;
    if (rc == RS_OK) {
      hit_locked(e);
    } else if (rc == RS_CKPT) {
      drop_ck_locked(e, true);
    } else if (rc == RS_PAGE) {
      // a damaged page invalidates every checkpoint built on it
      std::vector<std::shared_ptr<CkEnt>> v;
      for (auto& o : g_ck)
        if (o->pub && !o->busy && std::count(o->chunks.begin(), o->chunks.end(), bad))
          v.push_back(o);
      for (auto& o : v) drop_ck_locked(o, true);
    }
  }
  flush_log();
  const double s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (rc != RS_OK) {
    fprintf(stderr, "%s kvsnap: restore of %s failed (%s, non-fatal)\n", log_ts(),
            e->name.c_str(),
            rc == RS_CKPT ? "bad checkpoint" : rc == RS_PAGE ? "bad page" : "device/geometry");
    return false;
  }
  hist.assign(prompt.begin(), prompt.begin() + e->n);
  if (tm.bytes >= (64ull << 20) && s > 0) {  // restore throughput (prefer_over_ram)
    std::lock_guard<std::mutex> lk(g_mtx);
    const double bps = tm.bytes / s;
    g_rs_bps = g_rs_bps > 0 ? 0.7 * g_rs_bps + 0.3 * bps : bps;
  }
  // ckpt = checkpoint read + verify; pages read/verify on the prefetch thread;
  // io wait = GPU thread waiting for pages; device = H2D + derived-state rebuild
  fprintf(stderr,
          "%s kvsnap: restored %d tokens from %s in %.2fs (%d pages, %.0f MB, %s IO: "
          "ckpt %.2fs, pages read %.2fs + verify %.2fs, io wait %.2fs, device %.2fs)%s\n",
          log_ts(), e->n, e->name.c_str(), s, npages, tm.bytes / 1048576.0,
          g_direct.load() ? "direct" : "buffered", tm.ck_s, tm.rd_s, tm.sum_s, tm.wait_s,
          tm.h2d_s, *mtp_out ? " (mtp live)" : "");
  crashdiag::note("kvsnap restored %d tokens in %.2fs", e->n, s);
  return true;
}

// ---- startup -----------------------------------------------------------------
static uint64_t fingerprint(const GpuModel& m) {
  const Geo G = geo(m);
  uint64_t h = mix64(0x6764656b76736e70ULL + FMT);
  auto add = [&](uint64_t v) { h = mix64(h ^ v); };
  auto bits = [](double v) {
    uint64_t u;
    memcpy(&u, &v, sizeof(u));
    return u;
  };
  for (const auto& f : g_model_files) add(file_id(f));
  for (uint64_t v : {(uint64_t)m.ple_on, (uint64_t)G.kvb, (uint64_t)G.has_ik,
                     (uint64_t)G.nqsa, (uint64_t)g_cfg.layers, (uint64_t)g_cfg.d,
                     (uint64_t)g_cfg.branches, (uint64_t)KV_PAGE,
                     bits(g_cfg.rope_theta), (uint64_t)g_cfg.rotary_dim,
                     bits(g_cfg.rope.factor),
                     (uint64_t)g_cfg.rope.original_max_position_embeddings,
                     bits(g_cfg.rope.beta_fast), bits(g_cfg.rope.beta_slow),
                     bits(g_cfg.rope.attention_factor)})
    add(v);
  // GGUF native dense weights (not the q8g32 requant): different numerics,
  // so snapshots of the two layouts never mix. Other setups keep their hash.
  if (g_gguf_dense && gguf_map::g_native_dense) add(0x4e41544956453031ULL);
  return h;
}
static bool ends_with(const std::string& s, const char* suf) {
  const size_t k = strlen(suf);
  return s.size() >= k && !s.compare(s.size() - k, k, suf);
}
// A name this cache writes (c_*.kvc, m_*.kvm, k_*.kck). Startup cleanup only
// ever touches these, their tmp_ temporaries and pre-A4 s_* snapshot
// directories, so a cache dir that is shared with other files is safe.
static bool own_name(const std::string& fn) {
  return (!fn.rfind("c_", 0) && ends_with(fn, ".kvc")) ||
         (!fn.rfind("m_", 0) && ends_with(fn, ".kvm")) ||
         (!fn.rfind("k_", 0) && ends_with(fn, ".kck"));
}
// Index entry of a checkpoint: one open; a 64 KiB first read holds the
// header and the page-key/tail lists of any common length.
static std::shared_ptr<CkEnt> load_ck(const std::string& fn) {
  const int fd = os_open_rd(path_of(fn).c_str());
  if (fd < 0) return nullptr;
  struct Fd {
    int fd;
    ~Fd() { os_close(fd); }
  } guard{fd};
  const int64_t fsz = os_fsize(fd);
  if (fsz < (int64_t)sizeof(CkHdr)) return nullptr;
  std::vector<char> v((size_t)std::min<int64_t>(fsz, (int64_t)64 << 10));
  if (!rd_all(fd, v.data(), v.size())) return nullptr;
  CkHdr h;
  memcpy(&h, v.data(), sizeof h);
  if (!ck_sane(h) || (uint64_t)fsz != sizeof h + ck_body_b(h) || fn != ck_name(h.n, h.id))
    return nullptr;
  const size_t nt = (size_t)h.n - (size_t)h.nc * KV_PAGE;
  const size_t need = sizeof h + (size_t)h.nc * 16 + nt * 4;
  if (need > v.size()) {
    const size_t have = v.size();
    v.resize(need);
    if (!rd_all(fd, v.data() + have, need - have)) return nullptr;
  }
  auto e = std::make_shared<CkEnt>();
  e->name = fn;
  e->seed = h.seed;
  e->base = h.base;
  e->id = h.id;
  e->bytes = (uint64_t)fsz;
  e->n = h.n;
  e->nc = h.nc;
  e->nl = h.nl;
  e->mtp = h.mtp != 0;
  e->pub = true;
  e->chunks.resize((size_t)h.nc);
  e->mtps.resize((size_t)h.nc);
  e->tail.resize(nt);
  memcpy(e->chunks.data(), v.data() + sizeof h, (size_t)h.nc * 8);
  memcpy(e->mtps.data(), v.data() + sizeof h + (size_t)h.nc * 8, (size_t)h.nc * 8);
  memcpy(e->tail.data(), v.data() + sizeof h + (size_t)h.nc * 16, nt * 4);
  return e;
}

// Index the cache directory (serve::run, before the first request). Torn
// temporaries, pre-A4 snapshot directories, unreadable checkpoints and
// checkpoints missing a trunk page are removed; unreferenced pages too.
// Cost is per file on a share, so pages are indexed from the directory
// listing alone (name -> key, listed size): their header and checksum are
// verified when a restore reads them, and a bad one then drops the
// checkpoints built on it. Checkpoints are read (header + key lists) by
// several threads. Bulk deletes go to the janitor.
static void init(GpuModel& m, Cfg& cfg) {
  namespace fs = std::filesystem;
  const auto ti = std::chrono::steady_clock::now();
  std::error_code ec;
  fs::create_directories(cfg.dir, ec);
  if (!fs::is_directory(cfg.dir, ec)) {
    fprintf(stderr, "kvsnap: cannot use %s: disabled\n", cfg.dir.c_str());
    cfg.dir.clear();
    return;
  }
  g_dir = cfg.dir;
  g_cap = cfg.cap_bytes;
  g_qcap = cfg.queue_bytes;
  const bool remote = os_path_is_remote(cfg.dir.c_str());
  g_readers = std::max(1, std::min(cfg.readers, 32));
  g_writers = std::max(1, std::min(cfg.writers > 0 ? cfg.writers : remote ? 4 : 1, 32));
  g_direct = cfg.direct;
  g_rs_bps0 = remote ? 0.15e9 : 2e9;
  g_hot_hits = std::max(0, cfg.hot_hits);
  g_hot_frac = std::max(0.0, std::min(1.0, cfg.hot_frac));
  g_fp = fingerprint(m);
  g_vis_id = g_vision_file.empty() ? 0 : file_id(g_vision_file);
  struct Ck {
    std::string fn;
    int64_t t;
    std::shared_ptr<CkEnt> e;
  };
  std::vector<Ck> cks;
  size_t legacy = 0, bad = 0;
  // directory_entry carries the listing's size/mtime on Windows (no stat)
  for (fs::directory_iterator it(cfg.dir, ec), end; !ec && it != end; it.increment(ec)) {
    const fs::directory_entry& de = *it;
    std::error_code e2;
    const std::string fn = de.path().filename().string();
    if (!fn.rfind("tmp_", 0)) {
      if ((own_name(fn.substr(4)) || fn.substr(4) == HITS_FILE) && de.is_regular_file(e2))
        fs::remove(de.path(), e2);
    } else if (!fn.rfind("s_", 0) && de.is_directory(e2) &&
               fs::is_regular_file(de.path() / "meta", e2)) {
      fs::remove_all(de.path(), e2);  // KVS1-3 whole-state snapshots
      legacy++;
    } else if ((!fn.rfind("c_", 0) && ends_with(fn, ".kvc")) ||
               (!fn.rfind("m_", 0) && ends_with(fn, ".kvm"))) {
      const bool mt = fn[0] == 'm';
      const uint64_t key =
          fn.size() == 22 ? strtoull(fn.substr(2, 16).c_str(), nullptr, 16) : 0;
      const uint64_t fsz = de.file_size(e2);
      if (e2 || fsz <= sizeof(ChHdr) || fn != (mt ? mt_name(key) : ch_name(key))) {
        fs::remove(de.path(), e2);
        bad++;
        continue;
      }
      Blk& b = (mt ? g_mt : g_ch)[key];
      b.bytes = fsz;
      b.st = ST_PUB;
    } else if (!fn.rfind("k_", 0) && ends_with(fn, ".kck")) {
      const int64_t t = (int64_t)de.last_write_time(e2).time_since_epoch().count();
      cks.push_back({fn, t, nullptr});
    }
  }
  const double list_s = secs_since(ti);
  const auto tc = std::chrono::steady_clock::now();
  {
    std::atomic<size_t> next{0};
    const size_t nth = std::min<size_t>(cks.size(), remote ? 16 : 4);
    std::vector<std::thread> th;
    for (size_t i = 0; i < nth; i++)
      th.emplace_back([&] {
        for (size_t j; (j = next++) < cks.size();) cks[j].e = load_ck(cks[j].fn);
      });
    for (auto& t : th) t.join();
  }
  const double ck_s = secs_since(tc);
  const std::unordered_map<std::string, int> hits = read_hits();
  std::unique_lock<std::mutex> lk(g_mtx);
  for (auto& c : cks) {
    if (!c.e) {
      std::error_code e2;
      fs::remove(path_of(c.fn), e2);
      bad++;
      continue;
    }
    const auto& e = c.e;
    e->t = c.t;
    if (auto h = hits.find(e->name); h != hits.end()) {
      e->hits = h->second;
      e->hot = g_hot_hits > 0 && e->hits >= g_hot_hits;
    }
    bool whole = true;
    for (uint64_t k : e->chunks) whole = whole && g_ch.count(k);
    if (!whole) {
      del_locked(e->name);
      bad++;
      continue;
    }
    for (uint64_t k : e->chunks) g_ch[k].refs++;
    for (uint64_t& k : e->mtps) {
      auto f = g_mt.find(k);
      if (f == g_mt.end()) k = 0;
      else f->second.refs++;
    }
    g_total += e->bytes;
    g_ck.push_back(e);
  }
  size_t orphans = 0;
  for (auto* mp : {&g_ch, &g_mt})
    for (auto it = mp->begin(); it != mp->end();) {
      if (it->second.refs == 0) {
        del_locked(mp == &g_mt ? mt_name(it->first) : ch_name(it->first));
        it = mp->erase(it);
        orphans++;
      } else {
        g_total += it->second.bytes;
        ++it;
      }
    }
  evict_locked(nullptr);
  size_t nhot = 0;
  for (auto& e : g_ck) nhot += e->hot;
  lk.unlock();  // writer not started, no request yet: the index is ours alone
  flush_log();
  if (legacy)
    fprintf(stderr, "kvsnap: removed %zu pre-A4 snapshot director%s (old format)\n",
            legacy, legacy == 1 ? "y" : "ies");
  if (bad || orphans)
    fprintf(stderr, "kvsnap: removing %zu unusable and %zu unreferenced files\n", bad,
            orphans);
  fprintf(stderr, "kvsnap: index %zu checkpoints (%zu hot), %zu pages, %zu mtp sections, "
          "%.2f GB (cap %.1f GB, queue %.0f MB, %s, %d readers, %d writers; scanned in %.2fs: "
          "listing %.2fs, %zu checkpoint headers %.2fs)\n",
          g_ck.size(), nhot, g_ch.size(), g_mt.size(), g_total / 1073741824.0,
          g_cap / 1073741824.0, g_qcap / 1048576.0, remote ? "network" : "local",
          g_readers, g_writers, secs_since(ti), list_s, cks.size(), ck_s);
  g_on = true;
  std::thread(janitor).detach();
  for (int i = 0; i < g_writers; i++) std::thread(writer).detach();
}

}  // namespace kvsnap

namespace serve {

struct LineBuf {
  sockfd_t fd;
  std::string buf;
  bool take(std::string& out) {
    size_t nl = buf.find('\n');
    if (nl == std::string::npos) return false;
    out = buf.substr(0, nl);
    buf.erase(0, nl + 1);
    return true;
  }
  // blocking read of one line; false on EOF/error
  bool readline(std::string& out) {
    for (;;) {
      if (take(out)) return true;
      char tmp[65536];
      ssize_t n = sock_recv(fd, tmp, sizeof tmp, false);
      if (n <= 0) return false;
      buf.append(tmp, n);
    }
  }
  // non-blocking: true + line iff a complete line is available right now
  bool try_readline(std::string& out) {
    for (;;) {
      if (take(out)) return true;
      char tmp[65536];
      ssize_t n = sock_recv(fd, tmp, sizeof tmp, true);
      if (n <= 0) return false;  // EAGAIN (or EOF: surfaced by next readline)
      buf.append(tmp, n);
    }
  }
  // Blocking read of exactly n bytes (binary frames after a GEN line).
  // Bytes already swallowed into the line buffer are drained first, so the
  // client may pipeline frame bytes right behind the newline. False on
  // EOF/error/timeout (SO_RCVTIMEO, armed by the caller for bulk reads).
  bool read_full(void* dst, size_t n) {
    char* p = (char*)dst;
    size_t head = std::min(n, buf.size());
    if (head) {
      memcpy(p, buf.data(), head);
      buf.erase(0, head);
      p += head;
      n -= head;
    }
    while (n > 0) {
      ssize_t r = sock_recv(fd, p, n, false);
      if (r <= 0) return false;
      p += r;
      n -= (size_t)r;
    }
    return true;
  }
};

static bool send_all(sockfd_t fd, const std::string& s) {
  size_t off = 0;
  while (off < s.size()) {
    ssize_t n = sock_send(fd, s.data() + off, s.size() - off);
    if (n <= 0) return false;
    off += n;
  }
  return true;
}

// VIMG frames (and MROPE grids) accepted per GEN line. This is only a
// protocol sanity ceiling, not the image policy: the per-request image cap is
// the user's MAX_IMAGES (service.conf, range 1–256), enforced by the API
// (src/api/vision.cpp, same 256 ceiling), so every configurable value is
// accepted here. It used to be 16 while the config allowed 256, which turned
// MAX_IMAGES>16 into a silent clamp. parse_gen keeps the parsed count even
// when it rejects the value, so the serve loop can still drain that many
// binary frames before replying — a reject that skips the drain leaves frame
// bytes to be misread as protocol lines and silently wedges the connection
// (73159c9).
constexpr int kVimgMax = 256;

struct GenReq {
  int64_t req = 0;
  int max_tokens = 0;
  std::vector<int> eos;
  std::vector<int> ids;
  int drafter = -1;  // omitted: engine default; 0 serial; 1 MTP; 3 ngram-mod; 4 chain
  bool reject = false;  // malformed or unsupported optional field
  const char* reject_why = nullptr;  // parse note for the serve log
  // Sampler (SAMPLE suffix). sample=true implies temp>0; greedy otherwise.
  bool sample = false;
  float temp = 1.0f, top_p = 0.0f, min_p = 0.0f;  int top_k = 0;
  uint64_t seed = 0;
  // SEEDED: the client gave this seed explicitly (the API otherwise draws a
  // random one). Seeded sampled requests pick their MTP γ from a fixed
  // occupancy (PARALLEL) so the output is reproducible under concurrency;
  // unseeded ones follow the live rendezvous size (Model::OccGamma).
  bool seeded = false;
  // PENALTY (presence, frequency) / BIAS (tid,val pairs) / LOGPROBS: sampler-only.
  float presence = 0.0f, frequency = 0.0f;
  std::vector<std::pair<int, float>> bias;
  bool logprobs = false;
  // SNAP / SNAP2: frontend-hinted prefix lengths worth an SSD snapshot (the
  // next turn repeats them verbatim; see serve_api.py render_prompt).
  // SNAPS: same idea as a counted list (semantic message boundaries); the
  // engine keeps cheap RAM checkpoints (rckpt) at these cuts.
  size_t snap = 0, snap2 = 0;
  std::vector<size_t> snaps;
  // MROPE <k> <t> <h> <w> ...: per-image grid triples, expanded in-engine
  // (stage 1: fresh prefill + serial decode only; cont/spec are rejected).
  std::vector<std::array<int, 3>> mrope_grids;
  // VIMG <k>: k binary patch frames follow the GEN line (one per MROPE grid,
  // same order; stage 4). Frame: u32 magic 0x56494D31, u32 P, u64 nbytes,
  // then P*1536 fp32 LE patches in block-major (C,T,ph,pw) order.
  int vimg = 0;
  // CKPT <n> <ids...>: generated tokens worth a mid-decode RAM checkpoint
  // (rckpt), e.g. <tool_call> / </think>. The API's token cache can only
  // splice right after such added tokens; without a checkpoint there a
  // retokenization mismatch inside a tool call forces a re-prefill back to
  // the previous prompt-end cut.
  std::vector<int> ckpt;
};



// Host-side sampler: logits come back over unified memory (~1 MB), all filtering
// is O(vocab) or less. Filter order follows HF warpers: penalties/bias on raw
// logits, then temperature, top_k, top_p, min_p. top_p uses a binary-search
// threshold (mass(p >= t) >= top_p) instead of a full sort; ties keep all tied
// tokens, unlike sort-and-cut. RNG is splitmix64: deterministic per
// (seed, emitted sequence).
struct HostSampler {
  float temp, top_p, min_p;
  int top_k;
  uint64_t rng;
  float presence, frequency;
  std::vector<std::pair<int, float>> bias;
  std::unordered_map<int, int> hist;  // emitted token counts (not the prompt)
  std::vector<int> cand;              // scratch: candidate token ids
  std::vector<float> work;            // scratch: top-k preservation / residual

  static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }

  uint64_t next64() {  // splitmix64
    return mix64(rng += 0x9E3779B97F4A7C15ull);
  }

  double uniform() {
    return (double)(next64() >> 11) * (1.0 / 9007199254740992.0);
  }

  HostSampler fork_stream(uint64_t tag) const {
    HostSampler out = *this;
    out.rng = mix64(rng ^ tag);
    out.cand.clear();
    out.work.clear();
    return out;
  }

  void remember(int id) { hist[id]++; }

  // Convert raw logits in place to the normalized distribution after the
  // engine's penalty/bias -> temperature -> top-k -> top-p -> min-p pipeline.
  void prepare(std::vector<float>& logits) {
    const int n = (int)logits.size();
    for (auto& b : bias)
      if (b.first >= 0 && b.first < n) logits[b.first] += b.second;
    if (presence != 0.0f || frequency != 0.0f)
      for (auto& h : hist)
        if (h.first >= 0 && h.first < n)
          logits[h.first] -= presence + frequency * (float)h.second;
    float lmax = -INFINITY;
    for (int i = 0; i < n; i++) lmax = fmaxf(lmax, logits[i]);
    // unnormalized probs after temperature
    for (int i = 0; i < n; i++) logits[i] = expf((logits[i] - lmax) / temp);
    cand.clear();
    if (top_k > 0 && top_k < n) {
      cand.resize(n);
      for (int i = 0; i < n; i++) cand[i] = i;
      std::nth_element(cand.begin(), cand.begin() + top_k, cand.end(),
                       [&](int a, int b) { return logits[a] > logits[b]; });
      cand.resize(top_k);
      work.resize(cand.size());
      for (size_t j = 0; j < cand.size(); j++) work[j] = logits[cand[j]];
      std::fill(logits.begin(), logits.end(), 0.0f);
      for (size_t j = 0; j < cand.size(); j++) logits[cand[j]] = work[j];
    } else {
      cand.reserve(n);
      for (int i = 0; i < n; i++)
        if (logits[i] > 0.0f) cand.push_back(i);
    }
    if (top_p > 0.0f && top_p < 1.0f && !cand.empty()) {
      // largest threshold t with mass(p >= t) >= top_p * total
      double total = 0;
      for (int i : cand) total += logits[i];
      double lo = 0, hi = 1.0;  // probs are scaled so max == 1
      for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi), m = 0;
        for (int i : cand)
          if (logits[i] >= (float)mid) m += logits[i];
        if (m >= top_p * total) lo = mid; else hi = mid;
      }
      std::vector<int> keep;
      keep.reserve(cand.size());
      for (int i : cand)
        if (logits[i] >= (float)lo)
          keep.push_back(i);
        else
          logits[i] = 0.0f;
      cand.swap(keep);
    }
    if (min_p > 0.0f && !cand.empty()) {
      float pmax = 0;
      for (int i : cand) pmax = fmaxf(pmax, logits[i]);
      float thr = min_p * pmax;
      std::vector<int> keep;
      keep.reserve(cand.size());
      for (int i : cand)
        if (logits[i] >= thr) keep.push_back(i);
      if (!keep.empty()) {
        for (int i : cand)
          if (logits[i] < thr) logits[i] = 0.0f;
        cand.swap(keep);  // never empty the candidate set
      }
    }
    double z = 0;
    for (int i : cand) z += logits[i];
    for (int i : cand) logits[i] = (float)(logits[i] / z);
  }

  // Draw from a distribution produced by prepare(). Candidate order is kept
  // stable with the serial sampler so existing seeded behavior is unchanged.
  int draw_prepared(const std::vector<float>& probs, float* lp,
                    bool add_hist) {
    double z = 0;
    for (int i : cand) z += probs[i];
    double u = uniform() * z;
    int sel = cand.back();
    for (int i : cand) {
      u -= probs[i];
      if (u <= 0) { sel = i; break; }
    }
    if (lp) *lp = (float)(log((double)probs[sel]) - log(z));
    if (add_hist) remember(sel);
    return sel;
  }

  int sample(std::vector<float>& logits, float* lp) {
    prepare(logits);
    return draw_prepared(logits, lp, true);
  }

  // Standard speculative correction distribution: normalize max(p-q, 0).
  // The reported logprob remains the target probability p(token), matching
  // serial LOGPROBS semantics rather than the residual draw probability.
  int sample_residual(const std::vector<float>& p,
                      const std::vector<float>& q, float* lp) {
    const int n = (int)p.size();
    work.assign(n, 0.0f);
    cand.clear();
    double z = 0.0;
    for (int i = 0; i < n; i++) {
      float d = fmaxf(0.0f, p[i] - q[i]);
      if (d > 0.0f) {
        work[i] = d;
        cand.push_back(i);
        z += d;
      }
    }
    if (cand.empty() || z <= 0.0) {
      for (int i = 0; i < n; i++)
        if (p[i] > 0.0f) {
          work[i] = p[i];
          cand.push_back(i);
          z += p[i];
        }
    }
    double u = uniform() * z;
    int sel = cand.back();
    for (int i : cand) {
      u -= work[i];
      if (u <= 0.0) {
        sel = i;
        break;
      }
    }
    if (lp) *lp = p[sel] > 0.0f ? logf(p[sel]) : -INFINITY;
    return sel;
  }

  // ---- sparse candidates (GPU top-k path) ----
  // SDist is the sparse mirror of a prepare()d row: the candidate ids in draw
  // scan order and their normalized probabilities. prepare_sparse() replays
  // the post-top_k filter chain (temperature -> top-p -> min-p -> normalize)
  // on exactly the top-k corrected logits supplied by the GPU.
  struct SDist {
    std::vector<int> ids;
    std::vector<float> pr;
  };

  void prepare_sparse(const int* ids, const float* vals, int k, SDist& out) {
    float lmax = -INFINITY;
    for (int j = 0; j < k; j++) lmax = fmaxf(lmax, vals[j]);
    cand.assign(ids, ids + k);
    work.resize(k);
    for (int j = 0; j < k; j++) work[j] = expf((vals[j] - lmax) / temp);
    if (top_p > 0.0f && top_p < 1.0f && !cand.empty()) {
      double total = 0;
      for (float w : work) total += w;
      double lo = 0, hi = 1.0;  // probs are scaled so max == 1
      for (int it = 0; it < 50; it++) {
        double mid = 0.5 * (lo + hi), m = 0;
        for (float w : work)
          if (w >= (float)mid) m += w;
        if (m >= top_p * total) lo = mid; else hi = mid;
      }
      std::vector<int> keep;
      std::vector<float> kwork;
      keep.reserve(cand.size());
      kwork.reserve(cand.size());
      for (size_t j = 0; j < cand.size(); j++)
        if (work[j] >= (float)lo) {
          keep.push_back(cand[j]);
          kwork.push_back(work[j]);
        }
      cand.swap(keep);
      work.swap(kwork);
    }
    if (min_p > 0.0f && !cand.empty()) {
      float pmax = 0;
      for (float w : work) pmax = fmaxf(pmax, w);
      float thr = min_p * pmax;
      std::vector<int> keep;
      std::vector<float> kwork;
      for (size_t j = 0; j < cand.size(); j++)
        if (work[j] >= thr) {
          keep.push_back(cand[j]);
          kwork.push_back(work[j]);
        }
      if (!keep.empty()) {
        cand.swap(keep);  // never empty the candidate set
        work.swap(kwork);
      }
    }
    double z = 0;
    for (float w : work) z += w;
    out.ids = cand;
    out.pr.resize(cand.size());
    for (size_t j = 0; j < cand.size(); j++) out.pr[j] = (float)(work[j] / z);
  }

  static float sdist_prob(const SDist& d, int id) {
    for (size_t j = 0; j < d.ids.size(); j++)
      if (d.ids[j] == id) return d.pr[j];
    return 0.0f;
  }

  int draw_sparse(const SDist& d, float* lp, bool add_hist) {
    double z = 0;
    for (float p : d.pr) z += p;
    double u = uniform() * z;
    size_t si = d.ids.size() - 1;
    for (size_t j = 0; j < d.ids.size(); j++) {
      u -= d.pr[j];
      if (u <= 0) { si = j; break; }
    }
    const int sel = d.ids[si];
    if (lp) *lp = (float)(log((double)d.pr[si]) - log(z));
    if (add_hist) remember(sel);
    return sel;
  }

  // Sparse residual: max(p-q, 0) renormalized. p's support suffices (d > 0
  // implies p > 0); ids are scanned in ascending order like the dense loop.
  int sample_residual_sparse(const SDist& p, const SDist& q, float* lp) {
    cand = p.ids;
    std::sort(cand.begin(), cand.end());
    work.clear();
    std::vector<int> sel_ids;
    std::vector<float> sel_w;
    double z = 0.0;
    for (int id : cand) {
      const float pv = sdist_prob(p, id);
      const float d = fmaxf(0.0f, pv - sdist_prob(q, id));
      if (d > 0.0f) {
        sel_ids.push_back(id);
        sel_w.push_back(d);
        z += d;
      }
    }
    if (sel_ids.empty() || z <= 0.0) {
      sel_ids.clear();
      sel_w.clear();
      z = 0.0;
      for (int id : cand) {
        const float pv = sdist_prob(p, id);
        if (pv > 0.0f) {
          sel_ids.push_back(id);
          sel_w.push_back(pv);
          z += pv;
        }
      }
    }
    double u = uniform() * z;
    int sel = sel_ids.back();
    for (size_t j = 0; j < sel_ids.size(); j++) {
      u -= sel_w[j];
      if (u <= 0.0) {
        sel = sel_ids[j];
        break;
      }
    }
    if (lp) {
      const float pv = sdist_prob(p, sel);
      *lp = pv > 0.0f ? logf(pv) : -INFINITY;
    }
    return sel;
  }
};

// Default drafter for GEN lines that omit the field (the HTTP API always
// omits it). Unset: chain — ngram drafts every round it can and MTP
// takes the rounds where the table has nothing, for greedy and sampled
// requests alike (sampled ngram rounds re-draw from the target distribution
// and accept by token-id match, llama.cpp-style; only the pure-ngram
// default stays greedy-only, see parse_gen). QWENOX_DRAFTER overrides:
// "serial"/"0" forces serial, "ngram"/"3" pure ngram, "chain"/"4" explicit
// chain, "dflash2"/"df2"/"2" the DFlash2 draft head (needs QWENOX_DFLASH2),
// "mtp" the legacy MTP default.
static int env_default_drafter() {
  const char* e = getenv("QWENOX_DRAFTER");
  if (!e) return 4;
  if (!strcmp(e, "serial") || !strcmp(e, "0")) return 0;
  if (!strcmp(e, "ngram") || !strcmp(e, "ngram-mod") || !strcmp(e, "3")) return 3;
  if (!strcmp(e, "chain") || !strcmp(e, "4")) return 4;
  if (!strcmp(e, "dflash2") || !strcmp(e, "df2") || !strcmp(e, "2")) return 2;
  return -1;
}

// GEN <req> <max_tokens> <n_eos> <eos...> <n_ids> <ids...> [drafter] [suffixes]
static GenReq parse_gen(std::istringstream& ss) {
  GenReq g;
  int n_eos = 0, n_ids = 0;
  if (!(ss >> g.req >> g.max_tokens >> n_eos) || n_eos < 0 || n_eos > 64 ||
      g.max_tokens <= 0) {
    g.reject = true;
    return g;
  }
  for (int i = 0; i < n_eos; i++) {
    int e;
    if (!(ss >> e)) { g.reject = true; return g; }
    g.eos.push_back(e);
  }
  if (!(ss >> n_ids) || n_ids <= 0) { g.reject = true; return g; }
  g.ids.resize(n_ids);
  for (int i = 0; i < n_ids; i++)
    if (!(ss >> g.ids[i])) { g.reject = true; return g; }
  std::string tok;
  while (ss >> tok) {
    if (tok == "SAMPLE") {
      // SAMPLE <temp> <top_k> <top_p> <min_p> <seed:u64>
      if (!(ss >> g.temp >> g.top_k >> g.top_p >> g.min_p >> g.seed) ||
          g.temp <= 0.0f || g.top_k < 0) {
        g.reject = true;
        return g;
      }
      g.sample = true;
    } else if (tok == "NGRAM") {
      g.drafter = 3;
    } else if (tok == "PENALTY") {
      if (!(ss >> g.presence >> g.frequency)) { g.reject = true; return g; }
    } else if (tok == "BIAS") {
      int nb;
      if (!(ss >> nb) || nb < 0 || nb > 20480) { g.reject = true; return g; }
      g.bias.resize(nb);
      for (int i = 0; i < nb; i++) {
        if (!(ss >> g.bias[i].first >> g.bias[i].second) ||
            g.bias[i].first < 0 || g.bias[i].first >= 248320) {
          g.reject = true;
          return g;
        }
      }
    } else if (tok == "LOGPROBS") {
      g.logprobs = true;
    } else if (tok == "SEEDED") {
      g.seeded = true;
    } else if (tok == "MROPE") {
      int k;
      // One grid per image: same ceiling as VIMG (was 64, a hidden 2nd cap).
      if (!(ss >> k) || k <= 0 || k > kVimgMax) { g.reject = true; return g; }
      for (int i = 0; i < k; i++) {
        std::array<int, 3> g3{0, 0, 0};
        if (!(ss >> g3[0] >> g3[1] >> g3[2])) { g.reject = true; return g; }
        g.mrope_grids.push_back(g3);
      }
    } else if (tok == "VIMG") {
      // g.vimg stays recorded even when the value is rejected, so the serve
      // loop can drain that many frames before it replies.
      if (!(ss >> g.vimg)) { g.reject = true; return g; }
      if (g.vimg <= 0 || g.vimg > kVimgMax) {
        g.reject = true;
        g.reject_why = "bad VIMG frame count";
        return g;
      }
    } else if (tok == "SNAP" || tok == "SNAP2") {
      long long v;
      if (!(ss >> v) || v < 0) { g.reject = true; return g; }
      (tok == "SNAP" ? g.snap : g.snap2) = (size_t)v;
    } else if (tok == "SNAPS") {
      long long n;
      if (!(ss >> n) || n < 0 || n > 64) { g.reject = true; return g; }
      for (long long i = 0; i < n; i++) {
        long long v;
        if (!(ss >> v) || v < 0) { g.reject = true; return g; }
        g.snaps.push_back((size_t)v);
      }
    } else if (tok == "CKPT") {
      int n;
      if (!(ss >> n) || n < 0 || n > 64) { g.reject = true; return g; }
      for (int i = 0; i < n; i++) {
        int v;
        if (!(ss >> v) || v < 0) { g.reject = true; return g; }
        g.ckpt.push_back(v);
      }
    } else if (isdigit(tok[0]) || tok[0] == '-') {
      int v = atoi(tok.c_str());
      if (v < 0 || v > 4) g.reject = true;
      else g.drafter = v;
    }
    // anything else: drop (lenient by design)
  }
  // Omitted drafter: resolve the QWENOX_DRAFTER default. Pure ngram only
  // drafts greedy requests (sampled ones keep the legacy default); chain
  // covers greedy and sampled alike.
  if (g.drafter < 0) {
    static const int def = env_default_drafter();
    g.drafter = (def == 3 && g.sample) ? -1 : def;
  }
  // PENALTY/BIAS/LOGPROBS are sampler-only (protocol spec): no sampler, no-op
  // fields would silently mislead the client.
  if (!g.sample &&
      (g.presence != 0.0f || g.frequency != 0.0f || !g.bias.empty() || g.logprobs))
    g.reject = true;
  return g;
}

// ---------------------------------------------------------------------------
// Concurrent serving (QWENOX_PARALLEL=N, paged KV only; 2026-09-24).
//
// N sequence slots (GpuModel::slots_init) time-share the GPU. Every
// connection gets a reader thread -- PING/INFO/MEM/X answer at once, even
// mid-request -- and each GEN runs on its own worker thread. A FIFO ticket
// lock (Turnstile) owns the GPU; the owner hands it over at yield points
// (spec round tops, prompt prefill chunk boundaries, serial decode tokens)
// when someone is waiting, so sequences interleave but each one computes
// exactly what it would alone (bit-exact per sequence).
//
// All slots map KV pages from ONE pool (default: one maxctx sequence, like
// llama.cpp's unified KV). When the pool runs dry (GpuModel::kv_evict_one):
// RAM checkpoints go first, then idle slots' pages (LRU), then the
// latest-admitted busy request later than the one asking is aborted; if
// the asker is itself the latest, IT fails (KvOverflow). An earlier request
// is never sacrificed for a later one.
//
// N=1 keeps the old batch-1 semantics: live-state reuse is per connection
// (a new connection starts with an empty hist), no hooks are installed.

// QWENOX_TURN_PROF=1: GPU-turn time accounting for concurrent decode
// (diagnostic only; off = zero cost). Every release syncs the stream so a
// hold's time includes the GPU work it enqueued; time with no owner is the
// hand-off gap (thread wake-ups, rendezvous waits). Tags say who held it.
enum TurnTag { TT_WORKER = 0, TT_VR = 1, TT_DR = 2, TT_N = 3 };
static thread_local int tl_turn_tag = TT_WORKER;
static const bool g_turn_prof = [] {
  const char* e = getenv("QWENOX_TURN_PROF");
  return e && *e && strcmp(e, "0") != 0;
}();
struct TurnProf {
  using clk = std::chrono::steady_clock;
  clk::time_point acq{}, rel{}, last_print{};
  double held[TT_N] = {}, idle = 0;  // ms, since last print
  long holds[TT_N] = {};
  bool have_rel = false;
  // Called under the turnstile (only the owner touches it).
  void on_acquire() {
    acq = clk::now();
    if (have_rel) idle += std::chrono::duration<double, std::milli>(acq - rel).count();
  }
  void on_release(int tag) {
    (void)hipStreamSynchronize(g_str);
    rel = clk::now();
    have_rel = true;
    held[tag] += std::chrono::duration<double, std::milli>(rel - acq).count();
    holds[tag]++;
    if (last_print.time_since_epoch().count() == 0) last_print = rel;
    const double span = std::chrono::duration<double, std::milli>(rel - last_print).count();
    if (span >= 10000.0) {
      const double tot = held[0] + held[1] + held[2] + idle;
      fprintf(stderr,
              "%s [turnprof] %.1fs: verify-exec %.1f%% (%ld, %.1fms avg) "
              "draft-exec %.1f%% (%ld, %.1fms avg) worker %.1f%% (%ld, %.2fms avg) "
              "idle %.1f%%\n",
              log_ts(), span / 1000.0, 100 * held[1] / tot, holds[1],
              holds[1] ? held[1] / holds[1] : 0.0, 100 * held[2] / tot, holds[2],
              holds[2] ? held[2] / holds[2] : 0.0, 100 * held[0] / tot, holds[0],
              holds[0] ? held[0] / holds[0] : 0.0, 100 * idle / tot);
      if (g_vseg_calls)
        fprintf(stderr,
                "%s [turnprof] batched verify: %ld calls, %.1f rows avg, "
                "PLE join wait %.1fms avg (gather %.1fms), host enqueue %.1fms avg\n",
                log_ts(), g_vseg_calls, (double)g_vseg_rows / g_vseg_calls,
                g_vseg_ple_ms / g_vseg_calls, g_vseg_pleh_ms / g_vseg_calls,
                g_vseg_enq_ms / g_vseg_calls);
      g_vseg_calls = g_vseg_rows = 0;
      g_vseg_ple_ms = g_vseg_pleh_ms = g_vseg_enq_ms = 0;
      for (int i = 0; i < TT_N; i++) held[i] = 0, holds[i] = 0;
      idle = 0;
      last_print = rel;
    }
  }
};
static TurnProf g_turnprof;

// FIFO ticket lock around the (single-threaded) GpuModel.
// 10-13: a waiter may tag itself HEAVY with its remaining prompt tokens
// (acquire(rem > 0)): its next GPU work is a big prompt chunk. Grant order is
// unchanged (FIFO); the owner only reads the tags (waiters()) to decide whether
// lending or yielding can help anyone (see yield_hook / borrow_query).
struct Turnstile {
  std::mutex mu;
  std::condition_variable cv;
  uint64_t next = 0, serving = 0;
  std::vector<std::pair<uint64_t, int>> heavy;  // (ticket, remaining) still waiting
  uint64_t acquire(int heavy_rem = 0) {  // returns this grant's ticket (QWENOX_VR_CHECK)
    std::unique_lock<std::mutex> lk(mu);
    const uint64_t t = next++;
    if (heavy_rem > 0) heavy.emplace_back(t, heavy_rem);
    cv.wait(lk, [&] { return serving == t; });
    if (heavy_rem > 0)
      for (size_t i = 0; i < heavy.size(); i++)
        if (heavy[i].first == t) {
          heavy.erase(heavy.begin() + (long)i);
          break;
        }
    if (g_turn_prof) g_turnprof.on_acquire();
    return t;
  }
  struct Waiters {
    int light = 0, heavy = 0;
    int min_heavy_rem = INT_MAX;
  };
  Waiters waiters() {  // owner only: who is queued behind it
    std::lock_guard<std::mutex> lk(mu);
    Waiters w;
    const int all = (int)(next - serving) - 1;
    w.heavy = (int)heavy.size();
    w.light = std::max(0, all - w.heavy);
    for (const auto& h : heavy) w.min_heavy_rem = std::min(w.min_heavy_rem, h.second);
    return w;
  }
  void release() {
    if (g_turn_prof) g_turnprof.on_release(tl_turn_tag);
    {
      std::lock_guard<std::mutex> lk(mu);
      serving++;
    }
    cv.notify_all();
  }
  bool contended() {  // someone besides the owner is queued
    std::lock_guard<std::mutex> lk(mu);
    return next - serving > 1;
  }
};

struct Slot {
  bool busy = false;
  bool abort = false;  // pool-overflow victim: fail at the next GPU acquire
  // MTP KV continuity: true after a spec-enabled turn (prefill tap capture
  // + spec decode keep the MTP layer's KV in lockstep with the trunk).
  // False after kvsnap/rckpt restores and any plain (non-speculative) turn.
  bool mtp_live = false;
  // tokens INGESTED by this slot's live state (prompt + every token actually
  // fed to forward). KV reuse: a GEN whose prompt extends hist exactly
  // continues from live state instead of re-prefilling.
  std::vector<int> hist;
  uint64_t seq = 0;       // admission order of the request it runs
  uint64_t last_use = 0;  // LRU among idle slots
  uint64_t owner = 0;     // connection id of the last request (N=1 affinity)
  int reserved = 0;
  // Set at admission: longest common prefix of the prompt with any idle
  // slot's live tokens when none continues it (0 for a live continuation).
  // Feeds the fork checkpoint (see fork_cut).
  size_t fork_lcp = 0;
};

static Turnstile g_turn;
static std::mutex g_sm;  // g_slots bookkeeping + admission queue
static std::condition_variable g_scv;
static std::vector<Slot> g_slots;
static std::list<uint64_t> g_wait;  // admission tickets, FIFO
static uint64_t g_ticket = 0, g_seq = 0, g_use = 0;
static std::atomic<uint64_t> g_conn_id{0};
static thread_local int tl_slot = -1;  // slot of the GPU-owning worker
static int g_reserve_decode = 4096;

// Idle slot for a prompt: the longest hist that is a strict prefix (live
// continuation), else an empty slot, else the least recently used one whose
// connection is gone (owner 0), else the least recently used one.
static int pick_slot(const std::vector<int>& ids) {
  int best = -1, empty = -1, lru = -1, orphan = -1;
  size_t bestn = 0;
  for (int i = 0; i < (int)g_slots.size(); i++) {
    const Slot& s = g_slots[(size_t)i];
    if (s.busy) continue;
    const size_t n = s.hist.size();
    if (n == 0) {
      if (empty < 0) empty = i;
      continue;
    }
    if (n < ids.size() && n > bestn &&
        std::equal(s.hist.begin(), s.hist.end(), ids.begin())) {
      best = i;
      bestn = n;
    }
    if (lru < 0 || s.last_use < g_slots[(size_t)lru].last_use) lru = i;
    if (s.owner == 0 && (orphan < 0 || s.last_use < g_slots[(size_t)orphan].last_use))
      orphan = i;
  }
  return best >= 0 ? best : empty >= 0 ? empty : orphan >= 0 ? orphan : lru;
}

// Admission miss diagnostic (under g_sm, before claiming `chosen`): when no
// idle slot's live tokens are a strict prefix of the prompt, log the closest
// one and the tokens around its first mismatch. An lcp deep inside an earlier
// reply means the client re-tokenized history differently from what was
// generated; a short lcp is just another conversation. Returns that lcp
// (0 for a live continuation or when no idle slot holds tokens).
static size_t log_prefix_miss(long long req, const std::vector<int>& ids, int chosen) {
  const std::vector<int>& ch = g_slots[(size_t)chosen].hist;
  if (!ch.empty() && ch.size() < ids.size() &&
      std::equal(ch.begin(), ch.end(), ids.begin()))
    return 0;  // live continuation
  int best = -1;
  size_t bl = 0;
  for (int i = 0; i < (int)g_slots.size(); i++) {
    const Slot& s = g_slots[(size_t)i];
    if (s.busy || s.hist.empty()) continue;
    size_t l = 0;
    while (l < s.hist.size() && l < ids.size() && s.hist[l] == ids[l]) l++;
    if (best < 0 || l > bl) {
      best = i;
      bl = l;
    }
  }
  if (best < 0) return 0;
  const std::vector<int>& h = g_slots[(size_t)best].hist;
  std::string why;
  if (bl < h.size() && bl < ids.size()) {
    char b[64];
    why = "first mismatch at " + std::to_string(bl) + ": live";
    for (size_t k = bl; k < h.size() && k < bl + 4; k++) {
      snprintf(b, sizeof b, " %d", h[k]);
      why += b;
    }
    why += " vs prompt";
    for (size_t k = bl; k < ids.size() && k < bl + 4; k++) {
      snprintf(b, sizeof b, " %d", ids[k]);
      why += b;
    }
  } else {  // a strict-prefix slot would have been picked
    why = "prompt ends within the live tokens";
  }
  fprintf(stderr, "%s req %lld: no live prefix; closest slot %d matches %zu "
          "of %zu live tokens (prompt %zu), %s\n", log_ts(), req, best, bl,
          h.size(), ids.size(), why.c_str());
  return bl;
}

// Wait (FIFO) for an idle slot and claim it; -1 if cancelled while queued,
// -2 if the shared KV pool cannot fit the request alongside the reservations
// of the requests already running.
static int slot_admit(GpuModel& m, const std::vector<int>& ids, int max_tokens,
                      const std::atomic<bool>& cancel, long long req) {
  std::unique_lock<std::mutex> lk(g_sm);
  const uint64_t t = g_ticket++;
  g_wait.push_back(t);
  for (;;) {
    if (cancel) {
      g_wait.remove(t);
      g_scv.notify_all();
      return -1;
    }
    if (g_wait.front() == t && pick_slot(ids) >= 0) {
      lk.unlock();
      g_turn.acquire();
      lk.lock();
      if (cancel) {
        g_wait.pop_front();
        g_scv.notify_all();
        lk.unlock();
        g_turn.release();
        return -1;
      }
      const int s = pick_slot(ids);
      if (s >= 0) {
        if (m.nslots > 1 && m.kv_paged) {
          const Slot& sl = g_slots[(size_t)s];
          size_t lcp = 0;
          while (lcp < sl.hist.size() && lcp < ids.size() &&
                 sl.hist[lcp] == ids[lcp])
            lcp++;
          const bool cont = !sl.hist.empty() && lcp == sl.hist.size() &&
                            lcp < ids.size();
          const int target = kv_admission_target(ids.size(), max_tokens,
                                                  g_reserve_decode, m.maxctx,
                                                  KV_PAGE);
          std::vector<KvAdmissionState> states(g_slots.size());
          for (size_t index = 0; index < g_slots.size(); index++)
            states[index] = {g_slots[index].busy && !g_slots[index].abort,
                             m.slot_pages((int)index), g_slots[index].reserved};
          const int need = kv_admission_need(target, states[(size_t)s].mapped, cont);
          const int64_t supply = kv_admission_available(m.kv_pool_pages, states,
                                                        (size_t)s);
          if (need > supply) {
            fprintf(stderr,
                    "%s req %lld rejected at admission: needs %d pages "
                    "(%zu prompt tokens + %d reserved decode), only %lld of %d pool "
                    "pages available to it\n",
                    log_ts(), req, need, ids.size(),
                    std::max(0, std::min(max_tokens, g_reserve_decode)),
                    (long long)supply, m.kv_pool_pages);
            g_wait.pop_front();
            g_scv.notify_all();
            lk.unlock();
            g_turn.release();
            return -2;
          }
          g_slots[(size_t)s].reserved = target;
        }
        g_slots[(size_t)s].fork_lcp = log_prefix_miss(req, ids, s);
        g_wait.pop_front();
        Slot& sl = g_slots[(size_t)s];
        sl.busy = true;
        sl.abort = false;
        sl.seq = ++g_seq;
        g_scv.notify_all();
        return s;
      }
      lk.unlock();
      g_turn.release();
      lk.lock();
    }
    g_scv.wait_for(lk, std::chrono::milliseconds(50));
  }
}

static void slot_leave(int s) {
  {
    std::lock_guard<std::mutex> lk(g_sm);
    g_slots[(size_t)s].busy = false;
    g_slots[(size_t)s].reserved = 0;
    g_slots[(size_t)s].last_use = ++g_use;
  }
  g_scv.notify_all();
}

// 10-13 prefill scheduling knobs (both default on; =0 restores the old rule
// for A/B). QWENOX_CONC_LEND_LIGHT_ONLY: mid-chunk lending only when a light
// worker waits. QWENOX_PREFILL_SRPT: between chunks of a long prompt, hand the
// GPU to another prompt only if it has less left (else lend to light workers
// and keep going), instead of alternating chunk by chunk.
static bool lend_light_only() {
  static const bool v = !GpuModel::env_zero("QWENOX_CONC_LEND_LIGHT_ONLY");
  return v;
}
static bool prefill_srpt() {
  static const bool v = !GpuModel::env_zero("QWENOX_PREFILL_SRPT");
  return v;
}

// Fork checkpoint (the branch-point state of SGLang's MambaRadixCache). A
// prompt that an earlier sequence shared up to position F, with no
// checkpoint at or below F, is recomputed from the last one below -- often
// from 0: an agent's new session whose system prompt ends with a fresh git
// commit hash diverges from the previous session ~170 tokens before the
// first message-boundary cut. The GDN state at F never existed, so this
// request still pays; checkpointing F now lets every later sequence that
// forks at the same place resume there. F is the longest common prefix with
// an idle slot's live tokens, a RAM checkpoint, or the SSD page index (page
// granular). Exact lcps are rounded down to FORK_ALIGN, keeping at least one
// token of margin, so a BPE merge that moves the boundary by a token still
// hits. A fork less than FORK_MIN_GAIN past where this prefill starts is
// skipped: that is the usual retokenized tail of the previous reply, which
// the </think> / <tool_call> checkpoints already cover and no later prompt
// will repeat. Returns the cut, or 0. QWENOX_FORK_CKPT=0 disables.
static size_t fork_cut(const std::vector<int>& ids, size_t slot_lcp, size_t base,
                       size_t min_tokens, long long req) {
  static const bool on = !GpuModel::env_zero("QWENOX_FORK_CKPT");
  constexpr size_t FORK_ALIGN = 64, FORK_MIN_GAIN = 1024;
  if (!on) return 0;
  auto down = [](size_t l) { return l > FORK_ALIGN ? (l - 1) / FORK_ALIGN * FORK_ALIGN : 0; };
  const size_t from_slot = down(slot_lcp);
  const size_t from_ram = rckpt::enabled() ? down(rckpt::lcp(ids)) : 0;
  const size_t from_ssd = kvsnap::shared_tokens(ids, kvsnap::g_fp);
  const size_t cc = std::max({from_slot, from_ram, from_ssd});
  if (cc < base + FORK_MIN_GAIN || cc >= ids.size() || cc < min_tokens) return 0;
  fprintf(stderr, "%s fork: req %lld shares a prefix with an earlier sequence (%s); "
          "checkpointing at %zu\n", log_ts(), req,
          cc == from_slot ? "live slot" : cc == from_ram ? "RAM checkpoint" : "SSD pages", cc);
  return cc;
}

static void borrow_abort_check();
static void borrow_turn(GpuModel& m);

// GpuModel::yield_fn: hand the GPU to the next queued worker, then fail
// this request if a pool overflow picked it as the victim meanwhile.
// Between chunks of a long prompt (pf_rem() > 0) with SRPT on: yield to a
// heavy waiter only if it has less prompt left; otherwise give light workers
// one round each (borrow_turn: heavy waiters requeue) and keep the GPU.
static void yield_hook(GpuModel& m) {
  if (tl_slot < 0) return;
  const int rem = m.pf_rem();
  if (rem > 0 && prefill_srpt()) {
    const Turnstile::Waiters w = g_turn.waiters();
    if (w.heavy > 0 && w.min_heavy_rem < rem) {
      g_turn.release();
      g_turn.acquire(rem);
      m.bind(tl_slot);
    } else if (w.light > 0) {
      m.heavy_rem_ = rem;
      borrow_turn(m);  // includes the abort check
      return;
    }
  } else if (g_turn.contended()) {
    g_turn.release();
    g_turn.acquire(rem);
    m.bind(tl_slot);
  }
  bool ab;
  {
    std::lock_guard<std::mutex> lk(g_sm);
    ab = g_slots[(size_t)tl_slot].abort;
  }
  if (ab) throw KvOverflow("aborted: KV pool needed by an earlier request");
}

// D1b borrow callbacks (GpuModel::borrow_*_fn). borrow_turn runs on the
// chunk owner's thread: it raises the light-round flag, hands the turnstile
// around the FIFO once (borrowers may start only chunks of <= conc_rmax_
// rows while it is up), then rebinds our slot. borrow_skip runs on a
// borrower whose next chunk is heavy: requeue behind the owner and retry.
static bool borrow_query() {
  if (!lend_light_only()) return g_turn.contended();
  return g_turn.waiters().light > 0;
}
// GpuModel::heavy_gate_fn: the chunk about to start (rem prompt tokens left)
// should requeue because a waiting prompt has strictly less left.
static bool heavy_gate(GpuModel&, int rem) {
  if (!prefill_srpt()) return false;
  return g_turn.waiters().min_heavy_rem < rem;
}
static void borrow_abort_check() {
  bool ab;
  {
    std::lock_guard<std::mutex> lk(g_sm);
    ab = g_slots[(size_t)tl_slot].abort;
  }
  if (ab) throw KvOverflow("aborted: KV pool needed by an earlier request");
}
// Both re-acquire tagged heavy with m.heavy_rem_ (set by the caller: the
// rest of this prompt), so the owner that gets the GPU next knows a prompt
// is waiting and how much it has left.
static void borrow_turn(GpuModel& m) {
  const int rem = m.heavy_rem_;
  g_borrow_light.fetch_add(1, std::memory_order_acq_rel);
  g_turn.release();
  g_turn.acquire(rem);
  g_borrow_light.fetch_sub(1, std::memory_order_acq_rel);
  m.bind(tl_slot);
  borrow_abort_check();
}
static void borrow_skip(GpuModel& m) {
  const int rem = m.heavy_rem_;
  g_turn.release();
  g_turn.acquire(rem);
  m.bind(tl_slot);
  borrow_abort_check();
}

// ---- D2-S2a: verify rendezvous pool (VerifyPool) ---------------------------
// Greedy spec workers reach the rendezvous after drafting a round: they hand
// the GPU turn over and wait; the last submitter (or a waiter whose window
// elapsed) becomes the executor and runs ONE batched multi-segment verify
// for the round (GpuModel::verify_segments, bit-identical to serial per the
// S1c selftest). Judging/emit/rollback stay on each worker after it
// re-acquires the turn, so abort/cancel semantics are unchanged.
struct VrReq {
  int slot;
  const std::vector<int>* vt;  // worker stack; valid while the worker blocks
  int base;                    // q+1
  int* vout;                   // worker stack v.data(), room for vt->size()
  const std::function<void(int)>* post = nullptr;  // see GpuModel::vseg_pool_fn
  bool done = false;
  bool dropped = false;        // abort/guard refusal -> worker serial rerun
  // QWENOX_VR_CHECK / QWENOX_VR_RESTORE (10-13): this segment's row window in
  // the shared post-verify scratch (-1 = not recorded), its hash at the
  // executor's release, and the executor's turn ticket.
  int win_row0 = -1;
  bool saved = false;
  std::array<uint64_t, 4> win_hash{};
  uint64_t exec_ticket = 0;
};
static bool vr_check_on() {
  static const bool v = getenv("QWENOX_VR_CHECK") && !GpuModel::env_zero("QWENOX_VR_CHECK");
  return v;
}
static bool vr_restore_on() { return GpuModel::vr_restore_env(); }  // default on
static std::atomic<long> g_vrc_ok{0}, g_vrc_bad{0}, g_vrc_restored{0};
struct VerifyPool {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<VrReq*> pend;    // current round's submissions
  int active = 0;              // workers able to submit this round (under mu)
  bool executing = false;
  std::chrono::steady_clock::time_point first_submit;
  long batches = 0, solo = 0, timeouts = 0, fallbacks = 0;
  long hist[9] = {};           // batch-size (segments) histogram
};
static VerifyPool g_vr;
static int g_vr_wait_ms = 50;  // QWENOX_VR_WAIT_MS

// Membership counting. Every exit path (eos, cancel, KvOverflow unwind) must
// decrement, else the all-arrived trigger never fires and the pool stalls to
// the timeout each round. Both spec_loop (VrActive) and spec_loop_chain
// (VrLoop, via the callbacks) count the WHOLE decode loop — the round body
// runs only while holding the GPU turn, so any finer scope never sees a
// peer. Chain ngram rounds excuse themselves per round (they never submit).
static void vseg_round_enter() {
  std::lock_guard<std::mutex> lk(g_vr.mu);
  g_vr.active++;
}
static void vseg_round_exit() {
  {
    std::lock_guard<std::mutex> lk(g_vr.mu);
    g_vr.active--;
  }
  g_vr.cv.notify_all();  // the round may now be complete without us
}
struct VrActive {
  VrActive() { vseg_round_enter(); }
  ~VrActive() { vseg_round_exit(); }
};

static int vseg_active() {
  std::lock_guard<std::mutex> lk(g_vr.mu);
  return g_vr.active;
}

// Executor half. Runs on the promoting worker's thread; acquires the GPU
// turn (FIFO behind the peers, so their drafts finish first), runs the
// batched verify, scatters the per-segment argmax rows into each worker's
// vout, and wakes everyone. Call with the turn NOT held; returns with it
// released again.
static void vr_execute(GpuModel& m) {
  tl_turn_tag = TT_VR;
  const uint64_t my_ticket = g_turn.acquire();
  m.bind(tl_slot);
  std::vector<VrReq*> mine;
  {
    std::lock_guard<std::mutex> lk(g_vr.mu);
    mine.swap(g_vr.pend);
  }
  std::vector<GpuModel::VSeg> segs;
  std::vector<VrReq*> run;  // run[i] owns segs[i]
  int rows = 0;
  for (VrReq* r : mine) {
    {
      std::lock_guard<std::mutex> lk(g_sm);
      if (g_slots[(size_t)r->slot].abort) { r->dropped = true; continue; }
    }
    const int P = (int)r->vt->size();
    if (rows + P > m.kVSnapRows) {  // hard row budget
      r->dropped = true;
      g_vr.fallbacks++;
      continue;
    }
    rows += P;
    segs.push_back(GpuModel::VSeg{r->slot, r->vt, r->base});
    run.push_back(r);
  }
  if (!segs.empty()) {
    std::vector<int> v((size_t)rows);
    if (m.verify_segments(segs.data(), (int)segs.size(), v.data())) {
      int off = 0;
      for (size_t i = 0; i < segs.size(); i++) {
        const int P = (int)segs[i].toks->size();
        memcpy(run[i]->vout, v.data() + off, (size_t)P * sizeof(int));
        if (run[i]->post) (*run[i]->post)(off);  // rows still in d_logits_b
        // 10-13: record this segment's shared row window before the turn is
        // released (see GpuModel::vr_rows_each).
        run[i]->exec_ticket = my_ticket;
        if (vr_check_on() || vr_restore_on()) {
          run[i]->win_row0 = off;
          if (vr_check_on()) run[i]->win_hash = m.vr_rows_hash(off, P);
          if (vr_restore_on()) run[i]->saved = m.vr_rows_save(segs[i].slot, off, P);
        }
        off += P;
      }
    } else {
      // Batched pass refused by a guard: serial per segment — today's
      // time-sliced behavior, still inside this worker's turn.
      g_vr.fallbacks++;
      fprintf(stderr, "%s [vr] batched verify refused (%zu segments, %d rows);"
                      " serial fallback\n", log_ts(), segs.size(), rows);
      // Every serial segment packs at its slot's vsnap_row0_ (0 from the
      // spec loop), so segment i+1 overwrites segment i's rows: record and
      // save each window right after its own verify (10-13).
      for (size_t i = 0; i < segs.size(); i++) {
        m.bind(segs[i].slot);
        if (!m.verify_segments(&segs[i], 1, run[i]->vout)) {
          run[i]->dropped = true;  // worker re-runs prefill_batch+batch_argmax
          continue;
        }
        if (run[i]->post) (*run[i]->post)(0);
        run[i]->exec_ticket = my_ticket;
        if (vr_check_on() || vr_restore_on()) {
          const int P = (int)segs[i].toks->size(), r0 = m.vsnap_row0_;
          run[i]->win_row0 = r0;
          if (vr_check_on()) run[i]->win_hash = m.vr_rows_hash(r0, P);
          if (vr_restore_on()) run[i]->saved = m.vr_rows_save(segs[i].slot, r0, P);
        }
      }
    }
  }
  static const bool vr_dbg = getenv("QWENOX_VR_DBG") != nullptr;
  {
    std::lock_guard<std::mutex> lk(g_vr.mu);
    for (VrReq* r : mine) r->done = true;
    g_vr.executing = false;
    g_vr.batches++;
    g_vr.hist[std::min(segs.size(), (size_t)8)]++;
    if (segs.size() <= 1) g_vr.solo++;
    if (vr_dbg)
      fprintf(stderr, "%s [vr] batch: %zu segs %d rows (batches %ld, solo %ld,"
                      " timeouts %ld, fallbacks %ld)\n",
              log_ts(), segs.size(), rows, g_vr.batches, g_vr.solo,
              g_vr.timeouts, g_vr.fallbacks);
  }
  g_vr.cv.notify_all();
  g_turn.release();
  tl_turn_tag = TT_WORKER;
}

// Worker half (GpuModel::vseg_pool_fn), called from spec_loop holding the
// GPU turn, drafts done. Returns true with vout filled, or false when the
// pool could not run this segment (caller runs the serial verify).
static bool vseg_pool(GpuModel& m, const GpuModel::VSeg& self, int* vout,
                      const std::function<void(int)>* post) {
  {
    std::lock_guard<std::mutex> lk(g_vr.mu);
    if (g_vr.active <= 1) return false;  // sole decoder: keep the serial path
  }
  VrReq req{self.slot, self.toks, self.base, vout, post, false, false};
  bool am_exec = false;
  {
    std::lock_guard<std::mutex> lk(g_vr.mu);
    if (g_vr.pend.empty()) g_vr.first_submit = std::chrono::steady_clock::now();
    g_vr.pend.push_back(&req);
    if ((int)g_vr.pend.size() >= g_vr.active && !g_vr.executing)
      am_exec = g_vr.executing = true;  // everyone is in: run the round
  }
  g_turn.release();  // let the peers' drafts/judging run while we wait
  if (!am_exec) {
    std::unique_lock<std::mutex> lk(g_vr.mu);
    for (;;) {
      if (req.done) break;
      if (!g_vr.executing) {
        // The round completed without us (a peer left its loop), or the wait
        // window elapsed: run what we have instead of waiting forever.
        if ((int)g_vr.pend.size() >= g_vr.active ||
            std::chrono::steady_clock::now() - g_vr.first_submit >=
                std::chrono::milliseconds(g_vr_wait_ms)) {
          am_exec = g_vr.executing = true;
          if ((int)g_vr.pend.size() < g_vr.active) g_vr.timeouts++;
          break;
        }
        g_vr.cv.wait_until(lk, g_vr.first_submit +
                                   std::chrono::milliseconds(g_vr_wait_ms));
      } else {
        g_vr.cv.wait(lk);
      }
    }
    lk.unlock();
  }
  if (am_exec) vr_execute(m);
  const uint64_t back = g_turn.acquire();
  m.bind(tl_slot);
  if (!req.dropped && req.win_row0 >= 0) {
    const int P = (int)self.toks->size();
    // grants that ran between the executor's turn and ours (peers + others)
    const long between = (long)(back - req.exec_ticket) - 1;
    if (vr_check_on()) {
      const auto h = m.vr_rows_hash(req.win_row0, P);
      static const char* nm[4] = {"d_Rb", "d_grow", "d_csnap", "d_iproj_snap"};
      std::string bad;
      for (int k = 0; k < 4; k++)
        if (h[k] != req.win_hash[k]) bad += std::string(bad.empty() ? "" : ",") + nm[k];
      if (bad.empty()) {
        g_vrc_ok++;
      } else {
        const long nb = ++g_vrc_bad;
        fprintf(stderr, "%s [vr-check] slot %d rows [%d,%d) CLOBBERED: %s (%ld grants in between;"
                        " ok %ld bad %ld)\n",
                log_ts(), self.slot, req.win_row0, req.win_row0 + P, bad.c_str(), between,
                g_vrc_ok.load(), nb);
      }
      const long tot = g_vrc_ok.load() + g_vrc_bad.load();
      if (tot % 100 == 0)
        fprintf(stderr, "%s [vr-check] ok %ld bad %ld\n", log_ts(), g_vrc_ok.load(),
                g_vrc_bad.load());
    }
    if (vr_restore_on() && req.saved) {
      m.vr_rows_load(self.slot, req.win_row0, P);
      g_vrc_restored++;
    }
  }
  borrow_abort_check();
  return !req.dropped;
}

// ---- D2-S3c: draft rendezvous pool (DraftPool) ----------------------------
// Same rendezvous shape as the verify pool above: spec workers reach their
// draft burst holding the GPU turn, submit the round (slot, gamma, tok0),
// hand the turn over and wait; the last submitter (or a waiter whose window
// elapsed) becomes the executor and runs ONE batched cross-slot draft for
// the round (GpuModel::mtp_draft_batch, bit-identical to the serial per-slot
// mtp_step bursts). Membership mirrors the verify pool: spec_loop joins at
// the serve call site (DrActive), spec_loop_chain joins per loop and its
// ngram rounds excuse themselves per round (they never submit).
struct DrReq {
  int slot;
  int g;
  int tok0;
  int* drafts;                 // worker stack; room for g ints
  const std::function<bool(int, const float*, int&)>* step = nullptr;  // C7 sampled
  bool done = false;
  bool dropped = false;        // abort/guard refusal -> worker serial rerun
};
struct DraftPool {
  std::mutex mu;
  std::condition_variable cv;
  std::vector<DrReq*> pend;    // current round's submissions
  int active = 0;              // workers able to submit this round (under mu)
  bool executing = false;
  std::chrono::steady_clock::time_point first_submit;
  long batches = 0, solo = 0, timeouts = 0, fallbacks = 0;
  long hist[9] = {};           // batch-size (segments) histogram
};
static DraftPool g_dr;
static int g_dr_wait_ms = 25;  // QWENOX_DR_WAIT_MS

static void dr_round_enter() {
  std::lock_guard<std::mutex> lk(g_dr.mu);
  g_dr.active++;
}
static void dr_round_exit() {
  {
    std::lock_guard<std::mutex> lk(g_dr.mu);
    g_dr.active--;
  }
  g_dr.cv.notify_all();  // the round may now be complete without us
}
// Loop-scope membership at the spec_loop call site. on=false when this
// worker can never submit (GpuModel::dr_worker_ok): it must not be counted,
// else the all-arrived trigger waits for it until every timeout.
struct DrActive {
  bool on;
  explicit DrActive(bool on) : on(on) { if (on) dr_round_enter(); }
  ~DrActive() { if (on) dr_round_exit(); }
};

// Executor half. Runs on the promoting worker's thread; acquires the GPU
// turn (FIFO behind the peers, so their rounds land first), runs the batched
// draft, and wakes everyone. Call with the turn NOT held; returns with it
// released again.
static void dr_execute(GpuModel& m) {
  tl_turn_tag = TT_DR;
  g_turn.acquire();
  m.bind(tl_slot);
  std::vector<DrReq*> mine;
  {
    std::lock_guard<std::mutex> lk(g_dr.mu);
    mine.swap(g_dr.pend);
  }
  std::vector<GpuModel::DrSeg> segs;
  for (DrReq* r : mine) {
    {
      std::lock_guard<std::mutex> lk(g_sm);
      if (g_slots[(size_t)r->slot].abort) { r->dropped = true; continue; }
    }
    segs.push_back(GpuModel::DrSeg{r->slot, r->g, r->tok0, r->drafts, r->step});
  }
  // Guard refusal (incl. a borrow window being up: mtp_draft_batch rejects
  // ws_lent_) -> every worker re-runs its own serial burst.
  if (!segs.empty() && !m.mtp_draft_batch(segs.data(), (int)segs.size())) {
    g_dr.fallbacks++;
    for (DrReq* r : mine) r->dropped = true;
  }
  static const bool vr_dbg = getenv("QWENOX_VR_DBG") != nullptr;
  {
    std::lock_guard<std::mutex> lk(g_dr.mu);
    for (DrReq* r : mine) r->done = true;
    g_dr.executing = false;
    g_dr.batches++;
    g_dr.hist[std::min(segs.size(), (size_t)8)]++;
    if (segs.size() <= 1) g_dr.solo++;
    if (vr_dbg)
      fprintf(stderr, "%s [dr] batch: %zu segs (batches %ld, solo %ld,"
                      " timeouts %ld, fallbacks %ld)\n",
              log_ts(), segs.size(), g_dr.batches, g_dr.solo, g_dr.timeouts,
              g_dr.fallbacks);
  }
  g_dr.cv.notify_all();
  g_turn.release();
  tl_turn_tag = TT_WORKER;
}

// Worker half (GpuModel::dr_pool_fn), called from the spec loops holding the
// GPU turn, checkpoint saved. Returns true with drafts filled, or false when
// the pool could not run this round (caller runs the serial draft burst).
static bool dr_pool(GpuModel& m, const GpuModel::DrSeg& self) {
  {
    std::lock_guard<std::mutex> lk(g_dr.mu);
    if (g_dr.active <= 1) return false;  // sole decoder: keep the serial path
  }
  DrReq req{self.slot, self.g, self.tok0, self.drafts, self.step, false, false};
  bool am_exec = false;
  {
    std::lock_guard<std::mutex> lk(g_dr.mu);
    if (g_dr.pend.empty()) g_dr.first_submit = std::chrono::steady_clock::now();
    g_dr.pend.push_back(&req);
    if ((int)g_dr.pend.size() >= g_dr.active && !g_dr.executing)
      am_exec = g_dr.executing = true;  // everyone is in: run the round
  }
  g_turn.release();  // let the peers' rounds run while we wait
  if (!am_exec) {
    std::unique_lock<std::mutex> lk(g_dr.mu);
    for (;;) {
      if (req.done) break;
      if (!g_dr.executing) {
        // The round completed without us (a peer left its loop), or the wait
        // window elapsed: run what we have instead of waiting forever.
        if ((int)g_dr.pend.size() >= g_dr.active ||
            std::chrono::steady_clock::now() - g_dr.first_submit >=
                std::chrono::milliseconds(g_dr_wait_ms)) {
          am_exec = g_dr.executing = true;
          if ((int)g_dr.pend.size() < g_dr.active) g_dr.timeouts++;
          break;
        }
        g_dr.cv.wait_until(lk, g_dr.first_submit +
                                   std::chrono::milliseconds(g_dr_wait_ms));
      } else {
        g_dr.cv.wait(lk);
      }
    }
    lk.unlock();
  }
  if (am_exec) dr_execute(m);
  g_turn.acquire();
  m.bind(tl_slot);
  borrow_abort_check();
  return !req.dropped;
}

// GpuModel::kv_starve_fn (pool dry after rckpt eviction), GPU owner only.
static bool starve(GpuModel& m) {
  const int cur = tl_slot;
  if (cur < 0) return false;
  std::lock_guard<std::mutex> lk(g_sm);
  const int n = (int)g_slots.size();
  int v = -1;
  for (int i = 0; i < n; i++) {
    const Slot& s = g_slots[(size_t)i];
    if (i != cur && !s.busy && m.slot_pages(i) > 0 &&
        (v < 0 || s.last_use < g_slots[(size_t)v].last_use))
      v = i;
  }
  if (v >= 0) {
    fprintf(stderr, "%s [kvpage] pool dry: dropped idle slot %d (%zu tokens, %d pages)\n",
            log_ts(), v, g_slots[(size_t)v].hist.size(), m.slot_pages(v));
    m.slot_reset(v, cur);
    g_slots[(size_t)v].hist.clear();
    g_slots[(size_t)v].mtp_live = false;
    return true;
  }
  const uint64_t mine = g_slots[(size_t)cur].seq;
  for (int i = 0; i < n; i++) {
    const Slot& s = g_slots[(size_t)i];
    if (i != cur && s.busy && !s.abort && s.seq > mine && m.slot_pages(i) > 0 &&
        (v < 0 || s.seq > g_slots[(size_t)v].seq))
      v = i;
  }
  if (v >= 0) {
    fprintf(stderr, "%s [kvpage] pool dry: aborting the later request on slot %d "
            "(%d pages) for slot %d\n", log_ts(), v, m.slot_pages(v), cur);
    m.slot_reset(v, cur);
    g_slots[(size_t)v].abort = true;
    return true;
  }
  throw KvOverflow("KV pool exhausted: concurrent contexts exceed the shared pool");
}

struct Conn {
  sockfd_t fd;
  uint64_t id = 0;
  std::mutex wmu;  // reader (PONG/INFO/...) and worker (T/D) both write
  std::atomic<bool> cancel{false}, in_flight{false};
  std::atomic<int64_t> req{0};
  bool send(const std::string& s) {
    std::lock_guard<std::mutex> lk(wmu);
    return send_all(fd, s);
  }
};

// One GEN, on its own worker thread. Protocol lines and semantics are the
// batch-1 ones; only scheduling changed.
static void gen_one(GpuModel& m, Conn& c, GenReq g,
                    std::vector<std::vector<float>> vpatches,
                    const kvsnap::Cfg& kvs) {
  hip_use_selected_device();
  kvsnap::Turn kv_turn;  // KVSYNC waits for this turn's SSD commit
  const std::atomic<bool>& cancel = c.cancel;
  char db[192];
  auto err_line = [&]() {
    snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
             g.ids.size());
    return std::string(db);
  };
  // The request's last line. in_flight drops first so the GEN the client
  // sends the moment it reads D is not taken for a pipelined one.
  auto finish = [&](const std::string& d) {
    c.in_flight = false;
    if (!c.send(d)) sock_shutdown(c.fd);
  };
  // Vision request validation (0.5.8 semantics), before state mutation:
  // patches need a loaded tower and one MROPE grid each; any image
  // placeholder without a grid is rejected (first position logged).
  if (g.vimg > 0) {
    const char* why = nullptr;
    if (!m.vision_on)
      why = "vision tower not loaded (start the engine with "
            "--vision-tower <sidecar.hgn>)";
    else if ((int)g.mrope_grids.size() != g.vimg)
      why = "VIMG frame count != MROPE grid count";
    else
      for (int i = 0; i < g.vimg; i++) {
        auto gr = g.mrope_grids[i];
        if (gr[0] != 1 ||
            (int)vpatches[i].size() != gr[0] * gr[1] * gr[2] * 1536) {
          why = "patches/grid size mismatch (or t != 1)";
          break;
        }
      }
    if (why) {
      fprintf(stderr, "vision: req %lld rejected: %s\n", (long long)g.req, why);
      finish(err_line());
      return;
    }
  }
  if (g.mrope_grids.empty()) {
    size_t bad = g.ids.size();
    for (size_t i = 0; i < g.ids.size(); i++)
      if (g.ids[i] == 248056 || g.ids[i] == 248057) {
        bad = i;
        break;
      }
    if (bad < g.ids.size()) {
      fprintf(stderr, "vision: req %lld rejected: image placeholder id %d at "
              "prompt pos %zu without MROPE/VIMG\n", (long long)g.req,
              g.ids[bad], bad);
      finish(err_line());
      return;
    }
  }
  const int s = slot_admit(m, g.ids, g.max_tokens, cancel, (long long)g.req);
  if (s == -2) {
    fprintf(stderr, "%s serve: req %lld failed: KV pool cannot fit it "
            "alongside the running request(s)\n", log_ts(), (long long)g.req);
    finish(err_line());
    return;
  }
  if (s < 0) {
    snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)g.req);
    finish(db);
    return;
  }
  tl_slot = s;
  m.bind(s);
  m.pf_end = 0;  // not ingesting a prompt yet (a thrown earlier request may have left it)
  // Leaving: the slot turns idle while this worker still owns the GPU, so a
  // starving peer never sees a finished request as busy.
  struct Lease {
    int s;
    ~Lease() {
      tl_slot = -1;
      slot_leave(s);
      g_turn.release();
    }
  } lease{s};
  Slot& sl = g_slots[(size_t)s];
  std::vector<int>& hist = sl.hist;
  bool& mtp_live = sl.mtp_live;
  bool victim;
  {
    std::lock_guard<std::mutex> lk(g_sm);
    victim = sl.abort;
    if (m.nslots == 1 && sl.owner != c.id) {  // batch-1: reuse per connection
      hist.clear();
      mtp_live = false;
    }
    sl.owner = c.id;
  }
  if (victim) {  // reset by a pool overflow while queued for the GPU
    fprintf(stderr, "%s serve: req %lld aborted: KV pool needed by an earlier "
            "request\n", log_ts(), (long long)g.req);
    hist.clear();
    mtp_live = false;
    finish(err_line());
    return;
  }
  if (m.nslots > 1)
    fprintf(stderr, "%s req %lld -> slot %d (%zu live tokens)\n", log_ts(),
            (long long)g.req, s, hist.size());
  crashdiag::note("req %lld -> slot %d (live %zu, prompt %zu, max %d, %s)", (long long)g.req, s,
                  hist.size(), g.ids.size(), g.max_tokens, g.sample ? "sampled" : "greedy");
  crashdiag::rearm();
  crash_test_point("gen");
  bool owns_cut = false;
  auto clear_cut = [&] {
    m.cut_abs = -1;
    m.cut_r_ = 0;
    m.cut_hit = m.cut_mtp_hit = false;
    if (owns_cut) {
      m.cut_busy = false;
      owns_cut = false;
    }
  };
  try {
    // KV reuse: longest common prefix of live state and this prompt.
    // Continue only when hist is a STRICT prefix (mcp == hist.size() and
    // new tokens follow); anything else falls to the SSD snapshot cache
    // and only then to a full recompute.
    size_t mcp = 0;
    while (mcp < hist.size() && mcp < g.ids.size() && hist[mcp] == g.ids[mcp])
      mcp++;
    bool cont = !hist.empty() && mcp == hist.size() && mcp < g.ids.size();
    int n_cached = cont ? (int)mcp : 0;
    // M-RoPE restrictions, decided before any state mutation:
    // KV reuse (cont) IS sound for M-RoPE — prefix positions are
    // suffix-independent and RoPE is applied before K is cached, so the
    // cached prefix is identical to what a recompute would produce —
    // but only when the cached image KV came from the SAME pixels:
    // placeholder ids are pixel-agnostic, so the patch-bytes hash
    // guards cont; a mismatch (or QWENOX_VISION_NOCONT) falls back to a
    // full recompute. MTP speculation IS allowed for M-RoPE: verify
    // positions come from the d_ropecs table (ensure_rope_cs handles
    // base=q+1 out-of-range rows with rope_delta_), and the drafter
    // paths rotate with *d_rdelta, so only acceptance rate is at stake.
    // Only the images the live prefix holds need the same pixels (its
    // request's grids/hash; none for a text prefix, which is text KV
    // whatever this request's images are — see has_image_tok): a
    // follow-up that keeps the earlier images and adds new ones continues.
    const auto& live_g = m.mrope_grids_;
    const bool live_imgs_same =
        live_g.size() <= g.mrope_grids.size() &&
        std::equal(live_g.begin(), live_g.end(), g.mrope_grids.begin()) &&
        fnv1a_patches(vpatches, live_g.size()) == m.img_hash_;
    if (!g.mrope_grids.empty() && cont &&
        ((!live_imgs_same && has_image_tok(hist.data(), mcp)) ||
         getenv("QWENOX_VISION_NOCONT"))) {
      fprintf(stderr, "%s vision: req %lld: live prefix %zu not reused "
              "(its images differ)\n", log_ts(), (long long)g.req, mcp);
      cont = false;
      n_cached = 0;
    }
    // Below the live slot: the RAM tier (rckpt), then the SSD tier
    // (kvsnap). The SSD match (index only, no IO) is looked up first: when
    // it covers a much longer prefix than the best RAM checkpoint, restoring
    // it beats prefilling the difference (kvsnap::prefer_over_ram).
    const bool is_mrope = !g.mrope_grids.empty();
    const uint64_t req_ih = is_mrope && !cont ? fnv1a_patches(vpatches) : 0;
    std::shared_ptr<kvsnap::CkEnt> ssd_ck;  // busy until restored or released
    if (!cont && !kvs.dir.empty())
      // M-RoPE requests: placeholder ids are pixel-agnostic, so the vision
      // tower, grids and patch-bytes hash are folded into the page-key seed;
      // a different image is simply a cache miss. + the text prefix of a
      // vision prompt (g_fp).
      ssd_ck = kvsnap::match(g.ids, kvsnap::seed_for(is_mrope, req_ih, g.mrope_grids),
                             kvsnap::g_fp);
    if (!cont && rckpt::enabled()) {
      // RAM checkpoint tier (llama.cpp-style PARTIAL_ONLY): only GDN
      // state is stored; validity requires the checkpoint tokens to be a
      // prefix of the live token stream (m.chist), which guarantees the
      // QSA KV/indexer prefix is still intact (paged: the checkpoint owns
      // its pages, any slot may restore it). B0: the pinned pages include
      // the MTP layer, so a checkpoint saved with the MTP layer in lockstep
      // restores it too and speculation stays available.
      auto r0 = std::chrono::steady_clock::now();
      int ci = rckpt::match(g.ids, m.chist, is_mrope,
                            is_mrope ? &g.mrope_grids : nullptr, req_ih);
      if (ci >= 0 && ssd_ck &&
          kvsnap::prefer_over_ram(*ssd_ck, rckpt::length(ci), rckpt::has_mtp(ci),
                                  (long long)g.req))
        ci = -1;
      bool ck_mtp = false;
      if (ci >= 0 && rckpt::restore(m, ci, &hist, &ck_mtp)) {
        mcp = hist.size();
        cont = true;
        n_cached = (int)mcp;
        mtp_live = ck_mtp;
        double rs = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - r0)
                        .count();
        fprintf(stderr, "%s rckpt: restored %zu tokens in %.2fs%s\n",
                log_ts(), mcp, rs, ck_mtp ? " (mtp live)" : "");
        crashdiag::note("req %lld rckpt restored %zu tokens", (long long)g.req, mcp);
      }
    }
    if (ssd_ck && cont) {  // the RAM tier won: drop the SSD match
      kvsnap::release(ssd_ck);
      ssd_ck.reset();
    }
    if (ssd_ck) {
      // SSD tier. A checkpoint saved with the MTP layer in lockstep restores
      // it too. restore() releases the match.
      std::shared_ptr<kvsnap::CkEnt> ck = std::move(ssd_ck);
      bool ck_mtp = false;
      if (kvsnap::restore(m, ck, g.ids, hist, &ck_mtp)) {  // hist := its tokens
        mcp = hist.size();
        cont = true;
        n_cached = (int)mcp;
        mtp_live = ck_mtp;
      } else {
        hist.clear();  // live state undefined after a failed restore
      }
    }
    if (!cont) m.reset_state();
    // Ngram proposals are request-local. Clear learned suffixes so a
    // previous request cannot collide with this prompt's hash history.
    if (g.drafter == 3 || g.drafter == 4) m.ngram.reset();
    m.ng_reset();  // gate + chunk level are per request too (reproducibility)
    // Vision requests expand their grids in-engine (validated against the
    // prompt's image token runs); plain requests drop stale mrope state
    // (a cont turn skips reset_state).
    if (!g.mrope_grids.empty()) {
      if (!m.set_mrope(g.ids, g.mrope_grids)) {
        fprintf(stderr, "vision: req %lld rejected: grids do not cover "
                "the image token runs\n", (long long)g.req);
        finish(err_line());
        return;
      }
      if (g.vimg > 0 &&
          !m.vision_embeds(g.ids, g.mrope_grids, vpatches, n_cached)) {
        finish(err_line());
        return;
      }
      vpatches.clear();
      vpatches.shrink_to_fit();  // ~50 MB/image: release before decode
    } else {
      m.clear_mrope();
    }
    std::unordered_set<int> eos(g.eos.begin(), g.eos.end());
    // CKPT tokens (e.g. <tool_call>, </think>): pause mid-decode right before
    // one commits and checkpoint that position (rckpt), so the next turn --
    // whose token-cache splice can only cut right after such an added token --
    // falls back a few tokens instead of to the previous prompt-end cut.
    // QWENOX_CKPT_TOKENS=0 disables; eos membership wins over a ckpt request.
    std::unordered_set<int> ckpt_toks(g.ckpt.begin(), g.ckpt.end());
    for (int e : g.eos) ckpt_toks.erase(e);
    if (!rckpt::enabled()) ckpt_toks.clear();
    {
      const char* e = getenv("QWENOX_CKPT_TOKENS");
      if (e && !atoi(e)) ckpt_toks.clear();
    }
    HostSampler smp;
    std::vector<float> lbuf;
    if (g.sample) {
      smp.temp = g.temp;
      smp.top_p = g.top_p;
      smp.min_p = g.min_p;
      smp.top_k = g.top_k;
      smp.rng = g.seed;
      smp.presence = g.presence;
      smp.frequency = g.frequency;
      smp.bias = g.bias;
      // penalties count GENERATED tokens only: hist starts empty and
      // sample() records each emitted token
    }
    auto t0 = std::chrono::steady_clock::now();
    int cur;
    float cur_lp = 0.0f;
    // MTP speculation is available for greedy and rejection-sampled
    // requests when its KV can be made complete: a fresh prefill, or a
    // restore (rckpt/kvsnap) of a state saved with the MTP layer live.
    // QWENOX_SPEC_GAMMA 覆盖；未设时按模式自选：greedy 4（09-28 重扫 γ=4：MTP 8K
    // +6%/32K +9%，chain-greedy +5.6%/+11.5%），采样 3（拒绝采样验收率低，
    // 深链不划算：8K 三种子 γ=3 46.7 vs γ=4 44.8 tok/s，32K 持平）。
    // 09-29 起采样 auto = 自适应 γ（spec_gamma 传 0，见 Model::GammaCtl 两状态版；
    // tools/gamma_sample_check.sh 18 样本：真实长文 1.004× γ3，代码/重复文本 1.040×），
    // QWENOX_SPEC_ADAPT=0 退回固定 3。greedy auto 固定 4：固定轨迹成对比较
    // （tools/gamma_force_check.sh）里自适应与 γ4 相差 ±2% 以内，没有可拿的收益，
    // 固定 γ 还让 greedy 输出不随控制器状态变化（gamma_real_check 里的 0.92 是
    // 换 γ 导致轨迹分叉的噪声）；QWENOX_SPEC_ADAPT_GREEDY=1 让 greedy 也走自适应（试验用）。
    static const char* sg_env = getenv("QWENOX_SPEC_GAMMA");
    static const bool sg_adapt = [] {
      const char* e = getenv("QWENOX_SPEC_ADAPT");
      return !e || e[0] != '0';
    }();
    static const bool sg_adapt_greedy = [] {
      const char* e = getenv("QWENOX_SPEC_ADAPT_GREEDY");
      return e && e[0] == '1';
    }();
    const int spec_gamma =
        sg_env ? std::max(1, std::min(8, atoi(sg_env)))
        : g.sample ? (sg_adapt ? 0 : 3)
                   : (sg_adapt_greedy ? 0 : 4);
    if (!sg_env) {  // 让 auto 默认在 serve 日志里可观测（每模式只打一次）
      static bool said[2] = {false, false};
      if (!said[g.sample]) {
        said[g.sample] = true;
        if (spec_gamma == 0)
          fprintf(stderr, "[gamma] QWENOX_SPEC_GAMMA 未设：%s 请求自适应 gamma（起点 %d）\n",
                  g.sample ? "采样" : "greedy", g.sample ? 3 : 4);
        else
          fprintf(stderr, "[gamma] QWENOX_SPEC_GAMMA 未设：%s 请求默认 gamma=%d\n",
                  g.sample ? "采样" : "greedy", spec_gamma);
      }
    }
    if (g.drafter == 1 && !m.mtp_avail) {
      finish(err_line());
      return;
    }
    const bool use_ngram = g.drafter == 3 && !g.sample;
    // Chain mode (llama.cpp-style fallback): ngram drafts every round it
    // can, MTP takes the rounds where the table has nothing. Sampled
    // requests run spec_loop_chain_sample (ngram rounds re-draw from the
    // target distribution; QWENOX_NOSPEC_SAMPLE forces plain serial, same
    // as use_spec). The MTP leg requires a complete MTP KV: a fresh
    // prefill always has one; a cont/restored turn only if the previous
    // request (or the restored checkpoint) left MTP live.
    const bool use_chain = g.drafter == 4 &&
                           (!g.sample ||
                            getenv("QWENOX_NOSPEC_SAMPLE") == nullptr);
    const bool chain_mtp = use_chain && m.mtp_avail &&
                           (!cont || mtp_live) &&
                           getenv("QWENOX_NOSPEC") == nullptr;
    const bool use_spec = !use_ngram && !use_chain && m.mtp_avail &&
                          g.drafter != 0 && g.drafter != 2 &&
                          g.drafter != 3 && g.drafter != 4 &&
                          (!cont || mtp_live) &&
                          (!g.sample ||
                           getenv("QWENOX_NOSPEC_SAMPLE") == nullptr) &&
                          getenv("QWENOX_NOSPEC") == nullptr;
    // DFlash2 (drafter 2): 5-block masked draft + selector. v1 is greedy,
    // single-slot, and requires a fresh prompt: target taps only exist for
    // positions this process prefilled (a restored/continued prefix has no
    // tap ring). Anything else falls back to plain serial decode.
    const bool use_df2 = g.drafter == 2 && m.df2.on && m.nslots == 1 &&
                         !cont && ckpt_toks.empty() &&
                         getenv("QWENOX_NOSPEC") == nullptr;
    // Hybrid (QWENOX_DF2_HYBRID=1, greedy, needs MTP weights): the MTP layer
    // is prefilled too and every round picks the drafter with the better
    // recent tokens/ms (df2_spec_loop); MTP stays in lockstep for the next
    // turn. Off by default: 10-20 A/B pooled 0.98x of pure MTP (the per-round
    // winner changes faster than any history estimate tracks).
    static const bool df2_hyb_env = [] {
      const char* e = getenv("QWENOX_DF2_HYBRID");
      return e && e[0] == '1';
    }();
    const bool df2_hyb = use_df2 && !g.sample && m.mtp_avail && df2_hyb_env;
    if (g.drafter == 2) {
      static bool said_df2 = false;
      if (!said_df2 && !use_df2) {
        said_df2 = true;
        fprintf(stderr,
                "%s dflash2: drafter 2 requested but unavailable "
                "(on=%d slots=%d sample=%d cont=%d ckpts=%zu) -> serial\n",
                log_ts(), (int)m.df2.on, m.nslots, (int)g.sample, (int)cont,
                ckpt_toks.size());
      }
    }
    m.last_spec_rounds = m.last_spec_commit = m.last_spec_rollbacks = 0;
    // Prompt ingest. A failure here (or a KvOverflow anywhere below) lands
    // in the catch at the end: the slot is reset, the client gets D error.
    if (use_spec || chain_mtp) {
      m.mtp_tap_capture = true;  // prefill_batch ingests MTP per chunk
      if (cont) {
        // Boundary: MTP position mcp-1 pairs the live d_R (stream at
        // the last ingested position) with the first new token.
        std::vector<int> bt{g.ids[mcp]};
        m.mtp_ingest_b(m.d_R, bt, 1, (int)mcp - 1);
      }
    } else if (use_df2) {
      m.df2.cap_mode = 1;  // capture tap ring + build draft KV per chunk
      if (df2_hyb) m.mtp_tap_capture = true;  // MTP KV over the prompt too
    }
    // Frontend-hinted snapshot points (SNAPS/SNAP/SNAP2) that fall
    // inside the uncached range: pause the prefill there, checkpoint,
    // resume. The next turn re-renders history as TEXT whose ids match
    // the prompt only up to these boundaries (the assistant opener and
    // generated text retokenize differently), so they are the cuts
    // that hit. Checkpoints go to the RAM tier (rckpt) and are captured
    // for the SSD tier (kvsnap); the SSD captures are committed at the end
    // of the turn together with the final state, where only the pages no
    // cached file holds yet are written (content-addressed, shared).
    std::vector<size_t> cuts;
    if (!kvs.dir.empty() || rckpt::enabled()) {
      // Prefill starts here. A full recompute starts at 0 while mcp still
      // holds the lcp with the stale live tokens; that must not drop the
      // cuts below it.
      const size_t base = cont ? mcp : 0;
      for (size_t cc : g.snaps)
        if (cc > base && cc < g.ids.size() && cc >= kvs.min_tokens)
          cuts.push_back(cc);
      for (size_t cc : {g.snap2, g.snap})
        if (cc > base && cc < g.ids.size() && cc >= kvs.min_tokens)
          cuts.push_back(cc);
      if (g.mrope_grids.empty())
        if (size_t cc = fork_cut(g.ids, sl.fork_lcp, base, kvs.min_tokens, (long long)g.req))
          cuts.push_back(cc);
    }
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    if (!cont) hist.clear();
    size_t done_upto = cont ? mcp : 0;
    // QWENOX_VIZ: token log for the VIZ verb; positions < done_upto keep
    // their records only when the slot continues its live state
    viz::begin(s, (long long)g.req, g.ids, (int)done_upto, cont);
    // Arm the 1s prefill progress reporter for the prompt ingest only
    // (spec-verify batches never see it armed). prog_base accumulates
    // across the kvsnap cut segments inside prefill_chunk.
    m.prog_total = (int)(g.ids.size() - done_upto);
    m.pf_end = (int)g.ids.size();  // prompt ingest: scheduling sees what is left
    m.prog_base = 0;
    m.prog_last_est = 0;
    m.prog_req = g.req;
    m.prog_t0 = m.prog_last = std::chrono::steady_clock::now();
    crashdiag::note("req %lld prefill %zu..%zu (cached %d, %s, spec %d chain %d df2 %d, cuts %zu)",
                    (long long)g.req, done_upto, g.ids.size(), n_cached, cont ? "cont" : "fresh",
                    (int)use_spec, (int)chain_mtp, (int)use_df2, cuts.size());
    // live spec counters reset per request, or a serial decode after a
    // speculative one would report the previous request's acceptance
    m.live_spec_rounds = m.live_spec_commits = m.live_spec_proposed = 0;
    std::vector<kvsnap::Pending> pend;  // SSD captures, committed at turn end
    // prefill 期间的断连中断：armed 仅覆盖下面的 ingest 段（cuts 循环 + 最后一段），
    // prefill_batch 在分段间检查并抛 ReqCancelled。ingest 一结束立即 disarm——
    // 后面的投机 verify 也走 prefill_batch，但回合本地状态存活时不能抛。
    m.abort_flag = &c.cancel;
    static const bool cut_prof = getenv("QWENOX_PROF") != nullptr;
    auto cut_tp = t0;
    auto cut_lap = [&](const char* what) {
      if (!cut_prof) return;
      CK(hipStreamSynchronize(g_str));
      auto now = std::chrono::steady_clock::now();
      fprintf(stderr, "cutprof: %-12s %8.1f ms\n", what,
              std::chrono::duration<double, std::milli>(now - cut_tp).count());
      cut_tp = now;
    };
    cut_lap("pre-cuts");
    // aggr 10-05: the LAST cut is taken in-pass (no separate tiny tail pass);
    // earlier cuts keep the split path. QWENOX_CUT_INPASS=0 disables.
    size_t inpass_cc = 0;
    if (!cuts.empty() && m.cut_snap_supported() && g.mrope_grids.empty() && !m.cut_busy) {
      inpass_cc = cuts.back();
      cuts.pop_back();
      m.cut_busy = true;  // released after the save below
      owns_cut = true;
    }
    for (size_t cc : cuts) {
      std::vector<int> seg(g.ids.begin() + done_upto, g.ids.begin() + cc);
      cur = m.prefill_batch(seg, (int)done_upto);
      cut_lap("seg-prefill");
      hist.assign(g.ids.begin(), g.ids.begin() + cc);
      // The segment leaves MTP position cc-1 pending (mtp_pos == pos-1,
      // d_R == h_{cc-1}): checkpoint that lockstep state, then pair h_{cc-1}
      // with token cc, which the next segment's ingest would otherwise skip.
      rckpt::save(m, hist, use_spec || chain_mtp);
      cut_lap("rckpt-save");
      if (!kvs.dir.empty()) kvsnap::capture(m, hist, use_spec || chain_mtp, pend);
      cut_lap("kvsnap-cap");
      if (use_spec || chain_mtp) {
        std::vector<int> bt{g.ids[cc]};
        m.mtp_ingest_b(m.d_R, bt, 1, (int)cc - 1);
      }
      done_upto = cc;
      m.yield_point();  // consistent state between segments
      cut_lap("mtp+yield");
    }
    std::vector<int> seg(g.ids.begin() + done_upto, g.ids.end());
    if (inpass_cc) {
      m.cut_alloc();
      m.cut_abs = (int)inpass_cc;
      m.cut_hit = m.cut_mtp_hit = false;
    }
    cur = m.prefill_batch(seg, (int)done_upto);
    m.abort_flag = nullptr;  // ingest 完成：后续 verify 批次不许抛 ReqCancelled
    cut_lap("last-prefill");
    if (inpass_cc) {
      const bool mtp_cut = (use_spec || chain_mtp) && m.cut_mtp_hit;
      if (m.cut_hit) {
        std::vector<int> chist(g.ids.begin(), g.ids.begin() + inpass_cc);
        m.with_cut_state(mtp_cut, [&] {
          rckpt::save(m, chist, mtp_cut);
          if (!kvs.dir.empty()) kvsnap::capture(m, chist, mtp_cut, pend);
        });
      } else {
        fprintf(stderr, "%s cut: in-pass snapshot at %zu missed (req %lld)\n", log_ts(),
                inpass_cc, (long long)g.req);
      }
      clear_cut();
      cut_lap("cut-save");
    }
    m.prog_total = 0;
    m.pf_end = 0;
    m.mtp_tap_capture = false;  // decode/verify must not capture taps
    if (use_df2) m.df2.cap_mode = 0;  // prompt tap ring is complete
    hist = g.ids;  // prompt fully ingested on both paths
    if (use_spec || chain_mtp || df2_hyb) mtp_live = true;  // MTP KV complete & in lockstep
    if (g.sample) {  // d_logits still holds the last position's logits
      m.get_logits(lbuf);
      cur = smp.sample(lbuf, &cur_lp);
    }
    cut_lap("sample");
    double pre_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    kvsnap::note_prefill(g.ids.size() - (size_t)std::max(0, n_cached), pre_ms);
    // Phase 1: snapshot only at end-of-turn. This removes the first-token
    // stall while the full asynchronous capture path is being designed.
    // prompt-end RAM checkpoint (cheap, deduped)
    rckpt::save(m, hist, use_spec || chain_mtp);
    auto d0 = std::chrono::steady_clock::now();
    int ngen = 0;
    bool dead = false;
    std::string reason = "length";
    // 1s-cadence decode stats (2026-09-13): instantaneous + average t/s,
    // with the live MTP acceptance the spec loops bump every round.
    // committed drafts = commits - rounds (each round also emits the
    // verify/bonus token), acceptance = committed / proposed.
    auto dl_t0 = std::chrono::steady_clock::now();
    auto dl_last = dl_t0;
    int dl_last_ngen = 0;
    auto dl_tick = [&]() {
      auto now = std::chrono::steady_clock::now();
      double iv = std::chrono::duration<double>(now - dl_last).count();
      if (iv < 1.0) return;
      double el = std::chrono::duration<double>(now - dl_t0).count();
      if (m.live_spec_rounds > 0) {
        int acc_n = m.live_spec_commits - m.live_spec_rounds;
        fprintf(stderr,
                "%s req %lld | n_decoded = %d | t = %.0f s | tg = %.1f "
                "tok/s (avg %.1f) | accept %.1f%% (%d/%d)\n",
                log_ts(), (long long)g.req, ngen, el,
                (ngen - dl_last_ngen) / iv, el > 0 ? ngen / el : 0.0,
                m.live_spec_proposed
                    ? 100.0 * acc_n / m.live_spec_proposed
                    : 0.0,
                acc_n, m.live_spec_proposed);
      } else {
        fprintf(stderr,
                "%s req %lld | n_decoded = %d | t = %.0f s | tg = %.1f "
                "tok/s (avg %.1f)\n",
                log_ts(), (long long)g.req, ngen, el,
                (ngen - dl_last_ngen) / iv, el > 0 ? ngen / el : 0.0);
      }
      perfsnap::write_decode((uint64_t)g.req, (ngen - dl_last_ngen) / iv,
                             el > 0 ? ngen / el : 0.0,
                             m.live_spec_rounds > 0 && m.live_spec_proposed
                                 ? 100.0 * (m.live_spec_commits - m.live_spec_rounds) /
                                       (double)m.live_spec_proposed
                                 : -1.0);
      dl_last = now;
      dl_last_ngen = ngen;
      crashdiag::note("req %lld decode n=%d pos=%d rounds=%d", (long long)g.req, ngen, m.pos,
                      m.live_spec_rounds);
    };
    // Token line; the X verb (reader thread) raises `cancel`.
    auto send_tok = [&](int tok, bool with_lp, float lp) -> bool {
      char tb[96];
      int tl = with_lp ? snprintf(tb, sizeof tb, "T %lld %d %.6f\n",
                                  (long long)g.req, tok, (double)lp)
                       : snprintf(tb, sizeof tb, "T %lld %d\n",
                                  (long long)g.req, tok);
      if (!c.send(std::string(tb, tl))) {
        dead = true;
        return false;
      }
      ngen++;
      viz::token(s, tok);
      return true;
    };
    // CKPT pause/resume. The emit lambdas below return false on a ckpt token
    // BEFORE sending it; the spec loops roll the round back to exclude it
    // (their eos semantics), leaving the trunk/MTP state exactly at that
    // token's position. ckpt_pause checkpoints that position (the next turn's
    // token-cache splice can only cut right after such an added token, so
    // without this a retokenization mismatch inside a tool call re-prefills
    // from the previous prompt-end cut), then ingests the token serially --
    // MTP boundary pairing first, while d_R is still h_{pos-1} -- and the
    // caller re-enters the loop. The loops reset their spec counters at
    // entry, so they are accumulated across pauses and written back.
    int pause_tok = -1;
    float pause_lp = 0.0f;
    int spec_r = 0, spec_c = 0, spec_p = 0;
    // The one pause checkpoint per turn that also goes to the SSD tier (from
    // its RAM copy, at the end of the turn: kvsnap::capture_ram), so a
    // re-rendered reply still resumes inside it after a restart or RAM
    // eviction. The first pause on g.ckpt[0] -- the API sends <tool_call>
    // first, and a re-serialized tool call is where a re-rendered reply
    // usually stops matching (later pauses sit past it) -- else the last one.
    size_t ssd_pause_n = 0;
    bool ssd_pause_fixed = false;
    auto note_pause = [&](size_t n, int tok) {
      if (ssd_pause_fixed) return;
      ssd_pause_n = n;
      ssd_pause_fixed = !g.ckpt.empty() && tok == g.ckpt[0];
    };
    auto ckpt_pause = [&](bool mtp_ok) -> bool {
      hist.assign(m.chist.begin(), m.chist.end());
      if (rckpt::save(m, hist, mtp_ok && mtp_live)) {
        fprintf(stderr, "%s ckpt: saved %zu tokens (before token %d, req %lld)\n",
                log_ts(), hist.size(), pause_tok, (long long)g.req);
        note_pause(hist.size(), pause_tok);
      }
      if (!send_tok(pause_tok, g.sample && g.logprobs, pause_lp)) return false;
      if (g.sample) smp.remember(pause_tok);  // the loop's sampler view is gone
      dl_tick();
      if (mtp_ok && mtp_live) {
        std::vector<int> bt{pause_tok};
        m.mtp_ingest_b(m.d_R, bt, 1, (int)hist.size() - 1);
      }
      cur = m.forward(pause_tok);
      hist.push_back(pause_tok);
      if (g.sample) {
        m.get_logits(lbuf);
        cur = smp.sample(lbuf, &cur_lp);
      }
      m.yield_point();
      if (eos.count(cur)) {
        reason = "done";
        return false;
      }
      if (ngen >= g.max_tokens) {  // pause_tok hit the budget: cur unsent
        reason = "length";
        return false;
      }
      // spec-loop entry invariant: cur is already streamed, not yet ingested.
      if (!send_tok(cur, g.sample && g.logprobs, cur_lp)) return false;
      return true;
    };
    if (use_ngram || use_chain) {
      // chain with a live MTP keeps it live (the loop patches its KV
      // every ngram round); pure/degraded ngram is trunk-only decode and
      // invalidates the previous MTP state.
      const bool chain = use_chain && chain_mtp;
      if (!chain) mtp_live = false;
      if (!m.ple_on) m.chist.assign(g.ids.begin(), g.ids.end());
      if (eos.count(cur)) {
        reason = "done";
      } else {
        send_tok(cur, g.sample && g.logprobs, cur_lp);
      }
      auto emit_ng = [&](int tok) -> bool {
        if (ngen >= g.max_tokens) return false;
        if (eos.count(tok)) { reason = "done"; return false; }
        if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = 0.0f; return false; }
        if (!send_tok(tok, false, 0.0f)) return false;
        dl_tick();
        return !cancel;
      };
      auto emit_ng_smp = [&](int tok, float lp) -> bool {
        if (ngen >= g.max_tokens) return false;
        if (eos.count(tok)) { reason = "done"; return false; }
        if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = lp; return false; }
        if (!send_tok(tok, g.logprobs, lp)) return false;
        dl_tick();
        return !cancel;
      };
      if (!dead && !eos.count(cur)) {
        for (;;) {
          pause_tok = -1;
          if (g.sample)
            m.spec_loop_chain_sample(cur, g.max_tokens - 1, spec_gamma, smp,
                                     chain_mtp, emit_ng_smp, g.seeded);
          else if (chain) m.spec_loop_chain(cur, g.max_tokens - 1, spec_gamma, emit_ng);
          else m.spec_loop_ngram(cur, g.max_tokens - 1, m.ngram.n_max, emit_ng);
          spec_r += m.last_spec_rounds;
          spec_c += m.last_spec_commit;
          spec_p += m.last_spec_proposed;
          if (pause_tok < 0 || dead || cancel || reason == "done") break;
          if (!ckpt_pause(chain)) break;
        }
        m.last_spec_rounds = spec_r;
        m.last_spec_commit = spec_c;
        m.last_spec_proposed = spec_p;
      }
      hist.assign(m.chist.begin(), m.chist.end());
    } else if (use_df2) {
      // DFlash2 greedy speculative decode. The loop verifies every round
      // against the trunk, so committed ids are exactly serial-decode ids.
      // Hybrid keeps the MTP layer in lockstep; plain DFlash2 leaves it stale.
      mtp_live = df2_hyb;
      if (eos.count(cur)) {
        reason = "done";
      } else if (send_tok(cur, g.sample && g.logprobs, cur_lp)) {
        for (;;) {
          pause_tok = -1;
          if (g.sample) {
            auto emit = [&](int tok, float lp) -> bool {
              if (ngen >= g.max_tokens) return false;
              if (eos.count(tok)) {
                reason = "done";
                return false;
              }
              if (!send_tok(tok, g.logprobs, lp)) return false;
              dl_tick();
              return !cancel;
            };
            m.df2_spec_loop_sample(cur, g.max_tokens - 1, smp, emit);
          } else {
            auto emit = [&](int tok) -> bool {
              if (ngen >= g.max_tokens) return false;
              if (eos.count(tok)) {
                reason = "done";
                return false;
              }
              if (!send_tok(tok, false, 0.0f)) return false;
              dl_tick();
              return !cancel;
            };
            m.df2_spec_loop(cur, g.max_tokens - 1, spec_gamma, emit, df2_hyb,
                            spec_gamma);
          }
          spec_r += m.last_spec_rounds;
          spec_c += m.last_spec_commit;
          spec_p += m.last_spec_proposed;
          break;  // no ckpt-pause tokens on the dflash2 path
        }
        m.last_spec_rounds = spec_r;
        m.last_spec_commit = spec_c;
        m.last_spec_proposed = spec_p;
        hist.assign(m.chist.begin(), m.chist.end());
      }
    } else if (use_spec) {
      // Stream each token as the spec loop commits it. eos/cancel/length
      // return false, rolling the round back so trunk and MTP stop at the
      // last token actually sent to the client.
      if (eos.count(cur)) {
        reason = "done";
      } else if (send_tok(cur, g.sample && g.logprobs, cur_lp)) {
        for (;;) {
          pause_tok = -1;
          if (g.sample) {
            auto emit = [&](int tok, float lp) -> bool {
              if (ngen >= g.max_tokens) return false;
              if (eos.count(tok)) {
                reason = "done";
                return false;
              }
              if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = lp; return false; }
              if (!send_tok(tok, g.logprobs, lp)) return false;
              dl_tick();
              return !cancel;
            };
            m.spec_loop_sample(cur, g.max_tokens - 1, spec_gamma, smp, emit);
          } else {
            auto emit = [&](int tok) -> bool {
              if (ngen >= g.max_tokens) return false;
              if (eos.count(tok)) {
                reason = "done";
                return false;
              }
              if (ckpt_toks.count(tok)) { pause_tok = tok; pause_lp = 0.0f; return false; }
              if (!send_tok(tok, false, 0.0f)) return false;
              dl_tick();
              return !cancel;
            };
            {
              VrActive vr_act;  // rendezvous membership; every exit counts
              // D2-S3c: draft rendezvous membership, same scope and exit
              // discipline. dr_worker_ok is the submit predicate too, so the
              // count never includes a worker that cannot submit.
              DrActive dr_act(m.dr_worker_ok());
              m.spec_loop(cur, g.max_tokens - 1, spec_gamma, emit);
            }
          }
          spec_r += m.last_spec_rounds;
          spec_c += m.last_spec_commit;
          spec_p += m.last_spec_proposed;
          if (pause_tok < 0 || dead || cancel || reason == "done") break;
          if (!ckpt_pause(true)) break;
        }
        m.last_spec_rounds = spec_r;
        m.last_spec_commit = spec_c;
        m.last_spec_proposed = spec_p;
        // Trunk ingested exactly chist tokens; hist must match for the
        // next turn's KV-reuse prefix check.
        hist.assign(m.chist.begin(), m.chist.end());
      }
    } else {
      mtp_live = false;  // plain decode leaves the MTP layer stale
    }
    if (!use_spec && !use_ngram && !use_chain && !use_df2)
      for (int i = 0; i < g.max_tokens; i++) {
        if (eos.count(cur)) {
          reason = "done";
          break;
        }
        if (!send_tok(cur, g.logprobs, cur_lp)) break;
        if (i + 1 < g.max_tokens) {
          int fed = cur;  // the token forward() ingests (pre-sampling)
          cur = m.forward(fed);
          hist.push_back(fed);
          // Serial decode advances one token at a time: checkpoint right
          // after a ckpt token (mtp is stale on this path).
          if (!ckpt_toks.empty() && ckpt_toks.count(fed) &&
              rckpt::save(m, hist, false)) {
            fprintf(stderr, "%s ckpt: saved %zu tokens (after token %d, req %lld)\n",
                    log_ts(), hist.size(), fed, (long long)g.req);
            note_pause(hist.size(), fed);
          }
          if (g.sample) {
            m.get_logits(lbuf);
            cur = smp.sample(lbuf, &cur_lp);
          }
        }
        dl_tick();
        if (cancel) break;
        m.yield_point();
      }
    if (dead) {  // client gone: no D line, no end-of-turn snapshot
      perfsnap::write_idle();
      c.in_flight = false;
      sock_shutdown(c.fd);
      return;
    }
    double dec_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - d0)
                        .count();
    if (cancel)
      snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)g.req);
    else if ((use_spec || use_ngram || use_chain || use_df2) &&
             m.last_spec_rounds > 0)
      snprintf(db, sizeof db,
               "D %lld %s %zu %d %.1f %.1f %d %d %d %d %d\n",
               (long long)g.req, reason.c_str(), g.ids.size(), ngen, pre_ms,
               ngen > 1 ? dec_ms : 0.0,
               use_chain ? (chain_mtp ? 4 : 3)
                         : (use_ngram ? 3 : (use_df2 ? 2 : 1)),
               m.last_spec_rounds,
               m.last_spec_commit, n_cached, m.last_spec_proposed);
    else
      // trailing "0 0 0 <n_cached> 0": drafter/rounds/commit placeholders
      // so n_cached lands in the protocol's cache column (parts[10]);
      // the last 0 is the proposed-drafts column (parts[11])
      snprintf(db, sizeof db, "D %lld %s %zu %d %.1f %.1f 0 0 0 %d 0\n",
               (long long)g.req, reason.c_str(), g.ids.size(), ngen, pre_ms,
               ngen > 1 ? dec_ms : 0.0, n_cached);
    finish(db);
    {
      // llama.cpp-style per-request timing block (human-facing; the wire
      // D line above stays the machine-readable record).
      size_t n_proc =
          g.ids.size() - (n_cached > 0 ? (size_t)n_cached : 0);
      fprintf(stderr,
              "%s req %lld | prompt eval = %.1f ms / %zu tokens "
              "(%.2f ms/token, %.1f tok/s)%s\n",
              log_ts(), (long long)g.req, pre_ms, n_proc,
              n_proc ? pre_ms / n_proc : 0.0,
              pre_ms > 0 ? n_proc / (pre_ms / 1000.0) : 0.0,
              n_cached ? " | cached" : "");
      if (ngen > 0)
        fprintf(stderr,
                "%s req %lld | decode eval = %.1f ms / %d tokens "
                "(%.2f ms/token, %.1f tok/s) | %s\n",
                log_ts(), (long long)g.req, dec_ms, ngen,
                dec_ms > 0 ? dec_ms / ngen : 0.0,
                dec_ms > 0 ? ngen / (dec_ms / 1000.0) : 0.0,
                cancel ? "cancel" : reason.c_str());
      else
        fprintf(stderr, "%s req %lld | decode eval = no tokens | %s\n",
                log_ts(), (long long)g.req,
                cancel ? "cancel" : reason.c_str());
      crashdiag::note("req %lld done gen=%d (%s)", (long long)g.req, ngen,
                      cancel ? "cancel" : reason.c_str());
      if ((use_spec || use_ngram || use_chain) && m.last_spec_rounds > 0) {
        int acc_n = m.live_spec_commits - m.live_spec_rounds;
        fprintf(stderr,
                "%s req %lld | draft acceptance = %.3f (%d accepted / %d "
                "proposed), mean len = %.2f, drafter %s\n",
                log_ts(), (long long)g.req,
                m.live_spec_proposed
                    ? (double)acc_n / m.live_spec_proposed
                    : 0.0,
                acc_n, m.live_spec_proposed,
                m.last_spec_rounds
                    ? (double)m.live_spec_commits / m.last_spec_rounds
                    : 0.0,
                use_chain ? (chain_mtp ? "chain" : "ngram")
                          : (use_ngram ? "ngram" : "mtp"));
      }
      // D2-S2: rendezvous pool snapshot + batch-size histogram (histogram
      // buckets = segments per batch, 8 = 8+).
      if (m.vseg_pool_fn && g_vr.batches) {
        std::lock_guard<std::mutex> lk(g_vr.mu);
        fprintf(stderr,
                "%s [vr] batches=%ld multi=%ld solo=%ld timeouts=%ld "
                "fallbacks=%ld | segs 1:%ld 2:%ld 3:%ld 4:%ld 5:%ld 6:%ld "
                "7:%ld 8+:%ld\n",
                log_ts(), g_vr.batches, g_vr.batches - g_vr.solo, g_vr.solo,
                g_vr.timeouts, g_vr.fallbacks, g_vr.hist[1], g_vr.hist[2],
                g_vr.hist[3], g_vr.hist[4], g_vr.hist[5], g_vr.hist[6],
                g_vr.hist[7], g_vr.hist[8]);
      }
    }
    perfsnap::write_idle();  // tooltip 下一轮轮询即回归纯地址
    viz::finish(s);
    // SSD prompt cache: capture the end-of-turn state, then commit it with
    // the cut captures (new pages + MTP sections + checkpoints go to the
    // background writer). Cancelled turns are valid too -- hist tracks
    // exactly the ingested tokens. The MTP layer is saved only when live.
    // + this turn's SSD pause checkpoint (QWENOX_KVSNAP_PAUSE=0: none).
    if (!kvs.dir.empty()) {
      if (kvs.pause_ckpt && ssd_pause_n >= kvs.min_tokens)
        kvsnap::capture_ram(m, hist, ssd_pause_n, pend);
      if (hist.size() >= kvs.min_tokens) kvsnap::capture(m, hist, mtp_live, pend);
      kvsnap::commit(m, hist, pend);
    }
  } catch (const ReqCancelled& e) {
    // 客户端在 prompt 预填期间断连/取消：decode 取消的同款收尾，D cancel 行。
    // 已写入的分段 KV 属槽位持久状态，直接 reset 回收（同 KvOverflow 注释的语义）。
    fprintf(stderr, "%s serve: req %lld %s\n", log_ts(), (long long)g.req,
            e.what());
    m.abort_flag = nullptr;
    m.prog_total = 0;
    m.pf_end = 0;
    clear_cut();
    m.mtp_tap_capture = m.spec_snap = m.ngram_verify_active = false;
    m.reset_state();
    hist.clear();
    mtp_live = false;
    char cb[96];
    snprintf(cb, sizeof cb, "D %lld cancel 0 0 0.0 0.0\n", (long long)g.req);
    finish(cb);
    perfsnap::write_idle();
    viz::finish(s);
  } catch (const std::exception& e) {
    // prefill/decode failure, or KvOverflow (pool exhausted by concurrent
    // requests: this one is the later): the partial state is unknown, so
    // drop it (returning the slot's pages to the pool) and fail the request
    fprintf(stderr, "%s serve: req %lld failed: %s\n", log_ts(),
            (long long)g.req, e.what());
    crashdiag::note("req %lld failed: %.80s", (long long)g.req, e.what());
    m.abort_flag = nullptr;
    m.prog_total = 0;
    m.pf_end = 0;
    clear_cut();
    m.mtp_tap_capture = m.spec_snap = m.ngram_verify_active = false;
    m.reset_state();
    hist.clear();
    mtp_live = false;
    finish(err_line());
    perfsnap::write_idle();
    viz::finish(s);
  }
}

// Connection reader: control verbs inline, GEN handed to a worker thread.
static void conn_main(GpuModel& m, sockfd_t cfd, const kvsnap::Cfg& kvs) {
  hip_use_selected_device();
  Conn c;
  c.fd = cfd;
  c.id = ++g_conn_id;
  LineBuf lb{cfd, {}};
  std::thread worker;
  std::string line;
  bool alive = true;
  int bad_lines = 0;  // unrecognized lines seen on this connection
  // Connection log policy: the tray/console liveness probe (one SNAP or PING
  // per connection, every 2 s) stays silent. "connected" is printed when the
  // first other verb arrives (or the first unrecognized line), and only then
  // is the matching "disconnected" printed. A connection that closes without
  // sending any line at all is logged too: no internal client does that, so
  // it points at a broken client or a foreign scanner.
  bool announced = false, any_line = false;
  while (alive && lb.readline(line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string verb;
    ss >> verb;
    any_line = true;
    if (!announced && verb != "PING" && verb != "SNAP") {
      announced = true;
      fprintf(stderr, "%s serve: conn %llu: client connected (first verb %.16s)\n",
              log_ts(), (unsigned long long)c.id, verb.c_str());
      crashdiag::note("conn %llu connected (%.16s)", (unsigned long long)c.id, verb.c_str());
    }
    if (verb == "PING") {
      alive = c.send("PONG\n");
    } else if (verb == "INFO") {
      char ib[160];
      // fields: mtp draft_head ctx spec_rows default drafter_weights dflash2
      //         cache_mb cache_align kv_slots slot_ctx cache_mode sampling
      //         kv_pool_tokens
      // The default drafter is the chain (ngram first, MTP fallback) unless
      // QWENOX_DRAFTER overrides; draft_head is a separate shortlist readout
      // and is not part of this checkpoint. kv_slots: concurrent sequences.
      // kv_pool_tokens: KV total shared by all slots (paged mode); without
      // paging each sequence owns slot_ctx, so report nslots * slot_ctx.
      const long long kv_pool_tokens =
          m.kv_paged ? (long long)m.kv_pool_pages * KV_PAGE
                     : (long long)m.nslots * m.maxctx;
      snprintf(ib, sizeof ib, "I %d 0 %d 8 %d %d %d 0 0 %d %d 1 1 %lld\n",
               m.mtp_avail ? 1 : 0, m.maxctx,
               env_default_drafter() >= 0 ? env_default_drafter()
                                          : (m.mtp_avail ? 1 : 0),
               m.mtp_avail ? 1 : 0, m.df2.on ? 1 : 0, m.nslots, m.maxctx,
               kv_pool_tokens);
      alive = c.send(ib);
    } else if (verb == "MEM") {
      EngineMemStats ms = memstats_snapshot();
      char mb[512];
      // M version device_cur device_peak registered_cur pinned_cur
      //   pinned_peak committed_cur hip_free hip_total hip_delta rss locked
      snprintf(mb, sizeof mb,
               "M 1 %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu\n",
               (unsigned long long)ms.device_current,
               (unsigned long long)ms.device_peak,
               (unsigned long long)ms.registered_current,
               (unsigned long long)ms.pinned_current,
               (unsigned long long)ms.pinned_peak,
               (unsigned long long)ms.committed_current,
               (unsigned long long)ms.hip_free,
               (unsigned long long)ms.hip_total,
               (unsigned long long)ms.hip_used_since_start,
               (unsigned long long)ms.process_rss,
               (unsigned long long)ms.process_locked);
      alive = c.send(mb);
    } else if (verb == "KVSYNC") {
      // wait until every in-flight turn committed and the SSD writer is
      // idle (tests/ops); replies "K <ckpts> <pages> <mtp> <bytes>"
      alive = c.send(kvs.dir.empty() ? std::string("K 0 0 0 0\n") : kvsnap::sync_line());
    } else if (verb == "SNAP") {
      // 实时性能快照（托盘 tooltip 3s 轮询）：P ts_ms state req pp_toks
      // done total tg_inst tg_avg accept_pct；state 0=idle 1=prefill 2=decode
      PerfSnap ps = perfsnap::snapshot();
      char pb[192];
      snprintf(pb, sizeof pb, "P %llu %u %llu %.1f %llu %llu %.1f %.1f %.1f\n",
               (unsigned long long)ps.ts_ms, (unsigned)ps.state,
               (unsigned long long)ps.req_id, ps.pp_toks,
               (unsigned long long)ps.prompt_done,
               (unsigned long long)ps.prompt_total, ps.tg_inst, ps.tg_avg,
               ps.accept_pct);
      alive = c.send(pb);
    } else if (verb == "DIAG") {
      // 在线诊断快照（API 诊断包用）：多行文本，以单独一行 "." 结束
      alive = c.send(crashdiag::live_text() + "\n.\n");
    } else if (verb == "VIZ") {
      // QWENOX_VIZ runtime taps (19_viz.inc), one JSON line:
      //   VIZ                    -> V {"on":..,"slots":[..]}
      //   VIZ <slot> <from> <n>  -> V {"items":[per position ..]}
      int vs = -1, vf = 0, vn = 64;
      std::string o;
      if (ss >> vs) {
        ss >> vf >> vn;
        o = viz::range_json(vs, vf, vn);
      } else {
        o = viz::summary_json();
      }
      alive = c.send("V " + o + "\n");
    } else if (verb == "X") {
      int64_t req = 0;
      ss >> req;
      if (c.in_flight && req == c.req) {
        c.cancel = true;  // the worker ends it with its own D cancel line
      } else {
        char db[96];
        snprintf(db, sizeof db, "D %lld cancel 0 0 0.0 0.0\n", (long long)req);
        alive = c.send(db);
      }
    } else if (verb == "GEN") {
      GenReq g = parse_gen(ss);
      const bool rejected =
          g.reject || (int)g.ids.size() + g.max_tokens > m.maxctx;
      if (rejected) {
        const char* why = !g.reject ? "prompt + max_tokens exceeds maxctx"
                          : g.reject_why ? g.reject_why
                                         : "malformed or unsupported GEN line";
        fprintf(stderr, "%s serve: req %lld rejected before dispatch: %s "
                "(ids %zu, max_tokens %d, maxctx %d, vimg %d)\n",
                log_ts(), (long long)g.req, why, g.ids.size(), g.max_tokens,
                m.maxctx, g.vimg);
      }
      // VIMG binary frames follow the GEN line immediately and must be
      // drained before ANY reply path — rejects included — or their bytes
      // desync the line protocol: leftover frame bytes are then misread as
      // protocol lines and silently swallow every later GEN on this
      // connection. parse_gen keeps the parsed VIMG count even for lines it
      // rejects, so the drain still knows how many frames to consume; a
      // rejected request drains into a scratch buffer (its pixels are never
      // used). 120 s ceiling on the bulk transfer (~50 MB per 1080p image
      // over loopback is sub-second; the ceiling only guards against a
      // stalled client wedging this connection).
      std::vector<std::vector<float>> vpatches;
      if (g.vimg > 0) {
        bool ok = true;
        sock_rcvtimeo(cfd, 120);
        if (!rejected) vpatches.resize(g.vimg);
        for (int i = 0; i < g.vimg && ok; i++) {
          struct {
            uint32_t magic, P;
            uint64_t nbytes;
          } fh;
          ok = lb.read_full(&fh, sizeof fh) && fh.magic == 0x56494D31u &&
               fh.P > 0 && fh.P <= 65536 &&
               fh.nbytes == (uint64_t)fh.P * 1536 * 4;
          if (!ok) break;
          if (rejected) {
            char scratch[65536];
            uint64_t left = fh.nbytes;
            while (left > 0 && ok) {
              const size_t chunk =
                  (size_t)std::min<uint64_t>(left, sizeof scratch);
              ok = lb.read_full(scratch, chunk);
              left -= chunk;
            }
          } else {
            vpatches[i].resize((size_t)fh.P * 1536);
            ok = lb.read_full(vpatches[i].data(), fh.nbytes);
          }
        }
        sock_rcvtimeo(cfd, 0);
        if (!ok) {
          fprintf(stderr, "vision: frame read failed for req %lld "
                  "(timeout/short/garbled); dropping the connection\n",
                  (long long)g.req);
          char db[128];
          snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n",
                   (long long)g.req, g.ids.size());
          c.send(db);
          break;  // protocol desync: cannot trust anything after this
        }
      }
      if (rejected) {
        char db[128];
        snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
                 g.ids.size());
        alive = c.send(db);
        if (g.reject && g.vimg <= 0) {
          // The line failed before the VIMG token was parsed, so the frame
          // count is unknown and any frames the client did send cannot be
          // drained. Closing the connection is the only safe resync.
          fprintf(stderr, "%s serve: conn %llu: closing after an "
                  "undrainable rejected GEN\n", log_ts(),
                  (unsigned long long)c.id);
          break;
        }
        continue;
      }
      if (c.in_flight) {  // one GEN at a time per connection
        fprintf(stderr, "serve: req %lld rejected: another GEN is in flight "
                "on this connection\n", (long long)g.req);
        char db[128];
        snprintf(db, sizeof db, "D %lld error %zu 0 0 0\n", (long long)g.req,
                 g.ids.size());
        alive = c.send(db);
        continue;
      }
      if (worker.joinable()) worker.join();  // previous request's post-D tail
      c.cancel = false;
      c.req = g.req;
      c.in_flight = true;
      worker = std::thread(gen_one, std::ref(m), std::ref(c), std::move(g),
                           std::move(vpatches), std::cref(kvs));
    } else {
      // Not a protocol verb: foreign garbage, or binary frame bytes misread
      // as lines after a stream desync. Legit clients never send any, so a
      // flood means this byte stream can no longer be trusted: close the
      // connection and let the client reconnect resynced, instead of
      // silently swallowing everything that follows (a connection wedged
      // that way looks idle but drops every GEN).
      if (++bad_lines == 4 || bad_lines == 64)
        fprintf(stderr, "%s serve: conn %llu: %d unrecognized lines "
                "(desynced stream?)%s\n", log_ts(), (unsigned long long)c.id,
                bad_lines, bad_lines >= 64 ? "; closing the connection" : "");
      if (bad_lines >= 64) break;
    }
  }
  c.cancel = true;  // disconnect cancels the request in flight
  if (worker.joinable()) worker.join();
  {  // its slots become the first LRU victims (live conversations keep theirs)
    std::lock_guard<std::mutex> lk(g_sm);
    for (Slot& s : g_slots)
      if (s.owner == c.id) s.owner = 0;
  }
  sock_close(cfd);
  if (announced) {
    fprintf(stderr, "%s serve: conn %llu: client disconnected\n", log_ts(),
            (unsigned long long)c.id);
    crashdiag::note("conn %llu disconnected", (unsigned long long)c.id);
  }
  else if (!any_line)
    fprintf(stderr, "%s serve: conn %llu: client connected and closed without "
            "sending anything\n", log_ts(), (unsigned long long)c.id);
}

static int run(GpuModel& m, const std::string& host, int port) {
  if (!kv_admission_decode_cap(getenv("QWENOX_KV_RESERVE_DECODE"), g_reserve_decode)) {
    fprintf(stderr, "serve: QWENOX_KV_RESERVE_DECODE must be an integer from 0 to %d\n",
            INT_MAX);
    return 2;
  }
#ifndef _WIN32
  signal(SIGPIPE, SIG_IGN);
#endif
  sockaddr_in addr{};
  if (!engine_net::address(host, port, &addr)) {
    fprintf(stderr, "serve: invalid IPv4 listen address %s:%d\n", host.c_str(), port);
    return 1;
  }
  sockfd_t lfd = sock_tcp();
  if (lfd < 0) {
    fprintf(stderr, "serve: socket failed: %s\n", sock_strerror(sock_errno()));
    return 1;
  }
  int one = 1;
  setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof one);
  if (bind(lfd, (sockaddr*)&addr, sizeof addr) || listen(lfd, 16)) {
    fprintf(stderr, "serve: bind/listen failed on %s:%d: %s\n", host.c_str(), port,
            sock_strerror(sock_errno()));
    sock_close(lfd);
    return 1;
  }
  m.slots_init(eff_parallel());
  if (m.nslots > 1) memstats_log("slots");  // per-slot state allocated
  g_slots.assign((size_t)m.nslots, Slot{});
  if (m.nslots > 1) {
    m.kv_starve_fn = &starve;
    m.yield_fn = &yield_hook;
    m.borrow_query_fn = &borrow_query;
    m.borrow_fn = &borrow_turn;
    m.borrow_skip_fn = &borrow_skip;
    m.heavy_gate_fn = &heavy_gate;
    fprintf(stderr, "qwenox: prefill scheduling: lend %s, between prompts %s "
                    "(QWENOX_CONC_LEND_LIGHT_ONLY / QWENOX_PREFILL_SRPT)\n",
            lend_light_only() ? "light-only" : "any waiter",
            prefill_srpt() ? "shortest-remaining-first" : "alternate per chunk");
    // D2-S2a verify rendezvous: opt out with QWENOX_NO_VR=1 (serial
    // single-segment verify, the pre-S2 behavior). QWENOX_VR_WAIT_MS bounds
    // how long a submitter waits for stragglers before running the round.
    static const bool no_vr =
        getenv("QWENOX_NO_VR") && !GpuModel::env_zero("QWENOX_NO_VR");
    if (!no_vr) {
      m.vseg_pool_fn = &vseg_pool;
      m.vseg_active_fn = &vseg_active;
      // D2-S2b/c: chain loops join the pool per loop (ngram rounds excuse
      // themselves — they never submit and would stall the trigger).
      m.vseg_round_enter_fn = &vseg_round_enter;
      m.vseg_round_exit_fn = &vseg_round_exit;
    }
    // D2-S3c draft rendezvous: opt out with QWENOX_NO_DR=1 (serial per-slot
    // draft bursts, the pre-S3c behavior). QWENOX_DR_WAIT_MS bounds how long
    // a submitter waits for stragglers before running the round itself.
    static const bool no_dr =
        getenv("QWENOX_NO_DR") && !GpuModel::env_zero("QWENOX_NO_DR");
    if (!no_dr) {
      m.dr_pool_fn = &dr_pool;
      m.dr_round_enter_fn = &dr_round_enter;
      m.dr_round_exit_fn = &dr_round_exit;
    }
    if (const char* v = getenv("QWENOX_DR_WAIT_MS")) {
      g_dr_wait_ms = atoi(v);
      if (g_dr_wait_ms < 1) g_dr_wait_ms = 1;
      if (g_dr_wait_ms > 5000) g_dr_wait_ms = 5000;
    }
    if (const char* v = getenv("QWENOX_VR_WAIT_MS")) {
      g_vr_wait_ms = atoi(v);
      if (g_vr_wait_ms < 1) g_vr_wait_ms = 1;
      if (g_vr_wait_ms > 5000) g_vr_wait_ms = 5000;
    }
  }
  rckpt::prewarm(m.kv_nl > g_cfg.layers / 4);  // pinned checkpoint pool, background
  fprintf(stderr, "serve: listening on %s:%d (maxctx %d, %d slot%s)\n", host.c_str(), port,
          m.maxctx, m.nslots, m.nslots > 1 ? "s" : "");
  static kvsnap::Cfg kvs;  // read-only once serving; workers share it
  kvs.dir = "data/kvsnap";
  if (const char* v = getenv("QWENOX_KVSNAP_DIR")) kvs.dir = v;
  if (const char* v = getenv("QWENOX_KVSNAP"))
    if (!strcmp(v, "0")) kvs.dir.clear();
  if (const char* v = getenv("QWENOX_KVSNAP_MAX_GB"))
    kvs.cap_bytes = (uint64_t)(atof(v) * 1073741824.0);
  if (!kvs.cap_bytes) kvs.dir.clear();
  if (const char* v = getenv("QWENOX_KVSNAP_MIN")) kvs.min_tokens = (size_t)atol(v);
  if (const char* v = getenv("QWENOX_KVSNAP_QUEUE_MB"))
    kvs.queue_bytes = (uint64_t)std::max(64.0, atof(v)) << 20;
  if (const char* v = getenv("QWENOX_KVSNAP_DIRECT")) kvs.direct = strcmp(v, "0") != 0;
  if (const char* v = getenv("QWENOX_KVSNAP_READERS")) kvs.readers = atoi(v);
  if (const char* v = getenv("QWENOX_KVSNAP_WRITERS")) kvs.writers = atoi(v);
  if (const char* v = getenv("QWENOX_KVSNAP_HOT_HITS")) kvs.hot_hits = atoi(v);
  if (const char* v = getenv("QWENOX_KVSNAP_HOT_FRAC")) kvs.hot_frac = atof(v);
  if (const char* v = getenv("QWENOX_KVSNAP_PAUSE")) kvs.pause_ckpt = strcmp(v, "0") != 0;
  if (!kvs.dir.empty()) {
    fprintf(stderr, "kvsnap: dir %s, cap %.1f GB, min %zu tokens, %s IO\n",
            kvs.dir.c_str(), kvs.cap_bytes / 1073741824.0, kvs.min_tokens,
            kvs.direct ? "direct" : "buffered");
    kvsnap::init(m, kvs);  // may disable it (dir unusable)
  }
  for (;;) {
    sockfd_t cfd = accept(lfd, nullptr, nullptr);
    if (cfd < 0) continue;
    setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof one);
    // "client connected" is logged by conn_main once it has seen a verb
    std::thread(conn_main, std::ref(m), cfd, std::cref(kvs)).detach();
  }
  return 0;
}

}  // namespace serve
