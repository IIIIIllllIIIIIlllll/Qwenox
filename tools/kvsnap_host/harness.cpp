// Host-only harness for namespace kvsnap (51_host_cfg.inc): the real cache
// code against a fake GpuModel whose "device" buffers are host memory and
// whose KV rows are a pure function of the token prefix (as in the model).
// Build: tools/kvsnap_host/run.sh extracts the namespace, strips <<<...>>> launches, compiles.
//   harness save  DIR   -- fresh cache: save/restore/evict/re-create races
//   harness index DIR   -- restart on DIR: index == disk, restores, bad page
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "platform_gen.inc"  // POSIX os_* file shims of 00_platform.inc
static const char* log_ts() { return "[t]"; }

// ---- stubs for what kvsnap uses from the engine -------------------------------
static const int KV_PAGE = 256, KV_PAGE_BLOCKS = 64;
struct Rope {
  double factor = 0, beta_fast = 0, beta_slow = 0, attention_factor = 0;
  int original_max_position_embeddings = 0;
};
struct Cfg0 {
  int layers = 4, d = 16, branches = 1, rotary_dim = 64;
  double rope_theta = 1e6;
  Rope rope;
} g_cfg;
static int g_str = 0;
static bool g_gguf_dense = false;
namespace gguf_map {
static bool g_native_dense = false;
}
enum hipError_t { hipSuccess = 0 };
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost };
static hipError_t hipMemcpy(void* d, const void* s, size_t n, hipMemcpyKind) {
  memcpy(d, s, n);
  return hipSuccess;
}
static hipError_t hipStreamSynchronize(int) { return hipSuccess; }
static hipError_t hipGetLastError() { return hipSuccess; }
#define CK(x) (void)(x)
namespace crashdiag {  // 飞行记录：桩里不记
static inline void note(const char*, ...) {}
}  // namespace crashdiag
template <class... A>
static void k_bf16_rows_to_bt(A...) {}
template <class... A>
static void k_ik_twin_build(A...) {}
template <class T>
static bool has_image_tok(const T* t, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (t[i] == 248056 || t[i] == 248057) return true;
  return false;
}

static const int MAXCTX = 4096;
struct GpuModel {
  // geometry: nqsa = 1 QSA layer + 1 MTP layer, BF16 KV, indexer keys on
  bool qsa_kv_fp32 = false, qsa_wmma_btv = false, kv_paged = false, ple_on = true;
  bool mrope_active_ = false, graphs_ok = false;
  bool spec_snap = false, ngram_verify_active = false, mtp_tap_capture = false;
  int rope_delta_ = 0, pf_end = 0, cut_abs = -1, cut_r_ = 0;
  std::vector<std::array<int, 3>> mrope_pos_;
  std::vector<std::array<int, 2>> img_inj;
  struct {
    int mapped = 0;
    std::vector<int> tab;
  } kv_seq;
  struct {
    int npages = 0;
    std::vector<uint16_t> ref;
    int nfree() const { return npages; }
  } kv_pool;
  uint64_t img_hash_ = 0;
  std::vector<std::array<int, 3>> mrope_grids_;
  int kv_nl = 2, kv_layer_rows = MAXCTX, index_blocks = MAXCTX / 4,
      ik_layer_blocks = MAXCTX / 4, maxctx = MAXCTX;
  int pos = 0, mtp_pos = 0, ple_ringpos = 0;
  std::vector<int> chist;
  std::vector<uint16_t> kc, vc;
  std::vector<float> ik, S, conv, ring, iraw, R, miraw;
  uint16_t *d_kcb, *d_vcb;
  float *d_kc = nullptr, *d_vc = nullptr, *d_ik, *d_ikh = nullptr;
  float *d_S, *d_convst, *d_ring, *d_iraw, *d_R, *d_miraw;
  int pos_dev = 0, mpos_dev = 0, ringpos_dev = 0;
  int *d_pos = &pos_dev, *d_mpos = &mpos_dev, *d_ringpos = &ringpos_dev, *d_ptab = nullptr;
  GpuModel() {
    kc.resize((size_t)kv_nl * MAXCTX * 512);
    vc.resize(kc.size());
    ik.resize((size_t)kv_nl * ik_layer_blocks * 128);
    S.resize((size_t)3 * 48 * 128 * 128);  // ngdn = 3
    conv.resize((size_t)3 * 3 * 10240);
    ring.resize((size_t)9 * 10240);
    iraw.resize((size_t)1 * 512);
    R.resize((size_t)g_cfg.branches * g_cfg.d);
    miraw.resize(4 * 128);
    d_kcb = kc.data(), d_vcb = vc.data(), d_ik = ik.data(), d_S = S.data();
    d_convst = conv.data(), d_ring = ring.data(), d_iraw = iraw.data();
    d_R = R.data(), d_miraw = miraw.data();
  }
  void kv_runs(int r0, int r1, int, const std::function<void(int, size_t, int)>& f) {
    if (r1 > r0) f(r0, (size_t)r0, r1 - r0);  // identity layout
  }
  void kv_trim(int) {}
  void kv_reserve(int) {}
  uint16_t* vct_layer(int) { return nullptr; }
  static bool env_zero(const char* k) {
    const char* v = getenv(k);
    return v && v[0] == '0' && !v[1];
  }
};

// rckpt (49_rckpt.inc) as kvsnap::capture_ram sees it: host copies of one
// RAM checkpoint. The harness keeps none (find -> nullptr) unless a test adds.
namespace rckpt {
struct Buf {
  std::vector<float> v;
  const float* data() const { return v.data(); }
  size_t size() const { return v.size(); }
};
struct Ck {
  std::vector<int> toks;
  int pos = 0, ringpos = 0;
  Buf s, convst, ring, iraw;
  bool mtp_ok = false;
  int mtp_pos = 0;
  Buf mR, miraw;
};
static std::vector<Ck> g_list;
static const Ck* find(const std::vector<int>& hist, size_t n) {
  for (const Ck& ck : g_list)
    if (ck.toks.size() == n && n <= hist.size() &&
        std::equal(ck.toks.begin(), ck.toks.end(), hist.begin()))
      return &ck;
  return nullptr;
}
}  // namespace rckpt

#include "kvsnap_gen.inc"  // the real namespace kvsnap (extracted by run.sh)

// ---- deterministic "model": KV row r depends on tokens [0, r] ------------------
static uint64_t sm(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static std::vector<uint64_t> prefix_h(const std::vector<int>& t) {
  std::vector<uint64_t> h(t.size());
  uint64_t a = 1;
  for (size_t i = 0; i < t.size(); i++) h[i] = a = sm(a ^ (uint32_t)t[i]);
  return h;
}
template <class T>
static void fill(T* p, size_t n, uint64_t seed) {
  for (size_t i = 0; i < n; i++) p[i] = (T)(sm(seed + i) & 0xffff);
}
// Live state after "prefilling" t: rows [0, n) of every layer, indexer
// blocks [0, n/4+2), recurrent state from the last prefix hash.
static void prefill(GpuModel& m, const std::vector<int>& t) {
  std::fill(m.kc.begin(), m.kc.end(), 0);
  std::fill(m.vc.begin(), m.vc.end(), 0);
  std::fill(m.ik.begin(), m.ik.end(), 0.f);
  const auto h = prefix_h(t);
  for (int l = 0; l < m.kv_nl; l++)
    for (size_t r = 0; r < t.size(); r++) {
      fill(m.d_kcb + ((size_t)l * MAXCTX + r) * 512, 512, h[r] * 3 + l);
      fill(m.d_vcb + ((size_t)l * MAXCTX + r) * 512, 512, h[r] * 5 + l);
    }
  // indexer block b pools rows [4b, 4b+4): a function of the prefix to row 4b+3
  for (int l = 0; l < m.kv_nl; l++)
    for (size_t b = 0; 4 * b + 3 < t.size(); b++)
      fill(m.d_ik + ((size_t)l * m.ik_layer_blocks + b) * 128, 128, h[4 * b + 3] * 7 + l);
  const uint64_t last = h.back();
  fill(m.d_S, m.S.size(), last), fill(m.d_convst, m.conv.size(), last + 1);
  fill(m.d_ring, m.ring.size(), last + 2), fill(m.d_iraw, m.iraw.size(), last + 3);
  fill(m.d_R, m.R.size(), last + 4), fill(m.d_miraw, m.miraw.size(), last + 5);
  m.pos = (int)t.size();
  m.mtp_pos = m.pos - 1;
  m.ple_ringpos = m.pos % 9;
}
// Compare the restored live state with what prefill(t) produces.
static bool same_as(GpuModel& m, const std::vector<int>& t) {
  GpuModel g;
  prefill(g, t);
  const size_t n = t.size();
  bool ok = m.pos == (int)n && m.ple_ringpos == g.ple_ringpos;
  for (int l = 0; l < m.kv_nl && ok; l++) {
    const size_t o = (size_t)l * MAXCTX * 512;
    ok = !memcmp(&m.kc[o], &g.kc[o], n * 1024) && !memcmp(&m.vc[o], &g.vc[o], n * 1024) &&
         !memcmp(&m.ik[(size_t)l * m.ik_layer_blocks * 128],
                 &g.ik[(size_t)l * m.ik_layer_blocks * 128], (n / 4) * 512);
  }
  return ok && m.S == g.S && m.conv == g.conv && m.ring == g.ring && m.iraw == g.iraw;
}
static void save(GpuModel& m, const std::vector<int>& t) {
  prefill(m, t);
  std::vector<kvsnap::Pending> pend;
  kvsnap::capture(m, t, true, pend);
  kvsnap::commit(m, t, pend);
}
// Restore the longest cached prefix of t + 1 token; returns tokens restored.
static int restore(GpuModel& m, const std::vector<int>& t, bool* mtp = nullptr) {
  std::vector<int> q = t;
  q.push_back(7);
  auto e = kvsnap::match(q, kvsnap::g_fp);
  if (!e) return 0;
  std::fill(m.kc.begin(), m.kc.end(), 0);  // wipe the live state first
  std::fill(m.vc.begin(), m.vc.end(), 0);
  std::fill(m.ik.begin(), m.ik.end(), 0.f);
  std::fill(m.S.begin(), m.S.end(), 0.f);
  std::vector<int> hist;
  bool mt = false;
  if (!kvsnap::restore(m, e, q, hist, &mt)) return -1;
  if (mtp) *mtp = mt;
  return (int)hist.size();
}

// Slow deletes (3 ms, like a share): widens the window in which a queued or
// running delete meets a re-save of the same content-addressed name.
extern "C" int remove(const char* p) {
  usleep(3000);
  return unlink(p);
}
static bool wait_pub(const std::string& name) {
  for (int i = 0; i < 2000; i++) {
    {
      std::lock_guard<std::mutex> lk(kvsnap::g_mtx);
      for (auto& e : kvsnap::g_ck)
        if (e->name == name && e->pub) return true;
    }
    usleep(5000);
  }
  return false;
}
static void set_cap(uint64_t c) {
  std::lock_guard<std::mutex> lk(kvsnap::g_mtx);
  kvsnap::g_cap = c;
}
static std::string ckname(const std::vector<int>& t) {
  return kvsnap::ck_name((int)t.size(), kvsnap::ck_id(kvsnap::g_fp, t.data(), t.size()));
}
static int fails = 0;
static void check(bool ok, const char* what, const std::string& info = "") {
  printf("  %s  %s%s%s\n", ok ? "PASS" : "FAIL", what, info.empty() ? "" : "  ", info.c_str());
  fflush(stdout);
  if (!ok) fails++;
}
// KVSYNC counts == files on disk, and nothing else of ours is there.
static bool disk_ok(const std::string& dir, std::string* info) {
  const std::string s = kvsnap::sync_line();
  size_t ck = 0, pg = 0, mt = 0, other = 0;
  unsigned long long by = 0;
  for (auto& de : std::filesystem::directory_iterator(dir)) {
    const std::string fn = de.path().filename().string();
    if (fn == kvsnap::HITS_FILE || kvsnap::req_dir_name(fn)) continue;
    if (!de.is_regular_file()) { other++; continue; }
    by += de.file_size();
    if (fn[0] == 'k') ck++;
    else if (fn[0] == 'c') pg++;
    else if (fn[0] == 'm') mt++;
    else other++;
  }
  char b[160];
  snprintf(b, sizeof b, "K %zu %zu %zu %llu\n", ck, pg, mt, by);
  *info = "KVSYNC " + s.substr(0, s.size() - 1) + " / disk " + std::string(b, strlen(b) - 1);
  return s == b && !other;
}
static std::vector<int> toks(int n, int seed, const std::vector<int>& prefix = {}) {
  std::vector<int> t(prefix.begin(), prefix.end());
  while ((int)t.size() < n) t.push_back((int)(sm(seed * 1000003ULL + t.size()) % 200000));
  return t;
}
static bool exists(const std::string& dir, const std::string& name) {
  return std::filesystem::exists(dir + "/" + name);
}
// Index entry of t's checkpoint (copied out under the lock); hits -1: none.
struct CkInfo {
  int hits = -1;
  bool hot = false;
};
static CkInfo ckinfo(const std::vector<int>& t) {
  const std::string n = ckname(t);
  std::lock_guard<std::mutex> lk(kvsnap::g_mtx);
  for (auto& e : kvsnap::g_ck)
    if (e->name == n && e->pub) return {e->hits, e->hot};
  return {};
}
static std::string read_file_str(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
// Two-segment eviction (K2): a checkpoint restored twice survives a cap
// squeeze although it is the LRU one; the hot segment over its share is
// demoted; restore counts land in kvsnap_hits.txt.
static void test_two_segment(GpuModel& m, const std::string& dir) {
  std::string info;
  const auto E = toks(1200, 20), F = toks(1250, 21), G = toks(1300, 22);
  save(m, E), save(m, F), save(m, G);
  kvsnap::sync_line();
  restore(m, E), restore(m, E);
  const CkInfo ie = ckinfo(E);
  check(ie.hot && ie.hits == 2, "2Q：恢复两次进保护段",
        "hits " + std::to_string(ie.hits) + " hot " + std::to_string(ie.hot));
  // every other checkpoint restored once after E (F first: the oldest
  // probation one; it is larger than the next save, so it alone is evicted
  // and A..D stay for the restart stage): E is now the LRU one
  const auto A = toks(1300, 1), B = toks(1700, 2, std::vector<int>(A.begin(), A.begin() + 600)),
             C = toks(1500, 3), D = toks(1400, 9);
  for (const auto* t : {&F, &G, &A, &B, &C, &D}) restore(m, *t);
  kvsnap::sync_line();
  uint64_t total;
  {
    std::lock_guard<std::mutex> lk(kvsnap::g_mtx);
    total = kvsnap::g_total;
  }
  set_cap(total - 1);  // the next save must evict at least one checkpoint
  save(m, toks(1100, 23));
  kvsnap::sync_line();
  check(exists(dir, ckname(E)) && !exists(dir, ckname(F)) && exists(dir, ckname(G)) &&
            exists(dir, ckname(A)) && disk_ok(dir, &info),
        "2Q：上限挤压时只淘汰试用段最旧的 F，保护段的 E（最旧）保留", info);
  const std::string hits = read_file_str(dir + "/" + kvsnap::HITS_FILE);
  check(hits.find(ckname(E) + " 2\n") != std::string::npos &&
            hits.find(ckname(B) + " 1\n") != std::string::npos,
        "2Q：命中数写入 kvsnap_hits.txt");
  {
    std::lock_guard<std::mutex> lk(kvsnap::g_mtx);
    kvsnap::g_hot_frac = 1e-12;  // any hot checkpoint is over its share
  }
  set_cap(1ull << 40);
  save(m, toks(1150, 24));  // a publish runs evict_locked -> demote_locked
  kvsnap::sync_line();
  {
    std::lock_guard<std::mutex> lk(kvsnap::g_mtx);
    kvsnap::g_hot_frac = 0.5;
  }
  const CkInfo id = ckinfo(E);
  check(!id.hot && id.hits == 0, "2Q：保护段超额时降回试用段、命中数清零",
        "hits " + std::to_string(id.hits) + " hot " + std::to_string(id.hot));
}

int main(int argc, char** argv) {
  // serve-only entry points (tier choice, pause checkpoints): not exercised here
  (void)&kvsnap::capture_ram;
  (void)&kvsnap::note_prefill;
  (void)&kvsnap::release;
  (void)&kvsnap::prefer_over_ram;
  if (argc < 3) return 2;
  const std::string mode = argv[1], dir = argv[2];
  kvsnap::g_model_files = {"/dev/null"};
  GpuModel m;
  kvsnap::Cfg cfg;
  cfg.dir = dir;
  cfg.min_tokens = 1;
  cfg.cap_bytes = 1ull << 40;
  cfg.writers = 4;  // network-style parallel writers: the racy configuration
  const auto A = toks(1300, 1), B = toks(1700, 2, std::vector<int>(A.begin(), A.begin() + 600)),
             C = toks(1500, 3);
  std::string info;
  if (mode == "save") {
    kvsnap::init(m, cfg);
    bool no_private_dir = true;
    for (const auto& de : std::filesystem::directory_iterator(dir))
      if (kvsnap::req_dir_name(de.path().filename().string())) no_private_dir = false;
    check(no_private_dir, "初始化不创建私有目录");
    save(m, A), save(m, B);
    check(disk_ok(dir, &info), "保存 A、B 后索引 == 磁盘", info);
    bool mt = false;
    int r = restore(m, A, &mt);
    check(r == (int)A.size() && same_as(m, A) && mt, "恢复 A 逐字节一致（含 MTP）",
          "restored " + std::to_string(r));
    // A cap of 1 byte evicts everything but the newest checkpoint as soon as
    // it is published; A and B are then saved again at once, while the
    // janitor is still deleting their files (same content-addressed names).
    const std::string ckA = ckname(A);
    bool race_ok = true;
    for (int it = 0; it < 12; it++) {
      kvsnap::sync_line();
      const auto Ci = toks(1500, 100 + it);
      set_cap(1);
      save(m, Ci);
      const bool pub = wait_pub(ckname(Ci));
      size_t queued;
      {
        std::lock_guard<std::mutex> lk(kvsnap::g_mtx);
        queued = kvsnap::g_jq.size();
      }
      set_cap(1ull << 40);
      save(m, A), save(m, B);
      const bool d = disk_ok(dir, &info);
      const int ra = restore(m, A);
      const bool okA = ra == (int)A.size() && same_as(m, A);
      const int rb = restore(m, B);
      const bool okB = rb == (int)B.size() && same_as(m, B);
      if (it == 0 || !(pub && d && okA && okB))
        printf("    round %d: published %d, deletes queued at re-save %zu, disk %d, A %d, B %d  %s\n",
               it, pub, queued, d, okA, okB, info.c_str());
      race_ok = race_ok && pub && d && okA && okB;
    }
    check(race_ok, "淘汰后立刻重存同名文件 ×12：索引 == 磁盘，A/B 恢复逐字节一致", info);
    const auto D = toks(1400, 9);
    set_cap(1);
    save(m, D);
    check(disk_ok(dir, &info) && !exists(dir, ckA) && info.find("KVSYNC K 1 ") == 0,
          "上限压到 1 字节：只剩最新的 D，其余文件已删", info);
    set_cap(1ull << 40);
    save(m, A), save(m, B), save(m, C);
    check(disk_ok(dir, &info) && exists(dir, ckA), "再存 A、B、C：索引 == 磁盘", info);
    test_two_segment(m, dir);
    // B: one restore above, kept across the restart (kvsnap_hits.txt)
    FILE* f = fopen((dir + ".index").c_str(), "w");
    fputs(kvsnap::sync_line().c_str(), f);
    fclose(f);
  } else {
    char want[160] = {0};
    FILE* f = fopen((dir + ".index").c_str(), "r");
    if (!f || !fgets(want, sizeof want, f)) return 2;
    fclose(f);
    // a stray temp file, a foreign file, and an orphan page from a crash
    { FILE* t = fopen((dir + "/tmp_c_0000000000000001.kvc").c_str(), "w"); fputs("x", t); fclose(t); }
    { FILE* t = fopen((dir + "/notes.txt").c_str(), "w"); fputs("keep", t); fclose(t); }
    for (auto& de : std::filesystem::directory_iterator(dir))
      if (de.path().filename().string().rfind("c_", 0) == 0) {
        std::filesystem::copy_file(de.path(), dir + "/" + kvsnap::ch_name(0x123));
        break;
      }
    const std::string old = dir + "/kvsnap_req_0000000000000001";
    const std::string empty_old = dir + "/kvsnap_req_0000000000000002";
    std::filesystem::create_directory(old);
    std::filesystem::create_directory(empty_old);
    { std::ofstream(old + "/s_0000000000000001.kck") << "old"; }
    { std::ofstream(old + "/tmp_s_0000000000000001_00000000.kvm") << "old"; }
    { std::ofstream(old + "/notes.txt") << "keep"; }
    { std::ofstream(empty_old + "/s_0000000000000001_00000000.kvc") << "old"; }
    kvsnap::init(m, cfg);
    check(!std::filesystem::exists(old + "/s_0000000000000001.kck") &&
              !std::filesystem::exists(old + "/tmp_s_0000000000000001_00000000.kvm") &&
              std::filesystem::exists(old + "/notes.txt") &&
              !std::filesystem::exists(empty_old + "/s_0000000000000001_00000000.kvc"),
          "旧私有目录仅清理已识别文件，保留其他文件");
    const CkInfo ib = ckinfo(B);
    check(ib.hits == 1 && !ib.hot, "重启后从 kvsnap_hits.txt 读回命中数（B 1 次）",
          "hits " + std::to_string(ib.hits));
    const std::string s = kvsnap::sync_line();
    check(s == want, "重启后索引 == 重启前", "now " + s.substr(0, s.size() - 1) + " / before " + std::string(want, strlen(want) - 1));
    check(!exists(dir, "tmp_c_0000000000000001.kvc") && exists(dir, "notes.txt") && !exists(dir, kvsnap::ch_name(0x123)),
          "启动清理：临时文件、孤儿页已删，别人的文件保留");
    std::filesystem::remove(dir + "/notes.txt");
    check(disk_ok(dir, &info), "重启后索引 == 磁盘", info);
    check(kvsnap::shared_tokens(toks(1800, 5, B), kvsnap::g_fp) == 1536, "按页共享前缀长度（B 的 6 个整页）");
    int ra = restore(m, A);
    const bool okA = ra == (int)A.size() && same_as(m, A);
    int rc = restore(m, C);
    const bool okC = rc == (int)C.size() && same_as(m, C);
    check(okA && okC, "重启后恢复 A、C 逐字节一致");
    // damage every page of C: header/checksum are checked at restore now
    const auto ch = kvsnap::chain(kvsnap::g_fp, C.data(), (int)(C.size() - 1) / KV_PAGE);
    for (size_t i = 1; i < ch.size(); i++) {
      const std::string p = dir + "/" + kvsnap::ch_name(ch[i]);
      int fd = open(p.c_str(), O_RDWR);
      char c;
      pread(fd, &c, 1, 5000);
      c ^= 0x5a;
      pwrite(fd, &c, 1, 5000);
      close(fd);
    }
    rc = restore(m, C);
    check(rc == -1, "C 的页被改坏：恢复失败（不致命）", "rc " + std::to_string(rc));
    check(disk_ok(dir, &info) && !kvsnap::match(toks(1501, 4, C), kvsnap::g_fp), "坏页的检查点被删除，索引 == 磁盘", info);
    ra = restore(m, A);
    check(ra == (int)A.size() && same_as(m, A), "A 不受影响，仍能恢复");
  }
  printf("%s: %s\n", mode.c_str(), fails ? "FAIL" : "PASS");
  fflush(stdout);
  fflush(stderr);
  _exit(fails ? 1 : 0);  // detached writer/janitor threads still wait on g_cv
}
