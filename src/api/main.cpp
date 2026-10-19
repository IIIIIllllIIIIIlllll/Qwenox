// main.cpp — qwenox-api: C++ OpenAI-compatible front-end.
//
// Independently implements the OpenAI-compatible front-end from public API
// schemas and project-owned black-box conformance fixtures. The tokenizer and
// chat template are verified by tools/tok_ab.py and tools/template_ab.py.
//
// KV reuse: prompts reuse the exact ids of earlier requests for their
// byte-identical prefix (TokenCache), so a history holding a generated reply
// stays a token prefix of what the engine holds. Chat/responses requests also
// send SNAPS hints (semantic message-boundary token cuts before the first
// image, see compute_snap_cuts); the engine keeps RAM checkpoints (rckpt) at
// those cuts, so edit-and-resend hits the cache instead of re-prefilling. Raw
// completions send no hints — the engine's `cont` strict-prefix reuse and the
// SSD kvsnap tier still apply.
#include <atomic>
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include "../gpu/os_win32.h"  // wsa_init()
#include <direct.h>           // _chdir
#else
#include <unistd.h>           // chdir, readlink
#endif

#include "../service_conf.h"
#include "../engine_net.h"
#include "../version.h"
#include "chat_template.h"
#include "engine_client.h"
#include "engine_sup.h"
#include "http.h"
#include "json_py.h"
#include "power.h"
#include "reqlog.h"
#include "reqstat.h"
#include "reqstat_read.h"
#include "snap_cuts.h"
#include "tokenizer.h"
#include "toolparse.h"
#include "vision.h"
#include "../rope.h"

#ifdef _WIN32
#include "tray_win.h"
#endif

namespace {

using json = nlohmann::ordered_json;

// ----------------------------------------------------------------- config --

struct Config {
    std::string tokenizer_dir = "models/tokenizer";
    std::string engine_addr = "127.0.0.1:8730";
    std::string listen = "0.0.0.0:8731";
    std::string model = "qwen3.8-flash-next";
    int context = 262144;
    RopeConfig rope;
    // Server-side sampling/thinking overrides (see apply_overrides). Relative
    // to the working directory, which every launcher sets to the repo root.
    std::string overrides_path = "data/api-overrides.json";
    std::string admin_key;  // env QWENOX_API_ADMIN_KEY; empty = POST open (LAN)
    // service.conf 路径：--service-conf > QWENOX_SERVICE_CONF > 自动探测
    //（CWD 的 service.conf，否则 exe 上一级的 service.conf——build/ 里的
    // qwenox-api 直接运行也能找到项目根）。
    std::string service_conf;
};

Config g_cfg;
svcconf::Conf g_svc_conf;     // service.conf 有效值（env > conf）
std::mutex g_svc_conf_mtx;    // /admin/config 写与引擎 start 读之间互斥
std::string g_conf_path;      // 实际使用的 conf 路径
std::string g_conf_root = ".";  // conf 所在目录（引擎 exe/logs 的基准）
qwenox::Tokenizer g_tok;
bool g_tok_ready = false;  // 词表缺失降级：控制台/配置/引擎管理可用，推理 503
std::mutex g_conn_mtx;  // serialize (re)connects
std::atomic<long long> g_req_seq{0};

// The engine runs INFO kv_slots sequences at once (QWENOX_PARALLEL, sharing
// one KV pool; 1 = batch-1) and takes one GEN per connection.  Up to g_slots
// generations are in flight here, each on its own pooled engine connection;
// the rest wait in this front-end queue until a slot is released.  Control
// queries (MEM/VIZ) use g_ctl and never wait for a generation slot.
int g_slots = 1;
int g_kv_pool = 0;  // shared KV pool tokens from INFO (0 = old engine, unknown)
std::vector<std::unique_ptr<qwenox::EngineClient>> g_pool;  // g_slots clients
std::vector<qwenox::EngineClient*> g_pool_free;             // under g_slot_mtx
qwenox::EngineClient g_ctl;
std::mutex g_slot_mtx;
std::condition_variable g_slot_cv;
int g_in_flight = 0;
int g_queued = 0;
std::chrono::steady_clock::time_point g_busy_since;  // idle -> busy edge

// generation_config.json of this checkpoint.
const std::vector<int> g_eos = {248046, 248044};

long long now_unix() {
    return (long long)std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string hex24() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char* kHex = "0123456789abcdef";
    uint64_t v = rng();
    std::string s;
    s.reserve(24);
    for (int i = 0; i < 24; ++i) {
        if (i == 12) v = rng();
        s += kHex[(v >> (((i % 12) * 4) + 4)) & 0xf];
    }
    return s;
}

std::string make_id(const char* prefix) { return std::string(prefix) + hex24(); }

const char* kEngineRejected =
    "the engine rejected this request. Check /health for what this build "
    "supports, and that the prompt fits the context.";
const char* kEngineRejectedPar =
    "the engine rejected this request. Check /health for what this build "
    "supports, and that the prompt fits the context; the context is shared by "
    "concurrent requests, so it may also be full right now (retry later).";
const char* kEngineAborted =
    "the engine aborted this request: concurrent requests filled the shared "
    "context (earlier requests keep it). Retry when the server is less busy.";

// ------------------------------------------------------------- slot guard --

class SlotGuard {
  public:
    explicit SlotGuard(const std::function<bool()>& on_wait = {}) {
        std::unique_lock<std::mutex> lk(g_slot_mtx);
        const bool queued = g_in_flight >= g_slots;
        if (queued) ++g_queued;
        while (g_in_flight >= g_slots) {
            if (!on_wait) {
                g_slot_cv.wait(lk, [] { return g_in_flight < g_slots; });
                break;
            }
            // Do not hold the slot mutex while writing to the client.  The
            // callback also lets a disconnected stream leave the queue.
            lk.unlock();
            const bool keep_waiting = on_wait();
            lk.lock();
            if (!keep_waiting) {
                --g_queued;
                g_slot_cv.notify_all();
                return;
            }
            if (g_in_flight >= g_slots) g_slot_cv.wait_for(lk, std::chrono::seconds(1));
        }
        if (queued) --g_queued;
        if (g_in_flight++ == 0) g_busy_since = std::chrono::steady_clock::now();
        eng_ = g_pool_free.back();  // g_in_flight < g_slots: one is free
        g_pool_free.pop_back();
        acquired_ = true;
    }
    ~SlotGuard() {
        if (!acquired_) return;
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        g_pool_free.push_back(eng_);
        --g_in_flight;
        g_slot_cv.notify_all();
    }
    bool acquired() const { return acquired_; }
    qwenox::EngineClient& engine() const { return *eng_; }
    SlotGuard(const SlotGuard&) = delete;
    SlotGuard& operator=(const SlotGuard&) = delete;

  private:
    bool acquired_ = false;
    qwenox::EngineClient* eng_ = nullptr;
};

bool engine_ready(qwenox::EngineClient& eng, std::string* err) {
    std::lock_guard<std::mutex> lk(g_conn_mtx);
    if (eng.connected()) return true;
    return eng.connect(g_cfg.engine_addr, err);
}

// ------------------------------------------------------- incremental text --

// ------------------------------------------------------------------ utf8 --

const char* kReplacement = "\xEF\xBF\xBD";  // U+FFFD

// Largest cut <= n that does not split a UTF-8 character. Used wherever a
// look-ahead window is cut by a byte count (the </think> splitter): cutting at
// a lead-byte boundary keeps every emitted frame independently valid UTF-8,
// which per-frame JSON parsers require.
size_t utf8_boundary_at_or_before(const std::string& s, size_t n) {
    if (n > s.size()) n = s.size();
    while (n > 0 && n < s.size() && ((unsigned char)s[n] & 0xC0) == 0x80) --n;
    return n;
}

// Consumes well-formed text from `pending`, repairing ill-formed bytes to
// U+FFFD the same way the tokenizer's lossy decode does. Without `final`, a
// trailing incomplete character stays buffered.
std::string take_utf8(std::string* pending, bool final) {
    const std::string& s = *pending;
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        if (c < 0x80) {
            out += (char)c;
            ++i;
            continue;
        }
        size_t need = 0;
        unsigned lo2 = 0x80, hi2 = 0xBF;
        if (c >= 0xC2 && c <= 0xDF) need = 2;
        else if (c >= 0xE0 && c <= 0xEF) {
            need = 3;
            if (c == 0xE0) lo2 = 0xA0;
            if (c == 0xED) hi2 = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 4;
            if (c == 0xF0) lo2 = 0x90;
            if (c == 0xF4) hi2 = 0x8F;
        }
        if (need == 0) {  // ill-formed lead byte
            out += kReplacement;
            ++i;
            continue;
        }
        if (i + need > s.size()) {
            if (!final) break;
            out += kReplacement;  // truncated tail at end of stream
            i = s.size();
            break;
        }
        size_t valid = 1;  // maximal subpart: the lead byte always counts
        for (size_t k = 1; k < need; ++k) {
            const unsigned char cc = (unsigned char)s[i + k];
            if (cc < (k == 1 ? lo2 : 0x80) || cc > (k == 1 ? hi2 : 0xBF)) break;
            ++valid;
        }
        if (valid == need) {
            out.append(s, i, need);
            i += need;
        } else {
            out += kReplacement;
            i += valid;
        }
    }
    pending->erase(0, i);
    return out;
}

// Streams token -> text, holding back a multi-byte character split across
// tokens until it is complete (decode() would otherwise emit U+FFFD first).
class Detokenizer {
  public:
    explicit Detokenizer(const qwenox::Tokenizer* tok) : tok_(tok) {}
    std::string push(int id) {
        pending_ += tok_->decode_bytes(std::vector<int>{id}, /*skip_special=*/false);
        return take_utf8(&pending_, /*final=*/false);
    }
    std::string flush() { return take_utf8(&pending_, /*final=*/true); }

  private:
    const qwenox::Tokenizer* tok_;
    std::string pending_;
};

// ------------------------------------------------------ transcript cache ---
// The engine reuses KV only for an exact token prefix of what it holds, but
// generated ids are not canonical BPE (the model may write "/meta" as
// 14+5317 where encode() gives 66862). Re-encoding a history that contains a
// long reply therefore diverges inside the reply and the next turn re-prefills
// everything. So every request's sent + generated ids are remembered, and a
// prompt whose text starts with the same bytes reuses those ids; only the rest
// is encoded. QWENOX_API_TOKCACHE=<tokens> sizes the cache (0 = off).

constexpr int kImagePadId = 248056, kVideoPadId = 248057;
constexpr const char* kVisionPadSpan = "<|vision_start|><|image_pad|><|vision_end|>";

bool is_pad_id(int id) { return id == kImagePadId || id == kVideoPadId; }

class TokenCache {
  public:
    void set_capacity(size_t tokens) { cap_tokens_ = tokens; }
    bool enabled() const { return cap_tokens_ > 0; }

    // Remember what the engine saw for one request (prompt + generated ids).
    void record(std::vector<int> ids) {
        if (!enabled() || ids.empty() || ids.size() > cap_tokens_) return;
        auto t = build_entry(std::move(ids));
        {
            std::lock_guard<std::mutex> lk(mtx_);
            insert(std::move(t));
        }
        autosave();
    }

    // Persistence (QWENOX_API_TOKCACHE_FILE, default data/tcache.bin, "" = off).
    // Layout, little-endian: magic "GDTC1\0\0\0" | u32 version(2) | u32 vocab
    // | u64 count | per entry u64 n_ids + n_ids*i32. Byte offsets and pad runs
    // are rebuilt from the ids at load, so the only compatibility guard is the
    // vocab size. Written tmp+rename; a torn file just means an empty cache.
    void set_file(std::string f) { file_ = std::move(f); }
    void set_save_interval(int s) { save_s_ = s; }

    void save_file() const {
        if (file_.empty()) return;
        std::vector<std::vector<int>> snap;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            snap.reserve(entries_.size());
            for (const auto& e : entries_) snap.push_back(e->ids);
        }
        const std::string tmp = file_ + ".tmp";
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) return;
        const char magic[8] = "GDTC1";
        const uint32_t ver = 2, vocab = (uint32_t)g_tok.vocab_size();
        const uint64_t count = snap.size();
        bool ok = fwrite(magic, 1, 8, f) == 8 && fwrite(&ver, 4, 1, f) == 1 &&
                  fwrite(&vocab, 4, 1, f) == 1 && fwrite(&count, 8, 1, f) == 1;
        for (const auto& ids : snap) {
            const uint64_t n = ids.size();
            if (ok) ok = fwrite(&n, 8, 1, f) == 1;
            if (ok) ok = fwrite(ids.data(), 4, n, f) == n;
        }
        ok = fclose(f) == 0 && ok;
        if (!ok) {
            fprintf(stderr, "qwenox-api: tcache: save %s failed: %s\n", tmp.c_str(),
                    strerror(errno));
            remove(tmp.c_str());
            return;
        }
        remove(file_.c_str());  // Windows rename(2) fails when the target exists
        rename(tmp.c_str(), file_.c_str());
    }

    void load_file() {
        if (file_.empty() || !enabled()) return;
        FILE* f = fopen(file_.c_str(), "rb");
        if (!f) return;  // first boot: no cache yet
        char magic[8];
        uint32_t ver = 0, vocab = 0;
        uint64_t count = 0;
        bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, "GDTC1\0\0\0", 8) &&
                  fread(&ver, 4, 1, f) == 1 && ver == 2 &&
                  fread(&vocab, 4, 1, f) == 1 && (int)vocab == g_tok.vocab_size() &&
                  fread(&count, 8, 1, f) == 1 && count <= kMaxEntries * 4;
        size_t entries = 0, tokens = 0;
        std::vector<std::shared_ptr<Entry>> loaded;
        for (uint64_t i = 0; ok && i < count; i++) {
            uint64_t n = 0;
            std::vector<int> ids;
            ok = fread(&n, 8, 1, f) == 1 && n > 0 && n <= cap_tokens_;
            if (!ok) break;
            ids.resize(n);
            ok = fread(ids.data(), 4, n, f) == n;
            if (!ok) break;
            loaded.push_back(build_entry(std::move(ids)));
            tokens += n;
        }
        fclose(f);
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (auto& t : loaded) insert(std::move(t));
            entries = entries_.size();
        }
        if (!ok)
            fprintf(stderr, "qwenox-api: tcache: %s truncated/mismatched, loaded what parsed\n",
                    file_.c_str());
        if (entries)
            fprintf(stderr, "qwenox-api: tcache: loaded %zu entries (%zu tokens) from %s\n",
                    entries, tokens, file_.c_str());
    }

    struct Hit {
        std::vector<int> ids;        // reused ids
        std::vector<size_t> starts;  // their byte offsets in the text
        size_t bytes = 0;            // text bytes they cover
        size_t images = 0;           // image pad runs among them
    };

    // Longest reusable prefix of `text`. pad_counts = pad tokens per image of
    // this request, in order (an image run is only reused when it matches).
    Hit splice(const std::string& text, const std::vector<int>& pad_counts) {
        Hit best;
        if (!enabled()) return best;
        std::shared_ptr<Entry> pick;
        size_t pick_k = 0;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (const auto& t : entries_) {
                const size_t k = cut(*t, text, pad_counts);
                if (k == 0) continue;
                const size_t b = t->off[k];
                if (!pick || b > pick->off[pick_k] ||
                    (b == pick->off[pick_k] && t->used > pick->used)) {
                    pick = t;
                    pick_k = k;
                }
            }
            if (pick) pick->used = ++clock_;
        }
        if (!pick) return best;
        best.ids.assign(pick->ids.begin(), pick->ids.begin() + pick_k);
        best.starts.assign(pick->off.begin(), pick->off.begin() + pick_k);
        best.bytes = pick->off[pick_k];
        for (const auto& run : pick->runs)
            if (run.first < pick_k) best.images++;
        return best;
    }

  private:
    struct Entry {
        std::vector<int> ids;
        std::string bytes;           // what the ids decode to (a pad run = one pad)
        std::vector<uint32_t> off;   // off[i] = byte offset of ids[i]; off[n] = bytes.size()
        std::vector<std::pair<uint32_t, uint32_t>> runs;  // image pad runs: (first id, count)
        int vstart = -1;
        uint64_t used = 0;
    };

    // Reusable id count of `t` for `text` (0 = none).
    size_t cut(const Entry& t, const std::string& text, const std::vector<int>& pad_counts) const {
        const size_t n = t.ids.size();
        const size_t lim = std::min(t.bytes.size(), text.size());
        const size_t same = static_cast<size_t>(
            std::mismatch(t.bytes.begin(), t.bytes.begin() + lim, text.begin()).first -
            t.bytes.begin());
        // Last id boundary inside the equal bytes.
        size_t k = static_cast<size_t>(
            std::upper_bound(t.off.begin(), t.off.end(), static_cast<uint32_t>(same)) -
            t.off.begin()) - 1;
        // The whole transcript matched: the engine's `cont` resumes right at its
        // end, so cut there (unless the text continues inside a UTF-8 char).
        bool whole = k == n && !(same < text.size() && (uint8_t(text[same]) & 0xC0) == 0x80);
        for (;;) {
            // Otherwise the engine can only resume from a checkpoint below k
            // anyway, so cut right after an added token: the rest of the text
            // then encodes exactly as encode() would encode it in full. Never
            // inside or right after an image placeholder (its span must stay whole).
            if (!whole) {
                while (k > 0 && !(g_tok.is_added_token(t.ids[k - 1]) &&
                                  t.ids[k - 1] != t.vstart && !is_pad_id(t.ids[k - 1])))
                    k--;
            } else if (k > 0 && (t.ids[k - 1] == t.vstart || is_pad_id(t.ids[k - 1]))) {
                whole = false;
                k--;
                continue;
            }
            // Every image run inside the prefix must match this request's image.
            size_t bad = SIZE_MAX;
            for (size_t j = 0; j < t.runs.size() && t.runs[j].first < k; ++j) {
                if (j >= pad_counts.size() ||
                    static_cast<int>(t.runs[j].second) != pad_counts[j]) {
                    bad = t.runs[j].first;
                    break;
                }
            }
            if (bad == SIZE_MAX) return k;
            whole = false;
            k = bad;  // before the pad; the loop steps back past <|vision_start|>
        }
    }

    // Decode + offset bookkeeping of one recorded transcript.
    static std::shared_ptr<Entry> build_entry(std::vector<int> ids) {
        auto t = std::make_shared<Entry>();
        t->off.reserve(ids.size() + 1);
        const int vstart = g_tok.added_token_id("<|vision_start|>");
        for (size_t i = 0; i < ids.size(); ++i) {
            t->off.push_back(static_cast<uint32_t>(t->bytes.size()));
            // The prompt text holds one <|image_pad|> per image; the ids hold
            // pad_tokens() copies. Only the first copy of a run has bytes.
            const bool run_cont = i > 0 && is_pad_id(ids[i]) && is_pad_id(ids[i - 1]);
            if (is_pad_id(ids[i]) && !run_cont) {
                t->runs.push_back({static_cast<uint32_t>(i), 0});
            }
            if (is_pad_id(ids[i])) {
                t->runs.back().second++;
                if (run_cont) continue;
            }
            t->bytes += g_tok.decode_bytes(std::vector<int>{ids[i]}, /*skip_special=*/false);
        }
        t->off.push_back(static_cast<uint32_t>(t->bytes.size()));
        t->vstart = vstart;
        t->ids = std::move(ids);
        return t;
    }

    // Caller holds mtx_.
    void insert(std::shared_ptr<Entry> t) {
        t->used = ++clock_;
        // A multi-turn conversation re-sends its own history: the new
        // transcript supersedes every entry that is a prefix of it.
        for (auto it = entries_.begin(); it != entries_.end();) {
            const auto& e = (*it)->ids;
            if (e.size() <= t->ids.size() && std::equal(e.begin(), e.end(), t->ids.begin())) {
                total_ -= e.size();
                it = entries_.erase(it);
            } else {
                ++it;
            }
        }
        total_ += t->ids.size();
        entries_.push_back(std::move(t));
        while (entries_.size() > kMaxEntries || total_ > cap_tokens_) {
            auto lru = std::min_element(entries_.begin(), entries_.end(),
                                        [](const auto& a, const auto& b) { return a->used < b->used; });
            total_ -= (*lru)->ids.size();
            entries_.erase(lru);
        }
    }

    // Throttled persist after record(); 0 interval = save on every record.
    void autosave() {
        if (file_.empty()) return;
        const auto now = std::chrono::steady_clock::now();
        auto last = last_save_.load(std::memory_order_relaxed);
        if (save_s_ > 0 && now - last < std::chrono::seconds(save_s_)) return;
        if (!last_save_.compare_exchange_strong(last, now, std::memory_order_relaxed)) return;
        save_file();
    }

    static constexpr size_t kMaxEntries = 64;
    mutable std::mutex mtx_;  // mutable: save_file() is const
    std::vector<std::shared_ptr<Entry>> entries_;
    size_t total_ = 0;
    size_t cap_tokens_ = 4u << 20;
    uint64_t clock_ = 0;
    std::string file_;
    int save_s_ = 10;
    std::atomic<std::chrono::steady_clock::time_point> last_save_{
        std::chrono::steady_clock::time_point{}};
};

TokenCache g_tcache;

// Token ids the engine checkpoints its RAM state at mid-generation (CKPT
// hints on GEN). Mid-reply cuts happen right after these added tokens
// (<tool_call>, </think>): the TokenCache slices a next-turn prefix there, so
// a checkpoint at the same spot turns a template-roundtrip mismatch from a
// full reply re-prefill into re-decoding only the tool call + tool result.
// QWENOX_CKPT_TOKENS=0 disables; the engine ignores unknown/empty hints.
std::vector<int> g_ckpt_ids;

std::vector<int> default_ckpt_tokens() {
    if (const char* e = std::getenv("QWENOX_CKPT_TOKENS"))
        if (!strcmp(e, "0")) return {};
    std::vector<int> ids;
    for (const char* s : {"<tool_call>", "</think>"}) {
        const int id = g_tok.added_token_id(s);
        if (id >= 0) ids.push_back(id);
    }
    if (!ids.empty()) {
        std::string names;
        for (int id : ids) {
            if (!names.empty()) names += ",";
            names += std::to_string(id);
        }
        fprintf(stderr, "qwenox-api: ckpt tokens: %s\n", names.c_str());
    }
    return ids;
}

// Encode a rendered prompt, reusing the transcript cache for its prefix.
// frames == nullptr: raw text (no image expansion, like encode()). starts
// receives each id's byte offset in `text`.
bool encode_spliced(const std::string& text, const std::vector<vision::Frame>* frames,
                    std::vector<int>* ids, std::vector<size_t>* starts, std::string* error) {
    std::vector<int> pad_counts;
    if (frames)
        for (const auto& f : *frames) pad_counts.push_back(f.pad_tokens());
    TokenCache::Hit hit = g_tcache.splice(text, pad_counts);
    // The reused text must hold exactly the images the reused ids hold.
    if (hit.bytes > 0) {
        size_t spans = 0;
        const std::string head = text.substr(0, hit.bytes);
        for (size_t at = 0; (at = head.find(kVisionPadSpan, at)) != std::string::npos;
             at += std::strlen(kVisionPadSpan))
            spans++;
        if (spans != hit.images || (!frames && hit.images > 0)) hit = TokenCache::Hit{};
    }
    const std::string tail_text = text.substr(hit.bytes);
    std::vector<int> tail;
    std::vector<size_t> tail_starts;
    if (frames) {
        std::vector<vision::Frame> rest;
        for (size_t i = hit.images; i < frames->size(); ++i) {
            vision::Frame f;
            f.grid = (*frames)[i].grid;
            rest.push_back(f);
        }
        if (!vision::encode_prompt(tail_text, g_tok, rest, &tail, error, &tail_starts))
            return false;
    } else {
        // Text-only request: a user-typed <|image_pad|>/<|video_pad|> literal
        // must tokenize as ordinary text. As the added-token ids it would
        // make the engine read this as a vision request without images and
        // reject it ("image placeholder ... without MROPE/VIMG"); only the
        // vision path (encode_prompt) may produce those ids.
        static const std::unordered_set<int> kTextNoPads = {kImagePadId, kVideoPadId};
        for (const auto& s : g_tok.encode_with_offsets(tail_text, false, &kTextNoPads)) {
            tail.push_back(s.id);
            tail_starts.push_back(s.start);
        }
    }
    if (!hit.ids.empty())
        fprintf(stderr, "tcache: reused %zu ids (%zu of %zu bytes), encoded %zu\n",
                hit.ids.size(), hit.bytes, text.size(), tail.size());
    *ids = std::move(hit.ids);
    ids->insert(ids->end(), tail.begin(), tail.end());
    *starts = std::move(hit.starts);
    for (size_t s : tail_starts) starts->push_back(s + hit.bytes);
    return true;
}

// ------------------------------------------------------------ generation ---

struct GenSpec {
    std::vector<int> ids;
    int max_tokens = 0;  // 0 => context budget
    bool sample = false;
    float temp = 1.0f;
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    unsigned long long seed = 0;
    bool seeded = false;  // client-supplied seed (engine SEEDED flag)
    float presence = 0.0f;
    float frequency = 0.0f;
    std::vector<std::pair<int, float>> bias;
    bool logprobs = false;
    std::vector<std::string> stop;
    std::vector<std::array<int, 3>> mrope_grids;
    std::vector<std::vector<float>> patches;
    std::vector<long long> snaps;  // SNAPS hints: semantic boundary token cuts
    // Test hook (request field think_guard_test_cut): the first run stops
    // after this many tokens as if the model had ended its turn there, so the
    // think guard can be exercised on a real model. 0 = off.
    int think_guard_test_cut = 0;
};

struct GenOutcome {
    std::string text;
    std::string reason;  // OpenAI finish_reason
    long long n_prompt = 0;
    long long n_gen = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ttft_ms = 0.0;
    int n_cached = 0, rounds = 0, commit = 0, proposed = 0;
    int served_max_tokens = 0;
    int clamped_from = 0;
    std::vector<float> logprobs;
    bool hit_stop = false;
    std::string stop_hit;  // the stop sequence that ended the turn (hit_stop)
    bool client_gone = false;
    std::vector<int> tokens;  // generated ids (the think guard resumes from them)
    // Think guard (run_generation_guarded): "" when it did not act, else
    // "resumed_unclosed" / "resumed_empty_answer" / "resume_failed" /
    // "tool_call_from_reasoning".
    std::string guard;
    // tool_call_from_reasoning: the complete <tool_call> block(s) the model
    // wrote at the end of its reasoning; the caller feeds them to the tool
    // parser after the answer so they are made as real calls.
    std::string reasoning_calls;
};

// on_delta receives each incremental piece of text; returning false aborts the
// generation (stop sequence matched, or the HTTP client vanished). May be
// empty for non-streaming use. `held` runs the generation on a slot the
// caller already holds (the think guard keeps one slot across its runs);
// null acquires one here.
GenOutcome run_generation(GenSpec& spec,
                          const std::function<bool(const std::string&)>& on_delta,
                          const std::string& log_tag,
                          const std::function<bool()>& on_wait = {},
                          SlotGuard* held = nullptr) {
    GenOutcome out;
    std::function<bool()> wait_callback;
    if (on_wait) {
        wait_callback = [&]() {
            const bool keep_waiting = on_wait();
            if (!keep_waiting) out.client_gone = true;
            return keep_waiting;
        };
    }
    std::optional<SlotGuard> own_slot;
    SlotGuard* slot = held;
    if (slot == nullptr) {
        own_slot.emplace(wait_callback);
        slot = &*own_slot;
    }
    if (!slot->acquired()) {
        out.client_gone = true;
        return out;
    }
    qwenox::EngineClient& eng = slot->engine();
    std::string err;
    if (!engine_ready(eng, &err)) {
        fprintf(stderr, "REQ %s 502 engine connect failed: %s\n", log_tag.c_str(), err.c_str());
        http::fail(502, "engine not reachable: " + err, "server_error", "server_error");
    }

    const long long req_id = ++g_req_seq;

    qwenox::GenParams p;
    p.req = req_id;
    p.ids = spec.ids;
    const long long budget = static_cast<long long>(g_cfg.context) -
                             static_cast<long long>(spec.ids.size());
    if (budget <= 0) http::fail(400, kEngineRejected);
    const long long requested = spec.max_tokens > 0 ? spec.max_tokens : budget;
    p.max_tokens = static_cast<int>(std::min(requested, budget));
    p.eos = g_eos;
    p.sample = spec.sample;
    p.temp = spec.temp;
    p.top_k = spec.top_k;
    p.top_p = spec.top_p;
    p.min_p = spec.min_p;
    p.seed = spec.seed;
    p.seeded = spec.seeded;
    p.presence = spec.presence;
    p.frequency = spec.frequency;
    p.bias = spec.bias;
    p.logprobs = spec.logprobs;
    p.mrope_grids = spec.mrope_grids;
    p.patches = std::move(spec.patches);
    p.snaps = spec.snaps;
    p.ckpt = g_ckpt_ids;

    out.served_max_tokens = p.max_tokens;
    if (spec.max_tokens > p.max_tokens) out.clamped_from = spec.max_tokens;
    Detokenizer detok(&g_tok);
    std::string full;
    std::string pending_emit;
    size_t max_stop_len = 0;
    for (const std::string& stop : spec.stop) max_stop_len = std::max(max_stop_len, stop.size());
    double ttft = 0.0;
    bool first_piece = true;
    long long emitted = 0;
    auto t0 = std::chrono::steady_clock::now();
    fprintf(stderr, "REQ %lld start %s prompt=%zu max=%d %s\n", req_id, log_tag.c_str(),
            spec.ids.size(), p.max_tokens, spec.sample ? "sampled" : "greedy");

    auto deliver = [&](const std::string& piece) -> bool {
        if (piece.empty()) return true;
        if (first_piece) {
            ttft = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
            first_piece = false;
        }
        full += piece;
        if (on_delta && !on_delta(piece)) {
            out.client_gone = true;
            return false;
        }
        return true;
    };

    auto emit = [&](const std::string& piece) -> bool {
        pending_emit += piece;
        // Hold back enough suffix bytes that a stop sequence spanning two
        // decoded token pieces can still be removed before it reaches SSE.
        size_t cut = std::string::npos;
        const std::string* hit = nullptr;
        for (const std::string& s : spec.stop) {
            size_t at = pending_emit.find(s);
            if (at != std::string::npos && (cut == std::string::npos || at < cut)) {
                cut = at;
                hit = &s;
            }
        }
        if (cut != std::string::npos) {
            std::string delta = pending_emit.substr(0, cut);
            pending_emit.clear();
            out.hit_stop = true;
            out.stop_hit = *hit;
            if (!deliver(delta)) out.client_gone = true;
            return false;
        }
        size_t safe = pending_emit.size();
        if (max_stop_len > 1) {
            const size_t hold = max_stop_len - 1;
            safe = safe > hold ? safe - hold : 0;
            safe = utf8_boundary_at_or_before(pending_emit, safe);
        }
        if (safe == 0) return true;
        std::string delta = pending_emit.substr(0, safe);
        pending_emit.erase(0, safe);
        return deliver(delta);
    };

    qwenox::GenResult r = eng.generate(p, [&](int tok, float lp) {
        (void)lp;
        ++emitted;
        return emit(detok.push(tok));
    }, wait_callback);
    out.ttft_ms = ttft;
    if (r.transport_ok && r.reason != "error") {
        // Before the response ends, so a client's next turn always sees it.
        std::vector<int> seen = spec.ids;
        seen.insert(seen.end(), r.tokens.begin(), r.tokens.end());
        g_tcache.record(std::move(seen));
    }

    if (!r.transport_ok) {
        if (r.timed_out) http::fail(504, "engine timed out while generating", "server_error", "server_error");
        http::fail(502, "engine connection lost", "server_error", "server_error");
    }
    if (!out.hit_stop && !out.client_gone) {
        emit(detok.flush());
        if (!out.hit_stop && !out.client_gone && !pending_emit.empty()) {
            std::string tail = std::move(pending_emit);
            deliver(tail);
        }
    }

    out.text = full;
    out.n_prompt = r.n_prompt;
    // A cancelled request comes back as `D <req> cancel 0 0 0.0 0.0`, which
    // carries no counts, so fall back to what we actually emitted (a stop
    // sequence cancels the turn after the tokens that produced it).
    out.n_gen = r.n_gen > 0 ? r.n_gen : emitted;
    out.prefill_ms = r.prefill_ms;
    out.decode_ms = r.decode_ms;
    out.n_cached = r.n_cached;
    out.rounds = r.rounds;
    out.commit = r.commit;
    out.proposed = r.proposed;
    out.logprobs = std::move(r.logprobs);
    out.tokens = std::move(r.tokens);

    if (r.reason == "error") {
        // Concurrent requests share one KV pool; when it runs out the engine
        // fails the later request (the D line carries no reason).  After tokens
        // have streamed that is the only way a request can fail.
        if (emitted > 0) {
            fprintf(stderr, "REQ %lld aborted by the engine after %lld tokens (shared KV "
                            "context full)\n", req_id, emitted);
            http::fail(503, kEngineAborted, "server_error", "server_error");
        }
        http::fail(400, g_slots > 1 ? kEngineRejectedPar : kEngineRejected);
    }
    if (out.hit_stop || r.reason == "done") out.reason = "stop";
    else if (r.reason == "length") out.reason = "length";
    else out.reason = "stop";  // "cancel": client gone, or stopped by us

    fprintf(stderr,
            "REQ %lld end finish=%s prompt=%lld gen=%lld ttft=%.0fms prefill=%.0fms "
            "decode=%.0fms cached=%d%s\n",
            req_id, out.reason.c_str(), out.n_prompt, out.n_gen, out.ttft_ms,
            out.prefill_ms, out.decode_ms, out.n_cached,
            out.hit_stop ? " (stop)" : (out.client_gone ? " (client gone)" : ""));
    if (out.rounds > 0)
        fprintf(stderr, "REQ %lld mtp rounds=%d commit=%d proposed=%d acc=%.1f%%\n", req_id,
                out.rounds, out.commit, out.proposed,
                out.proposed ? 100.0 * (out.commit - out.rounds) / out.proposed : 0.0);
    {
        auto us = [](double ms) -> uint32_t {
            double v = ms * 1000.0;
            return v >= 4294967295.0 ? 4294967295u : (uint32_t)(v > 0.0 ? v : 0.0);
        };
        reqstat::Entry e{};
        e.ts_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        e.req_seq = (uint64_t)req_id;
        e.n_prompt = (uint32_t)std::min<long long>(out.n_prompt, UINT32_MAX);
        e.n_cached = (uint32_t)out.n_cached;
        e.n_gen = (uint32_t)std::min<long long>(out.n_gen, UINT32_MAX);
        e.proposed = (uint32_t)out.proposed;
        e.commit = (uint32_t)out.commit;
        e.rounds = (uint32_t)out.rounds;
        e.prefill_us = us(out.prefill_ms);
        e.decode_us = us(out.decode_ms);
        e.ttft_us = us(out.ttft_ms);
        const uint32_t finish = out.client_gone || r.reason == "cancel" ? 3u
                                : out.reason == "length"               ? 2u
                                                                       : 1u;
        e.flags = finish | ((uint32_t)(r.drafter & 0xF) << 4) |
                  (!p.patches.empty() ? 0x100u : 0u);
        reqstat::record(e);
    }
    // Hand the image patches back: the think guard may resume this prompt.
    spec.patches = std::move(p.patches);
    return out;
}

// -------------------------------------------------------------- request io --

json parse_body(const http::Request& q) {
    if (q.body.empty()) return json::object();
    json body;
    try {
        body = json::parse(q.body);
    } catch (...) {
        http::fail(400, "1: JSON decode error");
    }
    if (!body.is_object()) http::fail(400, "request body must be a JSON object");
    return body;
}

bool bool_field(const json& body, const char* key, bool fallback) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return fallback;
    if (!it->is_boolean()) http::fail(400, std::string(key) + " must be a boolean");
    return it->get<bool>();
}

double number_field(const json& body, const char* key, double fallback) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return fallback;
    if (!it->is_number()) http::fail(400, std::string(key) + " must be a number");
    const double value = it->get<double>();
    if (!std::isfinite(value)) http::fail(400, std::string(key) + " must be finite");
    return value;
}

long long integer_value(const json& value, const char* key) {
    if (value.is_number_unsigned()) {
        const unsigned long long v = value.get<unsigned long long>();
        if (v > static_cast<unsigned long long>(std::numeric_limits<long long>::max()))
            http::fail(400, std::string(key) + " is too large");
        return static_cast<long long>(v);
    }
    if (value.is_number_integer()) return value.get<long long>();
    http::fail(400, std::string(key) + " must be an integer");
}

long long integer_field(const json& body, const char* key, long long fallback) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return fallback;
    return integer_value(*it, key);
}

void require_optional_boolean(const json& body, const char* key) {
    auto it = body.find(key);
    if (it != body.end() && !it->is_null() && !it->is_boolean())
        http::fail(400, std::string(key) + " must be a boolean");
}

bool request_wants_logprobs(const json& body) {
    if (!body.contains("logprobs") || body["logprobs"].is_null()) return false;
    const json& lp = body["logprobs"];
    if (lp.is_boolean()) return lp.get<bool>();
    if (lp.is_number_integer() || lp.is_number_unsigned()) {
        const long long value = integer_value(lp, "logprobs");
        if (value < 0) http::fail(400, "logprobs must be non-negative");
        return value > 0;
    }
    http::fail(400, "logprobs must be a boolean or integer");
}

// Fills the shared sampling / budget fields of a GenSpec.
void apply_sampling(const json& body, GenSpec* spec) {
    // temperature omitted => the checkpoint's Qwen defaults (sampled);
    // explicit 0 => greedy.
    const double temp = number_field(body, "temperature", 1.0);
    if (temp < 0.0 || temp > 2.0)
        http::fail(400, "temperature=" + json_py::py_float(temp) +
                            " is outside 0.0..2.0. Values past the range are NOT "
                            "clamped, because a clamped request is indistinguishable, "
                            "from the response, from one that was served as asked.");
    spec->sample = temp > 0.0;
    spec->temp = (float)temp;
    const long long top_k = integer_field(body, "top_k", 20);
    if (top_k < 0 || top_k > std::numeric_limits<int>::max())
        http::fail(400, "top_k must be between 0 and 2147483647");
    spec->top_k = static_cast<int>(top_k);
    spec->top_p = static_cast<float>(number_field(body, "top_p", 0.95));
    spec->min_p = static_cast<float>(number_field(body, "min_p", 0.0));
    if (spec->top_p < 0.0f || spec->top_p > 1.0f)
        http::fail(400, "top_p must be between 0 and 1");
    if (spec->min_p < 0.0f || spec->min_p > 1.0f)
        http::fail(400, "min_p must be between 0 and 1");
    spec->presence = static_cast<float>(number_field(body, "presence_penalty", 0.0));
    spec->frequency = static_cast<float>(number_field(body, "frequency_penalty", 0.0));
    if (spec->presence < -2.0f || spec->presence > 2.0f)
        http::fail(400, "presence_penalty must be between -2 and 2");
    if (spec->frequency < -2.0f || spec->frequency > 2.0f)
        http::fail(400, "frequency_penalty must be between -2 and 2");

    spec->logprobs = request_wants_logprobs(body);
    if (spec->logprobs && !spec->sample)
        http::fail(400,
                   "logprobs applies to the sampler, and this request decodes greedy "
                   "(explicit temperature: 0). Send temperature > 0 (or omit it for "
                   "the server-default sampling), or drop logprobs: a request silently "
                   "served without it would be indistinguishable from one that "
                   "honoured it.");

    if (body.contains("seed") && !body["seed"].is_null()) {
        const json& s = body["seed"];
        if (s.is_number_unsigned()) spec->seed = s.get<unsigned long long>();
        else if (s.is_number_integer())
            spec->seed = static_cast<unsigned long long>(s.get<long long>());
        else if (s.is_string()) {
            const std::string& text = s.get_ref<const std::string&>();
            unsigned long long value = 0;
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size())
                http::fail(400, "seed must be an integer or decimal integer string");
            spec->seed = value;
        } else {
            http::fail(400, "seed must be an integer or decimal integer string");
        }
        spec->seeded = true;
    } else {
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        spec->seed = rng();  // the engine's RNG is counter-based: a seed is required
    }

    // Test hook for the think guard (see GenSpec::think_guard_test_cut).
    if (body.contains("think_guard_test_cut") && !body["think_guard_test_cut"].is_null()) {
        const long long cut = integer_field(body, "think_guard_test_cut", 0);
        if (cut < 0 || cut > std::numeric_limits<int>::max())
            http::fail(400, "think_guard_test_cut must be a non-negative integer");
        spec->think_guard_test_cut = static_cast<int>(cut);
    }

    if (body.contains("logit_bias") && body["logit_bias"].is_object()) {
        if (body["logit_bias"].size() > 20480)
            http::fail(400, "logit_bias has more than 20480 entries");
        for (auto it = body["logit_bias"].begin(); it != body["logit_bias"].end(); ++it) {
            int token = 0;
            const std::string& key = it.key();
            const auto parsed = std::from_chars(key.data(), key.data() + key.size(), token);
            if (parsed.ec != std::errc() || parsed.ptr != key.data() + key.size() || token < 0 ||
                token >= g_tok.vocab_size())
                http::fail(400, "logit_bias keys must be token ids");
            if (!it.value().is_number()) http::fail(400, "logit_bias values must be numbers");
            const double value = it.value().get<double>();
            if (!std::isfinite(value)) http::fail(400, "logit_bias values must be finite");
            if (value < -100.0 || value > 100.0)
                http::fail(400, "logit_bias values must be between -100 and 100");
            spec->bias.emplace_back(token, static_cast<float>(value));
        }
    } else if (body.contains("logit_bias") && !body["logit_bias"].is_null()) {
        http::fail(400, "logit_bias must be an object");
    }

    spec->max_tokens = 0;
    const char* selected_budget = nullptr;
    for (const char* k : {"max_tokens", "max_completion_tokens", "max_output_tokens"}) {
        if (body.contains(k) && !body[k].is_null()) {
            const long long value = integer_value(body[k], k);
            if (value <= 0 || value > std::numeric_limits<int>::max())
                http::fail(400, std::string(k) + " must be between 1 and 2147483647");
            if (selected_budget != nullptr && spec->max_tokens != value)
                http::fail(400, std::string(k) + " conflicts with " + selected_budget);
            spec->max_tokens = static_cast<int>(value);
            selected_budget = k;
        }
    }

    if (body.contains("stop") && !body["stop"].is_null()) {
        const json& s = body["stop"];
        if (s.is_string()) spec->stop.push_back(s.get<std::string>());
        else if (s.is_array()) {
            for (const auto& e : s) {
                if (!e.is_string()) http::fail(400, "stop entries must be strings");
                spec->stop.push_back(e.get<std::string>());
            }
        } else {
            http::fail(400, "stop must be a string or array of strings");
        }
        for (const std::string& stop : spec->stop)
            if (stop.empty()) http::fail(400, "stop strings must not be empty");
    }
}

json usage_json(const GenOutcome& o, const std::vector<int>& prompt_ids) {
    const long long pt = o.n_prompt > 0 ? o.n_prompt : (long long)prompt_ids.size();
    json u;
    u["prompt_tokens"] = pt;
    u["completion_tokens"] = o.n_gen;
    u["total_tokens"] = pt + o.n_gen;
    if (o.n_cached > 0) u["prompt_tokens_details"] = {{"cached_tokens", o.n_cached}};
    if (o.clamped_from > 0) u["clamped_from"] = o.clamped_from;
    if (!o.guard.empty()) u["think_guard"] = o.guard;
    return u;
}

json logprobs_json(const std::vector<float>& values) {
    json content = json::array();
    for (float value : values) {
        content.push_back({{"token", nullptr},
                           {"logprob", static_cast<double>(value)},
                           {"bytes", nullptr},
                           {"top_logprobs", json::array()}});
    }
    return json{{"content", std::move(content)}};
}

bool send_frame(http::Stream* st, const json& j) {
    return st->send_event(json_py::dumps(j, /*spaced=*/true));
}

bool stream_wants_usage(const json& body) {
    if (!body.contains("stream_options") || body["stream_options"].is_null()) return false;
    if (!body["stream_options"].is_object())
        http::fail(400, "stream_options must be an object");
    return bool_field(body["stream_options"], "include_usage", false);
}

const json* normalize_reasoning_effort(const json* value, json* storage) {
    if (value == nullptr || value->is_null()) return nullptr;
    if (!value->is_string()) http::fail(400, "reasoning_effort must be a string");
    std::string effort = value->get_ref<const std::string&>();
    std::transform(effort.begin(), effort.end(), effort.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (effort == "high" || effort == "max" || effort == "ultra" ||
        effort == "ultracode" || effort == "extreme") {
        *storage = "xhigh";
        return storage;
    }
    if (effort == "none" || effort == "off") {
        *storage = effort;
        return storage;
    }
    if (effort == "minimal") {
        *storage = "low";
        return storage;
    }
    if (effort == "low" || effort == "medium" || effort == "xhigh") {
        *storage = effort;
        return storage;
    }
    http::fail(400, "unsupported reasoning_effort: " + effort);
    return nullptr;
}

// ------------------------------------------------------- server overrides --
// Operator policy for sampling / thinking, applied to the request body before
// any parsing so the normal validation still runs on the result. A client
// therefore cannot switch thinking or send a broken sampler by mistake.
// Each entry is {"mode": "force" | "default", "value": v}:
//   force   -- replaces what the client sent (max_tokens: caps it instead);
//   default -- used only when the client omitted the field.
// Fields without an entry follow the client. Edited from the dashboard via
// GET/POST /admin/overrides and persisted to g_cfg.overrides_path.

struct OverrideField {
    const char* key;
    char kind;  // b = bool, n = number, i = integer, e = reasoning effort
    double lo, hi;
};
const OverrideField kOverrideFields[] = {
    {"enable_thinking", 'b', 0, 0},
    {"preserve_thinking", 'b', 0, 0},
    {"preserve_reasoning", 'b', 0, 0},
    {"auto_disable_thinking_with_tools", 'b', 0, 0},
    {"reasoning_effort", 'e', 0, 0},
    {"temperature", 'n', 0.0, 2.0},
    {"top_p", 'n', 0.0, 1.0},
    {"top_k", 'i', 0, 2147483647.0},
    {"min_p", 'n', 0.0, 1.0},
    {"presence_penalty", 'n', -2.0, 2.0},
    {"frequency_penalty", 'n', -2.0, 2.0},
    {"max_tokens", 'i', 1, 2147483647.0},
};

std::mutex g_ovr_mtx;
json g_overrides = json::object();

// Checks a posted table and returns its canonical form (mode "client" and
// null entries dropped, effort aliases resolved like the request path does).
json validate_overrides(const json& in) {
    if (!in.is_object()) http::fail(400, "overrides must be a JSON object");
    json out = json::object();
    for (auto it = in.begin(); it != in.end(); ++it) {
        const std::string& k = it.key();
        const OverrideField* f = nullptr;
        for (const auto& c : kOverrideFields)
            if (k == c.key) f = &c;
        if (f == nullptr) http::fail(400, "unknown override field " + k);
        const json& e = it.value();
        if (e.is_null()) continue;
        if (!e.is_object() || !e.contains("mode") || !e["mode"].is_string())
            http::fail(400, k + ": expected {\"mode\": ..., \"value\": ...}");
        const std::string mode = e["mode"].get<std::string>();
        if (mode == "client") continue;
        if (mode != "force" && mode != "default")
            http::fail(400, k + ": mode must be force, default or client");
        if (!e.contains("value") || e["value"].is_null()) http::fail(400, k + ": value is required");
        json v = e["value"];
        if (f->kind == 'b') {
            if (!v.is_boolean()) http::fail(400, k + " must be a boolean");
        } else if (f->kind == 'e') {
            json storage;
            const json* n = normalize_reasoning_effort(&v, &storage);
            const std::string s = n->get<std::string>();
            if (s != "xhigh" && s != "medium" && s != "low" && s != "none" && s != "off")
                http::fail(400, k + " must be one of none, off, xhigh, high, medium, low, minimal");
            v = s;
        } else {
            const bool ok = f->kind == 'i' ? (v.is_number_integer() || v.is_number_unsigned())
                                           : v.is_number();
            const double d = ok ? v.get<double>() : 0.0;
            if (!ok || !std::isfinite(d) || d < f->lo || d > f->hi)
                http::fail(400, k + " must be " + (f->kind == 'i' ? "an integer" : "a number") +
                                    " in " + json_py::py_float(f->lo) + ".." +
                                    json_py::py_float(f->hi));
        }
        out[k] = json{{"mode", mode}, {"value", std::move(v)}};
    }
    return out;
}

bool save_overrides(const json& table, std::string* err) {
    if (g_cfg.overrides_path.empty()) return true;
    const std::string tmp = g_cfg.overrides_path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f << table.dump(2) << "\n";
        if (!f.good()) {
            *err = "cannot write " + tmp;
            return false;
        }
    }
#ifdef _WIN32
    std::remove(g_cfg.overrides_path.c_str());  // rename() does not replace on Windows
#endif
    if (std::rename(tmp.c_str(), g_cfg.overrides_path.c_str()) != 0) {
        *err = "cannot rename " + tmp + " -> " + g_cfg.overrides_path;
        return false;
    }
    return true;
}

void load_overrides() {
    if (g_cfg.overrides_path.empty()) return;
    std::ifstream f(g_cfg.overrides_path, std::ios::binary);
    if (!f) return;  // no file yet: every field follows the client
    try {
        g_overrides = validate_overrides(json::parse(f));
        fprintf(stderr, "qwenox-api: overrides from %s: %s\n", g_cfg.overrides_path.c_str(),
                g_overrides.dump().c_str());
    } catch (const http::Error& e) {
        fprintf(stderr, "qwenox-api: ignoring %s: %s\n", g_cfg.overrides_path.c_str(),
                e.message.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "qwenox-api: ignoring %s: %s\n", g_cfg.overrides_path.c_str(), e.what());
    }
}

enum class Api { Completions, Chat, Responses };

// Rewrites `body` per the override table and reports the fields it changed in
// the X-Qwenox-Overrides response header.
void apply_overrides(json* body, Api api, http::Response* r) {
    json table;
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        table = g_overrides;
    }
    if (table.empty()) return;
    std::vector<std::string> changed;
    auto put = [&](json& obj, const char* key, const json& e, const std::string& label) {
        const bool present = obj.contains(key) && !obj[key].is_null();
        if (present && (e["mode"] != "force" || obj[key] == e["value"])) return;
        obj[key] = e["value"];
        changed.push_back(label);
    };
    for (auto it = table.begin(); it != table.end(); ++it) {
        const std::string& k = it.key();
        const json& e = it.value();
        if (k == "enable_thinking" || k == "preserve_thinking" ||
            k == "preserve_reasoning" || k == "auto_disable_thinking_with_tools") {
            if (api != Api::Completions) put(*body, k.c_str(), e, k);
        } else if (k == "reasoning_effort") {
            if (api == Api::Chat) {
                put(*body, "reasoning_effort", e, k);
            } else if (api == Api::Responses) {
                if (!body->contains("reasoning") || (*body)["reasoning"].is_null())
                    (*body)["reasoning"] = json::object();
                // A malformed `reasoning` is left for the normal 400.
                if ((*body)["reasoning"].is_object()) put((*body)["reasoning"], "effort", e, k);
            }
        } else if (k == "max_tokens") {
            const long long cap = e["value"].get<long long>();
            bool any = false;
            for (const char* a : {"max_tokens", "max_completion_tokens", "max_output_tokens"}) {
                if (!body->contains(a) || (*body)[a].is_null()) continue;
                any = true;
                json& v = (*body)[a];
                if (e["mode"] == "force" && v.is_number_integer() && v.get<long long>() > cap) {
                    v = cap;
                    changed.push_back(a);
                }
            }
            if (!any) {
                (*body)[api == Api::Responses ? "max_output_tokens" : "max_tokens"] = cap;
                changed.push_back("max_tokens");
            }
        } else {
            put(*body, k.c_str(), e, k);
        }
    }
    // A forced greedy decode cannot honour logprobs (apply_sampling rejects
    // the pair); the operator's choice wins, so the request loses logprobs.
    if (std::find(changed.begin(), changed.end(), "temperature") != changed.end() &&
        (*body)["temperature"] == 0 && body->contains("logprobs")) {
        body->erase("logprobs");
        body->erase("top_logprobs");
        changed.push_back("logprobs");
    }
    if (changed.empty()) return;
    std::string list;
    for (const auto& c : changed) list += (list.empty() ? "" : ",") + c;
    r->set("X-Qwenox-Overrides", list);
}

bool admin_ok(const http::Request& q) {
    if (g_cfg.admin_key.empty()) return true;
    return q.header("authorization") == "Bearer " + g_cfg.admin_key ||
           q.header("x-admin-key") == g_cfg.admin_key;
}

json overrides_state() {
    json j;
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        j["overrides"] = g_overrides;
    }
    j["persist_path"] = g_cfg.overrides_path;
    j["admin_key_required"] = !g_cfg.admin_key.empty();
    return j;
}

// GET /admin/overrides
void handle_overrides_get(const http::Request&, http::Response* r, http::Stream*) {
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(overrides_state(), /*spaced=*/false);
}

// POST /admin/overrides — replaces the whole table; {} clears it.
void handle_overrides_post(const http::Request& q, http::Response* r, http::Stream*) {
    if (!admin_ok(q))
        http::fail(401, "admin key required (Authorization: Bearer <key> or X-Admin-Key)");
    const json body = parse_body(q);
    json table = validate_overrides(body.contains("overrides") ? body["overrides"] : body);
    std::string err;
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        g_overrides = table;
    }
    const bool saved = save_overrides(table, &err);
    fprintf(stderr, "qwenox-api: overrides set by %s: %s%s\n", q.remote.c_str(),
            table.dump().c_str(), saved ? "" : (" (NOT persisted: " + err + ")").c_str());
    json j = overrides_state();
    j["saved"] = saved;
    if (!saved) j["save_error"] = err;
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// ------------------------------------------------------ service.conf 管理 --
// /admin/config：网页前端读写 service.conf 的受管键（与托盘面板同款，
// svcconf::kManagedKeys）。写回只改 conf 文件：运行中的引擎与 API 监听端口
// 不受影响，下次启动生效（响应 applied:"next_start"）。
//
// 权重版本（V1 单文件 / V2 主权重+独立 n-gram / GGUF）：与旧托盘面板一致，
// 版本选择持久化为注释行 "# start_win: weights=v1|v2|gguf"；无标记时按
// NGRAM_FILE 是否跟随 MODEL_FILE 推断 V1/V2（GGUF 只能来自标记）。
// 保存语义：V1 强制 NGRAM_FILE=$MODEL_FILE；V2 强制 OVERLAY_FILE 置空；
// GGUF 只写 GGUF_*（Windows 启停按标记 weights=gguf 改用 GGUF_* 键，enginesup::start）。

const char* kWeightsMarker = "# start_win: weights=";

std::string eff_engine_host() {
    std::lock_guard<std::mutex> lk(g_svc_conf_mtx);
    return svcconf::cfg(g_svc_conf, "ENGINE_HOST", "127.0.0.1");
}

int eff_engine_port() {
    std::lock_guard<std::mutex> lk(g_svc_conf_mtx);
    long p = 8730;
    svcconf::parse_int(svcconf::cfg(g_svc_conf, "ENGINE_PORT", "8730"), &p);
    return static_cast<int>(p);
}

json engine_status_json() {
    const enginesup::Status st = enginesup::status(eff_engine_host(), eff_engine_port());
    json j;
    j["state"] = st.state;
    j["pid"] = st.pid;
    j["exit_code"] = st.exit_code;
    j["log"] = st.log;
    j["external"] = st.external;
    j["detail"] = st.detail;
    j["can_control"] = enginesup::supported();
    return j;
}

// GET /admin/config
void handle_config_get(const http::Request&, http::Response* r, http::Stream*) {
    json j;
    j["path"] = g_conf_path;
    j["can_control_engine"] = enginesup::supported();
#ifdef _WIN32
    j["platform"] = "windows";
#else
    j["platform"] = "linux";
#endif
    j["admin_key_required"] = !g_cfg.admin_key.empty();
    json vals = json::object();
    json env_ovr = json::array();
    {
        std::lock_guard<std::mutex> lk(g_svc_conf_mtx);
        for (size_t i = 0; i < svcconf::kManagedKeysCount; ++i) {
            const char* key = svcconf::kManagedKeys[i];
            if (getenv(key)) env_ovr.push_back(key);  // 环境变量覆盖：conf 改了也不生效
            vals[key] = svcconf::cfg_optional(g_svc_conf, key, "");
        }
    }
    j["values"] = vals;
    j["env_overridden"] = env_ovr;
    {
        std::string model, ngram;
        {
            std::lock_guard<std::mutex> lk(g_svc_conf_mtx);
            model = svcconf::cfg(g_svc_conf, "MODEL_FILE", "");
            ngram = svcconf::cfg(g_svc_conf, "NGRAM_FILE", "");
        }
        const std::string marker = svcconf::conf_marker(g_conf_path, kWeightsMarker);
        if (marker == "v1" || marker == "v2" || marker == "gguf")
            j["weights_ver"] = marker;
        else
            j["weights_ver"] = (ngram.empty() || ngram == model) ? "v1" : "v2";
    }
    j["engine"] = engine_status_json();
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// POST /admin/config — {"values": {KEY: "value", ...}, "weights_ver": "v1|v2|gguf"}，
// 只接受受管键。带 weights_ver 时按版本语义过滤权重键（无关键静默丢弃）并写版本标记。
void handle_config_post(const http::Request& q, http::Response* r, http::Stream*) {
    if (!admin_ok(q))
        http::fail(401, "admin key required (Authorization: Bearer <key> or X-Admin-Key)");
    const json body = parse_body(q);
    const json& vals_j = body.contains("values") ? body["values"] : body;
    if (!vals_j.is_object()) http::fail(400, "expected {\"values\": {KEY: \"value\"}}");
    std::string weights_ver;
    if (body.contains("weights_ver") && !body["weights_ver"].is_null()) {
        if (!body["weights_ver"].is_string()) http::fail(400, "weights_ver must be a string");
        weights_ver = body["weights_ver"].get<std::string>();
        if (weights_ver != "v1" && weights_ver != "v2" && weights_ver != "gguf")
            http::fail(400, "weights_ver must be one of v1, v2, gguf");
    }
    std::map<std::string, std::string> vals;
    for (auto it = vals_j.begin(); it != vals_j.end(); ++it) {
        if (!svcconf::is_managed_key(it.key()))
            http::fail(400, "unknown or unmanaged key: " + it.key());
        if (!it.value().is_string()) http::fail(400, "value must be a string: " + it.key());
        vals[it.key()] = it.value().get<std::string>();
    }
    if (!weights_ver.empty()) {
        static const std::unordered_set<std::string> kWeightKeys = {
            "MODEL_FILE", "NGRAM_FILE", "OVERLAY_FILE", "MTP_FILE", "VISION_FILE",
            "GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE"};
        static const std::unordered_set<std::string> kVerKeys[] = {
            /* v1 */ {"MODEL_FILE", "OVERLAY_FILE", "MTP_FILE", "VISION_FILE"},
            /* v2 */ {"MODEL_FILE", "NGRAM_FILE", "MTP_FILE", "VISION_FILE"},
            /* gguf */ {"GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE"},
        };
        const auto& allow = kVerKeys[weights_ver == "v2"   ? 1
                                     : weights_ver == "gguf" ? 2
                                                             : 0];
        for (auto it = vals.begin(); it != vals.end();) {
            if (kWeightKeys.count(it->first) && !allow.count(it->first))
                it = vals.erase(it);
            else
                ++it;
        }
        if (weights_ver == "v1") vals["NGRAM_FILE"] = "$MODEL_FILE";
        if (weights_ver == "v2") vals["OVERLAY_FILE"] = "";
    }
    if (vals.empty() && weights_ver.empty()) http::fail(400, "no keys to save");
    std::string err;
    json env_ovr = json::array();
    {
        std::lock_guard<std::mutex> lk(g_svc_conf_mtx);
        // 合并到当前有效值后整表校验（端口互异、YaRN 上限、KV 池上限等）
        svcconf::Conf merged = g_svc_conf;
        for (const auto& kv : vals) merged[kv.first] = kv.second;
        if (!svcconf::validate_conf(merged, &err)) http::fail(400, err);
        if (!svcconf::save_conf(g_conf_path, vals, &err,
                                weights_ver.empty() ? nullptr : kWeightsMarker,
                                weights_ver))
            http::fail(500, err);
        // 从文件重解析作为内存有效值：V1 写入的 "$MODEL_FILE" 等需要展开
        g_svc_conf.clear();
        svcconf::load_conf(g_conf_path, &g_svc_conf);
        for (const auto& kv : vals)
            if (getenv(kv.first.c_str())) env_ovr.push_back(kv.first);
    }
    fprintf(stderr, "qwenox-api: service.conf updated by %s (%zu keys)\n", q.remote.c_str(),
            vals.size());
    json j;
    j["saved"] = true;
    j["applied"] = "next_start";  // 运行中的引擎/API 不受影响，下次启动生效
    if (!env_ovr.empty())
        j["env_overridden"] = env_ovr;  // 这些键被环境变量覆盖，conf 改动不生效
    j["engine"] = engine_status_json();
    r->body = json_py::dumps(j, /*spaced=*/false);
}

std::map<std::string, std::string> parse_query(const std::string& qs);

// p 的逐段前缀是否覆盖 root（两者都应是 canonical 后的路径）。
// Windows 下路径不区分大小写，逐段比较也不区分。
bool path_within_root(const std::filesystem::path& root, const std::filesystem::path& p) {
    auto ri = root.begin();
    auto pi = p.begin();
    for (; ri != root.end(); ++ri, ++pi) {
        if (pi == p.end()) return false;
#ifdef _WIN32
        if (_stricmp(ri->u8string().c_str(), pi->u8string().c_str()) != 0) return false;
#else
        if (*ri != *pi) return false;
#endif
    }
    return true;
}

// GET /admin/files[?path=子目录] — 供配置面板的权重路径键浏览选择文件。
// path 是相对 conf 根目录（项目根）的子路径：绝对路径、".." 段直接 400；
// canonical 解析（符号链接/junction 展开）后落在根外同样 400，不得越界。
// 目录与文件分开返回（TOKENIZER_DIR 选目录，其余键选文件）。
void handle_admin_files(const http::Request& q, http::Response* r, http::Stream*) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = fs::weakly_canonical(fs::path(g_conf_root), ec);
    if (ec) {
        r->status = 500;
        r->body = http::error_json("cannot resolve workspace root: " + ec.message(),
                                   "server_error", "server_error");
        return;
    }
    std::string rel;
    const auto qp = parse_query(q.query);
    if (auto it = qp.find("path"); it != qp.end()) rel = it->second;
    std::replace(rel.begin(), rel.end(), '\\', '/');
    while (!rel.empty() && rel.front() == '/') rel.erase(rel.begin());
    while (!rel.empty() && rel.back() == '/') rel.pop_back();
    std::string rel_out;
    fs::path target = root;
    if (!rel.empty() && rel != ".") {
        const fs::path sub = fs::u8path(rel).lexically_normal();
        bool bad = sub.is_absolute();
        for (const auto& part : sub)
            if (part == "..") bad = true;
        if (bad) http::fail(400, "path must stay within the workspace root");
        rel_out = sub.generic_u8string();
        if (rel_out == ".") rel_out.clear();
        target = root / sub;
    }
    const fs::path canon = fs::canonical(target, ec);
    if (ec || !path_within_root(root, canon))
        http::fail(400, "path is outside the workspace root or does not exist");
    if (!fs::is_directory(canon, ec) || ec)
        http::fail(400, "path is not a directory");
    json dirs = json::array();
    json files = json::array();
    for (const auto& de : fs::directory_iterator(canon, ec)) {
        const std::string name = de.path().filename().u8string();
        if (name.empty() || name[0] == '.') continue;  // 隐藏条目不进列表
        std::error_code ec2;
        json e;
        e["name"] = name;
        const auto ft = fs::last_write_time(de.path(), ec2);
        if (!ec2) {
            // C++17 没有 file_clock -> system_clock 的直接转换
            const auto st = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            e["mtime"] = (long long)std::chrono::system_clock::to_time_t(st);
        }
        if (de.is_directory(ec2) && !ec2) {
            dirs.push_back(std::move(e));
        } else if (de.is_regular_file(ec2) && !ec2) {
            const auto sz = fs::file_size(de.path(), ec2);
            if (!ec2) e["size"] = (long long)sz;
            files.push_back(std::move(e));
        }
        // 其它类型（管道等）不提供选择
    }
    if (ec) {
        r->status = 500;
        r->body = http::error_json("cannot list directory: " + ec.message(),
                                   "server_error", "server_error");
        return;
    }
    const auto by_name = [](const json& a, const json& b) {
        return a["name"].get_ref<const std::string&>() < b["name"].get_ref<const std::string&>();
    };
    std::sort(dirs.begin(), dirs.end(), by_name);
    std::sort(files.begin(), files.end(), by_name);
    json j;
    j["root"] = root.u8string();
    j["path"] = rel_out;
    j["dirs"] = std::move(dirs);
    j["files"] = std::move(files);
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// GET /admin/engine/status
void handle_engine_status(const http::Request&, http::Response* r, http::Stream*) {
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(engine_status_json(), /*spaced=*/false);
}

// POST /admin/engine/start — 仅 Windows；Linux 返回 501（改配置后由启动脚本生效）。
void handle_engine_start(const http::Request& q, http::Response* r, http::Stream*) {
    if (!admin_ok(q))
        http::fail(401, "admin key required (Authorization: Bearer <key> or X-Admin-Key)");
    if (!enginesup::supported())
        http::fail(501,
                   "engine start/stop is only supported on Windows; on Linux edit the "
                   "config and restart via start_hgn.sh / start_gguf.sh");
    const enginesup::Status st = enginesup::status(eff_engine_host(), eff_engine_port());
    if (st.state == "running" || st.state == "starting")
        http::fail(409, st.external ? "engine already running (started externally)"
                                    : "engine already running");
    svcconf::Conf conf;
    {
        std::lock_guard<std::mutex> lk(g_svc_conf_mtx);
        conf = g_svc_conf;
    }
    conf[enginesup::kWeightsVerKey] = svcconf::conf_marker(g_conf_path, kWeightsMarker);
    std::string err;
    if (!svcconf::validate_conf(conf, &err)) http::fail(400, err);
    if (!enginesup::start(g_conf_root, conf, &err)) http::fail(400, err);
    fprintf(stderr, "qwenox-api: engine start requested by %s\n", q.remote.c_str());
    r->body = json_py::dumps(engine_status_json(), /*spaced=*/false);
}

// POST /admin/engine/stop — 仅 Windows；外部引擎（非本进程拉起）返回 409。
void handle_engine_stop(const http::Request& q, http::Response* r, http::Stream*) {
    if (!admin_ok(q))
        http::fail(401, "admin key required (Authorization: Bearer <key> or X-Admin-Key)");
    if (!enginesup::supported())
        http::fail(501,
                   "engine start/stop is only supported on Windows; on Linux edit the "
                   "config and restart via start_hgn.sh / start_gguf.sh");
    const enginesup::Status st = enginesup::status(eff_engine_host(), eff_engine_port());
    if (st.external)
        http::fail(409, "engine was not started by this API process; stop it from its "
                        "own launcher");
    std::string err;
    if (!enginesup::stop(&err)) http::fail(409, err);
    fprintf(stderr, "qwenox-api: engine stop requested by %s\n", q.remote.c_str());
    r->body = json_py::dumps(engine_status_json(), /*spaced=*/false);
}

struct ToolSetup {
    json tools = json::array();
    toolparse::ToolChoice choice;

    bool enabled() const { return choice.tools_enabled() && !tools.empty(); }
};

ToolSetup parse_tool_setup(const json& body) {
    ToolSetup result;
    std::string error;
    const json* raw_tools = body.contains("tools") ? &body["tools"] : nullptr;
    if (!toolparse::normalize_tools(raw_tools, &result.tools, &error))
        http::fail(400, error);
    const json* raw_choice = body.contains("tool_choice") ? &body["tool_choice"] : nullptr;
    if (!toolparse::parse_tool_choice(raw_choice, result.tools, &result.choice, &error))
        http::fail(400, error);
    return result;
}

// A request with no callable tools (none declared, or tool_choice "none") has
// no use for <tool_call> / </tool_call>, yet the model sometimes opens a call
// anyway: an agent's title request, told to return only {"title":...},
// answered with a tool call and the client showed "<tool_call>" as the session
// title. Sampled requests get OpenAI's ban value (-100) on both tokens, except
// one the client biased itself. Text that spells the tags out is not touched
// (a prompt may ask for it). Greedy requests are left alone: the engine's bias
// is sampler-only and it rejects a greedy request that carries one.
// QWENOX_NOTOOLS_BAN=0 turns this off.
void ban_tool_call_without_tools(const ToolSetup& tools, GenSpec* spec) {
    static const bool off = [] {
        const char* e = std::getenv("QWENOX_NOTOOLS_BAN");
        return e != nullptr && std::strcmp(e, "0") == 0;
    }();
    if (off || tools.enabled() || !spec->sample || !g_tok_ready) return;
    static const int ids[] = {g_tok.added_token_id("<tool_call>"),
                              g_tok.added_token_id("</tool_call>")};
    for (int id : ids) {
        if (id < 0) continue;
        bool client = false;
        for (const auto& b : spec->bias) client = client || b.first == id;
        if (!client) spec->bias.emplace_back(id, -100.0f);
    }
}

bool responses_media_item(const json& item) {
    if (!item.is_object()) return false;
    const std::string type =
        item.contains("type") && item["type"].is_string()
            ? item["type"].get<std::string>()
            : std::string();
    return type == "image" || type == "image_url" || type == "input_image" ||
           type == "computer_screenshot" || type == "video" ||
           type == "input_video" || item.contains("image") ||
           item.contains("image_url") || item.contains("video") ||
           item.contains("video_url");
}

json responses_message_content(const json& content, const char* field,
                               bool allow_media) {
    if (content.is_string()) return content;
    if (!content.is_array())
        http::fail(400, std::string(field) + " must be a string or a content array");
    json normalized = json::array();
    for (const auto& part : content) {
        if (!part.is_object())
            http::fail(400, std::string(field) + " content items must be objects");
        if (responses_media_item(part)) {
            if (!allow_media)
                http::fail(400, std::string(field) + " must contain only text items");
            normalized.push_back(part);
            continue;
        }
        if (!part.contains("text") || !part["text"].is_string())
            http::fail(400, std::string(field) + " contains an unsupported content item");
        normalized.push_back(part);
    }
    return normalized;
}

std::string responses_text_content(const json& content, const char* field) {
    const json normalized = responses_message_content(content, field, false);
    if (normalized.is_string()) return normalized.get<std::string>();
    std::string text;
    for (const auto& part : normalized) text += part["text"].get<std::string>();
    return text;
}

json normalize_responses_input(const json& body) {
    if (!body.contains("input") || body["input"].is_null())
        http::fail(400, "input is required");

    std::string system_text;
    bool has_system = false;
    if (body.contains("instructions") && !body["instructions"].is_null()) {
        if (!body["instructions"].is_string())
            http::fail(400, "instructions must be a string");
        system_text = body["instructions"].get<std::string>();
        has_system = true;
    }

    json ordinary = json::array();
    std::string pending_reasoning;
    bool has_pending_reasoning = false;
    auto append_system = [&](const json& content) {
        const std::string text = responses_text_content(content, "system content");
        if (has_system && !system_text.empty() && !text.empty()) system_text += "\n\n";
        system_text += text;
        has_system = true;
    };
    auto append_user_content = [&](json content) {
        ordinary.push_back(json{{"role", "user"}, {"content", std::move(content)}});
    };
    auto append_user_text = [&](const std::string& text) {
        append_user_content(text);
    };
    auto append_item = [&](const json& item) {
        if (item.is_string()) {
            append_user_text(item.get<std::string>());
            return;
        }
        if (!item.is_object())
            http::fail(400, "input items must be strings or message objects");

        const std::string type =
            item.contains("type") && item["type"].is_string()
                ? item["type"].get<std::string>()
                : std::string();
        if (type == "reasoning") {
            pending_reasoning.clear();
            if (item.contains("content") && !item["content"].is_null())
                pending_reasoning = responses_text_content(item["content"], "reasoning content");
            has_pending_reasoning = true;
            return;
        }
        if (type == "function_call") {
            const json* call_id = nullptr;
            if (item.contains("call_id")) call_id = &item["call_id"];
            else if (item.contains("id")) call_id = &item["id"];
            if (call_id == nullptr || !call_id->is_string() ||
                call_id->get_ref<const std::string&>().empty())
                http::fail(400, "function_call items require a string call_id");
            if (!item.contains("name") || !item["name"].is_string() ||
                item["name"].get_ref<const std::string&>().empty())
                http::fail(400, "function_call items require a string name");
            json arguments = item.contains("arguments") ? item["arguments"] : json("{}");
            json call = {{"id", *call_id},
                         {"type", "function"},
                         {"function", {{"name", item["name"]}, {"arguments", arguments}}}};
            if (!has_pending_reasoning && !ordinary.empty() && ordinary.back().is_object() &&
                ordinary.back().contains("role") && ordinary.back()["role"] == "assistant") {
                if (!ordinary.back().contains("tool_calls"))
                    ordinary.back()["tool_calls"] = json::array();
                ordinary.back()["tool_calls"].push_back(std::move(call));
            } else {
                json assistant = {{"role", "assistant"},
                                  {"content", ""},
                                  {"tool_calls", json::array({std::move(call)})}};
                if (has_pending_reasoning) {
                    assistant["reasoning_content"] = pending_reasoning;
                    pending_reasoning.clear();
                    has_pending_reasoning = false;
                }
                ordinary.push_back(std::move(assistant));
            }
            return;
        }
        if (type == "function_call_output") {
            if (!item.contains("call_id") || !item["call_id"].is_string() ||
                item["call_id"].get_ref<const std::string&>().empty())
                http::fail(400, "function_call_output items require a string call_id");
            if (!item.contains("output"))
                http::fail(400, "function_call_output items require output");
            ordinary.push_back(
                json{{"role", "tool"},
                     {"tool_call_id", item["call_id"]},
                     {"content", responses_text_content(item["output"], "function output")}});
            return;
        }
        if (type == "input_text") {
            if (!item.contains("text") || !item["text"].is_string())
                http::fail(400, "input_text items require a string text field");
            append_user_text(item["text"].get<std::string>());
            return;
        }
        if (!item.contains("role") || !item["role"].is_string())
            http::fail(400, "input message items require a string role");
        if (!item.contains("content"))
            http::fail(400, "input message items require content");

        const std::string role = item["role"].get<std::string>();
        if (role == "system" || role == "developer") {
            append_system(item["content"]);
            return;
        }
        if (role != "user" && role != "assistant")
            http::fail(400, "only user and assistant input messages are supported in this build");
        json content = responses_message_content(item["content"], "message content",
                                                 role == "user");
        json message = {{"role", role}, {"content", std::move(content)}};
        if (role == "assistant" && has_pending_reasoning) {
            message["reasoning_content"] = pending_reasoning;
            pending_reasoning.clear();
            has_pending_reasoning = false;
        }
        if (item.contains("tool_calls")) message["tool_calls"] = item["tool_calls"];
        ordinary.push_back(std::move(message));
    };

    const json& input = body["input"];
    if (input.is_string()) {
        append_user_text(input.get<std::string>());
    } else if (input.is_array()) {
        if (input.empty()) http::fail(400, "input must not be empty");
        const bool flat_content = input[0].is_object() && !input[0].contains("role") &&
                                  input[0].contains("type") && input[0]["type"].is_string() &&
                                  input[0]["type"] != "function_call" &&
                                  input[0]["type"] != "function_call_output" &&
                                  input[0]["type"] != "reasoning";
        if (flat_content)
            append_user_content(responses_message_content(input, "input", true));
        else
            for (const auto& item : input) append_item(item);
    } else {
        http::fail(400, "input must be a string or array");
    }

    json messages = json::array();
    if (has_system) messages.push_back(json{{"role", "system"}, {"content", system_text}});
    for (auto& item : ordinary) messages.push_back(std::move(item));
    std::string error;
    if (!toolparse::normalize_messages(&messages, &error)) http::fail(400, error);
    return messages;
}

// ------------------------------------------------------------------ routes --

void handle_models(const http::Request&, http::Response* r, http::Stream*) {
    json j;
    j["object"] = "list";
    j["data"] = json::array({json{{"id", g_cfg.model},
                                  {"object", "model"},
                                  {"owned_by", "local"},
                                  {"created", 0}}});
    r->body = json_py::dumps(j, /*spaced=*/false);
}

void handle_memory(const http::Request&, http::Response* r, http::Stream*) {
    std::string line, err;
    {
        std::lock_guard<std::mutex> lk(g_conn_mtx);
        if (!g_ctl.connected()) g_ctl.connect(g_cfg.engine_addr, &err);
    }
    if (!g_ctl.memory(&line, &err)) {
        r->status = 500;
        r->body = http::error_json("engine memory query failed: " + err,
                                   "server_error", "server_error");
        return;
    }
    std::istringstream ss(line);
    std::string tag;
    unsigned long long version = 0, device = 0, device_peak = 0, registered = 0,
                       pinned = 0, pinned_peak = 0, committed = 0, hip_free = 0,
                       hip_total = 0, hip_delta = 0, rss = 0, locked = 0;
    if (!(ss >> tag >> version >> device >> device_peak >> registered >> pinned >>
          pinned_peak >> committed >> hip_free >> hip_total >> hip_delta >> rss >>
          locked) ||
        tag != "M" || version != 1) {
        r->status = 500;
        r->body = http::error_json("invalid engine memory reply", "server_error",
                                   "server_error");
        return;
    }
    json j;
    j["device_current_bytes"] = device;
    j["device_peak_bytes"] = device_peak;
    j["registered_mmap_bytes"] = registered;
    j["pinned_host_current_bytes"] = pinned;
    j["pinned_host_peak_bytes"] = pinned_peak;
    j["gpu_accessible_committed_bytes"] = committed;
    j["hip_free_bytes"] = hip_free;
    j["hip_total_bytes"] = hip_total;
    j["hip_used_since_engine_start_bytes"] = hip_delta;
    j["process_rss_bytes"] = rss;
    j["process_locked_bytes"] = locked;
    j["accounting"] =
        "engine-requested HIP allocations; unregistered pageable mmap is excluded";
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// GET /viz[?slot=S&from=F&n=N] — 运行时可视化数据（引擎 QWENOX_VIZ=1 时有效）。
// 无参数：各 slot 概况；带 slot：该 slot 当前请求 [F, F+N) 位置的逐层专家路由、
// 路由集中度/熵、top-5 候选（字段说明见 src/gpu/parts/19_viz.inc），每个位置
// 另附 b = 该 token 的原始字节（hex，前端按流拼 UTF-8）。允许跨域读取。
void handle_viz(const http::Request& q, http::Response* r, http::Stream*) {
    r->set("Access-Control-Allow-Origin", "*");
    r->set("Cache-Control", "no-store");
    const auto qp = parse_query(q.query);
    std::string args;
    const auto it = qp.find("slot");
    if (it != qp.end()) {
        const auto num = [&](const char* k, int d) {
            const auto f = qp.find(k);
            return f == qp.end() ? d : atoi(f->second.c_str());
        };
        args = std::to_string(atoi(it->second.c_str())) + " " +
               std::to_string(num("from", 0)) + " " + std::to_string(num("n", 64));
    }
    std::string line, err;
    {
        std::lock_guard<std::mutex> lk(g_conn_mtx);
        if (!g_ctl.connected()) g_ctl.connect(g_cfg.engine_addr, &err);
    }
    if (!g_ctl.viz(args, &line, &err) || line.size() < 2 || line[0] != 'V') {
        r->status = 502;
        r->body = http::error_json("engine viz query failed: " + err, "server_error",
                                   "server_error");
        return;
    }
    json j = json::parse(line.substr(2), nullptr, false);
    if (j.is_discarded()) {
        r->status = 502;
        r->body = http::error_json("invalid engine viz reply", "server_error", "server_error");
        return;
    }
    if (g_tok_ready && j.contains("items") && j["items"].is_array()) {
        static const char* hx = "0123456789abcdef";
        for (auto& itm : j["items"]) {
            if (!itm.contains("t")) continue;
            const int t = itm["t"].get<int>();
            const std::string b = g_tok.decode_bytes({t}, false);
            std::string h;
            h.reserve(b.size() * 2);
            for (unsigned char ch : b) {
                h += hx[ch >> 4];
                h += hx[ch & 15];
            }
            itm["b"] = h;
            if (g_tok.is_special_token(t)) itm["sp"] = 1;
            if (itm.contains("k") && itm["k"].is_array()) {  // top-5 candidates' bytes
                json kb = json::array();
                for (const auto& kv : itm["k"]) {
                    const int kt = kv.get<int>();
                    std::string kh;
                    if (kt >= 0)
                        for (unsigned char ch : g_tok.decode_bytes({kt}, false)) {
                            kh += hx[ch >> 4];
                            kh += hx[ch & 15];
                        }
                    kb.push_back(kh);
                }
                itm["kb"] = kb;
            }
        }
    }
    r->body = j.dump();
}

// GET /power — APU 功耗与温度瞬时值（Windows ADL PMLog / Linux amdgpu hwmon +
// k10temp），
// 读不到的字段为 null，完全不可用时 available=false。
void handle_power(const http::Request&, http::Response* r, http::Stream*) {
    const power::Reading p = power::read();
    json j;
    j["available"] = p.available;
    if (p.socket_watts >= 0) j["socket_watts"] = p.socket_watts;
    else j["socket_watts"] = nullptr;
    if (p.gfx_watts >= 0) j["gfx_watts"] = p.gfx_watts;
    else j["gfx_watts"] = nullptr;
    if (p.cpu_watts >= 0) j["cpu_watts"] = p.cpu_watts;
    else j["cpu_watts"] = nullptr;
    // 温度（°C），与功耗同一次快照；读不到的为 null。
    j["temp_available"] = p.temp_available;
    const auto put_temp = [&j](const char* k, int v) {
        if (v >= 0) j[k] = v;
        else j[k] = nullptr;
    };
    put_temp("gpu_temp_c", p.gpu_temp_c);
    put_temp("gpu_hotspot_c", p.gpu_hotspot_c);
    put_temp("cpu_temp_c", p.cpu_temp_c);
    put_temp("soc_temp_c", p.soc_temp_c);
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// --------------------------------------------------------- request stats --

// GET /reqstat/summary?from=YYYY-MM-DD&to=YYYY-MM-DD — aggregates over the
// reqstat file chain (UTC dates, same semantics as tools/reqstat_dump.py;
// no params = all history). GET /reqstat/tail?n=N — last N records,
// oldest-first. GET /reqstat/page?offset=K&limit=M — paged slice counting
// back from the newest record (offset=0&limit=n equals tail(n)), plus
// "total" so a client can compute the page count. All read-only, served
// without touching the engine.

std::string url_decode(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size()) {
            const auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            const int hi = hex(s[i + 1]), lo = hex(s[i + 2]);
            if (hi >= 0 && lo >= 0) {
                out += (char)(hi * 16 + lo);
                i += 2;
                continue;
            }
        }
        out += s[i] == '+' ? ' ' : s[i];
    }
    return out;
}

std::map<std::string, std::string> parse_query(const std::string& qs) {
    std::map<std::string, std::string> out;
    for (size_t i = 0; i <= qs.size();) {
        const size_t amp = qs.find('&', i);
        const std::string kv = qs.substr(i, amp == std::string::npos
                                                 ? std::string::npos
                                                 : amp - i);
        if (!kv.empty()) {
            const size_t eq = kv.find('=');
            out[url_decode(kv.substr(0, eq))] =
                eq == std::string::npos ? "" : url_decode(kv.substr(eq + 1));
        }
        if (amp == std::string::npos) break;
        i = amp + 1;
    }
    return out;
}

// Howard Hinnant's days_from_civil: days since 1970-01-01, proleptic
// Gregorian — no timezone dependence, so UTC dates behave the same on
// Linux and Windows.
int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = (unsigned)(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + (int64_t)doe - 719468;
}

bool parse_date_ms(const std::string& s, bool end_of_day, uint64_t* ms) {
    int y;
    unsigned m, d;
    if (sscanf(s.c_str(), "%d-%u-%u", &y, &m, &d) != 3 || m < 1 || m > 12 ||
        d < 1 || d > 31)
        return false;
    int64_t v = days_from_civil(y, m, d) * 86400000LL +
                (end_of_day ? 86399999LL : 0);
    *ms = (uint64_t)(v > 0 ? v : 0);
    return true;
}

const char* finish_name(uint32_t f) {
    switch (f) {
        case 1: return "stop";
        case 2: return "length";
        case 3: return "cancel";
        case 4: return "error";
        default: return "other";
    }
}

const char* drafter_name(uint32_t d) {
    switch (d) {
        case 0: return "serial";
        case 1: return "mtp";
        case 2: return "dflash2";
        case 3: return "ngram";
        case 4: return "chain";
        default: return "other";
    }
}

void handle_reqstat_summary(const http::Request& q, http::Response* r,
                            http::Stream*) {
    const auto qp = parse_query(q.query);
    uint64_t t0 = 0, t1 = ~(uint64_t)0;
    if (auto it = qp.find("from"); it != qp.end() && !it->second.empty() &&
        !parse_date_ms(it->second, false, &t0))
        http::fail(400, "from must be YYYY-MM-DD (UTC)");
    if (auto it = qp.find("to"); it != qp.end() && !it->second.empty() &&
        !parse_date_ms(it->second, true, &t1))
        http::fail(400, "to must be YYYY-MM-DD (UTC)");
    if (t0 > t1) http::fail(400, "from is after to");

    uint64_t n = 0, tp = 0, tc = 0, tg = 0;
    double pre_s = 0, dec_s = 0;
    std::vector<uint32_t> ttfts;
    std::map<std::string, uint64_t> finish;
    struct DStat {
        uint64_t requests = 0, accepted = 0, proposed = 0, gen = 0;
        double decode_s = 0;
    };
    std::map<std::string, DStat> drafters;
    reqstat::ScanInfo info;
    std::string err;
    if (!reqstat::scan(
            t0, t1,
            [&](const reqstat::QEntry& e) {
                n++;
                tp += e.n_prompt;
                tc += e.n_cached;
                tg += e.n_gen;
                pre_s += e.prefill_us / 1e6;
                dec_s += e.decode_us / 1e6;
                if (e.ttft_us > 0) ttfts.push_back(e.ttft_us);
                finish[finish_name(e.flags & 0xF)]++;
                // 速度按起草器分桶（串行也要出现：它就是拉低总速度的那类）；
                // 接受率只在真正起草过的请求上有意义。
                auto& s = drafters[drafter_name((e.flags >> 4) & 0xF)];
                s.requests++;
                s.gen += e.n_gen;
                s.decode_s += e.decode_us / 1e6;
                if (e.proposed > 0) {
                    s.proposed += e.proposed;
                    s.accepted +=
                        e.commit >= e.rounds ? e.commit - e.rounds : 0;
                }
                return true;
            },
            &info, &err)) {
        r->status = 500;
        r->body = http::error_json("reqstat scan failed: " + err,
                                   "server_error", "server_error");
        return;
    }

    json j;
    j["from_ms"] = t0;
    if (t1 != ~(uint64_t)0) j["to_ms"] = t1; else j["to_ms"] = nullptr;
    j["requests"] = n;
    j["input_tokens"] = tp;
    j["cached_tokens"] = tc;
    j["output_tokens"] = tg;
    j["cache_hit_rate"] = tp ? json(tc * 1.0 / tp) : json(nullptr);
    j["prefill_seconds"] = pre_s;
    j["prefill_tok_per_s"] =
        tp > tc && pre_s > 0 ? json((tp - tc) / pre_s) : json(nullptr);
    j["decode_seconds"] = dec_s;
    j["decode_tok_per_s"] =
        dec_s > 0 ? json(tg / dec_s) : json(nullptr);
    if (!ttfts.empty()) {
        std::sort(ttfts.begin(), ttfts.end());
        j["ttft"] = {{"count", ttfts.size()},
                     {"p50_ms", ttfts[ttfts.size() / 2] / 1000.0},
                     {"p95_ms", ttfts[(size_t)(ttfts.size() * 0.95)] / 1000.0}};
    } else {
        j["ttft"] = nullptr;
    }
    j["finish"] = finish.empty() ? json::object() : json(finish);
    json dj = json::object();
    for (const auto& [name, s] : drafters) {
        dj[name] = {{"requests", s.requests},
                    {"accepted", s.accepted},
                    {"proposed", s.proposed},
                    {"acceptance",
                     s.proposed ? json(s.accepted * 1.0 / s.proposed)
                                : json(nullptr)},
                    {"decode_tok_per_s",
                     s.decode_s > 0 ? json(s.gen / s.decode_s)
                                    : json(nullptr)}};
    }
    j["drafters"] = dj;
    j["files_total"] = info.files_total;
    j["files_scanned"] = info.files_scanned;
    j["bad_crc"] = info.bad_crc;
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// GET /reqstat/usage?from_ms=K&to_ms=M — hourly aggregates over the reqstat
// chain, cached in-process. Records are stamped when the request completes,
// so an hour that has already ended is immutable: it is scanned at most once.
// Hours that have not ended yet are rescanned, and the first/last hours are
// scanned exactly for the requested window instead of using the cache.
// Buckets are keyed by UTC hour start (epoch ms); the client re-buckets them
// into local days, so the server needs no timezone/DST policy.
struct UsageBucket {
    uint64_t requests = 0, prompt = 0, cached = 0, gen = 0;
};

struct UsageCache {
    std::mutex mu;
    std::map<uint64_t, UsageBucket> hours;  // hour_start_ms -> counters
    uint64_t last_now_ms = 0;               // clock-rollback detection
};

UsageCache g_usage_cache;

void handle_reqstat_usage(const http::Request& q, http::Response* r,
                          http::Stream*) {
    constexpr uint64_t kHour = 3600000;
    constexpr uint64_t kMaxRangeMs = 366ull * 86400000ull;
    const auto qp = parse_query(q.query);
    const auto parse_ms = [&](const char* key, uint64_t* out) {
        const auto it = qp.find(key);
        if (it == qp.end() || it->second.empty())
            http::fail(400, std::string(key) + " is required");
        char* end = nullptr;
        errno = 0;
        const unsigned long long v = strtoull(it->second.c_str(), &end, 10);
        if (!end || *end || errno == ERANGE)
            http::fail(400, std::string(key) + " must be epoch milliseconds");
        *out = (uint64_t)v;
    };
    uint64_t from = 0, to = 0;
    parse_ms("from_ms", &from);
    parse_ms("to_ms", &to);
    if (from > to) http::fail(400, "from_ms is after to_ms");
    if (to - from > kMaxRangeMs)
        http::fail(400, "range too large (max 366 days)");

    const uint64_t now = (uint64_t)std::chrono::duration_cast<
                             std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();
    // Keeps all hour arithmetic away from UINT64_MAX and rejects nonsense
    // windows. One day of slack tolerates a client clock slightly ahead.
    if (to > now + 86400000ull)
        http::fail(400, "to_ms is too far in the future");
    const uint64_t first_hour = from / kHour * kHour;
    const uint64_t last_hour = to / kHour * kHour;

    UsageCache& c = g_usage_cache;
    std::lock_guard<std::mutex> lock(c.mu);
    if (now < c.last_now_ms) c.hours.clear();  // clock moved backwards
    c.last_now_ms = std::max(c.last_now_ms, now);

    uint64_t files_scanned = 0, bad_crc = 0;
    std::string err;
    const auto scan_into = [&](uint64_t t0, uint64_t t1,
                               UsageBucket* out) -> bool {
        reqstat::ScanInfo si;
        if (!reqstat::scan(
                t0, t1,
                [&](const reqstat::QEntry& e) {
                    out->requests++;
                    out->prompt += e.n_prompt;
                    out->cached += e.n_cached;
                    out->gen += e.n_gen;
                    return true;
                },
                &si, &err)) {
            r->status = 500;
            r->body = http::error_json("reqstat scan failed: " + err,
                                       "server_error", "server_error");
            return false;
        }
        files_scanned += si.files_scanned;
        bad_crc += si.bad_crc;
        return true;
    };

    // Hours fully covered by [from,to] come from the cache; the first and
    // last hours are partial and are scanned exactly, so totals match the
    // requested window to the millisecond.
    const uint64_t first_full =
        from % kHour == 0 ? first_hour : first_hour + kHour;
    uint64_t last_full = last_hour;
    bool has_full = true;
    if (to % kHour != kHour - 1) {
        if (last_hour == 0) has_full = false;
        else last_full = last_hour - kHour;
    }
    if (first_full > last_full) has_full = false;

    // Cache misses plus hours that have not ended yet. A short grace period
    // on the previous hour covers a record timestamped before the boundary
    // that is still on its way to the writer.
    constexpr uint64_t kGraceMs = 60000;
    const uint64_t open_cutoff =
        now > kHour + kGraceMs ? now - kHour - kGraceMs : 0;
    bool any_missing = false;
    uint64_t scan_first = 0, scan_last = 0;
    if (has_full) {
        for (uint64_t h = first_full;; h += kHour) {
            const bool still_open = h >= open_cutoff;
            if (still_open || c.hours.find(h) == c.hours.end()) {
                if (!any_missing) {
                    scan_first = h;
                    any_missing = true;
                }
                scan_last = h;
            }
            if (h >= last_full) break;
        }
    }
    if (any_missing) {
        std::map<uint64_t, UsageBucket> local;
        for (uint64_t h = scan_first;; h += kHour) {
            local[h];  // materialize zero buckets across the whole scan range
            if (h >= scan_last) break;
        }
        reqstat::ScanInfo si;
        const uint64_t scan_t1 = scan_last > ~(uint64_t)0 - (kHour - 1)
                                     ? ~(uint64_t)0
                                     : scan_last + kHour - 1;
        uint64_t cur_hour = ~(uint64_t)0;
        UsageBucket* cur = nullptr;
        if (!reqstat::scan(
                scan_first, scan_t1,
                [&](const reqstat::QEntry& e) {
                    const uint64_t h = e.ts_ms / kHour * kHour;
                    if (h != cur_hour) {  // records are approximately ordered
                        const auto it = local.find(h);
                        cur = it == local.end() ? nullptr : &it->second;
                        cur_hour = h;
                    }
                    if (!cur) return true;
                    cur->requests++;
                    cur->prompt += e.n_prompt;
                    cur->cached += e.n_cached;
                    cur->gen += e.n_gen;
                    return true;
                },
                &si, &err)) {
            r->status = 500;
            r->body = http::error_json("reqstat scan failed: " + err,
                                       "server_error", "server_error");
            return;
        }
        files_scanned += si.files_scanned;
        bad_crc += si.bad_crc;
        for (const auto& [h, b] : local) c.hours[h] = b;
    }

    bool head_partial = from % kHour != 0;
    bool tail_partial = to % kHour != kHour - 1;
    const bool same_hour = first_hour == last_hour;
    UsageBucket edge_first, edge_last, edge_single;
    const bool single_partial = same_hour && (head_partial || tail_partial);
    if (single_partial) {
        if (!scan_into(from, to, &edge_single)) return;
        head_partial = tail_partial = false;
    } else {
        if (head_partial &&
            !scan_into(from, first_hour + kHour - 1, &edge_first))
            return;
        if (tail_partial && !scan_into(last_hour, to, &edge_last)) return;
    }

    json arr = json::array();
    for (uint64_t h = first_hour;; h += kHour) {
        UsageBucket b;
        if (single_partial && h == first_hour) b = edge_single;
        else if (head_partial && h == first_hour) b = edge_first;
        else if (tail_partial && h == last_hour) b = edge_last;
        else {
            const auto it = c.hours.find(h);
            if (it != c.hours.end()) b = it->second;
        }
        json o;
        o["ts_ms"] = h;
        o["requests"] = b.requests;
        o["input_tokens"] = b.prompt;
        o["cached_tokens"] = b.cached;
        o["output_tokens"] = b.gen;
        arr.push_back(o);
        if (h >= last_hour) break;
    }
    // Evict only after the response snapshot: an old range must not be
    // dropped from the cache before it has been serialized.
    while (c.hours.size() > 100000) c.hours.erase(c.hours.begin());  // ~11y
    json j;
    j["from_ms"] = from;
    j["to_ms"] = to;
    j["buckets"] = arr;
    j["cache_hours"] = (uint64_t)c.hours.size();
    j["files_scanned"] = files_scanned;
    j["bad_crc"] = bad_crc;
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// One record as JSON — shared by /reqstat/tail and /reqstat/page.
json reqstat_record_json(const reqstat::QEntry& e) {
    json o;
    o["ts_ms"] = e.ts_ms;
    o["req_seq"] = e.req_seq;
    o["finish"] = finish_name(e.flags & 0xF);
    o["drafter"] = drafter_name((e.flags >> 4) & 0xF);
    o["vision"] = (e.flags & 0x100) != 0;
    o["prompt_tokens"] = e.n_prompt;
    o["cached_tokens"] = e.n_cached;
    o["output_tokens"] = e.n_gen;
    o["ttft_ms"] = e.ttft_us / 1000.0;
    o["prefill_ms"] = e.prefill_us / 1000.0;
    o["decode_ms"] = e.decode_us / 1000.0;
    o["proposed"] = e.proposed;
    o["commit"] = e.commit;
    o["rounds"] = e.rounds;
    o["acceptance"] =
        e.proposed
            ? json((e.commit >= e.rounds ? e.commit - e.rounds : 0) * 1.0 /
                   e.proposed)
            : json(nullptr);
    return o;
}

void handle_reqstat_tail(const http::Request& q, http::Response* r,
                         http::Stream*) {
    const auto qp = parse_query(q.query);
    long n = 20;
    if (auto it = qp.find("n"); it != qp.end() && !it->second.empty()) {
        char* end = nullptr;
        n = strtol(it->second.c_str(), &end, 10);
        if (!end || *end || n < 1) http::fail(400, "n must be a positive integer");
        if (n > 1000) n = 1000;
    }
    std::vector<reqstat::QEntry> recs;
    reqstat::ScanInfo info;
    std::string err;
    if (!reqstat::tail((uint32_t)n, &recs, &info, &err)) {
        r->status = 500;
        r->body = http::error_json("reqstat read failed: " + err,
                                   "server_error", "server_error");
        return;
    }
    json arr = json::array();
    for (const auto& e : recs) arr.push_back(reqstat_record_json(e));
    json j;
    j["records"] = arr;
    j["files_total"] = info.files_total;
    j["files_scanned"] = info.files_scanned;
    j["bad_crc"] = info.bad_crc;
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

void handle_reqstat_page(const http::Request& q, http::Response* r,
                         http::Stream*) {
    const auto qp = parse_query(q.query);
    long long offset = 0;
    if (auto it = qp.find("offset"); it != qp.end() && !it->second.empty()) {
        char* end = nullptr;
        offset = strtoll(it->second.c_str(), &end, 10);
        if (!end || *end || offset < 0)
            http::fail(400, "offset must be a non-negative integer");
    }
    long limit = 20;
    if (auto it = qp.find("limit"); it != qp.end() && !it->second.empty()) {
        char* end = nullptr;
        limit = strtol(it->second.c_str(), &end, 10);
        if (!end || *end || limit < 1)
            http::fail(400, "limit must be a positive integer");
        if (limit > 1000) limit = 1000;
    }
    std::vector<reqstat::QEntry> recs;
    uint64_t total = 0;
    reqstat::ScanInfo info;
    std::string err;
    if (!reqstat::page((uint64_t)offset, (uint32_t)limit, &recs, &total,
                       &info, &err)) {
        r->status = 500;
        r->body = http::error_json("reqstat read failed: " + err,
                                   "server_error", "server_error");
        return;
    }
    json arr = json::array();
    for (const auto& e : recs) arr.push_back(reqstat_record_json(e));
    json j;
    j["records"] = arr;
    j["total"] = total;
    j["offset"] = (uint64_t)offset;
    j["limit"] = (uint64_t)limit;
    j["files_total"] = info.files_total;
    j["files_scanned"] = info.files_scanned;
    j["bad_crc"] = info.bad_crc;
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// GET 静态页 —— src/api/static/ 下的文件由构建时工具(tools/gen_static_inc.cpp)
// 嵌入 static_gen.inc，路由在 kStaticAssets，启动时逐条注册 GET。
// dashboard 页面加载后用同源 fetch 轮询 /health 与 /memory，这里只负责回内容。
//
// 磁盘覆盖：g_web_dir（service.conf/环境变量 WEB_DIR，默认 conf 所在目录的 web/）
// 下有同名文件时，每次请求现读现回，改页面只需刷新浏览器、不用重编 API。只按
// kStaticAssets 里的固定文件名查找，URL 不参与拼路径；读不到就回内置版本。
#include "static_gen.inc"

std::string g_web_dir;  // 静态页磁盘覆盖目录；空 = 只用内置

static std::string web_override_path(const StaticAsset& a) { return g_web_dir + "/" + a.file; }

http::Server::Handler make_static_handler(const StaticAsset& asset) {
    return [asset](const http::Request&, http::Response* r, http::Stream*) {
        r->content_type = asset.mime;
        r->set("Cache-Control", "no-store");
        if (!g_web_dir.empty()) {
            std::string body;
            if (svcconf::read_text_file(web_override_path(asset), &body)) {
                r->set("X-Qwenox-Static", "disk");
                r->body = std::move(body);
                return;
            }
        }
        r->body.assign(reinterpret_cast<const char*>(asset.data), asset.len);
    };
}

// 启动时解析 WEB_DIR 并列出当前被磁盘版覆盖的页面（之后增删文件不必重启，
// 只是这份清单不再更新）。相对路径以 conf 所在目录为基准。
static void init_web_dir() {
    const std::string conf_v = svcconf::cfg(g_svc_conf, "WEB_DIR", "");
    std::string dir = conf_v.empty() ? g_conf_root + "/web" : conf_v;
    const bool absolute = dir[0] == '/' || dir[0] == '\\' || (dir.size() > 1 && dir[1] == ':');
    if (!absolute) dir = g_conf_root + "/" + dir;
    while (dir.size() > 1 && (dir.back() == '/' || dir.back() == '\\')) dir.pop_back();
    std::error_code ec;
    if (!std::filesystem::is_directory(std::filesystem::path(dir), ec)) {
        // 默认的 web/ 不存在是常态，不提示；显式配了却找不到才警告
        if (!conf_v.empty())
            fprintf(stderr, "[warn] WEB_DIR=%s 不是目录，静态页只用内置版本\n", dir.c_str());
        g_web_dir = dir;  // 仍保留：之后建了目录、放进文件也能生效
        return;
    }
    g_web_dir = dir;
    std::string list;
    for (const StaticAsset& a : kStaticAssets)
        if (std::filesystem::is_regular_file(std::filesystem::path(web_override_path(a)), ec)) list += std::string(" ") + a.file;
    fprintf(stderr, "qwenox-api: web: %s 下的同名文件优先于内置页面%s%s\n", dir.c_str(),
            list.empty() ? "（当前没有）" : "，当前覆盖:", list.c_str());
}

void handle_health(const http::Request&, http::Response* r, http::Stream*) {
    json j;
    j["status"] = "ok";
    j["version"] = QWENOX_VERSION;
    j["build"] = QWENOX_GIT;
    j["model"] = g_cfg.model;
    j["tokenizer"] = g_tok_ready;  // false = 词表缺失，推理端点 503
    j["endpoints"] = json::array(
        {"/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/responses",
         "/v1/messages", "/v1/messages/count_tokens", "/v1/tokenize", "/v1/detokenize", "/dashboard", "/reqstat/summary", "/reqstat/usage", "/reqstat/tail", "/reqstat/page"});
    j["context"] = g_cfg.context;
    if (g_cfg.rope.factor > 1.0) {
        j["rope_scaling"] = {{"type", "yarn"},
                              {"factor", g_cfg.rope.factor},
                              {"original_max_position_embeddings",
                               g_cfg.rope.original_max_position_embeddings},
                              {"beta_fast", g_cfg.rope.beta_fast},
                              {"beta_slow", g_cfg.rope.beta_slow},
                              {"attention_factor", rope_attention_factor(g_cfg.rope)}};
    } else {
        j["rope_scaling"] = nullptr;
    }
    {
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        j["busy"] = g_in_flight > 0;
        j["in_flight"] = g_in_flight;
        j["busy_for_s"] = g_in_flight > 0
                              ? (int)std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - g_busy_since)
                                    .count()
                              : 0;
    }
    j["slots"] = g_slots;
    j["slot_ctx"] = g_cfg.context;
    j["kv_pool"] = g_kv_pool;
    {
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        j["queued"] = g_queued;
    }
    // 引擎状态给控制台指示灯用（ENG 灯）：stopped/starting/running/exited；
    // external = 外部拉起的引擎（脚本/旧启动器），不可由 API 停止
    const enginesup::Status est = enginesup::status(eff_engine_host(), eff_engine_port());
    j["engine"] = est.state;
    j["engine_external"] = est.external;
    j["decode"] = std::string(
        "greedy at explicit temperature 0; Qwen-default sampling (temp 1.0, top_k 20, "
        "top_p 0.95) when temperature is omitted; sampled at temperature > 0");
    j["prompt_cache"] = {{"enabled", false}, {"cap_mb", 0}};
    j["tool_calls"] = {{"parsed", true},
                       {"template_injection", true},
                       {"wire_format", "qwen-xml (<function=>/<parameter=>)"},
                       {"streaming", true},
                       {"tool_choice", json::array({"auto", "none", "required", "named"})},
                       {"parallel_tool_calls", true},
                       {"constrained_decoding", false}};
    j["vision"] = {{"accepted", true},
                   {"chat_content_type", "image_url"},
                   {"responses_content_type", "input_image"},
                   {"inline_data_url", true},
                   {"bare_base64", true},
                   {"http_urls", false},
                   {"formats", json::array({"png", "jpeg", "webp"})},
                   {"max_images", vision::max_images()},
                   {"video", false},
                   {"runtime", "native-cpp"}};
    j["supported"] = json::array(
        {"reasoning_effort", "enable_thinking", "preserve_thinking", "preserve_reasoning",
         "auto_disable_thinking_with_tools", "tool_call_format", "max_tool_arg_chars",
         "max_tool_response_chars", "stop",
         "max_tokens", "max_completion_tokens", "max_output_tokens", "stream",
         "seed", "temperature", "top_p", "top_k", "min_p", "presence_penalty",
         "frequency_penalty", "logit_bias", "logprobs", "tools", "tool_choice",
         "parallel_tool_calls", "image_url", "input_image", "add_vision_id"});
    j["token_budget_aliases"] =
        json::array({"max_tokens", "max_completion_tokens", "max_output_tokens"});
    j["max_tokens_default"] = "unbounded (budget = context minus prompt length)";
    j["error_format"] = "openai";
    j["accepted_but_ignored"] = json::array({"n", "drafter"});
    j["partial"] = json::array();
    j["sampling"] = {
        {"implemented", json::array({"temperature", "top_p", "top_k", "min_p", "seed",
                                     "presence_penalty", "frequency_penalty",
                                     "logit_bias", "logprobs"})}};
    j["max_tokens_cap"] = nullptr;
    j["reasoning_effort_values"] =
        json::array({"none", "off", "minimal", "low", "medium", "high", "xhigh",
                     "max", "ultracode", "extreme"});
    {
        std::lock_guard<std::mutex> lk(g_ovr_mtx);
        j["server_overrides"] = g_overrides;
    }
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// POST /v1/completions — raw prompt, no chat template.
void handle_completions(const http::Request& q, http::Response* r, http::Stream* st) {
    if (!g_tok_ready)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    json mutable_body = parse_body(q);
    apply_overrides(&mutable_body, Api::Completions, r);
    const json& body = mutable_body;
    GenSpec spec;
    if (body.contains("prompt")) {
        const json& p = body["prompt"];
        const json* text = p.is_string() ? &p
                           : (p.is_array() && !p.empty() && p[0].is_string()) ? &p[0]
                                                                               : nullptr;
        if (text == nullptr) http::fail(400, kEngineRejected);
        std::vector<size_t> starts;
        std::string err;
        encode_spliced(text->get_ref<const std::string&>(), nullptr, &spec.ids, &starts, &err);
    } else {
        http::fail(400, kEngineRejected);  // an empty prompt is an engine rejection
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    apply_sampling(body, &spec);

    const bool stream = bool_field(body, "stream", false) && st != nullptr;
    const bool include_usage = stream_wants_usage(body);
    if (stream && spec.logprobs)
        http::fail(400, "logprobs with stream=true is not supported in this build");
    const std::string id = make_id("cmpl-");
    const long long created = now_unix();
    auto frame_base = [&]() {
        json j;
        j["id"] = id;
        j["created"] = created;
        j["model"] = g_cfg.model;
        j["object"] = "text_completion";
        return j;
    };

    if (!stream) {
        GenOutcome o = run_generation(
            spec, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
        json j = frame_base();
        json choice{{"index", 0}, {"finish_reason", o.reason}, {"text", o.text}};
        if (spec.logprobs) choice["logprobs"] = logprobs_json(o.logprobs);
        j["choices"] = json::array({std::move(choice)});
        j["usage"] = usage_json(o, spec.ids);
        r->body = json_py::dumps(j, /*spaced=*/false);
        return;
    }

    r->sse = true;
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        json frame = frame_base();
        frame["choices"] = json::array(
            {json{{"index", 0}, {"finish_reason", nullptr}, {"text", ""}}});
        return send_frame(st, frame);
    };
    GenOutcome o = run_generation(
        spec,
        [&](const std::string& d) {
            json frame = frame_base();
            frame["choices"] = json::array(
                {json{{"index", 0}, {"finish_reason", nullptr}, {"text", d}}});
            return send_frame(st, frame);
        },
        id, on_wait);
    json fin = frame_base();
    fin["choices"] =
        json::array({json{{"index", 0}, {"finish_reason", o.reason}, {"text", ""}}});
    send_frame(st, fin);
    if (include_usage) {
        json u = frame_base();
        u["choices"] = json::array();
        u["usage"] = usage_json(o, spec.ids);
        send_frame(st, u);
    }
    // The terminator is a bare payload, not JSON: send_event() wraps it as
    // `data: [DONE]`. Passing it through a `const json&` helper would quote it
    // into `data: "[DONE]"`, which streaming clients reject at end-of-stream.
    st->send_event("[DONE]");
}

// Splits a reply at </think>: before it is reasoning_content, after it content.
void split_reasoning(const std::string& text, bool thinking_enabled,
                     std::string* reasoning, std::string* content) {
    if (!thinking_enabled) {
        reasoning->clear();
        *content = text;
        return;
    }
    struct Marker { const char* close; size_t length; };
    static constexpr Marker markers[] = {{"</think>", 8}, {"</thinking>", 11},
                                         {"</ think>", 9}, {"</think >", 9}};
    size_t at = std::string::npos;
    const Marker* hit = nullptr;
    for (const auto& marker : markers) {
        const size_t found = text.find(marker.close);
        if (found != std::string::npos && (at == std::string::npos || found < at)) {
            at = found;
            hit = &marker;
        }
    }
    if (at == std::string::npos) {
        *reasoning = text;
        content->clear();
        while (!reasoning->empty() && reasoning->back() == '\n') reasoning->pop_back();
        return;
    }
    *reasoning = text.substr(0, at);
    const size_t open = reasoning->find("<think>");
    const size_t open_alt = reasoning->find("<thinking>");
    if (open != std::string::npos) *reasoning = reasoning->substr(open + 7);
    else if (open_alt != std::string::npos) *reasoning = reasoning->substr(open_alt + 10);
    while (!reasoning->empty() && reasoning->back() == '\n') reasoning->pop_back();
    *content = text.substr(at + hit->length);
    while (!content->empty() && (*content)[0] == '\n') content->erase(0, 1);
}

// Incremental version of the same split. A naive diff of split_reasoning()
// over the growing text shrinks the reasoning string once the marker lands
// (and re-emits it as content), so this holds back a partial-marker tail and
// keeps two independent cursors.
class ThinkSplitter {
  public:
    explicit ThinkSplitter(bool thinking_enabled) {
        if (!thinking_enabled) {
            split_ = true;
            content_started_ = true;
        }
    }

    void feed(const std::string& piece, std::string* reasoning_out,
              std::string* content_out) {
        buf_ += piece;
        if (!split_) {
            size_t at = buf_.find(kMarker);
            size_t marker_len = kMarkerLen;
            const size_t alt = buf_.find("</thinking>");
            if (alt != std::string::npos && (at == std::string::npos || alt < at)) {
                at = alt;
                marker_len = 11;
            }
            if (at == std::string::npos) {
                // Hold back len(marker)-1 bytes in case the marker straddles
                // this piece and the next one, then snap that window to a
                // character boundary: a byte-count cut would split a
                // multi-byte character across two frames and make each frame
                // individually invalid UTF-8.
                // Hold back only a tail that is a real partial marker ("<",
                // "</th", ...), not a blind 7 bytes: the blind window kept
                // the first reasoning token(s) ("We", 2 bytes) off the wire
                // until the first speculative round landed (~65 ms TTFT).
                size_t hold = 0;
                for (size_t n = std::min(buf_.size(), kMarkerLen - 1); n > 0; --n)
                    if (buf_.compare(buf_.size() - n, n, kMarker, n) == 0) {
                        hold = n;
                        break;
                    }
                // trailing newlines may precede the marker (trimmed off the
                // reasoning once it lands): keep them too
                while (hold < buf_.size() && buf_[buf_.size() - 1 - hold] == '\n') ++hold;
                size_t safe = buf_.size() - hold;
                safe = utf8_boundary_at_or_before(buf_, std::max(safe, rpos_));
                if (safe > rpos_) {
                    *reasoning_out = buf_.substr(rpos_, safe - rpos_);
                    rpos_ = safe;
                }
                return;
            }
            split_ = true;
            r_end_ = at;
            while (r_end_ > 0 && buf_[r_end_ - 1] == '\n') --r_end_;
            cpos_ = at + marker_len;
        }
        if (rpos_ < r_end_) {
            *reasoning_out = buf_.substr(rpos_, r_end_ - rpos_);
            rpos_ = r_end_;
        }
        // Skip the newlines the template puts between </think> and the answer.
        // Only before the first content byte: afterwards a newline at the
        // cursor is real content ("```python\nprint(...") and must be emitted.
        // The marker can arrive before those bytes do, so this cannot be a
        // one-shot adjustment at split time -- it runs on every feed until
        // content starts, or the streamed text would keep a leading "\n\n"
        // that the non-streaming path strips.
        if (!content_started_) {
            while (cpos_ < buf_.size() && (buf_[cpos_] == '\n' || buf_[cpos_] == '\r')) ++cpos_;
            if (cpos_ < buf_.size()) content_started_ = true;
        }
        if (content_started_ && buf_.size() > cpos_) {
            *content_out = buf_.substr(cpos_);
            cpos_ = buf_.size();
        }
    }

    // Flush the tail when the stream ends without a marker.
    void finish(std::string* reasoning_out, std::string* content_out) {
        if (!split_) {
            size_t end = buf_.size();
            while (end > 0 && buf_[end - 1] == '\n') --end;
            if (end > rpos_) {
                *reasoning_out = buf_.substr(rpos_, end - rpos_);
                rpos_ = end;
            }
            return;
        }
        feed("", reasoning_out, content_out);
    }

  private:
    static constexpr const char* kMarker = "</think>";
    static constexpr size_t kMarkerLen = 8;
    std::string buf_;
    size_t rpos_ = 0, r_end_ = 0, cpos_ = 0;
    bool split_ = false, content_started_ = false;
};

// ----------------------------------------------------------- think guard ---
// The model sometimes ends its turn with nothing but reasoning: no answer, no
// tool call, finish_reason "stop". Clients then show a reasoning-only reply
// and an agent loop stalls on it. Three shapes, all only when the reply
// opened inside <think> and ended normally (not at max_tokens, not on a
// client stop string) with a blank answer:
//
//   A  the turn ended inside the unclosed think block. The server writes
//      Qwen's thinking-budget sentence and </think> into the stream (so the
//      client's text matches the model's context), then resumes on the same
//      slot from prompt + generated ids + that closer.
//   B  the model closed </think> and ended without an answer. The client
//      already has the </think>; the resume context swaps it for the budget
//      sentence + </think> (the bare context would replay the same end of
//      turn under greedy) and the answer streams on.
//   C  the reasoning ends with complete <tool_call> block(s) for declared
//      tools. They are handed back in GenOutcome::reasoning_calls and the
//      caller makes them as real calls; the text stays in the reasoning,
//      where it was already streamed. No resume.
//
// A resume that ends at once (blank, finish "stop") is retried with a fresh
// seed when sampling, at most twice; greedy would replay the same tokens, so
// it is not retried. When nothing comes back the reply stays as it was
// (guard = "resume_failed"). The resume holds the slot it ran on, so its
// prefix is still in that slot's KV and only the closer is prefilled.
// QWENOX_THINK_GUARD=0 turns all of this off.

struct ThinkGuardCtx {
    bool thinking = false;               // the reply opens inside <think>
    const ToolSetup* tools = nullptr;    // shape C needs the declared tools
    std::string tool_format = "xml";
};

bool think_guard_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("QWENOX_THINK_GUARD");
        return e == nullptr || std::strcmp(e, "0") != 0;
    }();
    return on;
}

bool is_blank_text(const std::string& s) {
    return s.find_first_not_of(" \t\r\n\f\v") == std::string::npos;
}

// Qwen's own sentence for closing a think block early (its thinking-budget
// recipe), so the model reads the cut as "answer now".
const char* const kThinkBudgetSentence =
    "Considering the limited time by the user, I have to give the solution based on "
    "the thinking directly now.";

// The complete trailing <tool_call> block(s) of `reasoning`, if they parse as
// calls of declared tools with nothing else in between; "" otherwise.
std::string trailing_reasoning_calls(const std::string& reasoning, const ThinkGuardCtx& g) {
    if (g.tools == nullptr || !g.tools->enabled()) return {};
    static const std::string kOpen = "<tool_call>", kClose = "</tool_call>";
    auto rtrim_len = [](const std::string& s, size_t n) {
        while (n > 0 && std::isspace(static_cast<unsigned char>(s[n - 1]))) --n;
        return n;
    };
    auto ends_with_close = [&](size_t n) {
        return n >= kClose.size() && reasoning.compare(n - kClose.size(), kClose.size(), kClose) == 0;
    };
    const size_t end = rtrim_len(reasoning, reasoning.size());
    if (!ends_with_close(end)) return {};
    size_t begin = reasoning.rfind(kOpen, end - kClose.size());
    if (begin == std::string::npos) return {};
    for (;;) {  // take in parallel calls written back to back
        const size_t before = rtrim_len(reasoning, begin);
        if (!ends_with_close(before)) break;
        const size_t prev = reasoning.rfind(kOpen, before - kClose.size());
        if (prev == std::string::npos) break;
        begin = prev;
    }
    std::string block = reasoning.substr(begin, end - begin);
    toolparse::StreamParser parser(g.tools->tools, nullptr, true, g.tool_format, "");
    parser.feed(block);
    parser.finish();
    if (parser.calls().empty() || parser.has_partial_call() || !is_blank_text(parser.content()))
        return {};
    return block;
}

GenOutcome run_generation_guarded(GenSpec& spec, const ThinkGuardCtx& g,
                                  const std::function<bool(const std::string&)>& on_delta,
                                  const std::string& log_tag,
                                  const std::function<bool()>& on_wait = {}) {
    const int test_cut = spec.think_guard_test_cut;
    if (!g.thinking || !think_guard_enabled() || !g_tok_ready) {
        return run_generation(spec, on_delta, log_tag, on_wait);
    }
    SlotGuard slot(on_wait);
    if (!slot.acquired()) {
        GenOutcome gone;
        gone.client_gone = true;
        return gone;
    }

    const int max_tokens = spec.max_tokens;
    if (test_cut > 0) spec.max_tokens = test_cut;
    GenOutcome out = run_generation(spec, on_delta, log_tag, on_wait, &slot);
    spec.max_tokens = max_tokens;
    if (test_cut > 0 && out.reason == "length" && !out.client_gone &&
        (max_tokens <= 0 || test_cut < max_tokens)) {
        out.reason = "stop";  // test hook: treat the cut as the model's own end of turn
        fprintf(stderr, "REQ %s think guard test: first run cut at %d tokens\n",
                log_tag.c_str(), test_cut);
    }
    if (out.client_gone || out.reason != "stop" || out.hit_stop) return out;

    std::string reasoning, answer;
    split_reasoning(out.text, true, &reasoning, &answer);
    if (!is_blank_text(answer)) return out;

    // C: a finished tool call left inside the reasoning.
    std::string calls = trailing_reasoning_calls(reasoning, g);
    if (!calls.empty()) {
        out.guard = "tool_call_from_reasoning";
        out.reasoning_calls = std::move(calls);
        fprintf(stderr, "REQ %s think guard: tool call written inside the reasoning, "
                        "made as a call\n", log_tag.c_str());
        return out;
    }

    // A / B: build the resume context.
    static const int think_close_id = g_tok.added_token_id("</think>");
    size_t marker = std::string::npos;
    for (const char* m : {"</think>", "</thinking>", "</ think>", "</think >"}) {
        const size_t at = out.text.find(m);
        if (at != std::string::npos && (marker == std::string::npos || at < marker)) marker = at;
    }
    const bool unclosed = marker == std::string::npos;
    std::vector<int> context = out.tokens;
    std::string closer;
    if (unclosed) {
        const size_t nl = out.text.size() - out.text.find_last_not_of('\n') - 1;
        closer = out.text.empty() ? "" : std::string(nl >= 2 ? 0 : 2 - nl, '\n');
        closer += kThinkBudgetSentence;
        closer += "\n</think>\n\n";
    } else {
        // The close must be the model's own </think> token to cut the
        // context there; a spelled-out variant is left as it is.
        const auto it = std::find(context.rbegin(), context.rend(), think_close_id);
        if (think_close_id < 0 || it == context.rend()) return out;
        context.erase(std::next(it).base(), context.end());
        const std::string before = out.text.substr(0, marker);
        const size_t nl = before.size() - before.find_last_not_of('\n') - 1;
        closer = before.empty() ? "" : std::string(nl >= 2 ? 0 : 2 - nl, '\n');
        closer += kThinkBudgetSentence;
        closer += "\n</think>\n\n";
    }
    const std::vector<int> closer_ids = g_tok.encode(closer);
    context.insert(context.end(), closer_ids.begin(), closer_ids.end());
    // The resume needs room for the context and at least a short answer;
    // without it the engine would refuse the resume mid-stream.
    if (static_cast<long long>(spec.ids.size() + context.size()) + 16 >
        static_cast<long long>(g_cfg.context)) {
        fprintf(stderr, "REQ %s think guard: empty answer, but no context left to resume\n",
                log_tag.c_str());
        return out;
    }
    if (unclosed) {
        if (on_delta && !on_delta(closer)) {
            out.client_gone = true;
            return out;
        }
        out.text += closer;
    }
    fprintf(stderr, "REQ %s think guard: reply ended %s with no answer, resuming "
                    "(%zu generated + %zu closer tokens)\n",
            log_tag.c_str(), unclosed ? "inside <think>" : "right after </think>",
            out.tokens.size(), closer_ids.size());

    // Resume on the same slot. spec is borrowed and restored: callers read
    // spec.ids (prompt size) afterwards.
    std::vector<int> prompt_ids = spec.ids;
    spec.ids.insert(spec.ids.end(), context.begin(), context.end());
    std::vector<long long> snaps = std::move(spec.snaps);
    spec.snaps.clear();  // the hints index the original prompt only
    const unsigned long long seed = spec.seed;
    if (max_tokens > 0) {
        const long long left = max_tokens - out.n_gen;
        spec.max_tokens = static_cast<int>(std::max<long long>(left, 1));
    }
    const int attempts = spec.sample ? 3 : 1;
    bool answered = false;
    int resume_cached = 0;
    long long resume_prompt = 0;
    for (int attempt = 1; attempt <= attempts && !answered; ++attempt) {
        if (attempt > 1) {
            // splitmix64 step: same seed => the same early end of turn
            unsigned long long z = (spec.seed += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            spec.seed = z ^ (z >> 31);
        }
        GenOutcome rest = run_generation(spec, on_delta, log_tag, on_wait, &slot);
        resume_cached = rest.n_cached;
        resume_prompt = rest.n_prompt;
        out.text += rest.text;
        out.n_gen += rest.n_gen;
        out.prefill_ms += rest.prefill_ms;
        out.decode_ms += rest.decode_ms;
        out.rounds += rest.rounds;
        out.commit += rest.commit;
        out.proposed += rest.proposed;
        out.logprobs.insert(out.logprobs.end(), rest.logprobs.begin(), rest.logprobs.end());
        out.reason = rest.reason;
        out.hit_stop = rest.hit_stop;
        out.stop_hit = rest.stop_hit;
        if (rest.client_gone) {
            out.client_gone = true;
            break;
        }
        answered = !is_blank_text(rest.text) || rest.reason != "stop" || rest.hit_stop;
        if (!answered)
            fprintf(stderr, "REQ %s think guard: resume %d/%d came back empty\n",
                    log_tag.c_str(), attempt, attempts);
    }
    spec.ids = std::move(prompt_ids);
    spec.snaps = std::move(snaps);
    spec.max_tokens = max_tokens;
    spec.seed = seed;
    out.guard = answered ? (unclosed ? "resumed_unclosed" : "resumed_empty_answer")
                         : "resume_failed";
    fprintf(stderr, "REQ %s think guard: %s (resume prompt=%lld cached=%d)\n", log_tag.c_str(),
            out.guard.c_str(), resume_prompt, resume_cached);
    return out;
}

// POST /v1/chat/completions
void handle_chat(const http::Request& q, http::Response* r, http::Stream* st) {
    if (!g_tok_ready)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    json mutable_body = parse_body(q);
    apply_overrides(&mutable_body, Api::Chat, r);
    const json& body = mutable_body;
    if (!body.contains("messages")) http::fail(400, "No messages provided.");
    if (!body["messages"].is_array() || body["messages"].empty())
        http::fail(400, "No messages provided.");
    json messages = body["messages"];
    std::string normalize_error;
    if (!toolparse::normalize_messages(&messages, &normalize_error))
        http::fail(400, normalize_error);
    std::vector<vision::Frame> frames;
    if (!vision::prepare_messages(messages, &frames, &normalize_error))
        http::fail(400, normalize_error);

    ToolSetup tool_setup = parse_tool_setup(body);
    chat_template::Options opts;
    if (tool_setup.enabled()) opts.tools = &tool_setup.tools;
    static const json kTrue = true;
    opts.add_generation_prompt = &kTrue;
    json normalized_effort;
    if (body.contains("reasoning_effort"))
        opts.reasoning_effort =
            normalize_reasoning_effort(&body["reasoning_effort"], &normalized_effort);
    require_optional_boolean(body, "enable_thinking");
    require_optional_boolean(body, "preserve_thinking");
    require_optional_boolean(body, "preserve_reasoning");
    require_optional_boolean(body, "auto_disable_thinking_with_tools");
    require_optional_boolean(body, "add_vision_id");
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null())
        opts.enable_thinking = &body["enable_thinking"];
    if (body.contains("preserve_thinking") && !body["preserve_thinking"].is_null())
        opts.preserve_thinking = &body["preserve_thinking"];
    if (body.contains("preserve_reasoning") && !body["preserve_reasoning"].is_null())
        opts.preserve_reasoning = &body["preserve_reasoning"];
    if (body.contains("auto_disable_thinking_with_tools"))
        opts.auto_disable_thinking_with_tools = &body["auto_disable_thinking_with_tools"];
    std::string tool_format = "xml";
    if (body.contains("tool_call_format")) {
        if (!body["tool_call_format"].is_string() ||
            (body["tool_call_format"] != "xml" && body["tool_call_format"] != "json"))
            http::fail(400, "tool_call_format must be 'xml' or 'json'");
        tool_format = body["tool_call_format"].get<std::string>();
        opts.tool_call_format = &body["tool_call_format"];
    }
    if (body.contains("max_tool_arg_chars")) {
        if (!body["max_tool_arg_chars"].is_number_integer() &&
            !body["max_tool_arg_chars"].is_number_unsigned())
            http::fail(400, "max_tool_arg_chars must be a non-negative integer");
        opts.max_tool_arg_chars = &body["max_tool_arg_chars"];
    }
    if (body.contains("max_tool_response_chars")) {
        if (!body["max_tool_response_chars"].is_number_integer() &&
            !body["max_tool_response_chars"].is_number_unsigned())
            http::fail(400, "max_tool_response_chars must be a non-negative integer");
        opts.max_tool_response_chars = &body["max_tool_response_chars"];
    }
    if (body.contains("add_vision_id")) opts.add_vision_id = &body["add_vision_id"];

    const auto prep_t0 = std::chrono::steady_clock::now();
    chat_template::RenderResult rr = chat_template::render_chat_template(&messages, opts);
    if (!rr.ok) http::fail(400, rr.error);
    rr.text += tool_setup.choice.prompt_suffix(rr.thinking_enabled, tool_format);
    const auto prep_t1 = std::chrono::steady_clock::now();

    GenSpec spec;
    std::vector<size_t> starts;
    if (!encode_spliced(rr.text, &frames, &spec.ids, &starts, &normalize_error))
        http::fail(400, normalize_error);
    const auto prep_t2 = std::chrono::steady_clock::now();
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    spec.snaps = compute_snap_cuts(messages, opts, rr.text, spec.ids, starts);
    // T2 (agent 短增量专项)：API 侧固定开销拆分。引擎层截距 ~159 ms vs 真实日志
    // ~345 ms，差值在这里——每请求一行，格式对齐 tcache 日志。
    {
        const auto prep_t3 = std::chrono::steady_clock::now();
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr,
                "qwenox-api: prep msgs=%zu render=%.1fms encode=%.1fms cuts=%.1fms"
                " (n=%zu) ids=%zu\n",
                messages.size(), ms(prep_t0, prep_t1), ms(prep_t1, prep_t2),
                ms(prep_t2, prep_t3), spec.snaps.size(), spec.ids.size());
    }
    apply_sampling(body, &spec);
    ban_tool_call_without_tools(tool_setup, &spec);

    const bool stream = bool_field(body, "stream", false) && st != nullptr;
    const bool include_usage = stream_wants_usage(body);
    if (stream && spec.logprobs)
        http::fail(400, "logprobs with stream=true is not supported in this build");
    const std::string id = make_id("chatcmpl-");
    const long long created = now_unix();
    auto chunk = [&](json delta, const char* finish) {
        json j;
        j["id"] = id;
        j["created"] = created;
        j["model"] = g_cfg.model;
        j["object"] = "chat.completion.chunk";
        j["choices"] = json::array(
            {json{{"index", 0},
                  {"finish_reason", finish != nullptr ? json(finish) : json(nullptr)},
                  {"delta", delta}}});
        return j;
    };
    auto text_chunk = [&](const char* field, const std::string& value) {
        json delta = json::object();
        delta[field] = value;
        return chunk(std::move(delta), nullptr);
    };

    const ThinkGuardCtx guard{rr.thinking_enabled && !tool_setup.choice.forced(), &tool_setup,
                              tool_format};
    if (!stream) {
        GenOutcome o = run_generation_guarded(
            spec, guard, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
        std::string reasoning, answer;
        if (tool_setup.choice.forced())
            answer = o.text;
        else
            split_reasoning(o.text, rr.thinking_enabled, &reasoning, &answer);
        toolparse::StreamParser parser(
            tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled(), tool_format,
        tool_setup.choice.name);
        parser.feed(tool_setup.choice.parser_prefix(tool_format));
        parser.feed(answer);
        parser.feed(o.reasoning_calls);
        parser.finish();
        json msg;
        msg["role"] = "assistant";
        msg["content"] = parser.content();
        msg["reasoning_content"] = reasoning;
        if (!parser.calls().empty()) {
            msg["tool_calls"] = json::array();
            for (const auto& call : parser.calls()) {
                msg["tool_calls"].push_back(
                    json{{"id", call.id},
                         {"type", "function"},
                         {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
            }
        }
        json j;
        j["id"] = id;
        j["created"] = created;
        j["model"] = g_cfg.model;
        j["object"] = "chat.completion";
        const std::string finish = (o.reason == "length" || parser.has_partial_call())
                                       ? "length"
                                       : (parser.calls().empty() ? o.reason : "tool_calls");
        json choice{{"index", 0}, {"finish_reason", finish}, {"message", msg}};
        if (spec.logprobs) choice["logprobs"] = logprobs_json(o.logprobs);
        j["choices"] = json::array({std::move(choice)});
        j["usage"] = usage_json(o, spec.ids);
        r->body = json_py::dumps(j, /*spaced=*/false);
        return;
    }

    r->sse = true;
    ThinkSplitter splitter(rr.thinking_enabled);
    toolparse::StreamParser parser(
        tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled(), tool_format,
        tool_setup.choice.name);
    auto dispatch_tool_events = [&](const std::vector<toolparse::Event>& events) {
        for (const auto& event : events) {
            if (event.type == toolparse::EventType::Content) {
                if (!send_frame(st, text_chunk("content", event.data))) return false;
            } else if (event.type == toolparse::EventType::CallStart) {
                json tc;
                tc["index"] = event.index;
                tc["id"] = event.id;
                tc["type"] = "function";
                tc["function"] = {{"name", event.name}, {"arguments", ""}};
                json delta = json::object();
                delta["tool_calls"] = json::array({std::move(tc)});
                if (!send_frame(st, chunk(std::move(delta), nullptr))) return false;
            } else if (event.type == toolparse::EventType::ArgumentsDelta) {
                json tc;
                tc["index"] = event.index;
                tc["function"] = {{"arguments", event.data}};
                json delta = json::object();
                delta["tool_calls"] = json::array({std::move(tc)});
                if (!send_frame(st, chunk(std::move(delta), nullptr))) return false;
            }
        }
        return true;
    };
    if (!dispatch_tool_events(parser.feed(tool_setup.choice.parser_prefix(tool_format)))) return;
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        // 协议中性 keep-alive（与 /v1/responses 相同）：空 content 帧会被转换层
        // 当成"正文开始"，把 thinking 截成 text 在前、reasoning 在后的非法序列。
        return st->send_raw(": keep-alive\n\n");
    };
    GenOutcome o = run_generation_guarded(
        spec, guard,
        [&](const std::string& d) {
            if (st->disconnected()) return false;
            if (tool_setup.choice.forced()) return dispatch_tool_events(parser.feed(d));
            std::string rp, ct;
            splitter.feed(d, &rp, &ct);
            if (!rp.empty() && !send_frame(st, text_chunk("reasoning_content", rp)))
                return false;
            return dispatch_tool_events(parser.feed(ct));
        },
        id, on_wait);
    if (o.client_gone) return;
    if (!tool_setup.choice.forced()) {
        std::string rp, ct;
        splitter.finish(&rp, &ct);
        if (!rp.empty() && !send_frame(st, text_chunk("reasoning_content", rp))) return;
        if (!dispatch_tool_events(parser.feed(ct))) return;
        if (!dispatch_tool_events(parser.feed(o.reasoning_calls))) return;
    }
    if (!dispatch_tool_events(parser.finish())) return;
    const std::string finish = (o.reason == "length" || parser.has_partial_call())
                                   ? "length"
                                   : (parser.calls().empty() ? o.reason : "tool_calls");
    send_frame(st, chunk(json::object(), finish.c_str()));
    if (include_usage) {
        json u;
        u["id"] = id;
        u["created"] = created;
        u["model"] = g_cfg.model;
        u["object"] = "chat.completion.chunk";
        u["choices"] = json::array();
        u["usage"] = usage_json(o, spec.ids);
        send_frame(st, u);
    }
    // The terminator is a bare payload, not JSON: send_event() wraps it as
    // `data: [DONE]`. Passing it through a `const json&` helper would quote it
    // into `data: "[DONE]"`, which streaming clients reject at end-of-stream.
    st->send_event("[DONE]");
}

// POST /v1/responses -- stateless text and function-call subset.
void handle_responses(const http::Request& q, http::Response* r, http::Stream* st) {
    if (!g_tok_ready)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    json mutable_body = parse_body(q);
    apply_overrides(&mutable_body, Api::Responses, r);
    const json& body = mutable_body;
    json messages = normalize_responses_input(body);
    std::vector<vision::Frame> frames;
    std::string vision_error;
    if (!vision::prepare_messages(messages, &frames, &vision_error))
        http::fail(400, vision_error);
    const bool stream = bool_field(body, "stream", false) && st != nullptr;
    const bool parallel_tool_calls = bool_field(body, "parallel_tool_calls", true);
    require_optional_boolean(body, "enable_thinking");
    require_optional_boolean(body, "preserve_thinking");

    ToolSetup tool_setup = parse_tool_setup(body);
    chat_template::Options opts;
    static const json kTrue = true;
    opts.add_generation_prompt = &kTrue;
    if (tool_setup.enabled()) opts.tools = &tool_setup.tools;
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null())
        opts.enable_thinking = &body["enable_thinking"];
    if (body.contains("preserve_thinking") && !body["preserve_thinking"].is_null())
        opts.preserve_thinking = &body["preserve_thinking"];
    require_optional_boolean(body, "preserve_reasoning");
    require_optional_boolean(body, "auto_disable_thinking_with_tools");
    if (body.contains("preserve_reasoning") && !body["preserve_reasoning"].is_null())
        opts.preserve_reasoning = &body["preserve_reasoning"];
    if (body.contains("auto_disable_thinking_with_tools"))
        opts.auto_disable_thinking_with_tools = &body["auto_disable_thinking_with_tools"];
    std::string tool_format = "xml";
    if (body.contains("tool_call_format")) {
        if (!body["tool_call_format"].is_string() ||
            (body["tool_call_format"] != "xml" && body["tool_call_format"] != "json"))
            http::fail(400, "tool_call_format must be 'xml' or 'json'");
        tool_format = body["tool_call_format"].get<std::string>();
        opts.tool_call_format = &body["tool_call_format"];
    }
    if (body.contains("max_tool_arg_chars")) {
        if (!body["max_tool_arg_chars"].is_number_integer() &&
            !body["max_tool_arg_chars"].is_number_unsigned())
            http::fail(400, "max_tool_arg_chars must be a non-negative integer");
        opts.max_tool_arg_chars = &body["max_tool_arg_chars"];
    }
    if (body.contains("max_tool_response_chars")) {
        if (!body["max_tool_response_chars"].is_number_integer() &&
            !body["max_tool_response_chars"].is_number_unsigned())
            http::fail(400, "max_tool_response_chars must be a non-negative integer");
        opts.max_tool_response_chars = &body["max_tool_response_chars"];
    }

    json normalized_effort;
    if (body.contains("reasoning") && !body["reasoning"].is_null()) {
        if (!body["reasoning"].is_object()) http::fail(400, "reasoning must be an object");
        const json& reasoning = body["reasoning"];
        if (reasoning.contains("effort"))
            opts.reasoning_effort =
                normalize_reasoning_effort(&reasoning["effort"], &normalized_effort);
        for (const char* key : {"summary", "generate_summary"}) {
            if (reasoning.contains(key) && !reasoning[key].is_null() &&
                !reasoning[key].is_string())
                http::fail(400, std::string("reasoning.") + key + " must be a string");
        }
    }

    chat_template::RenderResult rr = chat_template::render_chat_template(&messages, opts);
    if (!rr.ok) http::fail(400, rr.error);
    rr.text += tool_setup.choice.prompt_suffix(rr.thinking_enabled, tool_format);

    GenSpec spec;
    std::vector<size_t> starts;
    if (!encode_spliced(rr.text, &frames, &spec.ids, &starts, &vision_error))
        http::fail(400, vision_error);
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    spec.snaps = compute_snap_cuts(messages, opts, rr.text, spec.ids, starts);
    apply_sampling(body, &spec);
    ban_tool_call_without_tools(tool_setup, &spec);

    const std::string id = make_id("resp_");
    const long long created = now_unix();
    auto output_part = [](const std::string& text) {
        return json{{"type", "output_text"},
                    {"text", text},
                    {"annotations", json::array()},
                    {"logprobs", json::array()}};
    };
    auto message_item = [&](const std::string& item_id, const std::string& status,
                            const std::string& text) {
        return json{{"id", item_id},
                    {"type", "message"},
                    {"status", status},
                    {"role", "assistant"},
                    {"content", json::array({output_part(text)})}};
    };
    auto function_item = [](const std::string& item_id, const std::string& call_id,
                            const std::string& status, const std::string& name,
                            const std::string& arguments) {
        return json{{"id", item_id},
                    {"type", "function_call"},
                    {"status", status},
                    {"call_id", call_id},
                    {"name", name},
                    {"arguments", arguments}};
    };
    auto response_object = [&](const std::string& status, const json& output,
                               const GenOutcome* outcome) {
        json response;
        response["id"] = id;
        response["object"] = "response";
        response["created_at"] = created;
        response["status"] = status;
        response["model"] = g_cfg.model;
        response["output"] = output;
        response["parallel_tool_calls"] = parallel_tool_calls;
        response["max_output_tokens"] =
            spec.max_tokens > 0 ? json(spec.max_tokens) : json(nullptr);
        if (outcome == nullptr) {
            response["usage"] = nullptr;
        } else {
            const long long pt = outcome->n_prompt > 0
                                     ? outcome->n_prompt
                                     : static_cast<long long>(spec.ids.size());
            response["usage"] = {{"input_tokens", pt},
                                 {"output_tokens", outcome->n_gen},
                                 {"total_tokens", pt + outcome->n_gen}};
            if (!outcome->guard.empty()) response["usage"]["think_guard"] = outcome->guard;
            if (status == "incomplete")
                response["incomplete_details"] = {
                    {"reason", outcome->reason == "length" ? "max_output_tokens"
                                                              : "invalid_tool_call"}};
        }
        return response;
    };

    const ThinkGuardCtx guard{rr.thinking_enabled && !tool_setup.choice.forced(), &tool_setup,
                              tool_format};
    if (!stream) {
        GenOutcome o = run_generation_guarded(
            spec, guard, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
        std::string reasoning, answer;
        if (tool_setup.choice.forced())
            answer = o.text;
        else
            split_reasoning(o.text, rr.thinking_enabled, &reasoning, &answer);
        toolparse::StreamParser parser(
            tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled(), tool_format,
        tool_setup.choice.name);
        parser.feed(tool_setup.choice.parser_prefix(tool_format));
        parser.feed(answer);
        parser.feed(o.reasoning_calls);
        parser.finish();

        const bool incomplete = o.reason == "length" || parser.has_partial_call();
        json output = json::array();
        if (!reasoning.empty())
            output.push_back(
                json{{"id", make_id("rs_")},
                     {"type", "reasoning"},
                     {"status", "completed"},
                     {"summary", json::array({json{{"type", "summary_text"},
                                                   {"text", reasoning}}})}});
        if (!parser.content().empty() || (!incomplete && parser.calls().empty())) {
            output.push_back(message_item(make_id("msg_"),
                                          incomplete ? "incomplete" : "completed",
                                          parser.content()));
        }
        for (const auto& call : parser.calls()) {
            output.push_back(function_item(make_id("fc_"), call.id, "completed",
                                           call.name, call.arguments));
        }
        json response = response_object(incomplete ? "incomplete" : "completed", output, &o);
        r->body = json_py::dumps(response, /*spaced=*/false);
        return;
    }

    r->sse = true;
    long long sequence = 0;
    bool created_sent = false;
    int next_output_index = 0;
    json output_slots = json::array();
    auto put_output = [&](int index, json item) {
        while (output_slots.size() <= static_cast<size_t>(index))
            output_slots.push_back(nullptr);
        output_slots[index] = std::move(item);
    };
    auto event = [&](const char* type, json fields) {
        json e;
        e["type"] = type;
        e["sequence_number"] = sequence++;
        for (auto it = fields.begin(); it != fields.end(); ++it) e[it.key()] = it.value();
        return send_frame(st, e);
    };
    auto ensure_created = [&]() {
        if (created_sent) return true;
        created_sent = true;
        return event("response.created",
                     json{{"response", response_object("in_progress", json::array(), nullptr)}});
    };

    // reasoning 输出项（summary 单 part）：先于 message/function_call 打开，
    // 首个正文/工具事件前关闭。对应 chat/completions 的 reasoning_content。
    struct ReasoningState {
        bool open = false;
        int output_index = -1;
        std::string id;
        std::string summary;
    } rstate;
    auto reasoning_item = [&](const char* status) {
        return json{{"id", rstate.id},
                    {"type", "reasoning"},
                    {"status", status},
                    {"summary", rstate.summary.empty()
                                    ? json::array()
                                    : json::array({json{{"type", "summary_text"},
                                                        {"text", rstate.summary}}})}};
    };
    auto start_reasoning = [&]() {
        if (rstate.open) return true;
        if (!ensure_created()) return false;
        rstate.open = true;
        rstate.output_index = next_output_index++;
        rstate.id = make_id("rs_");
        rstate.summary.clear();
        if (!event("response.output_item.added",
                   json{{"output_index", rstate.output_index},
                        {"item", reasoning_item("in_progress")}}))
            return false;
        return event("response.reasoning_summary_part.added",
                     json{{"item_id", rstate.id},
                          {"output_index", rstate.output_index},
                          {"summary_index", 0},
                          {"part", json{{"type", "summary_text"}, {"text", ""}}}});
    };
    auto finish_reasoning = [&]() {
        if (!rstate.open) return true;
        if (!event("response.reasoning_summary_text.done",
                   json{{"item_id", rstate.id},
                        {"output_index", rstate.output_index},
                        {"summary_index", 0},
                        {"text", rstate.summary}}))
            return false;
        if (!event("response.reasoning_summary_part.done",
                   json{{"item_id", rstate.id},
                        {"output_index", rstate.output_index},
                        {"summary_index", 0},
                        {"part", json{{"type", "summary_text"},
                                      {"text", rstate.summary}}}}))
            return false;
        json item = reasoning_item("completed");
        if (!event("response.output_item.done",
                   json{{"output_index", rstate.output_index}, {"item", item}}))
            return false;
        put_output(rstate.output_index, std::move(item));
        rstate.open = false;
        return true;
    };
    auto emit_reasoning = [&](const std::string& delta) {
        if (delta.empty()) return true;
        if (!start_reasoning()) return false;
        rstate.summary += delta;
        return event("response.reasoning_summary_text.delta",
                     json{{"item_id", rstate.id},
                          {"output_index", rstate.output_index},
                          {"summary_index", 0},
                          {"delta", delta}});
    };

    struct MessageState {
        bool open = false;
        int output_index = -1;
        std::string id;
        std::string content;
    } message;
    auto start_message = [&]() {
        if (message.open) return true;
        if (!finish_reasoning()) return false;
        if (!ensure_created()) return false;
        message.open = true;
        message.output_index = next_output_index++;
        message.id = make_id("msg_");
        message.content.clear();
        if (!event("response.output_item.added",
                   json{{"output_index", message.output_index},
                        {"item", message_item(message.id, "in_progress", "")}}))
            return false;
        return event("response.content_part.added",
                     json{{"item_id", message.id},
                          {"output_index", message.output_index},
                          {"content_index", 0},
                          {"part", output_part("")}});
    };
    auto finish_message = [&]() {
        if (!message.open) return true;
        if (!event("response.output_text.done",
                   json{{"item_id", message.id},
                        {"output_index", message.output_index},
                        {"content_index", 0},
                        {"text", message.content},
                        {"logprobs", json::array()}}))
            return false;
        if (!event("response.content_part.done",
                   json{{"item_id", message.id},
                        {"output_index", message.output_index},
                        {"content_index", 0},
                        {"part", output_part(message.content)}}))
            return false;
        json item = message_item(message.id, "completed", message.content);
        if (!event("response.output_item.done",
                   json{{"output_index", message.output_index}, {"item", item}}))
            return false;
        put_output(message.output_index, std::move(item));
        message.open = false;
        return true;
    };
    auto emit_text = [&](const std::string& delta) {
        if (delta.empty()) return true;
        if (!start_message()) return false;
        message.content += delta;
        return event("response.output_text.delta",
                     json{{"item_id", message.id},
                          {"output_index", message.output_index},
                          {"content_index", 0},
                          {"delta", delta},
                          {"logprobs", json::array()}});
    };

    struct CallState {
        bool open = false;
        int output_index = -1;
        std::string item_id;
        std::string call_id;
        std::string name;
        std::string arguments;
    };
    std::vector<CallState> call_states;
    auto dispatch_tool_events = [&](const std::vector<toolparse::Event>& events) {
        for (const auto& parsed : events) {
            if (parsed.type == toolparse::EventType::Content) {
                if (!emit_text(parsed.data)) return false;
                continue;
            }
            if (parsed.index >= call_states.size()) call_states.resize(parsed.index + 1);
            CallState& call = call_states[parsed.index];
            if (parsed.type == toolparse::EventType::CallStart) {
                if (!finish_reasoning() || !finish_message() || !ensure_created()) return false;
                call.open = true;
                call.output_index = next_output_index++;
                call.item_id = make_id("fc_");
                call.call_id = parsed.id;
                call.name = parsed.name;
                json item = function_item(call.item_id, call.call_id, "in_progress",
                                          call.name, "");
                if (!event("response.output_item.added",
                           json{{"output_index", call.output_index}, {"item", item}}))
                    return false;
            } else if (parsed.type == toolparse::EventType::ArgumentsDelta) {
                call.arguments += parsed.data;
                if (!event("response.function_call_arguments.delta",
                           json{{"item_id", call.item_id},
                                {"output_index", call.output_index},
                                {"delta", parsed.data}}))
                    return false;
            } else if (parsed.type == toolparse::EventType::CallEnd) {
                call.arguments = parsed.data;
                if (!event("response.function_call_arguments.done",
                           json{{"item_id", call.item_id},
                                {"output_index", call.output_index},
                                {"arguments", call.arguments},
                                {"name", call.name}}))
                    return false;
                json item = function_item(call.item_id, call.call_id, "completed",
                                          call.name, call.arguments);
                if (!event("response.output_item.done",
                           json{{"output_index", call.output_index}, {"item", item}}))
                    return false;
                put_output(call.output_index, std::move(item));
                call.open = false;
            }
        }
        return true;
    };

    ThinkSplitter splitter(rr.thinking_enabled);
    toolparse::StreamParser parser(
        tool_setup.tools, []() { return make_id("call_"); }, tool_setup.enabled(), tool_format,
        tool_setup.choice.name);
    if (!dispatch_tool_events(parser.feed(tool_setup.choice.parser_prefix(tool_format)))) return;
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        // No response item exists until generation starts, so use a standard
        // SSE comment as a protocol-neutral keep-alive during queue/prefill.
        return st->send_raw(": keep-alive\n\n");
    };
    GenOutcome o = run_generation_guarded(
        spec, guard,
        [&](const std::string& delta) {
            if (st->disconnected()) return false;
            if (tool_setup.choice.forced()) return dispatch_tool_events(parser.feed(delta));
            std::string reasoning_delta, content_delta;
            splitter.feed(delta, &reasoning_delta, &content_delta);
            if (!emit_reasoning(reasoning_delta)) return false;
            return dispatch_tool_events(parser.feed(content_delta));
        },
        id, on_wait);
    if (o.client_gone) return;
    if (!tool_setup.choice.forced()) {
        std::string reasoning_delta, content_delta;
        splitter.finish(&reasoning_delta, &content_delta);
        if (!emit_reasoning(reasoning_delta)) return;
        if (!dispatch_tool_events(parser.feed(content_delta))) return;
        if (!dispatch_tool_events(parser.feed(o.reasoning_calls))) return;
    }
    if (!dispatch_tool_events(parser.finish())) return;
    if (!finish_reasoning() || !finish_message() || !ensure_created()) return;

    bool partial_call = parser.has_partial_call();
    for (CallState& call : call_states) {
        if (!call.open) continue;
        partial_call = true;
        json item = function_item(call.item_id, call.call_id, "incomplete",
                                  call.name, call.arguments);
        if (!event("response.output_item.done",
                   json{{"output_index", call.output_index}, {"item", item}}))
            return;
        put_output(call.output_index, std::move(item));
        call.open = false;
    }

    const bool incomplete = o.reason == "length" || partial_call;
    if (output_slots.empty() && !incomplete) {
        if (!start_message() || !finish_message()) return;
    }
    json final_response =
        response_object(incomplete ? "incomplete" : "completed", output_slots, &o);
    event(incomplete ? "response.incomplete" : "response.completed",
          json{{"response", std::move(final_response)}});
}

// ------------------------------------------------- Anthropic Messages API --
// POST /v1/messages and /v1/messages/count_tokens. The Anthropic request is
// translated into the chat-completions body shape and then runs through the
// same override / template / tool-parser path as /v1/chat/completions; only
// the wire format differs: content blocks in, content blocks and named SSE
// events out, errors in Anthropic's {"type":"error",...} envelope.
// x-api-key / anthropic-version / anthropic-beta headers are accepted and
// ignored (the inference endpoints carry no auth, same as the OpenAI ones).

const char* anthropic_error_type(int status) {
    switch (status) {
        case 400: return "invalid_request_error";
        case 401: return "authentication_error";
        case 403: return "permission_error";
        case 404: return "not_found_error";
        case 413: return "request_too_large";
        case 429: return "rate_limit_error";
        case 503: return "overloaded_error";
        default: return status >= 500 ? "api_error" : "invalid_request_error";
    }
}

json anthropic_error_json(int status, const std::string& message) {
    return json{{"type", "error"},
                {"error", {{"type", anthropic_error_type(status)}, {"message", message}}}};
}

// One named SSE event: `event: <type>\ndata: <json>\n\n`.
bool send_anthropic_event(http::Stream* st, const char* type, const json& data) {
    return st->send_raw(std::string("event: ") + type + "\ndata: " +
                        json_py::dumps(data, /*spaced=*/false) + "\n\n");
}

std::string block_type(const json& block) {
    if (!block.is_object()) http::fail(400, "content blocks must be objects");
    if (!block.contains("type") || !block["type"].is_string())
        http::fail(400, "content blocks require a string type");
    return block["type"].get<std::string>();
}

const std::string& block_string(const json& block, const char* key, const char* type) {
    if (!block.contains(key) || !block[key].is_string())
        http::fail(400, std::string(type) + " blocks require a string " + key);
    return block[key].get_ref<const std::string&>();
}

// Text of a string or an array of text blocks (system prompt), joined by `sep`.
std::string anthropic_text(const json& content, const char* field, const char* sep) {
    if (content.is_null()) return {};
    if (content.is_string()) return content.get<std::string>();
    if (!content.is_array())
        http::fail(400, std::string(field) + " must be a string or an array of text blocks");
    std::string text;
    for (const auto& block : content) {
        if (block_type(block) != "text")
            http::fail(400, std::string(field) + " must contain only text blocks");
        if (!text.empty()) text += sep;
        text += block_string(block, "text", "text");
    }
    return text;
}

// Anthropic image source -> the image_url part the vision path decodes. URL
// sources are passed through so vision:: rejects them with its usual message.
json anthropic_image(const json& block) {
    if (!block.contains("source") || !block["source"].is_object())
        http::fail(400, "image blocks require a source object");
    const json& source = block["source"];
    const std::string type =
        source.contains("type") && source["type"].is_string() ? source["type"].get<std::string>()
                                                              : std::string();
    std::string url;
    if (type == "base64") {
        if (!source.contains("media_type") || !source["media_type"].is_string() ||
            !source.contains("data") || !source["data"].is_string())
            http::fail(400, "base64 image sources require string media_type and data");
        url = "data:" + source["media_type"].get<std::string>() + ";base64," +
              source["data"].get<std::string>();
    } else if (type == "url") {
        url = block_string(source, "url", "url image source");
    } else {
        http::fail(400, "image source type '" + type +
                            "' is not supported; send the image as base64");
    }
    return json{{"type", "image_url"}, {"image_url", {{"url", std::move(url)}}}};
}

// Appends a text part, merging it into a preceding text part: the template
// concatenates content parts without a separator, which would glue Claude
// Code's separate reminder / prompt blocks into one line.
void push_text_part(json* parts, const std::string& text) {
    if (!parts->empty() && parts->back().contains("text")) {
        std::string merged = parts->back()["text"].get<std::string>();
        if (!merged.empty() && !text.empty()) merged += "\n\n";
        parts->back()["text"] = merged + text;
        return;
    }
    parts->push_back(json{{"type", "text"}, {"text", text}});
}

// A single text part collapses to a plain string.
json collapse_parts(json parts) {
    if (parts.size() == 1 && parts[0].contains("text")) return parts[0]["text"];
    return parts;
}

json anthropic_tool_result(const json& block) {
    const std::string& call_id = block_string(block, "tool_use_id", "tool_result");
    json content = "";
    if (block.contains("content") && !block["content"].is_null()) {
        const json& raw = block["content"];
        if (raw.is_string()) {
            content = raw;
        } else if (raw.is_array()) {
            json parts = json::array();
            for (const auto& part : raw) {
                const std::string type = block_type(part);
                if (type == "text") push_text_part(&parts, block_string(part, "text", "text"));
                else if (type == "image") parts.push_back(anthropic_image(part));
                else http::fail(400, "unsupported block type in tool_result content: " + type);
            }
            content = parts.empty() ? json("") : collapse_parts(std::move(parts));
        } else {
            http::fail(400, "tool_result content must be a string or an array of blocks");
        }
    }
    return json{{"role", "tool"}, {"tool_call_id", call_id}, {"content", std::move(content)}};
}

// Anthropic system + messages -> chat messages. tool_result blocks become
// role "tool" messages placed before the rest of their user turn: the
// template needs them right after the assistant turn that made the calls.
json anthropic_messages(const json& in) {
    json out = json::array();
    if (in.contains("system") && !in["system"].is_null())
        out.push_back(json{{"role", "system"},
                           {"content", anthropic_text(in["system"], "system", "\n\n")}});
    if (!in.contains("messages") || !in["messages"].is_array() || in["messages"].empty())
        http::fail(400, "messages: at least one message is required");
    for (const auto& m : in["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string())
            http::fail(400, "messages entries require a string role");
        const std::string role = m["role"].get<std::string>();
        if (role != "user" && role != "assistant")
            http::fail(400, "messages: role must be 'user' or 'assistant'");
        if (!m.contains("content")) http::fail(400, "messages entries require content");
        const json& content = m["content"];
        if (content.is_string()) {
            out.push_back(json{{"role", role}, {"content", content}});
            continue;
        }
        if (!content.is_array())
            http::fail(400, "message content must be a string or an array of blocks");

        if (role == "user") {
            json parts = json::array();
            for (const auto& block : content) {
                const std::string type = block_type(block);
                if (type == "text") {
                    push_text_part(&parts, block_string(block, "text", "text"));
                } else if (type == "image") {
                    parts.push_back(anthropic_image(block));
                } else if (type == "tool_result") {
                    out.push_back(anthropic_tool_result(block));
                } else if (type == "document") {
                    const json* source = block.contains("source") ? &block["source"] : nullptr;
                    if (source == nullptr || !source->is_object() || !source->contains("type") ||
                        (*source)["type"] != "text")
                        http::fail(400, "only plain-text document sources are supported");
                    push_text_part(&parts, block_string(*source, "data", "document source"));
                } else {
                    http::fail(400, "unsupported content block type in a user message: " + type);
                }
            }
            if (!parts.empty())
                out.push_back(json{{"role", "user"}, {"content", collapse_parts(std::move(parts))}});
            continue;
        }

        std::string text, reasoning;
        json calls = json::array();
        for (const auto& block : content) {
            const std::string type = block_type(block);
            if (type == "text") {
                text += block_string(block, "text", "text");
            } else if (type == "thinking") {
                reasoning += block_string(block, "thinking", "thinking");
            } else if (type == "redacted_thinking") {
                continue;  // opaque to this server
            } else if (type == "tool_use") {
                json input = block.contains("input") && !block["input"].is_null()
                                 ? block["input"]
                                 : json::object();
                if (!input.is_object()) http::fail(400, "tool_use input must be an object");
                calls.push_back(json{{"id", block_string(block, "id", "tool_use")},
                                     {"type", "function"},
                                     {"function",
                                      {{"name", block_string(block, "name", "tool_use")},
                                       {"arguments", std::move(input)}}}});
            } else {
                http::fail(400, "unsupported content block type in an assistant message: " + type);
            }
        }
        json message{{"role", "assistant"}, {"content", text}};
        if (!reasoning.empty()) message["reasoning_content"] = reasoning;
        if (!calls.empty()) message["tool_calls"] = std::move(calls);
        out.push_back(std::move(message));
    }
    return out;
}

// Anthropic body -> chat-completions body. A trailing assistant message is a
// prefill: it is removed from `messages` and returned in *prefill, and the
// reply continues from it (the prefill itself is not echoed back).
json anthropic_to_chat(const json& in, std::string* prefill) {
    json out = json::object();
    out["messages"] = anthropic_messages(in);
    if (in["messages"].back()["role"] == "assistant") {
        json& last = out["messages"].back();
        if (last.contains("tool_calls"))
            http::fail(400, "the final assistant message cannot contain tool_use blocks");
        *prefill = last["content"].get<std::string>();
        out["messages"].erase(out["messages"].size() - 1);
    }

    // Anthropic sampling fields share the chat names; the rest are this
    // server's chat extensions, accepted here too.
    for (const char* k : {"max_tokens", "temperature", "top_p", "top_k", "stream", "seed",
                          "min_p", "presence_penalty", "frequency_penalty", "logit_bias",
                          "enable_thinking", "preserve_thinking", "preserve_reasoning",
                          "auto_disable_thinking_with_tools", "reasoning_effort",
                          "tool_call_format", "max_tool_arg_chars", "max_tool_response_chars",
                          "add_vision_id", "think_guard_test_cut"})
        if (in.contains(k)) out[k] = in[k];
    if (in.contains("stop_sequences") && !in["stop_sequences"].is_null()) {
        if (!in["stop_sequences"].is_array())
            http::fail(400, "stop_sequences must be an array of strings");
        out["stop"] = in["stop_sequences"];
    }

    // thinking -> enable_thinking. budget_tokens has no engine equivalent and
    // is ignored. Omitted => left unset, so the server override table (or the
    // template default) decides.
    if (in.contains("thinking") && !in["thinking"].is_null() && !out.contains("enable_thinking")) {
        const json& t = in["thinking"];
        if (!t.is_object() || !t.contains("type") || !t["type"].is_string())
            http::fail(400, "thinking must be an object with a string type");
        const std::string type = t["type"].get<std::string>();
        if (type == "enabled" || type == "adaptive") out["enable_thinking"] = true;
        else if (type == "disabled") out["enable_thinking"] = false;
        else http::fail(400, "thinking.type must be 'enabled', 'adaptive' or 'disabled'");
    }
    if (in.contains("output_config") && in["output_config"].is_object() &&
        in["output_config"].contains("effort") && !out.contains("reasoning_effort"))
        out["reasoning_effort"] = in["output_config"]["effort"];

    if (in.contains("tools") && !in["tools"].is_null()) {
        if (!in["tools"].is_array()) http::fail(400, "tools must be an array");
        json tools = json::array();
        for (const auto& t : in["tools"]) {
            if (!t.is_object() || !t.contains("name") || !t["name"].is_string())
                http::fail(400, "tools entries require a string name");
            // Server tools (web_search, code_execution, ...) have a versioned
            // type and no input_schema: Anthropic runs those, this server can't.
            if (t.contains("type") && t["type"] != "custom" && !t.contains("input_schema"))
                continue;
            json function{{"name", t["name"]}};
            if (t.contains("description")) function["description"] = t["description"];
            function["parameters"] =
                t.contains("input_schema")
                    ? t["input_schema"]
                    : json{{"type", "object"}, {"properties", json::object()}};
            tools.push_back(json{{"type", "function"}, {"function", std::move(function)}});
        }
        out["tools"] = std::move(tools);
    }
    if (in.contains("tool_choice") && !in["tool_choice"].is_null()) {
        const json& c = in["tool_choice"];
        if (!c.is_object() || !c.contains("type") || !c["type"].is_string())
            http::fail(400, "tool_choice must be an object with a string type");
        const std::string type = c["type"].get<std::string>();
        if (type == "auto") out["tool_choice"] = "auto";
        else if (type == "any") out["tool_choice"] = "required";
        else if (type == "none") out["tool_choice"] = "none";
        else if (type == "tool")
            out["tool_choice"] = json{{"type", "function"},
                                      {"function", {{"name", block_string(c, "name", "tool_choice")}}}};
        else http::fail(400, "tool_choice.type must be 'auto', 'any', 'tool' or 'none'");
    }
    return out;
}

// Request state shared by /v1/messages and /v1/messages/count_tokens. Not
// copyable: opts points into body and tool_setup.
struct MessagesPrep {
    json body;
    json messages;
    std::string prefill;
    ToolSetup tool_setup;
    chat_template::Options opts;
    json normalized_effort;
    std::string tool_format = "xml";
    bool thinking = false;  // the reply opens inside <think>
    GenSpec spec;

    MessagesPrep() = default;
    MessagesPrep(const MessagesPrep&) = delete;
    MessagesPrep& operator=(const MessagesPrep&) = delete;
};

void prepare_messages_request(const http::Request& q, http::Response* r, MessagesPrep* p,
                              bool count_only) {
    if (!g_tok_ready)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    p->body = anthropic_to_chat(parse_body(q), &p->prefill);
    apply_overrides(&p->body, Api::Chat, r);
    const json& body = p->body;
    p->messages = body["messages"];
    std::string error;
    if (!toolparse::normalize_messages(&p->messages, &error)) http::fail(400, error);
    std::vector<vision::Frame> frames;
    if (!vision::prepare_messages(p->messages, &frames, &error)) http::fail(400, error);

    p->tool_setup = parse_tool_setup(body);
    if (p->tool_setup.choice.forced() && !p->prefill.empty())
        http::fail(400, "an assistant prefill cannot be combined with a forced tool_choice");
    chat_template::Options& opts = p->opts;
    if (p->tool_setup.enabled()) opts.tools = &p->tool_setup.tools;
    static const json kTrue = true;
    opts.add_generation_prompt = &kTrue;
    if (body.contains("reasoning_effort"))
        opts.reasoning_effort =
            normalize_reasoning_effort(&body["reasoning_effort"], &p->normalized_effort);
    require_optional_boolean(body, "enable_thinking");
    require_optional_boolean(body, "preserve_thinking");
    require_optional_boolean(body, "preserve_reasoning");
    require_optional_boolean(body, "auto_disable_thinking_with_tools");
    require_optional_boolean(body, "add_vision_id");
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null())
        opts.enable_thinking = &body["enable_thinking"];
    if (body.contains("preserve_thinking") && !body["preserve_thinking"].is_null())
        opts.preserve_thinking = &body["preserve_thinking"];
    if (body.contains("preserve_reasoning") && !body["preserve_reasoning"].is_null())
        opts.preserve_reasoning = &body["preserve_reasoning"];
    if (body.contains("auto_disable_thinking_with_tools"))
        opts.auto_disable_thinking_with_tools = &body["auto_disable_thinking_with_tools"];
    if (body.contains("tool_call_format")) {
        if (!body["tool_call_format"].is_string() ||
            (body["tool_call_format"] != "xml" && body["tool_call_format"] != "json"))
            http::fail(400, "tool_call_format must be 'xml' or 'json'");
        p->tool_format = body["tool_call_format"].get<std::string>();
        opts.tool_call_format = &body["tool_call_format"];
    }
    if (body.contains("max_tool_arg_chars")) {
        if (!body["max_tool_arg_chars"].is_number_integer() &&
            !body["max_tool_arg_chars"].is_number_unsigned())
            http::fail(400, "max_tool_arg_chars must be a non-negative integer");
        opts.max_tool_arg_chars = &body["max_tool_arg_chars"];
    }
    if (body.contains("max_tool_response_chars")) {
        if (!body["max_tool_response_chars"].is_number_integer() &&
            !body["max_tool_response_chars"].is_number_unsigned())
            http::fail(400, "max_tool_response_chars must be a non-negative integer");
        opts.max_tool_response_chars = &body["max_tool_response_chars"];
    }
    if (body.contains("add_vision_id")) opts.add_vision_id = &body["add_vision_id"];

    chat_template::RenderResult rr = chat_template::render_chat_template(&p->messages, opts);
    if (!rr.ok) http::fail(400, rr.error);
    // Forced tool calls and prefills both close the think block in the prompt
    // (ToolChoice::prompt_suffix does it for the former), so the reply carries
    // no reasoning to split off.
    p->thinking = rr.thinking_enabled && !p->tool_setup.choice.forced() && p->prefill.empty();
    rr.text += p->tool_setup.choice.prompt_suffix(rr.thinking_enabled, p->tool_format);
    if (!p->prefill.empty()) {
        if (rr.thinking_enabled) rr.text += "</think>\n\n";
        rr.text += p->prefill;
    }

    GenSpec& spec = p->spec;
    std::vector<size_t> starts;
    if (!encode_spliced(rr.text, &frames, &spec.ids, &starts, &error)) http::fail(400, error);
    if (count_only) return;
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    spec.snaps = compute_snap_cuts(p->messages, opts, rr.text, spec.ids, starts);
    apply_sampling(body, &spec);
    ban_tool_call_without_tools(p->tool_setup, &spec);
}

// Complete tool calls carry valid JSON objects; anything else (cannot happen
// for a finished call) degrades to an empty input rather than a 500.
json tool_use_input(const std::string& arguments) {
    if (arguments.empty()) return json::object();
    try {
        json value = json::parse(arguments);
        if (value.is_object()) return value;
    } catch (...) {
    }
    return json::object();
}

bool has_visible_text(const std::string& s) {
    return s.find_first_not_of(" \t\r\n") != std::string::npos;
}

void messages_impl(const http::Request& q, http::Response* r, http::Stream* st) {
    MessagesPrep p;
    prepare_messages_request(q, r, &p, /*count_only=*/false);
    GenSpec& spec = p.spec;
    const ToolSetup& tool_setup = p.tool_setup;
    const bool stream = bool_field(p.body, "stream", false) && st != nullptr;
    const std::string id = make_id("msg_");

    // Anthropic counts prompt cache reads apart from input_tokens. KVSNAP
    // resume counts restored generation state in n_cached, but that state is
    // not client-visible prompt cache and must not produce negative usage.
    auto usage = [&](const GenOutcome* o) {
        const long long prompt = static_cast<long long>(spec.ids.size());
        const long long reported_cached = o != nullptr ? o->n_cached : 0;
        const long long cached = reported_cached >= 0 && reported_cached <= prompt
                                     ? reported_cached
                                     : 0;
        json u{{"input_tokens", prompt - cached},
               {"cache_creation_input_tokens", 0},
               {"cache_read_input_tokens", cached},
               {"output_tokens", o != nullptr ? o->n_gen : 0}};
        if (o != nullptr && !o->guard.empty()) u["think_guard"] = o->guard;
        return u;
    };
    auto stop_reason = [](const GenOutcome& o, bool partial_call, bool has_calls) {
        if (o.reason == "length" || partial_call) return "max_tokens";
        if (has_calls) return "tool_use";
        if (o.hit_stop) return "stop_sequence";
        return "end_turn";
    };
    auto stop_sequence = [](const GenOutcome& o) {
        return o.hit_stop ? json(o.stop_hit) : json(nullptr);
    };
    auto message_object = [&](json content, json reason, json sequence, json u) {
        json m;
        m["id"] = id;
        m["type"] = "message";
        m["role"] = "assistant";
        m["model"] = g_cfg.model;
        m["content"] = std::move(content);
        m["stop_reason"] = std::move(reason);
        m["stop_sequence"] = std::move(sequence);
        m["usage"] = std::move(u);
        return m;
    };
    auto make_parser = [&]() {
        return toolparse::StreamParser(
            tool_setup.tools, []() { return make_id("toolu_"); }, tool_setup.enabled(),
            p.tool_format, tool_setup.choice.name);
    };

    const ThinkGuardCtx guard{p.thinking, &tool_setup, p.tool_format};
    if (!stream) {
        GenOutcome o = run_generation_guarded(
            spec, guard, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
        std::string reasoning, answer;
        split_reasoning(o.text, p.thinking, &reasoning, &answer);
        toolparse::StreamParser parser = make_parser();
        parser.feed(tool_setup.choice.parser_prefix(p.tool_format));
        parser.feed(answer);
        parser.feed(o.reasoning_calls);
        parser.finish();
        json content = json::array();
        if (!reasoning.empty())
            content.push_back(json{{"type", "thinking"}, {"thinking", reasoning}, {"signature", ""}});
        if (has_visible_text(parser.content()))
            content.push_back(json{{"type", "text"}, {"text", parser.content()}});
        for (const auto& call : parser.calls())
            content.push_back(json{{"type", "tool_use"},
                                   {"id", call.id},
                                   {"name", call.name},
                                   {"input", tool_use_input(call.arguments)}});
        json m = message_object(
            std::move(content),
            stop_reason(o, parser.has_partial_call(), !parser.calls().empty()),
            stop_sequence(o), usage(&o));
        r->body = json_py::dumps(m, /*spaced=*/false);
        return;
    }

    // message_start is sent lazily (first ping or first block) so that errors
    // raised before the slot is granted still get a real HTTP status.
    r->sse = true;
    bool started = false;
    auto ensure_started = [&]() {
        if (started) return true;
        started = true;
        return send_anthropic_event(
            st, "message_start",
            json{{"type", "message_start"},
                 {"message", message_object(json::array(), nullptr, nullptr, usage(nullptr))}});
    };

    enum class Block { None, Thinking, Text, Tool };
    Block open_kind = Block::None;
    int open_index = -1;
    int next_index = 0;
    auto close_block = [&]() {
        if (open_kind == Block::None) return true;
        open_kind = Block::None;
        return send_anthropic_event(st, "content_block_stop",
                                    json{{"type", "content_block_stop"}, {"index", open_index}});
    };
    auto open_block = [&](Block kind, json block) {
        if (!ensure_started() || !close_block()) return false;
        open_kind = kind;
        open_index = next_index++;
        return send_anthropic_event(st, "content_block_start",
                                    json{{"type", "content_block_start"},
                                         {"index", open_index},
                                         {"content_block", std::move(block)}});
    };
    auto block_delta = [&](json delta) {
        return send_anthropic_event(st, "content_block_delta",
                                    json{{"type", "content_block_delta"},
                                         {"index", open_index},
                                         {"delta", std::move(delta)}});
    };
    auto emit_thinking = [&](const std::string& text) {
        if (text.empty()) return true;
        if (open_kind != Block::Thinking &&
            !open_block(Block::Thinking,
                        json{{"type", "thinking"}, {"thinking", ""}, {"signature", ""}}))
            return false;
        return block_delta(json{{"type", "thinking_delta"}, {"thinking", text}});
    };
    // Whitespace-only text (the newlines Qwen puts before <tool_call>) is held
    // back and only opens a text block once real text follows, matching the
    // non-streaming path, which drops whitespace-only text.
    std::string held_space;
    auto emit_text = [&](const std::string& text) {
        if (text.empty()) return true;
        if (open_kind != Block::Text) {
            if (!has_visible_text(text)) {
                held_space += text;
                return true;
            }
            if (!open_block(Block::Text, json{{"type", "text"}, {"text", ""}})) return false;
            std::string first = std::move(held_space);
            held_space.clear();
            return block_delta(json{{"type", "text_delta"}, {"text", first + text}});
        }
        return block_delta(json{{"type", "text_delta"}, {"text", text}});
    };
    std::string tool_arguments;
    auto dispatch_tool_events = [&](const std::vector<toolparse::Event>& events) {
        for (const auto& event : events) {
            if (event.type == toolparse::EventType::Content) {
                if (!emit_text(event.data)) return false;
            } else if (event.type == toolparse::EventType::CallStart) {
                held_space.clear();
                tool_arguments.clear();
                if (!open_block(Block::Tool, json{{"type", "tool_use"},
                                                  {"id", event.id},
                                                  {"name", event.name},
                                                  {"input", json::object()}}))
                    return false;
            } else if (event.type == toolparse::EventType::ArgumentsDelta) {
                if (event.data.empty()) continue;
                tool_arguments += event.data;
                if (!block_delta(json{{"type", "input_json_delta"}, {"partial_json", event.data}}))
                    return false;
            } else if (event.type == toolparse::EventType::CallEnd) {
                if (tool_arguments.empty() && !event.data.empty() &&
                    !block_delta(json{{"type", "input_json_delta"}, {"partial_json", event.data}}))
                    return false;
                if (!close_block()) return false;
            }
        }
        return true;
    };

    ThinkSplitter splitter(p.thinking);
    toolparse::StreamParser parser = make_parser();
    if (!dispatch_tool_events(parser.feed(tool_setup.choice.parser_prefix(p.tool_format)))) return;
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        if (!ensure_started()) return false;
        return send_anthropic_event(st, "ping", json{{"type", "ping"}});
    };
    GenOutcome o = run_generation_guarded(
        spec, guard,
        [&](const std::string& delta) {
            if (st->disconnected()) return false;
            if (!p.thinking) return dispatch_tool_events(parser.feed(delta));
            std::string reasoning_delta, content_delta;
            splitter.feed(delta, &reasoning_delta, &content_delta);
            if (!emit_thinking(reasoning_delta)) return false;
            return dispatch_tool_events(parser.feed(content_delta));
        },
        id, on_wait);
    if (o.client_gone) return;
    if (p.thinking) {
        std::string reasoning_delta, content_delta;
        splitter.finish(&reasoning_delta, &content_delta);
        if (!emit_thinking(reasoning_delta)) return;
        if (!dispatch_tool_events(parser.feed(content_delta))) return;
        if (!dispatch_tool_events(parser.feed(o.reasoning_calls))) return;
    }
    if (!dispatch_tool_events(parser.finish())) return;
    // A tool block still open_kind here was cut off mid-call.
    const bool partial_call = parser.has_partial_call() || open_kind == Block::Tool;
    if (!ensure_started() || !close_block()) return;
    if (!send_anthropic_event(
            st, "message_delta",
            json{{"type", "message_delta"},
                 {"delta",
                  {{"stop_reason", stop_reason(o, partial_call, !parser.calls().empty())},
                   {"stop_sequence", stop_sequence(o)}}},
                 {"usage", usage(&o)}}))
        return;
    send_anthropic_event(st, "message_stop", json{{"type", "message_stop"}});
}

void messages_count_impl(const http::Request& q, http::Response* r, http::Stream*) {
    MessagesPrep p;
    prepare_messages_request(q, r, &p, /*count_only=*/true);
    r->body = json_py::dumps(
        json{{"input_tokens", static_cast<long long>(p.spec.ids.size())}}, /*spaced=*/false);
}

// Re-wraps errors in Anthropic's envelope. Once the SSE stream has started
// the status line is gone, so the error goes out as an `error` event.
void with_anthropic_errors(void (*impl)(const http::Request&, http::Response*, http::Stream*),
                           const http::Request& q, http::Response* r, http::Stream* st) {
    int status = 0;
    std::string message;
    try {
        impl(q, r, st);
        return;
    } catch (const http::Error& e) {
        status = e.status;
        message = e.message;
    } catch (const std::exception& e) {
        status = 500;
        message = std::string("internal error: ") + e.what();
    }
    if (st != nullptr && st->started()) {
        send_anthropic_event(st, "error", anthropic_error_json(status, message));
        return;  // r->sse is still set: the server closes the chunked body
    }
    r->sse = false;
    r->status = status;
    r->content_type = "application/json";
    r->body = json_py::dumps(anthropic_error_json(status, message), /*spaced=*/false);
}

// POST /v1/messages
void handle_messages(const http::Request& q, http::Response* r, http::Stream* st) {
    with_anthropic_errors(messages_impl, q, r, st);
}

// POST /v1/messages/count_tokens
void handle_messages_count(const http::Request& q, http::Response* r, http::Stream* st) {
    with_anthropic_errors(messages_count_impl, q, r, st);
}

// POST /v1/tokenize — text -> token ids. `text` may also be spelled `prompt`
// (vLLM's /tokenize convention). Offsets, when requested, are UTF-8 byte
// ranges into the input text, not character indexes.
void handle_tokenize(const http::Request& q, http::Response* r, http::Stream*) {
    if (!g_tok_ready)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    const json body = parse_body(q);
    const char* key = body.contains("text")      ? "text"
                      : body.contains("prompt")  ? "prompt"
                                                 : nullptr;
    if (key == nullptr) http::fail(400, "text is required");
    const json& tv = body[key];
    if (!tv.is_string()) http::fail(400, std::string(key) + " must be a string");
    const bool want_offsets = bool_field(body, "return_offsets", false);

    const auto spans = g_tok.encode_with_offsets(tv.get_ref<const std::string&>());
    json ids = json::array();
    ids.get_ref<json::array_t&>().reserve(spans.size());
    json offsets = json::array();
    if (want_offsets) offsets.get_ref<json::array_t&>().reserve(spans.size());
    for (const auto& t : spans) {
        ids.push_back(t.id);
        if (want_offsets) offsets.push_back(json::array({t.start, t.end}));
    }
    json j;
    j["tokens"] = std::move(ids);
    j["count"] = spans.size();
    if (want_offsets) j["offsets"] = std::move(offsets);
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// POST /v1/detokenize — token ids -> text. `tokens` may also be spelled `ids`.
void handle_detokenize(const http::Request& q, http::Response* r,
                       http::Stream*) {
    if (!g_tok_ready)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    const json body = parse_body(q);
    const json* arr = body.contains("tokens") ? &body["tokens"]
                      : body.contains("ids")  ? &body["ids"]
                                              : nullptr;
    if (arr == nullptr) http::fail(400, "tokens is required");
    if (!arr->is_array()) http::fail(400, "tokens must be an array of integers");
    const bool skip_special = bool_field(body, "skip_special_tokens", true);

    std::vector<int> ids;
    ids.reserve(arr->size());
    for (const auto& v : *arr) {
        if (!v.is_number_integer() && !v.is_number_unsigned())
            http::fail(400, "tokens must be an array of integers");
        const long long id = v.is_number_unsigned()
                                 ? (long long)v.get<unsigned long long>()
                                 : v.get<long long>();
        if (id < 0 || id >= g_tok.vocab_size())
            http::fail(400, "token id out of range: " + std::to_string(id));
        ids.push_back((int)id);
    }
    json j;
    j["text"] = g_tok.decode(ids, skip_special);
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// -------------------------------------------------------------------- main --

// INFO 行：I mtp draft_head ctx spec_rows default drafter_weights dflash2
//   cache_mb cache_align kv_slots slot_ctx cache_mode sampling kv_pool_tokens
struct EngineInfo {
    int context = 0;
    int kv_slots = 0;  // 0 = 老引擎未上报（按单槽处理）
    int kv_pool = 0;
};

bool parse_engine_info(const std::string& line, EngineInfo* ei) {
    std::vector<long long> f;
    std::istringstream ss(line);
    std::string tok;
    ss >> tok;  // "I"
    long long v;
    while (ss >> v) f.push_back(v);
    if (f.size() < 3) return false;
    ei->context = (int)f[2];
    if (f.size() >= 10 && f[9] >= 1 && f[9] <= 64) ei->kv_slots = (int)f[9];
    if (f.size() >= 14 && f[13] > 0) ei->kv_pool = (int)f[13];
    return true;
}

// 初始探测失败（GUI/监管路径下 API 先于引擎监听）时由后台线程调用：
// 引擎就绪后应用 INFO 并把连接池扩到引擎的槽位数。只增不缩——运行中的
// SlotGuard 持有池内对象指针，缩减需要排空在途请求；引擎重启改 PARALLEL
// 的场景请重启 API。
void apply_engine_info(const EngineInfo& ei) {
    {
        std::lock_guard<std::mutex> lk(g_slot_mtx);
        if (ei.context > 0) g_cfg.context = ei.context;
        if (ei.kv_pool > 0) g_kv_pool = ei.kv_pool;
        if (ei.kv_slots > g_slots) g_slots = ei.kv_slots;
        // 池以 g_pool.size() 为基线补齐到 g_slots：启动时为空全量建，
        // 重探测时只补差额；kv_slots=0（老引擎）也保证至少有 1 条连接。
        for (int i = (int)g_pool.size(); i < g_slots; ++i) {
            g_pool.push_back(std::make_unique<qwenox::EngineClient>());
            g_pool_free.push_back(g_pool.back().get());
        }
    }
    g_slot_cv.notify_all();  // 排队中的请求重新评估 g_in_flight < g_slots
}

void log_slots() {
    fprintf(stderr, "qwenox-api: %d concurrent generation slot%s (per-slot context %d,"
            " shared KV pool %d)\n",
            g_slots, g_slots > 1 ? "s" : "", g_cfg.context,
            g_kv_pool > 0 ? g_kv_pool : g_cfg.context);
}

void reprobe_engine_loop() {
    // 引擎冷加载可达分钟级：前 5 分钟每 2s 一探，之后降到 15s（覆盖用户
    // 稍后在面板上手动启动引擎的场景）。connect 失败是廉价的本机拒绝。
    for (int attempt = 0;; ++attempt) {
        std::this_thread::sleep_for(std::chrono::seconds(attempt < 150 ? 2 : 15));
        qwenox::EngineClient c;  // 独立连接：不与 g_ctl 的控制查询竞争
        c.set_timeouts(5.0, 5.0);
        std::string line, err;
        if (!c.connect(g_cfg.engine_addr, &err, 0.3)) continue;
        if (!c.info(&line, &err)) continue;
        EngineInfo ei;
        if (!parse_engine_info(line, &ei)) continue;
        apply_engine_info(ei);
        fprintf(stderr, "qwenox-api: engine INFO (re-probe): %s\n", line.c_str());
        log_slots();
        return;
    }
}

bool probe_engine() {
    std::string err, line;
    // Startup must not stall for the generation budget: an unreachable or
    // wedged engine falls back to the configured defaults (and one slot)
    // instead of blocking the listen socket for minutes; the caller then
    // retries in reprobe_engine_loop so the pool still reaches full width
    // once the engine is up. The control connection keeps the short
    // timeouts: MEM answers at once. The connect itself is bounded
    // too: on some machines a refused loopback connect takes ~2s to come
    // back (filter driver), delaying startup.
    g_ctl.set_timeouts(5.0, 5.0);
    bool ok = false;
    if (!g_ctl.connect(g_cfg.engine_addr, &err, 0.3)) {
        fprintf(stderr, "qwenox-api: engine connect failed: %s\n", err.c_str());
    } else if (g_ctl.info(&line, &err)) {
        EngineInfo ei;
        if (parse_engine_info(line, &ei)) {
            apply_engine_info(ei);
            ok = true;
        }
        fprintf(stderr, "qwenox-api: engine INFO: %s\n", line.c_str());
    } else {
        fprintf(stderr, "qwenox-api: engine INFO unavailable (%s); continuing with "
                        "ctx=%d\n", err.c_str(), g_cfg.context);
    }
    // No usable INFO: still build the g_slots (= 1) default connection so a
    // request that lands before reprobe_engine_loop widens the pool finds a
    // client (SlotGuard pops g_pool_free; an empty pool crashed the API --
    // 10-13, API started before the engine, first request beat the re-probe).
    if (!ok) apply_engine_info(EngineInfo{});
    log_slots();
    return ok;
}

}  // namespace

// exe 所在目录：service.conf 自动探测的兜底基准（build/ 里的 exe → 上一级项目根）。
static std::string exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n ? static_cast<size_t>(n) : 0);
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string p = n > 0 ? std::string(buf, static_cast<size_t>(n)) : std::string();
#endif
    const size_t slash = p.find_last_of("\\/");
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

// 致命退出：双击运行（独占控制台）时窗口会随进程关闭，暂停一下让错误可读。
// 从终端运行（共享控制台）时不暂停；托盘模式（无控制台）弹 MessageBox。
#ifdef _WIN32
static bool g_tray_mode = true;       // 托盘模式（默认）；--console 时为 false
static std::string g_api_log_abs;     // 托盘模式 stdout/stderr 重定向的日志（绝对路径）

// UTF-8 → UTF-16 的 MessageBox：A 版弹窗在非中文系统上中文会乱码
static void mb_qwenox(const char* text, UINT icon) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, text, -1, w.data(), n);
    MessageBoxW(nullptr, w.c_str(), L"Qwenox", icon | MB_OK | MB_SETFOREGROUND | MB_TOPMOST);
}

// 本程序是 GUI 子系统（双击不出黑窗）。--console 需要控制台：先尝试挂到调用者
// （cmd）的控制台上，不行就新开一个窗口。stdout 已被调用方重定向（如
// start_win.sh 的日志文件）时不抢控制台，保持重定向原样。
static void console_open() {
    const DWORD t = GetFileType(GetStdHandle(STD_OUTPUT_HANDLE));
    if (t == FILE_TYPE_DISK || t == FILE_TYPE_PIPE) return;
    const bool attached = AttachConsole(ATTACH_PARENT_PROCESS) != 0;
    if (!attached && !AllocConsole()) return;
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE out = CreateFileA("CONOUT$", GENERIC_READ | GENERIC_WRITE,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    HANDLE in = CreateFileA("CONIN$", GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);
    if (out != INVALID_HANDLE_VALUE) {
        SetStdHandle(STD_OUTPUT_HANDLE, out);
        SetStdHandle(STD_ERROR_HANDLE, out);
    }
    if (in != INVALID_HANDLE_VALUE) SetStdHandle(STD_INPUT_HANDLE, in);
    freopen("CONOUT$", "w", stdout);
    freopen("CONOUT$", "w", stderr);
    freopen("CONIN$", "r", stdin);
    SetConsoleOutputCP(CP_UTF8);
    if (attached) printf("\n");  // cmd 的提示符已经先打印了，另起一行
}
#endif

static int gexit(int code) {
#ifdef _WIN32
    if (g_tray_mode) {
        fflush(stdout);
        fflush(stderr);
        mb_qwenox("Qwenox failed to start. See logs\\api-win-*.log for details.\n"
                  "Qwenox 启动失败，详情见 logs\\api-win-*.log",
                  MB_ICONERROR);
        return code;
    }
    DWORD ids[8];
    if (GetConsoleWindow() && GetConsoleProcessList(ids, 8) <= 1) {
        fflush(stdout);
        fflush(stderr);
        system("pause");
    }
#endif
    return code;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    // 预扫描 --console：本程序是 GUI 子系统，默认托盘模式（无控制台）；
    // --console 立即开控制台，让用法/错误输出可见。
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--console") == 0) g_tray_mode = false;
    if (!g_tray_mode) console_open();
    if (!wsa_init()) {
        std::fprintf(stderr, "WSAStartup failed\n");
        return gexit(1);
    }
#endif
    // 记录 argv 显式给定的项：未给定的由 service.conf 补默认（独立启动自举）。
    bool arg_tokenizer = false, arg_engine = false, arg_listen = false,
         arg_context = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (a == "--tokenizer") { g_cfg.tokenizer_dir = next(); arg_tokenizer = true; }
        else if (a == "--engine") { g_cfg.engine_addr = next(); arg_engine = true; }
        else if (a == "--port") {
            const std::string p = next();
            g_cfg.listen = g_cfg.listen.substr(0, g_cfg.listen.rfind(':') + 1) + p;
            arg_listen = true;
        } else if (a == "--host") {
            const std::string h = next();
            g_cfg.listen = h + ":" + g_cfg.listen.substr(g_cfg.listen.rfind(':') + 1);
            arg_listen = true;
        } else if (a == "--context") { g_cfg.context = std::atoi(next().c_str()); arg_context = true; }
        else if (a == "--model") g_cfg.model = next();
        else if (a == "--listen") { g_cfg.listen = next(); arg_listen = true; }
        else if (a == "--overrides") g_cfg.overrides_path = next();  // "" = memory only
        else if (a == "--service-conf") g_cfg.service_conf = next();
#ifdef _WIN32
        else if (a == "--console") g_tray_mode = false;  // 已在 main 开头预扫描并开控制台
#endif
        else {
            fprintf(stderr,
                    "usage: %s [--tokenizer DIR] [--engine H:P] [--listen H:P] "
                    "[--port N] [--host H] [--context N] [--model NAME] "
                    "[--overrides FILE] [--service-conf FILE]"
#ifdef _WIN32
                    " [--console]"
#endif
                    "\n",
                    argv[0]);
            return gexit(2);
        }
    }
    // ---- service.conf：独立启动（无 argv）时从 conf 自举 ----------------------
    {
        std::string path = g_cfg.service_conf;
        if (path.empty())
            if (const char* e = std::getenv("QWENOX_SERVICE_CONF")) path = e;
        bool from_exe_fallback = false;
        if (path.empty()) {
            path = "service.conf";
            std::string probe;
            if (!svcconf::read_text_file(path, &probe)) {
                // build/ 里的 exe 直接运行：CWD 没有 conf 时找 exe 上一级的项目根
                const std::string dir = exe_dir();
                const std::string alt = dir + "/../service.conf";
                if (svcconf::read_text_file(alt, &probe)) {
                    path = alt;
                    from_exe_fallback = true;
                }
            }
        }
        g_conf_path = path;
        const size_t slash = path.find_last_of("\\/");
        g_conf_root = slash == std::string::npos ? "." : path.substr(0, slash);
        if (g_conf_root.empty()) g_conf_root = "/";
        if (from_exe_fallback) {
            // 相对路径（tokenizer、overrides、logs、模型）都以项目根为基准
#ifdef _WIN32
            _chdir(g_conf_root.c_str());
#else
            if (chdir(g_conf_root.c_str()) != 0)
                fprintf(stderr, "[warn] chdir(%s) 失败: %s\n", g_conf_root.c_str(), strerror(errno));
#endif
        }
        svcconf::load_conf(path, &g_svc_conf);
        // 优先级 argv > env > conf > 内置（svcconf::cfg 已含 env > conf）
        if (!arg_tokenizer)
            g_cfg.tokenizer_dir =
                svcconf::cfg(g_svc_conf, "TOKENIZER_DIR", g_cfg.tokenizer_dir);
        if (!arg_engine)
            g_cfg.engine_addr =
                engine_net::connect_host(svcconf::cfg(g_svc_conf, "ENGINE_HOST",
                                                      engine_net::kDefaultHost)) +
                ":" + svcconf::cfg(g_svc_conf, "ENGINE_PORT", "8730");
        if (!arg_listen)
            g_cfg.listen = svcconf::cfg(g_svc_conf, "API_HOST", "0.0.0.0") + ":" +
                           svcconf::cfg(g_svc_conf, "API_PORT", "8731");
        if (!arg_context) {
            long c = 0;
            if (svcconf::parse_int(svcconf::cfg(g_svc_conf, "MAX_CONTEXT", ""), &c) &&
                c > 0)
                g_cfg.context = static_cast<int>(c);
        }
        // vision::max_images 只读环境变量 QWENOX_API_MAX_IMAGES：Linux 启动器会
        // export，Windows 托盘模式下由 engine_sup 在拉起引擎时写入——但引擎可能
        // 外部启动、或晚于首个请求启动，上限就卡在默认 8。这里按 conf 补齐；
        // 已设置则不覆盖（engine_sup 启动引擎时仍会按当时的 conf 重写）。
        if (!std::getenv("QWENOX_API_MAX_IMAGES")) {
            long mi = 0;
            if (svcconf::parse_int(svcconf::cfg(g_svc_conf, "MAX_IMAGES", ""), &mi) &&
                mi >= 1) {
                const std::string v = std::to_string(mi);
#ifdef _WIN32
                // SetEnvironmentVariableA 只改 Win32 环境块，本进程 CRT 的
                // getenv 看不到；_putenv 两者都更新。
                _putenv((std::string("QWENOX_API_MAX_IMAGES=") + v).c_str());
#else
                setenv("QWENOX_API_MAX_IMAGES", v.c_str(), 0);
#endif
            }
        }
    }
    // 请求日志：REQLOG / REQLOG_DIR / REQLOG_LEVEL 只从 service.conf 读
    // （QWENOX_REQLOG* 环境变量优先，便于独立运行/测试）。刻意不加入
    // svcconf::kManagedKeys：设置面板和网页不显示、也不能改，只能手改配置。
    // 与大多数 conf 项一样，改动在下次启动 API 时生效。
    {
        auto eff = [](const char* env_key, const char* conf_key, const char* dflt) {
            if (const char* e = std::getenv(env_key); e != nullptr && *e != '\0')
                return std::string(e);
            return svcconf::cfg(g_svc_conf, conf_key, dflt);
        };
        const std::string on = eff("QWENOX_REQLOG", "REQLOG", "0");
        const std::string dir = eff("QWENOX_REQLOG_DIR", "REQLOG_DIR", "logs/reqlog");
        long level = 6;
        if (!svcconf::parse_int(eff("QWENOX_REQLOG_LEVEL", "REQLOG_LEVEL", "6"), &level) ||
            level < 1 || level > 10) {
            fprintf(stderr, "qwenox-api: REQLOG_LEVEL must be 1..10, using 6\n");
            level = 6;
        }
        reqlog::configure(on != "0", dir, static_cast<int>(level));
    }
#ifdef _WIN32
    // 托盘模式没有控制台：stdout/stderr 重定向到 logs\api-win-<时间戳>.log。
    // 放在 conf 探测之后：CWD 已是项目根，日志落在根的 logs\ 下。
    if (g_tray_mode) {
        // 防多开（--console 不受限：脚本场景由端口占用兜底）。必须在 listen 之前：
        // Windows 的 SO_REUSEADDR 允许后启动的实例抢绑同一端口。句柄不关，退出回收。
        CreateMutexW(nullptr, FALSE, L"Local\\Qwenox-api");
        if (GetLastError() == ERROR_ALREADY_EXISTS) {
            mb_qwenox("Qwenox is already running (check the tray icon in the "
                      "taskbar corner, possibly under the ^ overflow).\n"
                      "Qwenox 已经在运行了（看任务栏右下角的托盘图标，"
                      "可能藏在 ^ 展开区里）。",
                      MB_ICONWARNING);
            return 1;
        }
        CreateDirectoryA("logs", nullptr);
        SYSTEMTIME st;
        GetLocalTime(&st);
        char stamp[32];
        snprintf(stamp, sizeof(stamp), "%04d%02d%02d-%02d%02d%02d", st.wYear, st.wMonth,
                 st.wDay, st.wHour, st.wMinute, st.wSecond);
        const std::string lp = std::string("logs\\api-win-") + stamp + ".log";
        char cwd[MAX_PATH];
        GetCurrentDirectoryA(MAX_PATH, cwd);
        g_api_log_abs = std::string(cwd) + "\\" + lp;
        FILE* f = fopen(lp.c_str(), "wb");  // 先清空，再以追加方式共用
        if (f) fclose(f);
        if (freopen(lp.c_str(), "ab", stdout)) setvbuf(stdout, nullptr, _IONBF, 0);
        if (freopen(lp.c_str(), "ab", stderr)) setvbuf(stderr, nullptr, _IONBF, 0);
    }
#endif
    std::string rope_error;
    if (!rope_from_env(g_cfg.rope, rope_error) || !rope_config_valid(g_cfg.rope)) {
        if (rope_error.empty()) rope_error = "invalid YaRN configuration";
        fprintf(stderr, "qwenox-api: %s\n", rope_error.c_str());
        return gexit(2);
    }
    // Environment, not argv: keeps the key out of `ps`.
    if (const char* k = std::getenv("QWENOX_API_ADMIN_KEY")) g_cfg.admin_key = k;
    if (const char* t = std::getenv("QWENOX_API_TOKCACHE"))
        g_tcache.set_capacity(std::strtoull(t, nullptr, 10));
    {
        // "" disables persistence; default keeps the cache across restarts.
        const char* f = std::getenv("QWENOX_API_TOKCACHE_FILE");
        g_tcache.set_file(f ? f : "data/tcache.bin");
        if (const char* s = std::getenv("QWENOX_API_TOKCACHE_SAVE_S"))
            g_tcache.set_save_interval(std::atoi(s));
    }
    load_overrides();

    std::string err;
    // 词表缺失不再是致命错误：独立运行时控制台、配置与引擎管理照常可用，
    // 仅推理端点返回 503。TOKENIZER_DIR 指向的目录不可用时回退到仓库内置的
    // data/tokenizer（随项目分发的兜底词表）；显式 --tokenizer 失败则不回退。
    g_tok_ready = g_tok.load(g_cfg.tokenizer_dir, &err);
    if (!g_tok_ready && !arg_tokenizer) {
        const std::string fb = g_conf_root + "/data/tokenizer";
        std::string err2;
        if (g_tok.load(fb, &err2)) {
            fprintf(stderr,
                    "qwenox-api: tokenizer: %s unavailable (%s); using bundled %s\n",
                    g_cfg.tokenizer_dir.c_str(), err.c_str(), fb.c_str());
            g_tok_ready = true;
        }
    }
    if (!g_tok_ready)
        fprintf(stderr,
                "qwenox-api: WARNING: tokenizer not loaded: %s\n"
                "qwenox-api: console/config/engine endpoints work; inference returns 503\n",
                err.c_str());
    g_tcache.load_file();
    g_ckpt_ids = default_ckpt_tokens();
    if (!probe_engine()) std::thread(reprobe_engine_loop).detach();

    http::Server srv;
    if (reqlog::enabled()) {
        srv.set_request_logger([](const http::Request& q, int status,
                                  const std::string& response, bool sse,
                                  bool delivered, double duration_ms) {
            if (!reqlog::should_log(q.method, q.path)) return;
            reqlog::Record r;
            r.method = q.method;
            r.path = q.path;
            r.query = q.query;
            r.remote = q.remote;
            r.version = q.version;
            r.headers = q.headers;
            r.status = status;
            r.sse = sse;
            r.delivered = delivered;
            r.duration_ms = duration_ms;
            r.request_body = q.body;
            r.response_body = response;
            reqlog::write(r);
        });
        fprintf(stderr, "qwenox-api: request log: %s (zip level %d)\n",
                reqlog::directory().c_str(), reqlog::level());
    }
    if (!srv.listen(g_cfg.listen, &err)) {
        fprintf(stderr, "qwenox-api: listen: %s\n", err.c_str());
        return gexit(1);
    }
    fprintf(stderr, "qwenox-api %s: listening on :%d model=%s ctx=%d slots=%d\n",
            QWENOX_VERSION_FULL, srv.port(), g_cfg.model.c_str(), g_cfg.context, g_slots);
    // 双击运行时给用户一个可直接点的入口
    fprintf(stderr, "qwenox-api: console: http://127.0.0.1:%d/\n", srv.port());

    init_web_dir();
    for (const StaticAsset& a : kStaticAssets) srv.on("GET", a.url, make_static_handler(a));
    srv.on("GET", "/v1/models", handle_models);
    srv.on("GET", "/health", handle_health);
    srv.on("GET", "/memory", handle_memory);
    srv.on("GET", "/power", handle_power);
    srv.on("GET", "/viz", handle_viz);
    srv.on("GET", "/reqstat/summary", handle_reqstat_summary);
    srv.on("GET", "/reqstat/usage", handle_reqstat_usage);
    srv.on("GET", "/reqstat/tail", handle_reqstat_tail);
    srv.on("GET", "/reqstat/page", handle_reqstat_page);
    srv.on("GET", "/admin/overrides", handle_overrides_get);
    srv.on("POST", "/admin/overrides", handle_overrides_post);
    srv.on("GET", "/admin/config", handle_config_get);
    srv.on("POST", "/admin/config", handle_config_post);
    srv.on("GET", "/admin/files", handle_admin_files);
    srv.on("GET", "/admin/engine/status", handle_engine_status);
    srv.on("POST", "/admin/engine/start", handle_engine_start);
    srv.on("POST", "/admin/engine/stop", handle_engine_stop);
    srv.on("POST", "/v1/completions", handle_completions);
    srv.on("POST", "/v1/chat/completions", handle_chat);
    srv.on("POST", "/v1/responses", handle_responses);
    srv.on("POST", "/v1/messages", handle_messages);
    srv.on("POST", "/v1/messages/count_tokens", handle_messages_count);
    srv.on("POST", "/v1/tokenize", handle_tokenize);
    srv.on("POST", "/v1/detokenize", handle_detokenize);

#ifdef _WIN32
    if (g_tray_mode) {
        // 托盘模式：worker 线程跑 HTTP 服务，主线程跑托盘消息循环
        TrayHooks hooks;
        hooks.api_port = srv.port();
        hooks.api_host = g_cfg.listen.substr(0, g_cfg.listen.rfind(':'));
        hooks.api_log = g_api_log_abs;
        hooks.conf_snapshot = [] {
            std::lock_guard<std::mutex> lk(g_svc_conf_mtx);
            svcconf::Conf c = g_svc_conf;
            c[enginesup::kWeightsVerKey] = svcconf::conf_marker(g_conf_path, kWeightsMarker);
            return c;
        };
        hooks.engine_endpoint = [] {
            return std::make_pair(eff_engine_host(), eff_engine_port());
        };
        return tray_run(srv, hooks);
    }
#endif
    srv.run();
    return 0;
}
