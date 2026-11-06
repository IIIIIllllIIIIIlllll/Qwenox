// gguf.h — read-only GGUF (v2/v3) loader: multi-shard mmap, KV table, tensor views.
//
// Layout facts used by the engine (llama.cpp convention):
//   * dims are ne[0..nd): ne[0] is the contiguous (input / K) axis, so a 2D
//     weight [ne0=K, ne1=N] is N rows of K elements, i.e. HF [out, in].
//     3D expert tensors are [K, N, E]: expert e's rows start at
//     e * N * row_bytes(K).
//   * tensor data offsets are relative to the shard's data section, which
//     starts at the header end rounded up to general.alignment (default 32).
//   * shards: NAME-0000i-of-0000n.gguf; tensors are unique across shards.
//     Only shard 1 carries the full KV table (others hold split.* keys).
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include "gpu/os_win32.h"
#endif
#include "gguf_iq3s_grid.h"
#include "gguf_iq2xxs_iq2xs_grid.h"
#include "gguf_iq3xxs_iq2s_grid.h"

namespace gguf {

enum Type : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
  Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
  IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19,
  IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27,
  F64 = 28, IQ1_M = 29, BF16 = 30, MXFP4 = 39,
  Q2_0 = 42,
};

// (elements per block, bytes per block); {0,0} = unsupported.
inline void block_info(uint32_t t, uint32_t& be, uint32_t& bb) {
  switch (t) {
    case F32: be = 1; bb = 4; return;
    case F16: case BF16: be = 1; bb = 2; return;
    case I8: be = 1; bb = 1; return;
    case I16: be = 1; bb = 2; return;
    case I32: be = 1; bb = 4; return;
    case I64: case F64: be = 1; bb = 8; return;
    case Q4_0: be = 32; bb = 18; return;
    case Q4_1: be = 32; bb = 20; return;
    case Q5_0: be = 32; bb = 22; return;
    case Q5_1: be = 32; bb = 24; return;
    case Q8_0: be = 32; bb = 34; return;
    case IQ4_NL: be = 32; bb = 18; return;
    case Q2_K: be = 256; bb = 84; return;
    case Q3_K: be = 256; bb = 110; return;
    case Q4_K: be = 256; bb = 144; return;
    case Q5_K: be = 256; bb = 176; return;
    case Q6_K: be = 256; bb = 210; return;
    case IQ2_XXS: be = 256; bb = 66; return;
    case IQ2_XS: be = 256; bb = 74; return;
    case IQ4_XS: be = 256; bb = 136; return;
    case IQ3_S: be = 256; bb = 110; return;
    case IQ3_XXS: be = 256; bb = 98; return;
    case IQ2_S: be = 256; bb = 82; return;
    case Q2_0: be = 64; bb = 18; return;
    default: be = 0; bb = 0; return;
  }
}

inline const char* type_name(uint32_t t) {
  switch (t) {
    case F32: return "F32"; case F16: return "F16"; case BF16: return "BF16";
    case Q4_0: return "Q4_0"; case Q4_1: return "Q4_1"; case Q5_0: return "Q5_0";
    case Q5_1: return "Q5_1"; case Q8_0: return "Q8_0"; case Q2_K: return "Q2_K";
    case Q3_K: return "Q3_K"; case Q4_K: return "Q4_K"; case Q5_K: return "Q5_K";
    case Q6_K: return "Q6_K"; case IQ4_NL: return "IQ4_NL"; case IQ4_XS: return "IQ4_XS";
    case IQ3_S: return "IQ3_S";
    case IQ2_XXS: return "IQ2_XXS"; case IQ2_XS: return "IQ2_XS"; case IQ3_XXS: return "IQ3_XXS";
    case IQ1_S: return "IQ1_S"; case IQ2_S: return "IQ2_S"; case IQ1_M: return "IQ1_M";
    case MXFP4: return "MXFP4";
    case Q2_0: return "Q2_0";
    case I32: return "I32"; case I64: return "I64";
    default: return "?";
  }
}

// Bytes of one row of `k` elements (k must be a multiple of the block size).
inline uint64_t row_bytes(uint32_t t, uint64_t k) {
  uint32_t be, bb;
  block_info(t, be, bb);
  if (!be) throw std::runtime_error(std::string("gguf: unsupported type ") + type_name(t));
  if (k % be) throw std::runtime_error("gguf: row size not block-aligned");
  return k / be * bb;
}

struct Tensor {
  std::string name;
  uint32_t type = 0, nd = 0;
  uint64_t ne[4] = {1, 1, 1, 1};
  const uint8_t* data = nullptr;
  uint64_t nbytes = 0;
  int shard = 0;
  uint64_t file_off = 0;  // absolute offset inside the shard file
  uint64_t numel() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
  uint64_t rows() const { return ne[1] * ne[2] * ne[3]; }
  uint64_t row_bytes() const { return gguf::row_bytes(type, ne[0]); }
};

// KV value: scalars widened to int64/double; arrays keep numeric elements
// (widened) or strings. `type` is the GGUF value type of the scalar or of the
// array element; `is_array` distinguishes the two.
struct Value {
  uint32_t type = 0;
  bool is_array = false;
  int64_t i = 0;
  double f = 0;
  std::string s;
  std::vector<int64_t> ai;
  std::vector<double> af;
  std::vector<std::string> as;
};

class File {
 public:
  struct Mapping {
    const uint8_t* base = nullptr;
    size_t len = 0;
    std::string path;
    int fd = -1;
    // Windows: the file handle stays open (OVERLAPPED | NO_BUFFERING, like
    // hgn.h) so the weight loader can ReadFile shard ranges straight into
    // pinned staging, bypassing the page cache (32 GiB host RAM).
    void* os_handle = nullptr;
  };

  File() = default;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  explicit File(const std::string& path) { open(path); }
  ~File() {
#ifndef _WIN32
    for (auto& m : maps_) {
      if (m.base) munmap((void*)m.base, m.len);
      if (m.fd >= 0) ::close(m.fd);
    }
#else
    for (auto& m : maps_) {
      if (m.base) UnmapViewOfFile((void*)m.base);
      if (m.os_handle) CloseHandle((HANDLE)m.os_handle);
    }
#endif
  }

  // Opens `path`; if it matches *-0000i-of-0000n.gguf all n shards are opened.
  void open(const std::string& path) {
    std::vector<std::string> files{path};
    size_t p = path.rfind("-of-");
    if (p != std::string::npos && p >= 6 && path.size() == p + 4 + 5 + 5 &&
        path.compare(path.size() - 5, 5, ".gguf") == 0) {
      int n = atoi(path.substr(p + 4, 5).c_str());
      std::string stem = path.substr(0, p - 5);
      files.clear();
      for (int i = 1; i <= n; i++) {
        char buf[32];
        snprintf(buf, sizeof buf, "%05d-of-%05d.gguf", i, n);
        files.push_back(stem + buf);
      }
    }
    for (size_t i = 0; i < files.size(); i++) map_shard(files[i], (int)i);
  }

  const Tensor* find(const std::string& n) const {
    auto it = index_.find(n);
    return it == index_.end() ? nullptr : &it->second;
  }
  const Tensor& at(const std::string& n) const {
    auto* t = find(n);
    if (!t) throw std::runtime_error("gguf: missing tensor " + n);
    return *t;
  }
  const Value* kv(const std::string& k) const {
    auto it = kv_.find(k);
    return it == kv_.end() ? nullptr : &it->second;
  }
  int64_t kv_i(const std::string& k, int64_t def) const {
    auto* v = kv(k);
    if (!v || v->is_array) return def;
    return (v->type == 6 || v->type == 12) ? (int64_t)v->f : v->i;
  }
  double kv_f(const std::string& k, double def) const {
    auto* v = kv(k);
    if (!v || v->is_array) return def;
    return (v->type == 6 || v->type == 12) ? v->f : (double)v->i;
  }
  std::string kv_s(const std::string& k, const std::string& def = "") const {
    auto* v = kv(k);
    return (v && !v->is_array && v->type == 8) ? v->s : def;
  }
  const std::unordered_map<std::string, Tensor>& tensors() const { return index_; }
  const std::vector<std::string>& order() const { return order_; }
  const std::unordered_map<std::string, Value>& kvs() const { return kv_; }
  const std::vector<Mapping>& mappings() const { return maps_; }

  // Drop the mapped pages this process has touched from its working set
  // (Windows; no-op elsewhere). After load every weight lives in the device
  // arena, but the pages read through the mapping (gguf_map's dense / norm
  // conversion: ~5 GiB for UD-IQ4_XS) stay resident and show as "in use" in
  // Task Manager. VirtualUnlock on unlocked pages removes them from the
  // working set (documented behaviour; it reports ERROR_NOT_LOCKED): they move
  // to the standby list -- still cached, reclaimable, and soft-faulted back if
  // touched again (the PLE table is read with ReadFile, not through the map).
  // Returns the mapped bytes covered.
  size_t trim_resident() const {
    size_t n = 0;
#ifdef _WIN32
    for (const auto& m : maps_)
      if (m.base) {
        VirtualUnlock((LPVOID)m.base, m.len);
        n += m.len;
      }
#endif
    return n;
  }
  std::string arch() const { return kv_s("general.architecture"); }

 private:
  std::vector<Mapping> maps_;
  std::unordered_map<std::string, Tensor> index_;
  std::vector<std::string> order_;
  std::unordered_map<std::string, Value> kv_;

  struct Cur {
    const uint8_t* p;
    const uint8_t* end;
    template <class T>
    T get() {
      if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated header");
      T v;
      memcpy(&v, p, sizeof(T));
      p += sizeof(T);
      return v;
    }
    std::string str() {
      uint64_t n = get<uint64_t>();
      if (p + n > end) throw std::runtime_error("gguf: truncated string");
      std::string s((const char*)p, n);
      p += n;
      return s;
    }
  };

  static void scalar(Cur& c, uint32_t t, int64_t& i, double& f, std::string* s) {
    switch (t) {
      case 0: i = c.get<uint8_t>(); return;
      case 1: i = c.get<int8_t>(); return;
      case 2: i = c.get<uint16_t>(); return;
      case 3: i = c.get<int16_t>(); return;
      case 4: i = c.get<uint32_t>(); return;
      case 5: i = c.get<int32_t>(); return;
      case 6: f = c.get<float>(); return;
      case 7: i = c.get<uint8_t>(); return;
      case 8: if (s) *s = c.str(); else c.str(); return;
      case 10: i = (int64_t)c.get<uint64_t>(); return;
      case 11: i = c.get<int64_t>(); return;
      case 12: f = c.get<double>(); return;
      default: throw std::runtime_error("gguf: bad kv type " + std::to_string(t));
    }
  }

  void map_shard(const std::string& path, int shard) {
    Mapping m;
    m.path = path;
#ifdef _WIN32
    HANDLE fh = CreateFileA(path.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING, nullptr);
    if (fh == INVALID_HANDLE_VALUE) throw std::runtime_error("gguf: open " + path);
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz)) {
      CloseHandle(fh);
      throw std::runtime_error("gguf: GetFileSizeEx " + path);
    }
    m.len = (size_t)sz.QuadPart;
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mh) {
      m.base = (const uint8_t*)MapViewOfFile(mh, FILE_MAP_READ, 0, 0, m.len);
      CloseHandle(mh);
    }
    if (!m.base) {
      CloseHandle(fh);
      throw std::runtime_error("gguf: map " + path);
    }
    m.os_handle = (void*)fh;
#else
    m.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (m.fd < 0) throw std::runtime_error("gguf: open " + path + ": " + strerror(errno));
    struct stat st;
    if (fstat(m.fd, &st) < 0) throw std::runtime_error("gguf: fstat");
    m.len = (size_t)st.st_size;
    void* p = mmap(nullptr, m.len, PROT_READ, MAP_PRIVATE, m.fd, 0);
    if (p == MAP_FAILED) throw std::runtime_error("gguf: mmap " + path);
    m.base = (const uint8_t*)p;
#endif
    maps_.push_back(m);
    Cur c{m.base, m.base + m.len};
    if (memcmp(c.p, "GGUF", 4) != 0) throw std::runtime_error("gguf: bad magic " + path);
    c.p += 4;
    uint32_t ver = c.get<uint32_t>();
    if (ver < 2) throw std::runtime_error("gguf: version < 2 unsupported");
    uint64_t nt = c.get<uint64_t>(), nkv = c.get<uint64_t>();
    for (uint64_t i = 0; i < nkv; i++) {
      std::string k = c.str();
      Value v;
      v.type = c.get<uint32_t>();
      if (v.type == 9) {
        v.is_array = true;
        v.type = c.get<uint32_t>();
        uint64_t n = c.get<uint64_t>();
        // Large string arrays (tokenizer vocab/merges) are kept too: the
        // tokenizer may be loaded from here later. ~250K strings is fine.
        for (uint64_t j = 0; j < n; j++) {
          int64_t iv = 0;
          double fv = 0;
          if (v.type == 8) {
            v.as.push_back(c.str());
          } else {
            scalar(c, v.type, iv, fv, nullptr);
            if (v.type == 6 || v.type == 12) v.af.push_back(fv);
            else v.ai.push_back(iv);
          }
        }
      } else {
        scalar(c, v.type, v.i, v.f, &v.s);
      }
      if (shard == 0 || !kv_.count(k)) kv_[k] = std::move(v);
    }
    uint64_t align = 32;
    if (auto it = kv_.find("general.alignment"); it != kv_.end() && it->second.i > 0)
      align = (uint64_t)it->second.i;
    std::vector<Tensor> ts;
    std::vector<uint64_t> offs;
    for (uint64_t i = 0; i < nt; i++) {
      Tensor t;
      t.name = c.str();
      t.nd = c.get<uint32_t>();
      if (t.nd > 4) throw std::runtime_error("gguf: nd > 4 for " + t.name);
      for (uint32_t d = 0; d < t.nd; d++) t.ne[d] = c.get<uint64_t>();
      t.type = c.get<uint32_t>();
      offs.push_back(c.get<uint64_t>());
      t.shard = shard;
      ts.push_back(std::move(t));
    }
    uint64_t hdr = (uint64_t)(c.p - m.base);
    uint64_t base = (hdr + align - 1) / align * align;
    for (size_t i = 0; i < ts.size(); i++) {
      Tensor& t = ts[i];
      uint32_t be, bb;
      block_info(t.type, be, bb);
      t.file_off = base + offs[i];
      // Unknown block types are kept as zero-size views so mixed files can be
      // opened and their supported tensors used; row_bytes/dequant_row still
      // reject them with the type-specific error when something touches them.
      if (!be) {
        if (t.file_off > m.len) throw std::runtime_error("gguf: tensor past EOF: " + t.name);
        t.nbytes = 0;
        t.data = m.base + t.file_off;
        if (index_.count(t.name)) throw std::runtime_error("gguf: duplicate " + t.name);
        order_.push_back(t.name);
        index_.emplace(t.name, std::move(t));
        continue;
      }
      t.nbytes = t.numel() / be * bb;
      if (t.file_off + t.nbytes > m.len)
        throw std::runtime_error("gguf: tensor past EOF: " + t.name);
      t.data = m.base + t.file_off;
      if (index_.count(t.name)) throw std::runtime_error("gguf: duplicate " + t.name);
      order_.push_back(t.name);
      index_.emplace(t.name, std::move(t));
    }
  }
};

// ---- CPU reference dequant (llama.cpp ggml-quants.c semantics) ------------
inline float h2f(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v;
  if (e == 0) v = (float)m * 0x1p-24f;
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = ldexpf((float)(1024 + m), (int)e - 25);
  return s ? -v : v;
}

// K-quant 6-bit scale/min unpack (get_scale_min_k4).
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Dequantize one row of `k` elements of type `t` starting at `src`.
inline void dequant_row(uint32_t t, const uint8_t* src, float* y, uint64_t k) {
  static const uint8_t signs_tab[128] = GGUF_KSIGNS_IQ2XS_INIT;
  if (t == F32) { memcpy(y, src, k * 4); return; }
  if (t == F16) {
    for (uint64_t i = 0; i < k; i++) { uint16_t h; memcpy(&h, src + 2 * i, 2); y[i] = h2f(h); }
    return;
  }
  if (t == BF16) {
    for (uint64_t i = 0; i < k; i++) {
      uint16_t h; memcpy(&h, src + 2 * i, 2);
      uint32_t u = (uint32_t)h << 16; memcpy(&y[i], &u, 4);
    }
    return;
  }
  if (t == Q8_0) {
    for (uint64_t b = 0; b < k / 32; b++, src += 34) {
      uint16_t dh; memcpy(&dh, src, 2);
      float d = h2f(dh);
      for (int j = 0; j < 32; j++) y[b * 32 + j] = d * (float)(int8_t)src[2 + j];
    }
    return;
  }
  if (t == Q5_0) {
    for (uint64_t b = 0; b < k / 32; b++, src += 22) {
      uint16_t dh; memcpy(&dh, src, 2);
      uint32_t qh; memcpy(&qh, src + 2, 4);
      float d = h2f(dh);
      const uint8_t* qs = src + 6;
      for (int j = 0; j < 16; j++) {
        int x0 = (qs[j] & 0xF) | (((qh >> j) & 1) << 4);
        int x1 = (qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
        y[b * 32 + j] = (x0 - 16) * d;
        y[b * 32 + j + 16] = (x1 - 16) * d;
      }
    }
    return;
  }
  if (t == Q5_1) {
    for (uint64_t b = 0; b < k / 32; b++, src += 24) {
      uint16_t dh, mh; memcpy(&dh, src, 2); memcpy(&mh, src + 2, 2);
      uint32_t qh; memcpy(&qh, src + 4, 4);
      float d = h2f(dh), mn = h2f(mh);
      const uint8_t* qs = src + 8;
      for (int j = 0; j < 16; j++) {
        int x0 = (qs[j] & 0xF) | (((qh >> j) & 1) << 4);
        int x1 = (qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
        y[b * 32 + j] = x0 * d + mn;
        y[b * 32 + j + 16] = x1 * d + mn;
      }
    }
    return;
  }
  if (t == IQ4_NL) {
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1, 13, 25, 38, 53, 69, 89, 113};
    for (uint64_t b = 0; b < k / 32; b++, src += 18) {
      uint16_t dh; memcpy(&dh, src, 2);
      float d = h2f(dh);
      for (int j = 0; j < 16; j++) {
        y[b * 32 + j] = d * kv[src[2 + j] & 0xF];
        y[b * 32 + j + 16] = d * kv[src[2 + j] >> 4];
      }
    }
    return;
  }
  if (t == Q2_0) {  // 18 B / 64: d, qs[16] (4 x 2-bit); q=0..3 -> (-1,0,1,2)*d
    for (uint64_t b = 0; b < k / 64; b++, src += 18) {
      uint16_t dh; memcpy(&dh, src, 2);
      const float d = h2f(dh);
      for (int j = 0; j < 64; j++) {
        const int q = (src[2 + j / 4] >> (2 * (j % 4))) & 3;
        y[b * 64 + j] = (float)(q - 1) * d;
      }
    }
    return;
  }
  if (t == IQ4_XS) {  // 136 B / 256: d, scales_h, scales_l[4], qs[128]
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1, 13, 25, 38, 53, 69, 89, 113};
    for (uint64_t b = 0; b < k / 256; b++, src += 136) {
      uint16_t dh, sh; memcpy(&dh, src, 2); memcpy(&sh, src + 2, 2);
      const float d = h2f(dh);
      const uint8_t* sl = src + 4;
      const uint8_t* qs = src + 8;
      for (int ib = 0; ib < 8; ib++, qs += 16) {
        const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        float* yy = y + b * 256 + ib * 32;
        for (int j = 0; j < 16; j++) {
          yy[j] = dl * kv[qs[j] & 0xF];
          yy[j + 16] = dl * kv[qs[j] >> 4];
        }
      }
    }
    return;
  }
  if (t == IQ3_S) {  // 110 B / 256: d, qs[64], qh[8], signs[32], scales[4]
    static const uint32_t grid[512] = GGUF_IQ3S_GRID_INIT;
    for (uint64_t b = 0; b < k / 256; b++, src += 110) {
      uint16_t dh; memcpy(&dh, src, 2);
      const float d = h2f(dh);
      const uint8_t* qs = src + 2;
      const uint8_t* qh = src + 66;
      const uint8_t* sg = src + 74;
      const uint8_t* sc = src + 106;
      float* yy = y + b * 256;
      for (int ib = 0; ib < 8; ib++) {  // 32-element sub-block: 8 grid entries
        const float db = d * (float)(1 + 2 * ((sc[ib / 2] >> (4 * (ib & 1))) & 0xF));
        const uint8_t h = qh[ib];
        for (int l = 0; l < 4; l++) {
          const uint32_t g1 = grid[qs[ib * 8 + 2 * l] | ((h << (8 - 2 * l)) & 256)];
          const uint32_t g2 = grid[qs[ib * 8 + 2 * l + 1] | ((h << (7 - 2 * l)) & 256)];
          const uint8_t s = sg[ib * 4 + l];
          for (int j = 0; j < 4; j++) {
            const float m1 = (float)((g1 >> (8 * j)) & 0xFF), m2 = (float)((g2 >> (8 * j)) & 0xFF);
            yy[ib * 32 + l * 8 + j] = db * m1 * ((s >> j) & 1 ? -1.f : 1.f);
            yy[ib * 32 + l * 8 + j + 4] = db * m2 * ((s >> (j + 4)) & 1 ? -1.f : 1.f);
          }
        }
      }
    }
    return;
  }
  if (t == IQ2_XXS) {  // 66 B / 256: d, qs[32] (grid indices + scales/signs)
    static const uint64_t grid[256] = GGUF_IQ2XXS_GRID_INIT;
    for (uint64_t b = 0; b < k / 256; b++, src += 66) {
      uint16_t dh; memcpy(&dh, src, 2);
      const float d = h2f(dh);
      const uint8_t* qs = src + 2;
      float* yy = y + b * 256;
      for (int ib = 0; ib < 8; ib++) {  // 32 elements: one scale, 4 grids/signs
        uint32_t aux32[2]; memcpy(aux32, qs + 8 * ib, sizeof(aux32));
        const float db = d * (0.5f + (float)(aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; l++) {
          const uint32_t code = (aux32[l / 4] >> (8 * (l & 3))) & 0xFF;
          const uint64_t g = grid[code];
          const uint8_t signs = signs_tab[(aux32[1] >> (7 * l)) & 127];
          for (int j = 0; j < 8; j++)
            yy[ib * 32 + l * 8 + j] = db * (float)((g >> (8 * j)) & 0xFF) *
                                      ((signs >> j) & 1 ? -1.f : 1.f);
        }
      }
    }
    return;
  }
  if (t == IQ2_XS) {  // 74 B / 256: d, qs[32] (grid indices + signs), scales[8]
    static const uint64_t grid[512] = GGUF_IQ2XS_GRID_INIT;
    for (uint64_t b = 0; b < k / 256; b++, src += 74) {
      uint16_t dh; memcpy(&dh, src, 2);
      const float d = h2f(dh);
      const uint8_t* qs = src + 2;
      const uint8_t* sc = src + 66;
      float* yy = y + b * 256;
      for (int ib = 0; ib < 8; ib++) {  // Two 16-element scales per 32 elements
        const float db0 = d * (0.5f + (float)(sc[ib] & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
          uint16_t q; memcpy(&q, qs + 2 * (4 * ib + l), sizeof(q));
          const uint64_t g = grid[q & 511];
          const uint8_t signs = signs_tab[q >> 9];
          const float dl = l < 2 ? db0 : db1;
          for (int j = 0; j < 8; j++)
            yy[ib * 32 + l * 8 + j] = dl * (float)((g >> (8 * j)) & 0xFF) *
                                      ((signs >> j) & 1 ? -1.f : 1.f);
        }
      }
    }
    return;
  }
  if (t == IQ3_XXS) {  // 98 B / 256: d, qs[64], scales_and_signs[32]
    static const uint32_t grid[256] = GGUF_IQ3XXS_GRID_INIT;
    for (uint64_t b = 0; b < k / 256; b++, src += 98) {
      uint16_t dh; memcpy(&dh, src, 2);
      const float d = h2f(dh);
      const uint8_t* qs = src + 2;
      const uint8_t* ss = src + 66;
      float* yy = y + b * 256;
      for (int ib = 0; ib < 8; ib++) {  // 32 elements: 8 grid words + 4 sign bytes
        uint32_t aux; memcpy(&aux, ss + 4 * ib, 4);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        for (int l = 0; l < 4; l++) {
          const uint8_t signs = signs_tab[(aux >> (7 * l)) & 127];
          const uint32_t g1 = grid[qs[8 * ib + 2 * l + 0]];
          const uint32_t g2 = grid[qs[8 * ib + 2 * l + 1]];
          for (int j = 0; j < 4; j++) {
            const float m1 = (float)((g1 >> (8 * j)) & 0xFF), m2 = (float)((g2 >> (8 * j)) & 0xFF);
            yy[ib * 32 + l * 8 + j] = db * m1 * ((signs >> j) & 1 ? -1.f : 1.f);
            yy[ib * 32 + l * 8 + j + 4] = db * m2 * ((signs >> (j + 4)) & 1 ? -1.f : 1.f);
          }
        }
      }
    }
    return;
  }
  if (t == IQ2_S) {  // 82 B / 256: d, qs[64], qh[8], scales[8]
    static const uint64_t grid[1024] = GGUF_IQ2S_GRID_INIT;
    for (uint64_t b = 0; b < k / 256; b++, src += 82) {
      uint16_t dh; memcpy(&dh, src, 2);
      const float d = h2f(dh);
      const uint8_t* qs = src + 2;
      const uint8_t* qh = src + 66;
      const uint8_t* sc = src + 74;
      const uint8_t* signs = qs + 32;
      float* yy = y + b * 256;
      for (int ib = 0; ib < 8; ib++) {
        const float db0 = d * (0.5f + (float)(sc[ib] & 0xF)) * 0.25f;
        const float db1 = d * (0.5f + (float)(sc[ib] >> 4)) * 0.25f;
        for (int l = 0; l < 4; l++) {
          const uint64_t g = grid[qs[4 * ib + l] | ((qh[ib] << (8 - 2 * l)) & 0x300)];
          const uint8_t s = signs[4 * ib + l];
          const float dl = l < 2 ? db0 : db1;
          for (int j = 0; j < 8; j++)
            yy[ib * 32 + l * 8 + j] = dl * (float)((g >> (8 * j)) & 0xFF) *
                                      ((s >> j) & 1 ? -1.f : 1.f);
        }
      }
    }
    return;
  }
  if (t == Q6_K) {  // 210 B / 256: ql[128], qh[64], scales[16] (int8), d
    for (uint64_t b = 0; b < k / 256; b++, src += 210) {
      uint16_t dh; memcpy(&dh, src + 208, 2);
      const float d = h2f(dh);
      const uint8_t* ql = src;
      const uint8_t* qh = src + 128;
      const int8_t* sc = (const int8_t*)(src + 192);
      float* yy = y + b * 256;
      for (int n = 0; n < 256; n += 128, yy += 128, ql += 64, qh += 32, sc += 8) {
        for (int l = 0; l < 32; l++) {
          const int is = l / 16;
          const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
          const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
          const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
          const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
          yy[l] = d * sc[is] * q1;
          yy[l + 32] = d * sc[is + 2] * q2;
          yy[l + 64] = d * sc[is + 4] * q3;
          yy[l + 96] = d * sc[is + 6] * q4;
        }
      }
    }
    return;
  }
  if (t == Q4_K || t == Q5_K) {
    const uint64_t bs = t == Q4_K ? 144 : 176;
    for (uint64_t b = 0; b < k / 256; b++, src += bs) {
      uint16_t dh, mh; memcpy(&dh, src, 2); memcpy(&mh, src + 2, 2);
      float d = h2f(dh), dmin = h2f(mh);
      const uint8_t* sc = src + 4;
      const uint8_t* qh = t == Q5_K ? src + 16 : nullptr;
      const uint8_t* ql = src + (t == Q5_K ? 48 : 16);
      float* yy = y + b * 256;
      int is = 0;
      uint8_t u1 = 1, u2 = 2;
      for (int j = 0; j < 256; j += 64) {
        uint8_t s0, m0, s1, m1;
        scale_min_k4(is + 0, sc, s0, m0);
        scale_min_k4(is + 1, sc, s1, m1);
        float d1 = d * s0, dm1 = dmin * m0, d2 = d * s1, dm2 = dmin * m1;
        for (int l = 0; l < 32; l++) {
          int q = ql[l] & 0xF;
          if (qh) q += (qh[l] & u1) ? 16 : 0;
          *yy++ = d1 * q - dm1;
        }
        for (int l = 0; l < 32; l++) {
          int q = ql[l] >> 4;
          if (qh) q += (qh[l] & u2) ? 16 : 0;
          *yy++ = d2 * q - dm2;
        }
        ql += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
      }
    }
    return;
  }
  throw std::runtime_error(std::string("gguf: dequant_row: unsupported type ") + type_name(t));
}

}  // namespace gguf
