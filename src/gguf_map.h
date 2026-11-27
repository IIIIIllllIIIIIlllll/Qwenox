// gguf_map.h — build the engine's (hgn-named) non-expert tensors from a
// llama.cpp GGUF (qwen4exp arch, e.g. Unsloth Qwen3.8-Flash-Next UD-Q4_K_XL)
// plus the optional MTP sidecar (mtp-*-Q8_0.gguf: blk.48.* + blk.48.nextn.*).
//
// The result is a set of synthetic hgn::Tensor records (hgn::Checkpoint::
// add_synthetic) under the exact hgn names, so the loader, wview/gemv/gemm
// and the small-tensor uploads keep working unchanged:
//   * Q8_0 matrices  -> dtype 8 q8g32 planar (lossless repack)
//   * F32 matrices   -> dtype 0 bf16 (router gate, ssm_alpha/beta, shexp gate,
//                        HC inject; also the Q8_0 MTP inject)
//   * F32 vectors    -> dtype 1 f32 (norms, conv1d, A_log, dt_bias)
//   * BF16 indexer q_proj(512)+k_proj(128) -> one bf16 index_qk_proj(640)
//   * PLE config KV arrays -> dtype 4 u64
// Routed experts are NOT built here (read in place by 26_kernels_moe_gguf.inc);
// the PLE n-gram table is a borrowed view (add_ple_table, dtype 11 IQ4_NL or
// dtype 12 Q5_0).
//
// GGUF vs HF/engine transforms (semantics from gufo reference.cpp, MIT;
// verified numerically against hgn by tools/g3_map_check.cpp):
//   * layout: GGUF ne[0] is the torch last dim -> no transposes.
//   * GDN value heads are "tiled" in GGUF (value head g reads key head g%16)
//     while HF / the engine read key head h/3; engine head h = GGUF head
//     (h%3)*16 + h/3. Applies to the v rows of attn_qkv, attn_gate (z),
//     ssm_alpha/beta, ssm_a, ssm_dt.bias, the v channels of ssm_conv1d and
//     the 128-col head blocks of ssm_out's input. ssm_norm (128) is shared.
//   * ssm_a = -exp(A_log)  ->  A_log = log(-ssm_a).
//   * zero-centred RMSNorm weights are stored with +1 folded in (llama.cpp
//     converter); the engine's k_rmsnorm_zc* want the raw w -> w - 1.
//   * MTP eh_proj [2560][5120] = [embedding cols | hidden cols].
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gguf.h"
#include "hgn.h"

namespace gguf_map {

struct Opts {
  bool vperm = true;  // un-tile GDN value heads
  // +1 folded into these GGUF norm kinds (subtract 1 to get the hgn/HF value)
  bool fold_hc = true, fold_qk = true, fold_idx = true, fold_ssm = false,
       fold_ple = true, fold_mtp = true;
  std::function<bool(int)> layer;  // trunk layer filter (empty = all)
  bool globals = true;             // embed / lm_head / output HC / PLE (non-table)
  bool mtp = true;                 // build mtp.* when an MTP file is given
};

using Map = std::vector<uint32_t>;

// Debug/bisect knob: when set, only hgn names it accepts are written (every
// other tensor keeps its hgn record). Each written tensor is individually
// converted to engine order, so any subset is self-consistent.
inline std::function<bool(const std::string&)> g_keep;
inline void put(hgn::Checkpoint& ck, const hgn::Tensor& d, std::vector<uint8_t>&& buf) {
  if (g_keep && !g_keep(d.name)) return;
  ck.add_synthetic(d, std::move(buf));
}

inline uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;
  u += 0x7fffu + ((u >> 16) & 1u);
  return (uint16_t)(u >> 16);
}

inline void par_for(uint64_t n, const std::function<void(uint64_t, uint64_t)>& f,
                    uint64_t min_per = 4096) {
  unsigned nt = std::thread::hardware_concurrency();
  nt = nt ? std::min(nt, 16u) : 4u;
  if (n < min_per * 2 || nt < 2) { f(0, n); return; }
  nt = (unsigned)std::min<uint64_t>(nt, n / min_per);
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nt; i++)
    th.emplace_back([&, i] { f(n * i / nt, n * (i + 1) / nt); });
  for (auto& t : th) t.join();
}

inline hgn::Tensor desc(const std::string& name, uint32_t dtype,
                        std::initializer_list<uint64_t> dims) {
  hgn::Tensor t;
  t.name = name;
  t.dtype = dtype;
  t.ndims = (uint32_t)dims.size();
  int i = 0;
  for (uint64_t d : dims) t.dims[i++] = d;
  return t;
}

inline void need(bool c, const std::string& what) {
  if (!c) throw std::runtime_error("gguf_map: " + what);
}

// f32 -> f16 bits, round to nearest even (subnormals, overflow -> inf)
inline uint16_t f2h(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u, a = x & 0x7fffffffu;
  if (a >= 0x7f800000u) return (uint16_t)(sign | (a > 0x7f800000u ? 0x7e00u : 0x7c00u));
  if (a >= 0x477ff000u) return (uint16_t)(sign | 0x7c00u);  // rounds to >= 65536
  if (a < 0x38800000u) {                                     // below 2^-14: subnormal / zero
    if (a < 0x33000000u) return (uint16_t)sign;              // below 2^-25 -> 0
    const uint32_t e = a >> 23, m = (a & 0x7fffffu) | 0x800000u, sh = 126 - e;
    uint32_t r = m >> sh;
    const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) r++;
    return (uint16_t)(sign | r);
  }
  const uint32_t r = a + 0x0fffu + ((a >> 13) & 1u);
  return (uint16_t)(sign | ((r >> 13) - (112u << 10)));
}

// Native dense (engine opt-in, default off here so tools keep q8g32): Q4_K /
// Q5_K / Q6_K / IQ4_XS / IQ4_NL matrices without a column-block map keep their
// GGUF rows (row map applied) as dtype 100 + GGML type instead of the q8g32
// requant (1.3-2.0x the bytes; decode GEMV is bandwidth-bound). The token
// embedding stays q8g32 (gather kernels). Read by 26_kernels_dense_gguf.inc.
inline bool g_native_dense = false;
// QWENOX_GGUF_DENSE_Q6=1: dense matrices that would be requantized to q8g32
// go to dtype 24 q6g64 instead (~0.76x the bytes; decode GEMV is
// bandwidth-bound). The token embedding stays q8g32 (gather kernels), MTP
// tensors keep q8g32. Consumed by the hgn v2 q6g64 kernels.
inline bool g_dense_q6 = false;
inline bool native_dense_ok(const std::string& name, const gguf::Tensor& t, const Map& blks) {
  if (!g_native_dense || !blks.empty() || name == "embed_tokens.weight") return false;
  if (!(t.type == gguf::Q4_K || t.type == gguf::Q5_K || t.type == gguf::Q6_K ||
        t.type == gguf::IQ4_XS || t.type == gguf::IQ4_NL))
    return false;
  uint32_t be, bb;
  gguf::block_info(t.type, be, bb);
  return be && t.ne[0] % be == 0 && t.row_bytes() % 4 == 0;  // GEMV stages dwords
}

// Any dense matrix -> dtype 24 q6g64: per 64-col group f16 scale + bias,
// W = code * scale + bias, 6-bit codes packed as C/2 B low nibbles + C/4 B
// high-2-bit plane; row stride align16(13 C / 16) (k_q6g64_* decode kernels).
// Encoding mirrors tools/v2repro/v2enc.py q6g64_encode: scale candidates
// (hi - lo) / 63 * {1.0 .. 0.85} in f16, least squared error per group.
inline uint64_t q6g64_st(uint64_t C) { return (13 * C / 16 + 15) & ~15ull; }
inline void add_q6(hgn::Checkpoint& ck, const std::string& name, const gguf::Tensor& t,
                   std::initializer_list<uint64_t> dims, const Map& rows = {},
                   const Map& blks = {}) {
  uint32_t be, bb;
  gguf::block_info(t.type, be, bb);
  need(be != 0, name + " <- " + t.name + ": unsupported type " + gguf::type_name(t.type));
  const uint64_t C = t.ne[0], R = t.rows(), nbs = C / 32, rb = t.row_bytes();
  need(C % 32 == 0, name + ": cols not a multiple of 32");
  const uint64_t Ro = rows.empty() ? R : rows.size();
  const uint64_t nbo = blks.empty() ? nbs : blks.size(), Co = nbo * 32;
  need(Co % 64 == 0, name + ": cols not a multiple of 64");
  hgn::Tensor d = desc(name, 24, dims);
  need(d.numel() == Ro * Co, name + ": shape mismatch vs " + t.name);
  for (uint32_t r : rows) need(r < R, name + ": row map out of range");
  for (uint32_t b : blks) need(b < nbs, name + ": block map out of range");
  const uint64_t st = q6g64_st(Co), ng = Co / 64;
  std::vector<uint8_t> buf(Ro * st);
  static const float mults[6] = {1.0f, 0.97f, 0.94f, 0.91f, 0.88f, 0.85f};
  par_for(Ro, [&](uint64_t r0, uint64_t r1) {
    std::vector<float> f(Co);
    for (uint64_t r = r0; r < r1; r++) {
      if (blks.empty()) {
        gguf::dequant_row(t.type, t.data + (uint64_t)(rows.empty() ? r : rows[r]) * rb,
                          f.data(), C);
      } else {
        std::vector<float> tmp(C);
        gguf::dequant_row(t.type, t.data + (uint64_t)(rows.empty() ? r : rows[r]) * rb,
                          tmp.data(), C);
        for (uint64_t b = 0; b < nbo; b++)
          memcpy(f.data() + b * 32, tmp.data() + (uint64_t)blks[b] * 32, 32 * 4);
      }
      uint8_t* out = buf.data() + r * st;
      uint8_t* lo4 = out;                    // [0, Co/2)
      uint8_t* hi2 = out + Co / 2;           // [Co/2, 3Co/4)
      uint8_t* sb = out + 3 * Co / 4;        // ng x (f16 scale, f16 bias)
      for (uint64_t g = 0; g < ng; g++) {
        const float* v = f.data() + g * 64;
        float lo = v[0], hi = v[0];
        for (int j = 1; j < 64; j++) {
          lo = std::min(lo, v[j]);
          hi = std::max(hi, v[j]);
        }
        float best_e = 0;
        uint16_t best_s = 0, best_b = 0;
        uint8_t best_q[64];
        for (float m : mults) {
          const uint16_t sh = f2h((hi - lo) / 63.f * m);
          const uint16_t bh = f2h(lo);
          const float sf = gguf::h2f(sh), bf = gguf::h2f(bh);
          const float inv = sf > 0.f ? 1.f / sf : 0.f;
          float e = 0.f;
          uint8_t q[64];
          for (int j = 0; j < 64; j++) {
            const float c = std::max(0.f, std::min(63.f, std::nearbyint((v[j] - bf) * inv)));
            q[j] = (uint8_t)c;
            const float df = c * sf + bf - v[j];
            e += df * df;
          }
          if (m == mults[0] || e < best_e) {
            best_e = e;
            best_s = sh;
            best_b = bh;
            memcpy(best_q, q, 64);
          }
        }
        for (int j = 0; j < 32; j++)
          lo4[g * 32 + j] = (best_q[2 * j] & 15) | ((best_q[2 * j + 1] & 15) << 4);
        for (int j = 0; j < 16; j++)
          hi2[g * 16 + j] = (uint8_t)((best_q[4 * j] >> 4) | ((best_q[4 * j + 1] >> 4) << 2) |
                                      ((best_q[4 * j + 2] >> 4) << 4) |
                                      ((best_q[4 * j + 3] >> 4) << 6));
        memcpy(sb + g * 4, &best_s, 2);
        memcpy(sb + g * 4 + 2, &best_b, 2);
      }
    }
  }, 256);
  fprintf(stderr, "gguf_map: %s <- %s requantized %s -> q6g64\n", name.c_str(), t.name.c_str(),
          gguf::type_name(t.type));
  put(ck, d, std::move(buf));
}

// Q8_0 [R][C] -> q8g32 [R'][C'] taking source row rows[r] (identity if empty)
// and source 32-col block blks[b] (identity if empty). Other block types (the
// UD-IQ4_XS lm_head is Q6_K) are dequantized (ggml semantics) and requantized
// per 32: scale = f16(amax / 127), code = round(v / scale) — 8-bit codes on a
// 6-bit source, error <= 1/8 of the source quant step.
inline void add_q8(hgn::Checkpoint& ck, const std::string& name, const gguf::Tensor& t,
                   std::initializer_list<uint64_t> dims, const Map& rows = {},
                   const Map& blks = {}) {
  // Q6 dense opt-in: everything except the token embedding (gather kernels)
  // and the MTP head goes to q6g64 instead of q8g32. Full conversion measured
  // fastest end-to-end (30.46 ms/tok GPU vs 33.84 q8g32 vs 31.5 with the
  // hyper_connection/shared_expert/ple/kv tensors kept on q8g32, gfx1151
  // 2026-11-27): the q6g64 small-shape kernels' standalone-bench deficit
  // (>96 MB cycling) does not reproduce in-situ.
  if (g_dense_q6 && name != "embed_tokens.weight" && name.rfind("mtp.", 0) != 0) {
    uint32_t be, bb;
    gguf::block_info(t.type, be, bb);
    const uint64_t Co = (blks.empty() ? t.ne[0] / 32 : blks.size()) * 32;
    if (be != 0 && t.ne[0] % 32 == 0 && Co % 64 == 0) {
      add_q6(ck, name, t, dims, rows, blks);
      return;
    }
  }
  if (native_dense_ok(name, t, blks)) {
    const uint64_t C = t.ne[0], R = t.rows(), rb = t.row_bytes();
    const uint64_t Ro = rows.empty() ? R : rows.size();
    hgn::Tensor d = desc(name, 100 + t.type, dims);
    need(d.numel() == Ro * C, name + ": shape mismatch vs " + t.name);
    for (uint32_t r : rows) need(r < R, name + ": row map out of range");
    std::vector<uint8_t> buf(Ro * rb);
    par_for(Ro, [&](uint64_t r0, uint64_t r1) {
      for (uint64_t r = r0; r < r1; r++)
        memcpy(buf.data() + r * rb, t.data + (uint64_t)(rows.empty() ? r : rows[r]) * rb, rb);
    }, 256);
    put(ck, d, std::move(buf));
    return;
  }
  if (t.type != gguf::Q8_0) {
    uint32_t be, bb;
    gguf::block_info(t.type, be, bb);
    need(be != 0, name + " <- " + t.name + ": unsupported type " + gguf::type_name(t.type));
    const uint64_t C = t.ne[0], R = t.rows(), nbs = C / 32, rb = t.row_bytes();
    need(C % 32 == 0, name + ": cols not a multiple of 32");
    const uint64_t Ro = rows.empty() ? R : rows.size();
    const uint64_t nbo = blks.empty() ? nbs : blks.size(), Co = nbo * 32;
    hgn::Tensor d = desc(name, 8, dims);
    need(d.numel() == Ro * Co, name + ": shape mismatch vs " + t.name);
    for (uint32_t r : rows) need(r < R, name + ": row map out of range");
    for (uint32_t b : blks) need(b < nbs, name + ": block map out of range");
    std::vector<uint8_t> buf(Ro * Co + Ro * nbo * 2);
    int8_t* q = (int8_t*)buf.data();
    uint16_t* s = (uint16_t*)(buf.data() + Ro * Co);
    par_for(Ro, [&](uint64_t r0, uint64_t r1) {
      std::vector<float> f(C);
      for (uint64_t r = r0; r < r1; r++) {
        gguf::dequant_row(t.type, t.data + (uint64_t)(rows.empty() ? r : rows[r]) * rb, f.data(), C);
        for (uint64_t b = 0; b < nbo; b++) {
          const float* v = &f[(blks.empty() ? b : blks[b]) * 32];
          float amax = 0.f;
          for (int j = 0; j < 32; j++) amax = std::max(amax, std::fabs(v[j]));
          const uint16_t sh = f2h(amax / 127.f);
          s[r * nbo + b] = sh;
          const float sc = gguf::h2f(sh);
          const float inv = sc > 0.f ? 1.f / sc : 0.f;
          for (int j = 0; j < 32; j++) {
            const float c = std::nearbyint(v[j] * inv);
            q[r * Co + b * 32 + j] = (int8_t)std::max(-127.f, std::min(127.f, c));
          }
        }
      }
    }, 256);
    fprintf(stderr, "gguf_map: %s <- %s requantized %s -> q8g32\n", name.c_str(), t.name.c_str(),
            gguf::type_name(t.type));
    put(ck, d, std::move(buf));
    return;
  }
  const uint64_t C = t.ne[0], R = t.rows(), nbs = C / 32;
  const uint64_t Ro = rows.empty() ? R : rows.size();
  const uint64_t nbo = blks.empty() ? nbs : blks.size(), Co = nbo * 32;
  hgn::Tensor d = desc(name, 8, dims);
  need(d.numel() == Ro * Co, name + ": shape mismatch vs " + t.name);
  for (uint32_t r : rows) need(r < R, name + ": row map out of range");
  for (uint32_t b : blks) need(b < nbs, name + ": block map out of range");
  std::vector<uint8_t> buf(Ro * Co + Ro * nbo * 2);
  int8_t* q = (int8_t*)buf.data();
  uint16_t* s = (uint16_t*)(buf.data() + Ro * Co);
  par_for(Ro, [&](uint64_t r0, uint64_t r1) {
    for (uint64_t r = r0; r < r1; r++) {
      const uint8_t* src = t.data + (uint64_t)(rows.empty() ? r : rows[r]) * nbs * 34;
      for (uint64_t b = 0; b < nbo; b++) {
        const uint8_t* bk = src + (uint64_t)(blks.empty() ? b : blks[b]) * 34;
        memcpy(&s[r * nbo + b], bk, 2);
        memcpy(q + r * Co + b * 32, bk + 2, 32);
      }
    }
  }, 256);
  put(ck, d, std::move(buf));
}

// any GGUF type -> f32 [R'][C] with source row map
inline std::vector<float> rows_f32(const gguf::Tensor& t, const Map& rows = {}) {
  const uint64_t C = t.ne[0], R = t.rows(), rb = t.row_bytes();
  const uint64_t Ro = rows.empty() ? R : rows.size();
  for (uint32_t r : rows) need(r < R, t.name + ": row map out of range");
  std::vector<float> out(Ro * C);
  par_for(Ro, [&](uint64_t r0, uint64_t r1) {
    for (uint64_t r = r0; r < r1; r++)
      gguf::dequant_row(t.type, t.data + (uint64_t)(rows.empty() ? r : rows[r]) * rb,
                        &out[r * C], C);
  }, 64);
  return out;
}

inline void add_bf16v(hgn::Checkpoint& ck, const std::string& name, const std::vector<float>& f,
                      std::initializer_list<uint64_t> dims) {
  hgn::Tensor d = desc(name, 0, dims);
  need(d.numel() == f.size(), name + ": bf16 shape mismatch");
  std::vector<uint8_t> buf(f.size() * 2);
  uint16_t* p = (uint16_t*)buf.data();
  for (size_t i = 0; i < f.size(); i++) p[i] = f2bf(f[i]);
  put(ck, d, std::move(buf));
}
inline void add_bf16(hgn::Checkpoint& ck, const std::string& name, const gguf::Tensor& t,
                     std::initializer_list<uint64_t> dims, const Map& rows = {}) {
  add_bf16v(ck, name, rows_f32(t, rows), dims);
}
inline void add_f32v(hgn::Checkpoint& ck, const std::string& name, const std::vector<float>& f,
                     std::initializer_list<uint64_t> dims) {
  hgn::Tensor d = desc(name, 1, dims);
  need(d.numel() == f.size(), name + ": f32 shape mismatch");
  std::vector<uint8_t> buf(f.size() * 4);
  memcpy(buf.data(), f.data(), buf.size());
  put(ck, d, std::move(buf));
}
inline std::vector<float> norm_vec(const gguf::Tensor& t, bool fold) {
  std::vector<float> v = rows_f32(t);
  if (fold)
    for (float& x : v) x -= 1.f;
  return v;
}
inline void add_u64(hgn::Checkpoint& ck, const std::string& name, const gguf::File& g,
                    const std::string& key, size_t n) {
  const gguf::Value* v = g.kv(key);
  need(v && v->is_array && v->ai.size() == n, "KV " + key + " missing or wrong length");
  std::vector<uint8_t> buf(n * 8);
  for (size_t i = 0; i < n; i++) {
    uint64_t x = (uint64_t)v->ai[i];
    memcpy(buf.data() + i * 8, &x, 8);
  }
  put(ck, desc(name, 4, {(uint64_t)n}), std::move(buf));
}

// GDN head/row/channel maps (engine index -> GGUF index)
inline uint32_t vhead(const Opts& o, uint32_t h) { return o.vperm ? (h % 3) * 16 + h / 3 : h; }
inline Map map_heads(const Opts& o, uint32_t nh, uint32_t hd, uint32_t base_rows) {
  Map m;
  for (uint32_t r = 0; r < base_rows; r++) m.push_back(r);
  for (uint32_t h = 0; h < nh; h++)
    for (uint32_t d = 0; d < hd; d++) m.push_back(base_rows + vhead(o, h) * hd + d);
  return m;
}

// One transformer block. `g`/`bl`: GGUF file + block index; `P`: hgn prefix
// ("layers.7" / "mtp.layers.0").
inline void build_layer(hgn::Checkpoint& ck, const gguf::File& g, int bl, const std::string& P,
                        bool qsa, const Opts& o) {
  const std::string B = "blk." + std::to_string(bl) + ".";
  auto G = [&](const std::string& s) -> const gguf::Tensor& { return g.at(B + s); };
  const char* hk[2] = {"attn", "mlp"};
  const char* gk[2] = {"attn", "ffn"};
  for (int i = 0; i < 2; i++) {
    const std::string hp = P + "." + hk[i] + "_hyper_connection.";
    const std::string gp = std::string("hc_") + gk[i] + "_";
    add_q8(ck, hp + "input_mix_weight_down.weight", G(gp + "down.weight"), {320, 10240});
    add_q8(ck, hp + "input_mix_weight_up.weight", G(gp + "up.weight"), {10240, 320});
    add_bf16(ck, hp + "block_inject_weight.weight", G(gp + "inject.weight"), {4, 10240});
    add_f32v(ck, hp + "hc_norm.weight", norm_vec(G(gp + "norm.weight"), o.fold_hc), {10240});
  }
  add_bf16(ck, P + ".mlp.gate.weight", G("ffn_gate_inp.weight"), {512, 2560});
  add_bf16(ck, P + ".mlp.shared_expert_gate.weight", G("ffn_gate_inp_shexp.weight"), {1, 2560});
  add_q8(ck, P + ".mlp.shared_expert.gate_proj.weight", G("ffn_gate_shexp.weight"), {640, 2560});
  add_q8(ck, P + ".mlp.shared_expert.up_proj.weight", G("ffn_up_shexp.weight"), {640, 2560});
  add_q8(ck, P + ".mlp.shared_expert.down_proj.weight", G("ffn_down_shexp.weight"), {2560, 640});
  if (qsa) {
    const std::string A = P + ".self_attn.";
    add_q8(ck, A + "q_proj.weight", G("attn_q.weight"), {12288, 2560});
    add_q8(ck, A + "k_proj.weight", G("attn_k.weight"), {512, 2560});
    add_q8(ck, A + "v_proj.weight", G("attn_v.weight"), {512, 2560});
    add_q8(ck, A + "o_proj.weight", G("attn_output.weight"), {2560, 6144});
    add_f32v(ck, A + "q_norm.weight", norm_vec(G("attn_q_norm.weight"), o.fold_qk), {256});
    add_f32v(ck, A + "k_norm.weight", norm_vec(G("attn_k_norm.weight"), o.fold_qk), {256});
    std::vector<float> iq = rows_f32(G("indexer.q_proj.weight"));
    std::vector<float> ik = rows_f32(G("indexer.k_proj.weight"));
    need(iq.size() == 512 * 2560 && ik.size() == 128 * 2560, B + "indexer shapes");
    iq.insert(iq.end(), ik.begin(), ik.end());
    add_bf16v(ck, A + "indexer.index_qk_proj.weight", iq, {640, 2560});
    add_f32v(ck, A + "indexer.q_layernorm.weight", norm_vec(G("indexer.q_norm.weight"), o.fold_idx),
             {128});
    add_f32v(ck, A + "indexer.k_layernorm.weight", norm_vec(G("indexer.k_norm.weight"), o.fold_idx),
             {128});
    return;
  }
  const std::string L = P + ".linear_attn.";
  const Map mqkv = map_heads(o, 48, 128, 4096);  // q(2048) k(2048) | v 48x128
  const Map mz = map_heads(o, 48, 128, 0);
  const Map m48 = map_heads(o, 48, 1, 0);
  Map mout;  // ssm_out input cols: 48 heads x 4 Q8_0 blocks
  for (uint32_t h = 0; h < 48; h++)
    for (uint32_t j = 0; j < 4; j++) mout.push_back(vhead(o, h) * 4 + j);
  add_q8(ck, L + "in_proj_qkv.weight", G("attn_qkv.weight"), {10240, 2560}, mqkv);
  add_q8(ck, L + "in_proj_z.weight", G("attn_gate.weight"), {6144, 2560}, mz);
  add_bf16(ck, L + "in_proj_a.weight", G("ssm_alpha.weight"), {48, 2560}, m48);
  add_bf16(ck, L + "in_proj_b.weight", G("ssm_beta.weight"), {48, 2560}, m48);
  add_q8(ck, L + "out_proj.weight", G("ssm_out.weight"), {2560, 6144}, {}, mout);
  add_f32v(ck, L + "conv1d.weight", rows_f32(G("ssm_conv1d.weight"), mqkv), {10240, 1, 4});
  std::vector<float> a = rows_f32(G("ssm_a")), dt = rows_f32(G("ssm_dt.bias"));
  need(a.size() == 48 && dt.size() == 48, B + "ssm_a/dt shapes");
  std::vector<float> alog(48), dtb(48);
  for (uint32_t h = 0; h < 48; h++) {
    alog[h] = logf(-a[vhead(o, h)]);
    dtb[h] = dt[vhead(o, h)];
  }
  add_f32v(ck, L + "A_log", alog, {48});
  add_f32v(ck, L + "dt_bias", dtb, {48});
  add_f32v(ck, L + "norm.weight", norm_vec(G("ssm_norm.weight"), o.fold_ssm), {128});
}

// Main GGUF (+ optional MTP sidecar) -> synthetic tensors in `ck`.
// Returns the number of tensors written (new or overriding hgn records).
inline size_t build(hgn::Checkpoint& ck, const gguf::File& g, const gguf::File* mtp,
                    const Opts& o = Opts()) {
  const size_t n0 = ck.synthetic_count();
  const std::string arch = g.arch();
  const int nl = (int)g.kv_i(arch + ".block_count", 48);
  need(arch == "qwen4exp", "unexpected arch '" + arch + "'");
  for (int l = 0; l < nl && l < 48; l++) {
    if (o.layer && !o.layer(l)) continue;
    build_layer(ck, g, l, "layers." + std::to_string(l), l % 4 == 3, o);
  }
  if (o.globals) {
    add_q8(ck, "embed_tokens.weight", g.at("token_embd.weight"), {248320, 2560});
    add_q8(ck, "lm_head.weight", g.at("output.weight"), {248320, 2560});
    add_q8(ck, "hyper_connection_mixer.input_mix_weight_down.weight", g.at("output_hc_down.weight"),
           {320, 10240});
    add_q8(ck, "hyper_connection_mixer.input_mix_weight_up.weight", g.at("output_hc_up.weight"),
           {10240, 320});
    add_f32v(ck, "hyper_connection_mixer.hc_norm.weight",
             norm_vec(g.at("output_hc_norm.weight"), o.fold_hc), {10240});
    const std::string E = "layers.1.ple.";
    add_q8(ck, E + "key_proj.weight", g.at("blk.1.ple_key.weight"), {10240, 2560});
    add_q8(ck, E + "value_proj.weight", g.at("blk.1.ple_value.weight"), {2560, 2560});
    add_f32v(ck, E + "conv1d.weight", rows_f32(g.at("blk.1.ple_conv1d.weight")), {10240, 1, 4});
    add_f32v(ck, E + "norm_conv.weight", norm_vec(g.at("blk.1.ple_norm_conv.weight"), o.fold_ple),
             {10240});
    add_f32v(ck, E + "norm_key.weight", norm_vec(g.at("blk.1.ple_norm_key.weight"), o.fold_ple),
             {10240});
    add_f32v(ck, E + "norm_query.weight",
             norm_vec(g.at("blk.1.ple_norm_query.weight"), o.fold_ple), {10240});
    add_u64(ck, E + "ple_embedding.layer_multipliers", g, arch + ".ple.layer_multipliers", 3);
    add_u64(ck, E + "ple_embedding.ngram_heads_offsets", g, arch + ".ple.head_offsets", 16);
    add_u64(ck, E + "ple_embedding.ngram_heads_vocab_sizes", g, arch + ".ple.head_vocab_sizes", 16);
  }
  if (mtp && o.mtp) {
    const gguf::File& m = *mtp;
    const int bl = (int)m.kv_i(m.arch() + ".block_count", 49) - 1;
    const std::string N = "blk." + std::to_string(bl) + ".nextn.";
    const gguf::Tensor& eh = m.at(N + "eh_proj.weight");
    need(eh.ne[0] == 5120 && eh.ne[1] == 2560, "eh_proj shape");
    Map le, lh;
    for (uint32_t b = 0; b < 80; b++) le.push_back(b), lh.push_back(80 + b);
    add_q8(ck, "mtp.fc_embedding.weight", eh, {2560, 2560}, {}, le);
    add_q8(ck, "mtp.fc_hidden.weight", eh, {2560, 2560}, {}, lh);
    add_f32v(ck, "mtp.pre_fc_norm_embedding.weight", norm_vec(m.at(N + "enorm.weight"), o.fold_mtp),
             {2560});
    add_f32v(ck, "mtp.pre_fc_norm_hidden.weight", norm_vec(m.at(N + "hnorm.weight"), o.fold_mtp),
             {10240});
    add_q8(ck, "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
           m.at(N + "hc_head_down.weight"), {320, 10240});
    add_q8(ck, "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
           m.at(N + "hc_head_up.weight"), {10240, 320});
    add_f32v(ck, "mtp.hyper_connection_mixer.hc_norm.weight",
             norm_vec(m.at(N + "hc_head_norm.weight"), o.fold_hc), {10240});
    build_layer(ck, m, bl, "mtp.layers.0", true, o);
  }
  return ck.synthetic_count() - n0;
}

// PLE n-gram table: per_layer_token_embd.weight (IQ4_NL [rows][160], same row
// order as the hgn fp8 table; Q5_0 [rows][160] also accepted) as a zero-copy
// dtype 11 (IQ4_NL) / dtype 12 (Q5_0) view of the mmap; the engine streams
// rows from it (40_model.inc ple_rows*, 10_ple_io.inc).
inline void add_ple_table(hgn::Checkpoint& ck, const gguf::File& g) {
  const std::string name = "layers.1.ple.ngram_embedding.weight";
  if (g_keep && !g_keep(name)) return;
  const gguf::Tensor& t = g.at("per_layer_token_embd.weight");
  need(t.ne[0] == 160 && (t.type == gguf::IQ4_NL || t.type == gguf::Q5_0),
       "per_layer_token_embd: expected IQ4_NL or Q5_0 [rows][160]");
  need(t.row_bytes() == (t.type == gguf::IQ4_NL ? 90 : 110),
       "per_layer_token_embd: row size mismatch");
  hgn::Tensor d = desc(name, t.type == gguf::IQ4_NL ? 11 : 12, {t.rows(), 160});
  d.data = t.data;
  d.data_size = t.nbytes;
  ck.add_view(d);
}

// External standalone PLE table (QWENOX_GGUF_PLE=<path>): a GGUF holding
// per_layer_token_embd.weight on its own, either BF16 [rows][160] (the engram
// split — same bytes as the hgn bf16 n-gram file from tools/ple_ngram_bf16.py,
// registered with its (128, R, 160) dims) or IQ4_NL like the built-in table.
inline void add_ple_table_file(hgn::Checkpoint& ck, const gguf::File& g) {
  const std::string name = "layers.1.ple.ngram_embedding.weight";
  if (g_keep && !g_keep(name)) return;
  const gguf::Tensor& t = g.at("per_layer_token_embd.weight");
  if (t.type == gguf::IQ4_NL || t.type == gguf::Q5_0) {
    need(t.ne[0] == 160 && t.row_bytes() == (t.type == gguf::IQ4_NL ? 90 : 110),
         "external PLE: expected IQ4_NL/Q5_0 [rows][160]");
    hgn::Tensor d = desc(name, t.type == gguf::IQ4_NL ? 11 : 12, {t.rows(), 160});
    d.data = t.data;
    d.data_size = t.nbytes;
    ck.add_view(d);
    return;
  }
  need(t.type == gguf::BF16 && t.ne[0] == 160 && t.rows() % 128 == 0,
       "external PLE: expected BF16 [128*R][160] or IQ4_NL/Q5_0 [rows][160]");
  hgn::Tensor d = desc(name, 0, {128, t.rows() / 128, 160});
  d.data = t.data;
  d.data_size = t.nbytes;
  ck.add_view(d);
}

// mmproj GGUF (general.architecture clip, projector qwen3vl_merger, e.g.
// mmproj-BF16.gguf) -> the vision tower's hgn names (visual.*), all bf16
// (dtype 0) like the hgn vision file; 31_vision.inc then loads it unchanged.
// Only the Conv3d patch embed needs a transform: the converter split it into
// its two temporal slices v.patch_embd.weight / .weight.1 (each torch
// [1152][3][16][16]); the engine wants [1152][3*2*16*16], column
// c*512 + t*256 + y*16 + x. The F32 tensors (biases, norms, patch/pos embed)
// were upcast from bf16 sources, so bf16 is lossless
// (tools/g3_vision_check.cpp: bit-exact vs the hgn vision file).
inline size_t build_vision(hgn::Checkpoint& ck, const gguf::File& g) {
  const size_t n0 = ck.synthetic_count();
  need(g.arch() == "clip" && g.kv_s("clip.projector_type") == "qwen3vl_merger",
       "mmproj: expected a clip / qwen3vl_merger GGUF");
  const int nb = (int)g.kv_i("clip.vision.block_count", 0);
  const uint64_t H = (uint64_t)g.kv_i("clip.vision.embedding_length", 0);
  const uint64_t F = (uint64_t)g.kv_i("clip.vision.feed_forward_length", 0);
  const uint64_t O = (uint64_t)g.kv_i("clip.vision.projection_dim", 0);
  need(nb == 27 && H == 1152 && F == 4304 && O == 2560, "mmproj: unexpected vision dims");
  const uint64_t M = 4 * H;  // 2x2 spatial merge
  auto mat = [&](const std::string& n, const std::string& gn, uint64_t r, uint64_t c) {
    const gguf::Tensor& t = g.at(gn);
    need(t.ne[0] == c && t.rows() == r, n + " <- " + gn + ": shape");
    add_bf16(ck, n, t, {r, c});
  };
  auto vec = [&](const std::string& n, const std::string& gn, uint64_t c) {
    const gguf::Tensor& t = g.at(gn);
    need(t.numel() == c, n + " <- " + gn + ": shape");
    add_bf16(ck, n, t, {c});
  };
  {
    const gguf::Tensor& t0 = g.at("v.patch_embd.weight");
    const gguf::Tensor& t1 = g.at("v.patch_embd.weight.1");
    need(t0.ne[0] == 16 && t0.ne[1] == 16 && t0.ne[2] == 3 && t0.ne[3] == H &&
             t1.numel() == t0.numel(),
         "v.patch_embd: shape");
    const std::vector<float> a = rows_f32(t0), b = rows_f32(t1);
    std::vector<float> w(H * 1536);
    for (uint64_t o = 0; o < H; o++)
      for (uint64_t c = 0; c < 3; c++)
        for (uint64_t t = 0; t < 2; t++)
          memcpy(&w[o * 1536 + c * 512 + t * 256], &(t ? b : a)[(o * 3 + c) * 256],
                 256 * sizeof(float));
    add_bf16v(ck, "visual.patch_embed.proj.weight", w, {H, 1536});
  }
  vec("visual.patch_embed.proj.bias", "v.patch_embd.bias", H);
  mat("visual.pos_embed.weight", "v.position_embd.weight", 2304, H);
  for (int i = 0; i < nb; i++) {
    const std::string P = "visual.blocks." + std::to_string(i) + ".";
    const std::string B = "v.blk." + std::to_string(i) + ".";
    mat(P + "attn.qkv.weight", B + "attn_qkv.weight", 3 * H, H);
    vec(P + "attn.qkv.bias", B + "attn_qkv.bias", 3 * H);
    mat(P + "attn.proj.weight", B + "attn_out.weight", H, H);
    vec(P + "attn.proj.bias", B + "attn_out.bias", H);
    mat(P + "mlp.linear_fc1.weight", B + "ffn_up.weight", F, H);
    vec(P + "mlp.linear_fc1.bias", B + "ffn_up.bias", F);
    mat(P + "mlp.linear_fc2.weight", B + "ffn_down.weight", H, F);
    vec(P + "mlp.linear_fc2.bias", B + "ffn_down.bias", H);
    vec(P + "norm1.weight", B + "ln1.weight", H);
    vec(P + "norm1.bias", B + "ln1.bias", H);
    vec(P + "norm2.weight", B + "ln2.weight", H);
    vec(P + "norm2.bias", B + "ln2.bias", H);
  }
  vec("visual.merger.norm.weight", "v.post_ln.weight", H);
  vec("visual.merger.norm.bias", "v.post_ln.bias", H);
  mat("visual.merger.linear_fc1.weight", "mm.0.weight", M, M);
  vec("visual.merger.linear_fc1.bias", "mm.0.bias", M);
  mat("visual.merger.linear_fc2.weight", "mm.2.weight", O, M);
  vec("visual.merger.linear_fc2.bias", "mm.2.bias", O);
  return ck.synthetic_count() - n0;
}

}  // namespace gguf_map
