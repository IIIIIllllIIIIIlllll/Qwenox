#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

struct VisionMemDemand {
  int max_p = 0;
  size_t rows = 0;
};

constexpr size_t kVisionEmbeddingRowBytes = 2560 * 2;
constexpr size_t kVisionMemorySafetyBytes = (size_t)512 << 20;

inline size_t vision_mem_mul(size_t count, size_t bytes) {
  return count > std::numeric_limits<size_t>::max() / bytes
             ? std::numeric_limits<size_t>::max()
             : count * bytes;
}

inline size_t vision_mem_add(size_t a, size_t b) {
  return a > std::numeric_limits<size_t>::max() - b
             ? std::numeric_limits<size_t>::max()
             : a + b;
}

inline size_t vision_workspace_bytes(int p) {
  // Valid forwards have M=P/4. The 16 activation buffers total
  // 72032*P + 15360*M = 75872*P bytes.
  return p > 0 ? vision_mem_mul((size_t)p, 75872) : 0;
}

inline size_t vision_memory_peak_extra(int cap_p, size_t cap_rows,
                                       const VisionMemDemand& demand) {
  if (demand.max_p <= 0 || demand.rows == 0) return 0;
  const size_t target_rows =
      std::max(cap_rows, std::max<size_t>(4096, demand.rows));
  const size_t emb_alloc = target_rows > cap_rows
                               ? vision_mem_mul(target_rows, kVisionEmbeddingRowBytes)
                               : 0;
  const size_t emb_delta =
      vision_mem_mul(target_rows - cap_rows, kVisionEmbeddingRowBytes);
  const size_t ws_delta = demand.max_p > cap_p
                              ? vision_workspace_bytes(demand.max_p) -
                                    vision_workspace_bytes(cap_p)
                              : 0;
  // Embedding growth briefly keeps old+new. After its commit, ViT growth
  // releases the old workspace before allocating the larger one.
  return std::max(emb_alloc, vision_mem_add(emb_delta, ws_delta));
}

inline size_t vision_memory_budget(int cap_p, size_t cap_rows,
                                   const VisionMemDemand& demand) {
  const size_t extra = vision_memory_peak_extra(cap_p, cap_rows, demand);
  return extra ? vision_mem_add(extra, kVisionMemorySafetyBytes) : 0;
}
