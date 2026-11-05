// llama.cpp-style shared repeat/frequency/presence penalty state.
// The window contains accepted prompt and generated token ids, not KV state.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct LlamaPenaltyState {
  int64_t last_n = 64;
  float repeat = 1.0f;
  float frequency = 0.0f;
  float presence = 0.0f;

  size_t cap = 0;
  size_t next = 0;
  std::vector<int> recent;
  std::unordered_map<int, int> counts;

  void configure(int64_t n, float r, float f, float p, size_t max_history) {
    last_n = n;
    repeat = r;
    frequency = f;
    presence = p;
    cap = 0;
    if (n != 0 && (r != 1.0f || f != 0.0f || p != 0.0f)) {
      const size_t wanted = n < 0 ? max_history : (size_t)std::min<int64_t>(n, (int64_t)max_history);
      cap = std::min(wanted, max_history);
      recent.reserve(cap);
    }
    clear();
  }

  bool active() const { return cap != 0; }

  void clear() {
    next = 0;
    recent.clear();
    counts.clear();
  }

  void accept(int id) {
    if (!active()) return;
    counts[id]++;
    if (recent.size() < cap) {
      recent.push_back(id);
      return;
    }
    const int old = recent[next];
    if (--counts[old] == 0) counts.erase(old);
    recent[next] = id;
    next = (next + 1) % cap;
  }

  void apply(std::vector<float>& logits) const {
    if (!active()) return;
    const int n = (int)logits.size();
    for (const auto& h : counts) {
      if (h.first < 0 || h.first >= n) continue;
      float& logit = logits[h.first];
      if (logit <= 0.0f) logit *= repeat;
      else logit /= repeat;
      logit -= (float)h.second * frequency + presence;
    }
  }
};
