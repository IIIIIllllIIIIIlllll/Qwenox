// repeat_penalty_test.cpp — host-only checks for llama.cpp-style penalties.
// Build: clang++ -std=c++17 -O2 tools/repeat_penalty_test.cpp -o build/repeat_penalty_test
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/gpu/penalty_state.h"

static int fails = 0;

static void check(bool ok, const char* name) {
  std::printf("%s: %s\n", name, ok ? "PASS" : "FAIL");
  if (!ok) fails++;
}

static void accept_all(LlamaPenaltyState& s, const std::vector<int>& ids) {
  for (int id : ids) s.accept(id);
}

int main() {
  {
    LlamaPenaltyState s;
    s.configure(4, 2.0f, 0.25f, 0.5f, 5);
    accept_all(s, {1, 2, 1, 1, 3});
    std::vector<float> logits = {0.0f, 4.0f, -4.0f, 0.0f, 7.0f};
    s.apply(logits);
    const std::vector<float> expected = {0.0f, 1.0f, -8.75f, -0.75f, 7.0f};
    check(s.recent.size() == 4 && s.counts.at(1) == 2 && s.counts.at(2) == 1 &&
              s.counts.at(3) == 1 && logits == expected,
          "llama reference: eviction, counts, signs, frequency and presence");
  }
  {
    LlamaPenaltyState s;
    s.configure(2, 2.0f, 0.0f, 0.0f, 5);
    accept_all(s, {5, 6, 5});  // prompt tail [6, 5] affects the first generated token
    std::vector<float> logits(7, 0.0f);
    logits[5] = 8.0f;
    logits[6] = -8.0f;
    s.apply(logits);
    check(logits[5] == 4.0f && logits[6] == -16.0f,
          "prompt seeds the window before the first draw");
    s.accept(6);
    s.accept(6);  // accepted generated tokens affect subsequent draws only
    logits.assign(7, 0.0f);
    logits[5] = 8.0f;
    logits[6] = -8.0f;
    s.apply(logits);
    check(!s.counts.count(5) && s.counts.at(6) == 2 && logits[5] == 8.0f &&
              logits[6] == -16.0f,
          "duplicate eviction keeps membership; repeat is not exponentiated");
  }
  {
    LlamaPenaltyState s;
    s.configure(0, 2.0f, 0.25f, 0.5f, 8);
    accept_all(s, {1, 2, 3});
    std::vector<float> logits = {1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> before = logits;
    s.apply(logits);
    check(!s.active() && std::memcmp(logits.data(), before.data(), sizeof(float) * logits.size()) == 0,
          "last_n=0 disables repeat, frequency and presence");
    s.configure(64, 1.0f, 0.0f, 0.0f, 8);
    accept_all(s, {1, 2, 3});
    logits = before;
    s.apply(logits);
    check(!s.active() && logits == before, "neutral penalties disable the state");
  }
  {
    LlamaPenaltyState s;
    s.configure(-1, 2.0f, 0.0f, 0.0f, 3);
    accept_all(s, {1, 2, 1, 1, 3});
    std::vector<float> logits = {0.0f, 4.0f, -4.0f, 0.0f};
    s.apply(logits);
    check(s.cap == 3 && s.counts.at(1) == 2 && s.counts.at(3) == 1 &&
              !s.counts.count(2) && logits[1] == 2.0f && logits[2] == -4.0f,
          "last_n=-1 retains the whole request up to its maximum history");
  }
  {
    LlamaPenaltyState s;
    s.configure(1, 2.0f, 0.0f, 0.0f, 3);
    accept_all(s, {1, 1, 2});
    std::vector<float> logits = {0.0f, 8.0f, -8.0f};
    s.apply(logits);
    check(!s.counts.count(1) && s.counts.at(2) == 1 && logits[1] == 8.0f &&
              logits[2] == -16.0f,
          "oldest token is fully evicted");
  }
  {
    LlamaPenaltyState a;
    a.configure(4, 2.0f, 0.25f, 0.5f, 8);
    accept_all(a, {1, 2, 1, 1, 3});
    LlamaPenaltyState b = a;
    std::vector<float> x = {0.0f, 4.0f, -4.0f, 0.0f, 7.0f}, y = x;
    a.apply(x);
    b.apply(y);
    a.accept(2);
    b.accept(2);
    std::vector<float> u = {0.0f, 4.0f, -4.0f, 0.0f, 7.0f}, v = u;
    a.apply(u);
    b.apply(v);
    check(x == y && u == v && a.recent == b.recent && a.counts == b.counts,
          "copied speculative state preserves ring and counts");
  }
  std::printf("TOTAL: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
