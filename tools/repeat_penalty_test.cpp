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
  {
    LlamaPenaltyState base;
    base.configure(4, 1.25f, 0.5f, 0.25f, 16);
    accept_all(base, {1, 2, 1, 3, 4, 2});
    const int drafts[] = {1, 5, 6, 7, 2, 8};
    const auto saved_recent = base.recent;
    const auto saved_counts = base.counts;
    const size_t saved_next = base.next;
    LlamaPenaltyRows plan;
    check(llama_penalty_rows(base, drafts, 0, 6, 4, plan),
          "row plan accepts a wrapped full window");
    check(base.recent == saved_recent && base.counts == saved_counts && base.next == saved_next,
          "row plan leaves the committed rollback state untouched");

    LlamaPenaltyState view = base;
    bool exact = true;
    for (int r = 0; r < 6; r++) {
      std::vector<float> got(10), want(10);
      for (int i = 0; i < 10; i++) got[i] = want[i] = (i & 1 ? 0.75f : -0.75f) * (i + 1);
      view.apply(want);
      for (int j = 0; j < plan.counts[r]; j++) {
        const int id = plan.ids[(size_t)r * 4 + j];
        float& x = got[id];
        if (x <= 0.0f) x *= view.repeat;
        else x /= view.repeat;
        x -= plan.values[(size_t)r * 4 + j];
      }
      exact &= std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0;
      if (r + 1 < 6) view.accept(drafts[r]);
    }
    check(exact, "row plans match dense apply through repeats and evictions");

    LlamaPenaltyState tail = base;
    tail.accept(drafts[0]);
    tail.accept(drafts[1]);
    LlamaPenaltyRows chunk;
    check(llama_penalty_rows(tail, drafts, 2, 4, 4, chunk), "chunked row plan builds");
    bool chunk_same = chunk.counts.size() == 4;
    for (int r = 0; r < 4 && chunk_same; r++) {
      chunk_same &= chunk.counts[r] == plan.counts[r + 2];
      for (int j = 0; j < chunk.counts[r]; j++) {
        const int id = chunk.ids[(size_t)r * 4 + j];
        const float v = chunk.values[(size_t)r * 4 + j];
        bool found = false;
        for (int u = 0; u < plan.counts[r + 2]; u++)
          if (plan.ids[(size_t)(r + 2) * 4 + u] == id) {
            found = std::memcmp(&v, &plan.values[(size_t)(r + 2) * 4 + u], sizeof(float)) == 0;
            break;
          }
        chunk_same &= found;
      }
    }
    check(chunk_same, "chunked rows equal the corresponding unchunked rows");
    LlamaPenaltyRows small;
    check(!llama_penalty_rows(base, drafts, 0, 1, 2, small),
          "row-plan capacity failure is explicit");

    LlamaPenaltyState off;
    off.configure(0, 2.0f, 0.5f, 0.25f, 8);
    accept_all(off, {1, 2, 1});
    LlamaPenaltyRows disabled;
    check(llama_penalty_rows(off, drafts, 0, 3, 2, disabled) &&
              std::all_of(disabled.counts.begin(), disabled.counts.end(), [](int n) { return n == 0; }),
          "last_n=0 produces no corrections even with accepted history");
  }
  std::printf("TOTAL: %s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
