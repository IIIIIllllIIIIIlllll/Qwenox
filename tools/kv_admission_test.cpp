#include "../src/kv_admission.h"

#include <climits>
#include <cstdio>
#include <stdexcept>
#include <vector>

static constexpr int KV_PAGE = 256;

static void require(bool passed, const char* name) {
  if (!passed) throw std::runtime_error(name);
}

static void test_arithmetic() {
  int cap = -1;
  require(kv_admission_decode_cap(nullptr, cap) && cap == 4096,
          "default decode cap");
  require(kv_admission_decode_cap("0", cap) && cap == 0, "zero decode cap");
  require(kv_admission_decode_cap("2147483647", cap) && cap == INT_MAX,
          "maximum decode cap");
  for (const char* value : {"", "-1", "1.5", "oops", "2x",
                            "2147483648", "999999999999999999999999"})
    require(!kv_admission_decode_cap(value, cap), "invalid decode cap accepted");

  require(kv_admission_target(524287, 100, 4096, 524288, KV_PAGE) == 2048,
          "one-token context edge");
  require(kv_admission_target(524263, 100, 4096, 524288, KV_PAGE) == 2048,
          "25-token context edge");
  require(kv_admission_target(524032, 256, 4096, 524288, KV_PAGE) == 2048,
          "full context has no artificial slack");
  require(kv_admission_target(256, INT_MAX, 4096, 524288, KV_PAGE) == 17,
          "unbounded generation uses decode cap");
  require(kv_admission_target(256, 5, 0, 524288, KV_PAGE) == 1,
          "zero cap reserves prompt only");
  require(kv_admission_target(INT_MAX - 1, INT_MAX, INT_MAX, INT_MAX, KV_PAGE) ==
              8388608,
          "page rounding overflow");

  require(kv_admission_need(90, 400, false) == 90,
          "non-prefix slot reset");
  require(kv_admission_need(90, 100, true) == 100,
          "continuation keeps mapped pages");

  std::vector<KvAdmissionState> states = {{true, 80, 90}, {false, 1000, 0}};
  require(kv_admission_available(2048, states, 1) == 1958,
          "reset slot does not reserve released pages");
  states[0].mapped = 150;
  require(kv_admission_available(2048, states, 1) == 1898,
          "actual mappings above cap constrain peers");
  states[0] = {false, 2048, 0};
  require(kv_admission_available(2048, states, 1) == 2048,
          "idle pages do not count as active demand");
  states = {{true, 100, 120}, {true, 100, 120}, {false, 1000, 0}};
  require(kv_admission_available(2048, states, 2) == 1808,
          "active reservations are independent");
  require(kv_admission_available(2048, states, 0) == 1928,
          "chosen slot is excluded");
}

int main() {
  try {
    test_arithmetic();
    std::puts("PASS admission arithmetic / decode cap / context edge");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}
