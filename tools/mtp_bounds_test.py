#!/usr/bin/env python3
"""Host-only regression for the production pure-MTP round bounds."""

from pathlib import Path
import os
import re
import subprocess
import tempfile


source = (Path(__file__).resolve().parent.parent /
          "src/gpu/parts/40_model.inc").read_text(encoding="utf-8")
greedy = source[source.index("std::vector<int> spec_loop("):
                source.index("std::vector<int> spec_loop_chain(")]
sample = source[source.index("std::vector<int> spec_loop_sample("):
                source.index("std::vector<int> spec_loop_chain_sample(")]


def expression(body, pattern, label):
    match = re.search(pattern, body, re.S)
    if not match:
        raise RuntimeError(f"production {label} expression not found")
    return match.group(1).strip()


def bound(body):
    return expression(body, r"const int g_bound = (.*?);", "pure-MTP g_bound")


chain = source[source.index("std::vector<int> spec_loop_chain("):
               source.index("std::vector<int> spec_loop_sample(")]
chain_sample = source[source.index("std::vector<int> spec_loop_chain_sample("):]


def chain_expressions(body):
    # Restrict each extraction to its actual round-top or MTP fallback site;
    # do not substitute a desired bound for what production currently executes.
    top = body[body.index("while ((int)out.size() < gen && pos < maxctx)"):]
    fallback = top[top.index("// ---- MTP round (ngram table missed)"):]
    get = lambda text, pattern, name: expression(text, pattern, name)
    call = top[top.index("spec_round_point(") + len("spec_round_point("):]
    args, start, depth = [], 0, 0
    for i, char in enumerate(call):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                args.append(call[start:i].strip())
                break
            depth -= 1
        elif char == "," and depth == 0:
            args.append(call[start:i].strip())
            start = i + 1
    if len(args) < 3 or args[0] != "remaining":
        raise RuntimeError("production chain reservation arguments not found")
    before_pick = fallback[:fallback.index("g = occ.pick(")]
    guard = re.search(r"if\s*\((g\s*<\s*1)\)\s*break\s*;", before_pick)
    return {
        "NG": get(top, r"const int ng_bound\s*=\s*(.*?);", "chain ng_bound"),
        "MTP": get(top, r"const int mtp_bound\s*=\s*(.*?);", "chain mtp_bound"),
        "BEGIN": args[1],
        "END": args[2],
        "WANT": get(top, r"int want\s*=\s*(.*?);", "chain ngram want"),
        "G": get(fallback, r"int g\s*=\s*(.*?);", "chain fallback g"),
        "BREAK": guard.group(1) if guard else "false",
        "GD": get(fallback, r"const int gd\s*=\s*(.*?);", "chain fallback gd"),
    }


program = r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
struct Controller { int cur; };
int kSpecConfMax = SPEC_CONF_MAX;
int confidence = 0;
int spec_conf() { return confidence; }
int greedy(int maxctx, int pos, int remaining, Controller gctl) {
  return GREEDY;
}
int sample(int maxctx, int pos, int remaining, Controller gctl) {
  return SAMPLE;
}
int main() {
  for (int capacity = 1; capacity <= 20; ++capacity)
    for (int pos = 1; pos < capacity; ++pos)
      for (int remaining = 1; remaining <= 20; ++remaining)
        for (int gamma = 0; gamma <= 8; ++gamma)
          for (confidence = 0; confidence <= 1; ++confidence) {
            Controller ctl{gamma};
            for (int mode = 0; mode < 2; ++mode) {
              int g = mode ? sample(capacity, pos, remaining, ctl)
                           : greedy(capacity, pos, remaining, ctl);
              int requested = mode && confidence ? kSpecConfMax : gamma;
              int expected = std::min({requested, remaining - 1,
                                       capacity - pos - 1});
              assert(g == expected);
              assert(g >= 0 && g + 1 <= remaining);
              assert(pos + g + 1 <= capacity);
              // First MTP row is pos-1 even for g=0; later draft rows
              // and all verify rows lie inside the reserved interval.
              assert(pos - 1 >= 0 && pos - 1 + std::max(1, g) <= pos + g + 1);
            }
          }
  std::puts("pure MTP bounds: PASS");
}
'''.replace("GREEDY", bound(greedy)).replace("SAMPLE", bound(sample))
chain_function = r'''
void NAME(int maxctx, int pos, int remaining, int maxg, Controller gctl,
          bool mtp_ok, int picked, int& failures, int& zero_rounds) {
  // Reconstruct the production variables used by the extracted expressions.
  std::vector<int> out(1);
  int gen = remaining + (int)out.size();
  const int ng_bound = EXPR_NG;
  const int mtp_bound = EXPR_MTP;
  const int64_t begin = EXPR_BEGIN;
  const int64_t end = EXPR_END;
  int want = EXPR_WANT;
  const int q = pos - 1;
  int g = EXPR_G;
  const int requested = SAMPLED && confidence ? kSpecConfMax : gctl.cur;
  const int expected_ng = std::min({maxg, remaining - 1, maxctx - pos - 1});
  const int expected_mtp = std::min({requested, remaining - 1, maxctx - pos - 1});
  auto check = [&](bool ok, const char* what) {
    if (!ok) {
      if (failures < 12)
        std::fprintf(stderr, "%s: %s (ctx=%d pos=%d remaining=%d maxg=%d "
                     "gamma=%d confidence=%d mtp_ok=%d picked=%d g=%d "
                     "ng_bound=%d mtp_bound=%d reservation=[%lld,%lld))\n",
                     "NAME", what, maxctx, pos, remaining, maxg, gctl.cur,
                     confidence, (int)mtp_ok, picked, g, ng_bound, mtp_bound,
                     (long long)begin, (long long)end);
      ++failures;
    }
  };
  check(ng_bound == expected_ng && want == expected_ng, "ngram budget");
  check(begin == pos - (mtp_ok ? 1 : 0), "reservation begin");
  check(end >= pos + want + 1 && end <= maxctx, "ngram reservation end");
  if (!mtp_ok) {
    check(mtp_bound == 0, "disabled MTP bound");
    return;
  }
  check(mtp_bound == expected_mtp, "round-top MTP budget");
  check(g == expected_mtp, "fallback MTP budget");
  if (EXPR_BREAK) {
    check(false, "fallback breaks before a g=0 round");
    return;
  }
  // Model every possible occupancy clamp result, including zero. Actual
  // scheduling is not part of this host-only test.
  if (picked > g) return;
  g = picked;
  const int gd = EXPR_GD;
  if (g == 0) ++zero_rounds;
  check(gd == std::max(1, g), "draft row count");
  check(g + 1 <= remaining, "verify exceeds remaining output budget");
  check(q >= begin && q + gd <= end, "MTP draft outside reservation");
  check(pos >= begin && pos + g + 1 <= end, "verify outside reservation");
  check(pos + g + 1 <= maxctx, "verify exceeds context");
}
'''

functions = []
for name, body, sampled in (("chain_greedy", chain, False),
                            ("chain_sample", chain_sample, True)):
    function = chain_function.replace("NAME", name).replace("SAMPLED", str(int(sampled)))
    expressions = chain_expressions(body)
    for key in sorted(expressions, key=len, reverse=True):
        function = function.replace("EXPR_" + key, expressions[key])
    functions.append(function)

program = program.replace("#include <cstdio>",
                          "#include <cstdio>\n#include <cstdint>\n#include <vector>")
program = program.replace("int main() {", "\n".join(functions) + "\nint main() {")
program = program.replace('  std::puts("pure MTP bounds: PASS");', r'''
  std::puts("pure MTP bounds: PASS");
  int failures = 0, zero_rounds = 0;
  for (int capacity = 2; capacity <= 12; ++capacity)
    for (int pos = 1; pos < capacity; ++pos)
      for (int remaining = 1; remaining <= 12; ++remaining)
        for (int gamma = 0; gamma <= 8; ++gamma)
          for (int maxg : {0, 1, 4, 12})
            for (confidence = 0; confidence <= 1; ++confidence)
              for (int picked = 0; picked <= std::max(8, kSpecConfMax); ++picked) {
                Controller ctl{gamma};
                chain_greedy(capacity, pos, remaining, maxg, ctl, true,
                             picked, failures, zero_rounds);
                for (bool enabled : {false, true})
                  chain_sample(capacity, pos, remaining, maxg, ctl, enabled,
                               picked, failures, zero_rounds);
              }
  if (!zero_rounds) {
    std::fputs("chain: no occupancy-clamped g=0 rounds exercised\n", stderr);
    return 1;
  }
  if (failures) {
    std::fprintf(stderr, "chain bounds: FAIL (%d checks)\n", failures);
    return 1;
  }
  std::puts("chain MTP bounds and reservations: PASS");''')

conf_max = re.search(r"kSpecConfMax\s*=\s*(\d+)\s*;", source)
if not conf_max:
    raise RuntimeError("production kSpecConfMax not found")
program = program.replace("SPEC_CONF_MAX", conf_max.group(1))

with tempfile.TemporaryDirectory(prefix="mtp-bounds-") as directory:
    cpp = Path(directory) / "test.cpp"
    binary = Path(directory) / "test"
    cpp.write_text(program, encoding="utf-8")
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                    "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)],
                   check=True)
    subprocess.run([str(binary)], check=True)
