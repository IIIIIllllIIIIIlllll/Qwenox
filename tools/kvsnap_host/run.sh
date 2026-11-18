#!/usr/bin/env bash
# One-shot host test of kvsnap (no GPU): extract the namespace from the repo,
# compile with -Wall -Wextra -Werror, run save + restart-index stages.
# usage: tools/kvsnap_host/run.sh [REPO] [DIR] [bf16|fp16] (Linux; no GPU/model)
#  -> ends with KVSNAP HOST TEST: PASS/FAIL
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="${1:-$here/../..}"
dir="${2:-/tmp/kvsnap-host-test}"
precision="${3:-bf16}"
[[ "$precision" == bf16 || "$precision" == fp16 ]] || { echo 'expected bf16 or fp16' >&2; exit 2; }
src="$repo/src/gpu/parts/51_host_cfg.inc"
end=$(grep -n '^}  // namespace kvsnap' "$src" | cut -d: -f1)
build=/tmp/kvsnap-host-build
mkdir -p "$build"
# namespace kvsnap only; kernel launches k<<<g, b, s, q>>>(args) -> k(args)
{ echo "namespace kvsnap {"; sed -n "1,${end}p" "$src" | tr -d "\r" | sed -E "s/<<<[^>]*>>>//"; } > "$build/kvsnap_gen.inc"
pe=$(grep -n '^#endif' "$repo/src/gpu/parts/00_platform.inc" | head -1 | cut -d: -f1)
sed -n "1,${pe}p" "$repo/src/gpu/parts/00_platform.inc" | tr -d "\r" > "$build/platform_gen.inc"
cp "$here/harness.cpp" "$build/"
g++ -std=c++17 -O1 -g -Wall -Wextra -Werror -Wno-unused-parameter \
    -Wno-unused-result -Wno-unused-variable -fsanitize=address,undefined -pthread \
    -I"$build" "$build/harness.cpp" -o "$build/harness"
rm -rf "$dir" "$dir.index"
rc=0
timeout 120 "$build/harness" save "$dir" "$precision" 2>"$build/save.err" || rc=1
timeout 120 "$build/harness" index "$dir" "$precision" 2>"$build/index.err" || rc=1
echo "---- engine log (stderr) ----"; tail -n 12 "$build/save.err"; tail -n 8 "$build/index.err"
((rc == 0)) && echo "KVSNAP HOST TEST: PASS" || echo "KVSNAP HOST TEST: FAIL"
exit $rc
