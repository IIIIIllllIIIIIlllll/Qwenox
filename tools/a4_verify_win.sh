#!/usr/bin/env bash
# a4_verify_win.sh - one-shot check of the QWENOX_MOE_A4 switch (Windows Git
# Bash; Linux works too with BIN_OLD/BIN_NEW). v2 hgn, 2 slots, chunk 6144.
#
#   BIN_OLD=<engine before the change> bash tools/a4_verify_win.sh
#
#   CHUNK=8192 overrides the prefill chunk (unset = engine default).
# 1. old engine, A4 unset: a4_conc.py seq                       (reference)
# 2. new engine, A4 unset: seq -> tokens must equal 1 (default path unchanged);
#    then 2-way concurrent -> must equal seq (multi-chunk prefill + decode on
#    the other slot: the PLE flight reader must not share the decode IOCP)
# 3. new engine, MOE_A4=3: seq, then 2-way concurrent -> greedy tokens must be
#    identical (two multi-chunk 20K prefills interleave + short flows borrow
#    the GPU mid-chunk); the A4 log line must appear.
# Needs the service stopped (whole-GPU arena). Last line A4_VERIFY PASS|FAIL.
# Logs: logs/a4v_<tag>.log (engine), logs/a4v_*.txt, logs/conc/a4v_*.json
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
L=logs; mkdir -p $L logs/conc
win() { if command -v cygpath > /dev/null; then cygpath -m "$1"; else echo "$1"; fi; }
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then EXE=qwenox-engine-win.exe; else EXE=qwenox-engine; fi
BIN_OLD=${BIN_OLD:?set BIN_OLD to the engine built before the change}
BIN_NEW=${BIN_NEW:-$ROOT/build/$EXE}
M=$ROOT/models
MODEL_ARGS="${MODEL_ARGS:-$(win $M/qwen38-flash-next-v2.hgn) $(win $M/qwen38-flash-next-ngram.hgn) $(win $M/qwen38-flash-next-mtp.hgn)}"
MAXCTX=${MAXCTX:-262144}
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
FAILS=()
fail() { FAILS+=("$1"); echo "  FAIL: $1"; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

for b in "$BIN_OLD" "$BIN_NEW"; do
  [[ -f "$b" ]] || { echo "missing engine: $b"; echo "A4_VERIFY FAIL"; exit 1; }
done
if command -v tasklist > /dev/null && tasklist 2>/dev/null | grep -qi "qwenox"; then
  echo "a qwenox process is running - stop the service first (the test needs the whole GPU)"
  echo "A4_VERIFY FAIL"; exit 1
fi
echo "root $ROOT  head $(git log --oneline -1 | cut -c1-60)"
echo "old  $BIN_OLD"
echo "new  $BIN_NEW"

prod_env() {  # start_win.sh production kernel set, 2 slots; $1 = MOE_A4 (0 = unset)
  for e in $(compgen -e | grep '^QWENOX_'); do unset "$e"; done
  export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
  export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
  export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
  export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
  export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
  export QWENOX_KV_PAGED=1 QWENOX_ROPE_FACTOR=1 QWENOX_ROPE_ORIGINAL_CTX=262144
  export QWENOX_ROPE_BETA_FAST=32 QWENOX_ROPE_BETA_SLOW=1 QWENOX_ROPE_ATTN_SCALE=0
  export QWENOX_PARALLEL=2 QWENOX_PREFILL_CHUNK=${CHUNK:-6144}
  export QWENOX_KVSNAP=0 QWENOX_RCKPT=0 QWENOX_NGRAM_CHUNK=16
  if (( $1 )); then export QWENOX_MOE_A4=$1; fi
  return 0
}

ENG_PID=""
serve() {  # serve TAG BIN A4
  local tag=$1 bin=$2
  prod_env "$3"
  local log=$ROOT/$L/a4v_$tag.log
  local margs; read -r -a margs <<< "$MODEL_ARGS"
  (cd "$(dirname "$bin")" && exec "$bin" "${margs[@]}" --serve --host 127.0.0.1 --port 8732 \
      --maxctx "$MAXCTX" > "$log" 2>&1) &
  ENG_PID=$!
  local t0=$(date +%s)
  while :; do
    if tr '\r' '\n' < "$log" | grep -q "serve: listening"; then
      echo "  [$tag] listening after $(( $(date +%s) - t0 ))s"; return 0
    fi
    if ! kill -0 $ENG_PID 2>/dev/null; then
      echo "  [$tag] engine exited before listening"; tr '\r' '\n' < "$log" | grep -v "pload [0-9]" | tail -8; return 1
    fi
    (( $(date +%s) - t0 > 600 )) && { echo "  [$tag] timeout"; return 1; }
    sleep 2
  done
}
stop_eng() {
  [[ -n "$ENG_PID" ]] && kill $ENG_PID 2>/dev/null
  if command -v powershell > /dev/null; then
    powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { (\$_.Name -like 'qwenox-engine*') -and (\$_.CommandLine -match '8732') } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" > /dev/null 2>&1
  fi
  sleep 5; ENG_PID=""
}
trap stop_eng EXIT

engine_errors() {
  local e
  e=$(tr '\r' '\n' < $L/a4v_$1.log | grep -E "hipError|HIP error|Exception Code|FATAL|abort|KV pool exhausted|cannot fit|rejected" | head -3)
  if [[ -n "$e" ]]; then echo "$e" | sed 's/^/  engine log: /'; fail "engine error in $1"; fi
}
run() {  # run TAG seq|conc
  timeout 1800 python tools/a4_conc.py "$2" --tag "a4v_$1" --slots 2 2>&1 | tee $L/a4v_$1.txt
  [[ ${PIPESTATUS[0]} == 0 ]] || fail "$1 run had errors"
}
cmp() {  # cmp A B LABEL -> greedy flows must match token for token
  python - "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
a, b, label = sys.argv[1:]
C = Path('logs/conc')
try:
    A = json.loads((C / f'a4v_{a}.json').read_text())['res']
    B = json.loads((C / f'a4v_{b}.json').read_text())['res']
except Exception as e:
    print(f'{label}: missing results ({e})'); print('CMP FAIL'); raise SystemExit
bad = [k for k in A if k not in B or A[k]['tokens'] != B[k]['tokens']]
greedy_bad = [k for k in bad if not k.startswith('S-')]
print(f'{label}: {len(A) - len(bad)}/{len(A)} identical'
      + (f'; differ: {bad}' if bad else '') + ' (S-* = unseeded sampling, info only)')
print('CMP', 'PASS' if not greedy_bad else 'FAIL')
PY
}

stamp "1/3 old engine, A4 off"
if serve old "$BIN_OLD" 0; then run old seq; engine_errors old; else fail "old engine did not start"; fi
stop_eng

stamp "2/3 new engine, A4 off"
if serve new0 "$BIN_NEW" 0; then
  run new0 seq
  run new0conc conc
  engine_errors new0
  tr '\r' '\n' < $L/a4v_new0.log | grep -q "QWENOX_MOE_A4" && fail "A4 log line with the switch off"
else
  fail "new engine (A4 off) did not start"
fi
stop_eng

stamp "3/3 new engine, MOE_A4=3"
if serve a4 "$BIN_NEW" 3; then
  run a4seq seq
  run a4conc conc
  engine_errors a4
  tr '\r' '\n' < $L/a4v_a4.log | grep -m1 "QWENOX_MOE_A4" | sed 's/^/  engine log: /'
  tr '\r' '\n' < $L/a4v_a4.log | grep -q "W4A4 iu4 on" || fail "A4 path never ran (no log line)"
  echo "  D1b lends during the concurrent run: $(tr '\r' '\n' < $L/a4v_a4.log | grep -c 'lends GPU')"
else
  fail "new engine (A4=3) did not start"
fi
stop_eng

stamp "gate"
cmp old new0 "default path: old vs new (A4 off)" | tee $L/a4v_cmp1.txt
grep -q "^CMP PASS" $L/a4v_cmp1.txt || fail "default path changed"
cmp new0 new0conc "A4 off: serial vs 2-way concurrent" | tee $L/a4v_cmp0.txt
grep -q "^CMP PASS" $L/a4v_cmp0.txt || fail "default path concurrent differs from serial"
cmp a4seq a4conc "A4=3: serial vs 2-way concurrent" | tee $L/a4v_cmp2.txt
grep -q "^CMP PASS" $L/a4v_cmp2.txt || fail "A4 concurrent differs from serial"
cmp new0 a4seq "info: A4 off vs A4=3 (expected to differ somewhere)" | grep -v "^CMP"

echo
if (( ${#FAILS[@]} )); then
  printf '  - %s\n' "${FAILS[@]}"
  echo "A4_VERIFY FAIL"; exit 1
fi
echo "A4_VERIFY PASS"
