#!/usr/bin/env bash
# 一键验证 F1（API 特殊 token 注入，dev-docs/DEEPINFER-NOTES.md §4c）。
#   bash tools/f1_verify.sh
# 只编 API 侧（g++，不碰 GPU，不用停生产服务）到临时目录，然后跑：
#   inject_test        字面控制串不再变成控制 token；无字面控制串的 prompt 编码与旧版逐 id 相同
#   snap_cuts_test     切点计算不变
#   api_inject_tcache_test  token 缓存不把伪造前缀复用成真控制 token；正常多轮仍复用
#   api_regression / tokenize / think_guard / notools_ban  fake 引擎回归
# 耗时约 1–2 分钟。最后一行：F1 VERIFY: PASS 或 FAIL。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"
CXX="${CXX:-g++}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
FLAGS=(-O2 -std=c++17 -DQWENOX_VERSION_RAW=f1verify -DQWENOX_GIT_RAW=f1verify -Isrc/api
       -Ithird_party -Ithird_party/miniz -Wall -Wextra -Wpedantic -Werror
       -Wno-format-truncation)  # g++ 11 误报 json_py.cpp 的 snprintf，与本项无关
fail=0
step() { echo; echo "== $*"; }

step "编译（$OUT）"
VFLAGS=()
LIBS=(-lpng -ljpeg -lwebp -lpthread)
# 没有 libjpeg 头文件时仅 PNG/JPEG 改走 stb；共享 WebP 解码仍需 libwebp。
if ! echo '#include <jpeglib.h>' | "$CXX" -x c++ -E - >/dev/null 2>&1; then
  VFLAGS=(-D_WIN32 -Wno-error); LIBS=(-lwebp -lpthread)
  echo "（无 jpeglib.h：PNG/JPEG 用 stb 分支，WebP 仍链接 libwebp）"
fi
"$CXX" -O2 -std=c++17 tools/gen_static_inc.cpp -o "$OUT/gen_static_inc" &&
  "$OUT/gen_static_inc" src/api/static src/api/static_gen.inc || fail=1
objs=()
for m in miniz miniz_tdef miniz_tinfl miniz_zip; do
  gcc -O2 -x c -Ithird_party/miniz -c "third_party/miniz/$m.c" -o "$OUT/$m.o" 2>/dev/null || fail=1
  objs+=("$OUT/$m.o")
done
pids=()
SRCS=(http engine_client tokenizer chat_template json_py toolparse vision reqstat reqstat_read
      power engine_sup snap_cuts reqlog main inject_test snap_cuts_test)
for f in "${SRCS[@]}"; do
  extra=()
  [[ $f == vision ]] && extra=("${VFLAGS[@]}")
  "$CXX" "${FLAGS[@]}" "${extra[@]}" -c "src/api/$f.cpp" -o "$OUT/$f.o" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if (( fail == 0 )); then
  o() { for f in "$@"; do echo "$OUT/$f.o"; done; }
  "$CXX" $(o http engine_client tokenizer chat_template json_py toolparse vision reqstat \
           reqstat_read power engine_sup snap_cuts reqlog main) "${objs[@]}" "${LIBS[@]}" \
    -o "$OUT/qwenox-api" || fail=1
  "$CXX" $(o chat_template json_py tokenizer vision inject_test) "${LIBS[@]}" \
    -o "$OUT/inject_test" || fail=1
  "$CXX" $(o snap_cuts chat_template json_py tokenizer vision snap_cuts_test) "${LIBS[@]}" \
    -o "$OUT/snap_cuts_test" || fail=1
fi
if (( fail )); then echo "编译失败"; echo "F1 VERIFY: FAIL"; exit 1; fi

step inject_test
"$OUT/inject_test" data/tokenizer 3000 || fail=1
step snap_cuts_test
"$OUT/snap_cuts_test" data/tokenizer 2000 || fail=1
step api_inject_tcache_test
python3 tools/api_inject_tcache_test.py --api "$OUT/qwenox-api" | tail -3 || fail=1
[[ ${PIPESTATUS[0]} == 0 ]] || fail=1
step api_think_guard_test
python3 tools/api_think_guard_test.py --api "$OUT/qwenox-api" | tail -1
[[ ${PIPESTATUS[0]} == 0 ]] || fail=1
step api_notools_ban_test
python3 tools/api_notools_ban_test.py --api "$OUT/qwenox-api" | tail -1
[[ ${PIPESTATUS[0]} == 0 ]] || fail=1

step "api_regression_test + api_tokenize_test（fake 引擎）"
port() { python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }
EP="$(port)"; AP="$(port)"
python3 tools/api_fake_engine.py --port "$EP" >"$OUT/fake.log" 2>&1 &
FE=$!
sleep 0.5
QWENOX_API_TOKCACHE_FILE= QWENOX_API_ADMIN_KEY= QWENOX_REQSTAT=0 ROPE_FACTOR=1 \
  "$OUT/qwenox-api" --tokenizer data/tokenizer --engine "127.0.0.1:$EP" --host 127.0.0.1 \
  --port "$AP" --overrides "" >"$OUT/api.log" 2>&1 &
API=$!
for _ in $(seq 100); do curl -s "127.0.0.1:$AP/health" >/dev/null && break; sleep 0.1; done
python3 tools/api_regression_test.py --base "http://127.0.0.1:$AP" | tail -1
[[ ${PIPESTATUS[0]} == 0 ]] || fail=1
python3 tools/api_tokenize_test.py --base "http://127.0.0.1:$AP" | tail -1
[[ ${PIPESTATUS[0]} == 0 ]] || fail=1
kill "$API" "$FE" 2>/dev/null
wait "$API" "$FE" 2>/dev/null

echo
if (( fail )); then echo "F1 VERIFY: FAIL"; exit 1; fi
echo "F1 VERIFY: PASS"
