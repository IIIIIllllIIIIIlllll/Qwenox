#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/tools/service_processes.sh"
W="$(mktemp -d /tmp/qwenox-process-test.XXXXXX)"
pids=()
cleanup() {
  local pid
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  rm -rf -- "$W"
}
trap cleanup EXIT

cp "$(command -v bash)" "$W/ptyxis"
mkdir "$W/qwenox-0.1.4-linux-x64"
"$W/ptyxis" -c 'sleep 30; :' --working-directory "$W/qwenox-0.1.4-linux-x64" &
pids+=("$!")
sleep 0.1
if processes="$(running_service_processes)"; then
  echo "误报：$processes" >&2
  exit 1
fi
echo 'OK 终端参数中的 qwenox 目录不触发误报'

cp "$(command -v bash)" "$W/qwenox-engine"
"$W/qwenox-engine" -c 'sleep 30; :' &
pids+=("$!")
sleep 0.1
processes="$(running_service_processes)" || { echo '漏报：qwenox-engine' >&2; exit 1; }
grep -q 'qwenox-engine' <<<"$processes" || { echo "输出缺少 qwenox-engine：$processes" >&2; exit 1; }
echo 'OK 实际 qwenox-engine 可执行文件会被检测'
kill "${pids[-1]}"
wait "${pids[-1]}" 2>/dev/null || true
unset 'pids[-1]'

cat >"$W/serve_api.py" <<'PY'
import time
time.sleep(30)
PY
python3 "$W/serve_api.py" &
pids+=("$!")
sleep 0.1
processes="$(running_service_processes)" || { echo '漏报：serve_api.py' >&2; exit 1; }
grep -q 'serve_api.py' <<<"$processes" || { echo "输出缺少 serve_api.py：$processes" >&2; exit 1; }
echo 'OK Python serve_api.py 会被检测'
