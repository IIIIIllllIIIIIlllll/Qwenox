#!/usr/bin/env bash
# Windows (TheRock) 启动入口：与 Linux start.sh 并列。
# 起引擎（行协议 ENGINE_PORT，默认 8730）+ OpenAI HTTP API 前端
# （qwenox-win，API_PORT，默认 8731）；Ctrl+C 同时停两者。
#
# 用法:
#   bash start_win.sh            # 前台运行，Ctrl+C 停止
#   bash start_win.sh --check    # 只检查配置，不启动
#
# 覆盖（service.conf 风格，环境变量优先）:
#   MODEL_FILE=../path/x.hgn MTP_FILE="" VISION_FILE="" ENGINE_PORT=8730 \
#   API_PORT=8731 MAX_CONTEXT=262144 MTP_GAMMA=0 bash start_win.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
# 部署参数（模型/端口/上下文等）读根目录 service.conf，与 Linux start.sh、
# Windows start_win.exe 同一文件；环境变量优先。计时类平台项不取 conf
#（Windows 冷加载分钟级，START_TIMEOUT 内置 1800）。
if [[ -f service.conf ]]; then
  _env_start_timeout="${START_TIMEOUT-}"
  source service.conf
  START_TIMEOUT="${_env_start_timeout:-1800}"
fi
fail() { echo "错误：$*" >&2; exit 1; }
[[ $# -le 1 ]] || fail '用法：bash start_win.sh [--check]'
case "${1:-}" in
  -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
  ''|--check) ;;
  *) fail '用法：bash start_win.sh [--check]' ;;
esac

# build/ 自包含：HIP/rocBLAS/hipBLASLt DLL 与 gfx1151 kernel db 都在 exe 旁，
# 无需 TheRock/ROCm 运行库，也不需要 HIP_PATH/ROCM_PATH（已断根实测）。

MODEL_DIR="${MODEL_DIR:-./models}"
# 权重版本：service.conf 的 "# start_win: weights=gguf" 标记（设置页/网页选 GGUF
# 时写入）或环境变量 WEIGHTS=gguf → 用 GGUF_FILE 首分片纯 GGUF 启动，与托盘
# qwenox-win 的 enginesup::start 同一语义。
WEIGHTS="${WEIGHTS:-$(sed -n 's/^# start_win: weights=\([a-z0-9]*\).*/\1/p' service.conf 2>/dev/null | head -1)}"
if [[ "$WEIGHTS" == gguf ]]; then
  [[ -n "${GGUF_FILE:-}" ]] || fail 'weights=gguf 但 GGUF_FILE 未设置（填第 1 个分片 *-00001-of-0000N.gguf）'
  MODEL_FILE="$GGUF_FILE"
  NGRAM_FILE="$GGUF_FILE"
  OVERLAY_FILE=
  GGUF_MTP_FILE="${GGUF_MTP_FILE-}"
  VISION_FILE="${GGUF_VISION_FILE-}"
  MTP_FILE=
else
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/heretic.hgn}"
# PLE n-gram 表所在文件：默认同 MODEL_FILE（w4b 单文件）；v2 指向独立的 *-ngram.hgn。
NGRAM_FILE="${NGRAM_FILE:-$MODEL_FILE}"
MTP_FILE="${MTP_FILE-$MODEL_DIR/heretic-mtp.hgn}"
VISION_FILE="${VISION_FILE-$MODEL_DIR/heretic-vision.hgn}"
fi
# 覆盖层（可选的高精度替换张量）：默认空=不叠加；非空但文件不存在则警告并跳过。
OVERLAY_FILE="${OVERLAY_FILE-}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$MODEL_DIR/tokenizer}"
ENGINE_HOST="${ENGINE_HOST:-127.0.0.1}"
ENGINE_PORT="${ENGINE_PORT:-8730}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8731}"
MAX_CONTEXT="${MAX_CONTEXT:-262144}"
ROPE_FACTOR="${ROPE_FACTOR:-1}"
ROPE_ORIGINAL_CTX="${ROPE_ORIGINAL_CTX:-262144}"
ROPE_BETA_FAST="${ROPE_BETA_FAST:-32}"
ROPE_BETA_SLOW="${ROPE_BETA_SLOW:-1}"
ROPE_ATTN_SCALE="${ROPE_ATTN_SCALE:-0}"
MTP_GAMMA="${MTP_GAMMA:-0}"
KVSNAP_MAX_GB="${KVSNAP_MAX_GB:-20}"
KVSNAP_MIN="${KVSNAP_MIN:-4096}"
KVSNAP_PARK_MIN="${KVSNAP_PARK_MIN:-$KVSNAP_MIN}"
PARK="${PARK:-1}"
PARK_WAIT_MS="${PARK_WAIT_MS:-5000}"
PARK_MAX_ATTEMPTS="${PARK_MAX_ATTEMPTS:-8}"
KV_RESERVE_DECODE="${KV_RESERVE_DECODE:-4096}"
RCKPT_MAX="${RCKPT_MAX:-8}"
# 分页 KV（见 service.conf）：默认开启，页池 = 一条 MAX_CONTEXT 序列。
# Windows 的 arena 上限 95 GiB：v2 权重 512K 池（KV_POOL_TOKENS=524288）用 92–94 GiB，放得下；
# w4b 只放得下 256K。
KV_PAGED="${KV_PAGED:-1}"
KV_POOL_TOKENS="${KV_POOL_TOKENS:-0}"
PARALLEL="${PARALLEL:-1}"
# 单请求图片数上限（1–256）：多轮绘图/视觉对话会累计历史图片，按需调大。
MAX_IMAGES="${MAX_IMAGES:-8}"
# 本机 68 GiB 权重 cold-load 实测 ~9 分钟（NVMe 弱盘），超时给足。
START_TIMEOUT="${START_TIMEOUT:-1800}"

[[ "$ENGINE_HOST" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || fail 'ENGINE_HOST 必须是 IPv4 地址'
IFS=. read -r -a engine_octets <<< "$ENGINE_HOST"
for octet in "${engine_octets[@]}"; do
  [[ "$octet" =~ ^(0|[1-9][0-9]{0,2})$ ]] && (( 10#$octet <= 255 )) || fail 'ENGINE_HOST 必须是 IPv4 地址'
done
engine_connect_host="$ENGINE_HOST"
if [[ "$engine_connect_host" == 0.0.0.0 ]]; then engine_connect_host=127.0.0.1; fi
[[ "$ENGINE_PORT" =~ ^[1-9][0-9]*$ && "$ENGINE_PORT" -le 65535 ]] || fail 'ENGINE_PORT 必须为 1–65535'
[[ "$API_PORT" =~ ^[1-9][0-9]*$ && "$API_PORT" -le 65535 ]] || fail 'API_PORT 必须为 1–65535'
[[ "$ENGINE_PORT" != "$API_PORT" ]] || fail 'ENGINE_PORT 与 API_PORT 必须不同'
[[ "$MTP_GAMMA" =~ ^[0-8]$ ]] || fail 'MTP_GAMMA 范围为 0–8（0=引擎按模式自选 greedy 4 / 采样自适应）'
[[ "$KVSNAP_MAX_GB" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'KVSNAP_MAX_GB 必须为非负整数'
for key in KVSNAP_MIN KVSNAP_PARK_MIN PARK_WAIT_MS PARK_MAX_ATTEMPTS KV_RESERVE_DECODE; do
  value="${!key}"
  [[ "$value" =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 8 ]] || fail "$key 必须为非负整数"
done
[[ "$PARK" =~ ^[01]$ ]] || fail 'PARK 必须为 0 或 1'
(( PARK_WAIT_MS >= 500 )) || fail 'PARK_WAIT_MS 必须至少为 500 ms'
(( PARK_MAX_ATTEMPTS >= 1 && PARK_MAX_ATTEMPTS <= 32 )) || fail 'PARK_MAX_ATTEMPTS 范围为 1–32'
[[ "$RCKPT_MAX" =~ ^(0|[1-9][0-9]*)$ ]] || fail 'RCKPT_MAX 必须为非负整数'
[[ "$KV_PAGED" =~ ^[01]$ ]] || fail 'KV_PAGED 必须为 0 或 1'
[[ "$KV_POOL_TOKENS" =~ ^(0|[1-9][0-9]*)$ && ${#KV_POOL_TOKENS} -le 8 ]] || fail 'KV_POOL_TOKENS 必须为非负整数'
[[ "$PARALLEL" =~ ^[1-8]$ ]] || fail 'PARALLEL 范围为 1–8'
[[ "${MOE_A4:-0}" =~ ^[0-3]$ ]] || fail 'MOE_A4 范围为 0–3'
[[ "$MAX_IMAGES" =~ ^[1-9][0-9]*$ && "$MAX_IMAGES" -le 256 ]] || fail 'MAX_IMAGES 范围为 1–256'
[[ "$ROPE_ORIGINAL_CTX" =~ ^[1-9][0-9]*$ ]] || fail 'ROPE_ORIGINAL_CTX 必须为正整数'
[[ "$ROPE_FACTOR" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_FACTOR 必须为非负小数'
[[ "$ROPE_BETA_FAST" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_BETA_FAST 必须为非负小数'
[[ "$ROPE_BETA_SLOW" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_BETA_SLOW 必须为非负小数'
[[ "$ROPE_ATTN_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'ROPE_ATTN_SCALE 必须为非负小数'
awk "BEGIN { exit !($ROPE_FACTOR >= 1 && $ROPE_BETA_FAST > 0 && $ROPE_BETA_SLOW > 0 && $ROPE_ATTN_SCALE >= 0) }" || \
  fail 'ROPE_FACTOR 必须 >=1，beta 必须 >0，ROPE_ATTN_SCALE 必须 >=0'
# 与引擎 52_main.inc 一致：YaRN 开启时 MAX_CONTEXT 上限 = ROPE_FACTOR ×
# ROPE_ORIGINAL_CTX；并发容量用 KV_POOL_TOKENS 扩，不要拉伸 MAX_CONTEXT。
awk "BEGIN { exit !($ROPE_FACTOR <= 1 || $MAX_CONTEXT <= $ROPE_FACTOR * $ROPE_ORIGINAL_CTX) }" || \
  fail "MAX_CONTEXT=$MAX_CONTEXT 超出 YaRN 上限 ROPE_FACTOR×ROPE_ORIGINAL_CTX；调小 MAX_CONTEXT 或调大 ROPE_FACTOR（并发容量用 KV_POOL_TOKENS 扩）"
awk "BEGIN { exit !($ROPE_FACTOR > 1 || $MAX_CONTEXT <= $ROPE_ORIGINAL_CTX) }" || \
  echo "警告：ROPE_FACTOR=1 且 MAX_CONTEXT=$MAX_CONTEXT 超过原生 $ROPE_ORIGINAL_CTX，超出部分的位置编码未验证（需要时请设 ROPE_FACTOR）" >&2
(( PARALLEL == 1 || KV_PAGED )) || fail 'PARALLEL>1 需要 KV_PAGED=1'
[[ -f build/qwenox-engine-win.exe ]] || fail '缺少 build/qwenox-engine-win.exe，请先运行 bash build_win.sh'
[[ -f build/qwenox-win.exe ]] || fail '缺少 build/qwenox-win.exe，请先运行 bash build_win.sh api'
[[ -r "$MODEL_FILE" ]] || fail "找不到模型：$MODEL_FILE"
[[ -r "$NGRAM_FILE" ]] || fail "找不到 n-gram 表：$NGRAM_FILE（w4b 与 MODEL_FILE 相同）"
[[ -z "$MTP_FILE" || -r "$MTP_FILE" ]] || fail "找不到 MTP 权重：$MTP_FILE"
if [[ "$WEIGHTS" == gguf ]]; then
  [[ -z "$GGUF_MTP_FILE" || -r "$GGUF_MTP_FILE" ]] || fail "找不到 GGUF MTP sidecar：$GGUF_MTP_FILE"
fi
[[ -z "$VISION_FILE" || -r "$VISION_FILE" ]] || fail "找不到视觉塔：$VISION_FILE（纯文本可设 VISION_FILE=\"\"）"
if [[ -n "$OVERLAY_FILE" && ! -r "$OVERLAY_FILE" ]]; then  # 可选：只警告并跳过
  echo "警告：找不到 overlay：$OVERLAY_FILE，已跳过（v2 权重不用 overlay；不想看到这条可设 OVERLAY_FILE=\"\"）" >&2
  OVERLAY_FILE=
fi
[[ -r "$TOKENIZER_DIR/tokenizer.json" ]] || fail "找不到 tokenizer：$TOKENIZER_DIR"
[[ -d build/rocblas/library && -d build/hipblaslt/library ]] || fail '缺少 rocBLAS/hipBLASLt kernel db（build/*/library），请先运行 bash build_win.sh'
for port in "$ENGINE_PORT" "$API_PORT"; do
  if netstat -an | grep -E "[:.]$port\s+.*LISTENING" >/dev/null 2>&1; then
    fail "端口 $port 已被占用"
  fi
done
echo "项目：$ROOT"
echo "模型：$MODEL_FILE$([[ "$WEIGHTS" == gguf ]] && echo "（GGUF，MTP：${GGUF_MTP_FILE:-不使用}）")"
echo "配置：${MAX_CONTEXT} 上下文，MTP gamma=$([[ $MTP_GAMMA == 0 ]] && echo 'auto（greedy 4 / 采样自适应）' || echo "$MTP_GAMMA")，engine ${ENGINE_HOST}:${ENGINE_PORT}，API ${API_HOST}:${API_PORT}"
if (( KV_PAGED )); then
  echo "KV：分页，页池 $(( (KV_POOL_TOKENS > MAX_CONTEXT ? KV_POOL_TOKENS : MAX_CONTEXT) )) token（${PARALLEL} 路并发共享），RAM 检查点 ${RCKPT_MAX} 个"
  (( PARALLEL == 1 )) || echo "提示：每多一路并发约多占 0.27 GiB 设备内存，arena（95 GiB 上限）放不下的部分会回退 hipMalloc" >&2
  (( KV_POOL_TOKENS <= MAX_CONTEXT )) || echo "警告：KV_POOL_TOKENS 大于 MAX_CONTEXT，Windows arena（95 GiB 上限）可能放不下，超出部分会回退 hipMalloc" >&2
else
  echo "KV：不分页（KV_PAGED=0）"
fi
if [[ "${1:-}" == --check ]]; then
  echo '检查通过；没有启动引擎或 API。'
  exit 0
fi

# 生产选项与 Linux start.sh 一致，唯独不设 QWENOX_PREFILL_CHUNK：
# Windows 默认 6144（256K 下 16384 会顶破 95 GiB arena 上限；6144 与 8192 的 PP 差
# 约 0.5%，工作区少约 1.7 GiB，实测见 PORTING-WINDOWS.md）。
export QWENOX_QSA_KV_BF16=1 QWENOX_QSA_WMMA=1 QWENOX_QSA_WMMA_BTV=1
export QWENOX_MOE_LT=1 QWENOX_MOE_LT_BF16=1 QWENOX_GR_BF16=1
export QWENOX_GDN_STREAM=1 QWENOX_GDN_WAVE=1 QWENOX_NOWARMUP=1
export QWENOX_GEMM_WMMA=1 QWENOX_GDN_FUSED=1
export QWENOX_INDEX_FUSED2=1 QWENOX_PP_MOE_OUT=1 QWENOX_INDEX_STREAM_SELECT=1
if (( KVSNAP_MAX_GB )); then export QWENOX_KVSNAP=1; else export QWENOX_KVSNAP=0; fi
export QWENOX_KVSNAP_MAX_GB="$KVSNAP_MAX_GB" QWENOX_KVSNAP_MIN="$KVSNAP_MIN" \
  QWENOX_KVSNAP_PARK_MIN="$KVSNAP_PARK_MIN" QWENOX_RCKPT_MAX="$RCKPT_MAX"
if (( PARK )); then export QWENOX_PARK=1; else export QWENOX_PARK=0; fi
export QWENOX_PARK_WAIT_MS="$PARK_WAIT_MS" QWENOX_PARK_MAX_ATTEMPTS="$PARK_MAX_ATTEMPTS" \
  QWENOX_KV_RESERVE_DECODE="$KV_RESERVE_DECODE"
# KVSNAP_DIR（见 service.conf）留空时不动 QWENOX_KVSNAP_DIR
KVSNAP_DIR="${KVSNAP_DIR-}"
if [[ -n "$KVSNAP_DIR" ]]; then
  KVSNAP_DIR="${KVSNAP_DIR%/}"; export QWENOX_KVSNAP_DIR="${KVSNAP_DIR%\\}/qwenox-kvsnap"
fi
# MTP_GAMMA=0：不导出（并清掉外部残留），引擎按请求模式自选（greedy 4 / 采样自适应）。
if (( MTP_GAMMA > 0 )); then export QWENOX_SPEC_GAMMA="$MTP_GAMMA"; else unset QWENOX_SPEC_GAMMA; fi
unset QWENOX_KV_PAGED QWENOX_KV_POOL_TOKENS
if (( KV_PAGED )); then
  export QWENOX_KV_PAGED=1
  if (( KV_POOL_TOKENS )); then export QWENOX_KV_POOL_TOKENS="$KV_POOL_TOKENS"; fi
fi
export QWENOX_PARALLEL="$PARALLEL"
# VIZ=1：运行时可视化埋点（GET /viz，页面 /viz-live，见 docs/VIZ.md）
if (( ${VIZ:-0} )); then export QWENOX_VIZ=1; else unset QWENOX_VIZ; fi
# MOE_A4=1–3：prefill MoE 4-bit 激活（int4 WMMA，见 service.conf）
if (( ${MOE_A4:-0} )); then export QWENOX_MOE_A4="$MOE_A4"; else unset QWENOX_MOE_A4; fi
export QWENOX_API_MAX_IMAGES="$MAX_IMAGES"
export QWENOX_ROPE_FACTOR="$ROPE_FACTOR"
export QWENOX_ROPE_ORIGINAL_CTX="$ROPE_ORIGINAL_CTX"
export QWENOX_ROPE_BETA_FAST="$ROPE_BETA_FAST"
export QWENOX_ROPE_BETA_SLOW="$ROPE_BETA_SLOW"
export QWENOX_ROPE_ATTN_SCALE="$ROPE_ATTN_SCALE"
unset QWENOX_GGUF QWENOX_GGUF_DENSE QWENOX_GGUF_PLE QWENOX_GGUF_EXPERTS QWENOX_GGUF_MTP
if [[ "$WEIGHTS" == gguf ]]; then
  # MTP sidecar 由环境给出：空 = none（不投机，也不自动找 mtp-*.gguf）
  export QWENOX_GGUF_MTP="${GGUF_MTP_FILE:-none}"
fi

mkdir -p logs
ENGINE_LOG="logs/engine-win-$(date +%Y%m%d-%H%M%S).log"

engine=(build/qwenox-engine-win.exe "$MODEL_FILE")
[[ -z "$OVERLAY_FILE" ]] || engine+=("$OVERLAY_FILE")
[[ "$NGRAM_FILE" -ef "$MODEL_FILE" ]] || engine+=("$NGRAM_FILE")
[[ -z "$MTP_FILE" ]] || engine+=("$MTP_FILE")
engine+=(--serve --host "$ENGINE_HOST" --port "$ENGINE_PORT" --maxctx "$MAX_CONTEXT")
[[ -z "$VISION_FILE" ]] || engine+=(--vision-tower "$VISION_FILE")

echo "加载模型中，日志：$ENGINE_LOG；Ctrl+C 停止。"
"${engine[@]}" >"$ENGINE_LOG" 2>&1 &
engine_pid=$!
api_pid=''
API_LOG="logs/api-win-$(date +%Y%m%d-%H%M%S).log"
cleanup() {
  trap - EXIT INT TERM
  for pid in $api_pid $engine_pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" 2>/dev/null || true
    taskkill //PID "$pid" //F //T >/dev/null 2>&1 || true
  done
  echo "本次服务已停止。日志：$ENGINE_LOG $API_LOG"
}
trap cleanup EXIT INT TERM

begin=$SECONDS
until grep -q 'serve: listening' "$ENGINE_LOG" 2>/dev/null; do
  kill -0 "$engine_pid" 2>/dev/null || { tail -n 15 "$ENGINE_LOG" >&2; fail '引擎提前退出'; }
  (( SECONDS - begin < START_TIMEOUT )) || fail "引擎启动超过 ${START_TIMEOUT} 秒，查看 $ENGINE_LOG"
  sleep 2
done

build/qwenox-win.exe --console --tokenizer "$TOKENIZER_DIR" \
  --engine "$engine_connect_host:$ENGINE_PORT" --host "$API_HOST" \
  --port "$API_PORT" --context "$MAX_CONTEXT" >"$API_LOG" 2>&1 &
api_pid=$!
begin=$SECONDS
until netstat -an | grep -E "[:.]$API_PORT\s+.*LISTENING" >/dev/null 2>&1; do
  kill -0 "$engine_pid" 2>/dev/null || fail '引擎已退出'
  kill -0 "$api_pid" 2>/dev/null || { tail -n 15 "$API_LOG" >&2; fail 'API 提前退出'; }
  (( SECONDS - begin < 30 )) || fail "API 启动超时，查看 $API_LOG"
  sleep 1
done

echo "服务已就绪：http://${API_HOST}:${API_PORT}/v1（0.0.0.0 表示监听所有网卡）"
echo "日志：$ENGINE_LOG $API_LOG；Ctrl+C 同时停止 API 和引擎。"
if wait -n "$engine_pid" "$api_pid"; then
  fail '服务进程意外结束'
else
  result=$?
  echo "服务进程退出（$result），查看日志" >&2
  exit "$result"
fi
