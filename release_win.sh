#!/usr/bin/env bash
# release_win.sh — Windows 发行包：根目录编译 + 组装 dist/（含 PDB 与 bench token，可选 zip）。
#
# 编译调用 build_win.sh all（引擎、bench、API 托盘程序、start_win、settings），
# 然后把运行所需文件复制到 dist/<包名>/。发行包不含模型权重；PDB 调试符号默认
# 随包放进 build/，用户崩溃时用同版本的 PDB 还原 minidump。用法见 --help。
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
    cat <<'EOF'
用法: bash release_win.sh [选项]

编译（build_win.sh all）后组装 Windows 发行文件夹；PDB 调试符号默认随包分发。

选项:
  --release          正式包名：去掉 -preview（默认带 -preview）
  --name NAME        自定义包名（默认 qwenox-<版本>-win64-preview）
  --zip              额外生成 dist/<包名>.zip
  --stop             编译前结束正在运行的 qwenox-engine-win.exe / qwenox-win.exe
                     （只停止，不会自动重启服务）
  --skip-build       跳过编译，直接用 build/ 现有产物打包
  --force            目标文件夹已存在时覆盖
  --no-symbols       不包含 PDB 调试符号（包更小；正式发布不推荐）
  -h, --help         显示本帮助

产物:
  dist/<包名>/                          可直接拷到同架构 Windows 机器运行
  dist/<包名>/build/*.pdb               调试符号（崩溃 minidump 还原用，随包）
  dist/<包名>.zip                       --zip 时（含 PDB）

说明:
  - 编译在项目根目录进行；运行中的服务会锁住 build/ 下的 exe，脚本默认报错退出，
    加 --stop 可先结束服务（先停 API，再停引擎）。
  - 发行包不含模型权重，但默认包含 PDB 调试符号和 benchmark 的
    data/qsa-oracle/*.tokens。models/ 只有 README 占位；data/tokenizer 随包分发。
  - 包内 service.conf 原样复制（含 PARALLEL / KV_POOL_TOKENS / MOE_A4 等当前值）；
    BUILD-INFO.txt 会记录本次构建的关键配置与时间。
EOF
}

die() { echo "错误：$*" >&2; exit 1; }

PREVIEW=1
MAKE_ZIP=0
STOP_SERVICE=0
SKIP_BUILD=0
FORCE=0
SYMBOLS=1
NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release) PREVIEW=0 ;;
        --zip) MAKE_ZIP=1 ;;
        --stop) STOP_SERVICE=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        --force) FORCE=1 ;;
        --no-symbols) SYMBOLS=0 ;;
        --name)
            [[ $# -ge 2 ]] || die "--name 需要一个包名"
            NAME="$2"
            shift
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
    shift
done

VER="$(tr -d ' \r\n' < VERSION 2>/dev/null || true)"
[[ "$VER" =~ ^[0-9A-Za-z.-]+$ ]] || die "VERSION 文件缺失或格式不对（应形如 0.0.8）"
if [[ -z "$NAME" ]]; then
    NAME="qwenox-$VER-win64"
    # set -e: keep this branch's status 0 when PREVIEW=0
    if (( PREVIEW )); then NAME+="-preview"; fi
fi
[[ "$NAME" =~ ^[0-9A-Za-z._-]+$ ]] || die "包名只允许字母、数字、点、下划线、连字符"
OUT="dist/$NAME"
case "$OUT" in
    dist/*) ;;
    *) die "内部错误：输出路径不在 dist/ 下（$OUT）" ;;
esac
if (( ! MAKE_ZIP )) && [[ -f "$OUT.zip" ]]; then
    echo "提示：$OUT.zip 已存在，本次未加 --zip，保留旧 zip 不动" >&2
fi

GITREV="$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
GPU_ARCH="${GPU_ARCH:-gfx1151}"
TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"

# ---- 运行中的服务会锁住 build/ 下的 exe -------------------------------------
is_running() {  # $1=exe 名
    tasklist //FI "IMAGENAME eq $1" 2>/dev/null | grep -qi "$1"
}

if (( ! SKIP_BUILD )); then
    procs=()
    is_running qwenox-engine-win.exe && procs+=(qwenox-engine-win.exe)
    is_running qwenox-win.exe && procs+=(qwenox-win.exe)
    if (( ${#procs[@]} )); then
        if (( STOP_SERVICE )); then
            echo "[停止] ${procs[*]}（--stop）"
            taskkill //IM qwenox-win.exe //F >/dev/null 2>&1 || true
            taskkill //IM qwenox-engine-win.exe //F >/dev/null 2>&1 || true
            sleep 2
            is_running qwenox-engine-win.exe && die "无法结束 qwenox-engine-win.exe，请手动退出后再试"
            is_running qwenox-win.exe && die "无法结束 qwenox-win.exe，请手动退出后再试"
        else
            die "检测到正在运行的服务：${procs[*]}
      build/ 下的 exe 被占用，编译无法覆盖。
      先退出托盘程序/控制台服务，或加 --stop 让脚本结束它们。
      只想用现有 build/ 打包可加 --skip-build。"
        fi
    fi
fi

# ---- 编译 -------------------------------------------------------------------
mkdir -p build logs dist
BUILD_LOG="logs/release-win-$(date +%Y%m%d-%H%M%S).log"
if (( SKIP_BUILD )); then
    echo "[跳过编译] 使用现有 build/ 产物"
else
    [[ -x "$TR/bin/hipcc.exe" ]] || die "找不到 TheRock hipcc：$TR/bin/hipcc.exe（设 THEROCK=...）"
    [[ -x "$TR/lib/llvm/bin/clang++.exe" ]] || die "找不到 TheRock clang++：$TR/lib/llvm/bin/clang++.exe"
    echo "[编译] bash build_win.sh all（日志：$BUILD_LOG）"
    if ! bash build_win.sh all 2>&1 | tee "$BUILD_LOG"; then
        die "编译失败，详见 $BUILD_LOG"
    fi
fi

for f in build/qwenox-engine-win.exe build/qwenox-win.exe build/qwenox-bench.exe \
         start_win.exe settings_win.exe; do
    [[ -s "$f" ]] || die "缺少编译产物：$f（先运行 bash build_win.sh all）"
done
[[ -d "build/rocblas/library/$GPU_ARCH" ]] || die "缺少 rocBLAS kernel db：build/rocblas/library/$GPU_ARCH"
[[ -d "build/hipblaslt/library/$GPU_ARCH" ]] || die "缺少 hipBLASLt kernel db：build/hipblaslt/library/$GPU_ARCH"

# ---- 词表：优先仓库内置 data/tokenizer，其次 models/tokenizer ---------------
TOK_DIR=""
if [[ -f data/tokenizer/tokenizer.json ]]; then
    TOK_DIR=data/tokenizer
elif [[ -f models/tokenizer/tokenizer.json ]]; then
    TOK_DIR=models/tokenizer
else
    die "缺少词表：data/tokenizer/tokenizer.json（或 models/tokenizer/tokenizer.json）"
fi

# ---- 目标目录 ---------------------------------------------------------------
if [[ -e "$OUT" ]]; then
    if (( FORCE )); then
        echo "[覆盖] 删除已有 $OUT"
        rm -rf -- "$OUT"
    else
        die "目标已存在：$OUT（加 --force 覆盖，或用 --name 换名）"
    fi
fi
mkdir -p "$OUT/build/rocblas/library" "$OUT/build/hipblaslt/library" \
         "$OUT/data/tokenizer" "$OUT/models"

# ---- 运行文件（不含中间产物 / winlibs） ------------------------------------
RUNTIME_FILES=(
    qwenox-engine-win.exe qwenox-win.exe qwenox-bench.exe
    amdhip64_7.dll rocm_kpack.dll amd_comgr.dll rocblas.dll libhipblaslt.dll origami.dll
    msvcp140.dll vcruntime140.dll vcruntime140_1.dll
)
for f in "${RUNTIME_FILES[@]}"; do
    [[ -f "build/$f" ]] || die "缺少运行文件：build/$f"
    cp -a "build/$f" "$OUT/build/"
done
for f in COPYING PATENTS AUTHORS SOURCE.txt; do
    [[ -f "build/third-party-licenses/libwebp/$f" ]] || die "缺少 libwebp 许可文件：$f（请重建 Windows API）"
done
mkdir -p "$OUT/build/third-party-licenses"
cp -a build/third-party-licenses/libwebp "$OUT/build/third-party-licenses/"
cp -a "build/rocblas/library/$GPU_ARCH" "$OUT/build/rocblas/library/"
cp -a "build/hipblaslt/library/$GPU_ARCH" "$OUT/build/hipblaslt/library/"
# rocBLAS 顶层可能还有按架构命名的 fallback 分片（KB 级）
for f in build/rocblas/library/*"$GPU_ARCH"*; do
    [[ -f "$f" ]] || continue
    cp -a "$f" "$OUT/build/rocblas/library/"
done

for f in start_win.exe settings_win.exe start_win.sh service.conf \
         README.md README_EN.md CHANGELOG.md LICENSE VERSION; do
    [[ -e "$f" ]] || die "缺少根目录文件：$f"
    cp -a "$f" "$OUT/"
done
[[ -d docs ]] && cp -a docs "$OUT/"
[[ -d media ]] && cp -a media "$OUT/"

cp -a "$TOK_DIR/tokenizer.json" "$OUT/data/tokenizer/"
# qwenox-bench 的 prefill 基准直接读取 data/qsa-oracle/*.tokens；随包分发，
# 否则发行包里的 bench 会在 "no .tokens files found" 处失败。
if compgen -G "data/qsa-oracle/*.tokens" >/dev/null; then
    mkdir -p "$OUT/data/qsa-oracle"
    cp -a data/qsa-oracle/*.tokens "$OUT/data/qsa-oracle/"
    echo "[bench] 包含 $(ls -1 data/qsa-oracle/*.tokens | wc -l) 个 token 文件"
else
    die "缺少 data/qsa-oracle/*.tokens（qwenox-bench 的 prefill 基准输入）"
fi
if [[ -f data/tokenizer/README.md ]]; then
    cp -a data/tokenizer/README.md "$OUT/data/tokenizer/"
else
    cat > "$OUT/data/tokenizer/README.md" <<'EOF'
# data/tokenizer — 兜底词表

qwenox-api 在 `service.conf` 的 `TOKENIZER_DIR` 不可用时回退到本目录。
本目录只需 `tokenizer.json`（Qwen BPE 词表，约 20 MB）。
EOF
fi

cat > "$OUT/models/README.txt" <<'EOF'
模型权重目录（发行包不含权重，需要自行放入）
==============================================

默认使用 hgn v2（service.conf 末尾 "# start_win: weights=v2"）。
把下列文件放到本目录（models\）下：

  qwen38-flash-next-v2.hgn          主模型（62.1 GiB）
  qwen38-flash-next-ngram.hgn       PLE n-gram 表（47.7 GiB，常驻磁盘按需读取）
  qwen38-flash-next-mtp.hgn         8-bit MTP 草稿头
  qwen38-flash-next-vision.hgn      视觉塔；纯文本服务可删，并把
                                    service.conf 的 VISION_FILE 置空
  tokenizer\tokenizer.json          词表（hgn 与 GGUF 共用）

如果使用 GGUF（Windows 只支持 UD-IQ4_XS）：
  在控制台 #/engine 页把权重切到 GGUF，并按 README.md「支持的权重」一节
  下载 3 个分片、MTP sidecar 和视觉塔；路径写在 service.conf 的 GGUF_FILE /
  GGUF_MTP_FILE / GGUF_VISION_FILE。

下载命令、文件校验和更多说明见根目录 README.md「支持的权重」。
缺少文件时托盘/启动器会列出具体缺哪一项，不会带着半套权重启动。
EOF

# ---- PDB 调试符号（默认随包；--no-symbols 排除） ---------------------------
if (( SYMBOLS )); then
    pdb_n=0
    for f in build/*.pdb; do
        [[ -f "$f" ]] || continue
        cp -a "$f" "$OUT/build/"
        pdb_n=$((pdb_n + 1))
    done
    (( pdb_n )) || die "build/ 下没有 PDB（编译时设了 QWENOX_PDB=0？）。正式发布需要调试符号；如确定不要，加 --no-symbols"
    echo "[符号] 随包包含 $pdb_n 个 PDB"
else
    echo "[符号] --no-symbols，不包含 PDB"
fi

# ---- BUILD-INFO -------------------------------------------------------------
conf_val() {  # $1=service.conf 中的键；读文件里的默认值（不受当前环境变量影响）
    (
        set +u
        unset PARALLEL KV_POOL_TOKENS MAX_CONTEXT MOE_A4 KV_PAGED PREFILL_CHUNK
        # shellcheck disable=SC1091
        . ./service.conf >/dev/null 2>&1
        printf '%s' "${!1-}"
    )
}

PARALLEL_V="$(conf_val PARALLEL)"
KVPOOL_V="$(conf_val KV_POOL_TOKENS)"
MAXCTX_V="$(conf_val MAX_CONTEXT)"
MOEA4_V="$(conf_val MOE_A4)"
KVPAGED_V="$(conf_val KV_PAGED)"
PREFILL_V="$(conf_val PREFILL_CHUNK)"
WEIGHTS="$(sed -n 's/^# start_win: weights=\([a-z0-9]*\).*/\1/p' service.conf | head -1)"
[[ -n "$WEIGHTS" ]] || WEIGHTS="v2"

[[ "$PARALLEL_V" =~ ^[0-9]+$ ]] || PARALLEL_V="?"
[[ "$KVPOOL_V" =~ ^[0-9]+$ ]] || KVPOOL_V="?"
[[ "$MAXCTX_V" =~ ^[0-9]+$ ]] || MAXCTX_V="?"
[[ "$KVPAGED_V" =~ ^[01]$ ]] || KVPAGED_V="?"

case "$MOEA4_V" in
    1) A4_DESC="prefill 4-bit 激活，INT4 WMMA；仅 gate/up" ;;
    2) A4_DESC="prefill 4-bit 激活，INT4 WMMA；仅 down" ;;
    3) A4_DESC="prefill 4-bit 激活，INT4 WMMA；gate/up + down 全开" ;;
    0) A4_DESC="关闭" ;;
    *) A4_DESC="关闭，无法识别的值：$MOEA4_V" ;;
esac
if [[ "$KVPAGED_V" == 1 ]]; then
    KVPAGED_DESC="分页 KV，并发共享页池"
else
    KVPAGED_DESC="连续 KV，不分页"
fi
if [[ "$KVPOOL_V" == 0 ]]; then
    KVPOOL_DESC="0 = 等于单条 MAX_CONTEXT"
elif [[ "$MAXCTX_V" =~ ^[0-9]+$ && "$MAXCTX_V" -gt 0 ]] && (( KVPOOL_V % MAXCTX_V == 0 )); then
    KVPOOL_DESC="$(( KVPOOL_V / MAXCTX_V )) × MAX_CONTEXT 的共享页池"
else
    KVPOOL_DESC="全部并发槽位共享的 KV 页池"
fi
if [[ "$PREFILL_V" == 0 ]]; then
    PREFILL_DESC="0 = 引擎默认（Windows 6144 / Linux 16384）"
else
    PREFILL_DESC="每段 $PREFILL_V token"
fi

if (( PREVIEW )); then
    LABEL="预览发行版"
    PREVIEW_LINE="本包为预览版发行文件夹，不含模型权重（体积原因）。"
else
    LABEL="正式发行版"
    PREVIEW_LINE="本包为正式发行文件夹，不含模型权重（体积原因）。"
fi
if (( SYMBOLS )); then
    PDB_LINE="build\\ 内含 PDB 调试符号（qwenox-engine-win.pdb / qwenox-win.pdb），与 exe 同一次编译；收到崩溃 minidump 时用它还原函数名和源码行。"
else
    PDB_LINE="本包不含 PDB 调试符号（--no-symbols）；正式发布不推荐这样打包。"
fi

cat > "$OUT/BUILD-INFO.txt" <<EOF
Qwenox $VER — Windows x64 $LABEL
=====================================

构建时间：$(date '+%Y-%m-%d %H:%M:%S')（本机时间）
版本来源：VERSION = $VER（源码目录无 git 信息时提交号为 nogit）
提交：$GITREV
工具链：TheRock（hipcc / AMD clang，C++17，-O3）
目标 GPU：$GPU_ARCH（AMD Strix Halo / Ryzen AI Max+ 395）

本包 service.conf 关键配置
--------------------------
PARALLEL=$PARALLEL_V（同时运行的请求数）
KV_POOL_TOKENS=$KVPOOL_V（$KVPOOL_DESC）
MAX_CONTEXT=$MAXCTX_V（单条请求上限）
KV_PAGED=$KVPAGED_V（$KVPAGED_DESC）
MOE_A4=$MOEA4_V（$A4_DESC）
PREFILL_CHUNK=$PREFILL_V（$PREFILL_DESC）
weights=$WEIGHTS（默认启动权重）

目录内容
--------
start_win.exe             托盘入口：双击启动 API 与控制台网页，不自动加载模型
settings_win.exe          配置工具：编辑 service.conf、环境检查（不启动服务）
start_win.sh              Git Bash 命令行启动（行为同 Linux 启动器，可选）
service.conf              部署配置（同上表；环境变量可临时覆盖）
build\                    引擎 qwenox-engine-win.exe、API qwenox-win.exe、
                          benchmark qwenox-bench.exe、HIP/rocBLAS 运行库、
                          gfx1151 kernel db（自包含，无需安装 ROCm/TheRock）；
                          默认含 *.pdb 调试符号
data\tokenizer\           兜底词表 tokenizer.json（API 前端使用）
data\qsa-oracle\          benchmark 的 prefill token 文件（qwenox-bench 使用）
models\                   空目录，模型权重需自行放入（见 models\README.txt）
docs\、README.md          使用、配置、并发、YaRN、GGUF 等文档
media\                    README 引用的配图
SHA256SUMS.txt            包内主要文件的校验和

快速开始
--------
1. 把 hgn v2 权重与 tokenizer 放进 models\（文件名见 models\README.txt，
   或按 README.md「支持的权重」一节下载）。
2. 双击 start_win.exe。托盘图标右键：启动引擎；双击图标：打开控制台网页。
   也可以先在控制台 #/engine 页确认权重与参数，再启动引擎。
3. API 基址 http://127.0.0.1:8731/v1，控制台 http://127.0.0.1:8731/。

硬件前提：BIOS 为 GPU 划分 96 GiB 显存。参考占用：512K 页池（524288）+
2 路并发实测约 92 GiB；其它组合按每 256K 约 9.4 GiB、每路约 0.27 GiB 估算。

说明
----
- $PREVIEW_LINE
- $PDB_LINE
- MOE_A4 只影响 prefill（prompt/长输入），生成阶段不经过该路径；
  精度取舍与实测数字见 README.md「prefill 4-bit 加速」一节。
EOF

# ---- 校验和 -----------------------------------------------------------------
if command -v sha256sum >/dev/null 2>&1; then
    (
        cd "$OUT"
        files=(build/qwenox-engine-win.exe build/qwenox-win.exe build/qwenox-bench.exe
               start_win.exe settings_win.exe service.conf
               data/tokenizer/tokenizer.json BUILD-INFO.txt)
        for f in build/*.pdb; do [[ -f "$f" ]] && files+=("$f"); done
        for f in data/qsa-oracle/*.tokens; do [[ -f "$f" ]] && files+=("$f"); done
        sha256sum "${files[@]}" > SHA256SUMS.txt
    )
else
    echo "提示：找不到 sha256sum，跳过 SHA256SUMS.txt" >&2
fi

# ---- 完整性检查 -------------------------------------------------------------
miss=0
for f in start_win.exe settings_win.exe service.conf BUILD-INFO.txt \
         build/qwenox-engine-win.exe build/qwenox-win.exe build/qwenox-bench.exe \
         build/amdhip64_7.dll build/rocm_kpack.dll build/amd_comgr.dll \
         build/rocblas.dll build/libhipblaslt.dll build/origami.dll \
         build/msvcp140.dll build/vcruntime140.dll build/vcruntime140_1.dll \
         data/tokenizer/tokenizer.json; do
    [[ -s "$OUT/$f" ]] || { echo "  [缺失] $f" >&2; miss=1; }
done
for d in "build/rocblas/library/$GPU_ARCH" "build/hipblaslt/library/$GPU_ARCH"; do
    if [[ -d "$OUT/$d" ]] && [[ -n "$(ls -A "$OUT/$d" 2>/dev/null)" ]]; then
        :
    else
        echo "  [缺失] $d" >&2
        miss=1
    fi
done
if [[ -z "$(ls -A "$OUT/data/qsa-oracle" 2>/dev/null)" ]]; then
    echo "  [缺失] data/qsa-oracle/*.tokens" >&2
    miss=1
fi
if (( SYMBOLS )); then
    pdb_ok=0
    for f in "$OUT"/build/*.pdb; do [[ -s "$f" ]] && pdb_ok=1; done
    (( pdb_ok )) || { echo "  [缺失] build/*.pdb" >&2; miss=1; }
fi
(( miss == 0 )) || die "发行包不完整：$OUT"

# ---- 可选 zip ---------------------------------------------------------------
if (( MAKE_ZIP )); then
    ZIP="dist/$NAME.zip"
    rm -f -- "$ZIP"
    echo "[压缩] $ZIP"
    if command -v powershell >/dev/null 2>&1; then
        OUT_WIN="$(cygpath -w "$OUT")"
        ZIP_WIN="$(cygpath -w "$ZIP")"
        powershell -NoProfile -Command \
            "Add-Type -AssemblyName System.IO.Compression.FileSystem; [System.IO.Compression.ZipFile]::CreateFromDirectory('$OUT_WIN', '$ZIP_WIN', [System.IO.Compression.CompressionLevel]::Optimal, \$true)" \
            || die "压缩失败：$ZIP"
    else
        die "找不到 powershell，无法生成 zip（去掉 --zip 重试）"
    fi
fi

# ---- 汇总 -------------------------------------------------------------------
SIZE="$(du -sh "$OUT" | cut -f1)"
echo
echo "[完成] $OUT（$SIZE）"
echo "  配置：PARALLEL=$PARALLEL_V  KV_POOL_TOKENS=$KVPOOL_V  MAX_CONTEXT=$MAXCTX_V  MOE_A4=$MOEA4_V"
(( SKIP_BUILD )) || echo "  日志：$BUILD_LOG"
if (( SYMBOLS )); then
    echo "  PDB ：$OUT/build/*.pdb（随包）"
else
    echo "  PDB ：未包含（--no-symbols）"
fi
(( MAKE_ZIP )) && echo "  ZIP ：dist/$NAME.zip"
echo "  使用：把权重放进 $OUT/models/，双击 start_win.exe"
