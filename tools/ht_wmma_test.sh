#!/usr/bin/env bash
# ht_wmma_test.sh [ht|ht2|q6|q8] — build + run tools/ht_wmma_test.cu (k_ht_wmma vs
# k_ht_gemv), tools/ht2_wmma_test.cu (k_ht_wmma vs its frozen copy, bit-exact),
# tools/q6_wmma_test.cu (k_q6g64_wmma vs q6g64_gemv) or
# tools/q8_wmma_test.cu (k_q8g32_wmma vs CPU double, GGUF Q8_0 dense): accuracy,
# row invariance, determinism, timing. Ends HT_WMMA / Q6_WMMA TEST: PASS/FAIL.
# Windows (Git Bash, TheRock) or Linux (hipcc on PATH / ROCM).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build logs
T="${1:-ht}"
[[ "$T" == ht || "$T" == ht2 || "$T" == q6 || "$T" == q8 ]] || { echo "usage: $0 [ht|ht2|q6|q8]"; exit 2; }
if [[ "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
  TR="${THEROCK:-/c/therock-dist-windows-gfx1151-10.1.0}"
  HIPCC="$TR/bin/hipcc.exe"
  export HIP_PATH="$(cygpath -w "$TR")"
  unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
  EXE=build/${T}_wmma_test.exe
  FLAGS=(-O3 -std=c++17 --offload-arch=gfx1151 -D_CRT_SECURE_NO_WARNINGS
         -D_CRT_NONSTDC_NO_DEPRECATE -I"$TR/include" -Lbuild/winlibs -lrocblas -lhipblaslt)
else
  HIPCC="${HIPCC:-hipcc}"
  EXE=build/${T}_wmma_test
  FLAGS=(-O3 -std=c++17 --offload-arch=gfx1151 -lrocblas -lhipblaslt)
fi
echo "== ${T}_wmma_test $(date '+%F %T') build"
if ! "$HIPCC" "${FLAGS[@]}" ${HTW_FLAGS:-} -I src/gpu tools/${T}_wmma_test.cu -o "$EXE.tmp"; then
  echo "${T^^}_WMMA TEST: FAIL (build)"; exit 1
fi
mv -f "$EXE.tmp" "$EXE"
echo "== run"
if [[ "$EXE" == *.exe ]]; then (cd build && "./${T}_wmma_test.exe"); else "./$EXE"; fi
