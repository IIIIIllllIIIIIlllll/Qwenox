#!/usr/bin/env bash
# Compile-only check: src/gpu/qwenox.cpp with the modified
# src/gpu/parts/21_kernels_ple.inc swapped in (original restored afterwards).
set -euo pipefail
cd /c/Users/Mark/Workspace/Qwenox-Test
export HIP_PATH="C:\\therock-dist-windows-gfx1151-10.1.0"
unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
cp src/gpu/parts/21_kernels_ple.inc _serial_bench/21_kernels_ple.inc.orig
cp _serial_bench/21_kernels_ple.inc.new src/gpu/parts/21_kernels_ple.inc
restore() {
  cp _serial_bench/21_kernels_ple.inc.orig src/gpu/parts/21_kernels_ple.inc
}
trap restore EXIT
VER="$(tr -d ' \r\n' < VERSION)"
echo "[sampgpu-cc] compiling src/gpu/qwenox.cpp (compile-only, no link) ..."
/c/therock-dist-windows-gfx1151-10.1.0/bin/hipcc.exe -O3 -std=c++17 \
  --offload-arch=gfx1151 \
  -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE \
  -DQWENOX_VERSION_RAW="$VER" -DQWENOX_GIT_RAW=nogit \
  -I/c/therock-dist-windows-gfx1151-10.1.0/include \
  -c src/gpu/qwenox.cpp -o build/.sampgpu-qwenox.o
rm -f build/.sampgpu-qwenox.o
echo "[sampgpu-cc] COMPILE_OK"
