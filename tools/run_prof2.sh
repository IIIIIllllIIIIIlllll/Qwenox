#!/usr/bin/env bash
set -euo pipefail
cd /c/Users/Mark/Workspace/Qwenox-main
export HIP_PATH="C:\\therock-dist-windows-gfx1151-10.1.0"
unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
echo "[prof2] compiling qwenox-prof2.exe ..."
/c/therock-dist-windows-gfx1151-10.1.0/bin/hipcc.exe -O3 -std=c++17 --offload-arch=gfx1151 \
  -Xoffload-linker --lto-partitions=16 \
  -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE \
  -DQWENOX_VERSION_RAW=0.1.5-preview -DQWENOX_GIT_RAW=8e492fa \
  -I/c/therock-dist-windows-gfx1151-10.1.0/include -Lbuild/winlibs -lrocblas -lhipblaslt \
  _serial_bench/qwenox_prof2.cpp -o build/qwenox-prof2.exe
echo "[prof2] DONE build"
