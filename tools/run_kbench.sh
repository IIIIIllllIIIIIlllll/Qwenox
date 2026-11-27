#!/usr/bin/env bash
set -euo pipefail
cd /c/Users/Mark/Workspace/Qwenox-main
export HIP_PATH="C:\\therock-dist-windows-gfx1151-10.1.0"
unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
echo "[kbench] compiling kbench_hc_gdn.exe ..."
/c/therock-dist-windows-gfx1151-10.1.0/bin/hipcc.exe -O3 -std=c++17 --offload-arch=gfx1151 \
  -D_CRT_SECURE_NO_WARNINGS \
  -I/c/therock-dist-windows-gfx1151-10.1.0/include -Lbuild/winlibs -lrocblas -lhipblaslt \
  _serial_bench/kbench_hc_gdn.cu -o build/kbench_hc_gdn.exe
echo "[kbench] running ..."
./build/kbench_hc_gdn.exe
echo "[kbench] DONE"
