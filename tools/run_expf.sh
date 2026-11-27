#!/usr/bin/env bash
set -euo pipefail
cd /c/Users/Mark/Workspace/Qwenox-Test
export HIP_PATH="C:\\therock-dist-windows-gfx1151-10.1.0"
unset ROCM_PATH HIP_PATH_64 HIP_PATH_71 HIP_PATH_72
/c/therock-dist-windows-gfx1151-10.1.0/bin/hipcc.exe -O2 -std=c++17 --offload-arch=gfx1151 \
  -D_CRT_SECURE_NO_WARNINGS \
  -I/c/therock-dist-windows-gfx1151-10.1.0/include -Lbuild/winlibs -lrocblas -lhipblaslt \
  _serial_bench/expf_probe.cu -o build/expf_probe.exe
./build/expf_probe.exe
