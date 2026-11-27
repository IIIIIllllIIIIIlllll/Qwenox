#!/usr/bin/env bash
# usage: longctx_gate.sh <env-assignments> <tag> <ctx>
set -u
ENVS="$1"; TAG="$2"; CTX="${3:-10000}"
SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force ssh -o StrictHostKeyChecking=no Mark@192.168.5.12 "Get-Process | Where-Object { \$_.Name -match 'qwenox' } | Stop-Process -Force; Start-Sleep -Seconds 2; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd /c cd /d C:\Users\Mark\Workspace\Qwenox-Test && set \"QWENOX_NOSPEC=1\" && ${ENVS} build\qwenox-win.exe --console > logs\serve_lc.log 2>&1' }; echo x > \$null" >/dev/null 2>&1
sleep 5
curl -s -m 10 -X POST http://192.168.5.12:8731/admin/engine/start >/dev/null
for i in $(seq 1 60); do
  st=$(curl -s -m 5 http://192.168.5.12:8731/admin/engine/status 2>/dev/null)
  echo "$st" | grep -q '"state":"running"' && break
  sleep 5
done
python tools/r3_longctx_gate.py "$TAG" "$CTX"
