#!/usr/bin/env bash
# usage: abba.sh <env-assignments-in-cmd-syntax> <tag>
# e.g.: abba.sh 'set "QWENOX_SMS_GRAPH=1" &&' graph1
set -u
ENVS="$1"; TAG="$2"
SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force ssh -o StrictHostKeyChecking=no Mark@192.168.5.12 "Get-Process | Where-Object { \$_.Name -match 'qwenox' } | Stop-Process -Force; Start-Sleep -Seconds 2; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd /c cd /d C:\Users\Mark\Workspace\Qwenox-Test && set \"QWENOX_NOSPEC=1\" && ${ENVS} build\qwenox-win.exe --console > logs\serve_ab.log 2>&1' }; echo x > \$null" >/dev/null 2>&1
sleep 5
curl -s -m 10 -X POST http://192.168.5.12:8731/admin/engine/start >/dev/null
for i in $(seq 1 60); do
  st=$(curl -s -m 5 http://192.168.5.12:8731/admin/engine/status 2>/dev/null)
  echo "$st" | grep -q '"state":"running"' && break
  sleep 5
done
python tools/r2_quick_bench.py > "outputs/abba_${TAG}.txt" 2>&1
SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force ssh -o StrictHostKeyChecking=no Mark@192.168.5.12 "[Console]::OutputEncoding=[Text.Encoding]::UTF8; \$n = Get-ChildItem C:\Users\Mark\Workspace\Qwenox-Test\logs -Filter 'engine-api-*' | Sort-Object LastWriteTime | Select-Object -Last 1; Get-Content \$n.FullName -Encoding UTF8 | Select-String -Pattern 'decode eval' | ForEach-Object { \$_ -replace '.*\((.* ms/token).*','\$1' }" 2>/dev/null | head -5
