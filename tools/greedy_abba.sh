#!/usr/bin/env bash
# usage: greedy_abba.sh <env-assignments-in-cmd-syntax>
# Greedy request + QWENOX_TIME breakdown (launch/g_sync = submit/GPU split).
set -u
ENVS="$1"
SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force ssh -o StrictHostKeyChecking=no Mark@192.168.5.12 "Get-Process | Where-Object { \$_.Name -match 'qwenox' } | Stop-Process -Force; Start-Sleep -Seconds 2; \$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'cmd /c cd /d C:\Users\Mark\Workspace\Qwenox-Test && set \"QWENOX_NOSPEC=1\" && set \"QWENOX_TIME=1\" && ${ENVS} build\qwenox-win.exe --console > logs\serve_gt.log 2>&1' }; echo x > \$null" >/dev/null 2>&1
sleep 5
curl -s -m 10 -X POST http://192.168.5.12:8731/admin/engine/start >/dev/null
for i in $(seq 1 60); do
  st=$(curl -s -m 5 http://192.168.5.12:8731/admin/engine/status 2>/dev/null)
  echo "$st" | grep -q '"state":"running"' && break
  sleep 5
done
python - <<'EOF'
import json, urllib.request
body = json.dumps({"model":"qwen","messages":[{"role":"user","content":"Write a Python function that merges two sorted lists into one sorted list, with a docstring."}],"temperature":0,"max_tokens":192,"stream":False}).encode()
req = urllib.request.Request("http://192.168.5.12:8731/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    resp = json.loads(r.read())
print("ok", resp["usage"]["completion_tokens"])
EOF
SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force ssh -o StrictHostKeyChecking=no Mark@192.168.5.12 "[Console]::OutputEncoding=[Text.Encoding]::UTF8; \$n = Get-ChildItem C:\Users\Mark\Workspace\Qwenox-Test\logs -Filter 'engine-api-*' | Sort-Object LastWriteTime | Select-Object -Last 1; Get-Content \$n.FullName -Encoding UTF8 | Select-String -Pattern 'TIME:' | Select-Object -Last 2; Get-Content \$n.FullName -Encoding UTF8 | Select-String -Pattern 'decode eval' | Select-Object -Last 1" 2>/dev/null | head -4
