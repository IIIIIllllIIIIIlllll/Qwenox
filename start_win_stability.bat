@echo off
setlocal
rem Temporary low-pressure profile; does not modify service.conf or Windows.
rem Close the existing API from its tray menu before using this entry point.
tasklist /FI "IMAGENAME eq qwenox-win.exe" /NH 2>nul | find /I "qwenox-win.exe" >nul
if not errorlevel 1 (
  echo Close Qwenox from its tray menu, then run this profile again.
  pause
  exit /b 1
)
if not exist "%~dp0start_win.exe" (
  echo Missing start_win.exe. Use a complete Windows build of this version.
  pause
  exit /b 1
)
set "MAX_CONTEXT=65536"
set "KV_POOL_TOKENS=65536"
set "PARALLEL=1"
set "PREFILL_CHUNK=1024"
set "CONC_PREFILL_CHUNK=1024"
set "KVSNAP_MAX_GB=0"
set "RCKPT_MAX=2"
set "MAX_IMAGES=1"
set "QWENOX_HIP_TIMEOUT=120"
echo Starting Qwenox with the temporary stability profile.
echo Context 64K, one slot, prefill 1024, SSD snapshots disabled.
start "" /D "%~dp0" "%~dp0start_win.exe"
