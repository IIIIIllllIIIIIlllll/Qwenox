@echo off
cd /d C:\Users\Mark\Workspace\Qwenox-Test
"C:\Program Files\Git\bin\bash.exe" _serial_bench\run_sampgpu.sh > _serial_bench\sampgpu_out.log 2>&1
exit /b %ERRORLEVEL%
