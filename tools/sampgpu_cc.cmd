@echo off
cd /d C:\Users\Mark\Workspace\Qwenox-Test
"C:\Program Files\Git\bin\bash.exe" _serial_bench\sampgpu_cc.sh > _serial_bench\sampgpu_cc.log 2>&1
exit /b %ERRORLEVEL%
