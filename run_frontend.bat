@echo off
setlocal
set PY=D:\??\microfluidic_control_system\backend\venv\Scripts\python.exe
if not exist "%PY%" (
  echo [ERROR] venv python not found: %PY%
  exit /b 1
)
"%PY%" D:\??\microfluidic_control_system\frontend\run.py
endlocal
