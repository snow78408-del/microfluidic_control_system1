@echo off
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%backend\venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] venv python not found: %PY%
  echo Please run backend\venv setup first.
  exit /b 1
)
"%PY%" "%ROOT%frontend\run.py"
endlocal
