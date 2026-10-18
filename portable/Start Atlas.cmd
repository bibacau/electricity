@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime\python.exe" (
  echo This package is incomplete: bundled Windows runtime is missing.
  pause
  exit /b 1
)
"runtime\python.exe" -I -B "launcher.py"
if errorlevel 1 pause
