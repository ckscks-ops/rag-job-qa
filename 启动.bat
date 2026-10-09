@echo off
rem ============================================================
rem  RAG Job QA - one-click launcher (Windows)
rem  This file is intentionally ASCII-only: cmd.exe on a Chinese
rem  Windows uses code page 936, and non-ASCII bytes in a .bat are
rem  parsed unreliably. All Chinese messages come from setup.py.
rem ============================================================
setlocal enableextensions
cd /d "%~dp0"
title RAG Job QA

rem Console and Python must agree on UTF-8, otherwise Chinese output from
rem setup.py shows as mojibake on a code page 936 console.
chcp 65001 >nul 2>nul
set "PYTHONIOENCODING=utf-8"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo.
  echo [ERROR] Python not found.
  echo         Install Python 3.10+ and tick "Add python.exe to PATH".
  echo         https://www.python.org/downloads/
  echo.
  pause
  exit /b 1
)

if not exist "data\rag.db" (
  echo.
  echo   No database yet. The first run needs about 25 minutes:
  echo     - install dependencies
  echo     - download models  ^(~200 MB^)
  echo     - build the vector index
  echo.
  choice /c YN /m "  Run full setup now"
  if errorlevel 2 exit /b 0
  echo.
  "%PY%" setup.py all
  if errorlevel 1 (
    echo.
    echo   [ERROR] Setup failed - see the messages above.
    echo.
    pause
    exit /b 1
  )
)

"%PY%" setup.py serve --port 8000
echo.
pause
