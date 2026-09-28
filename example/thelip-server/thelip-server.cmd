@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title thelip-server

rem Python 3.10-3.12 (mediapipe has no wheels for newer Pythons); prefer 3.11.
set "PY="
for %%v in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.10, 3.11 or 3.12 is needed and none was found.
  echo Install Python 3.11 from https://www.python.org/downloads/windows/
  echo ^(tick "Add python.exe to PATH" and "py launcher"^), then run this file again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo creating the Python environment with %PY%
  %PY% -m venv .venv || (echo could not create .venv & pause & exit /b 1)
)
echo installing packages ^(PyTorch and mediapipe; a few minutes the first time^)
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt || (echo pip install failed; see above & pause & exit /b 1)

.venv\Scripts\python.exe run.py %*
echo.
pause
