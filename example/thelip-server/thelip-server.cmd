@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title thelip-server

rem The Python environment, the pipeline and the models live in an ASCII path:
rem PyTorch's DLLs fail to initialise from a path with non-ASCII characters
rem (a Hebrew user name, say). Logs and the link stay next to this file.
set "WORK=%PUBLIC%\thelip-server"
if not exist "%WORK%" mkdir "%WORK%" || (echo cannot create %WORK% & pause & exit /b 1)
if exist ".venv\Scripts\python.exe" (
  echo removing the old environment in this folder ^(it moved to %WORK%^)
  rmdir /s /q ".venv"
)

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

set "VENV=%WORK%\.venv"
if not exist "%VENV%\Scripts\python.exe" (
  echo creating the Python environment in %VENV% with %PY%
  %PY% -m venv "%VENV%" || (echo could not create the environment & pause & exit /b 1)
)
echo installing packages ^(PyTorch and mediapipe; a few minutes the first time^)
"%VENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip setuptools wheel
"%VENV%\Scripts\python.exe" -m pip install --quiet -r requirements.txt || (echo pip install failed; see above & pause & exit /b 1)

set "THELIP_HOME=%WORK%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
"%VENV%\Scripts\python.exe" run.py %*
echo.
pause
