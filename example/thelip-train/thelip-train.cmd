@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title thelip-train

rem thelip-train on this PC's NVIDIA card: the environment and the work live in
rem an ASCII path next to thelip-server's (PyTorch's DLLs fail from a path with
rem non-ASCII characters, such as a Hebrew user name). The window waits for runs
rem (see watch.py); closing it stops the run, which continues where it stopped
rem the next time.
set "ROOT=%PUBLIC%\thelip-server\train"
if not exist "%ROOT%" mkdir "%ROOT%" || (echo cannot create %ROOT% & pause & exit /b 1)

set "PY="
for %%v in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%v -c "import sys" >nul 2>&1 && set "PY=py -%%v"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo Python 3.10, 3.11 or 3.12 is needed and none was found.
  pause
  exit /b 1
)

set "VENV=%ROOT%\.venv"
if not exist "%VENV%\Scripts\python.exe" (
  echo creating the Python environment in %VENV% with %PY%
  %PY% -m venv "%VENV%" || (echo could not create the environment & pause & exit /b 1)
)
"%VENV%\Scripts\python.exe" -m pip install --quiet --upgrade pip wheel
"%VENV%\Scripts\python.exe" -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if errorlevel 1 (
  echo installing PyTorch with CUDA ^(about 3 GB, once^)
  "%VENV%\Scripts\python.exe" -m pip install --quiet torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 || (echo PyTorch did not install; see above & pause & exit /b 1)
)
echo installing the other packages
"%VENV%\Scripts\python.exe" -m pip install --quiet -r requirements-windows.txt || (echo pip install failed; see above & pause & exit /b 1)
"%VENV%\Scripts\python.exe" -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no NVIDIA card visible')"

set "THELIP_TRAIN_ROOT=%ROOT%"
set "THELIP_TRAIN_HOME=%ROOT%\work"
set "HF_HOME=%PUBLIC%\thelip-server\hf"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
rem One GPU of 12 GB: bf16, batches of about 25 seconds of video, a few data workers.
if not defined THELIP_PRECISION set "THELIP_PRECISION=bf16-mixed"
if not defined THELIP_MAX_FRAMES set "THELIP_MAX_FRAMES=640"
if not defined THELIP_NUM_WORKERS set "THELIP_NUM_WORKERS=4"
"%VENV%\Scripts\python.exe" watch.py %*
echo.
pause
