#!/usr/bin/env bash
# thelip-train on a fresh Linux GPU machine (Ubuntu 22.04, NVIDIA driver present;
# e.g. a RunPod "PyTorch" pod). One command, then it runs to the budget:
#
#   curl -sSL https://raw.githubusercontent.com/boggioMichael/syrup/claude/intents-2/example/thelip-train/bootstrap.sh \
#     | bash -s -- --languages en,he --hours 40 --hourly-cost 3.5 --video-hours en=150,he=80
#
# Everything lands under /workspace/thelip (a RunPod volume survives restarts).
set -euo pipefail
ROOT="${THELIP_TRAIN_ROOT:-/workspace/thelip}"
mkdir -p "$ROOT" && cd "$ROOT"
if ! command -v ffmpeg >/dev/null; then apt-get update -qq && apt-get install -y -qq ffmpeg git >/dev/null; fi
if [ ! -d syrup/.git ]; then git clone -q --depth 1 -b claude/intents-2 https://github.com/boggioMichael/syrup.git; fi
cd syrup/example/thelip-train
python3 -m pip install -q -r requirements.txt
export THELIP_TRAIN_HOME="$ROOT/work"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible: training will not run"
exec python3 run_all.py "$@" 2>&1 | tee -a "$ROOT/run.log"
