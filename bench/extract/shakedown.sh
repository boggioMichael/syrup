#!/bin/sh
# The P1 shakedown (SHAKEDOWN.md): train the tiny network, measure Syrup, OpenCV
# and the tiny network twice on both glyph splits, then summarise, compute the
# explicit-first measures and the G1 check.
#
#   bench/extract/shakedown.sh DATA SYRUP_BIN WORK
#
# DATA is the data directory `bench/vs-opencv/prepare.py --out` wrote (its
# hashes are in bench/vs-opencv/results/inputs.json); SYRUP_BIN the
# `vs_opencv` example binary; WORK a scratch directory.
set -eu
DATA=$1 BIN=$2 WORK=$3
HERE=$(cd "$(dirname "$0")" && pwd)
CASCADES=$(python3 -c "import cv2; print(cv2.data.haarcascades)")
mkdir -p "$WORK"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$HERE/tinynet.py" train --data "$DATA" --out "$WORK/model"
for split in dev test; do
  for run in a b; do
    out="$WORK/$split-$run"
    python3 "$HERE/../vs-opencv/run.py" measure --data "$DATA" --split $split --threads 1 \
      --impl syrup-after="$BIN" --impl opencv --impl tinynet="$HERE/tinynet" \
      --args "tinynet=--model $WORK/model/glyphs-int8.onnx" \
      --tasks glyphs --rounds 5 --cascades "$CASCADES" --out "$out"
    python3 "$HERE/../vs-opencv/run.py" summarise --data "$DATA" --out "$out" > /dev/null
    python3 "$HERE/cascade.py" --data "$DATA" --results "$out" --explicit syrup-after \
      --fallback opencv --fallback tinynet --out "$out/cascade.json"
  done
  python3 "$HERE/repeat_check.py" "$WORK/$split-a" "$WORK/$split-b" --out "$WORK/$split-g1.json"
done
