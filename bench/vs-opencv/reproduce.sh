#!/usr/bin/env bash
# The whole comparison in one command, from the repository root:
#
#     bench/vs-opencv/reproduce.sh WORK_DIR MS_REPO
#
# fetches the outside data at pinned commits, prepares the inputs (their
# SHA-256 must match results/inputs.json), builds Syrup before (156475f,
# the library as pinned by ms) and after (HEAD) with the same harness,
# measures the held-out test split with one and with two threads (5 rounds
# each, every implementation in its own process, rotated order), runs the
# lip-reading pipeline three times per detector, and writes the summaries
# under WORK_DIR/results. Needs: git, ffmpeg, cargo (rustc 1.97), python3
# with numpy and opencv-contrib-python 5.0.0.93. About two hours on two
# cores; nothing else should run meanwhile.
set -euo pipefail
work=$(mkdir -p "${1:?work directory}" && cd "$1" && pwd)
ms=$(cd "${2:?ms repository (github.com/boggioMichael/ms at 8a14965)}" && pwd)
repo=$(cd "$(dirname "$0")/../.." && pwd)
here="$repo/bench/vs-opencv"
before=156475f
data="$work/data" inputs="$work/inputs" bin="$work/bin" results="$work/results"

"$here/fetch.sh" "$data"
cascades="$data/opencv-src/data/haarcascades"
python3 "$here/prepare.py" --out "$inputs" --manifest "$work/inputs.json" \
    --opencv-samples "$data/opencv-src/samples/data" --opencv-extra "$data/opencv_extra" \
    --lip-fixtures "$data/lip-fixtures" --ms "$ms"
python3 - "$work/inputs.json" "$here/results/inputs.json" <<'PY'
import json, sys
got, want = (json.load(open(p)) for p in sys.argv[1:])
g = {e["file"]: e["sha256"] for e in got["files"]}
w = {e["file"]: e["sha256"] for e in want["files"]}
bad = sorted(f for f in w if g.get(f) != w[f])
print(f"inputs: {len(w) - len(bad)} of {len(w)} identical to the recorded hashes")
if bad:
    print("different:", bad[:10])
PY
"$here/build.sh" "$before" HEAD "$bin"

for threads in 1 2; do
    out="$results/test-t$threads"
    common=(--data "$inputs" --split test --threads "$threads" --rounds 5 --cascades "$cascades" --out "$out")
    python3 "$here/run.py" measure "${common[@]}" --tasks faces \
        --impl syrup-before="$bin/vs_opencv-before" --impl syrup-after="$bin/vs_opencv-after" \
        --impl syrup-after-gate="$bin/vs_opencv-after" --args "syrup-after-gate=--min-deviation 10" --impl opencv
    python3 "$here/run.py" measure "${common[@]}" --tasks templates,motion,glyphs \
        --impl syrup-before="$bin/vs_opencv-before" --impl syrup-after="$bin/vs_opencv-after" --impl opencv
    python3 "$here/run.py" measure "${common[@]}" --tasks yunet --impl syrup="$bin/vs_opencv_yunet" --impl opencv
    python3 "$here/run.py" summarise --data "$inputs" --out "$out"
done

# F1-pipeline: the lip-reading evaluation, three runs per detector in
# rotated order (the lipreader's OpenCV detector loads its cascades from
# cv2.data.haarcascades, which OpenCV 5 wheels leave empty).
haar=$(python3 -c "import cv2; print(cv2.data.haarcascades)")
for f in haarcascade_frontalface_alt2.xml haarcascade_eye.xml haarcascade_profileface.xml; do
    [ -f "$haar/$f" ] || cp "$cascades/$f" "$haar/"
done
lip="$results/lipreader"
mkdir -p "$lip"
order=(syrup-before syrup-after opencv)
for r in 0 1 2; do
    for k in 0 1 2; do
        impl=${order[$(((k + r) % 3))]}
        case $impl in
            syrup-before) env=(SYRUP_LIBRARY="$bin/libsyrup-before.so") ;;
            syrup-after) env=(SYRUP_LIBRARY="$bin/libsyrup-after.so") ;;
            opencv) env=(LIPREADER_DETECTOR=opencv) ;;
        esac
        (cd "$repo/example/lipreader" && env "${env[@]}" PYTHONPATH="$repo/python" \
            python3 -m lipreader evaluate "$data/lip-fixtures" --json "$lip/$impl.r$r.json")
    done
done
python3 "$here/lipreader_summary.py" "$lip"
