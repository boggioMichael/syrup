#!/usr/bin/env bash
# Fetch the comparison's outside data at pinned commits into DATA_DIR:
# OpenCV's samples and cascades, opencv_extra's face photographs, and
# LipNet's GRID clips (from which the lip-reading fixtures are composed).
#
#     bench/vs-opencv/fetch.sh DATA_DIR
set -euo pipefail
data=$(mkdir -p "${1:?data directory}" && cd "$1" && pwd)
repo=$(cd "$(dirname "$0")/../.." && pwd)

sparse() { # url commit dir paths...
    local url=$1 commit=$2 dir=$3; shift 3
    if [ ! -d "$dir/.git" ]; then
        git clone --filter=blob:none --no-checkout "$url" "$dir"
    fi
    git -C "$dir" sparse-checkout set "$@"
    git -C "$dir" checkout -q "$commit"
}
sparse https://github.com/opencv/opencv 1f2e79334576ed9792f7633fcd4df22b8da96767 "$data/opencv-src" \
    data/haarcascades samples/data modules/objdetect/src
sparse https://github.com/opencv/opencv_extra 22b4a7bc7cf5e4ddd3a0426eafc88bae28d3dfc3 "$data/opencv_extra" \
    testdata/cv/cascadeandhog
lipnet="$repo/python/thelip/LipNet"
if [ ! -d "$lipnet/.git" ]; then
    git clone https://github.com/rizkiarm/LipNet "$lipnet"
fi
git -C "$lipnet" checkout -q bec592aa1e66378bc7ab33f6110f1c8a431b3b9a
python3 "$repo/example/lipreader/tests/make_fixtures.py" --lipnet "$lipnet" --out "$data/lip-fixtures"
