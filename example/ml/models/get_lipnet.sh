#!/bin/sh
# LipNet weights, GRID dictionary and sample clips: github.com/rizkiarm/LipNet (MIT;
# GRID corpus CC BY 4.0). Cloned next to The Lip, where lipreader looks for them
# (or set LIPNET_DIR). Nothing is redistributed from this repository.
set -e
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TARGET="${LIPNET_DIR:-$ROOT/python/thelip/LipNet}"
if [ -f "$TARGET/evaluation/models/overlapped-weights368.h5" ]; then
    echo "LipNet already at $TARGET"
    exit 0
fi
git clone --depth 1 https://github.com/rizkiarm/LipNet "$TARGET"
echo "LipNet at $TARGET"
