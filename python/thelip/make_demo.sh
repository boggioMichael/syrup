#!/bin/sh
# Builds the-lip-demo.mp4: title card, the GRID sample clips muted and
# subtitled from the lips, end card. Needs python3 with numpy, opencv and
# pillow, ffmpeg with libx264, and the built syrup library
# (`cargo build --release` at the repository root).
set -e
cd "$(dirname "$0")"
LIPNET=${LIPNET_DIR:-./LipNet}
if [ ! -d "$LIPNET" ]; then
    git clone --depth 1 https://github.com/rizkiarm/LipNet "$LIPNET"
fi
SAMPLES=$LIPNET/evaluation/samples
python3 the_lip.py --lipnet "$LIPNET" --truth --out clips.mp4 "$SAMPLES"/GRID/*.mpg "$SAMPLES"/*.mpg
python3 cards.py --summary clips.json --out-dir .
printf "file 'card_in.mp4'\nfile 'clips.mp4'\nfile 'card_out.mp4'\n" > concat.txt
ffmpeg -v error -y -f concat -safe 0 -i concat.txt -c copy the-lip-demo.mp4
echo "wrote the-lip-demo.mp4"
