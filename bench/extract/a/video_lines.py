"""Task A dev lines from the public Zakum video (exploratory, dev data only).

    python3 bench/extract/a/video_lines.py crop --video ms/chaos-zakum-solo-lvl230.mp4 --out PRIVATE/lines.npz
    python3 bench/extract/a/video_lines.py runs --lines PRIVATE/lines.npz --out PRIVATE/runs.json

`crop` cuts the HP, MP and EXP text lines out of every frame, at the boxes a
layout gives on this recording (the game window does not move in it), and
marks the frames where the HUD is not drawn (the black frames of a map
change). `runs` groups consecutive frames whose line did not change into runs:
one run is one value on the screen, and a run is the unit that is labelled.

The crops are game graphics: they stay outside the repository (DATA.md). The
video itself is `chaos-zakum-solo-lvl230.mp4` of the `ms` repository, hashed in
`manifests/dev-v1.sha256`.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np

W, H = 1152, 720
# (top, bottom, left, right) of each field's line, as the layout locates it:
# the inside of the HP and MP bars, and the band under them for EXP.
BOXES = {
    "hp": (603, 618, 445, 621),
    "mp": (622, 637, 445, 621),
    "exp": (641, 656, 440, 650),
}
# A line is "white text": every channel above this.
WHITE = 170


def frames(video: str):
    cmd = ["ffmpeg", "-loglevel", "error", "-i", video, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        size = W * H * 3
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)


def crop(args):
    out = {name: [] for name in BOXES}
    hud = []
    for frame in frames(args.video):
        for name, (y0, y1, x0, x1) in BOXES.items():
            out[name].append(frame[y0:y1, x0:x1].copy())
        # The HP bar's fill is bright red where the HUD is drawn.
        y0, y1, x0, x1 = BOXES["hp"]
        bar = frame[y0:y1, x0:x1].astype(int)
        hud.append(bool((bar[..., 0] > 150).mean() > 0.3))
    arrays = {name: np.stack(v) for name, v in out.items()}
    np.savez_compressed(args.out, hud=np.array(hud), **arrays)
    print(json.dumps({k: list(v.shape) for k, v in arrays.items()} | {"hud_frames": int(sum(hud))}))


def runs(args):
    """Runs of frames that show one value, and each run's mean picture.

    A frame starts a new run when more than `--pixels` of its values differ by
    more than `--level` from the run's first frame (not from the frame before,
    so a slow drift cannot carry a run across a change of value). Splitting a
    value into two runs costs nothing; joining two values would mislabel, so
    the thresholds sit just above the codec's noise.
    """
    data = np.load(args.lines)
    hud = data["hud"]
    result = {}
    means = {}
    for name in BOXES:
        lines = data[name].astype(np.int16)
        spans = []
        start = None
        for i in range(len(lines)):
            if not hud[i]:
                if start is not None:
                    spans.append((start, i))
                    start = None
                continue
            if start is None:
                start = i
                continue
            if (np.abs(lines[i] - lines[start]) > args.level).sum() > args.pixels:
                spans.append((start, i))
                start = i
        if start is not None:
            spans.append((start, len(lines)))
        result[name] = spans
        means[name] = np.stack([lines[a:b].mean(axis=0) for a, b in spans]).astype(np.float32)
        lengths = np.array([b - a for a, b in spans])
        print(name, len(spans), "runs;", int((lengths >= 3).sum()), "of 3 frames or more")
    with open(args.out, "w") as f:
        json.dump(result, f)
    np.savez_compressed(args.out.replace(".json", "-means.npz"), **means)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crop")
    c.add_argument("--video", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("runs")
    r.add_argument("--lines", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--level", type=int, default=50)
    r.add_argument("--pixels", type=int, default=3)
    args = p.parse_args()
    {"crop": crop, "runs": runs}[args.cmd](args)


if __name__ == "__main__":
    main()
