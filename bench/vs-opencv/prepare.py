"""Prepare the inputs of the Syrup vs OpenCV comparison (see PREREGISTRATION.md).

Every input is written as PNG (so both libraries decode the same pixels), with
its SHA-256 in results/inputs.json. Nothing here depends on results.

    python3 bench/vs-opencv/prepare.py --out DATA \
        --opencv-samples OPENCV/samples/data --opencv-extra OPENCV_EXTRA \
        --lip-fixtures LIPFIX --ms MS_REPO

Sources: OpenCV `samples/data` (opencv/opencv 4.x), `opencv_extra`
`testdata/cv/cascadeandhog/images`, the lip-reading fixtures made by
`example/lipreader/tests/make_fixtures.py` from LipNet's GRID samples, and
the ms repository's MapleStory still and recording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261012
TEMPLATE_SIDES = (16, 32, 48, 64)
TEMPLATES_PER_IMAGE = 10
MOTION_PAIRS = 100

FACE_TEST = [
    "addams-family.png", "audrybt1.png", "bttf301.png", "churchill-downs.png", "class57.png", "er.png",
    "image_00000000_0.png", "karen-and-rob.png", "larroquette.png", "mickymouse-self-p.png",
    "mona-lisa.png", "rehg-thanksgiving-1994.png", "waynesworld2.png",
]
TEMPLATE_TEST = [
    "baboon.jpg", "fruits.jpg", "building.jpg", "home.jpg", "box_in_scene.png", "messi5.jpg",
    "aero1.jpg", "butterfly.jpg", "starry_night.jpg", "board.jpg", "lena.jpg",
]
MOTION_TEST = ["vtest.avi", "tree.avi", "Megamind.avi"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Manifest:
    def __init__(self, out: str):
        self.out = out
        self.entries = []

    def write_png(self, rel: str, image: np.ndarray, **meta) -> str:
        path = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not cv2.imwrite(path, image):
            raise SystemExit(f"could not write {path}")
        self.entries.append({"file": rel, "sha256": sha256(path), **meta})
        return rel


def frames_of(video: str, start: int, count: int, step: int = 1, size=None):
    """`count` frames from `start`, every `step`-th (BGR), optionally resized
    with ffmpeg's bilinear scaler (as the ms bench scales the recording)."""
    if size is not None:
        w, h = size
        cmd = ["ffmpeg", "-v", "error", "-i", video, "-vf",
               f"select='gte(n\\,{start})*not(mod(n-{start}\\,{step}))',scale={w}:{h}:flags=bilinear",
               "-vsync", "0", "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout
        frame_bytes = w * h * 3
        return [np.frombuffer(raw[i * frame_bytes:(i + 1) * frame_bytes], np.uint8).reshape(h, w, 3)
                for i in range(len(raw) // frame_bytes)]
    cap = cv2.VideoCapture(video)
    frames, n = [], 0
    while len(frames) < count:
        ok, bgr = cap.read()
        if not ok:
            break
        if n >= start and (n - start) % step == 0:
            frames.append(bgr)
        n += 1
    return frames


def face_image(m: Manifest, split: str, stem: str, bgr: np.ndarray, source: str):
    """The colour image (for YuNet) and its grey (for the cascades: both
    libraries get these same bytes, OpenCV's Rec. 601 conversion)."""
    m.write_png(f"faces/{split}/{stem}.png", bgr, task="faces", split=split, source=source)
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m.write_png(f"faces/{split}/{stem}.grey.png", grey, task="faces", split=split, source=source + " (grey)")


def faces(m: Manifest, args):
    for name in ("one_speaker", "two_simultaneous", "five_faces"):
        for i, bgr in enumerate(frames_of(os.path.join(args.lip_fixtures, f"{name}.mp4"), 0, 15, step=5)):
            face_image(m, "dev", f"{name}_{i * 5:03d}", bgr, f"grid:{name}")
    images = [(os.path.join(args.opencv_extra, "testdata/cv/cascadeandhog/images", n), f"opencv_extra:{n}") for n in FACE_TEST]
    images.append((os.path.join(args.opencv_samples, "lena.jpg"), "opencv_samples:lena.jpg"))
    images.append((os.path.join(HERE, "..", "..", "tests", "fixtures", "astronaut_320.jpg"), "syrup:astronaut_320.jpg"))
    for path, source in images:
        bgr = cv2.imread(path)
        if bgr is None:
            raise SystemExit(f"missing {path}")
        stem = os.path.splitext(os.path.basename(path))[0]
        face_image(m, "test", stem, bgr, source)


def template_cases(m: Manifest, split: str, images, rng):
    cases = []
    for index, (stem, bgr, source) in enumerate(images):
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = grey.shape
        clean = m.write_png(f"templates/{split}/{stem}.png", grey, task="templates", split=split, source=source)
        noise = rng.normal(0.0, 4.0, grey.shape)
        perturbed = np.clip(0.85 * grey.astype(np.float64) + 12.0 + noise, 0, 255).round().astype(np.uint8)
        noisy = m.write_png(f"templates/{split}/{stem}.perturbed.png", perturbed, task="templates", split=split, source=source + " (perturbed)")
        made = 0
        for k in range(TEMPLATES_PER_IMAGE):
            side = TEMPLATE_SIDES[k % len(TEMPLATE_SIDES)]
            if side >= w or side >= h:
                continue
            for _ in range(100):
                x = int(rng.integers(0, w - side + 1))
                y = int(rng.integers(0, h - side + 1))
                crop = grey[y:y + side, x:x + side]
                if crop.std() >= 10.0:
                    break
            else:
                continue
            tpl = m.write_png(f"templates/{split}/{stem}.t{k:02d}.png", crop, task="templates", split=split, source=source)
            for variant, image in (("clean", clean), ("perturbed", noisy)):
                cases.append({"image": image, "template": tpl, "x": x, "y": y, "side": side,
                              "variant": variant, "source": source})
            made += 1
    return cases


def templates(m: Manifest, args):
    rng = np.random.default_rng(SEED)
    dev = [("maplestory", cv2.imread(os.path.join(args.ms, "resources", "maplestory.png")), "ms:resources/maplestory.png")]
    for i, bgr in enumerate(frames_of(os.path.join(args.ms, "chaos-zakum-solo-lvl230.mp4"), 300, 20, step=30)):
        dev.append((f"zakum_{300 + i * 30:05d}", bgr, f"ms:chaos-zakum-solo-lvl230.mp4#{300 + i * 30}"))
    test = []
    for name in TEMPLATE_TEST:
        bgr = cv2.imread(os.path.join(args.opencv_samples, name))
        if bgr is None:
            raise SystemExit(f"missing {name}")
        test.append((os.path.splitext(name)[0], bgr, f"opencv_samples:{name}"))
    for name in FACE_TEST:
        bgr = cv2.imread(os.path.join(args.opencv_extra, "testdata/cv/cascadeandhog/images", name))
        test.append(("photo_" + os.path.splitext(name)[0], bgr, f"opencv_extra:{name}"))
    for split, images in (("dev", dev), ("test", test)):
        cases = template_cases(m, split, images, rng)
        path = os.path.join(m.out, "templates", split, "cases.json")
        with open(path, "w") as f:
            json.dump(cases, f, indent=1)


def motion(m: Manifest, args):
    sources = [("dev", "zakum", os.path.join(args.ms, "chaos-zakum-solo-lvl230.mp4"), 900, (1366, 768),
                "ms:chaos-zakum-solo-lvl230.mp4@1366x768")]
    for name in MOTION_TEST:
        sources.append(("test", os.path.splitext(name)[0], os.path.join(args.opencv_samples, name), 0, None,
                        f"opencv_samples:{name}"))
    for split, stem, video, start, size, source in sources:
        frames = frames_of(video, start, MOTION_PAIRS + 1, size=size)
        # (tree.avi holds 68 frames, not the 444 its header says: all of
        # them are used — AMENDMENTS.md.)
        if len(frames) < 51:
            raise SystemExit(f"{video}: only {len(frames)} frames")
        for i, bgr in enumerate(frames):
            m.write_png(f"motion/{split}/{stem}/{i:04d}.png", bgr, task="motion", split=split, source=source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--opencv-samples", required=True)
    ap.add_argument("--opencv-extra", required=True)
    ap.add_argument("--lip-fixtures", required=True)
    ap.add_argument("--ms", required=True)
    ap.add_argument("--manifest", default=os.path.join(HERE, "results", "inputs.json"))
    args = ap.parse_args()
    m = Manifest(args.out)
    faces(m, args)
    templates(m, args)
    motion(m, args)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump({"seed": SEED, "files": m.entries}, f, indent=1)
    by_task = {}
    for e in m.entries:
        by_task.setdefault((e["task"], e["split"]), 0)
        by_task[(e["task"], e["split"])] += 1
    for (task, split), n in sorted(by_task.items()):
        print(f"{task:10s} {split:5s} {n:5d} files")


if __name__ == "__main__":
    sys.exit(main())
