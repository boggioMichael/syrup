"""Pilot A's readers 5 and 6 on the port's lines (export.py): OpenCV and
Tesseract, learning (OpenCV) from the same training lines as the others.

    baselines.py --port PORT_DIR --out FILE.json [--rounds 3]

- OpenCV: the vs-opencv recipe (opencv_side.glyph_cells: grey, Otsu,
  8-connected components, overlapping columns grouped, each glyph's box
  resized to 16 x 16) and one nearest neighbour, trained on the cells of
  every training line that falls into as many glyphs as its label has
  characters. It always answers when it finds glyphs. Timed in-process:
  3 warm-up and 5 timed calls a line, the median; `--rounds` processes.
- Tesseract 5 (the CLI): line mode (--psm 7), the line's characters only
  (a whitelist), each line scaled up four times (cubic) first, as text this
  small is read badly at its own size. All lines in one run (a list file),
  so the time a line is the run's time over the lines: the engine is loaded
  once.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "vs-opencv"))
from opencv_side import glyph_cells  # noqa: E402

CHARS = "0123456789,.[]%"


def load(port, prefix=""):
    meta = {}
    for line in open(os.path.join(port, "meta.txt")).read().splitlines():
        k, v = line.split(" ", 1)
        meta.setdefault(k, v)
    _, h, w = (int(v) for v in meta["lines"].split())
    n = int(meta["lines"].split()[0]) if not prefix else int(meta["train"])
    px = np.fromfile(os.path.join(port, f"{prefix}lines.u8"), np.uint8).reshape(n, h, w, 3)
    truth = open(os.path.join(port, f"{prefix}truth.txt")).read().splitlines()
    kind = open(os.path.join(port, f"{prefix}kind.txt")).read().splitlines()
    return px, truth, kind


def opencv_reader(train, labels):
    samples, ys, used = [], [], 0
    for line, label in zip(train, labels):
        cells = glyph_cells(cv2.cvtColor(line, cv2.COLOR_RGB2BGR))
        if len(cells) != len(label):
            continue
        used += 1
        samples += cells
        ys += [ord(c) for c in label]
    knn = cv2.ml.KNearest_create()
    knn.train(np.array(samples, np.float32), cv2.ml.ROW_SAMPLE, np.array(ys, np.float32))

    def read(bgr):
        cells = glyph_cells(bgr)
        if not cells:
            return None
        _, result, _, _ = knn.findNearest(np.array(cells, np.float32), k=1)
        return "".join(chr(int(v)) for v in result.ravel())

    return read, used


def tally(found, truth, kind):
    out = {}
    for k in ("made", "real"):
        idx = [i for i, kk in enumerate(kind) if kk == k]
        right = sum(found[i] is not None and found[i] == truth[i] for i in idx)
        refused = sum(found[i] is None for i in idx)
        out[k] = {"lines": len(idx), "right": right, "wrong": len(idx) - right - refused, "refused": refused}
    return out


def opencv_round(port):
    """One process's worth: learn, read every line, time every line."""
    train, labels, _ = load(port, "train-")
    test, truth, kind = load(port)
    read, used = opencv_reader(train, labels)
    found, per = [], {"made": [], "real": []}
    for line, k in zip(test, kind):
        bgr = cv2.cvtColor(line, cv2.COLOR_RGB2BGR)
        for _ in range(3):
            text = read(bgr)
        ns = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            text = read(bgr)
            ns.append(time.perf_counter_ns() - t0)
        found.append(text)
        per[k].append(int(np.median(ns)))
    return {"learned from": used, "found": found, "times": per}


def tesseract(port):
    test, truth, kind = load(port)
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, line in enumerate(test):
            big = cv2.resize(cv2.cvtColor(line, cv2.COLOR_RGB2BGR), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            path = os.path.join(tmp, f"{i:05d}.png")
            cv2.imwrite(path, big)
            paths.append(path)
        listing = os.path.join(tmp, "list.txt")
        open(listing, "w").write("\n".join(paths) + "\n")
        t0 = time.perf_counter()
        subprocess.run(["tesseract", listing, os.path.join(tmp, "out"), "--psm", "7", "-c",
                        f"tessedit_char_whitelist={CHARS}"], check=True, capture_output=True,
                       env=dict(os.environ, OMP_THREAD_LIMIT="1"))
        seconds = time.perf_counter() - t0
        pages = open(os.path.join(tmp, "out.txt")).read().split("\f")
    found = [p.strip().replace(" ", "") or None for p in pages[: len(test)]]
    return {"found": found, "seconds": seconds, "us per line, amortised": round(seconds * 1e6 / len(test), 1),
            "version": subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.split("\n")[0]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--one-round", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.one_round:
        json.dump(opencv_round(a.port), open(a.out, "w"))
        return
    _, truth, kind = load(a.port)
    rounds = []
    for i in range(a.rounds):
        out = f"{a.out}.opencv-{i}.json"
        subprocess.run([sys.executable, __file__, "--port", a.port, "--out", out, "--one-round"], check=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1"))
        rounds.append(json.load(open(out)))
        os.remove(out)
    report = {"opencv": {"version": cv2.__version__, "learned from training lines": rounds[0]["learned from"],
                         **tally(rounds[0]["found"], truth, kind)}}
    for k in ("made", "real"):
        report["opencv"][k]["p50 us"] = round(float(np.median([np.percentile(r["times"][k], 50) for r in rounds])) / 1e3, 1)
        report["opencv"][k]["p95 us"] = round(float(np.median([np.percentile(r["times"][k], 95) for r in rounds])) / 1e3, 1)
    t = tesseract(a.port)
    report["tesseract"] = {"version": t["version"], "us per line, amortised": t["us per line, amortised"],
                           **tally(t["found"], truth, kind)}
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
