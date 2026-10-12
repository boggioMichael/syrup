"""The OpenCV side of the Syrup vs OpenCV comparison (see PREREGISTRATION.md).

Same command line and same JSON as `examples/vs_opencv.rs`:

    python3 bench/vs-opencv/opencv_side.py faces|templates|motion \
        --data DIR --split dev|test [--threads N] [--calls K] [--warmup W] \
        [--dump DIR] --cascades DIR --out FILE.json

Each image input is called `warmup` times untimed, then `calls` times timed;
a video is processed once, front to back, its first `warmup` pairs untimed.
Only the OpenCV calls (and, for motion, the region filter that Syrup's call
also does) are inside the timer; decoding is not.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time

import cv2
import numpy as np

# Rec. 709 luma weights as Syrup has them: 16-bit fixed point summing to
# exactly 65536. In float32 every product and partial sum of these with a
# byte difference is exact, so OpenCV computes the same luma as Syrup.
LUMA_BGR = np.array([[4732 / 65536, 46871 / 65536, 13933 / 65536, -0.5]], np.float32)
# Syrup: moved when luma >= 28 exactly. transform() rounds to the nearest
# integer (ties to even), so with the -0.5 offset round(s - 0.5) >= 28
# exactly when s >= 28; threshold() keeps values > 27.
THRESHOLD = 27
MIN_RUN = 4
MIN_HEIGHT = 4
MIN_AREA = 24


def usage_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime, r.ru_stime, r.ru_maxrss


def timed(fn):
    start = time.perf_counter_ns()
    value = fn()
    return value, time.perf_counter_ns() - start


def sorted_pngs(directory, keep=lambda n: True):
    return sorted(os.path.join(directory, n) for n in os.listdir(directory) if n.endswith(".png") and keep(n))


def faces(a):
    directory = os.path.join(a.data, "faces", a.split)
    inputs = sorted_pngs(directory, lambda n: n.endswith(".grey.png"))
    path = os.path.join(a.cascades, "haarcascade_frontalface_alt2.xml")
    cascade, load_ns = timed(lambda: cv2.CascadeClassifier(path))
    if cascade.empty():
        raise SystemExit(f"could not load {path}")
    items, first_call = [], None
    for p in inputs:
        image = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        found = None
        for i in range(a.warmup):
            found, t = timed(lambda: cascade.detectMultiScale2(image, scaleFactor=1.1, minNeighbors=3, minSize=(0, 0)))
            if first_call is None:
                first_call = t
        times = []
        for _ in range(a.calls):
            found, t = timed(lambda: cascade.detectMultiScale2(image, scaleFactor=1.1, minNeighbors=3, minSize=(0, 0)))
            times.append(t)
        rects, neighbours = found
        boxes = [[int(x), int(y), int(w), int(h), int(n)] for (x, y, w, h), n in zip(rects, neighbours)]
        items.append({"input": os.path.relpath(p, a.data), "times_ns": times, "output": {"boxes": boxes}})
    return items, {"load_ns": load_ns, "first_call_ns": first_call or 0}


def templates(a):
    directory = os.path.join(a.data, "templates", a.split)
    with open(os.path.join(directory, "cases.json")) as f:
        cases = json.load(f)
    images, items, first_call = {}, [], None
    for index, case in enumerate(cases):
        if case["image"] not in images:
            images[case["image"]] = cv2.imread(os.path.join(a.data, case["image"]), cv2.IMREAD_GRAYSCALE)
        image = images[case["image"]]
        template = cv2.imread(os.path.join(a.data, case["template"]), cv2.IMREAD_GRAYSCALE)
        label = f"{index}|{case['template']}|{case['variant']}"
        # The score map is allocated once per case, outside the timer, and
        # reused by every call (as a C++ caller would reuse its Mat).
        scores = np.empty((image.shape[0] - template.shape[0] + 1, image.shape[1] - template.shape[1] + 1), np.float32)

        def search():
            cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, result=scores)
            return cv2.minMaxLoc(scores)

        best = None
        for i in range(a.warmup):
            best, t = timed(search)
            if first_call is None:
                first_call = t
        times = []
        for _ in range(a.calls):
            best, t = timed(search)
            times.append(t)
        _, score, _, (x, y) = best
        items.append({"input": label, "times_ns": times, "output": {"x": int(x), "y": int(y), "score": float(score)}})
    return items, {"first_call_ns": first_call or 0}


KERNEL = np.ones((1, MIN_RUN), np.uint8)


class Motion:
    """OpenCV's moved-pixel mask and changed regions of two frames, with
    every full-frame buffer allocated once and reused (as a C++ caller
    would). Connected components by Grana's BBDT: on the dev frames it gave
    the same output as the default algorithm, faster."""

    def __init__(self, shape):
        h, w = shape[:2]
        self.diff = np.empty((h, w, 3), np.uint8)
        self.luma = np.empty((h, w), np.uint8)
        self.mask = np.empty((h, w), np.uint8)
        self.runs = np.empty((h, w), np.uint8)
        self.labels = np.empty((h, w), np.int32)

    def __call__(self, a_bgr, b_bgr):
        cv2.absdiff(a_bgr, b_bgr, dst=self.diff)
        cv2.transform(self.diff, LUMA_BGR, dst=self.luma)
        cv2.threshold(self.luma, THRESHOLD, 255, cv2.THRESH_BINARY, dst=self.mask)
        cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, KERNEL, dst=self.runs,
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        n, _, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
            self.runs, 8, cv2.CV_32S, cv2.CCL_GRANA, labels=self.labels)
        stats = stats[1:n]
        keep = (stats[:, cv2.CC_STAT_HEIGHT] >= MIN_HEIGHT) & (
            stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT] >= MIN_AREA)
        return self.mask, stats[keep, :4]


def motion_regions(a_bgr, b_bgr):
    """The plain recipe (fresh buffers, default components algorithm): the
    reference the reusing version is checked against."""
    diff = cv2.absdiff(a_bgr, b_bgr)
    luma = cv2.transform(diff, LUMA_BGR)
    _, mask = cv2.threshold(luma, THRESHOLD, 255, cv2.THRESH_BINARY)
    runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    stats = stats[1:n]
    keep = (stats[:, cv2.CC_STAT_HEIGHT] >= MIN_HEIGHT) & (
        stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT] >= MIN_AREA)
    return mask, stats[keep, :4]


def motion(a):
    root = os.path.join(a.data, "motion", a.split)
    items = []
    for name in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
        frames = [cv2.imread(p, cv2.IMREAD_COLOR) for p in sorted_pngs(os.path.join(root, name))]
        detect = Motion(frames[0].shape)
        for i in range(len(frames) - 1):
            (mask, regions), t = timed(lambda: detect(frames[i], frames[i + 1]))
            if a.dump:
                os.makedirs(os.path.join(a.dump, name), exist_ok=True)
                cv2.imwrite(os.path.join(a.dump, name, f"{i:04d}.png"), mask)
            if i < a.warmup:
                continue
            items.append({"input": f"{name}/{i:04d}", "times_ns": [t],
                          "output": {"moved": int(np.count_nonzero(mask)),
                                     "regions": [[int(v) for v in r] for r in regions]}})
    return items, {}


GLYPH_CELL = 16


def glyph_cells(bgr):
    """The usual OpenCV recipe for a line of light text: grey, Otsu's
    threshold, 8-connected components (2 pixels or more), components whose
    columns overlap grouped into one glyph, left to right, each glyph's
    box of the binary image resized to a 16x16 cell."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = sorted((int(x), int(y), int(w), int(h)) for x, y, w, h, area in stats[1:n] if area >= 2)
    groups = []
    for x, y, w, h in boxes:
        if groups and x <= groups[-1][2]:
            g = groups[-1]
            groups[-1] = [g[0], min(g[1], y), max(g[2], x + w - 1), max(g[3], y + h - 1)]
        else:
            groups.append([x, y, x + w - 1, y + h - 1])
    cells = [cv2.resize(binary[y0:y1 + 1, x0:x1 + 1], (GLYPH_CELL, GLYPH_CELL), interpolation=cv2.INTER_AREA)
             .reshape(-1).astype(np.float32) / 255.0 for x0, y0, x1, y1 in groups]
    return cells


def glyphs(a):
    directory = os.path.join(a.data, "glyphs", a.split)
    with open(os.path.join(directory, "cases.json")) as f:
        spec = json.load(f)
    samples, labels, learned = [], [], []

    def train():
        knn = cv2.ml.KNearest_create()
        for path in spec["train"].values():
            cells = glyph_cells(cv2.imread(os.path.join(a.data, path), cv2.IMREAD_COLOR))
            if len(cells) != len(spec["chars"]):
                learned.append({"file": path, "error": f"{len(cells)} glyphs for {len(spec['chars'])} characters"})
                continue
            samples.extend(cells)
            labels.extend(ord(c) for c in spec["chars"])
            learned.append({"file": path, "learned": len(cells)})
        if samples:
            knn.train(np.array(samples, np.float32), cv2.ml.ROW_SAMPLE, np.array(labels, np.float32))
        return knn

    knn, learn_ns = timed(train)

    def read(bgr):
        cells = glyph_cells(bgr)
        if not cells or not samples:
            return None
        _, result, _, _ = knn.findNearest(np.array(cells, np.float32), k=1)
        return "".join(chr(int(v)) for v in result.ravel())

    items, first_call = [], None
    for case in spec["cases"]:
        image = cv2.imread(os.path.join(a.data, case["image"]), cv2.IMREAD_COLOR)
        text = None
        for i in range(a.warmup):
            text, t = timed(lambda: read(image))
            if first_call is None:
                first_call = t
        times = []
        for _ in range(a.calls):
            text, t = timed(lambda: read(image))
            times.append(t)
        output = {"text": text} if text is not None else {"refused": "no glyphs"}
        items.append({"input": case["image"], "times_ns": times, "output": output})
    return items, {"load_ns": learn_ns, "first_call_ns": first_call or 0, "learned": learned}


def binding_overhead_ns():
    """Median cost of a Python→OpenCV call that does nothing measurable."""
    tiny = np.zeros((1, 1), np.uint8)
    samples = []
    for _ in range(2000):
        _, t = timed(lambda: cv2.absdiff(tiny, tiny))
        samples.append(t)
    return int(np.median(samples))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["faces", "templates", "motion", "glyphs"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--calls", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dump")
    ap.add_argument("--cascades", default=os.path.join(os.path.dirname(cv2.__file__), "data"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)
    cv2.setUseOptimized(True)
    base_user, base_sys, base_rss = usage_now()
    started = time.perf_counter()
    items, init = {"faces": faces, "templates": templates, "motion": motion, "glyphs": glyphs}[a.task](a)
    wall = time.perf_counter() - started
    user, system, rss = usage_now()
    init["binding_overhead_ns"] = binding_overhead_ns()
    out = {
        "impl": "opencv", "version": cv2.__version__, "task": a.task, "split": a.split,
        "threads": a.threads, "calls": a.calls, "warmup": a.warmup, "init": init,
        "process": {"wall_s": wall, "cpu_user_s": user - base_user, "cpu_sys_s": system - base_sys,
                    "max_rss_kb": rss, "rss_after_imports_kb": base_rss},
        "items": items,
    }
    with open(a.out, "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    sys.exit(main())
