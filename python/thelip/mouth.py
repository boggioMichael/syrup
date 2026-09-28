"""Mouth crops for LipNet, located with syrup (face + eyes) and refined on
the lip line, replicating the original preprocessing (100x50 crops of the
frame resized so the padded mouth spans 100 px, centred on the mouth).
"""
import os
import sys

import cv2
import numpy as np

try:
    import syrup
except ImportError:  # run from the checkout: python/thelip -> python/syrup
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import syrup

MOUTH_WIDTH, MOUTH_HEIGHT = 100, 50
HORIZONTAL_PAD = 0.19


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    return frames, fps


class MouthLocator:
    """Where the mouth is, frame by frame: the eyes give the scale and the
    horizontal centre, the darkest row below them (the line between the
    lips) gives the vertical centre, and the dark run along that row gives
    the corners. Smoothed over time; the last good estimate is kept when a
    frame yields nothing."""

    def __init__(self):
        self.centre = None
        self.corners = None
        self.eyes = None

    def _eyes(self, frame):
        found = syrup.find_eyes(frame).value or []
        if len(found) == 2:
            (a, b) = sorted(found, key=lambda m: m.centre[0])
            return np.array(a.centre), np.array(b.centre)
        return None

    def locate(self, frame):
        eyes = self._eyes(frame)
        if eyes is not None:
            self.eyes = eyes
        if self.eyes is None:
            return self.centre, self.corners
        left, right = self.eyes
        d = float(np.linalg.norm(right - left))
        cx = (left[0] + right[0]) / 2.0
        ey = (left[1] + right[1]) / 2.0
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        # The line between the lips: the row below the eyes along which the
        # most pixels are clearly darker than the skin around the mouth. A
        # nostril shadow is dark but short; the lip line is dark and long.
        y0, y1 = int(ey + 0.95 * d), min(int(ey + 1.5 * d), h - 1)
        x0, x1 = max(int(cx - 0.45 * d), 0), min(int(cx + 0.45 * d), w - 1)
        band = gray[y0:y1, x0:x1]
        if band.size == 0:
            return self.centre, self.corners
        skin = np.percentile(band, 80)
        darkest = band.min()
        threshold = darkest + 0.4 * (skin - darkest)
        dark_counts = (band <= threshold).sum(axis=1).astype(np.float32)
        dark_counts = np.convolve(dark_counts, np.ones(3) / 3, mode="same")
        # Prefer, among rows within 90% of the longest run, the darkest one.
        candidates = np.where(dark_counts >= 0.9 * dark_counts.max())[0]
        cy = y0 + int(candidates[np.argmin(band[candidates].mean(axis=1))])
        # Corners: along the lip line (3 rows), the run of pixels darker than
        # the skin, allowing tiny gaps.
        wx0, wx1 = max(int(cx - 0.8 * d), 0), min(int(cx + 0.8 * d), w)
        rows = gray[max(cy - 1, 0):cy + 2, wx0:wx1]
        row = rows.min(axis=0)
        offset = wx0
        skin = np.percentile(row, 85)
        darkest = row.min()
        threshold = darkest + 0.45 * (skin - darkest)
        dark = row <= threshold
        centre_i = int(cx) - offset
        li = centre_i
        gap = 0
        while li > 0 and gap <= 2:
            li -= 1
            gap = gap + 1 if not dark[li] else 0
        ri = centre_i
        gap = 0
        while ri < len(dark) - 1 and gap <= 2:
            ri += 1
            gap = gap + 1 if not dark[ri] else 0
        lx, rx = offset + li + 1, offset + ri - 1
        width = rx - lx
        if width < 0.5 * d or width > 1.4 * d:
            lx, rx = cx - 0.47 * d, cx + 0.47 * d
        centre = np.array([(lx + rx) / 2.0, cy], dtype=np.float32)
        corners = (float(lx), float(rx))
        if self.centre is None:
            self.centre, self.corners = centre, corners
        else:
            self.centre = 0.6 * self.centre + 0.4 * centre
            self.corners = (0.6 * self.corners[0] + 0.4 * corners[0], 0.6 * self.corners[1] + 0.4 * corners[1])
        return self.centre, self.corners


def mouth_crops(frames, locator=None, ratio=None):
    """LipNet's crops for a list of RGB frames. Returns (crops as
    (T, 100, 50, 3) float in [0, 1], per-frame boxes in frame coordinates,
    normalize ratio). Frames where the mouth was not found borrow the
    nearest frame's location."""
    locator = locator or MouthLocator()
    located = [locator.locate(frame) for frame in frames]
    # Fill gaps from the nearest located frame, forwards then backwards.
    last = None
    for i, (centre, corners) in enumerate(located):
        if centre is None:
            located[i] = last
        else:
            last = (centre, corners)
    last = None
    for i in range(len(located) - 1, -1, -1):
        if located[i] is None:
            located[i] = last
        else:
            last = located[i]
    crops, boxes = [], []
    for frame, place in zip(frames, located):
        if place is None:
            crops.append(np.zeros((MOUTH_WIDTH, MOUTH_HEIGHT, 3), np.float32))
            boxes.append(None)
            continue
        centre, corners = place
        if ratio is None:
            mouth_left = corners[0] * (1.0 - HORIZONTAL_PAD)
            mouth_right = corners[1] * (1.0 + HORIZONTAL_PAD)
            ratio = MOUTH_WIDTH / float(mouth_right - mouth_left)
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        cxn, cyn = centre * ratio
        l, r = int(cxn - MOUTH_WIDTH / 2), int(cxn + MOUTH_WIDTH / 2)
        t, b = int(cyn - MOUTH_HEIGHT / 2), int(cyn + MOUTH_HEIGHT / 2)
        crop = resized[max(t, 0):b, max(l, 0):r]
        if crop.shape[0] != MOUTH_HEIGHT or crop.shape[1] != MOUTH_WIDTH:
            padded = np.zeros((MOUTH_HEIGHT, MOUTH_WIDTH, 3), np.uint8)
            padded[: crop.shape[0], : crop.shape[1]] = crop[:MOUTH_HEIGHT, :MOUTH_WIDTH]
            crop = padded
        crops.append(crop.swapaxes(0, 1).astype(np.float32) / 255.0)  # W x H x C, as LipNet stores frames
        boxes.append((l / ratio, t / ratio, MOUTH_WIDTH / ratio, MOUTH_HEIGHT / ratio))
    return np.array(crops, dtype=np.float32), boxes, ratio
