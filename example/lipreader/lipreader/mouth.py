"""Where the mouth is inside a face, and the crop a lip reader wants.

With two eyes the geometry is exact: the eye distance is the ruler, the line
between the lips is the longest clearly-dark row below the eyes, the corners
are the ends of that dark run (this is The Lip's method, per face). Without
eyes the mouth is placed from the face box alone, which is good enough to
track speaking activity but less good for reading.

The crop replicates LipNet's preprocessing (100x50, the padded mouth
spanning 100 px, a scale fixed per track) so the published weights see what
they were trained on; other models get the same crop resized.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .schema import Box

Point = Tuple[float, float]

MOUTH_WIDTH, MOUTH_HEIGHT = 100, 50
HORIZONTAL_PAD = 0.19
# LipNet's preprocessing pads the mouth by multiplying the corners' absolute
# x coordinates by (1 -/+ 0.19), so the crop scale it was trained with depends
# on where the mouth sat in the 360x288 GRID frame: about x = 180. The same
# scale is reproduced here for a mouth anywhere in any frame by padding as if
# it were at that position; otherwise a face on the right of a wide frame
# gets a crop the model has never seen.
GRID_MOUTH_X = 180.0


@dataclass
class MouthEstimate:
    centre: Point
    corners: Tuple[float, float]  # left x, right x
    from_eyes: bool
    openness: float  # 0..1: dark pixels in the cavity band, a proxy for how open the mouth is


def locate_from_eyes(gray: np.ndarray, left: Point, right: Point) -> Optional[MouthEstimate]:
    """The lip line and corners below a pair of eyes; None if the mouth would
    fall outside the frame."""
    h, w = gray.shape
    d = float(np.hypot(right[0] - left[0], right[1] - left[1]))
    if d < 8:
        return None
    cx = (left[0] + right[0]) / 2.0
    ey = (left[1] + right[1]) / 2.0
    y0, y1 = int(ey + 0.95 * d), min(int(ey + 1.5 * d), h - 1)
    x0, x1 = max(int(cx - 0.45 * d), 0), min(int(cx + 0.45 * d), w - 1)
    if y1 - y0 < 4 or x1 - x0 < 8:
        return None
    band = gray[y0:y1, x0:x1].astype(np.float32)
    skin = float(np.percentile(band, 80))
    darkest = float(band.min())
    threshold = darkest + 0.4 * (skin - darkest)
    dark = band <= threshold
    dark_counts = dark.sum(axis=1).astype(np.float32)
    dark_counts = np.convolve(dark_counts, np.ones(3) / 3, mode="same")
    candidates = np.where(dark_counts >= 0.9 * dark_counts.max())[0]
    cy = y0 + int(candidates[np.argmin(band[candidates].mean(axis=1))])
    # Corners along the lip line.
    wx0, wx1 = max(int(cx - 0.8 * d), 0), min(int(cx + 0.8 * d), w)
    rows = gray[max(cy - 1, 0) : cy + 2, wx0:wx1].astype(np.float32)
    row = rows.min(axis=0)
    skin_row = float(np.percentile(row, 85))
    darkest_row = float(row.min())
    dark_row = row <= darkest_row + 0.45 * (skin_row - darkest_row)
    centre_i = min(max(int(cx) - wx0, 0), len(dark_row) - 1)
    li, gap = centre_i, 0
    while li > 0 and gap <= 2:
        li -= 1
        gap = gap + 1 if not dark_row[li] else 0
    ri, gap = centre_i, 0
    while ri < len(dark_row) - 1 and gap <= 2:
        ri += 1
        gap = gap + 1 if not dark_row[ri] else 0
    lx, rx = wx0 + li + 1, wx0 + ri - 1
    width = rx - lx
    if width < 0.5 * d or width > 1.4 * d:
        lx, rx = cx - 0.47 * d, cx + 0.47 * d
    openness = measure_openness(gray, ((lx + rx) / 2.0, float(cy)), (float(lx), float(rx)), d)
    return MouthEstimate(((lx + rx) / 2.0, float(cy)), (float(lx), float(rx)), True, openness)


def measure_openness(gray: np.ndarray, centre: Point, corners: Tuple[float, float], d: float) -> float:
    """The share of clearly-dark pixels in a band around the lip line,
    relative to the mouth width: a closed mouth is a thin line, an open one
    a cavity. `d` is the eye distance (the scale)."""
    h, w = gray.shape
    cy = int(centre[1])
    lx, rx = corners
    y0, y1 = max(cy - int(0.25 * d), 0), min(cy + int(0.25 * d), h)
    x0, x1 = int(max(lx, 0)), int(min(rx, w))
    if y1 - y0 < 2 or x1 - x0 < 2:
        return 0.0
    cav = gray[y0:y1, x0:x1].astype(np.float32)
    skin = float(np.percentile(cav, 80))
    darkest = float(cav.min())
    dark = (cav <= darkest + 0.35 * (skin - darkest)).sum()
    return float(min(1.0, dark / max(1.0, (rx - lx) * 0.5 * d)))


def locate_from_box(gray: np.ndarray, face: Box) -> MouthEstimate:
    """A mouth placed from the face box alone: roughly where a frontal
    cascade's box puts it."""
    cx = face.x + face.w / 2.0
    cy = face.y + face.h * 0.78
    half = face.w * 0.22
    h, w = gray.shape
    y0, y1 = int(max(cy - face.h * 0.1, 0)), int(min(cy + face.h * 0.1, h))
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
    openness = 0.0
    if y1 > y0 and x1 > x0:
        region = gray[y0:y1, x0:x1].astype(np.float32)
        skin = float(np.percentile(region, 80))
        darkest = float(region.min())
        openness = float(((region <= darkest + 0.35 * (skin - darkest)).sum()) / max(1.0, region.size * 0.5))
    return MouthEstimate((cx, cy), (cx - half, cx + half), False, min(1.0, openness))


class MouthTracker:
    """Per person: smoothed mouth position and a fixed crop scale."""

    def __init__(self, smoothing: float = 0.4):
        self.smoothing = smoothing
        self.centre: Optional[Point] = None
        self.corners: Optional[Tuple[float, float]] = None
        self.ratio: Optional[float] = None
        self.from_eyes = False
        self.eye_distance: Optional[float] = None
        self.last_face: Optional[Box] = None

    def update(self, gray: np.ndarray, face: Box, eyes: Optional[Tuple[Point, Point]]) -> MouthEstimate:
        estimate = locate_from_eyes(gray, *eyes) if eyes is not None else None
        if estimate is not None:
            self.eye_distance = float(np.hypot(eyes[1][0] - eyes[0][0], eyes[1][1] - eyes[0][1]))
        elif self.from_eyes and self.centre is not None and self.eye_distance:
            # No eyes this frame: keep the lip-line geometry, moved with the
            # face box, and measure openness the same way as before so the
            # signal stays comparable from frame to frame.
            dx = dy = 0.0
            if self.last_face is not None:
                dx = face.x + face.w / 2.0 - (self.last_face.x + self.last_face.w / 2.0)
                dy = face.y + face.h / 2.0 - (self.last_face.y + self.last_face.h / 2.0)
            centre = (self.centre[0] + dx, self.centre[1] + dy)
            corners = (self.corners[0] + dx, self.corners[1] + dx)
            estimate = MouthEstimate(centre, corners, False, measure_openness(gray, centre, corners, self.eye_distance))
        else:
            estimate = locate_from_box(gray, face)
        self.last_face = face
        a = self.smoothing
        if self.centre is None:
            self.centre, self.corners = estimate.centre, estimate.corners
        else:
            self.centre = (self.centre[0] + (estimate.centre[0] - self.centre[0]) * a,
                           self.centre[1] + (estimate.centre[1] - self.centre[1]) * a)
            self.corners = (self.corners[0] + (estimate.corners[0] - self.corners[0]) * a,
                            self.corners[1] + (estimate.corners[1] - self.corners[1]) * a)
        self.from_eyes = self.from_eyes or estimate.from_eyes
        if self.ratio is None or (estimate.from_eyes and not self._ratio_from_eyes):
            width = self.corners[1] - self.corners[0]
            padded = width + 2.0 * HORIZONTAL_PAD * GRID_MOUTH_X
            self.ratio = MOUTH_WIDTH / max(padded, 1.0)
            self._ratio_from_eyes = estimate.from_eyes
        return MouthEstimate(self.centre, self.corners, estimate.from_eyes, estimate.openness)

    _ratio_from_eyes = False

    def box(self) -> Optional[Box]:
        """The crop's footprint in frame coordinates."""
        if self.centre is None or self.ratio is None:
            return None
        w, h = MOUTH_WIDTH / self.ratio, MOUTH_HEIGHT / self.ratio
        return Box(self.centre[0] - w / 2.0, self.centre[1] - h / 2.0, w, h)

    def crop(self, rgb: np.ndarray) -> np.ndarray:
        """LipNet's 100x50 crop as float32 (W x H x C, values in [0, 1])."""
        box = self.box()
        assert box is not None
        h, w = rgb.shape[:2]
        # Resize only the region around the mouth, not the whole frame.
        margin = 2.0
        x0 = int(max(box.x - box.w * (margin - 1) / 2, 0))
        y0 = int(max(box.y - box.h * (margin - 1) / 2, 0))
        x1 = int(min(box.x2 + box.w * (margin - 1) / 2, w))
        y1 = int(min(box.y2 + box.h * (margin - 1) / 2, h))
        region = rgb[y0:y1, x0:x1]
        if region.size == 0:
            return np.zeros((MOUTH_WIDTH, MOUTH_HEIGHT, 3), np.float32)
        resized = cv2.resize(region, (max(1, int(round(region.shape[1] * self.ratio))), max(1, int(round(region.shape[0] * self.ratio)))),
                             interpolation=cv2.INTER_LINEAR)
        cxn = (self.centre[0] - x0) * self.ratio
        cyn = (self.centre[1] - y0) * self.ratio
        l, t = int(cxn - MOUTH_WIDTH / 2), int(cyn - MOUTH_HEIGHT / 2)
        crop = resized[max(t, 0) : t + MOUTH_HEIGHT, max(l, 0) : l + MOUTH_WIDTH]
        if crop.shape[0] != MOUTH_HEIGHT or crop.shape[1] != MOUTH_WIDTH:
            padded = np.zeros((MOUTH_HEIGHT, MOUTH_WIDTH, 3), np.uint8)
            padded[: crop.shape[0], : crop.shape[1]] = crop[:MOUTH_HEIGHT, :MOUTH_WIDTH]
            crop = padded
        return crop.swapaxes(0, 1).astype(np.float32) / 255.0
