"""Shot-cut detection: a frame that looks nothing like the previous one.

Grey-level histograms are compared frame to frame; a jump far above the
running level of change is a cut. Camera pans and people moving change a
histogram slowly, a cut changes it at once.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class CutDetector:
    def __init__(self, threshold: float = 0.25, min_gap_frames: int = 5):
        self.threshold = threshold
        self.min_gap = min_gap_frames
        self.previous: Optional[np.ndarray] = None
        self.since_cut = 10**9
        self.last_distance = 0.0

    @staticmethod
    def _histogram(rgb: np.ndarray) -> np.ndarray:
        small = cv2.resize(rgb, (160, 90), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).ravel()
        return hist / max(hist.sum(), 1.0)

    def update(self, rgb: np.ndarray) -> bool:
        """True when this frame starts a new shot."""
        hist = self._histogram(rgb)
        cut = False
        if self.previous is not None:
            self.last_distance = float(0.5 * np.abs(hist - self.previous).sum())  # 0 identical .. 1 disjoint
            if self.last_distance > self.threshold and self.since_cut >= self.min_gap:
                cut = True
                self.since_cut = 0
        self.previous = hist
        self.since_cut += 1
        return cut
