"""Is this person speaking right now?

Two signals per tracked mouth, both from pixels: how the openness of the
mouth varies over a short window (speech opens and closes the mouth several
times a second; a still face does not), and how much the mouth crop changes
from frame to frame relative to the face as a whole (so that a head turning
does not count as talking). The two are combined into a probability with a
fixed, documented calibration; spans of speech are cut where it drops.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .schema import SpeakingSpan

WINDOW = 9  # frames each side, at 25 fps: ±0.36 s


@dataclass
class ActivityConfig:
    openness_scale: float = 0.03    # rolling std of openness that counts as "clearly speaking"
    motion_scale: float = 0.05      # mouth-vs-face motion ratio excess that counts as "clearly speaking"
    threshold: float = 0.5          # probability above which a frame is speech
    min_span_seconds: float = 0.2
    merge_gap_seconds: float = 0.6     # pauses inside a sentence are shorter than this
    pad_seconds: float = 0.16


def rolling_std(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out[i] = values[lo:hi].std()
    return out


def rolling_mean(values: np.ndarray, window: int = WINDOW) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo, hi = max(0, i - window), min(n, i + window + 1)
        out[i] = values[lo:hi].mean()
    return out


def speaking_probability(openness: np.ndarray, mouth_motion: np.ndarray, face_motion: np.ndarray,
                         config: ActivityConfig = ActivityConfig()) -> np.ndarray:
    """Per-frame probability of speech from the three per-frame signals
    (all the same length; motions are mean absolute frame differences)."""
    if len(openness) == 0:
        return np.zeros(0, dtype=np.float32)
    var = rolling_std(np.asarray(openness, dtype=np.float32))
    excess = np.clip(np.asarray(mouth_motion, dtype=np.float32) - np.asarray(face_motion, dtype=np.float32), 0.0, None)
    motion = rolling_mean(excess)
    a = var / config.openness_scale
    b = motion / config.motion_scale
    evidence = np.maximum(a, b) + 0.5 * np.minimum(a, b)
    return (1.0 - np.exp(-evidence)).astype(np.float32)


def spans(probability: np.ndarray, times: np.ndarray, config: ActivityConfig = ActivityConfig()) -> List[SpeakingSpan]:
    """Continuous stretches of speech, short gaps bridged, each padded a
    little so the first and last visemes are inside."""
    if len(probability) == 0:
        return []
    active = probability >= config.threshold
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(active) - 1))
    fps = 1.0 / max(np.median(np.diff(times)), 1e-6) if len(times) > 1 else 25.0
    merged: List[Tuple[int, int]] = []
    for s, e in out:
        if merged and (s - merged[-1][1]) / fps <= config.merge_gap_seconds:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    pad = int(round(config.pad_seconds * fps))
    result = []
    for s, e in merged:
        if (e - s + 1) / fps < config.min_span_seconds:
            continue
        s2, e2 = max(0, s - pad), min(len(active) - 1, e + pad)
        result.append(SpeakingSpan(float(times[s2]), float(times[e2]), float(probability[s : e + 1].mean())))
    return result
