"""Several faces in one stretch of video: which detection in each frame
belongs to which person, and which of them was speaking.

Detections come per frame as (box, keypoints): box = (x, y, w, h) in
pixels, keypoints = 4 x 2 (right eye, left eye, nose tip, mouth centre) —
what BlazeFace gives through mediapipe. A track follows a face from frame
to frame by overlap with where it was last seen (up to half a second ago),
greedily, best overlap first. A track seen in less than half the frames is
dropped (a passer-by, a false detection); at most MAX_FACES are kept, the
largest first. Frames where a kept face was not found stay None, which
Chaplin's alignment interpolates.

Whether a face was speaking is read from the visual encoder's features of
its mouth crops: a mouth that forms words changes what the encoder sees
from frame to frame; a still one barely does. `activity` is the mean
cosine distance between consecutive frames' features. The threshold is set
from measurements through the running server; see SPEAKING.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

MAX_FACES = 4
MIN_PRESENCE = 0.5       # of the frames
MIN_IOU = 0.3
MAX_GAP = 12             # frames a track may go unseen and still continue (0.5 s at 25 fps)
SPEAKING = 0.05          # activity at or above which a face counts as speaking (first setting; see the measurement notes in README)
SPEAKING_MEASURE = "enc" # which of measures() is the activity

Box = Tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def track_faces(dets: Sequence[Sequence[Tuple[Box, np.ndarray]]], width: int, height: int) -> List[dict]:
    """dets: per frame, a list of (box, keypoints). -> tracks, largest first:
    {"landmarks": [keypoints or None per frame], "box": (x, y, w, h) as
    fractions of the frame (median over the frames it was seen), "presence": 0..1}"""
    n = len(dets)
    tracks: List[dict] = []
    for t, faces in enumerate(dets):
        live = [k for k, tr in enumerate(tracks) if t - tr["last_t"] <= MAX_GAP]
        pairs = sorted(((iou(tracks[k]["last"], faces[i][0]), k, i) for k in live for i in range(len(faces))), reverse=True)
        used_tracks, used_faces = set(), set()
        for score, k, i in pairs:
            if score < MIN_IOU or k in used_tracks or i in used_faces:
                continue
            used_tracks.add(k)
            used_faces.add(i)
            box, kps = faces[i]
            tracks[k]["seen"][t] = (box, kps)
            tracks[k]["last"], tracks[k]["last_t"] = box, t
        for i, (box, kps) in enumerate(faces):
            if i not in used_faces:
                tracks.append({"seen": {t: (box, kps)}, "last": box, "last_t": t})
    kept = [tr for tr in tracks if len(tr["seen"]) >= MIN_PRESENCE * n]
    for tr in kept:
        boxes = np.array([b for b, _ in tr["seen"].values()], dtype=np.float64)
        tr["area"] = float(np.median(boxes[:, 2] * boxes[:, 3]))
        mx, my, mw, mh = np.median(boxes, axis=0)
        tr["box"] = (round(mx / width, 4), round(my / height, 4), round(mw / width, 4), round(mh / height, 4))
    kept.sort(key=lambda tr: -tr["area"])
    return [{"landmarks": [tr["seen"][t][1] if t in tr["seen"] else None for t in range(n)],
             "box": tr["box"], "presence": round(len(tr["seen"]) / max(1, n), 3)} for tr in kept[:MAX_FACES]]


def activity(features: Optional[np.ndarray]) -> float:
    """Mean cosine distance between consecutive frames' features (T, D)."""
    if features is None or len(features) < 2:
        return 0.0
    f = np.asarray(features, dtype=np.float32)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(1.0 - np.sum(f[1:] * f[:-1], axis=1)))


def measures(crops: np.ndarray, features: Optional[np.ndarray], resnet: Optional[np.ndarray]) -> dict:
    """Ways to tell a speaking mouth from a still one, all kept so that the
    choice can be made from measurements: the encoder's and the front end's
    change from frame to frame, and the lips' band of the aligned crop — how
    much its brightness moves over time (relative to its contrast), and its
    mean change between frames in grey levels."""
    c = np.asarray(crops, dtype=np.float32)
    band = c[:, 36:66, 24:72] if c.ndim == 3 and c.shape[1] >= 66 and c.shape[2] >= 72 else c
    level = band.reshape(len(band), -1).mean(axis=1) if len(band) else np.zeros(1)
    contrast = float(band.std(axis=(1, 2)).mean()) if len(band) else 1.0
    return {"enc": round(activity(features), 4), "resnet": round(activity(resnet), 4),
            "band_std": round(float(np.std(level)) / (contrast + 1.0), 4),
            "band_diff": round(float(np.mean(np.abs(np.diff(band, axis=0)))) if len(band) > 1 else 0.0, 3)}
