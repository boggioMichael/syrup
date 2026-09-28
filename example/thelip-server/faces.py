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

Whether a face was speaking is read from its aligned 96x96 mouth crops:
the lips' band of the crop darkens and lightens as the mouth opens and
closes, so the standard deviation over time of that band's mean
brightness, relative to its contrast, is high for a speaking mouth and
near zero for a still one. Measured through the running server
(2026-09-28, GRID's sample clips composed two to a frame): speaking faces
0.10 - 0.25 over a whole three-second clip, silences included; still
mouths (a clip's first quarter second, played back and forth) 0.004 -
0.025; a frozen frame 0. The encoder's own frame-to-frame change was
tried first and does not work: it gave 0.136 for a frozen frame and
0.15 - 0.21 for both speaking and still mouths; the visual front end's
overlapped too (0.22 - 0.40 against 0.25 - 0.34).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

MAX_FACES = 4
MIN_PRESENCE = 0.5       # of the frames
MIN_IOU = 0.3
MAX_GAP = 12             # frames a track may go unseen and still continue (0.5 s at 25 fps)
SPEAKING = 0.06          # mouth activity at or above which one of several faces counts as speaking (measured above)
STILL = 0.03             # below this even a lone face is not read: nothing moved, and the model would make up a sentence

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


def mouth_activity(crops: np.ndarray) -> float:
    """How much the lips' band of aligned 96x96 mouth crops (T, 96, 96) moves:
    the standard deviation over time of its mean brightness, over its contrast."""
    c = np.asarray(crops, dtype=np.float32)
    if c.ndim != 3 or len(c) < 2:
        return 0.0
    band = c[:, 36:66, 24:72] if c.shape[1] >= 66 and c.shape[2] >= 72 else c
    level = band.reshape(len(band), -1).mean(axis=1)
    contrast = float(band.std(axis=(1, 2)).mean())
    return round(float(np.std(level)) / (contrast + 1.0), 4)
