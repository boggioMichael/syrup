"""Phrases the server learns from a reader's own kept sentences, and reads
back from the lips alone.

Every sentence a reader keeps (the improve switch) carries the visual
encoder's per-frame features of its mouth crops and, once the microphone
has heard it or the reader has corrected it, its text. For a language
with no lip-reading model (Hebrew, until one is trained), a new utterance
is compared to that reader's past ones with dynamic time warping over the
features, and the nearest phrase's text is the reading — when it is near
enough and clearly nearer than any other phrase. So a reader who said
"בוקר טוב" with the microphone on can say it silently afterwards.

Distances are cosine, summed along the warping path and divided by the
longer length: 0 is the same movement, about 2 the opposite. The
thresholds below are starting points; the numbers come back in every
answer so they can be set from real phones.

The index lives in memory, built from data/ on first use per reader
(the `profile`, a random id the page keeps for itself) and kept current
as samples are added and labelled.
"""
from __future__ import annotations

import json
import os
import re
import threading
from typing import Dict, List, Optional

import numpy as np

MAX_DISTANCE = 0.45      # a match must be this near
MIN_MARGIN = 1.15        # and the nearest other phrase this much farther (ratio)
BAND = 0.3               # Sakoe-Chiba band, as a share of the longer sequence
MIN_CONFIDENCE = 0.4     # a heard text below this is not a label


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = BAND) -> float:
    """Cosine distance summed along the best warping path of two feature
    sequences (T1, D) and (T2, D), divided by the longer length; inf when the
    lengths differ too much. Each row of the table is computed at once: the
    moves from above are a vector minimum, the moves from the left a running
    minimum over prefix sums."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or len(a) == 0 or len(b) == 0 or a.shape[1] != b.shape[1]:
        return float("inf")
    n, m = len(a), len(b)
    if max(n, m) > 2.5 * min(n, m):
        return float("inf")
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cost = (1.0 - a @ b.T).astype(np.float64)          # (n, m)
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    inf = float("inf")
    prev = np.full(m + 1, inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        c = cost[i - 1, lo - 1:hi]                       # costs of cells (i, lo..hi)
        d = np.minimum(prev[lo - 1:hi], prev[lo:hi + 1]) + c   # arriving from above-left or above
        p = np.cumsum(c)                                 # then any run of moves to the right
        cur = np.full(m + 1, inf)
        cur[lo:hi + 1] = p + np.minimum.accumulate(d - p)
        prev = cur
    total = prev[m]
    return float(total / max(n, m)) if total < inf else inf


def valid_profile(profile: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{32}", profile or ""))


def label_of(meta: dict) -> Optional[str]:
    """What the sample's text is, if known: typed, else heard with confidence, else confirmed."""
    if meta.get("corrected"):
        return meta["corrected"]
    if meta.get("heard") and (meta.get("heard_confidence") or 0) >= MIN_CONFIDENCE:
        return meta["heard"]
    if meta.get("confirmed") and meta.get("raw"):
        return meta["raw"]
    return None


def same_phrase(text: str) -> str:
    """The key two texts share when they are the same phrase: letters and digits, one space, lower case."""
    return re.sub(r"[^\w' ]+", " ", text.lower()).strip()


class LearnedPhrases:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.index: Dict[str, Dict[str, dict]] = {}   # profile -> sample id -> {"language", "text", "features"}

    def _load_profile(self, profile: str) -> Dict[str, dict]:
        entries: Dict[str, dict] = {}
        if os.path.isdir(self.data_dir):
            for sid in os.listdir(self.data_dir):
                meta_path = os.path.join(self.data_dir, sid, "meta.json")
                feat_path = os.path.join(self.data_dir, sid, "features.npy")
                if not (os.path.isfile(meta_path) and os.path.isfile(feat_path)):
                    continue
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                except (OSError, ValueError):
                    continue
                if meta.get("profile") != profile:
                    continue
                try:
                    features = np.load(feat_path)
                except (OSError, ValueError):
                    continue
                entries[sid] = {"language": meta.get("language", "en"), "text": label_of(meta), "features": features}
        return entries

    def _profile(self, profile: str) -> Dict[str, dict]:
        if profile not in self.index:
            self.index[profile] = self._load_profile(profile)
        return self.index[profile]

    def note(self, sample_id: str, meta: dict, features: Optional[np.ndarray] = None) -> None:
        """A sample was kept or labelled: keep the index current."""
        profile = meta.get("profile")
        if not profile:
            return
        with self.lock:
            entries = self._profile(profile)
            entry = entries.get(sample_id)
            if entry is None:
                if features is None:
                    return
                entries[sample_id] = {"language": meta.get("language", "en"), "text": label_of(meta), "features": np.asarray(features, dtype=np.float16)}
            else:
                entry["text"] = label_of(meta)
                if features is not None:
                    entry["features"] = np.asarray(features, dtype=np.float16)

    def count(self, profile: str, language: str) -> int:
        with self.lock:
            return sum(1 for e in self._profile(profile).values() if e["language"] == language and e["text"])

    def phrases(self, profile: str, language: str) -> List[dict]:
        with self.lock:
            counts: Dict[str, dict] = {}
            for e in self._profile(profile).values():
                if e["language"] == language and e["text"]:
                    key = same_phrase(e["text"])
                    counts.setdefault(key, {"text": e["text"], "examples": 0})["examples"] += 1
            return sorted(counts.values(), key=lambda p: -p["examples"])

    def match(self, profile: str, language: str, features: np.ndarray) -> dict:
        """The nearest phrase the reader said before in this language, with its
        distance and the margin to the nearest other phrase; `matched` says
        whether it counts as a reading."""
        with self.lock:
            best: Dict[str, tuple] = {}   # phrase key -> (distance, text)
            for e in self._profile(profile).values():
                if e["language"] != language or not e["text"]:
                    continue
                d = dtw_distance(features, e["features"])
                if d == float("inf"):
                    continue
                key = same_phrase(e["text"])
                if key not in best or d < best[key][0]:
                    best[key] = (d, e["text"])
        ranked = sorted(best.values())
        if not ranked:
            return {"matched": False, "examples": 0, "text": None, "distance": None, "margin": None, "nearest": None}
        distance, text = ranked[0]
        margin = (ranked[1][0] / distance) if len(ranked) > 1 and distance > 0 else None
        matched = distance <= MAX_DISTANCE and (margin is None or margin >= MIN_MARGIN)
        return {"matched": matched, "examples": len(ranked), "text": text if matched else None,
                "distance": round(distance, 4), "margin": None if margin is None else round(margin, 3), "nearest": text}
