from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """Pixels of the image you passed, covering [x, x+w) x [y, y+h)."""

    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Track:
    """An object's identity across a session's frames."""

    id: int
    age_frames: int
    # Movement of the box centre since the previous frame, in pixels.
    velocity: tuple[float, float]


@dataclass(frozen=True)
class Found:
    label: str
    box: Box
    # The provider's score in [0, 1]; not a calibrated probability.
    confidence: float
    keypoints: dict[str, tuple[float, float]] = field(default_factory=dict)
    # What a word says, for targets that read text.
    text: str | None = None
    # The measured quantity in [0, 1], for measure_* operations.
    value: float | None = None
    # Set by track_* sessions.
    track: Track | None = None


@dataclass(frozen=True)
class Provenance:
    operation: str
    intent: str
    plan_hash: str
    artifact: dict
    # "compiled", "loaded_from_disk" or "in_memory"
    artifact_status: str
    providers: list
    params: dict
    image: tuple
    prepare_ms: float
    execute_ms: float
    # The frame's number in its session, from 1.
    frame: int | None = None


class FindResult(Sequence):
    """Ordered results. Empty (and falsy) means the operation ran and found
    nothing; failures raise instead."""

    def __init__(self, items, provenance):
        self._items = tuple(items)
        self.provenance = provenance

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    def __repr__(self):
        return f"FindResult({list(self._items)!r})"


def from_json(text):
    data = json.loads(text)
    items = [
        Found(
            f["label"],
            Box(**f["box"]),
            f["confidence"],
            {k["name"]: (k["x"], k["y"]) for k in f["keypoints"]},
            f["text"],
            f["value"],
            Track(f["track"]["id"], f["track"]["age_frames"], tuple(f["track"]["velocity"])) if f["track"] else None,
        )
        for f in data["items"]
    ]
    provenance = dict(data["provenance"], image=tuple(data["provenance"]["image"]))
    return FindResult(items, Provenance(**provenance))
