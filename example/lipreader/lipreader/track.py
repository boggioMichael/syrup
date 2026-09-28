"""Keeping each face the same person from frame to frame.

Detections are matched to live tracks by intersection-over-union with the
Hungarian assignment; a track survives a few frames without a detection
(its box is held) and ends after that. A shot cut (`cuts.py`) ends every
track at once: after a cut nothing on screen is the same person, whatever
the boxes say.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detect import Detection
from .schema import Box, Keyframe, PersonTrack

Point = Tuple[float, float]


@dataclass
class TrackState:
    track_id: int
    box: Box
    first_t: float
    last_t: float
    last_seen_t: float
    frames_missing: int = 0
    detections: int = 0
    attempts: int = 0   # frames on which the detector actually ran
    frames: int = 0
    eyes: Optional[Tuple[Point, Point]] = None
    profile: bool = False
    keyframes: List[Keyframe] = field(default_factory=list)
    # Per sampled frame index: the box (held while missing) and eyes if any.
    boxes: Dict[int, Box] = field(default_factory=dict)
    eyes_by_frame: Dict[int, Tuple[Point, Point]] = field(default_factory=dict)
    detected: set = field(default_factory=set)

    def to_person_track(self) -> PersonTrack:
        confidence = self.detections / self.attempts if self.attempts else 0.0
        return PersonTrack(
            track_id=self.track_id, first_seen=self.first_t, last_seen=self.last_t,
            keyframes=list(self.keyframes), confidence=round(min(1.0, confidence), 3),
        )


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, grace_frames: int = 12, keyframe_every: int = 5, smoothing: float = 0.5):
        self.iou_threshold = iou_threshold
        self.grace_frames = grace_frames
        self.keyframe_every = keyframe_every
        self.smoothing = smoothing
        self.live: List[TrackState] = []
        self.finished: List[TrackState] = []
        self.next_id = 1

    def cut(self, t: float) -> None:
        """A shot change: every live track ends here."""
        for track in self.live:
            self.finished.append(track)
        self.live = []

    def update(self, frame_index: int, t: float, detections: Sequence[Detection]) -> List[TrackState]:
        """Assign this frame's detections; returns the tracks alive after it."""
        live = self.live
        if live and detections:
            cost = np.ones((len(live), len(detections)), dtype=np.float32)
            for i, track in enumerate(live):
                for j, det in enumerate(detections):
                    cost[i, j] = 1.0 - track.box.iou(det.box)
            rows, cols = linear_sum_assignment(cost)
            matched_tracks, matched_dets = set(), set()
            for i, j in zip(rows, cols):
                if cost[i, j] <= 1.0 - self.iou_threshold:
                    self._observe(live[i], detections[j], frame_index, t)
                    matched_tracks.add(i)
                    matched_dets.add(j)
        else:
            matched_tracks, matched_dets = set(), set()
        held_frame = bool(detections) and all(d.held for d in detections)
        for i, track in enumerate(live):
            if i not in matched_tracks:
                track.frames_missing += 1
                track.frames += 1
                if not held_frame:
                    track.attempts += 1
                track.boxes[frame_index] = track.box
        for j, det in enumerate(detections):
            if j not in matched_dets:
                self._start(det, frame_index, t)
        still, ended = [], []
        for track in self.live:
            (ended if track.frames_missing > self.grace_frames else still).append(track)
        for track in ended:
            self._trim(track)
            self.finished.append(track)
        self.live = still
        return list(self.live)

    def finish(self) -> List[TrackState]:
        for track in self.live:
            self._trim(track)
            self.finished.append(track)
        self.live = []
        return sorted(self.finished, key=lambda t: t.track_id)

    def _start(self, det: Detection, frame_index: int, t: float) -> None:
        track = TrackState(track_id=self.next_id, box=det.box, first_t=t, last_t=t, last_seen_t=t, profile=det.profile)
        self.next_id += 1
        self.live.append(track)
        self._observe(track, det, frame_index, t, new=True)

    def _observe(self, track: TrackState, det: Detection, frame_index: int, t: float, new: bool = False) -> None:
        if new:
            track.box = det.box
        else:
            a = self.smoothing
            b = track.box
            track.box = Box(b.x + (det.box.x - b.x) * a, b.y + (det.box.y - b.y) * a,
                            b.w + (det.box.w - b.w) * a, b.h + (det.box.h - b.h) * a)
        track.frames += 1
        track.last_t = t
        track.boxes[frame_index] = track.box
        if det.held:
            return
        track.frames_missing = 0
        track.detections += 1
        track.attempts += 1
        track.last_seen_t = t
        track.profile = det.profile
        track.detected.add(frame_index)
        if det.eyes is not None:
            track.eyes = det.eyes
            track.eyes_by_frame[frame_index] = det.eyes
        if not track.keyframes or track.detections % self.keyframe_every == 0:
            track.keyframes.append(Keyframe(t, track.box))

    def _trim(self, track: TrackState) -> None:
        """Drop the frames held after the last detection."""
        last = max(track.detected, default=-1)
        for i in [i for i in track.boxes if i > last]:
            del track.boxes[i]
        track.frames = len(track.boxes)
        track.last_t = track.last_seen_t
        if track.keyframes and track.keyframes[-1].t < track.last_seen_t:
            track.keyframes.append(Keyframe(track.last_seen_t, track.box))
