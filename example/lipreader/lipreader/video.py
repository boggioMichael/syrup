"""Frames in, with their timestamps. Video only: the audio track is opened
by `audio.py`, and only in the modes that use it.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .schema import VideoInfo


def probe(path: str) -> VideoInfo:
    """Duration, frame rate, size and whether there is an audio track, from
    ffprobe when available, from OpenCV otherwise."""
    info = None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path],
            capture_output=True, text=True, timeout=60,
        )
        if out.returncode == 0:
            info = json.loads(out.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        info = None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps > 0 and frames > 0 else 0.0
    has_audio = False
    if info:
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "audio":
                has_audio = True
            elif stream.get("codec_type") == "video":
                rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
                if "/" in rate:
                    num, den = rate.split("/")
                    if float(den) > 0:
                        fps = float(num) / float(den)
                width = int(stream.get("width", width) or width)
                height = int(stream.get("height", height) or height)
        try:
            duration = float(info.get("format", {}).get("duration", duration)) or duration
        except (TypeError, ValueError):
            pass
    return VideoInfo(duration=duration, fps=fps, width=width, height=height, source=os.path.basename(path), has_audio=has_audio)


@dataclass
class Frame:
    index: int      # index in the sampled stream
    t: float        # seconds
    rgb: np.ndarray  # H x W x 3 uint8


class FrameSource:
    """Iterates a video's frames at a target rate. When the source rate is
    close to the target the frames are passed through; otherwise the nearest
    source frame to each target instant is used (no interpolation: lip
    readers were trained on real frames)."""

    def __init__(self, path: str, sample_fps: Optional[float] = None, start: float = 0.0, end: Optional[float] = None):
        self.path = path
        self.info = probe(path)
        self.source_fps = self.info.fps if self.info.fps > 0 else 25.0
        self.fps = float(sample_fps) if sample_fps else self.source_fps
        self.start = max(0.0, start)
        self.end = end
        self.info.sampled_fps = self.fps

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise IOError(f"cannot open video {self.path}")
        try:
            step = self.source_fps / self.fps  # source frames per sampled frame
            source_index = 0
            next_pick = 0.0
            out_index = 0
            passthrough = abs(step - 1.0) < 1e-3
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                t = source_index / self.source_fps
                take = passthrough or source_index >= round(next_pick)
                if take:
                    next_pick += step
                    if t >= self.start and (self.end is None or t <= self.end):
                        yield Frame(out_index, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        out_index += 1
                    elif self.end is not None and t > self.end:
                        break
                source_index += 1
        finally:
            cap.release()


def frame_size(path: str) -> Tuple[int, int]:
    info = probe(path)
    return info.width, info.height
