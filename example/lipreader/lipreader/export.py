"""Transcripts as files: TXT, JSON, SRT, VTT. Uncertain words stay marked
in every format; the notice is written into the text formats."""
from __future__ import annotations

import json
from typing import Iterable, List, Optional, Sequence

from .schema import NOTICE, AnalysisResult, TranscriptSegment

FORMATS = ("txt", "json", "srt", "vtt")


def _timestamp(seconds: float, srt: bool) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{',' if srt else '.'}{ms:03d}"


def _selected(result: AnalysisResult, tracks: Optional[Iterable[int]]) -> List[TranscriptSegment]:
    wanted = set(tracks) if tracks is not None else None
    segments = [s for s in result.segments if wanted is None or s.track_id in wanted]
    return sorted(segments, key=lambda s: (s.start, s.track_id))


def _label(result: AnalysisResult, track_id: int) -> str:
    for t in result.tracks:
        if t.track_id == track_id:
            return t.label
    return "Unattributed" if track_id < 0 else f"Person {track_id}"


def to_txt(result: AnalysisResult, tracks: Optional[Iterable[int]] = None) -> str:
    lines = [f"# {NOTICE}", ""]
    for s in _selected(result, tracks):
        lines.append(f"[{_timestamp(s.start, False)[:-4]}] {_label(result, s.track_id)}: {s.text}   (confidence {s.confidence:.2f}, {s.language}, {s.mode})")
    return "\n".join(lines) + "\n"


def to_json(result: AnalysisResult, tracks: Optional[Iterable[int]] = None) -> str:
    data = result.to_dict()
    if tracks is not None:
        wanted = set(tracks)
        data["segments"] = [s for s in data["segments"] if s["trackId"] in wanted]
        data["tracks"] = [t for t in data["tracks"] if t["trackId"] in wanted]
    return json.dumps(data, indent=1, ensure_ascii=False)


def to_srt(result: AnalysisResult, tracks: Optional[Iterable[int]] = None, labels: bool = True) -> str:
    out = []
    for i, s in enumerate(_selected(result, tracks), 1):
        text = f"{_label(result, s.track_id)}: {s.text}" if labels else s.text
        out.append(f"{i}\n{_timestamp(s.start, True)} --> {_timestamp(s.end, True)}\n{text}\n")
    return "\n".join(out)


def to_vtt(result: AnalysisResult, tracks: Optional[Iterable[int]] = None, labels: bool = True) -> str:
    out = ["WEBVTT", f"NOTE {NOTICE}", ""]
    for s in _selected(result, tracks):
        text = f"<v {_label(result, s.track_id)}>{s.text}" if labels else s.text
        out.append(f"{_timestamp(s.start, False)} --> {_timestamp(s.end, False)}\n{text}\n")
    return "\n".join(out)


def export(result: AnalysisResult, fmt: str, tracks: Optional[Sequence[int]] = None) -> str:
    fmt = fmt.lower()
    if fmt == "txt":
        return to_txt(result, tracks)
    if fmt == "json":
        return to_json(result, tracks)
    if fmt == "srt":
        return to_srt(result, tracks)
    if fmt == "vtt":
        return to_vtt(result, tracks)
    raise ValueError(f"format must be one of {FORMATS}")


MIME = {"txt": "text/plain; charset=utf-8", "json": "application/json", "srt": "application/x-subrip", "vtt": "text/vtt"}
