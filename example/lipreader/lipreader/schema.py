"""The data every part of lipreader exchanges. Mirrors
example/shared-types/schema/*.schema.json; `to_dict` produces exactly that
JSON and the tests validate it against the schema.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

NOTICE = (
    "Lip-reading output is probabilistic. Many sounds look identical on the lips, so this "
    "transcript is a best guess, not a verbatim record; words marked [like this?] are uncertain."
)

Mode = str  # "visual" | "audiovisual" | "audio-attributed"
MODES = ("visual", "audiovisual", "audio-attributed")


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def centre(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def iou(self, other: "Box") -> float:
        ix = max(0.0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0.0, min(self.y2, other.y2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def scaled(self, factor: float) -> "Box":
        return Box(self.x * factor, self.y * factor, self.w * factor, self.h * factor)

    def to_dict(self) -> Dict[str, float]:
        return {"x": round(self.x, 2), "y": round(self.y, 2), "w": round(self.w, 2), "h": round(self.h, 2)}


@dataclass
class Word:
    text: str
    start: float
    end: float
    confidence: float
    uncertain: bool
    raw: Optional[str] = None

    def rendered(self) -> str:
        return f"[{self.text}?]" if self.uncertain else self.text

    def to_dict(self) -> Dict[str, Any]:
        d = {"text": self.text, "start": round(self.start, 3), "end": round(self.end, 3),
             "confidence": round(self.confidence, 3), "uncertain": self.uncertain}
        if self.raw is not None and self.raw != self.text:
            d["raw"] = self.raw
        return d


@dataclass
class TranscriptSegment:
    id: str
    track_id: int
    start: float
    end: float
    words: List[Word]
    language: str
    mode: Mode
    model: str

    @property
    def text(self) -> str:
        return " ".join(w.rendered() for w in self.words)

    @property
    def confidence(self) -> float:
        return sum(w.confidence for w in self.words) / len(self.words) if self.words else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "trackId": self.track_id, "start": round(self.start, 3), "end": round(self.end, 3),
            "text": self.text, "words": [w.to_dict() for w in self.words], "language": self.language,
            "confidence": round(self.confidence, 3), "mode": self.mode, "model": self.model,
        }


@dataclass
class SpeakingSpan:
    start: float
    end: float
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "probability": round(self.probability, 3)}


@dataclass
class Keyframe:
    t: float
    box: Box
    mouth: Optional[Box] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"t": round(self.t, 3), "box": self.box.to_dict()}
        if self.mouth is not None:
            d["mouth"] = self.mouth.to_dict()
        return d


@dataclass
class PersonTrack:
    """One visible person, numbered within this video only."""

    track_id: int
    first_seen: float
    last_seen: float
    keyframes: List[Keyframe] = field(default_factory=list)
    speaking: List[SpeakingSpan] = field(default_factory=list)
    detected_language: Optional[str] = None
    segment_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def label(self) -> str:
        return f"Person {self.track_id}"

    def box_at(self, t: float) -> Optional[Box]:
        k = self.keyframes
        if not k:
            return None
        if t <= k[0].t:
            return k[0].box
        if t >= k[-1].t:
            return k[-1].box
        lo, hi = 0, len(k) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if k[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = k[lo], k[hi]
        f = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
        return Box(a.box.x + (b.box.x - a.box.x) * f, a.box.y + (b.box.y - a.box.y) * f,
                   a.box.w + (b.box.w - a.box.w) * f, a.box.h + (b.box.h - a.box.h) * f)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trackId": self.track_id, "label": self.label, "firstSeen": round(self.first_seen, 3),
            "lastSeen": round(self.last_seen, 3), "keyframes": [k.to_dict() for k in self.keyframes],
            "speaking": [s.to_dict() for s in self.speaking], "detectedLanguage": self.detected_language,
            "segmentIds": list(self.segment_ids), "confidence": round(self.confidence, 3),
        }


@dataclass
class ModelInfo:
    name: str
    license: str
    languages: Sequence[str] = ()
    vocabulary: str = "open"
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "license": self.license, "languages": list(self.languages), "vocabulary": self.vocabulary}
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int
    source: str = ""
    sampled_fps: Optional[float] = None
    has_audio: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = {"duration": round(self.duration, 3), "fps": round(self.fps, 3), "width": self.width, "height": self.height}
        if self.source:
            d["source"] = self.source
        if self.sampled_fps:
            d["sampledFps"] = round(self.sampled_fps, 3)
        return d


@dataclass
class Options:
    mode: Mode = "visual"
    language: str = "auto"
    speakers: Union[str, List[int]] = "all"
    sample_fps: Optional[float] = None
    max_faces: int = 8
    start_time: float = 0.0
    end_time: Optional[float] = None
    uncertain_below: float = 0.5

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Options":
        d = d or {}
        o = cls()
        o.mode = d.get("mode", o.mode)
        if o.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {o.mode!r}")
        o.language = str(d.get("language", o.language)).lower()
        o.speakers = d.get("speakers", o.speakers)
        if not (o.speakers in ("all", "current") or (isinstance(o.speakers, list) and all(isinstance(i, int) for i in o.speakers))):
            raise ValueError("speakers must be \"all\", \"current\" or a list of track ids")
        o.sample_fps = d.get("sampleFps", o.sample_fps)
        o.max_faces = int(d.get("maxFaces", o.max_faces))
        o.start_time = float(d.get("startTime", o.start_time))
        o.end_time = d.get("endTime", o.end_time)
        o.uncertain_below = float(d.get("uncertainBelow", o.uncertain_below))
        return o

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"mode": self.mode, "language": self.language, "speakers": self.speakers,
                             "maxFaces": self.max_faces, "startTime": self.start_time, "uncertainBelow": self.uncertain_below}
        if self.sample_fps:
            d["sampleFps"] = self.sample_fps
        if self.end_time is not None:
            d["endTime"] = self.end_time
        return d


@dataclass
class AnalysisResult:
    video: VideoInfo
    mode: Mode
    language_requested: str
    language_used: Optional[str]
    language_detection: str  # requested | assumed | audio-detected | unavailable
    tracks: List[PersonTrack]
    segments: List[TranscriptSegment]
    models: Dict[str, Any]
    processing_seconds: float
    stages: Dict[str, float] = field(default_factory=dict)
    language_note: str = ""
    warnings: List[str] = field(default_factory=list)

    @property
    def realtime_factor(self) -> float:
        return self.processing_seconds / self.video.duration if self.video.duration > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        language = {"requested": self.language_requested, "used": self.language_used, "detection": self.language_detection}
        if self.language_note:
            language["note"] = self.language_note
        models: Dict[str, Any] = {}
        for key, value in self.models.items():
            models[key] = value.to_dict() if isinstance(value, ModelInfo) else value
        return {
            "version": "1",
            "video": self.video.to_dict(),
            "mode": self.mode,
            "language": language,
            "tracks": [t.to_dict() for t in self.tracks],
            "segments": [s.to_dict() for s in self.segments],
            "models": models,
            "timing": {"processingSeconds": round(self.processing_seconds, 3), "realtimeFactor": round(self.realtime_factor, 3),
                       "stages": {k: round(v, 3) for k, v in self.stages.items()}},
            "notice": NOTICE,
            "warnings": list(self.warnings),
        }
