"""The audio track — opened only by the modes that use it (`audiovisual`,
`audio-attributed`); `visual` mode never calls anything in this file.

Provides the samples, a plain energy voice-activity detector (which parts
of the sound are speech at all), and the speech-recogniser interface with
the two backends that exist: Whisper (optional dependency, any of the
`faster-whisper` or `openai-whisper` packages) and a fake for tests.
"""
from __future__ import annotations

import importlib.util
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .schema import ModelInfo

RATE = 16000


def samples(path: str, start: float = 0.0, end: Optional[float] = None) -> Optional[np.ndarray]:
    """Mono float32 at 16 kHz, or None when there is no audio track."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path]
    if end is not None:
        cmd += ["-t", f"{max(0.0, end - start):.3f}"]
    cmd += ["-vn", "-f", "f32le", "-ac", "1", "-ar", str(RATE), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or len(out.stdout) < 4 * RATE // 10:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32)


@dataclass
class VoiceSpan:
    start: float
    end: float
    energy: float


def voice_activity(audio: np.ndarray, frame_seconds: float = 0.02, min_span: float = 0.15, merge_gap: float = 0.25) -> List[VoiceSpan]:
    """Spans where the sound is loud enough to be speech: RMS per 20 ms frame
    against a threshold between the quiet floor and the loud level."""
    n = int(frame_seconds * RATE)
    if len(audio) < n * 2:
        return []
    frames = len(audio) // n
    rms = np.sqrt((audio[: frames * n].reshape(frames, n) ** 2).mean(axis=1))
    floor = np.percentile(rms, 10)
    loud = np.percentile(rms, 90)
    if loud - floor < 1e-4:
        return []
    threshold = floor + 0.25 * (loud - floor)
    active = rms > threshold
    spans: List[VoiceSpan] = []
    start = None
    for i, on in enumerate(active):
        if on and start is None:
            start = i
        elif not on and start is not None:
            spans.append(VoiceSpan(start * frame_seconds, i * frame_seconds, float(rms[start:i].mean())))
            start = None
    if start is not None:
        spans.append(VoiceSpan(start * frame_seconds, frames * frame_seconds, float(rms[start:].mean())))
    merged: List[VoiceSpan] = []
    for s in spans:
        if merged and s.start - merged[-1].end <= merge_gap:
            merged[-1] = VoiceSpan(merged[-1].start, s.end, max(merged[-1].energy, s.energy))
        else:
            merged.append(s)
    return [s for s in merged if s.end - s.start >= min_span]


@dataclass
class AsrWord:
    text: str
    start: float
    end: float
    confidence: float


@dataclass
class AsrResult:
    words: List[AsrWord] = field(default_factory=list)
    language: Optional[str] = None
    language_probability: float = 0.0


class SpeechRecogniser:
    info: ModelInfo

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> AsrResult:
        """language None = detect."""
        raise NotImplementedError


class WhisperRecogniser(SpeechRecogniser):
    """faster-whisper if installed, else openai-whisper. Word timestamps on."""

    def __init__(self, size: str = "small"):
        self.size = size
        if importlib.util.find_spec("faster_whisper"):
            from faster_whisper import WhisperModel  # type: ignore

            self.backend = "faster-whisper"
            self.model = WhisperModel(size, compute_type="int8")
        elif importlib.util.find_spec("whisper"):
            import whisper  # type: ignore

            self.backend = "openai-whisper"
            self.model = whisper.load_model(size)
        else:
            raise ImportError("no Whisper backend: pip install faster-whisper (or openai-whisper)")
        self.info = ModelInfo(name=f"whisper-{size} ({self.backend})", license="MIT (OpenAI Whisper weights and code)",
                              languages=("multilingual",), vocabulary="open")

    @staticmethod
    def available() -> bool:
        return bool(importlib.util.find_spec("faster_whisper") or importlib.util.find_spec("whisper"))

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> AsrResult:
        words: List[AsrWord] = []
        if self.backend == "faster-whisper":
            segments, info = self.model.transcribe(audio, language=language, word_timestamps=True)
            for segment in segments:
                for w in segment.words or []:
                    words.append(AsrWord(w.word.strip(), float(w.start), float(w.end), float(w.probability)))
            return AsrResult(words, info.language, float(info.language_probability))
        result = self.model.transcribe(audio, language=language, word_timestamps=True)
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append(AsrWord(w["word"].strip(), float(w["start"]), float(w["end"]), float(w.get("probability", 0.5))))
        return AsrResult(words, result.get("language"), 1.0 if language else 0.5)


class FakeRecogniser(SpeechRecogniser):
    """Returns the words it was given: tests supply ground truth timings."""

    def __init__(self, words: Sequence[AsrWord], language: str = "en"):
        self.words = list(words)
        self.language = language
        self.info = ModelInfo(name="fake-asr", license="n/a", languages=(language,), vocabulary="test")

    @staticmethod
    def available() -> bool:
        return True

    def transcribe(self, audio: np.ndarray, language: Optional[str]) -> AsrResult:
        return AsrResult(list(self.words), self.language, 1.0)


def default_recogniser() -> Optional[SpeechRecogniser]:
    return WhisperRecogniser() if WhisperRecogniser.available() else None
