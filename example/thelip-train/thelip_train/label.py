"""Transcripts with word times for every downloaded video.

Human captions win when the source has them (YouTube's own, NASA's .srt):
their cues become segments, and the words inside a cue get times by even
spread, good enough to cut clips. Everything else is transcribed by Whisper
through faster-whisper with word timestamps — `large-v3` for English and
most languages, ivrit.ai's Hebrew-tuned Whisper for Hebrew.

    labels/<source>/<id>.json   {"language", "how": "captions"|"whisper:<model>",
                                 "segments": [{"start", "end", "text", "words": [{"w","s","e","p"}]}]}
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from .common import DIRS, say

WHISPER = {"he": "ivrit-ai/whisper-large-v3-turbo-ct2", "*": "large-v3"}


def _parse_cues(path: str) -> List[dict]:
    """.vtt or .srt -> [{"start","end","text"}], times in seconds."""
    def t(s: str) -> float:
        s = s.strip().replace(",", ".")
        parts = s.split(":")
        parts = [float(p) for p in parts]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]

    cues, cur, text = [], None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            m = re.match(r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s+-->\s+(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})", line)
            if m:
                if cur and text:
                    cues.append({"start": cur[0], "end": cur[1], "text": " ".join(text)})
                cur, text = (t(m.group(1)), t(m.group(2))), []
            elif not line:
                if cur and text:
                    cues.append({"start": cur[0], "end": cur[1], "text": " ".join(text)})
                cur, text = None, []
            elif cur and not line.isdigit() and not line.startswith(("WEBVTT", "NOTE", "Kind:", "Language:")):
                text.append(re.sub(r"<[^>]+>", "", line))
    if cur and text:
        cues.append({"start": cur[0], "end": cur[1], "text": " ".join(text)})
    # Rolling captions repeat text across cues; keep the first appearance.
    out, last = [], ""
    for c in cues:
        if c["text"] and c["text"] != last:
            out.append(c)
            last = c["text"]
    return out


def _spread_words(seg: dict) -> dict:
    words = seg["text"].split()
    if not words:
        return dict(seg, words=[])
    step = (seg["end"] - seg["start"]) / len(words)
    return dict(seg, words=[{"w": w, "s": seg["start"] + i * step, "e": seg["start"] + (i + 1) * step, "p": 1.0} for i, w in enumerate(words)])


def captions_for(video: str) -> Optional[str]:
    base = os.path.splitext(video)[0]
    folder = os.path.dirname(video)
    for name in sorted(os.listdir(folder)):
        if name.startswith(os.path.basename(base) + ".") and name.endswith((".vtt", ".srt")):
            return os.path.join(folder, name)
    return None


_models: dict = {}


def whisper_model(language: str, device: str = "auto"):
    name = WHISPER.get(language, WHISPER["*"])
    if name not in _models:
        from faster_whisper import WhisperModel

        say(f"loading Whisper {name}")
        try:
            _models[name] = WhisperModel(name, device=device, compute_type="float16" if device != "cpu" else "int8")
        except Exception as e:  # noqa: BLE001
            if name != WHISPER["*"]:
                say(f"  {name} unavailable ({e}); falling back to {WHISPER['*']} with language={language}")
                name = WHISPER["*"]
                _models[name] = WhisperModel(name, device=device, compute_type="float16" if device != "cpu" else "int8")
            else:
                raise
    return name, _models[name]


def transcribe(video: str, language: str, device: str = "auto") -> dict:
    name, model = whisper_model(language, device)
    segments, info = model.transcribe(video, language=language, word_timestamps=True, vad_filter=True, beam_size=5)
    out = []
    for s in segments:
        words = [{"w": w.word.strip(), "s": float(w.start), "e": float(w.end), "p": float(w.probability)} for w in (s.words or []) if w.word.strip()]
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip(), "words": words})
    return {"language": language, "how": f"whisper:{name}", "segments": out}


def label_video(video: str, language: str, device: str = "auto") -> str:
    source = os.path.basename(os.path.dirname(video))
    out = os.path.join(DIRS["labels"], source, os.path.splitext(os.path.basename(video))[0] + ".json")
    if os.path.isfile(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    caps = captions_for(video)
    if caps:
        cues = _parse_cues(caps)
        if cues:
            data = {"language": language, "how": "captions", "segments": [_spread_words(c) for c in cues]}
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            return out
    data = transcribe(video, language, device)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return out


def label_source(name: str, language: str, device: str = "auto") -> int:
    folder = os.path.join(DIRS["raw"], name)
    if not os.path.isdir(folder):
        return 0
    n = 0
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".mp4"):
            label_video(os.path.join(folder, fn), language, device)
            n += 1
    say(f"{name}: {n} videos labelled")
    return n
