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

from .common import DIRS, ffmpeg_exe, say

WHISPER = {"he": "ivrit-ai/whisper-large-v3-turbo-ct2", "*": "large-v3"}
# The same models in Hugging Face's format, for the PyTorch backend: on a Windows PC
# with an NVIDIA card, CUDA PyTorch runs Whisper without CTranslate2's CUDA libraries.
HF_WHISPER = {"he": "ivrit-ai/whisper-large-v3-turbo", "*": "openai/whisper-large-v3-turbo"}
# Only the start of a long video is transcribed: segment keeps at most 20 minutes of
# clips per video, and a transcript of the rest would never be used.
MAX_LABEL_SECONDS = int(os.environ.get("THELIP_MAX_LABEL_SECONDS", "1800"))


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


def backend() -> str:
    """"torch" when PyTorch sees a CUDA card and transformers is installed (the Windows PC),
    else "ctranslate2" (faster-whisper; a Linux GPU box, or a CPU)."""
    choice = os.environ.get("THELIP_WHISPER_BACKEND")
    if choice:
        return choice
    try:
        import torch
        import transformers  # noqa: F401

        return "torch" if torch.cuda.is_available() else "ctranslate2"
    except Exception:  # noqa: BLE001
        return "ctranslate2"


def decode_audio(video: str, seconds: int = MAX_LABEL_SECONDS):
    """The sound of a video's first `seconds`, mono float32 at 16 kHz, through ffmpeg."""
    import subprocess

    import numpy as np

    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", video, "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


_hf: dict = {}


def hf_pipeline(language: str):
    name = HF_WHISPER.get(language, HF_WHISPER["*"])
    if name not in _hf:
        import torch
        from transformers import pipeline

        say(f"loading Whisper {name} (PyTorch, CUDA)")
        try:
            _hf[name] = pipeline("automatic-speech-recognition", model=name, torch_dtype=torch.float16, device="cuda:0")
        except Exception as e:  # noqa: BLE001
            if name == HF_WHISPER["*"]:
                raise
            say(f"  {name} unavailable ({e}); using {HF_WHISPER['*']} with language={language}")
            name = HF_WHISPER["*"]
            _hf[name] = _hf.get(name) or pipeline("automatic-speech-recognition", model=name, torch_dtype=torch.float16, device="cuda:0")
    return name, _hf[name]


def transcribe_torch(video: str, language: str, whole: bool = False) -> dict:
    """Whisper through transformers on CUDA, with word times (cross-attention alignment);
    the words grouped into segments at pauses over a second."""
    name, pipe = hf_pipeline(language)
    audio = decode_audio(video, 24 * 3600 if whole else MAX_LABEL_SECONDS)
    if audio.size < 16000:
        return {"language": language, "how": f"whisper:{name}", "segments": []}
    out = pipe({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30, batch_size=8, return_timestamps="word",
               generate_kwargs={"language": language, "task": "transcribe"})
    words = []
    for c in out.get("chunks") or []:
        text = (c.get("text") or "").strip()
        s, e = (c.get("timestamp") or (None, None))
        if not text or s is None:
            continue
        e = e if e is not None and e > s else s + 0.3
        words.append({"w": text, "s": float(s), "e": float(e), "p": 1.0})
    segments, cur = [], []
    for w in words:
        if cur and w["s"] - cur[-1]["e"] > 1.0:
            segments.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segments.append(cur)
    return {"language": language, "how": f"whisper:{name}",
            "segments": [{"start": g[0]["s"], "end": g[-1]["e"], "text": " ".join(w["w"] for w in g), "words": g} for g in segments]}


def transcribe(video: str, language: str, device: str = "auto", whole: bool = False) -> dict:
    if backend() == "torch":
        return transcribe_torch(video, language, whole)
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
    info_path = os.path.splitext(video)[0] + ".info.json"
    local = False
    if os.path.isfile(info_path):
        with open(info_path, encoding="utf-8") as f:
            local = bool(json.load(f).get("local"))
    data = transcribe(video, language, device, whole=local)   # the owner's own recordings: all of it
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
