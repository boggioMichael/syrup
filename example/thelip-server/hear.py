"""What was said, from the sound: speech recognition with faster-whisper,
used only to label the sentences a reader chose to keep (the page sends
the sound of an utterance along with the frames, when its microphone
switch is on) and to show the reader how the lips compared. The sound is
transcribed in memory and dropped; only the text is kept.

Models, loaded on first use per language: ivrit.ai's Hebrew-tuned Whisper
for Hebrew (Apache-2.0), OpenAI's multilingual Whisper (MIT) otherwise —
`--whisper` picks its size (small, medium, large-v3-turbo; medium by
default). On a CPU an utterance takes a few seconds: Whisper always hears
30 seconds, however short the clip.
"""
from __future__ import annotations

import io
import threading
import wave
from typing import Dict, Optional

import numpy as np

HEBREW = "ivrit-ai/whisper-large-v3-turbo-ct2"
RATE = 16000


def wav_to_float(data: bytes):
    """A WAV file's samples as float32 mono at 16 kHz (resampled by
    averaging when the file's rate differs) -> (samples, seconds)."""
    with wave.open(io.BytesIO(data)) as w:
        channels, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {width}")
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    seconds = len(x) / float(rate)
    if rate != RATE:
        idx = (np.arange(int(len(x) * RATE / rate) + 1) * rate / RATE).astype(np.int64)
        idx = idx[idx < len(x)]
        x = x[idx]
    return np.ascontiguousarray(x, dtype=np.float32), seconds


class Transcriber:
    def __init__(self, size: str = "medium", device: str = "cpu", fake: bool = False, threads: Optional[int] = None):
        self.size, self.device, self.fake, self.threads = size, device, fake, threads
        self.models: Dict[str, object] = {}
        self.lock = threading.Lock()
        self.available = fake or self._importable()

    @staticmethod
    def _importable() -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def model_name(self, language: str) -> str:
        return HEBREW if language == "he" else self.size

    def _model(self, language: str):
        name = self.model_name(language)
        if name in self.models:
            return self.models[name]
        from faster_whisper import WhisperModel

        print(f"loading speech recognition {name} for {language} (first use)", flush=True)
        kw = {"device": "cuda" if self.device.startswith("cuda") else "cpu",
              "compute_type": "float16" if self.device.startswith("cuda") else "int8"}
        if self.threads and kw["device"] == "cpu":
            kw["cpu_threads"] = self.threads
        self.models[name] = WhisperModel(name, **kw)
        return self.models[name]

    def hear(self, wav: bytes, language: str) -> dict:
        """-> {"heard": text, "confidence": 0..1, "seconds": float, "model": name}"""
        audio, seconds = wav_to_float(wav)
        if self.fake:
            return {"heard": f"FAKE HEARD {seconds:.1f}s" if seconds >= 0.2 else "", "confidence": 0.9 if seconds >= 0.2 else 0.0,
                    "seconds": round(seconds, 2), "model": "fake"}
        if seconds < 0.2 or float(np.abs(audio).max()) < 1e-4:
            return {"heard": "", "confidence": 0.0, "seconds": round(seconds, 2), "model": self.model_name(language)}
        with self.lock:
            model = self._model(language)
            segments, info = model.transcribe(audio, language=language, beam_size=5, vad_filter=True,
                                              condition_on_previous_text=False, without_timestamps=True)
            texts, probs, no_speech = [], [], []
            for s in segments:
                if s.text.strip():
                    texts.append(s.text.strip())
                    probs.append(float(np.exp(s.avg_logprob)))
                    no_speech.append(float(s.no_speech_prob))
        text = " ".join(texts)
        confidence = float(np.mean(probs)) * (1.0 - float(np.mean(no_speech))) if probs else 0.0
        return {"heard": text, "confidence": round(confidence, 3), "seconds": round(seconds, 2), "model": self.model_name(language)}
