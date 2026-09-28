"""What was said, from the sound: speech recognition with faster-whisper,
used only to label the sentences a reader chose to keep (the page sends
the sound of an utterance along with the frames, when its microphone
switch is on) and to show the reader how the lips compared. The sound is
transcribed in memory and dropped; only the text is kept.

Models, loaded on first use per language: ivrit.ai's Hebrew-tuned Whisper
for Hebrew (Apache-2.0), OpenAI's multilingual Whisper (MIT) otherwise —
`--whisper` picks its size (small, medium, large-v3-turbo; medium by
default). On a CPU an utterance takes a few seconds: Whisper always hears
30 seconds, however short the clip. The models run in hear_worker.py, a
process of their own: on Windows, loading CTranslate2 next to PyTorch
ended the server with an access violation.
"""
from __future__ import annotations

import io
import json
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
    """Speech recognition for the server. In the server's process it only
    talks to hear_worker.py (PyTorch and CTranslate2 do not share a process
    well on Windows); `in_process=True` is what the worker itself uses, and
    `fake=True` answers without any model."""

    def __init__(self, size: str = "medium", device: str = "cpu", fake: bool = False, threads: Optional[int] = None, in_process: bool = False,
                 fake_worker: bool = False):
        self.size, self.device, self.fake, self.threads, self.in_process = size, device, fake, threads, in_process
        self.fake_worker = fake_worker   # tests: a worker that answers without a model
        self.models: Dict[str, object] = {}
        self.lock = threading.Lock()
        self.proc = None
        self.error: Optional[str] = None
        if fake or in_process:
            self.available = fake or self._importable()
        else:   # the server: what counts is whether the worker starts, and it says why when it does not
            try:
                self._start()
                self.available = True
            except Exception as e:  # noqa: BLE001
                self.error = str(e)
                self.available = False

    # ---- the worker ------------------------------------------------------------------
    def _start(self) -> None:
        import os
        import subprocess
        import sys

        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hear_worker.py"),
               "--size", self.size, "--device", self.device]
        if self.threads:
            cmd += ["--threads", str(self.threads)]
        if self.fake_worker:
            cmd.append("--fake")
        # CTranslate2 picks AVX-512 code paths on CPUs that have them; on the
        # owner's AMD Zen 5 that ended the worker with an access violation.
        # AVX2 is as fast as it matters here and runs everywhere.
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                   CT2_FORCE_CPU_ISA=os.environ.get("THELIP_CT2_ISA", "AVX2"))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        line = self.proc.stdout.readline()
        try:
            info = json.loads(line.decode("utf-8")) if line else {}
        except ValueError:
            info = {}
        if not info.get("ready"):
            self.stop()
            raise RuntimeError(f"the speech worker did not start: {info.get('error') or line[:300]!r}")

    def _ask(self, wav: bytes, language: str) -> dict:
        for attempt in (1, 2):
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            try:
                self.proc.stdin.write(json.dumps({"language": language, "bytes": len(wav)}).encode("utf-8") + b"\n" + wav)
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if line:
                    reply = json.loads(line.decode("utf-8"))
                    if "error" in reply:
                        raise ValueError(reply["error"])
                    return reply
            except OSError:
                pass
            code = self.proc.poll() if self.proc else None
            why = f"exited with {code & 0xFFFFFFFF:#010x}" if code not in (None, 0) else "stopped answering"
            if attempt == 2:
                print(f"the speech worker {why} again; giving up on this utterance", flush=True)
                break
            print(f"the speech worker {why}; starting it again", flush=True)
            self.stop()
        raise RuntimeError(f"the speech worker gave no answer ({why})")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b'{"bytes": 0}\n')
                self.proc.stdin.flush()
                self.proc.wait(5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        self.proc = None

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
        if self.fake:
            audio, seconds = wav_to_float(wav)
            return {"heard": f"FAKE HEARD {seconds:.1f}s" if seconds >= 0.2 else "", "confidence": 0.9 if seconds >= 0.2 else 0.0,
                    "seconds": round(seconds, 2), "model": "fake"}
        if not self.in_process:
            wav_to_float(wav)   # a bad file is refused here, before it reaches the worker
            with self.lock:
                return self._ask(wav, language)
        audio, seconds = wav_to_float(wav)
        if seconds < 0.2 or float(np.abs(audio).max()) < 1e-4:
            return {"heard": "", "confidence": 0.0, "seconds": round(seconds, 2), "model": self.model_name(language)}
        with self.lock:
            model = self._model(language)
            # No voice-activity filter: it imports onnxruntime, which ended the
            # worker with an access violation next to CTranslate2 on Windows;
            # the page sends the stretch of one utterance anyway.
            segments, info = model.transcribe(audio, language=language, beam_size=5, vad_filter=False,
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
