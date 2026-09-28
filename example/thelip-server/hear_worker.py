"""The process speech recognition runs in. faster-whisper (CTranslate2,
onnxruntime for its voice detector) and PyTorch each bring their own
native libraries, and on Windows the combination in one process ended in
an access violation, so server.py starts this worker and talks to it over
pipes:

    request   one JSON line {"language": "en", "bytes": N}, then N bytes of WAV
    answer    one JSON line {"heard": ..., "confidence": ..., "seconds": ..., "model": ...} or {"error": ...}
    bytes = 0 stops

The first line the worker writes is {"ready": true}. Models load on first
use per language and stay loaded.

    python hear_worker.py --size medium [--device cpu] [--threads N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--fake", action="store_true", help="no model; a fixed answer (tests)")
    args = ap.parse_args()

    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    inp = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    sys.stdout = sys.stderr

    def answer(obj: dict) -> None:
        out.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

    def read_exact(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = inp.read(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return bytes(buf)

    def read_line() -> bytes:
        buf = bytearray()
        while True:
            ch = inp.read(1)
            if not ch:
                raise EOFError
            if ch == b"\n":
                return bytes(buf)
            buf += ch

    try:
        from hear import Transcriber
    except Exception as e:  # noqa: BLE001
        answer({"error": f"{type(e).__name__}: {e}"})
        return
    transcriber = Transcriber(args.size, args.device, fake=args.fake, threads=args.threads, in_process=True)
    if not transcriber.available:
        answer({"error": "faster-whisper is not installed"})
        return
    answer({"ready": True})
    while True:
        try:
            header = json.loads(read_line().decode("utf-8"))
            n = int(header.get("bytes", 0))
            if n <= 0:
                return
            wav = read_exact(n)
        except (EOFError, ValueError):
            return
        try:
            answer(transcriber.hear(wav, str(header.get("language", "en"))))
        except Exception as e:  # noqa: BLE001
            answer({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
