"""thelip-server: open-vocabulary English lip reading for thelip.syrup.

The page at https://thelip.ai/ reads 51 words on its own. Point it at this
server and it reads any English words: the server runs the LRS3 visual
speech recogniser of Ma, Petridis & Pantic (Auto-AVSR; 19.1% word error
rate on LRS3) through the Chaplin pipeline (Amanvir Parhar, MIT), which
carries the Imperial College preprocessing (Apache-2.0): a mediapipe face
detector, alignment to a mean face, a 96x96 mouth crop, a Conformer
encoder, a transformer decoder with a subword language model.

    python server.py [--port 8791] [--beam 20] [--no-lm] [--token secret]
    python server.py --fake            # no model: answers with a fixed text (tests)

Routes
    GET  /health            -> {"ok": true, "model": ..., "device": ..., "fake": bool}
    POST /read              multipart: fps=<number>, frames=<jpeg>... in order
                            -> {"text": "HELLO THERE", "frames": 62, "seconds": 3.1}
                            422 when no face is found in the frames

English only, no audio, one speaker: the largest face in the frames. Text
comes back in the model's upper-case tokens. Nothing is stored: frames are
decoded in memory and dropped when the answer is sent.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from typing import List, Optional

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

HERE = os.path.dirname(os.path.abspath(__file__))
CHAPLIN = os.environ.get("THELIP_CHAPLIN", os.path.join(HERE, "chaplin"))
CONFIG = "configs/LRS3_V_WER19.1.ini"
MODEL_NAME = "LRS3_V_WER19.1 (Auto-AVSR, Ma et al. 2023) via Chaplin"
MAX_FRAMES = 400


class Reader:
    """The model behind /read. `fake=True` needs no torch and no weights."""

    def __init__(self, fake: bool = False, beam: int = 20, lm: bool = True, threads: Optional[int] = None):
        self.fake = fake
        self.lock = threading.Lock()
        self.device = "none"
        self.beam = beam
        self.lm = lm
        if fake:
            return
        import torch  # noqa: WPS433 (loaded here so --fake works without it)

        # ESPnet checkpoints are plain state dicts; newer torch defaults to
        # weights_only=True, which some of Chaplin's loads do not pass.
        _load = torch.load

        def load(*a, **k):
            k.setdefault("weights_only", False)
            return _load(*a, **k)

        torch.load = load
        torch.set_num_threads(threads or max(1, os.cpu_count() or 1))
        if not os.path.isdir(os.path.join(CHAPLIN, "pipelines")):
            raise SystemExit(f"Chaplin is not at {CHAPLIN}; run.py downloads it")
        os.chdir(CHAPLIN)
        sys.path.insert(0, CHAPLIN)
        from configparser import ConfigParser

        cfg = ConfigParser()
        cfg.read(CONFIG)
        for key in ("model_path", "model_conf", "rnnlm", "rnnlm_conf"):
            path = cfg.get("model", key)
            if not os.path.isfile(path):
                raise SystemExit(f"missing {path} under {CHAPLIN}; run.py downloads the model files")
        from pipelines.model import AVSR  # noqa: E402
        from pipelines.data.data_module import AVSRDataLoader  # noqa: E402
        from pipelines.detectors.mediapipe.detector import LandmarksDetector  # noqa: E402

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.loader = AVSRDataLoader("video", speed_rate=1, detector="mediapipe")
        self.detector = LandmarksDetector()
        self.model = AVSR(
            "video", cfg.get("model", "model_path"), cfg.get("model", "model_conf"),
            cfg.get("model", "rnnlm") if lm else None, cfg.get("model", "rnnlm_conf") if lm else None,
            penalty=cfg.getfloat("decode", "penalty"), ctc_weight=cfg.getfloat("decode", "ctc_weight"),
            lm_weight=cfg.getfloat("decode", "lm_weight") if lm else 0.0, beam_size=beam, device=self.device,
        )
        self.torch = torch

    def read(self, video: np.ndarray) -> str:
        """video: (T, H, W, 3) RGB uint8 at 25 fps -> the model's transcript."""
        if self.fake:
            return f"FAKE READING OF {len(video)} FRAMES"
        with self.lock:
            landmarks = self.detector.detect(video, self.detector.full_range_detector)
            if all(l is None for l in landmarks):
                landmarks = self.detector.detect(video, self.detector.short_range_detector)
            if all(l is None for l in landmarks):
                raise NoFace()
            crops = self.loader.video_process(video, landmarks)
            data = self.loader.video_transform(self.torch.tensor(crops))
            return self.model.infer(data)


class NoFace(Exception):
    pass


def decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def create_app(reader: Reader, token: Optional[str] = None) -> Starlette:
    async def health(request: Request):
        return JSONResponse({"ok": True, "model": MODEL_NAME, "device": reader.device, "fake": reader.fake,
                             "beam": reader.beam, "lm": reader.lm, "max_frames": MAX_FRAMES})

    async def read(request: Request):
        if token and request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"error": "bad token"}, status_code=401)
        form = await request.form()
        files = [v for k, v in form.multi_items() if k == "frames"]
        if not files:
            return JSONResponse({"error": "no frames"}, status_code=400)
        if len(files) > MAX_FRAMES:
            return JSONResponse({"error": f"at most {MAX_FRAMES} frames"}, status_code=413)
        try:
            fps = float(form.get("fps", "25"))
        except ValueError:
            fps = 25.0
        started = time.time()
        frames: List[np.ndarray] = []
        for f in files:
            frames.append(decode_jpeg(await f.read()))
        if any(fr.shape != frames[0].shape for fr in frames):
            return JSONResponse({"error": "frames differ in size"}, status_code=400)
        video = np.stack(frames)
        if abs(fps - 25.0) > 1.0:
            # The model sees 25 fps; resample by nearest frame.
            idx = np.clip(np.round(np.arange(0, len(video) * 25.0 / fps) * fps / 25.0).astype(int), 0, len(video) - 1)
            video = video[idx]
        try:
            text = await _run(reader.read, video)
        except NoFace:
            return JSONResponse({"error": "no face found in the frames"}, status_code=422)
        return JSONResponse({"text": text, "frames": int(len(video)), "seconds": round(time.time() - started, 2)})

    return Starlette(
        routes=[Route("/health", health), Route("/read", read, methods=["POST"])],
        middleware=[Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])],
    )


async def _run(fn, *args):
    import asyncio

    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--beam", type=int, default=20, help="beam width (40 in Chaplin; smaller is faster)")
    ap.add_argument("--no-lm", action="store_true", help="decode without the subword language model")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--token", default=os.environ.get("THELIP_TOKEN") or None)
    ap.add_argument("--fake", action="store_true", help="no model; answer with a fixed text")
    args = ap.parse_args(argv)
    import uvicorn

    reader = Reader(fake=args.fake, beam=args.beam, lm=not args.no_lm, threads=args.threads)
    print(f"thelip-server: {MODEL_NAME if not args.fake else 'fake'} on {reader.device}; http://{args.host}:{args.port}")
    uvicorn.run(create_app(reader, args.token), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
