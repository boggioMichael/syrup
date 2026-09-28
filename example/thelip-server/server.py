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
    POST /read              multipart: fps=<number>, frames=<jpeg>... in order,
                            improve=1 to keep the mouth crops for training
                            -> {"text": "HELLO THERE", "frames": 62, "seconds": 3.1, "id": ...}
                            422 when no face is found in the frames
    POST /feedback          json: {"id": ..., "raw": ..., "corrected": ...}
                            -> {"ok": true}; the correction joins the kept crops

English only, no audio, one speaker: the largest face in the frames. Text
comes back in the model's upper-case tokens. Nothing is stored unless the
request says improve=1: then the 96x96 grey mouth crops the model saw (not
the frames, not the face) and the texts are written under data/<id>/, the
material for training a better model on real phones and real speakers.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from typing import List, Optional

# torch before anything else that loads native DLLs: on Windows, torch 2.9+
# fails to initialise (WinError 1114) when imported after some of them
# (pytorch/pytorch#166628). --fake runs without torch.
if "--fake" not in sys.argv:
    try:
        import torch  # noqa: F401
    except OSError as _err:  # the DLL error, explained before giving up
        from diagnose import explain_dll_failure

        explain_dll_failure(_err)
        raise

import numpy as np
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

HERE = os.path.dirname(os.path.abspath(__file__))
# Downloads live under THELIP_HOME (run.py sets it): on Windows an ASCII path,
# because PyTorch's DLLs fail to initialise from a path with non-ASCII characters.
WORK = os.environ.get("THELIP_HOME") or HERE
CHAPLIN = os.environ.get("THELIP_CHAPLIN") or os.path.join(WORK, "chaplin")
FACE_MODEL = os.environ.get("THELIP_FACE_MODEL") or os.path.join(WORK, "blaze_face_short_range.tflite")
DATA = os.environ.get("THELIP_DATA") or os.path.join(WORK, "data")
CONFIG = "configs/LRS3_V_WER19.1.ini"
MODEL_NAME = "LRS3_V_WER19.1 (Auto-AVSR, Ma et al. 2023) via Chaplin"
MAX_FRAMES = 400


class FaceKeypoints:
    """Four points per frame — right eye, left eye, nose tip, mouth centre —
    from BlazeFace through mediapipe's Tasks API: the same keypoints, in the
    same order, that Chaplin's detector took from the legacy Solutions API,
    which mediapipe 0.10.3x no longer ships. The largest face wins; a frame
    with no face gives None, which Chaplin's VideoProcess interpolates."""

    def __init__(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not os.path.isfile(model_path):
            raise SystemExit(f"missing the face detector model {model_path}; run.py downloads it")
        self.mp = mp
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.5)
        self.detector = vision.FaceDetector.create_from_options(options)

    def detect(self, video: np.ndarray) -> list:
        out = []
        for frame in video:
            h, w = frame.shape[:2]
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = self.detector.detect(image)
            if not result.detections:
                out.append(None)
                continue
            best = max(result.detections, key=lambda d: d.bounding_box.width + d.bounding_box.height)
            kps = best.keypoints
            out.append(np.array([[int(kps[i].x * w), int(kps[i].y * h)] for i in range(4)]))
        return out


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

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.loader = AVSRDataLoader("video", speed_rate=1, detector="mediapipe")
        self.detector = FaceKeypoints(FACE_MODEL)
        self.model = AVSR(
            "video", cfg.get("model", "model_path"), cfg.get("model", "model_conf"),
            cfg.get("model", "rnnlm") if lm else None, cfg.get("model", "rnnlm_conf") if lm else None,
            penalty=cfg.getfloat("decode", "penalty"), ctc_weight=cfg.getfloat("decode", "ctc_weight"),
            lm_weight=cfg.getfloat("decode", "lm_weight") if lm else 0.0, beam_size=beam, device=self.device,
        )
        self.torch = torch

    def read(self, video: np.ndarray):
        """video: (T, H, W, 3) RGB uint8 at 25 fps -> (transcript, mouth crops
        (T, 96, 96) uint8 grey — what the model saw, or None when fake)."""
        if self.fake:
            return f"FAKE READING OF {len(video)} FRAMES", np.zeros((len(video), 96, 96), np.uint8)
        with self.lock:
            landmarks = self.detector.detect(video)
            if all(l is None for l in landmarks):
                raise NoFace()
            crops = self.loader.video_process(video, landmarks)
            data = self.loader.video_transform(self.torch.tensor(crops))
            return self.model.infer(data), np.asarray(crops, dtype=np.uint8)


class NoFace(Exception):
    pass


def keep_sample(crops: np.ndarray, meta: dict) -> str:
    """Write one training sample: the crops and what was read. Returns its id."""
    import json
    import uuid

    sample_id = uuid.uuid4().hex
    folder = os.path.join(DATA, sample_id)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "crops.npy"), crops)
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta, id=sample_id, kept=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), f, ensure_ascii=False, indent=1)
    return sample_id


def update_sample(sample_id: str, fields: dict) -> bool:
    import json
    import re

    if not re.fullmatch(r"[0-9a-f]{32}", sample_id or ""):
        return False
    path = os.path.join(DATA, sample_id, "meta.json")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    meta.update(fields)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return True


def decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def create_app(reader: Reader, token: Optional[str] = None) -> Starlette:
    async def health(request: Request):
        return JSONResponse({"ok": True, "model": MODEL_NAME, "device": reader.device, "fake": reader.fake,
                             "beam": reader.beam, "lm": reader.lm, "max_frames": MAX_FRAMES, "keeps": "mouth crops and texts, only when asked (improve=1)"})

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
            text, crops = await _run(reader.read, video)
        except NoFace:
            return JSONResponse({"error": "no face found in the frames"}, status_code=422)
        sample_id = None
        if str(form.get("improve", "")) == "1":
            sample_id = keep_sample(crops, {"raw": text, "fps": 25, "frames": int(len(video)),
                                            "source": str(form.get("source", ""))[:40]})
        return JSONResponse({"text": text, "frames": int(len(video)), "seconds": round(time.time() - started, 2), "id": sample_id})

    async def feedback(request: Request):
        if token and request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"error": "bad token"}, status_code=401)
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "json body expected"}, status_code=400)
        corrected = str(body.get("corrected", ""))[:500]
        raw = str(body.get("raw", ""))[:500]
        ok = update_sample(str(body.get("id", "")), {"corrected": corrected, "confirmed": corrected.strip().lower() == raw.strip().lower()})
        return JSONResponse({"ok": ok}, status_code=200 if ok else 404)

    return Starlette(
        routes=[Route("/health", health), Route("/read", read, methods=["POST"]), Route("/feedback", feedback, methods=["POST"])],
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
