"""thelip-server: open-vocabulary lip reading for thelip.syrup.

The page at https://thelip.ai/ reads 51 English words on its own. Point it
at this server and it reads any words, in the languages of languages.py:
the server runs the visual speech recognisers of Ma, Petridis & Pantic
(Auto-AVSR; 19.1% word error rate on LRS3 for English) through the Chaplin
pipeline (Amanvir Parhar, MIT), which carries the Imperial College
preprocessing (Apache-2.0): a mediapipe face detector, alignment to a mean
face, a 96x96 mouth crop, a Conformer encoder, a transformer decoder with a
subword language model. Models that example/thelip-train produced (Hebrew,
or a better English) are served the same way, each in a worker process.

    python server.py [--port 8791] [--beam 20] [--no-lm] [--token secret]
    python server.py --fake            # no model: answers with a fixed text (tests)

Routes
    GET  /health            -> {"ok": true, "version": N, "model": ..., "device": ..., "fake": bool, "languages": [...]}
    POST /read              multipart: fps=<number>, frames=<jpeg>... in order,
                            language=<code> (en default; see languages.py),
                            improve=1 to keep the mouth crops for training
                            -> {"text": "HELLO THERE", "frames": 62, "seconds": 3.1, "id": ...}
                            422 when no face is found in the frames; 400 for an unknown
                            language; 503 for a language whose model is not on this server
    POST /feedback          json: {"id": ..., "raw": ..., "corrected": ...}
                            -> {"ok": true}; the correction joins the kept crops
    POST /hear              multipart: audio=<wav>, language=<code>, utt=<the read's utt>, improve=1
                            -> {"heard": "hello there", "confidence": 0.8, "seconds": 2.1, "id": ...}
                            what the microphone heard (faster-whisper), written to the sample the
                            read with the same `utt` kept; the sound itself is dropped

One speaker: the largest face in the frames. Languages are the models in
languages.py, loaded on first use; text comes back as the model writes it
(upper case for the English one). Nothing is stored unless the request
says improve=1: then the 96x96 grey mouth crops the model saw (not the
frames, not the face) and the texts are written under data/<id>/, the
material for training a better model on real phones and real speakers.
With the page's microphone switch on, the sound of the utterance comes
too (/hear): speech recognition writes down what was said, so that the
kept sample is labelled without anyone typing; the sound is not kept.
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
AUTO_AVSR = os.environ.get("THELIP_AUTO_AVSR") or os.path.join(WORK, "auto_avsr")
MODELS = os.environ.get("THELIP_MODELS") or os.path.join(WORK, "models")   # exports of example/thelip-train
sys.path.insert(0, HERE)
from hear import HEBREW, Transcriber  # noqa: E402
from languages import LANGUAGES, ORDER, describe, runnable, trained_models  # noqa: E402
from version import SERVER_VERSION  # noqa: E402
FACE_MODEL = os.environ.get("THELIP_FACE_MODEL") or os.path.join(WORK, "blaze_face_short_range.tflite")
DATA = os.environ.get("THELIP_DATA") or os.path.join(WORK, "data")
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


class TrainedModel:
    """A model example/thelip-train exported (Auto-AVSR's network, the
    language's own tokenizer). It runs in its own process, trained_worker.py,
    because Auto-AVSR's code and Chaplin's each carry an `espnet` package of
    their own and the two cannot be imported side by side; this class sends
    it the 96x96 crops and reads the text back. A worker that dies is
    started once more."""

    def __init__(self, folder: str, device: str, beam: int = 20, fake: bool = False):
        self.folder, self.device, self.beam, self.fake = folder, device, beam, fake
        self.proc = None
        self.info: dict = {}
        if not fake and not os.path.isdir(os.path.join(AUTO_AVSR, "espnet")):
            raise SystemExit(f"auto_avsr is needed for trained models; expected at {AUTO_AVSR} (run.py fetches it)")
        self._start()

    def _start(self) -> None:
        import json
        import subprocess

        cmd = [sys.executable, os.path.join(HERE, "trained_worker.py")]
        cmd += ["--fake"] if self.fake else [self.folder, "--auto-avsr", AUTO_AVSR, "--device", self.device, "--beam", str(self.beam)]
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        line = self.proc.stdout.readline()
        try:
            self.info = json.loads(line.decode("utf-8")) if line else {}
        except ValueError:
            self.info = {}
        if not self.info.get("ready"):
            self.stop()
            raise SystemExit(f"the worker for {self.folder} did not start: {self.info.get('error') or line[:200]!r}")

    def infer(self, crops: np.ndarray) -> str:
        import json
        import struct

        crops = np.ascontiguousarray(crops, dtype=np.uint8)
        if crops.ndim != 3 or crops.shape[1:] != (96, 96):
            raise ValueError(f"crops must be T x 96 x 96, got {crops.shape}")
        for attempt in (1, 2):
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            try:
                self.proc.stdin.write(struct.pack("<I", len(crops)) + crops.tobytes())
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if line:
                    reply = json.loads(line.decode("utf-8"))
                    if "error" in reply:
                        raise RuntimeError(reply["error"])
                    return reply["text"]
            except (OSError, ValueError):
                pass
            if attempt == 2:
                break
            print(f"the worker for {self.folder} stopped answering; starting it again", flush=True)
            self.stop()
        raise RuntimeError(f"the worker for {self.folder} gave no answer")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"\0\0\0\0")
                self.proc.stdin.flush()
                self.proc.wait(5)
            except (OSError, ValueError):
                pass
            except Exception:  # noqa: BLE001  (TimeoutExpired)
                self.proc.kill()
        self.proc = None


class Reader:
    """The models behind /read, one per language, loaded on first use.
    `fake=True` needs no torch and no weights."""

    def __init__(self, fake: bool = False, beam: int = 20, lm: bool = True, threads: Optional[int] = None):
        self.fake = fake
        self.lock = threading.Lock()
        self.device = "none"
        self.beam = beam
        self.lm = lm
        self.models: dict = {}
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
        from pipelines.data.data_module import AVSRDataLoader  # noqa: E402

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.loader = AVSRDataLoader("video", speed_rate=1, detector="mediapipe")
        self.detector = FaceKeypoints(FACE_MODEL)
        self.torch = torch
        self.model_for("en")  # the default language, ready before the first request

    def available(self, code: str) -> bool:
        return (runnable(code) and describe(code, CHAPLIN)["available"]) or code in trained_models(MODELS)

    def model_for(self, code: str):
        """The model of a language, loaded the first time it is asked for: the one
        example/thelip-train exported when that is the better one (or the only
        one, as for Hebrew), else the published model of languages.py."""
        if code in self.models:
            return self.models[code]
        trained = trained_models(MODELS).get(code)
        if trained and (not runnable(code) or trained.get("preferred")):
            print(f"loading the trained {LANGUAGES[code]['name']} model {trained['name']} ({trained.get('summary', '')})", flush=True)
            self.models[code] = TrainedModel(trained["folder"], self.device, beam=min(self.beam, 20))
            return self.models[code]
        from pipelines.model import AVSR  # noqa: E402

        spec = LANGUAGES[code]
        base = os.path.join(CHAPLIN, "benchmarks")
        model_dir, lm_dir = os.path.join(base, spec["model"]), os.path.join(base, spec["lm"])
        if not os.path.isfile(os.path.join(model_dir, "model.pth")):
            raise SystemExit(f"missing {model_dir}/model.pth; run.py downloads the model files")
        use_lm = self.lm and os.path.isfile(os.path.join(lm_dir, "model.pth"))
        beam = self.beam if code == "en" else min(self.beam, spec["beam"])
        print(f"loading {spec['name']} ({spec['quality']})", flush=True)
        self.models[code] = AVSR(
            "video", os.path.join(model_dir, "model.pth"), os.path.join(model_dir, "model.json"),
            os.path.join(lm_dir, "model.pth") if use_lm else None, os.path.join(lm_dir, "model.json") if use_lm else None,
            penalty=spec["penalty"], ctc_weight=spec["ctc_weight"], lm_weight=spec["lm_weight"] if use_lm else 0.0,
            beam_size=beam, device=self.device,
        )
        return self.models[code]

    def read(self, video: np.ndarray, language: str = "en"):
        """video: (T, H, W, 3) RGB uint8 at 25 fps -> (transcript, mouth crops
        (T, 96, 96) uint8 grey — what the model saw)."""
        if self.fake:
            return f"FAKE {language.upper()} READING OF {len(video)} FRAMES", np.zeros((len(video), 96, 96), np.uint8)
        with self.lock:
            model = self.model_for(language)
            landmarks = self.detector.detect(video)
            if all(l is None for l in landmarks):
                raise NoFace()
            crops = np.asarray(self.loader.video_process(video, landmarks), dtype=np.uint8)
            if isinstance(model, TrainedModel):
                return model.infer(crops), crops
            data = self.loader.video_transform(self.torch.tensor(crops))
            return model.infer(data), crops


class NoFace(Exception):
    pass


# A read and the sound of the same utterance arrive as two requests, in
# either order, sharing the page's `utt` id: this links them for an hour.
LINKS: dict = {}
LINKS_LOCK = threading.Lock()


def link(utt: str, **fields) -> dict:
    """Records fields under the utterance id; returns the entry (id, heard...)."""
    now = time.time()
    with LINKS_LOCK:
        for key in [k for k, v in LINKS.items() if now - v["t"] > 3600]:
            del LINKS[key]
        entry = LINKS.setdefault(utt, {"t": now})
        entry.update(fields)
        return dict(entry)


def valid_utt(utt: str) -> bool:
    import re

    return bool(re.fullmatch(r"[0-9a-f]{8,32}", utt or ""))


def keep_sample(crops: np.ndarray, meta: dict) -> str:
    """Write one training sample: the crops and what was read. Returns its id."""
    import json
    import uuid

    sample_id = uuid.uuid4().hex
    folder = os.path.join(DATA, sample_id)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "crops.npy"), crops)
    utt = meta.get("utt")
    if utt and valid_utt(utt):
        entry = link(utt, id=sample_id)
        if entry.get("heard") is not None:   # the sound was heard before the frames were read
            meta = dict(meta, heard=entry["heard"], heard_confidence=entry.get("heard_confidence"), heard_model=entry.get("heard_model"))
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


def create_app(reader: Reader, token: Optional[str] = None, transcriber: Optional[Transcriber] = None) -> Starlette:
    def languages():
        out = []
        trained = trained_models(MODELS)
        for code in ORDER:
            entry = describe(code, None if reader.fake else CHAPLIN)
            if code in trained:
                t = trained[code]
                entry.update(available=True, trained=t["name"], quality=t.get("summary") or entry.get("quality"), status=None)
                entry.pop("status", None)
            entry["loaded"] = code in reader.models
            out.append(entry)
        return out

    async def health(request: Request):
        hears = bool(transcriber and transcriber.available)
        return JSONResponse({"ok": True, "version": SERVER_VERSION, "model": MODEL_NAME, "device": reader.device, "fake": reader.fake,
                             "beam": reader.beam, "lm": reader.lm, "max_frames": MAX_FRAMES,
                             "languages": languages(), "keeps": "mouth crops and texts, only when asked (improve=1)",
                             "hears": hears, "whisper": {"default": transcriber.size, "he": HEBREW} if hears else None})

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
        language = str(form.get("language", "en")).lower()[:8]
        if language not in LANGUAGES:
            return JSONResponse({"error": f"unknown language {language!r}"}, status_code=400)
        spec = LANGUAGES[language]
        has_model = runnable(language) or language in trained_models(MODELS)
        if not has_model:
            return JSONResponse({"error": f"no model for {spec['name']}: {spec['status']}"}, status_code=503)
        if not reader.fake and not reader.available(language):
            return JSONResponse({"error": f"the {spec['name']} model is not downloaded on this server"}, status_code=503)
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
            text, crops = await _run(reader.read, video, language)
        except NoFace:
            return JSONResponse({"error": "no face found in the frames"}, status_code=422)
        except RuntimeError as e:  # the model's process failed on this clip
            return JSONResponse({"error": f"the {spec['name']} model failed: {e}"}, status_code=500)
        sample_id = None
        utt = str(form.get("utt", ""))[:32]
        if str(form.get("improve", "")) == "1":
            sample_id = keep_sample(crops, {"raw": text, "language": language, "fps": 25, "frames": int(len(video)),
                                            "source": str(form.get("source", ""))[:40], "utt": utt if valid_utt(utt) else None})
        return JSONResponse({"text": text, "language": language, "frames": int(len(video)),
                             "seconds": round(time.time() - started, 2), "id": sample_id})

    async def hear(request: Request):
        if token and request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"error": "bad token"}, status_code=401)
        if not (transcriber and transcriber.available):
            return JSONResponse({"error": "this server does not hear (faster-whisper is not installed, or --whisper none)"}, status_code=503)
        form = await request.form()
        audio = form.get("audio")
        if audio is None or not hasattr(audio, "read"):
            return JSONResponse({"error": "no audio"}, status_code=400)
        data = await audio.read()
        if len(data) > 4_000_000:
            return JSONResponse({"error": "at most 4 MB of audio"}, status_code=413)
        language = str(form.get("language", "en")).lower()[:8]
        if language not in LANGUAGES:
            return JSONResponse({"error": f"unknown language {language!r}"}, status_code=400)
        utt = str(form.get("utt", ""))[:32]
        started = time.time()
        try:
            result = await _run(transcriber.hear, data, language)
        except Exception as e:  # noqa: BLE001  (a bad wav, a model that failed to load)
            return JSONResponse({"error": f"could not hear: {type(e).__name__}: {e}"}, status_code=400)
        sample_id = None
        if str(form.get("improve", "")) == "1" and valid_utt(utt):
            entry = link(utt, heard=result["heard"], heard_confidence=result["confidence"], heard_model=result["model"])
            sample_id = entry.get("id")
            if sample_id:
                update_sample(sample_id, {"heard": result["heard"], "heard_confidence": result["confidence"], "heard_model": result["model"]})
        return JSONResponse(dict(result, id=sample_id, language=language, took=round(time.time() - started, 2)))

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
        routes=[Route("/health", health), Route("/read", read, methods=["POST"]), Route("/feedback", feedback, methods=["POST"]),
                Route("/hear", hear, methods=["POST"])],
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
    ap.add_argument("--whisper", default="medium", help="speech recognition model for /hear: small, medium, large-v3-turbo, or none")
    args = ap.parse_args(argv)
    import uvicorn

    reader = Reader(fake=args.fake, beam=args.beam, lm=not args.no_lm, threads=args.threads)
    transcriber = None if args.whisper == "none" else Transcriber(args.whisper, reader.device, fake=args.fake, threads=args.threads)
    if transcriber and not transcriber.available:
        print("faster-whisper is not installed: /hear is off (pip install faster-whisper)", flush=True)
    print(f"thelip-server: {MODEL_NAME if not args.fake else 'fake'} on {reader.device}; http://{args.host}:{args.port}")
    uvicorn.run(create_app(reader, args.token, transcriber), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
