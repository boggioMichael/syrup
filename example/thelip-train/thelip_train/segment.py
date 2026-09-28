"""From a labelled video to training clips: one stable, large-enough face,
the words said while it is on screen, the mouth cut exactly as the model
sees it (Chaplin/Auto-AVSR preprocessing: four points aligned to a mean
face, a 96x96 grey crop), 25 frames per second.

    clips/<source>/<id>/<n>.mp4     96x96, 25 fps
    clips/<source>/<id>/<n>.txt     the normalised text
    clips/<source>.jsonl            one row per clip: path, frames, text, language, licence, seconds

Face points come from mediapipe (BlazeFace, Tasks API) — right eye, left
eye, nose tip, mouth centre, the same four thelip-server uses — or, where
mediapipe is not installed, from syrup's own face and eye cascades with
the nose and mouth placed from the eyes (good enough to test the plumbing;
the real data is cut with mediapipe).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .common import DIRS, jsonl_append, normalise, say

FPS = 25
MIN_IOD = 32          # pixels between the eyes: smaller faces give crops too blurred to learn from
SAMPLE_FPS = 5        # face checks per second while scanning a video
MAX_MOVE = 0.35       # of the inter-ocular distance, between checks, for "the same face, still"
MIN_SECONDS, MAX_SECONDS, MAX_GAP = 0.8, 12.0, 0.6
MIN_WORDS = 2
PAD = 0.12
MAX_MINUTES_PER_VIDEO = 20


class Landmarker:
    """Four points (right eye, left eye, nose tip, mouth centre) of the largest face, or None."""

    def points(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class MediapipeLandmarker(Landmarker):
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

    def __init__(self, model_path: Optional[str] = None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = model_path or os.path.join(DIRS["tools"], "blaze_face_short_range.tflite")
        if not os.path.isfile(model_path):
            import urllib.request

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
        self.mp = mp
        self.detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path), running_mode=vision.RunningMode.IMAGE, min_detection_confidence=0.5))

    def points(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        h, w = rgb.shape[:2]
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not result.detections:
            return None
        best = max(result.detections, key=lambda d: d.bounding_box.width + d.bounding_box.height)
        k = best.keypoints
        return np.array([[k[i].x * w, k[i].y * h] for i in range(4)], dtype=np.float32)


class SyrupLandmarker(Landmarker):
    """syrup's cascades: the face box and the two eyes; nose and mouth from the eye line."""

    def __init__(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "python"))
        import syrup  # noqa: WPS433

        self.syrup = syrup

    def points(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        faces = self.syrup.find_face(rgb)
        if not faces or not faces.value:
            return None
        face = max(faces.value, key=lambda m: m.bounds[2] * m.bounds[3])
        x, y, w, h = face.bounds
        eyes = self.syrup.find_eyes(rgb, region=(x, y, w, int(h * 0.6)))
        pts = sorted((m.centre for m in (eyes.value or [])), key=lambda c: c[0])
        if len(pts) >= 2:
            left, right = pts[0], pts[-1]          # in image coordinates: left = viewer's left = subject's right eye
        else:
            left, right = (x + 0.3 * w, y + 0.38 * h), (x + 0.7 * w, y + 0.38 * h)
        mid = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        iod = max(1.0, right[0] - left[0])
        nose = (mid[0], mid[1] + 0.55 * iod)
        mouth = (mid[0], mid[1] + 1.15 * iod)
        return np.array([left, right, nose, mouth], dtype=np.float32)


HERE = os.path.dirname(os.path.abspath(__file__))


def make_landmarker() -> Landmarker:
    try:
        return MediapipeLandmarker()
    except Exception as e:  # noqa: BLE001
        say(f"mediapipe not usable ({e}); using syrup's cascades for the face points")
        return SyrupLandmarker()


def video_process():
    """Chaplin's VideoProcess (Imperial College preprocessing): alignment and the 96x96 crop."""
    chaplin = os.environ.get("THELIP_CHAPLIN") or os.path.join(DIRS["tools"], "chaplin")
    if not os.path.isdir(os.path.join(chaplin, "pipelines")):
        raise SystemExit(f"Chaplin is needed for the mouth crops; expected at {chaplin} (bootstrap.sh fetches it)")
    if chaplin not in sys.path:
        sys.path.insert(0, chaplin)
    from pipelines.detectors.mediapipe.video_process import VideoProcess  # noqa: E402

    return VideoProcess(convert_gray=True)


@dataclass
class FaceSample:
    t: float
    points: Optional[np.ndarray]

    @property
    def iod(self) -> float:
        return float(np.linalg.norm(self.points[0] - self.points[1])) if self.points is not None else 0.0


def scan_faces(path: str, landmarker: Landmarker) -> List[FaceSample]:
    """The largest face SAMPLE_FPS times a second through the whole video."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / SAMPLE_FPS)))
    samples, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                samples.append(FaceSample(i / fps, landmarker.points(rgb)))
        i += 1
    cap.release()
    return samples


def face_ok(samples: Sequence[FaceSample], start: float, end: float) -> bool:
    """A face big enough, present and still through [start, end]."""
    window = [s for s in samples if start - 1 / SAMPLE_FPS <= s.t <= end + 1 / SAMPLE_FPS]
    if not window:
        return False
    good = [s for s in window if s.points is not None and s.iod >= MIN_IOD]
    if len(good) < 0.95 * len(window):
        return False
    for a, b in zip(good, good[1:]):
        if np.linalg.norm(a.points[3] - b.points[3]) > MAX_MOVE * max(a.iod, b.iod) * max(1.0, (b.t - a.t) * SAMPLE_FPS):
            return False
    return True


def utterances(label: dict) -> List[dict]:
    """Words grouped into stretches of MIN..MAX seconds, split at long gaps and at sentence ends."""
    words = [w for seg in label["segments"] for w in seg.get("words", []) if w.get("w")]
    out, cur = [], []

    def flush():
        if len(cur) >= MIN_WORDS and cur[-1]["e"] - cur[0]["s"] >= MIN_SECONDS:
            out.append({"start": cur[0]["s"], "end": cur[-1]["e"], "text": " ".join(w["w"] for w in cur)})
        cur.clear()

    for w in words:
        if cur and (w["s"] - cur[-1]["e"] > MAX_GAP or w["e"] - cur[0]["s"] > MAX_SECONDS):
            flush()
        cur.append(w)
        if w["w"].rstrip()[-1:] in ".?!" and cur[-1]["e"] - cur[0]["s"] >= 1.5:
            flush()
    flush()
    return out


def frames_at_25fps(path: str, start: float, end: float) -> List[np.ndarray]:
    """RGB frames of [start, end] resampled to 25 fps by nearest source frame."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start * fps) - 1))
    wanted = [start + k / FPS for k in range(int((end - start) * FPS))]
    frames, k, last = [], 0, None
    while k < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        while k < len(wanted) and wanted[k] <= t + 0.5 / fps:
            frames.append(rgb if last is None or abs(t - wanted[k]) <= abs(last[0] - wanted[k]) else last[1])
            k += 1
        last = (t, rgb)
    cap.release()
    return frames


def write_clip(crops: np.ndarray, path: str) -> None:
    import cv2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (crops.shape[2], crops.shape[1]), True)
    for frame in crops:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()


def segment_video(video: str, label_path: str, info: dict, landmarker: Landmarker, processor=None) -> int:
    """Cuts one video into clips; returns how many. Skips a video already cut."""
    source = info["source"]
    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(DIRS["clips"], source, stem)
    manifest = os.path.join(DIRS["clips"], source + ".jsonl")
    if os.path.isfile(os.path.join(out_dir, "done")):
        return sum(1 for _ in open(os.path.join(out_dir, "done"), encoding="utf-8"))
    with open(label_path, encoding="utf-8") as f:
        label = json.load(f)
    language = info["language"]
    processor = processor or video_process()
    samples = scan_faces(video, landmarker)
    faced = sum(1 for s in samples if s.points is not None and s.iod >= MIN_IOD) / max(1, len(samples))
    say(f"  {stem}: face big enough in {faced * 100:.0f}% of the video ({len(samples)} checks)")
    kept, seconds, rows = 0, 0.0, []
    for n, u in enumerate(utterances(label)):
        if seconds > MAX_MINUTES_PER_VIDEO * 60:
            break
        start, end = max(0.0, u["start"] - PAD), u["end"] + PAD
        text = normalise(u["text"], language)
        if len(text.split()) < MIN_WORDS if language != "zh" else len(text) < 2:
            continue
        if not face_ok(samples, start, end):
            continue
        frames = frames_at_25fps(video, start, end)
        if len(frames) < int(MIN_SECONDS * FPS):
            continue
        landmarks = [landmarker.points(fr) for fr in frames]
        if sum(l is not None for l in landmarks) < 0.9 * len(landmarks):
            continue
        try:
            crops = processor(np.stack(frames), [None if l is None else l.astype(np.int64) for l in landmarks])
        except Exception as e:  # noqa: BLE001
            say(f"    clip {n}: crop failed ({e})")
            continue
        if crops is None or len(crops) != len(frames):
            continue
        clip = os.path.join(out_dir, f"{n:04d}.mp4")
        write_clip(np.asarray(crops, dtype=np.uint8), clip)
        with open(os.path.join(out_dir, f"{n:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        row = {"path": os.path.relpath(clip, DIRS["clips"]), "frames": len(frames), "text": text, "language": language,
               "source": source, "video": stem, "licence": info.get("licence"), "seconds": round(len(frames) / FPS, 2), "how": label.get("how")}
        jsonl_append(manifest, row)
        rows.append(row)
        kept += 1
        seconds += len(frames) / FPS
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "done"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r["path"] + "\n")
    say(f"  {stem}: {kept} clips, {seconds / 60:.1f} min")
    return kept


def segment_source(name: str, landmarker: Optional[Landmarker] = None) -> int:
    raw = os.path.join(DIRS["raw"], name)
    labels = os.path.join(DIRS["labels"], name)
    if not os.path.isdir(raw):
        return 0
    landmarker = landmarker or make_landmarker()
    processor = video_process()
    total = 0
    for fn in sorted(os.listdir(raw)):
        if not fn.endswith(".mp4"):
            continue
        stem = os.path.splitext(fn)[0]
        label_path = os.path.join(labels, stem + ".json")
        info_path = os.path.join(raw, stem + ".info.json")
        if not (os.path.isfile(label_path) and os.path.isfile(info_path)):
            continue
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        total += segment_video(os.path.join(raw, fn), label_path, info, landmarker, processor)
    say(f"{name}: {total} clips")
    return total


def phone_samples(data_dir: str, language: str) -> List[dict]:
    """thelip-server's kept samples (crops.npy + meta.json) as clips: the ones
    with a text that is what was said — typed by the person, heard by the
    server's speech recognition, or confirmed by the person."""
    rows = []
    if not os.path.isdir(data_dir):
        return rows
    out_dir = os.path.join(DIRS["clips"], "phone")
    manifest = os.path.join(DIRS["clips"], "phone.jsonl")
    have = {r["path"] for r in _read_manifest(manifest)}
    for sid in sorted(os.listdir(data_dir)):
        meta_path = os.path.join(data_dir, sid, "meta.json")
        crops_path = os.path.join(data_dir, sid, "crops.npy")
        if not (os.path.isfile(meta_path) and os.path.isfile(crops_path)):
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("language", "en") != language:
            continue
        # The label: what the person typed, else what the microphone heard
        # (speech recognition, when it was sure enough), else the reading
        # the person confirmed.
        how = None
        if meta.get("corrected"):
            text, how = meta["corrected"], "corrected"
        elif meta.get("heard") and (meta.get("heard_confidence") or 0) >= 0.4:
            text, how = meta["heard"], "heard"
        elif meta.get("confirmed"):
            text, how = meta.get("raw"), "confirmed"
        else:
            text = None
        if not text:
            continue
        text = normalise(text, language)
        rel = os.path.join("phone", sid + ".mp4")
        if rel in have:
            continue
        crops = np.load(crops_path)
        if crops.ndim != 3 or len(crops) < int(MIN_SECONDS * FPS):
            continue
        write_clip(crops.astype(np.uint8), os.path.join(DIRS["clips"], rel))
        row = {"path": rel, "frames": int(len(crops)), "text": text, "language": language, "source": "phone", "video": sid,
               "licence": "consented users of thelip.ai", "seconds": round(len(crops) / FPS, 2), "how": how}
        jsonl_append(manifest, row)
        rows.append(row)
    say(f"phone: {len(rows)} new samples for {language}")
    return rows


def _read_manifest(path: str) -> List[dict]:
    from .common import jsonl_read

    return jsonl_read(path)
