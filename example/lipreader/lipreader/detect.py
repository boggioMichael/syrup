"""Face detection: where the faces are, and where their eyes are, per frame.

Two interchangeable backends. `SyrupDetector` uses the syrup library's
functions-from-names (`find_face`, `find_eyes`, `find_profile_face`) through
its Python package; `OpenCVDetector` uses the Haar cascades that ship with
opencv-python. Both are Viola-Jones cascades, so they see roughly the same
faces; syrup runs its cascade on every core.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .schema import Box

Point = Tuple[float, float]


@dataclass
class Detection:
    box: Box
    score: float
    eyes: Optional[Tuple[Point, Point]] = None  # left, right (image order)
    profile: bool = False
    held: bool = False  # not a detection: a live track's box carried over a frame the detector skipped


class FaceDetector:
    name = "abstract"

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        raise NotImplementedError


def _pair_eyes(face: Box, eyes: Sequence[Point]) -> Optional[Tuple[Point, Point]]:
    """The two eyes inside the upper 60% of a face box, left to right."""
    inside = [
        e for e in eyes
        if face.x <= e[0] <= face.x2 and face.y <= e[1] <= face.y + 0.6 * face.h
    ]
    if len(inside) < 2:
        return None
    inside.sort(key=lambda e: e[0])
    if len(inside) > 2:
        # Keep the pair whose separation is closest to 0.4 face widths.
        best = None
        for i in range(len(inside)):
            for j in range(i + 1, len(inside)):
                d = abs(inside[j][0] - inside[i][0]) / max(face.w, 1.0)
                score = abs(d - 0.4) + abs(inside[j][1] - inside[i][1]) / max(face.h, 1.0)
                if best is None or score < best[0]:
                    best = (score, (inside[i], inside[j]))
        return best[1]
    left, right = inside
    if right[0] - left[0] < 0.15 * face.w:
        return None
    return (left, right)


class OpenCVDetector(FaceDetector):
    name = "opencv-haar (frontalface_alt2 + eye)"

    def __init__(self, min_face: int = 40, profile: bool = False):
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(os.path.join(base, "haarcascade_frontalface_alt2.xml"))
        self.eye = cv2.CascadeClassifier(os.path.join(base, "haarcascade_eye.xml"))
        self.profile = cv2.CascadeClassifier(os.path.join(base, "haarcascade_profileface.xml")) if profile else None
        self.min_face = min_face

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(self.min_face, self.min_face))
        out: List[Detection] = []
        for (x, y, w, h) in (found if len(found) else []):
            box = Box(float(x), float(y), float(w), float(h))
            roi = gray[y : y + int(h * 0.6), x : x + w]
            eyes = self.eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(max(8, w // 10),) * 2)
            centres = [(x + ex + ew / 2.0, y + ey + eh / 2.0) for (ex, ey, ew, eh) in (eyes if len(eyes) else [])]
            out.append(Detection(box, 1.0, _pair_eyes(box, centres)))
        if self.profile is not None:
            for flipped in (False, True):
                image = cv2.flip(gray, 1) if flipped else gray
                faces = self.profile.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4, minSize=(self.min_face, self.min_face))
                for (x, y, w, h) in (faces if len(faces) else []):
                    if flipped:
                        x = gray.shape[1] - x - w
                    box = Box(float(x), float(y), float(w), float(h))
                    if all(box.iou(d.box) < 0.3 for d in out):
                        out.append(Detection(box, 0.8, None, profile=True))
        return out


def _import_syrup():
    try:
        import syrup  # noqa: F401
        return syrup
    except ImportError:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "..", "..", "python"))
        if os.path.isdir(os.path.join(candidate, "syrup")):
            sys.path.insert(0, candidate)
            import syrup
            return syrup
        raise


class SyrupDetector(FaceDetector):
    """Faces and eyes through syrup: `find_face` and `find_eyes` are built
    from their names the first time they are used. Needs the native library
    (`cargo build --release` in the syrup repository).

    Faces are searched on a copy no wider than `max_width` (a cascade's cost
    grows with the pixels, faces on video are large); eyes are searched
    inside each face, on a copy of the face about `eye_search_size` wide."""

    name = "syrup (find_face + find_eyes)"

    def __init__(self, profile: bool = False, max_width: int = 480, eye_search_size: int = 200):
        self.syrup = _import_syrup()
        self.find_face = self.syrup.find_face
        self.find_eyes = self.syrup.find_eyes
        self.find_profile = self.syrup.find_profile_face if profile else None
        self.max_width = max_width
        self.eye_search_size = eye_search_size

    @staticmethod
    def available() -> bool:
        try:
            s = _import_syrup()
            s.library_path()
            return True
        except Exception:  # noqa: BLE001 - any failure means "not available"
            return False

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        h, w = rgb.shape[:2]
        scale = 1.0
        search = rgb
        if w > self.max_width:
            scale = self.max_width / w
            search = cv2.resize(rgb, (self.max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        faces = self.find_face(search).value or []
        out = []
        for face in faces:
            x, y, fw, fh = face.bounds
            box = Box(x / scale, y / scale, fw / scale, fh / scale)
            out.append(Detection(box, float(face.score), self._eyes(rgb, box)))
        if self.find_profile is not None:
            for face in self.find_profile(search).value or []:
                x, y, fw, fh = face.bounds
                box = Box(x / scale, y / scale, fw / scale, fh / scale)
                if all(box.iou(d.box) < 0.3 for d in out):
                    out.append(Detection(box, float(face.score), None, profile=True))
        return out

    def _eyes(self, rgb: np.ndarray, face: Box) -> Optional[Tuple[Point, Point]]:
        """Eyes inside one face: the face region, expanded a little, resized
        to a fixed size, searched, and the positions scaled back."""
        h, w = rgb.shape[:2]
        x0 = int(max(face.x - 0.15 * face.w, 0))
        y0 = int(max(face.y - 0.15 * face.h, 0))
        x1 = int(min(face.x2 + 0.15 * face.w, w))
        y1 = int(min(face.y2 + 0.15 * face.h, h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        region = rgb[y0:y1, x0:x1]
        scale = 1.0
        if region.shape[1] > self.eye_search_size:  # never upsample: a small face is searched as it is
            scale = self.eye_search_size / region.shape[1]
            region = cv2.resize(region, (self.eye_search_size, max(1, int(round(region.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        region = np.ascontiguousarray(region)
        found = self.find_eyes(region).value or []
        eyes = [(x0 + m.centre[0] / scale, y0 + m.centre[1] / scale) for m in found]
        return _pair_eyes(face, eyes)


def default_detector(profile: bool = False) -> FaceDetector:
    """syrup when its library is built, OpenCV's cascades otherwise."""
    if os.environ.get("LIPREADER_DETECTOR", "").lower() == "opencv":
        return OpenCVDetector(profile=profile)
    if SyrupDetector.available():
        return SyrupDetector(profile=profile)
    return OpenCVDetector(profile=profile)
