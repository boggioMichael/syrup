"""What MediaPipe's own face detector finds in a few pictures, for faces.js
to be checked against (test/faces.test.mjs): test/faces-ref.json, and the
128x128 pictures themselves in test/faces-ref.rgb.gz (raw RGB, one after
the other).

    python3 tools/faces_reference.py blaze_face_short_range.tflite

Needs mediapipe (its Tasks FaceDetector, run with the same model file that
tools/export_blazeface.py reads) and ffmpeg. The pictures are frames of
GRID's sample clips in docs/test/grid (CC BY 4.0): each at 128x128, the
network's own input size, where MediaPipe's letterboxing does nothing and
both must agree to rounding; two faces side by side; a picture with no face;
and whole 360x288 frames (which the test takes from the clips with ffmpeg
too), which MediaPipe letterboxes into its input: there the page's own
letterboxing is compared, to a pixel or so.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(HERE, "..", "..", "..", "docs", "test", "grid")
CLIPS = ["bbaf2n", "lbax4n", "pwij3p", "swiz3n", "sbwe5n"]


def ffmpeg_rgb(clip: str, vf: str, size) -> np.ndarray:
    """A frame of a clip through an ffmpeg filter, as raw RGB (the test does the same)."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(GRID, clip + ".webm"), "-vf", vf, "-vframes", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3)


WHOLE = "select=eq(n\\,20)"
SQUARE = WHOLE + ",crop=288:288:36:0,scale=128:128:flags=bilinear"


def main():
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    model = sys.argv[1]
    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=model)))

    def faces(a):
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(a)))
        h, w = a.shape[:2]
        return [{"score": d.categories[0].score,
                 "box": [d.bounding_box.origin_x / w, d.bounding_box.origin_y / h, d.bounding_box.width / w, d.bounding_box.height / h],
                 "keypoints": [[k.x, k.y] for k in d.keypoints]} for d in res.detections]

    pictures, pixels = [], []
    for c in CLIPS:
        sq = ffmpeg_rgb(c, SQUARE, (128, 128))
        pictures.append({"name": f"{c} at 128x128", "size": [128, 128], "faces": faces(sq)})
        pixels.append(sq)
    left, right = ffmpeg_rgb("bbaf2n", WHOLE, (360, 288))[:, 82:226], ffmpeg_rgb("sbwe5n", WHOLE, (360, 288))[:, 110:254]
    two = np.concatenate([left, right], axis=1)   # two 144x288 halves, then 128x128 (Pillow)
    two = np.asarray(Image.fromarray(two).resize((128, 128), Image.BILINEAR))
    pictures.append({"name": "two faces at 128x128", "size": [128, 128], "faces": faces(two)})
    pixels.append(two)
    grey = np.full((128, 128, 3), 128, np.uint8)
    pictures.append({"name": "no face", "size": [128, 128], "faces": faces(grey)})
    pixels.append(grey)
    for c in CLIPS[:3]:
        pictures.append({"name": f"{c} whole (360x288, letterboxed)", "size": [360, 288], "clip": c, "vf": WHOLE,
                         "faces": faces(ffmpeg_rgb(c, WHOLE, (360, 288)))})
    det.close()
    import gzip

    with open(os.path.join(HERE, "..", "test", "faces-ref.rgb.gz"), "wb") as f:
        f.write(gzip.compress(b"".join(p.tobytes() for p in pixels), mtime=0))
    out = os.path.join(HERE, "..", "test", "faces-ref.json")
    with open(out, "w") as f:
        json.dump({"model_sha256": __import__("hashlib").sha256(open(model, "rb").read()).hexdigest(), "pictures": pictures}, f, indent=1)
    for p in pictures:
        print(p["name"], [(round(x["score"], 3), [round(v, 3) for v in x["keypoints"][3]]) for x in p["faces"]])
    print("wrote", out, os.path.getsize(out), "bytes")


if __name__ == "__main__":
    main()
