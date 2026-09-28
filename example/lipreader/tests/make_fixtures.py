"""Test videos with exact ground truth, composed from GRID sample clips.

GRID file names encode the sentence, every clip has one frontal speaker,
and the clips ship with LipNet's repository, so multi-person scenes can be
built whose words, speaker regions, speaking times and cut times are all
known exactly:

    one_speaker        one clip, with its audio
    two_alternating    two people side by side, one speaks then the other
    two_simultaneous   two people side by side, both speaking at once
    five_faces         five people, three speaking, two still
    low_res            two_simultaneous at half size
    rapid_cuts         four clips hard-cut every 1.2 s
    muted              one_speaker without an audio track

    python3 make_fixtures.py [--lipnet DIR] [--out DIR]

Each fixture gets a `<name>.json` beside it with the truth.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "..", "python", "thelip")))
from lipreader import audio as aud  # noqa: E402
from lipreader.vsr.lipnet import lipnet_dir  # noqa: E402
from lipnet_np import grid_sentence  # noqa: E402

FPS = 25
RATE = aud.RATE


def read_clip(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    audio = aud.samples(path)
    if audio is None:
        audio = np.zeros(int(len(frames) / FPS * RATE), np.float32)
    return frames, audio


def speech_span(audio):
    spans = aud.voice_activity(audio)
    if not spans:
        return [0.0, len(audio) / RATE]
    return [round(spans[0].start, 2), round(spans[-1].end, 2)]


def write_video(frames, audio, path, with_audio=True):
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "video.rgb")
        with open(raw, "wb") as f:
            for frame in frames:
                f.write(np.ascontiguousarray(frame).tobytes())
        cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", raw]
        if with_audio:
            wav = os.path.join(tmp, "audio.f32")
            with open(wav, "wb") as f:
                f.write(np.asarray(audio, np.float32).tobytes())
            cmd += ["-f", "f32le", "-ar", str(RATE), "-ac", "1", "-i", wav, "-c:a", "aac", "-b:a", "96k", "-shortest"]
        cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path]
        subprocess.run(cmd, check=True)


def person(region, words, spans):
    return {"region": [int(v) for v in region], "words": words, "speaking": [[round(a, 2), round(b, 2)] for a, b in spans]}


def build(lipnet, out):
    grid = os.path.join(lipnet, "evaluation", "samples", "GRID")
    codes = {"A": "sbwe5n", "B": "brbk7n", "C": "lbax4n", "D": "pwij3p", "E": "lrwp9a"}
    clips = {k: read_clip(os.path.join(grid, f"{c}.mpg")) for k, c in codes.items()}
    words = {k: grid_sentence(c) for k, c in codes.items()}
    spans = {k: speech_span(clips[k][1]) for k in codes}
    n = min(len(f) for f, _ in clips.values())
    h, w = clips["A"][0][0].shape[:2]
    os.makedirs(out, exist_ok=True)
    fixtures = {}

    # one_speaker / muted
    frames, audio = clips["A"]
    write_video(frames[:n], audio, os.path.join(out, "one_speaker.mp4"))
    write_video(frames[:n], audio, os.path.join(out, "muted.mp4"), with_audio=False)
    truth = {"size": [w, h], "duration": n / FPS, "cuts": [], "people": [person((0, 0, w, h), words["A"], [spans["A"]])]}
    fixtures["one_speaker"] = truth
    fixtures["muted"] = dict(truth, has_audio=False)

    # two_alternating: A speaks (B still), then B speaks (A still)
    fa, aa = clips["A"]
    fb, ab = clips["B"]
    left = fa[:n] + [fa[n - 1]] * n
    right = [fb[0]] * n + fb[:n]
    frames = [np.concatenate([l, r], axis=1) for l, r in zip(left, right)]
    audio = np.concatenate([aa[: n * RATE // FPS], ab[: n * RATE // FPS]])
    write_video(frames, audio, os.path.join(out, "two_alternating.mp4"))
    fixtures["two_alternating"] = {
        "size": [2 * w, h], "duration": 2 * n / FPS, "cuts": [],
        "people": [person((0, 0, w, h), words["A"], [spans["A"]]),
                   person((w, 0, w, h), words["B"], [[spans["B"][0] + n / FPS, spans["B"][1] + n / FPS]])],
    }

    # two_simultaneous
    frames = [np.concatenate([l, r], axis=1) for l, r in zip(fa[:n], fb[:n])]
    m = min(len(aa), len(ab))
    audio = 0.5 * (aa[:m] + ab[:m])
    write_video(frames, audio, os.path.join(out, "two_simultaneous.mp4"))
    fixtures["two_simultaneous"] = {
        "size": [2 * w, h], "duration": n / FPS, "cuts": [],
        "people": [person((0, 0, w, h), words["A"], [spans["A"]]), person((w, 0, w, h), words["B"], [spans["B"]])],
    }

    # low_res: two_simultaneous at half size
    small = [cv2.resize(f, (w, h // 2), interpolation=cv2.INTER_AREA) for f in frames]
    write_video(small, audio, os.path.join(out, "low_res.mp4"))
    fixtures["low_res"] = {
        "size": [w, h // 2], "duration": n / FPS, "cuts": [],
        "people": [person((0, 0, w // 2, h // 2), words["A"], [spans["A"]]), person((w // 2, 0, w // 2, h // 2), words["B"], [spans["B"]])],
    }

    # five_faces: 3 x 2 tiles; A, C, E speak; B, D still; one tile black
    layout = ["A", "B", "C", "D", "E", None]
    tiles_frames = []
    for i in range(n):
        rows = []
        for r in range(2):
            row = []
            for c in range(3):
                key = layout[r * 3 + c]
                if key is None:
                    row.append(np.zeros((h, w, 3), np.uint8))
                elif key in ("B", "D"):
                    row.append(clips[key][0][0])
                else:
                    row.append(clips[key][0][i])
            rows.append(np.concatenate(row, axis=1))
        tiles_frames.append(np.concatenate(rows, axis=0))
    m = min(len(clips[k][1]) for k in ("A", "C", "E"))
    audio = (clips["A"][1][:m] + clips["C"][1][:m] + clips["E"][1][:m]) / 3.0
    write_video(tiles_frames, audio, os.path.join(out, "five_faces.mp4"))
    people = []
    for idx, key in enumerate(layout):
        if key is None:
            continue
        r, c = divmod(idx, 3)
        region = (c * w, r * h, w, h)
        if key in ("B", "D"):
            people.append(person(region, "", []))
        else:
            people.append(person(region, words[key], [spans[key]]))
    fixtures["five_faces"] = {"size": [3 * w, 2 * h], "duration": n / FPS, "cuts": [], "people": people}

    # rapid_cuts: four clips, 1.2 s each, hard cuts
    piece = int(1.2 * FPS)
    order = ["A", "B", "C", "D"]
    frames = []
    audio_parts = []
    for key in order:
        frames.extend(clips[key][0][:piece])
        audio_parts.append(clips[key][1][: piece * RATE // FPS])
    write_video(frames, np.concatenate(audio_parts), os.path.join(out, "rapid_cuts.mp4"))
    fixtures["rapid_cuts"] = {
        "size": [w, h], "duration": len(frames) / FPS, "cuts": [round(i * piece / FPS, 2) for i in range(1, len(order))],
        "people": [person((0, 0, w, h), None, [[i * piece / FPS, (i + 1) * piece / FPS]]) for i in range(len(order))],
        "note": "each shot is the first 1.2 s of a different clip; the words spoken inside 1.2 s are not known exactly (null)",
    }

    for name, truth in fixtures.items():
        with open(os.path.join(out, f"{name}.json"), "w") as f:
            json.dump(truth, f, indent=1)
    print(f"wrote {len(fixtures)} fixtures to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lipnet", default=lipnet_dir())
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures"))
    args = ap.parse_args()
    if not args.lipnet:
        sys.exit("no LipNet checkout (git clone --depth 1 https://github.com/rizkiarm/LipNet python/thelip/LipNet)")
    build(args.lipnet, args.out)


if __name__ == "__main__":
    main()
