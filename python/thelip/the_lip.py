"""The Lip — lip reading from muted video, with live subtitles.

Named after Tony Lip (Green Book). syrup finds the face and the eyes and
locates the mouth; LipNet (Assael, Shillingford, Whiteson, de Freitas 2016),
run in numpy from the weights published with github.com/rizkiarm/LipNet
(MIT), reads the lips; the subtitle is decoded again on every frame from
what has been seen so far, so it appears while the person speaks.

    python3 the_lip.py <video>... --out demo.mp4 [--lipnet DIR] [--truth]

Each video is one utterance (GRID: 75 frames at 25 fps). Only the frames
are read: the audio track, if there is one, never enters. With `--truth`
the GRID file names give the spoken sentence, it is shown under the
subtitle, and a summary is written next to the output (`<out>.json`).

The LipNet checkout (weights, dictionary, sample clips) is `--lipnet`,
`LIPNET_DIR`, or `LipNet/` next to this file:

    git clone --depth 1 https://github.com/rizkiarm/LipNet
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lipnet_np import LipNet, Spell, greedy_decode, grid_sentence
from mouth import mouth_crops, read_frames

HERE = os.path.dirname(os.path.abspath(__file__))
LIPNET_DIR = os.environ.get("LIPNET_DIR", os.path.join(HERE, "LipNet"))
WEIGHTS = "evaluation/models/overlapped-weights368.h5"
DICTIONARY = "common/dictionaries/grid.txt"
# A sans-serif font wherever this runs; PIL's built-in font if none of them.
FONTS = {
    False: ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf", "segoeui.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf"],
    True: ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf",
           "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"],
}


def load_font(size, bold=False):
    for name in FONTS[bold]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow < 10.1: one size only
        return ImageFont.load_default()


# Characters the model emits in the last frames of a prefix are its guess at
# how the sentence continues; only what was emitted this many frames before
# the current one is shown while the person is still speaking.
SETTLE_FRAMES = 8


def streaming_decodes(net, crops):
    """The subtitle after each frame, using only frames seen so far:
    the greedy CTC path over the prefix, with characters emitted in its
    last SETTLE_FRAMES frames held back. Convolution features are
    temporally local, so the full-sequence features are reused and only
    the last frames are recomputed with the end padding a shorter sequence
    would have."""
    from lipnet_np import BLANK, LETTERS

    full = net.features(crops)
    decodes = []
    t0 = time.time()
    for t in range(len(crops)):
        prefix = full[: t + 1].copy()
        start = max(0, t - 5)
        tail = net.features(crops[start : t + 1])
        n = min(3, t + 1)
        prefix[t + 1 - n :] = tail[len(tail) - n :]
        probs = net.probabilities(prefix)
        path = probs.argmax(axis=1)[: max(0, t + 1 - SETTLE_FRAMES)]
        text, previous = [], -1
        for c in path:
            if c != previous and c != BLANK:
                text.append(" " if c == 26 else LETTERS[c])
            previous = c
        decodes.append("".join(text).strip())
    print(f"  {len(crops)} streaming decodes in {time.time() - t0:.1f}s", file=sys.stderr)
    return decodes


def render(frames, boxes, crops, decodes, final, spoken, scale, fonts, badge):
    """Annotated frames: the muted video, the mouth box, what the model
    sees, and the subtitle as it forms."""
    font, bold, small = fonts
    out = []
    h, w = frames[0].shape[:2]
    W, H = w * scale, h * scale + 96
    for i, frame in enumerate(frames):
        canvas = Image.new("RGB", (W, H), (12, 12, 16))
        big = Image.fromarray(frame).resize((w * scale, h * scale), Image.BILINEAR)
        canvas.paste(big, (0, 0))
        draw = ImageDraw.Draw(canvas)
        box = boxes[i]
        if box:
            x, y, bw, bh = [v * scale for v in box]
            draw.rectangle([x, y, x + bw, y + bh], outline=(80, 255, 120), width=2)
        # What the model sees, top right.
        crop = (crops[i].swapaxes(0, 1) * 255).astype(np.uint8)
        inset = Image.fromarray(crop).resize((200, 100), Image.NEAREST)
        canvas.paste(inset, (W - 212, 12))
        draw.rectangle([W - 213, 11, W - 12, 112], outline=(80, 255, 120), width=1)
        draw.text((W - 212, 116), "what the model sees", font=small, fill=(180, 180, 180))
        # Badges: which clip, and that nothing is heard.
        muted = "no audio: read from the lips"
        badge_w = max(draw.textlength(badge, font=small), draw.textlength(muted, font=small)) + 20
        draw.rounded_rectangle([12, 12, 12 + badge_w, 44], radius=6, fill=(0, 0, 0))
        draw.text((22, 18), badge, font=small, fill=(255, 230, 80))
        draw.rounded_rectangle([12, 52, 12 + badge_w, 84], radius=6, fill=(0, 0, 0))
        draw.text((22, 58), muted, font=small, fill=(255, 190, 190))
        # Subtitle bar.
        text = decodes[i]
        finished = i == len(frames) - 1
        line = final if finished else text
        draw.rectangle([0, h * scale, W, H], fill=(12, 12, 16))
        color = (255, 255, 255) if not finished else (120, 255, 150)
        tw = draw.textlength(line, font=bold)
        draw.text(((W - tw) / 2, h * scale + 14), line, font=bold, fill=color)
        if finished and spoken:
            note = f"spoken: {spoken}   {'correct' if spoken == final else 'differs'}"
            tw = draw.textlength(note, font=small)
            draw.text(((W - tw) / 2, h * scale + 58), note, font=small, fill=(170, 170, 170))
        else:
            tw = draw.textlength("reading...", font=small)
            draw.text(((W - tw) / 2, h * scale + 58), "reading...", font=small, fill=(120, 120, 120))
        out.append(canvas)
    # Hold the final frame so the sentence can be read.
    out.extend([out[-1]] * 25)
    return out


def encode(frames, path, fps):
    W, H = frames[0].size
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="clips, or glob patterns (expanded here, for shells that do not)")
    ap.add_argument("--out", default="the-lip.mp4")
    ap.add_argument("--lipnet", default=LIPNET_DIR, help="checkout of github.com/rizkiarm/LipNet")
    ap.add_argument("--weights", default=None, help=f"default: <lipnet>/{WEIGHTS}")
    ap.add_argument("--dictionary", default=None, help=f"default: <lipnet>/{DICTIONARY}")
    ap.add_argument("--truth", action="store_true", help="the GRID file name encodes the sentence; show it")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    weights = args.weights or os.path.join(args.lipnet, WEIGHTS)
    dictionary = args.dictionary or os.path.join(args.lipnet, DICTIONARY)
    if not os.path.exists(weights):
        sys.exit(f"no weights at {weights}: clone https://github.com/rizkiarm/LipNet as {args.lipnet} or pass --lipnet")
    net = LipNet(weights)
    spell = Spell(dictionary)
    fonts = (load_font(22), load_font(34, bold=True), load_font(18))
    videos = [path for pattern in args.videos for path in (sorted(glob.glob(pattern)) or [pattern])]
    if not videos:
        sys.exit("no videos")
    all_frames = []
    fps = 25.0
    results = []
    for k, path in enumerate(videos):
        code = os.path.splitext(os.path.basename(path))[0]
        frames, fps = read_frames(path)
        crops, boxes, ratio = mouth_crops(frames)
        decodes = streaming_decodes(net, crops)
        # Once the clip has ended nothing is held back: the whole utterance,
        # decoded, then corrected against the dictionary.
        final = spell.sentence(greedy_decode(net.predict(crops)))
        spoken = grid_sentence(code) if args.truth else None
        results.append((code, final, spoken))
        print(f"{code}: read {final!r}" + (f" spoken {spoken!r}" if spoken else ""), file=sys.stderr)
        badge = f"THE LIP  -  muted clip {k + 1}/{len(videos)}"
        all_frames.extend(render(frames, boxes, crops, decodes, final, spoken, args.scale, fonts, badge))
    encode(all_frames, args.out, fps)
    if args.truth:
        words = sum(len(s.split()) for _, _, s in results)
        right = sum(sum(1 for a, b in zip(f.split(), s.split()) if a == b) for _, f, s in results)
        sentences = sum(1 for _, f, s in results if f == s)
        print(f"{right}/{words} words right over {len(results)} clips, {sentences} sentences exact", file=sys.stderr)
        summary = {
            "clips": len(results),
            "words": words,
            "words_right": right,
            "sentences_right": sentences,
            "size": f"{all_frames[0].size[0]}x{all_frames[0].size[1]}",
            "fps": fps,
            "results": [{"clip": c, "read": f, "spoken": s} for c, f, s in results],
        }
        with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
