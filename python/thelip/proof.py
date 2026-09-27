"""The Lip — proof and code. A video in three parts:

1. Every sample clip twice: muted, the subtitle forming from the lips
   alone; then the same clip with its own sound, so the viewer hears
   whether the reading was right.
2. A user's session, recorded for real: clone, build, fetch the weights,
   run the tests, subtitle a clip, read one from Python. Every line of
   output on screen is what the commands printed.
3. The code behind it, stepped through with a highlight and a caption.

    python3 proof.py record --repo <git url or path> --workdir proofwork
    python3 proof.py render --workdir proofwork --out the-lip-proof.mp4

`record` needs cargo, git, python3 with numpy/opencv/pillow and network
access to github.com; `render` needs ffmpeg with libx264 and, for the
syntax colours, pygments (plain text without it).
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lipnet_np import LipNet, Spell, greedy_decode, grid_sentence  # noqa: E402
from mouth import mouth_crops, read_frames  # noqa: E402
from the_lip import DICTIONARY, WEIGHTS, load_font, settled_words, streaming_decodes  # noqa: E402

W, H, FPS = 1280, 720, 25
RATE = 48000
BG = (12, 12, 16)
PANEL = (22, 22, 28)
WHITE, GREY, DIM = (240, 240, 240), (190, 190, 190), (120, 120, 130)
YELLOW, GREEN, RED, BLUE = (255, 230, 80), (120, 255, 150), (255, 130, 130), (130, 190, 255)

USE_PY = '''from lipnet_np import LipNet, Spell, greedy_decode
from mouth import mouth_crops, read_frames

frames, fps = read_frames("LipNet/evaluation/samples/GRID/lrwp9a.mpg")  # any 25 fps clip of a face
crops, boxes, ratio = mouth_crops(frames)                               # syrup finds the mouth
net = LipNet("LipNet/evaluation/models/overlapped-weights368.h5")
letters = greedy_decode(net.predict(crops))                             # what the lips said
print(letters, "->", Spell("LipNet/common/dictionaries/grid.txt").sentence(letters))
'''

# The session: (display prompt, working directory under workdir, command,
# caption). `{repo}` is filled in at record time.
SESSION = [
    ("~/demo", ".", "git clone --branch claude/intents-2 {repo} syrup",
     "1  get the code"),
    ("~/demo/syrup", "syrup", "{cargo_env}cargo build --release",
     "2  build the library once; python/syrup finds target/release on its own"),
    ("~/demo/syrup/python/thelip", "syrup/python/thelip",
     'python3 -c "import numpy, cv2, PIL; print(numpy.__version__, cv2.__version__, PIL.__version__)"',
     "3  Python needs numpy, opencv and pillow (pip install numpy pillow opencv-python-headless)"),
    ("~/demo/syrup/python/thelip", "syrup/python/thelip", "git clone --depth 1 https://github.com/rizkiarm/LipNet",
     "4  the published weights, the dictionary and the sample clips"),
    ("~/demo/syrup/python/thelip", "syrup/python/thelip", "python3 test_thelip.py",
     "5  the tests: the numpy layers against naive references, the decoder, the mouth locator"),
    ("~/demo/syrup/python/thelip", "syrup/python/thelip",
     "python3 the_lip.py LipNet/evaluation/samples/GRID/pwij3p.mpg --out subtitled.mp4",
     "6  subtitle a clip from its lips; the file it writes plays next"),
    ("~/demo/syrup/python/thelip", "syrup/python/thelip", "cat use.py",
     "7  the same from your own code: eight lines"),
    ("~/demo/syrup/python/thelip", "syrup/python/thelip", "python3 use.py",
     "8  letters straight from the network, then the dictionary's correction"),
    ("~/demo/syrup/python/thelip", "syrup/python/thelip", "python3 evaluate.py",
     "9  every sample clip against the sentence its GRID name encodes"),
]


# --- recording ------------------------------------------------------------

def clean_output(text):
    """Terminal output as lines: carriage-return progress collapsed to its
    last state, trailing blanks dropped."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        if lines and "%" in line and "%" in lines[-1] and line[:12] == lines[-1][:12]:
            lines[-1] = line
        else:
            lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def record(repo, workdir, cargo_offline=False):
    os.makedirs(workdir, exist_ok=True)
    steps = []
    cargo_env = "CARGO_NET_OFFLINE=true " if cargo_offline else ""
    for prompt, cwd, command, caption in SESSION:
        command = command.format(repo=repo, cargo_env=cargo_env)
        if cargo_offline and command.startswith(cargo_env + "cargo"):
            caption += " (offline here: this machine cannot reach crates.io; the dependencies were already cached)"
        if command.startswith("git clone") and "syrup" in command and "://" not in repo:
            caption += " (a local copy here; from GitHub: git clone https://github.com/boggioMichael/syrup)"
        path = os.path.join(workdir, cwd)
        if command == "cat use.py":
            with open(os.path.join(path, "use.py"), "w") as f:
                f.write(USE_PY)
        print(f"$ {command}", file=sys.stderr)
        t0 = time.time()
        proc = subprocess.run(command, shell=True, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        seconds = time.time() - t0
        lines = clean_output(proc.stdout)
        print("\n".join(lines[-3:]), file=sys.stderr)
        steps.append({"prompt": prompt, "command": command, "caption": caption, "output": lines,
                      "seconds": round(seconds, 1), "code": proc.returncode})
        if proc.returncode != 0:
            sys.exit(f"step failed with code {proc.returncode}; the session is not recorded")
    with open(os.path.join(workdir, "session.json"), "w") as f:
        json.dump(steps, f, indent=1)
    print(f"recorded {len(steps)} steps to {workdir}/session.json", file=sys.stderr)


# --- drawing helpers ------------------------------------------------------

def wrap(draw, text, font, width):
    lines, line = [], ""
    for word in text.split():
        trial = (line + " " + word).strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def paragraph(draw, text, font, x, y, width, fill, gap=6):
    height = font.getbbox("Ag")[3] + gap
    for line in wrap(draw, text, font, width):
        draw.text((x, y), line, font=font, fill=fill)
        y += height
    return y


def centred(draw, text, font, y, fill, x0=0, x1=W):
    tw = draw.textlength(text, font=font)
    draw.text((x0 + (x1 - x0 - tw) / 2, y), text, font=font, fill=fill)


def card(blocks, margin=80):
    """A centred column of (text, font, colour, gap_after)."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    laid = []
    for text, font, colour, gap in blocks:
        lines = wrap(draw, text, font, W - 2 * margin) or [""]
        for line in lines:
            laid.append([line, font, colour, 0])
        laid[-1][3] = gap
    heights = [font.getbbox("Ag")[3] + 8 for _, font, _, _ in laid]
    total = sum(h + gap for h, (_, _, _, gap) in zip(heights, laid))
    y = (H - total) / 2
    for (line, font, colour, gap), h in zip(laid, heights):
        centred(draw, line, font, y, colour)
        y += h + gap
    return img


class Fonts:
    def __init__(self):
        self.title = load_font(64, bold=True)
        self.h1 = load_font(30, bold=True)
        self.h2 = load_font(24, bold=True)
        self.body = load_font(21)
        self.small = load_font(17)
        self.fine = load_font(14)
        self.sub = load_font(36, bold=True)
        self.mono = load_font(14, mono=True)
        self.mono_small = load_font(16, mono=True)


# --- syntax colouring -----------------------------------------------------

def coloured_lines(source):
    """[[(text, colour), ...] per line], pygments if present."""
    try:
        from pygments import lex
        from pygments.lexers import PythonLexer
        from pygments.token import Comment, Keyword, Name, Number, Operator, String
    except ImportError:
        return [[(line, WHITE)] for line in source.split("\n")]
    palette = [
        (Comment, (110, 118, 129)), (String, (152, 195, 121)), (Keyword, (198, 120, 221)),
        (Number, (209, 154, 102)), (Name.Function, (97, 175, 239)), (Name.Class, (229, 192, 123)),
        (Name.Builtin, (86, 182, 194)), (Name.Decorator, (86, 182, 194)), (Operator, (200, 200, 210)),
    ]
    lines, current = [], []
    for token, text in lex(source, PythonLexer()):
        colour = WHITE
        for kind, c in palette:
            if token in kind:
                colour = c
                break
        parts = text.split("\n")
        for i, part in enumerate(parts):
            if part:
                current.append((part, colour))
            if i < len(parts) - 1:
                lines.append(current)
                current = []
    if current:
        lines.append(current)
    return lines


def snippet(path, pieces):
    """Lines of `path` for a list of (start, end) markers, joined with an
    ellipsis line. `end` is included; None means the next blank line."""
    with open(path) as f:
        source = f.read().split("\n")
    out = []
    for start, end in pieces:
        i = next(k for k, line in enumerate(source) if start in line)
        if end is None:
            j = i
            while j + 1 < len(source) and source[j + 1].strip():
                j += 1
        else:
            j = next(k for k in range(i, len(source)) if end in source[k])
        if out:
            out.append("    ...")
        out.extend(source[i : j + 1])
    # Drop the common indentation.
    indent = min((len(l) - len(l.lstrip()) for l in out if l.strip() and l.strip() != "..."), default=0)
    return [l[indent:] if l.strip() else "" for l in out]


# --- segments -------------------------------------------------------------

class Timeline:
    """Segments in order; each yields frames and may carry audio (mono int16
    at RATE) aligned to its first frame."""

    def __init__(self):
        self.segments = []

    def add(self, frames, count, audio=None):
        self.segments.append((frames, count, audio))

    @property
    def total(self):
        return sum(count for _, count, _ in self.segments)

    def audio(self):
        samples = np.zeros(int(self.total * RATE / FPS) + RATE, dtype=np.int16)
        at = 0
        for _, count, audio in self.segments:
            if audio is not None:
                start = int(at * RATE / FPS)
                n = min(len(audio), len(samples) - start)
                samples[start : start + n] = audio[:n]
            at += count
        return samples[: int(self.total * RATE / FPS)]

    def frames(self):
        for frames, count, _ in self.segments:
            produced = 0
            for frame in frames():
                yield frame
                produced += 1
                if produced == count:
                    break
            while produced < count:  # a short generator: hold its last frame
                yield frame
                produced += 1


def hold(img, count):
    return (lambda: iter([img] * count)), count


def clip_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16)


def clip_pair(fonts, frames, boxes, crops, decodes, final, spoken, audio, index, count):
    """Two passes over one clip. Returns (generator factory, frame count)."""
    hold_frames = 25
    n = len(frames)
    total = 2 * (n + hold_frames)
    levels = np.zeros(n + hold_frames, dtype=np.float32)
    per = RATE // FPS
    for i in range(min(n, len(audio) // per)):
        chunk = audio[i * per : (i + 1) * per].astype(np.float32) / 32768.0
        levels[i] = float(np.sqrt((chunk**2).mean()))
    peak = max(levels.max(), 1e-3)
    inset_h, inset_w = 150, 300

    def make():
        for f in range(total):
            second = f >= n + hold_frames
            i = min(f - (n + hold_frames if second else 0), n - 1)
            finished = (f - (n + hold_frames if second else 0)) >= n
            canvas = Image.new("RGB", (W, H), BG)
            big = Image.fromarray(frames[i]).resize((720, 576), Image.BILINEAR)
            canvas.paste(big, (0, 0))
            draw = ImageDraw.Draw(canvas)
            box = boxes[i]
            if box:
                x, y, bw, bh = [v * 2 for v in box]
                draw.rectangle([x, y, x + bw, y + bh], outline=GREEN, width=2)
            # Subtitle bar under the video.
            draw.rectangle([0, 576, 720, H], fill=PANEL)
            if second:
                centred(draw, final, fonts.sub, 592, GREEN, 0, 720)
                if finished:
                    verdict = "correct" if spoken == final else "differs"
                    centred(draw, f"spoken: {spoken}   -   {verdict}", fonts.small, 648, GREY if spoken == final else RED, 0, 720)
                else:
                    centred(draw, "read from the lips before any sound was played", fonts.small, 648, DIM, 0, 720)
            else:
                line = final if finished else decodes[i]
                centred(draw, line, fonts.sub, 592, GREEN if finished else WHITE, 0, 720)
                centred(draw, "decoded from the mouth alone - no audio" if not finished else "the whole utterance, decoded once more",
                        fonts.small, 648, DIM, 0, 720)
            # Right panel.
            draw.rectangle([720, 0, W, H], fill=PANEL)
            draw.text((744, 22), "THE LIP", font=fonts.h1, fill=YELLOW)
            draw.text((744, 62), f"clip {index} of {count}", font=fonts.small, fill=GREY)
            for k, (label, active) in enumerate([("1   muted: the subtitle comes from the lips", not second),
                                                  ("2   the same clip, with its sound", second)]):
                y = 104 + k * 40
                if active:
                    draw.rounded_rectangle([736, y - 6, W - 24, y + 28], radius=6, fill=(40, 40, 52))
                draw.text((748, y), label, font=fonts.body, fill=WHITE if active else DIM)
            # What the model sees.
            crop = (crops[i].swapaxes(0, 1) * 255).astype(np.uint8)
            inset = Image.fromarray(crop).resize((inset_w, inset_h), Image.NEAREST)
            canvas.paste(inset, (744, 206))
            draw.rectangle([743, 205, 744 + inset_w, 206 + inset_h], outline=GREEN, width=1)
            draw.text((744, 362), "what the model sees: 100x50, around the mouth syrup found", font=fonts.fine, fill=GREY)
            y = 404
            if not second:
                draw.rounded_rectangle([744, y, 1010, y + 34], radius=6, fill=(70, 20, 20))
                draw.text((756, y + 6), "NO AUDIO", font=fonts.body, fill=RED)
                y = paragraph(draw, "The sound track is not read at all. After every frame the whole prefix is decoded again "
                              "from the mouth crops; the newest 8 frames are held back, so what you see has settled.",
                              fonts.small, 744, y + 48, W - 768, GREY)
            else:
                draw.rounded_rectangle([744, y, 1010, y + 34], radius=6, fill=(20, 60, 30))
                draw.text((756, y + 6), "SOUND ON", font=fonts.body, fill=GREEN)
                # Level meter from the clip's real audio.
                level = levels[min(i, len(levels) - 1)] / peak if not finished else 0.0
                bars = 24
                for b in range(bars):
                    lit = level * bars > b
                    colour = (80, 220, 120) if lit else (44, 44, 56)
                    draw.rectangle([1028 + b * 9, y + 6, 1028 + b * 9 + 6, y + 28], fill=colour)
                y = paragraph(draw, "Listen: the words you hear are the words the subtitle already showed, read with the sound off.",
                              fonts.small, 744, y + 48, W - 768, GREY)
                if finished:
                    paragraph(draw, f"The GRID file name encodes the sentence: \"{spoken}\".", fonts.small, 744, y + 12, W - 768, DIM)
            yield canvas

    return make, total


def pair_audio(audio, n_frames, hold_frames):
    """Audio for a clip pair: silence for the muted pass, the clip's own
    sound for the second."""
    out = np.zeros(int(2 * (n_frames + hold_frames) * RATE / FPS), dtype=np.int16)
    start = int((n_frames + hold_frames) * RATE / FPS)
    n = min(len(audio), len(out) - start)
    out[start : start + n] = audio[:n]
    return out


def terminal(fonts, step, chars_per_frame=3):
    """A command typed, its output revealed, a pause; the caption under."""
    prompt, command, lines, caption, seconds = step["prompt"], step["command"], step["output"], step["caption"], step["seconds"]
    font = fonts.mono_small
    line_h = 21
    top, bottom = 24, H - 100
    visible = (bottom - top) // line_h
    max_chars = 128
    wrapped = []
    for line in lines:
        wrapped.extend([line[k : k + max_chars] for k in range(0, len(line), max_chars)] or [""])
    typing = math.ceil(len(command) / chars_per_frame)
    reveal_per_frame = max(1, math.ceil(len(wrapped) / 60))
    revealing = math.ceil(len(wrapped) / reveal_per_frame)
    pause = 60 if len(wrapped) < 20 else 90
    total = typing + 10 + revealing + pause

    def make():
        for f in range(total):
            canvas = Image.new("RGB", (W, H), (18, 18, 24))
            draw = ImageDraw.Draw(canvas)
            typed = command[: min(len(command), f * chars_per_frame)]
            shown = [] if f < typing + 10 else wrapped[: min(len(wrapped), (f - typing - 10) * reveal_per_frame)]
            rows = [("prompt", f"{prompt}$ {typed}" + ("_" if f < typing + 10 and f % 10 < 5 else ""))]
            rows += [("out", l) for l in shown]
            if f >= typing + 10 + revealing:
                rows.append(("note", f"[took {seconds:.0f} s]" if seconds >= 1 else ""))
            rows = rows[-visible:]
            y = top
            for kind, text in rows:
                if kind == "prompt":
                    p = f"{prompt}$ "
                    draw.text((24, y), p, font=font, fill=GREEN)
                    draw.text((24 + draw.textlength(p, font=font), y), text[len(p):], font=font, fill=WHITE)
                else:
                    draw.text((24, y), text, font=font, fill=GREY if kind == "out" else DIM)
                y += line_h
            draw.rectangle([0, H - 88, W, H], fill=PANEL)
            paragraph(draw, caption, fonts.small, 24, H - 74, W - 48, YELLOW, gap=4)
            note = "recorded for real: every line is what the command printed"
            draw.text((W - 24 - draw.textlength(note, font=fonts.fine), 6), note, font=fonts.fine, fill=DIM)
            yield canvas

    return make, total


def play_file(fonts, path, caption):
    frames, fps = read_frames(path)
    step = max(1, round(fps / FPS))
    frames = frames[::step]
    fh, fw = frames[0].shape[:2]
    scale = min((W - 80) / fw, (H - 140) / fh)
    size = (int(fw * scale), int(fh * scale))
    x0, y0 = (W - size[0]) // 2, 20

    def make():
        for frame in frames:
            canvas = Image.new("RGB", (W, H), BG)
            canvas.paste(Image.fromarray(frame).resize(size, Image.BILINEAR), (x0, y0))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([x0 - 1, y0 - 1, x0 + size[0], y0 + size[1]], outline=(60, 60, 70))
            draw.rectangle([0, H - 84, W, H], fill=PANEL)
            draw.text((24, H - 66), caption, font=fonts.body, fill=YELLOW)
            draw.text((24, H - 34), os.path.basename(path), font=fonts.fine, fill=DIM)
            yield canvas

    return make, len(frames)


def wrap_code(lines, max_chars=96):
    """Display lines for source lines: long ones folded at a space with a
    continuation indent. Returns [(source index, text)]."""
    display = []
    for k, line in enumerate(lines):
        indent = len(line) - len(line.lstrip()) + 8
        rest = line
        while len(rest) > max_chars:
            cut = rest.rfind(" ", indent + 1, max_chars)
            if cut <= indent:
                cut = max_chars
            display.append((k, rest[:cut]))
            rest = " " * indent + rest[cut:].lstrip()
        display.append((k, rest))
    return display


def code_slide(fonts, title, filename, lines, steps, frames_per_step=90):
    """Code at left with a moving highlight; the step's caption at right.
    Snippets taller than the panel scroll to keep the highlight in view."""
    display = wrap_code(lines)
    coloured = coloured_lines("\n".join(text for _, text in display))
    origin = [k for k, _ in display]
    mono = fonts.mono
    line_h = 19
    code_w = 840
    max_rows = (H - 52) // line_h
    spans = []
    for start, end, _ in steps:
        i = next(k for k, l in enumerate(lines) if start in l)
        j = i if end is None else next(k for k in range(i, len(lines)) if end in lines[k])
        spans.append((i, j))
    total = frames_per_step * len(steps)

    def first_row(lo, hi):
        if len(display) <= max_rows:
            return 0
        rows_lo = next(r for r, k in enumerate(origin) if k == lo)
        rows_hi = max(r for r, k in enumerate(origin) if k == hi)
        start = rows_lo - (max_rows - (rows_hi - rows_lo + 1)) // 2
        return max(0, min(start, len(display) - max_rows))

    def make():
        for f in range(total):
            s = min(f // frames_per_step, len(steps) - 1)
            lo, hi = spans[s]
            top = first_row(lo, hi)
            canvas = Image.new("RGB", (W, H), (18, 18, 24))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([0, 0, code_w, 40], fill=PANEL)
            draw.text((20, 10), filename, font=fonts.small, fill=GREY)
            y = 52
            for row in range(top, min(len(display), top + max_rows)):
                parts = coloured[row] if row < len(coloured) else []
                lit = lo <= origin[row] <= hi
                if lit:
                    draw.rectangle([0, y - 2, code_w, y + line_h - 2], fill=(52, 52, 30))
                    draw.rectangle([0, y - 2, 5, y + line_h - 2], fill=YELLOW)
                x = 20
                for text, colour in parts:
                    fill = colour if lit else tuple(int(c * 0.55) for c in colour)
                    draw.text((x, y), text, font=mono, fill=fill)
                    x += draw.textlength(text, font=mono)
                y += line_h
            draw.rectangle([code_w, 0, W, H], fill=PANEL)
            x = code_w + 28
            y = paragraph(draw, title, fonts.h2, x, 28, W - x - 24, YELLOW, gap=8)
            draw.text((x, y + 6), f"step {s + 1} of {len(steps)}", font=fonts.fine, fill=DIM)
            paragraph(draw, steps[s][2], fonts.body, x, y + 40, W - x - 24, WHITE, gap=8)
            # Progress through the slide.
            progress = (f + 1) / total
            draw.rectangle([x, H - 30, W - 24, H - 26], fill=(44, 44, 56))
            draw.rectangle([x, H - 30, x + int((W - 24 - x) * progress), H - 26], fill=YELLOW)
            yield canvas

    return make, total


# --- the code walkthrough ---------------------------------------------------

def code_slides(fonts):
    mouth = os.path.join(HERE, "mouth.py")
    net = os.path.join(HERE, "lipnet_np.py")
    h5 = os.path.join(HERE, "minih5.py")
    lip = os.path.join(HERE, "the_lip.py")
    slides = []
    slides.append(code_slide(fonts, "1. One call finds the eyes", "mouth.py", snippet(mouth, [
        ("def _eyes(self, frame):", "return None"),
        ("def locate(self, frame):", "ey = (left[1] + right[1]) / 2.0"),
    ]), [
        ("syrup.find_eyes(frame)", None,
         "find_eyes did not exist until this line named it. syrup parses the name - verb find, noun eye - plans it "
         "(the frontal-face cascade, then the eye cascade inside each face) and runs it through its C API."),
        ("if len(found) == 2:", "return np.array(a.centre)",
         "Two eyes, left to right. Fewer or more, and this frame yields nothing; it will borrow a neighbour's estimate."),
        ("d = float(np.linalg.norm", "ey = (left[1]",
         "The eye distance is the ruler: every threshold that follows is a fraction of it, so the same code works at any resolution."),
    ]))
    slides.append(code_slide(fonts, "2. The line between the lips", "mouth.py", snippet(mouth, [
        ("y0, y1 = int(ey + 0.95 * d)", "cy = y0 + int(candidates"),
        ("wx0, wx1 = max(int(cx - 0.8 * d)", "lx, rx = cx - 0.47 * d, cx + 0.47 * d"),
    ]), [
        ("y0, y1 = int(ey + 0.95 * d)", "band = gray[y0:y1, x0:x1]",
         "Search a band below the eyes: 0.95 to 1.5 eye distances down, 0.45 to each side."),
        ("skin = np.percentile(band, 80)", "dark_counts = np.convolve",
         "Per row, count the pixels clearly darker than the skin. A nostril shadow is dark but short; the lip line is dark and long."),
        ("candidates = np.where", "cy = y0 + int(candidates",
         "Among the rows within 90% of the longest dark run, take the darkest one: that row is the lip line."),
        ("while li > 0 and gap <= 2:", "lx, rx = offset + li + 1",
         "Walk left and right along it while the pixels stay dark, allowing gaps of two pixels: the corners of the mouth."),
        ("if width < 0.5 * d or width > 1.4 * d:", "lx, rx = cx - 0.47 * d",
         "An implausible width is not trusted: fall back to 0.47 eye distances each side."),
    ]))
    slides.append(code_slide(fonts, "3. LipNet's crop, exactly", "mouth.py", snippet(mouth, [
        ("if ratio is None:", "boxes.append((l / ratio"),
    ]), [
        ("if ratio is None:", "ratio = MOUTH_WIDTH / float",
         "The scale is fixed on the first frame: the mouth, padded 19% on each side, must span 100 pixels. This is LipNet's own preprocessing."),
        ("resized = cv2.resize(frame", "crop = resized[max(t, 0):b, max(l, 0):r]",
         "Resize the whole frame by that ratio and cut 100x50 around the mouth centre."),
        ("crops.append(crop.swapaxes(0, 1)", None,
         "Width-major, W x H x C, as the original code stores frames; values in [0, 1]. Feed the network anything else and it reads nothing."),
    ]))
    slides.append(code_slide(fonts, "4. The network, in numpy", "lipnet_np.py", snippet(net, [
        ("def features(self, frames):", "return x.reshape(x.shape[0], -1)"),
        ("def probabilities(self, features):", "return e / e.sum(axis=1, keepdims=True)"),
    ]), [
        ("for layer, pad, stride in zip(self.conv, pads, strides):", "x = max_pool_ab(x)",
         "Three 3-D convolutions over time, width and height, each followed by batch norm, ReLU and 2x2 spatial pooling: the paper's STCNN."),
        ("return x.reshape(x.shape[0], -1)", None,
         "Every frame becomes 1728 numbers describing the mouth."),
        ("for layer in self.gru:", "x = np.concatenate([fw, bw], axis=1)",
         "Two bidirectional GRUs read the sequence forwards and backwards; concatenated, 512 numbers per frame."),
        ("logits = x @ self.dense[0]", "return e / e.sum(axis=1, keepdims=True)",
         "A dense layer and a softmax: for every frame, probabilities over 26 letters, the space and the CTC blank."),
    ]))
    slides.append(code_slide(fonts, "5. The GRU as Keras 2.0.2 computed it", "lipnet_np.py", snippet(net, [
        ("def gru(x, kernel, recurrent, bias, reverse=False):", "return out[::-1] if reverse else out"),
    ]), [
        ("xw = x @ kernel + bias", "u_z, u_r, u_h = recurrent",
         "Input and recurrent weights for the three gates, in Keras' order: update z, reset r, candidate h."),
        ("z = hard_sigmoid(x_z[t] + h @ u_z)", "r = hard_sigmoid(x_r[t] + h @ u_r)",
         "Hard sigmoid - a clipped line, not the smooth curve - because that is what these weights were trained with."),
        ("hh = np.tanh(x_h[t] + (r * h) @ u_h)", None,
         "The reset gate multiplies the state before the recurrent product (Keras 2.0), not after it (Keras 2.1+, CuDNN). "
         "Get this wrong and the letters come out as noise; get it right and published weights give published accuracy."),
        ("h = z * h + (1.0 - z) * hh", "out[t] = h",
         "The new state. 75 steps of this per direction, per layer; the whole clip decodes in a third of a second."),
    ]))
    slides.append(code_slide(fonts, "6. The weights, without TensorFlow", "minih5.py", snippet(h5, [
        ("def _walk_btree(self, addr, heap):", "return result"),
        ("def _snod(self, addr, heap):", "return result"),
        ("raw = self.buf[layout[1]:layout[1] + count * dtype.itemsize]", "return np.frombuffer(raw, dtype=dtype, count=count).reshape(dims)"),
    ]), [
        ("def _walk_btree(self, addr, heap):", "result = {}",
         "An .h5 file is HDF5. Groups are B-trees whose leaves are symbol-table nodes; the reader walks them by byte address."),
        ("def _snod(self, addr, heap):", "result[self._heap_string(heap, entry",
         "Each node lists its children: a name in the local heap and the address of the child's object header."),
        ("raw = self.buf[layout[1]:layout[1] + count * dtype.itemsize]", "return np.frombuffer(raw, dtype=dtype, count=count).reshape(dims)",
         "A dataset's header says its dimensions, its datatype and where the bytes are; numpy reads them in place. "
         "Under 300 lines replace h5py for this file - the container was never the hard part."),
    ]))
    slides.append(code_slide(fonts, "7. From probabilities to words", "lipnet_np.py", snippet(net, [
        ("def greedy_decode(probabilities):", 'return "".join(text)'),
        ("def correction(self, word):", "return self.known([word]) or self.known(self.edits1(word))"),
    ]), [
        ("best = probabilities.argmax(axis=1)", 'return "".join(text)',
         "CTC best path: the most likely symbol per frame, repeats collapsed, blanks dropped. 'sset  bluue' reads 'set blue'."),
        ("def correction(self, word):", "return self.known([word]) or self.known(self.edits1(word))",
         "Norvig's corrector over the GRID dictionary snaps a near miss ('fiv') to the closest word, as the original code does."),
    ]))
    slides.append(code_slide(fonts, "8. Subtitles while the person speaks", "the_lip.py", snippet(lip, [
        ("def streaming_decodes(net, crops):", "return decodes"),
    ]), [
        ("full = net.features(crops)", None,
         "Convolution features are local in time, so they are computed once for the whole clip..."),
        ("tail = net.features(crops[start : t + 1])", "prefix[t + 1 - n :] = tail[len(tail) - n :]",
         "...and only the last three frames are recomputed, because they see the zero padding a clip that ended here would have."),
        ("probs = net.probabilities(prefix)", None,
         "The GRUs run over the prefix seen so far. The backward direction cannot peek at the future: there is none yet."),
        ("path = probs.argmax(axis=1)[: max(0, t + 1 - SETTLE_FRAMES)]", None,
         "Characters emitted in the newest 8 frames are the network's guess at how the sentence continues. Hold them back; show what has settled."),
    ]))
    return slides


# --- rendering ------------------------------------------------------------

def render(workdir, out, lipnet, quick=False):
    fonts = Fonts()
    with open(os.path.join(workdir, "session.json")) as f:
        session = json.load(f)
    thelip = os.path.join(workdir, "syrup", "python", "thelip")
    lipnet = lipnet or os.path.join(thelip, "LipNet")
    samples = sorted(__import__("glob").glob(os.path.join(lipnet, "evaluation", "samples", "GRID", "*.mpg")))
    samples += sorted(__import__("glob").glob(os.path.join(lipnet, "evaluation", "samples", "*.mpg")))
    if quick:
        samples = samples[:2]
    net = LipNet(os.path.join(lipnet, WEIGHTS))
    spell = Spell(os.path.join(lipnet, DICTIONARY))

    tl = Timeline()
    tl.add(*hold(card([
        ("THE LIP", fonts.title, YELLOW, 10),
        ("proof, then code", fonts.h1, WHITE, 40),
        ("Every clip plays twice. First muted: the subtitle is decoded from the lips as the person speaks. "
         "Then with its own sound, so you can hear whether it was right.", fonts.body, GREY, 18),
        ("After that: a user installing and running it from a fresh clone, recorded for real, and the code, step by step.",
         fonts.body, GREY, 40),
        ("Video: GRID corpus (Cooke et al., CC BY 4.0), via github.com/rizkiarm/LipNet (MIT).", fonts.fine, DIM, 0),
    ]), 7 * FPS))

    results = []
    for k, path in enumerate(samples):
        code = os.path.splitext(os.path.basename(path))[0]
        frames, _ = read_frames(path)
        crops, boxes, _ = mouth_crops(frames)
        decodes = [settled_words(spell, d) for d in streaming_decodes(net, crops)]
        final = spell.sentence(greedy_decode(net.predict(crops)))
        spoken = grid_sentence(code)
        audio = clip_audio(path)
        results.append((code, final, spoken))
        print(f"{code}: read {final!r}, spoken {spoken!r}", file=sys.stderr)
        make, count = clip_pair(fonts, frames, boxes, crops, decodes, final, spoken, audio, k + 1, len(samples))
        tl.add(make, count, pair_audio(audio, len(frames), 25))
    words = sum(len(s.split()) for _, _, s in results)
    right = sum(sum(1 for a, b in zip(f.split(), s.split()) if a == b) for _, f, s in results)
    exact = sum(1 for _, f, s in results if f == s)

    tl.add(*hold(card([
        ("USE IT NOW", fonts.title, YELLOW, 10),
        ("a fresh clone, recorded for real", fonts.h1, WHITE, 40),
        ("Nine commands. Nothing on the following screens is typed in after the fact: the output is what each command printed, "
         "and the video the sixth one wrote is played back as it came out.", fonts.body, GREY, 0),
    ]), 6 * FPS))
    for step in session:
        tl.add(*terminal(fonts, step))
        if step["command"].startswith("python3 the_lip.py"):
            written = os.path.join(thelip, "subtitled.mp4")
            tl.add(*play_file(fonts, written, "the file that command wrote, played back as it is"))

    tl.add(*hold(card([
        ("THE CODE", fonts.title, YELLOW, 10),
        ("four files, no framework", fonts.h1, WHITE, 40),
        ("mouth.py finds the mouth with syrup.  lipnet_np.py is the network in numpy.  minih5.py reads the weights.  "
         "the_lip.py turns it into subtitles.", fonts.body, GREY, 0),
    ]), 5 * FPS))
    for make, count in code_slides(fonts):
        tl.add(make, count if not quick else min(count, 60))

    tl.add(*hold(card([
        ("THE LIP", fonts.title, YELLOW, 10),
        (f"{right}/{words} words from the lips alone, over {len(results)} muted clips; {exact}/{len(results)} sentences exact",
         fonts.h2, GREEN, 30),
        ("Honest limits: LipNet is trained on the GRID vocabulary - 51 words in a fixed sentence shape. It reads that vocabulary well "
         "and will not read open English; that takes a larger model. The mouth localisation, the streaming decode and the overlay "
         "carry over unchanged.", fonts.body, GREY, 36),
        ("github.com/boggioMichael/syrup  -  python/thelip", fonts.h2, YELLOW, 0),
    ]), 8 * FPS))

    # Audio first (its length is known from the timeline), then the video through a pipe.
    wav = os.path.join(workdir, "proof-audio.wav")
    write_wav(wav, tl.audio())
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest",
           "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = tl.total
    t0 = time.time()
    for n, frame in enumerate(tl.frames()):
        proc.stdin.write(frame.tobytes())
        if n % 500 == 0:
            print(f"  frame {n}/{total} ({time.time() - t0:.0f}s)", file=sys.stderr)
    proc.stdin.close()
    proc.wait()
    print(f"wrote {out}: {total / FPS:.0f} s", file=sys.stderr)


def write_wav(path, samples):
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("record")
    r.add_argument("--repo", required=True, help="git URL or path of the syrup repository")
    r.add_argument("--workdir", default="proofwork")
    r.add_argument("--cargo-offline", action="store_true", help="build from cargo's cache, and say so on screen")
    v = sub.add_parser("render")
    v.add_argument("--workdir", default="proofwork")
    v.add_argument("--out", default="the-lip-proof.mp4")
    v.add_argument("--lipnet", default=None, help="LipNet checkout; default: the one the session cloned")
    v.add_argument("--quick", action="store_true", help="two clips and short slides, to check the layout")
    args = ap.parse_args()
    if args.mode == "record":
        record(args.repo, args.workdir, args.cargo_offline)
    else:
        render(args.workdir, args.out, args.lipnet, args.quick)


if __name__ == "__main__":
    main()
