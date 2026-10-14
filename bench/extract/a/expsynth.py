"""EXP lines made from the video's own glyphs, for a reader to learn to read.

    expsynth.py bank --lines L.npz --labels EXP-LABELS.json --out BANK.npz

The Zakum recording shows only 15 different EXP values, so a network trained
on its lines alone learns to recognise those 15 lines rather than to read
(the first bootstrap did exactly that). Here every labelled EXP line is cut
into its glyphs: the line's mean picture over its run is split at the
columns with no text, and when the pieces are as many as the label's
characters, every frame of the run is cut at the same columns. A glyph is
kept as its pixels less the band's colour behind it (its "ink"), so it can
be laid on any stretch of the band. `render` then writes any EXP line, a
real glyph from the bank for each character, on a real stretch of
background, and compresses it as the video was.

Glyphs are banked by the time-split of the run they come from (train,
validation, test), so a line made for testing uses no glyph a reader saw in
training. The bank is game graphics: it stays outside the repository.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

SPLITS = {"train": (0, 31), "val": (31, 47), "test": (47, 56)}  # segment index ranges
# How wide each character's piece can be on the video's lines, in columns.
WIDTHS = {",": (1, 3), ".": (1, 2), "[": (1, 3), "]": (1, 3), "%": (6, 8), "1": (1, 3)}
TEXT_LEFT = 0  # filled from the data: where real lines start
HEIGHT, WIDTH = 15, 210


def luminance(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def background(img):
    """The band's colour behind the text, row by row (the median of each row)."""
    return np.median(img, axis=1, keepdims=True)


def columns(mean, contrast=40):
    """The column spans with text on a line's mean picture."""
    lum = luminance(mean)
    ink = (lum - np.median(lum, axis=1, keepdims=True)) > contrast
    col = ink.any(0)
    spans, x = [], 0
    while x < len(col):
        if col[x]:
            start = x
            while x < len(col) and col[x]:
                x += 1
            spans.append((start, x))
        else:
            x += 1
    return spans


def bank_cmd(args):
    data = np.load(args.lines)
    exp = data["exp"]
    labels = json.load(open(args.labels))
    glyphs = {name: {} for name in SPLITS}
    backgrounds = {name: [] for name in SPLITS}
    starts, ends, kept, skipped = [], [], 0, 0
    for item in labels:
        seg, (a, b), text = item["segment"], item["frames"], item["label"]
        split = next(n for n, (lo, hi) in SPLITS.items() if lo <= seg < hi)
        frames = exp[a:b].astype(np.float32)
        spans = columns(frames.mean(0))
        # as many pieces as characters, and each piece as wide as its
        # character can be (a digit 4-6 columns, a mark 1-3, '%' 6-8): a
        # line whose pieces merely add up to the right count is cut wrong
        if len(spans) != len(text) or not all(
            WIDTHS.get(ch, (4, 6))[0] <= x1 - x0 <= WIDTHS.get(ch, (4, 6))[1]
            for (x0, x1), ch in zip(spans, text)
        ):
            skipped += 1
            continue
        kept += 1
        starts.append(spans[0][0])
        ends.append(spans[-1][1])
        for frame in frames:
            bg = background(frame)
            for (x0, x1), ch in zip(spans, text):
                # the glyph as signed opacity: white text over the band
                # (> 0) and its dark edge (< 0), so it can be laid on a band
                # of another brightness as the game would have drawn it
                ink = frame[:, x0:x1] - bg
                alpha = np.where(ink >= 0, ink / np.maximum(255 - bg, 1), ink / np.maximum(bg, 1))
                glyphs[split].setdefault(ch, []).append(np.clip(alpha, -1, 1).astype(np.float16))
            # the band left of the text, as background to write on
            backgrounds[split].append(frame[:, : max(8, spans[0][0] - 4)].astype(np.uint8))
    out = {"text_left": np.array(int(np.median(starts))), "text_right": np.array(int(np.median(ends)))}
    for split in SPLITS:
        for ch, items in glyphs[split].items():
            # an instance far from its class's mean picture is a bad cut (a
            # frame where something passed over the line): left out
            w0 = max(g.shape[1] for g in items)
            pad = np.zeros((len(items), HEIGHT, w0), np.float32)
            for i, g in enumerate(items):
                pad[i, :, : g.shape[1]] = g.astype(np.float32).mean(-1)
            mean = pad.mean(0).ravel()
            mean = (mean - mean.mean()) / (mean.std() + 1e-6)
            flat = pad.reshape(len(items), -1)
            flat = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-6)
            fit = flat @ mean / mean.size
            items = [g for g, f in zip(items, fit) if f > 0.6]
            # instances of one class can differ in width by a column: pad to the widest
            w = max(g.shape[1] for g in items)
            arr = np.zeros((len(items), HEIGHT, w, 3), np.float16)
            widths = np.zeros(len(items), np.int16)
            for i, g in enumerate(items):
                arr[i, :, : g.shape[1]] = g
                widths[i] = g.shape[1]
            key = f"{split}|{ord(ch)}"
            out[key] = arr
            out[key + "|w"] = widths
        out[f"{split}|bg"] = np.stack([np.resize(b, (HEIGHT, 48, 3)) if b.shape[1] < 48 else b[:, :48] for b in backgrounds[split]])
    np.savez_compressed(args.out, **out)
    print(json.dumps({"segments kept": kept, "skipped (pieces != characters)": skipped,
                      "text from column": int(out["text_left"]), "to": int(out["text_right"]),
                      "classes per split": {s: sorted(glyphs[s]) for s in SPLITS},
                      "instances": {s: sum(len(v) for v in glyphs[s].values()) for s in SPLITS}}))


class Bank:
    def __init__(self, path, split):
        z = np.load(path)
        self.left, self.right = int(z["text_left"]), int(z["text_right"])
        self.glyphs, self.widths = {}, {}
        for key in z.files:
            parts = key.split("|")
            if parts[0] != split or parts[1] == "bg":
                continue
            ch = chr(int(parts[1]))
            if len(parts) == 3:
                self.widths[ch] = z[key]
            else:
                self.glyphs[ch] = z[key].astype(np.float32)
        self.backgrounds = z[f"{split}|bg"]

    def has(self, text):
        return all(ch in self.glyphs for ch in text)


# The empty columns between two characters on the video's lines, by kind
# (d = a digit), and how often each width is seen there (measured on the
# labelled runs): mostly one or two, four before the bracket.
GAPS = {
    "dd": ([1, 2], [0.74, 0.26]), "d,": ([1, 2], [0.47, 0.53]), ",d": ([1, 2], [0.68, 0.32]),
    "d.": ([1, 2], [0.85, 0.15]), ".d": ([1, 2], [0.38, 0.62]), "d[": ([3, 4, 5], [0.1, 0.8, 0.1]),
    "[d": ([1, 2, 3], [0.5, 0.47, 0.03]), "d%": ([1, 2], [0.3, 0.7]), "%]": ([1, 2], [0.3, 0.7]),
}


def gap(a, b, rng):
    kind = ("d" if a.isdigit() else a) + ("d" if b.isdigit() else b)
    widths, p = GAPS.get(kind, ([1, 2], [0.6, 0.4]))
    return int(rng.choice(widths, p=p))


def random_exp_text(rng):
    digits = int(rng.integers(1, 13))
    value = int(rng.integers(10 ** (digits - 1), 10 ** digits)) if digits > 1 else int(rng.integers(0, 10))
    decimals = 3 if rng.random() < 0.6 else 2
    pct = rng.random() * 100
    return f"{value:,}[{pct:.{decimals}f}%]"


def render(bank, text, rng, compress=True, where=None):
    """One EXP line (15 x 210 RGB, uint8) reading `text`, from banked glyphs.
    `where`, a list, receives each character's (first column, width)."""
    import cv2
    # background: stretches of real band side by side, some mirrored
    pieces, width = [], 0
    while width < WIDTH:
        piece = bank.backgrounds[rng.integers(len(bank.backgrounds))].astype(np.float32)
        if rng.random() < 0.5:
            piece = piece[:, ::-1]
        pieces.append(piece)
        width += piece.shape[1]
    line = np.concatenate(pieces, axis=1)[:, :WIDTH]
    # glyphs, one real instance each, a column apart as in the game
    inks = []
    for ch in text:
        i = rng.integers(len(bank.glyphs[ch]))
        w = int(bank.widths[ch][i])
        inks.append(bank.glyphs[ch][i][:, :w])
    gaps = [gap(text[i], text[i + 1], rng) for i in range(len(text) - 1)] + [0]
    total = sum(g.shape[1] for g in inks) + sum(gaps)
    centre = (bank.left + bank.right) / 2 + rng.integers(-6, 7)
    x = int(round(centre - total / 2))
    x = max(0, min(WIDTH - total, x))
    for a, g in zip(inks, gaps):
        w = a.shape[1]
        bg = line[:, x:x + w]
        line[:, x:x + w] = np.where(a >= 0, bg + a * (255 - bg), bg * (1 + a))
        if where is not None:
            where.append((x, w))
        x += w + g
    line = np.clip(line, 0, 255).astype(np.uint8)
    if compress:
        if rng.random() < 0.3:
            line = cv2.GaussianBlur(line, (3, 3), rng.uniform(0.3, 0.7))
        quality = int(rng.integers(55, 96))
        ok, buf = cv2.imencode(".jpg", line[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
        line = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    return line


def make(bank, n, rng, texts=None, places=None):
    """n made lines and their texts; `places`, a list, receives each line's
    characters' (first column, width)."""
    lines, out = [], []
    while len(lines) < n:
        text = texts[len(lines) % len(texts)] if texts else random_exp_text(rng)
        if not bank.has(text):
            continue
        where = []
        lines.append(render(bank, text, rng, where=where))
        out.append(text)
        if places is not None:
            places.append(where)
    return np.stack(lines), out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--lines", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--out", required=True)
    args = p.parse_args()
    {"bank": bank_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
