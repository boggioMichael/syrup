"""From the teacher to explicit code.

The explicit reader both arms produce (`Reader`) has two steps, both plain:

1. Placement. A line is a run of columns, each either background or part of
   one glyph. A glyph of class c covers w_c columns and costs its squared
   distance from c's mean picture; a background column costs its distance
   from the band's colour. The cheapest cover of the line, found by dynamic
   programming over the columns, places the glyphs: touching glyphs need no
   cutting first.
2. Reading. Each placed glyph is read again by one linear template per class
   (a weight for every pixel of the 15 x 9 window round its centre): the
   best-scoring class of its kind (a digit stays a digit, a mark a mark). The
   smallest margin between the best and the second class over the line's
   glyphs is the line's confidence; below a level fitted on validation lines,
   the line is refused.

What both steps need is where each character is on the lines to learn from,
and what it is. The two arms of PILOT-A.md get that differently:

- white box (`align_white`): the teacher's per-column posteriors say which
  character it read where it peaks, and the teacher's gradient there (which
  pixels move that score) says which columns are the glyph.
- black box (`align_black`): only the lines and the teacher's strings. The
  ink of each line is cut at its emptiest columns into as many pieces as the
  string has characters; mean pictures are made; every line is placed again
  against its own string by the dynamic programming of step 1; and again,
  until the pictures stop moving.
- oracle (`align_truth`, for made lines only): the places the renderer used.
  No reader can have them; they bound what the other two can reach.
"""
from __future__ import annotations

import numpy as np

import linenet

ROWS = 15
HW = 4  # the reading window: 2*HW+1 columns round a glyph's centre
DIGITS = [linenet.ALPHABET.index(d) + 1 for d in "0123456789"]
MARKS = [linenet.ALPHABET.index(d) + 1 for d in ",.[]%/"]


def gray(lines):
    """Lines as the teacher sees them (`linenet.to_input`), made grey: (n, 15, 216)."""
    x = linenet.to_input(lines)
    return x[:, :ROWS].mean(-1)


def band(g):
    """The band's colour, a column of 15 values: the median of all columns."""
    return np.median(g.transpose(1, 0, 2).reshape(ROWS, -1), axis=1)


# ------------------------------------------------------------- alignment

def align_truth(texts, places):
    return [[(linenet.ALPHABET.index(ch) + 1, x, w) for ch, (x, w) in zip(t, p)] for t, p in zip(texts, places)]


def spikes(logits):
    """Where the teacher's best path emits each character: (class, step)."""
    path = logits.argmax(-1)
    out, t = [], 0
    while t < len(path):
        k = path[t]
        if k == linenet.BLANK:
            t += 1
            continue
        start = t
        while t < len(path) and path[t] == k:
            t += 1
        out.append((int(k), (start + t - 1) // 2))
    return out


def align_white(net, lines, per_class=60, seed=0):
    """Where the teacher reads each character, and which columns it looks at.

    For each class, the gradient of the teacher's score at its peak with
    respect to the line (over up to `per_class` peaks) gives the columns that
    move the score: the glyph's columns, as an offset and a width from the
    peak's column."""
    x = linenet.to_input(lines)
    logits = linenet.logits_of(net, x)
    sp = [spikes(l) for l in logits]
    rng = np.random.default_rng(seed)
    weight, lines_of = {}, {}
    for i, s in enumerate(sp):
        for j, (k, t) in enumerate(s):
            lines_of.setdefault(k, []).append((i, j))
    for k, items in lines_of.items():
        pick = [items[q] for q in rng.permutation(len(items))[:per_class]]
        idx = np.array([i for i, _ in pick])
        lg = net.forward(x[idx])
        d = np.zeros_like(lg)
        for b, (i, j) in enumerate(pick):
            d[b, sp[i][j][1], k] = 1.0
        grad = np.abs(net.backward(d))[:, :ROWS].sum((1, 3))  # (n, 216)
        # centred on each peak's column (step t covers columns 2t, 2t+1)
        cols = np.stack([np.roll(gr, -2 * sp[i][j][1] + 12)[:26] for gr, (i, j) in zip(grad, pick)])
        weight[k] = cols.mean(0)  # columns 2t-12 .. 2t+13
    shape = {}
    for k, w in weight.items():
        peak = int(np.argmax(w))
        on = w > 0.35 * w[peak]
        a = b = peak
        while a > 0 and on[a - 1]:
            a -= 1
        while b < len(w) - 1 and on[b + 1]:
            b += 1
        shape[k] = (a - 12, b - a + 1)  # offset from column 2t, and width
    placed = [[(k, 2 * t + shape[k][0], shape[k][1]) for k, t in s] for s in sp]
    return placed, logits, shape


def cover(gline, templates, background, penalty=2.0, forced=None):
    """Step 1, the placement: [(class, first column)] of the cheapest cover."""
    width = gline.shape[1]
    bg = ((gline - background[:, None]) ** 2).sum(0)
    classes = list(templates)
    w_of = {k: templates[k].shape[1] for k in classes}
    cost = {}
    for k in classes:
        win = np.lib.stride_tricks.sliding_window_view(gline, (ROWS, w_of[k]))[0]
        c = ((win - templates[k][None]) ** 2).sum((1, 2))
        full = np.full(width, np.inf)
        full[: len(c)] = c
        cost[k] = full
    if forced is None:
        best = np.full(width + 1, np.inf)
        best[0] = 0.0
        back = [None] * (width + 1)
        for x in range(1, width + 1):
            v, arg = best[x - 1] + bg[x - 1], (None, x - 1)
            for k in classes:
                s = x - w_of[k]
                if s >= 0:
                    c = best[s] + cost[k][s] + penalty
                    if c < v:
                        v, arg = c, (k, s)
            best[x], back[x] = v, arg
        out, x = [], width
        while x > 0:
            k, s = back[x]
            if k is not None:
                out.append((k, s))
            x = s
        return out[::-1]
    n = len(forced)
    best = np.full((n + 1, width + 1), np.inf)
    best[0, 0] = 0.0
    back = np.zeros((n + 1, width + 1, 2), np.int64)
    for x in range(1, width + 1):
        best[:, x] = best[:, x - 1] + bg[x - 1]
        back[:, x, 0] = np.arange(n + 1)
        back[:, x, 1] = x - 1
        for i in range(1, n + 1):
            k = forced[i - 1]
            s = x - w_of[k]
            if s >= 0:
                c = best[i - 1, s] + cost[k][s] + penalty
                if c < best[i, x]:
                    best[i, x] = c
                    back[i, x] = (i - 1, s)
    out, i, x = [], n, width
    while x > 0:
        pi, s = back[i, x]
        if pi == i - 1:
            out.append((forced[i - 1], int(s)))
        i, x = int(pi), int(s)
    return out[::-1]


def ink_runs(gline, background, level=0.2):
    """The runs of columns that stand out from the band: [(first, width)]."""
    d = np.sqrt(((gline - background[:, None]) ** 2).mean(0))
    on = d > level * d.max()
    runs, x = [], 0
    while x < len(on):
        if on[x]:
            a = x
            while x < len(on) and on[x]:
                x += 1
            runs.append((a, x - a))
        else:
            x += 1
    return runs


SPAN = 6  # windows of 2*SPAN+1 columns round a placed character's centre


def pictures(g, placed, background):
    """Each class's picture and width, from the windows round its placed
    characters: the columns that stand out from the band the same way every
    time (far from the band, little spread) are the glyph's."""
    pad = np.pad(g, ((0, 0), (0, 0), (SPAN, SPAN + 1)))
    per = {}
    for i, pl in enumerate(placed):
        for k, x, w in pl:
            c = x + w // 2
            if 0 <= c < g.shape[2]:
                per.setdefault(k, []).append(pad[i, :, c:c + 2 * SPAN + 1])
    templates, offset = {}, {}
    for k, items in per.items():
        win = np.stack(items)
        far = np.sqrt(((win.mean(0) - background[:, None]) ** 2).mean(0))
        score = far / (np.sqrt(win.var(0).mean(0)) + 0.05)
        peak = SPAN if score[SPAN] > 0.5 * score.max() else int(np.argmax(score))
        on = score > 0.5 * score[peak]
        a = b = peak
        while a > 0 and on[a - 1]:
            a -= 1
        while b < 2 * SPAN and on[b + 1]:
            b += 1
        templates[k] = win[:, :, a:b + 1].mean(0)
        offset[k] = a - SPAN
    return templates, offset


def refine(lines, labels, placed, rounds=6, widths=None):
    """Mean pictures from the placements (at each class's width), the
    placements again from the pictures (every line against its own string),
    until nothing moves."""
    g = gray(lines)
    background = band(g)
    rounds_run = 0
    for _ in range(rounds):
        rounds_run += 1
        templates = mean_pictures(g, [p for p in placed if p], widths)
        moved = 0
        for i, (gl, lab) in enumerate(zip(g, labels)):
            if not lab or any(k not in templates for k in lab):
                continue
            out = cover(gl, templates, background, forced=lab)
            if len(out) != len(lab):
                continue
            new = [(k, s, templates[k].shape[1]) for k, s in out]
            if not placed[i] or any(a[1] != b[1] for a, b in zip(new, placed[i])):
                moved += 1
            placed[i] = new
        if moved == 0:
            break
    return placed, rounds_run


def align_black(lines, texts, rounds=6):
    """Where each character is, found from the lines and the strings alone.

    Start: the lines whose ink falls into exactly as many runs as the string
    has characters are cut there (the usual way to find glyphs without a
    model); then `refine`."""
    g = gray(lines)
    background = band(g)
    labels = [list(linenet.encode(t)) for t in texts]
    placed = []
    for gl, lab in zip(g, labels):
        runs = ink_runs(gl, background)
        placed.append([(k, a, w) for k, (a, w) in zip(lab, runs)] if len(runs) == len(lab) else [])
    started = sum(1 for p in placed if p)
    # each class's width: the most common width of its pieces on those lines
    seen = {}
    for p in placed:
        for k, _, w in p:
            seen.setdefault(k, []).append(w)
    widths = {k: int(np.bincount(v).argmax()) for k, v in seen.items()}
    placed, rounds_run = refine(lines, labels, placed, rounds, widths)
    return placed, {"started_from_lines": started, "rounds": rounds_run, "widths": widths}


def align_white_refined(net, lines, rounds=6):
    """The white box's start (the teacher's own places), then `refine`."""
    placed, logits, shape = align_white(net, lines)
    labels = [[k for k, _ in spikes(l)] for l in logits]
    placed, rounds_run = refine(lines, labels, placed, rounds, {k: w for k, (_, w) in shape.items()})
    return placed, {"rounds": rounds_run, "teacher_strings": ["".join(linenet.ALPHABET[k - 1] for k in lab) for lab in labels]}


# ------------------------------------------------------------- the reader

def mean_pictures(g, placed, widths=None):
    """Each class's mean picture: at `widths[k]` when given, else at its
    most common width among the placements."""
    per = {}
    for gl, pl in zip(g, placed):
        for k, x, w in pl:
            w = widths.get(k, w) if widths else w
            if 0 <= x and x + w <= gl.shape[1]:
                per.setdefault(k, []).append(gl[:, x:x + w])
    out = {}
    for k, items in per.items():
        w = int(np.bincount([it.shape[1] for it in items]).argmax())
        out[k] = np.mean([it for it in items if it.shape[1] == w], 0)
    return out


def windows(gline, centres):
    pad = np.pad(gline, ((0, 0), (HW, HW + 1)))
    if not centres:
        return np.zeros((0, ROWS * (2 * HW + 1)))
    return np.stack([pad[:, c:c + 2 * HW + 1].ravel() for c in centres])


def train_templates(g, placed, steps=400, lr=0.05, decay=1e-3):
    """Step 2's linear templates: multinomial logistic regression on the
    windows round the placed glyphs' centres."""
    xs, ys = [], []
    for gl, pl in zip(g, placed):
        ok = [(k, x, w) for k, x, w in pl if 0 <= x + w // 2 < gl.shape[1]]
        xs.append(windows(gl, [x + w // 2 for _, x, w in ok]))
        ys += [k for k, _, _ in ok]
    x = np.concatenate(xs).astype(np.float32)
    x = np.concatenate([x, np.ones((len(x), 1), np.float32)], 1)
    y = np.array(ys)
    w = np.zeros((x.shape[1], linenet.NCLASS), np.float32)
    m, v = np.zeros_like(w), np.zeros_like(w)
    onehot = np.eye(linenet.NCLASS, dtype=np.float32)[y]
    for step in range(1, steps + 1):
        z = x @ w
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        gw = x.T @ (p - onehot) / len(x) + decay * w
        m = 0.9 * m + 0.1 * gw
        v = 0.999 * v + 0.001 * gw * gw
        w -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
    return w


class Reader:
    def __init__(self, templates, background, weights, penalty=2.0, min_margin=-np.inf):
        self.templates, self.background, self.weights = templates, background, weights
        self.penalty, self.min_margin = penalty, min_margin

    def read(self, gline):
        """(text, confidence)."""
        placed = cover(gline, self.templates, self.background, self.penalty)
        if not placed:
            return "", -np.inf
        centres = [x + self.templates[k].shape[1] // 2 for k, x in placed]
        f = windows(gline, centres)
        scores = np.concatenate([f, np.ones((len(f), 1))], 1) @ self.weights
        text, conf = [], np.inf
        for (k, _), sc in zip(placed, scores):
            group = DIGITS if k in DIGITS else MARKS
            s = sc[group]
            order = np.argsort(-s)
            text.append(linenet.ALPHABET[group[order[0]] - 1])
            conf = min(conf, float(s[order[0]] - s[order[1]]))
        return "".join(text), conf

    def size(self):
        return int(sum(t.size for t in self.templates.values()) + self.weights.size + self.background.size)


def build(lines, placed, penalty=2.0):
    g = gray(lines)
    return Reader(mean_pictures(g, placed), band(g), train_templates(g, placed), penalty)
