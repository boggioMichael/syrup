"""The explicit reader as one layer of templates: a line read column by column.

For every second column t of a line, the 15 x 14 window of grey pixels
around it (columns 2t-6 .. 2t+7, the teacher's own stride) is scored against
one template per class (blank, digits, marks); the best class per column, with
repeats merged and blanks dropped, is the reading (CTC's greedy decoding,
the same as the teacher's). That is the whole program: 17 templates of 210
weights and a bias each, about 3,600 numbers, and 17 dot products a column.

Its templates come from the teacher in one of two ways (PILOT-A.md):

- white box (`fit_to_logits`): the teacher's score for every class at every
  column is there to be read, so the templates are the least-squares fit of
  those scores from the windows (ridge regression, one solve).
- black box (`fit_to_strings`): only the teacher's strings, so the templates
  are trained by CTC against them, the way the teacher itself was trained.

`confidence` is the smallest margin, over the characters read, between the
best class and the runner-up at the column that emitted it; a reader refuses
a line whose confidence is below a level fitted on validation lines.
"""
from __future__ import annotations

import time

import numpy as np

import linenet

LEFT, RIGHT = 6, 8  # window: columns 2t-6 .. 2t+7
STEPS = linenet.WIDTH // 2


def features(lines):
    """(n, 108, 15*14 + 1): each column step's window of the grey line, and a 1."""
    x = linenet.to_input(lines)[:, :15].mean(-1)  # (n, 15, 216)
    pad = np.pad(x, ((0, 0), (0, 0), (LEFT, RIGHT)))
    win = np.lib.stride_tricks.sliding_window_view(pad, LEFT + RIGHT, axis=2)  # (n, 15, 216+1, 14)
    win = win[:, :, 0:2 * STEPS:2, :]  # every second column: (n, 15, 108, 14)
    f = win.transpose(0, 2, 1, 3).reshape(len(x), STEPS, -1)
    return np.concatenate([f, np.ones((len(x), STEPS, 1), np.float32)], axis=2).astype(np.float32)


class Linear:
    def __init__(self, w):
        self.w = w  # (211, 17)

    def logits(self, f):
        return f @ self.w

    def read(self, f):
        """Readings and confidences for features f (n, 108, 211)."""
        lg = self.logits(f)
        out = []
        for row in lg:
            path = row.argmax(-1)
            top2 = np.sort(row, axis=-1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            text, conf, prev = [], np.inf, linenet.BLANK
            for t, k in enumerate(path):
                if k != prev and k != linenet.BLANK:
                    text.append(linenet.ALPHABET[k - 1])
                    conf = min(conf, float(margin[t]))
                prev = k
            out.append(("".join(text), conf if text else -np.inf))
        return out

    def size(self):
        return int(self.w.size)


def fit_to_logits(f, teacher_logits, ridge=1.0):
    """White box: templates whose scores best match the teacher's, column by column."""
    x = f.reshape(-1, f.shape[-1]).astype(np.float64)
    y = teacher_logits.reshape(-1, teacher_logits.shape[-1]).astype(np.float64)
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    w = np.linalg.solve(gram, x.T @ y)
    return Linear(w.astype(np.float32))


def fit_to_strings(f, texts, epochs=30, lr=0.01, batch=64, seed=0, log=None):
    """Black box: the same templates trained by CTC on the teacher's strings."""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((f.shape[-1], linenet.NCLASS)) * 0.01).astype(np.float32)
    labels = [linenet.encode(t) for t in texts]
    m, v = np.zeros_like(w), np.zeros_like(w)
    step = 0
    n = len(f)
    total_steps = epochs * ((n + batch - 1) // batch)
    for epoch in range(epochs):
        order = rng.permutation(n)
        t0, total = time.time(), 0.0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            fb = f[idx]
            lg = fb @ w
            loss, grad, _ = linenet.ctc(lg, [labels[i] for i in idx])
            g = np.einsum("btf,btk->fk", fb, grad)
            step += 1
            rate = lr * 0.5 * (1 + np.cos(np.pi * step / total_steps))
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            w -= rate * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            total += loss * len(idx)
        if log is not None:
            log({"epoch": epoch + 1, "loss": round(total / n, 4), "seconds": round(time.time() - t0, 1)})
    return Linear(w)
