"""A small CNN over a whole text line, trained with CTC, in NumPy alone.

This is the line reader of amendment 1 (at most 100,000 parameters), and the
teacher the extraction pilot reads from the inside. NumPy is all this
environment has (no PyTorch: its wheels cannot be downloaded here), so the
layers, their gradients and CTC are written out below; `check` compares every
gradient with a numerical one.

Layout is NHWC. A line is 16 rows (the box's 15 and one copied row) by 216
columns, RGB:

    conv 3x3, 3->16, ReLU
    conv 3x3, 16->32, ReLU, max-pool 2x2      -> 8 x 108
    conv 3x3, 32->48, ReLU
    conv 8x3 (all rows), 48->64, ReLU          -> 1 x 108
    conv 1x1, 64->17                            -> 108 columns of 17 scores

The 17 classes are the CTC blank and the characters of a HUD line.
"""
from __future__ import annotations

import json
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ALPHABET = "0123456789,./[]%"
BLANK = 0
HEIGHT, WIDTH = 16, 216
NCLASS = len(ALPHABET) + 1


def encode(text: str) -> np.ndarray:
    return np.array([ALPHABET.index(c) + 1 for c in text], dtype=np.int64)


def decode_path(path) -> str:
    out, prev = [], BLANK
    for k in path:
        if k != prev and k != BLANK:
            out.append(ALPHABET[k - 1])
        prev = k
    return "".join(out)


# ------------------------------------------------------------------ layers

class Conv:
    """A convolution with zero padding; weights as (kh*kw*cin, cout)."""

    def __init__(self, cin, cout, kh, kw, ph, pw, rng):
        self.cin, self.cout, self.kh, self.kw, self.ph, self.pw = cin, cout, kh, kw, ph, pw
        fan_in = cin * kh * kw
        self.w = (rng.standard_normal((kh * kw * cin, cout)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def params(self):
        return [self.w, self.b]

    def forward(self, x, keep=True):
        b, h, w, c = x.shape
        xp = np.pad(x, ((0, 0), (self.ph, self.ph), (self.pw, self.pw), (0, 0)))
        win = sliding_window_view(xp, (self.kh, self.kw), axis=(1, 2))  # b, ho, wo, c, kh, kw
        ho, wo = win.shape[1], win.shape[2]
        cols = np.ascontiguousarray(win.transpose(0, 1, 2, 4, 5, 3)).reshape(b * ho * wo, -1)
        out = cols @ self.w + self.b
        if keep:
            self.cache = (cols, x.shape, ho, wo)
        return out.reshape(b, ho, wo, self.cout)

    def backward(self, dout):
        cols, shape, ho, wo = self.cache
        b, h, w, c = shape
        d = dout.reshape(-1, self.cout)
        self.dw = cols.T @ d
        self.db = d.sum(0)
        dcols = (d @ self.w.T).reshape(b, ho, wo, self.kh, self.kw, c)
        dxp = np.zeros((b, h + 2 * self.ph, w + 2 * self.pw, c), dout.dtype)
        for i in range(self.kh):
            for j in range(self.kw):
                dxp[:, i:i + ho, j:j + wo, :] += dcols[:, :, :, i, j, :]
        return dxp[:, self.ph:self.ph + h, self.pw:self.pw + w, :]

    def grads(self):
        return [self.dw, self.db]


class Relu:
    def forward(self, x, keep=True):
        if keep:
            self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, d):
        return d * self.mask

    def params(self):
        return []

    def grads(self):
        return []


class Pool2:
    def forward(self, x, keep=True):
        b, h, w, c = x.shape
        r = x.reshape(b, h // 2, 2, w // 2, 2, c)
        out = r.max(axis=(2, 4))
        if keep:
            self.mask = r == out[:, :, None, :, None, :]
            # one winner per window, so ties do not double the gradient
            first = np.cumsum(self.mask.reshape(b, h // 2, 2, w // 2, 2, c).transpose(0, 1, 3, 5, 2, 4)
                              .reshape(b, h // 2, w // 2, c, 4), axis=-1) == 1
            m = self.mask.transpose(0, 1, 3, 5, 2, 4).reshape(b, h // 2, w // 2, c, 4) & first
            self.mask = m.reshape(b, h // 2, w // 2, c, 2, 2).transpose(0, 1, 4, 2, 5, 3)
            self.shape = x.shape
        return out

    def backward(self, d):
        b, h, w, c = self.shape
        return (self.mask * d[:, :, None, :, None, :]).reshape(b, h, w, c)

    def params(self):
        return []

    def grads(self):
        return []


class LineNet:
    def __init__(self, seed=0, widths=(16, 32, 48, 64)):
        rng = np.random.default_rng(seed)
        c1, c2, c3, c4 = widths
        self.layers = [
            Conv(3, c1, 3, 3, 1, 1, rng), Relu(),
            Conv(c1, c2, 3, 3, 1, 1, rng), Relu(), Pool2(),
            Conv(c2, c3, 3, 3, 1, 1, rng), Relu(),
            Conv(c3, c4, HEIGHT // 2, 3, 0, 1, rng), Relu(),
            Conv(c4, NCLASS, 1, 1, 0, 0, rng),
        ]
        self.widths = widths

    def params(self):
        return [p for l in self.layers for p in l.params()]

    def grads(self):
        return [g for l in self.layers for g in l.grads()]

    def count(self):
        return int(sum(p.size for p in self.params()))

    def forward(self, x, keep=True, taps=None):
        """Scores (b, T, classes). `taps`, a list, receives every layer's output."""
        for layer in self.layers:
            x = layer.forward(x, keep)
            if taps is not None:
                taps.append(x)
        return x[:, 0, :, :]

    def backward(self, dlogits):
        d = dlogits[:, None, :, :]
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return d

    def save(self, path, meta=None):
        arrays = {f"p{i}": p for i, p in enumerate(self.params())}
        np.savez(path, widths=np.array(self.widths), meta=json.dumps(meta or {}), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        net = cls(widths=tuple(int(v) for v in z["widths"]))
        for i, p in enumerate(net.params()):
            p[...] = z[f"p{i}"]
        return net


# --------------------------------------------------------------------- CTC

def log_softmax(z):
    m = z.max(-1, keepdims=True)
    e = z - m
    return e - np.log(np.exp(e).sum(-1, keepdims=True))


def ctc(logits, labels):
    """Mean negative log likelihood over the batch, and its gradient.

    logits: (b, T, K); labels: list of int arrays (classes 1..K-1).
    """
    b, t_len, _ = logits.shape
    logp = log_softmax(logits.astype(np.float64))
    lmax = max(len(l) for l in labels)
    s_max = 2 * lmax + 1
    ext = np.zeros((b, s_max), np.int64)
    s_len = np.zeros(b, np.int64)
    for i, l in enumerate(labels):
        ext[i, 1:2 * len(l):2] = l
        s_len[i] = 2 * len(l) + 1
    valid = np.arange(s_max)[None, :] < s_len[:, None]
    skip = np.zeros((b, s_max), bool)
    skip[:, 2:] = (ext[:, 2:] != BLANK) & (ext[:, 2:] != ext[:, :-2])
    neg = -np.inf
    emit = np.take_along_axis(logp, ext[:, None, :].repeat(t_len, 1), axis=2)  # b, T, S
    alpha = np.full((t_len, b, s_max), neg)
    alpha[0, :, 0] = emit[:, 0, 0]
    alpha[0, :, 1] = np.where(s_len > 1, emit[:, 0, 1], neg)
    for t in range(1, t_len):
        a = alpha[t - 1]
        a1 = np.concatenate([np.full((b, 1), neg), a[:, :-1]], 1)
        a2 = np.concatenate([np.full((b, 2), neg), a[:, :-2]], 1)
        a2 = np.where(skip, a2, neg)
        alpha[t] = np.logaddexp(np.logaddexp(a, a1), a2) + emit[:, t]
        alpha[t][~valid] = neg
    beta = np.full((t_len, b, s_max), neg)
    rows = np.arange(b)
    beta[-1, rows, s_len - 1] = emit[rows, -1, s_len - 1]
    beta[-1, rows, s_len - 2] = emit[rows, -1, s_len - 2]
    skip_next = np.zeros((b, s_max), bool)
    skip_next[:, :-2] = skip[:, 2:]
    for t in range(t_len - 2, -1, -1):
        n = beta[t + 1]
        n1 = np.concatenate([n[:, 1:], np.full((b, 1), neg)], 1)
        n2 = np.concatenate([n[:, 2:], np.full((b, 2), neg)], 1)
        n2 = np.where(skip_next, n2, neg)
        beta[t] = np.logaddexp(np.logaddexp(n, n1), n2) + emit[:, t]
        beta[t][~valid] = neg
    ll = np.logaddexp(alpha[-1, rows, s_len - 1], alpha[-1, rows, s_len - 2])
    gamma = alpha + beta - emit.transpose(1, 0, 2) - ll[None, :, None]  # T, b, S
    occ = np.exp(gamma).transpose(1, 0, 2)  # b, T, S
    grad = np.exp(logp)
    for i in range(b):
        np.add.at(grad[i].T, ext[i, :s_len[i]], -occ[i, :, :s_len[i]].T)
    return float(-ll.mean()), (grad / b).astype(logits.dtype), -ll


def greedy(logits):
    return [decode_path(p) for p in logits.argmax(-1)]


def line_logprob(logits, texts):
    """log P(text | line) under the network, for each line (0 for an empty text)."""
    out = np.zeros(len(texts))
    idx = [i for i, t in enumerate(texts) if t]
    if idx:
        _, _, nll = ctc(logits[idx], [encode(texts[i]) for i in idx])
        out[idx] = -nll
    return out


# ---------------------------------------------------------------- training

class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= (self.lr * mh / (np.sqrt(vh) + self.eps)).astype(p.dtype)


def to_input(lines: np.ndarray) -> np.ndarray:
    """uint8 (n, 15, w, 3) lines -> float32 (n, 16, 216, 3): each line less
    its mean colour and over its spread, so that a red bar, a blue one and
    the dark band under them look alike to the first layer."""
    n, h, w, _ = lines.shape
    out = np.zeros((n, HEIGHT, WIDTH, 3), np.float32)
    x = lines[:, :, :WIDTH].astype(np.float32) / 255.0
    x -= x.mean(axis=(1, 2), keepdims=True)
    x /= x.std(axis=(1, 2, 3), keepdims=True) + 0.02
    out[:, :h, :x.shape[2]] = x
    out[:, h:, :x.shape[2]] = x[:, -1:]
    return out


def augment(x, rng):
    """Generic changes only: a shift along the line, light and contrast, noise."""
    b = x.shape[0]
    out = np.empty_like(x)
    shifts = rng.integers(-8, 9, b)
    for i in range(b):
        out[i] = np.roll(x[i], shifts[i], axis=1)
    gain = rng.uniform(0.8, 1.2, (b, 1, 1, 1)).astype(np.float32)
    bias = rng.uniform(-0.08, 0.08, (b, 1, 1, 3)).astype(np.float32)
    noise = rng.standard_normal(x.shape).astype(np.float32) * rng.uniform(0, 0.03, (b, 1, 1, 1)).astype(np.float32)
    return out * gain + bias + noise


def train(net, x, texts, epochs, rng, batch=32, lr=2e-3, log=None, val=None, schedule=None, opt=None):
    """`epochs` passes over (x, texts). With `schedule=(epoch, of)` this call is
    epoch `epoch` of `of` (data drawn anew each epoch): the learning rate
    follows the whole run, and the optimiser `opt` (returned) carries over."""
    opt = opt or Adam(net.params(), lr=lr)
    labels = [encode(t) for t in texts]
    n = len(x)
    per_epoch = (n + batch - 1) // batch
    first, of = schedule if schedule else (0, epochs)
    steps = of * per_epoch
    step = first * per_epoch
    for epoch in range(epochs):
        order = rng.permutation(n)
        t0, total = time.time(), 0.0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            # a short warm-up, the rate held, then a cosine over the last 40%
            frac = step / steps
            opt.lr = lr * min(1.0, (step + 1) / 200) * (1.0 if frac < 0.6 else 0.5 * (1 + np.cos(np.pi * (frac - 0.6) / 0.4)))
            logits = net.forward(augment(x[idx], rng))
            loss, grad, _ = ctc(logits, [labels[i] for i in idx])
            net.backward(grad)
            opt.step(net.grads())
            total += loss * len(idx)
            step += 1
        msg = {"epoch": first + epoch + 1, "loss": round(total / n, 4), "seconds": round(time.time() - t0, 1)}
        if val is not None:
            vx, vt = val
            got = predict(net, vx)
            msg["val_exact"] = round(float(np.mean([a == b for a, b in zip(got, vt)])), 4)
        print(json.dumps(msg), flush=True)
        if log:
            log.write(json.dumps(msg) + "\n")
            log.flush()
    return opt


def predict(net, x, batch=128):
    out = []
    for s in range(0, len(x), batch):
        out += greedy(net.forward(x[s:s + batch], keep=False))
    return out


def logits_of(net, x, batch=128):
    return np.concatenate([net.forward(x[s:s + batch], keep=False) for s in range(0, len(x), batch)])


# ------------------------------------------------------------------- check

def check():
    """Every gradient against a numerical one, on a tiny net and batch."""
    rng = np.random.default_rng(1)
    net = LineNet(seed=3, widths=(4, 4, 4, 6))
    for layer in net.layers:
        if isinstance(layer, Conv):
            layer.w = layer.w.astype(np.float64)
            layer.b = layer.b.astype(np.float64)
    x = rng.standard_normal((2, HEIGHT, WIDTH, 3)) * 0.5
    labels = [encode("12/12"), encode("3,4[5.6%]")]

    def loss_of():
        return ctc(net.forward(x, keep=False), labels)[0]

    logits = net.forward(x)
    loss, grad, _ = ctc(logits, labels)
    net.backward(grad)
    worst = 0.0
    for p, g in zip(net.params(), net.grads()):
        flat = p.reshape(-1)
        for k in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[k]
            eps = 1e-5
            flat[k] = old + eps
            up = loss_of()
            flat[k] = old - eps
            down = loss_of()
            flat[k] = old
            num = (up - down) / (2 * eps)
            ana = g.reshape(-1)[k]
            rel = abs(num - ana) / max(1e-3, abs(num) + abs(ana))
            worst = max(worst, float(rel))
    print(json.dumps({"loss": float(loss), "worst_relative_gradient_error": worst}))
    return worst


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["check"]:
        assert check() < 0.05
