"""LipNet (Assael et al. 2016) inference in numpy, from the pretrained Keras
weights of github.com/rizkiarm/LipNet (MIT), read without h5py.

Architecture (model2.py of that repository):
  3 x [ZeroPad3D -> Conv3D -> BatchNorm -> ReLU -> MaxPool(1,2,2)]
  -> TimeDistributed(Flatten) -> BiGRU(256) -> BiGRU(256) -> Dense(28) -> softmax
Input: (T, 100, 50, 3) mouth crops (width-major, as the original code stores
them), values in [0, 1]. Output: per-frame character probabilities, 26
letters + space (26) + CTC blank (27).

Keras 2.0.2 semantics matter and are reproduced exactly: GRU gates in the
order z, r, h with hard-sigmoid recurrent activation and the reset gate
applied before the recurrent matmul; BatchNorm with epsilon 1e-3.
"""
import re
import string
from collections import Counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from minih5 import H5

LETTERS = "abcdefghijklmnopqrstuvwxyz"
BLANK = 27


def hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0.0, 1.0)


def conv3d(x, kernel, bias, pad, stride):
    """x: (T, A, B, C); kernel: (kt, ka, kb, C, F); 'valid' after zero padding."""
    pt, pa, pb = pad
    xp = np.pad(x, ((pt, pt), (pa, pa), (pb, pb), (0, 0)))
    kt, ka, kb, cin, cout = kernel.shape
    windows = sliding_window_view(xp, (kt, ka, kb), axis=(0, 1, 2))  # (T', A', B', C, kt, ka, kb)
    st, sa, sb = stride
    windows = windows[::st, ::sa, ::sb]
    out = np.tensordot(windows, kernel, axes=((3, 4, 5, 6), (3, 0, 1, 2)))
    return out + bias


def batch_norm(x, gamma, beta, mean, var, eps=1e-3):
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def max_pool_ab(x):
    """MaxPool3D(1, 2, 2) over the two spatial axes."""
    t, a, b, c = x.shape
    a2, b2 = a // 2, b // 2
    x = x[:, : a2 * 2, : b2 * 2]
    return x.reshape(t, a2, 2, b2, 2, c).max(axis=(2, 4))


def gru(x, kernel, recurrent, bias, reverse=False):
    """Keras 2.0 GRU over a sequence x: (T, D) -> (T, units)."""
    units = recurrent.shape[0]
    if reverse:
        x = x[::-1]
    xw = x @ kernel + bias  # (T, 3u): z, r, h
    x_z, x_r, x_h = xw[:, :units], xw[:, units:2 * units], xw[:, 2 * units:]
    u_z, u_r, u_h = recurrent[:, :units], recurrent[:, units:2 * units], recurrent[:, 2 * units:]
    h = np.zeros(units, dtype=np.float32)
    out = np.empty((x.shape[0], units), dtype=np.float32)
    for t in range(x.shape[0]):
        z = hard_sigmoid(x_z[t] + h @ u_z)
        r = hard_sigmoid(x_r[t] + h @ u_r)
        hh = np.tanh(x_h[t] + (r * h) @ u_h)
        h = z * h + (1.0 - z) * hh
        out[t] = h
    return out[::-1] if reverse else out


class LipNet:
    def __init__(self, weights_path):
        w = H5(weights_path).arrays()
        g = lambda k: np.ascontiguousarray(w[k]).astype(np.float32)
        self.conv = []
        for i in (1, 2, 3):
            self.conv.append(
                dict(
                    kernel=g(f"/conv{i}/conv{i}/kernel:0"),
                    bias=g(f"/conv{i}/conv{i}/bias:0"),
                    gamma=g(f"/batc{i}/batc{i}/gamma:0"),
                    beta=g(f"/batc{i}/batc{i}/beta:0"),
                    mean=g(f"/batc{i}/batc{i}/moving_mean:0"),
                    var=g(f"/batc{i}/batc{i}/moving_variance:0"),
                )
            )
        self.gru = []
        for i in (1, 2):
            p = f"/bidirectional_{i}/bidirectional_{i}/"
            self.gru.append(
                dict(
                    fw=(g(p + "kernel:0"), g(p + "recurrent_kernel:0"), g(p + "bias:0")),
                    bw=(g(p + "kernel_1:0"), g(p + "recurrent_kernel_1:0"), g(p + "bias_1:0")),
                )
            )
        self.dense = (g("/dense1/dense1/kernel:0"), g("/dense1/dense1/bias:0"))

    def features(self, frames):
        """Per-frame 1728-d features from mouth crops (T, 100, 50, 3) in [0, 1]."""
        x = frames.astype(np.float32)
        pads = [(1, 2, 2), (1, 2, 2), (1, 1, 1)]
        strides = [(1, 2, 2), (1, 1, 1), (1, 1, 1)]
        for layer, pad, stride in zip(self.conv, pads, strides):
            x = conv3d(x, layer["kernel"], layer["bias"], pad, stride)
            x = batch_norm(x, layer["gamma"], layer["beta"], layer["mean"], layer["var"])
            x = np.maximum(x, 0.0)
            x = max_pool_ab(x)
        return x.reshape(x.shape[0], -1)

    def probabilities(self, features):
        """Per-frame character probabilities (T, 28) from features (T, 1728)."""
        x = features
        for layer in self.gru:
            fw = gru(x, *layer["fw"])
            bw = gru(x, *layer["bw"], reverse=True)
            x = np.concatenate([fw, bw], axis=1)
        logits = x @ self.dense[0] + self.dense[1]
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, frames):
        return self.probabilities(self.features(frames))


def greedy_decode(probabilities):
    """Best-path CTC decoding: collapse repeats, drop blanks."""
    best = probabilities.argmax(axis=1)
    text = []
    previous = -1
    for c in best:
        if c != previous and c != BLANK:
            text.append(" " if c == 26 else LETTERS[c])
        previous = c
    return "".join(text)


class Spell:
    """Norvig's corrector over the GRID dictionary, as the original code uses
    (lipnet/utils/spell.py, MIT)."""

    def __init__(self, path):
        self.dictionary = Counter(list(string.punctuation) + re.findall(r"\w+", open(path).read().lower()))
        self.total = sum(self.dictionary.values())

    def correction(self, word):
        return max(self.candidates(word), key=lambda w: self.dictionary[w] / self.total)

    def candidates(self, word):
        return self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word]

    def known(self, words):
        return set(w for w in words if w in self.dictionary)

    def edits1(self, word):
        letters = LETTERS
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def sentence(self, text):
        return " ".join(self.correction(w) for w in text.split())


GRID_WORDS = {
    0: {"b": "bin", "l": "lay", "p": "place", "s": "set"},
    1: {"b": "blue", "g": "green", "r": "red", "w": "white"},
    2: {"a": "at", "b": "by", "i": "in", "w": "with"},
    4: {"z": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"},
    5: {"a": "again", "n": "now", "p": "please", "s": "soon"},
}


def grid_sentence(code):
    """The sentence a GRID file name encodes: 'sbwe5n' -> 'set blue with e five now'."""
    code = code.split("_")[-1]
    return " ".join(
        GRID_WORDS[i][c] if i in GRID_WORDS else c
        for i, c in enumerate(code[:6])
    )
