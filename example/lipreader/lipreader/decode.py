"""From per-frame character probabilities to timed, scored words.

CTC best-path decoding, kept honest: every word remembers which frames
emitted its letters (its timing) and how sure the network was at those
frames (its confidence). A dictionary correction may replace a word; the
replaced word keeps its raw form and loses some confidence, because a
correction is a guess about a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class DecodedWord:
    text: str
    start_frame: int
    end_frame: int  # inclusive
    confidence: float
    raw: Optional[str] = None


@dataclass
class Decoded:
    words: List[DecodedWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def best_path(probabilities: np.ndarray, alphabet: Sequence[str], blank: int, space: Optional[int]) -> Decoded:
    """Greedy CTC: argmax per frame, repeats collapsed, blanks dropped.
    `alphabet[i]` is the symbol for class i; `space` is the class that
    separates words (None when the alphabet has word pieces)."""
    best = probabilities.argmax(axis=1)
    peak = probabilities.max(axis=1)
    words: List[DecodedWord] = []
    letters: List[str] = []
    frames: List[int] = []
    confidences: List[float] = []
    previous = -1

    def flush():
        if letters:
            words.append(DecodedWord("".join(letters), frames[0], frames[-1], float(np.mean(confidences))))
        letters.clear()
        frames.clear()
        confidences.clear()

    for t, c in enumerate(best):
        if c != previous and c != blank:
            if space is not None and c == space:
                flush()
            else:
                letters.append(alphabet[c])
                frames.append(t)
                confidences.append(float(peak[t]))
        previous = c
    flush()
    return Decoded(words)


def correct(decoded: Decoded, corrector: Callable[[str], str], known: Optional[Callable[[str], bool]] = None,
            penalty: float = 0.8, unknown_penalty: float = 0.5) -> Decoded:
    """Apply a word-level dictionary correction, remembering the raw word. A
    word the dictionary still does not know after correction (a closed
    vocabulary model emitting letters that spell nothing) loses more."""
    out = []
    for w in decoded.words:
        fixed = corrector(w.text)
        confidence = w.confidence
        if fixed != w.text:
            confidence *= penalty
        if known is not None and not known(fixed):
            confidence *= unknown_penalty
        if fixed != w.text or confidence != w.confidence:
            out.append(DecodedWord(fixed, w.start_frame, w.end_frame, confidence, raw=w.text if fixed != w.text else None))
        else:
            out.append(w)
    return Decoded(out)
