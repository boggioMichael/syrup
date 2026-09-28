"""LipNet (Assael et al. 2016) as a lipreader backend, through the numpy
implementation in python/thelip (weights from github.com/rizkiarm/LipNet,
MIT; trained on GRID, CC BY 4.0). English, 51-word vocabulary, 25 fps,
clips of up to 75 frames.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import numpy as np

from ..decode import Decoded, best_path, correct
from ..schema import ModelInfo
from .base import ModelSpec, VisualSpeechModel

HERE = os.path.dirname(os.path.abspath(__file__))
THELIP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "python", "thelip"))
WEIGHTS = "evaluation/models/overlapped-weights368.h5"
DICTIONARY = "common/dictionaries/grid.txt"

INFO = ModelInfo(
    name="lipnet-grid",
    license="MIT (code and weights, github.com/rizkiarm/LipNet); GRID corpus CC BY 4.0",
    languages=("en",),
    vocabulary="GRID: 51 words in a fixed sentence shape (command colour preposition letter digit adverb)",
    note="reads GRID's vocabulary only; open English needs a larger model (see docs/models.md)",
)
HOW_TO_GET = "git clone --depth 1 https://github.com/rizkiarm/LipNet python/thelip/LipNet   (or set LIPNET_DIR)"


def lipnet_dir() -> Optional[str]:
    for candidate in (os.environ.get("LIPNET_DIR"), os.path.join(THELIP, "LipNet")):
        if candidate and os.path.exists(os.path.join(candidate, WEIGHTS)):
            return candidate
    return None


def spec() -> ModelSpec:
    found = lipnet_dir()
    return ModelSpec(
        key="lipnet-grid", info=INFO, native_fps=25.0, max_frames=75, crop="lipnet-100x50",
        available=found is not None,
        reason=f"weights at {found}" if found else "weights not found",
        how_to_get=HOW_TO_GET,
    )


class LipNetModel(VisualSpeechModel):
    def __init__(self, directory: Optional[str] = None):
        directory = directory or lipnet_dir()
        if directory is None:
            raise FileNotFoundError(f"LipNet weights not found; {HOW_TO_GET}")
        if THELIP not in sys.path:
            sys.path.insert(0, THELIP)
        from lipnet_np import BLANK, LETTERS, LipNet, Spell  # noqa: E402

        self.net = LipNet(os.path.join(directory, WEIGHTS))
        self.spell = Spell(os.path.join(directory, DICTIONARY))
        self.alphabet = list(LETTERS) + [" ", ""]  # 26 letters, space (26), blank (27)
        self.blank = BLANK
        self.space = 26
        self.spec = spec()

    def predict_batch(self, clips: Sequence[np.ndarray]) -> List[Decoded]:
        out = []
        for clip in clips:
            if len(clip) < 2:
                out.append(Decoded([]))
                continue
            probabilities = self.net.predict(np.asarray(clip, dtype=np.float32))
            decoded = best_path(probabilities, self.alphabet, self.blank, self.space)
            out.append(correct(decoded, self.spell.correction, known=lambda w: w in self.spell.dictionary))
        return out
