"""The visual speech recogniser interface. A model reads a batch of mouth
clips and returns timed, scored words for each; everything else in
lipreader is the same whatever the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..decode import Decoded
from ..schema import ModelInfo


@dataclass
class ModelSpec:
    """What a model is, whether it can run here, and how to get it."""

    key: str
    info: ModelInfo
    native_fps: float
    max_frames: int            # longest clip the model should see at once
    crop: str                  # "lipnet-100x50" | "mouth-96" ...
    available: bool
    reason: str = ""           # why not available, or how it was found
    how_to_get: str = ""       # download/licence instructions, shown to users


class VisualSpeechModel:
    spec: ModelSpec

    def predict_batch(self, clips: Sequence[np.ndarray]) -> List[Decoded]:
        """clips: each (T, ...) in the layout `spec.crop` names; one Decoded per clip."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class LanguageUnavailable(Exception):
    """No visual speech model for the requested language can run here; the
    message says which models exist, under what licence, and what is needed."""

    def __init__(self, language: str, candidates: Sequence[ModelSpec]):
        self.language = language
        self.candidates = list(candidates)
        if candidates:
            lines = [f"no visual speech model for {language!r} can run here:"]
            for c in candidates:
                lines.append(f"  - {c.info.name} ({c.info.license}): {c.reason or 'not available'}")
                if c.how_to_get:
                    lines.append(f"      {c.how_to_get}")
        else:
            lines = [f"no public visual speech model or corpus for {language!r} is known; "
                     "the language can only be read through the audio modes"]
        super().__init__("\n".join(lines))
