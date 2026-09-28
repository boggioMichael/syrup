"""Which visual speech model reads which language, whether it can run on
this machine, and under what licence. The table is the honest one: a
language whose only models are non-commercial research checkpoints that
need PyTorch and a download this machine cannot make is listed as such,
and asking for it raises LanguageUnavailable with the reasons.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..schema import ModelInfo
from . import lipnet
from .base import LanguageUnavailable, ModelSpec, VisualSpeechModel

LANGUAGE_NAMES = {
    "en": "English", "he": "Hebrew", "es": "Spanish", "ar": "Arabic", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "el": "Greek", "ru": "Russian", "zh": "Mandarin Chinese",
}


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _mpc001_spec(language: str, wer: str) -> ModelSpec:
    """Ma, Petridis, Pantic — Visual Speech Recognition for Multiple Languages
    in the Wild (2022). Checkpoints on Google Drive; PyTorch + ESPnet code."""
    directory = os.environ.get("LIPREADER_MPC001_DIR", "")
    have_weights = bool(directory) and os.path.isdir(directory)
    return ModelSpec(
        key=f"mpc001-vsr-{language}",
        info=ModelInfo(
            name=f"mpc001/Visual_Speech_Recognition_for_Multiple_Languages ({language})",
            license="non-commercial: \"comparative or benchmarking purposes\" only (repository LICENSE)",
            languages=(language,), vocabulary="open", note=f"reported visual-only WER {wer}",
        ),
        native_fps=25.0, max_frames=600, crop="mouth-96",
        available=False,
        reason=("adapter present but the model is not integrated in this build: needs PyTorch and the ESPnet-based "
                "code from that repository" if have_weights else
                "not runnable here: needs PyTorch, the repository code, and a Google Drive download"
                + (" (torch not installed)" if not _has("torch") else "")),
        how_to_get="git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages; download the "
                   f"{language} model from its README (Google Drive); set LIPREADER_MPC001_DIR; non-commercial use only",
    )


def _muavic_spec(language: str) -> ModelSpec:
    """Meta MuAViC AVSR checkpoints (AV-HuBERT, fairseq), CC BY-NC 4.0."""
    directory = os.environ.get("LIPREADER_MUAVIC_DIR", "")
    have_weights = bool(directory) and os.path.isdir(directory)
    return ModelSpec(
        key=f"muavic-avhubert-{language}",
        info=ModelInfo(
            name=f"MuAViC AV-HuBERT AVSR ({language})",
            license="CC BY-NC 4.0 (facebookresearch/muavic)",
            languages=(language,), vocabulary="open",
            note="audio-visual checkpoint; video-only accuracy is not reported by the authors",
        ),
        native_fps=25.0, max_frames=500, crop="mouth-96",
        available=False,
        reason=("adapter present but the model is not integrated in this build: needs fairseq and the av_hubert code"
                if have_weights else
                "not runnable here: needs PyTorch + fairseq + av_hubert and a download from dl.fbaipublicfiles.com"),
        how_to_get=f"see https://github.com/facebookresearch/muavic#models ({language}_avsr checkpoint, dict, tokenizer); "
                   "set LIPREADER_MUAVIC_DIR; non-commercial use only",
    )


def _auto_avsr_spec() -> ModelSpec:
    directory = os.environ.get("LIPREADER_AUTO_AVSR_DIR", "")
    have_weights = bool(directory) and os.path.isdir(directory)
    return ModelSpec(
        key="auto-avsr-en",
        info=ModelInfo(
            name="Auto-AVSR VSR base (vsr_trlrs2lrs3vox2avsp_base)",
            license="Apache-2.0 code; weights carry the training data's terms (LRS2/LRS3: non-commercial research)",
            languages=("en",), vocabulary="open", note="reported visual-only WER 20.3% on LRS3; 250M parameters",
        ),
        native_fps=25.0, max_frames=600, crop="mouth-96",
        available=False,
        reason=("adapter present but the model is not integrated in this build: needs PyTorch and the auto_avsr code"
                if have_weights else "not runnable here: needs PyTorch, the repository code, and a Google Drive download"),
        how_to_get="git clone https://github.com/mpc001/auto_avsr; download the checkpoint from its README (Google Drive); "
                   "set LIPREADER_AUTO_AVSR_DIR",
    )


def candidates(language: str) -> List[ModelSpec]:
    """Every known visual model for a language, best first."""
    language = language.lower()
    table: Dict[str, List[ModelSpec]] = {
        "en": [lipnet.spec(), _auto_avsr_spec(), _mpc001_spec("en", "32.3% (LRS3)"), _muavic_spec("en")],
        "es": [_mpc001_spec("es", "44.5% (CMU-MOSEAS)"), _muavic_spec("es")],
        "fr": [_mpc001_spec("fr", "58.6% (CMU-MOSEAS)"), _muavic_spec("fr")],
        "pt": [_mpc001_spec("pt", "51.4% (CMU-MOSEAS)"), _muavic_spec("pt")],
        "it": [_muavic_spec("it")],
        "ar": [_muavic_spec("ar")],
        "de": [_muavic_spec("de")],
        "el": [_muavic_spec("el")],
        "ru": [_muavic_spec("ru")],
        "zh": [_mpc001_spec("zh", "8.0% CER (CMLR)")],
        "he": [],
    }
    return table.get(language, [])


@dataclass
class LanguageStatus:
    code: str
    name: str
    visual_available: bool
    visual_model: Optional[str]
    visual_license: Optional[str]
    visual_note: str

    def to_dict(self) -> dict:
        return {
            "code": self.code, "name": self.name,
            "visual": {"available": self.visual_available, "model": self.visual_model,
                       "license": self.visual_license, "note": self.visual_note},
        }


def languages() -> List[LanguageStatus]:
    out = []
    for code in ("en", "he", "es", "ar", "fr", "de", "it", "pt", "el", "ru", "zh"):
        specs = candidates(code)
        ready = next((s for s in specs if s.available), None)
        if ready:
            status = LanguageStatus(code, LANGUAGE_NAMES[code], True, ready.info.name, ready.info.license, ready.info.vocabulary)
        elif specs:
            first = specs[0]
            status = LanguageStatus(code, LANGUAGE_NAMES[code], False, first.info.name, first.info.license,
                                    f"{first.reason}. {first.how_to_get}")
        else:
            status = LanguageStatus(code, LANGUAGE_NAMES[code], False, None, None,
                                    "no public visual speech model or corpus is known for this language; audio modes only")
        out.append(status)
    return out


def available_languages() -> List[str]:
    return [s.code for s in languages() if s.visual_available]


def resolve(language: str) -> ModelSpec:
    """The model to use for a language, or LanguageUnavailable."""
    specs = candidates(language)
    for s in specs:
        if s.available:
            return s
    raise LanguageUnavailable(language, specs)


def load(spec: ModelSpec) -> VisualSpeechModel:
    if spec.key == "lipnet-grid":
        return lipnet.LipNetModel()
    raise LanguageUnavailable(spec.info.languages[0] if spec.info.languages else "?", [spec])
