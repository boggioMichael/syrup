"""The languages thelip-server can read, and where their models come from.

Each entry names a visual speech model in the format Chaplin's pipeline
loads (ESPnet E2E: model.json + model.pth, optionally a language model),
its published quality, its licence and how it is fetched. `status` on an
entry means no runnable model exists yet, and says why.

Quality numbers are the authors', on their test sets, in the lab; a phone
is harder. All model weights here are for non-commercial use.
"""
from __future__ import annotations

from typing import Dict, Optional

HF = "https://huggingface.co"

LANGUAGES: Dict[str, dict] = {
    "en": {
        "name": "English", "native": "English",
        "model": "LRS3/models/LRS3_V_WER19.1", "lm": "LRS3/language_models/lm_en_subword",
        "quality": "19.1% word error rate on LRS3 (TED talks)",
        "credit": "Ma, Petridis & Pantic 2022/2023 (Auto-AVSR); weights re-hosted on Hugging Face by Amanvir Parhar",
        "licence": "non-commercial (comparative and benchmarking use)",
        "beam": 20, "ctc_weight": 0.1, "lm_weight": 0.3, "penalty": 0.0,
        "files": {
            "model.json": (f"{HF}/Amanvir/LRS3_V_WER19.1/resolve/main/model.json", 1_000),
            "model.pth": (f"{HF}/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth", 900_000_000),
            "lm/model.json": (f"{HF}/Amanvir/lm_en_subword/resolve/main/model.json", 500),
            "lm/model.pth": (f"{HF}/Amanvir/lm_en_subword/resolve/main/model.pth", 150_000_000),
        },
    },
    "es": {
        "name": "Spanish", "native": "Español",
        "model": "CMUMOSEAS/models/es/CMUMOSEAS_V_ES_WER44.5", "lm": "CMUMOSEAS/language_models/es/lm_es",
        "quality": "44.5% word error rate on CMU-MOSEAS",
        "credit": "Ma, Petridis & Pantic 2022 (VSR for Multiple Languages)",
        "licence": "non-commercial (comparative and benchmarking use)",
        "beam": 30, "ctc_weight": 0.1, "lm_weight": 0.4, "penalty": 0.0,
        "drive": {"model": "https://bit.ly/34MjWBW", "lm": "https://bit.ly/3rppyJN"},
    },
    "fr": {
        "name": "French", "native": "Français",
        "model": "CMUMOSEAS/models/fr/CMUMOSEAS_V_FR_WER58.6", "lm": "CMUMOSEAS/language_models/fr/lm_fr",
        "quality": "58.6% word error rate on CMU-MOSEAS",
        "credit": "Ma, Petridis & Pantic 2022 (VSR for Multiple Languages)",
        "licence": "non-commercial (comparative and benchmarking use)",
        "beam": 30, "ctc_weight": 0.1, "lm_weight": 0.4, "penalty": 0.0,
        "drive": {"model": "https://bit.ly/3Ik6owb", "lm": "https://bit.ly/3LDChSn"},
    },
    "pt": {
        "name": "Portuguese", "native": "Português",
        "model": "CMUMOSEAS/models/pt/CMUMOSEAS_V_PT_WER51.4", "lm": "CMUMOSEAS/language_models/pt/lm_pt",
        "quality": "51.4% word error rate on CMU-MOSEAS",
        "credit": "Ma, Petridis & Pantic 2022 (VSR for Multiple Languages)",
        "licence": "non-commercial (comparative and benchmarking use)",
        "beam": 30, "ctc_weight": 0.1, "lm_weight": 0.4, "penalty": 0.0,
        "drive": {"model": "https://bit.ly/3HjXCgo", "lm": "https://bit.ly/3gPvneF"},
    },
    "zh": {
        "name": "Mandarin Chinese", "native": "普通话",
        "model": "CMLR/models/CMLR_V_WER8.0", "lm": "CMLR/language_models/lm_zh",
        "quality": "8.0% character error rate on CMLR (news readers)",
        "credit": "Ma, Petridis & Pantic 2022 (VSR for Multiple Languages)",
        "licence": "non-commercial (comparative and benchmarking use)",
        "beam": 20, "ctc_weight": 0.1, "lm_weight": 0.3, "penalty": 0.3,
        "drive": {"model": "https://bit.ly/3fR8RkU", "lm": "https://bit.ly/3fPxXAJ"},
    },
    "he": {
        "name": "Hebrew", "native": "עברית",
        "status": "no public visual speech model or corpus exists for Hebrew; the first one is being trained from open-licence Hebrew video (example/thelip-train)",
    },
    "ar": {
        "name": "Arabic", "native": "العربية",
        "status": "research models exist (MuAViC, Meta AI, CC BY-NC) on the AV-HuBERT/fairseq stack, not yet wired into this server",
    },
    "de": {
        "name": "German", "native": "Deutsch",
        "status": "research models exist (MuAViC, Meta AI, CC BY-NC) on the AV-HuBERT/fairseq stack, not yet wired into this server",
    },
}

ORDER = ["en", "he", "es", "ar", "zh", "fr", "de", "pt"]


def runnable(code: str) -> bool:
    return "model" in LANGUAGES.get(code, {})


def describe(code: str, root: Optional[str] = None) -> dict:
    """One entry for /health: what it is and whether its files are present."""
    import os

    spec = LANGUAGES[code]
    out = {"code": code, "name": spec["name"], "native": spec["native"]}
    if not runnable(code):
        out.update(available=False, status=spec["status"])
        return out
    present = root is None or all(
        os.path.isfile(os.path.join(root, "benchmarks", spec["model"], "model.pth")) and
        os.path.isfile(os.path.join(root, "benchmarks", spec["model"], "model.json")) for _ in [0])
    out.update(available=present, quality=spec["quality"], credit=spec["credit"], licence=spec["licence"])
    if not present:
        out["status"] = "model files not downloaded on this server (run.py --languages)"
    return out


def trained_models(models_dir: Optional[str]) -> Dict[str, dict]:
    """Exports of example/thelip-train under models_dir: one per language, the
    highest version; `preferred` when its info.json shows it beat the
    published model on the phone samples (a language with no published model
    is always served by its trained one)."""
    import glob
    import json
    import os

    out: Dict[str, dict] = {}
    if not models_dir or not os.path.isdir(models_dir):
        return out
    for folder in sorted(glob.glob(os.path.join(models_dir, "*-thelip-v*"))):
        info_path = os.path.join(folder, "info.json")
        if not os.path.isfile(info_path) or not os.path.isfile(os.path.join(folder, "model.pth")):
            continue
        try:
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        code = info.get("language")
        if code not in LANGUAGES:
            continue
        results = info.get("results") or {}
        after = (results.get("v1") or {}).get("by_source", {}).get("phone", {}).get("rate")
        before = (results.get("base") or {}).get("by_source", {}).get("phone", {}).get("rate")
        overall = (results.get("v1") or {}).get("overall")
        preferred = not runnable(code) or (after is not None and before is not None and after < before)
        summary = ""
        if overall is not None:
            summary = f"{overall * 100:.1f}% {(results.get('v1') or {}).get('unit', 'word')} error rate on held-out clips"
            if after is not None and before is not None:
                summary += f"; on phone samples {after * 100:.1f}% (published model: {before * 100:.1f}%)"
        version = int(folder.rsplit("-v", 1)[-1]) if folder.rsplit("-v", 1)[-1].isdigit() else 0
        if code not in out or version > out[code]["version"]:
            out[code] = {"name": os.path.basename(folder), "folder": folder, "version": version, "preferred": preferred, "summary": summary}
    return out
