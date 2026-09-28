"""Package a fine-tuned model for thelip-server (the auto_avsr backend) and,
with a Hugging Face token, publish it.

    models/<lang>-thelip-v<N>/model.pth        the averaged parameters (Auto-AVSR E2E, video)
    models/<lang>-thelip-v<N>/unigram.model    the tokenizer, and unigram_units.txt
    models/<lang>-thelip-v<N>/info.json        base model, data hours by source and licence, results
    models/<lang>-thelip-v<N>/README.md        the model card, honest numbers included
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Optional

from .common import DIRS, say
from .tokenizer import train_tokenizer


def export(language: str, run: str, averaged: str, results: dict, split: dict, version: int = 1, push_to: Optional[str] = None) -> str:
    name = f"{language}-thelip-v{version}"
    out = os.path.join(DIRS["models"], name)
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(averaged, os.path.join(out, "model.pth"))
    model, units = train_tokenizer(language)
    shutil.copyfile(model, os.path.join(out, "unigram.model"))
    shutil.copyfile(units, os.path.join(out, "unigram_units.txt"))
    info = {
        "name": name, "language": language, "run": run, "backend": "auto_avsr",
        "base": "vsr_trlrs2lrs3vox2avsp_base (Auto-AVSR, Ma et al. 2023; 3,448 h; 20.3% WER on LRS3)",
        "transfer": "whole model" if language == "en" else "front end + encoder; new decoder and CTC head",
        "data": split, "results": results,
        "licence": "the base model's terms (non-commercial) and each source's licence, listed in data.by_source",
    }
    with open(os.path.join(out, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card(info))
    say(f"exported {out}")
    if push_to:
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            api.create_repo(push_to, exist_ok=True, private=False)
            api.upload_folder(folder_path=out, repo_id=push_to)
            say(f"published to https://huggingface.co/{push_to}")
        except Exception as e:  # noqa: BLE001
            say(f"(not published to {push_to}: {e})")
    return out


def model_card(info: dict) -> str:
    rows = info["results"]
    lines = [f"# {info['name']}", "", f"Visual speech recognition (lip reading) for language `{info['language']}`, fine-tuned by",
             "[thelip](https://thelip.ai) from " + info["base"] + f" ({info['transfer']}).", "",
             "## Data (hours by source, with licence)", ""]
    for source, h in (info["data"].get("by_source") or {}).items():
        lines.append(f"- {source}: {h} h")
    lines += ["", "## Results (error rate on held-out clips; test = a whole held-out source plus every consented phone sample)", ""]
    for name, r in rows.items():
        lines.append(f"- **{name}**: {r['overall'] * 100:.1f}% {r['unit']} error rate over {r['n']} clips; " +
                     ", ".join(f"{s} {v['rate'] * 100:.1f}% (n={v['n']})" for s, v in r["by_source"].items()))
    lines += ["", "Lip reading is probabilistic: many sounds look the same on the lips. These numbers are measured, not promised.",
              "", f"Licence: {info['licence']}."]
    return "\n".join(lines) + "\n"
