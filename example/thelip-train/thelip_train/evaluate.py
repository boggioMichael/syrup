"""Word (or character) error rate of a checkpoint on a manifest, per source,
with Auto-AVSR's own model code and test transform, so a number here means
the same thing as the authors' numbers. Used before and after fine-tuning,
and on the phone samples on their own.

    exp/<lang>/results.json     {run: {"overall": .., "by_source": {..}, "n": ..}}
"""
from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from typing import Dict, List, Optional

from .common import DIRS, error_rate, say
from .train import prepare_copy


def _module(language: str, checkpoint: str, beam: int = 20):
    """Auto-AVSR's ModelModule with the checkpoint loaded, in the language's copy."""
    copy = prepare_copy(language)
    if copy not in sys.path:
        sys.path.insert(0, copy)
    os.chdir(copy)
    import torch
    from lightning import ModelModule  # noqa: E402  (auto_avsr's)

    module = ModelModule(Namespace(modality="video", ctc_weight=0.1, pretrained_model_path=None))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = {k[len("model."):]: v for k, v in state["state_dict"].items() if k.startswith("model.")}
    module.model.load_state_dict(state)
    module.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    module.to(device)
    from lightning import get_beam_search_decoder  # noqa: E402

    module.beam_search = get_beam_search_decoder(module.model, module.token_list, beam_size=beam).to(device).eval()
    return module, device


def read_manifest(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            source, rel, frames, ids = line.rstrip("\n").split(",")
            rows.append({"source": source, "rel": rel, "frames": int(frames), "ids": [int(i) for i in ids.split()]})
    return rows


def evaluate(language: str, checkpoint: str, manifest: Optional[str] = None, beam: int = 20, limit: Optional[int] = None,
             unit: Optional[str] = None) -> dict:
    """Runs the evaluation in a process of its own: each language has its own
    copy of Auto-AVSR and its own tokenizer, and Python would keep the first
    copy's modules for the second language."""
    import subprocess

    out = os.path.join(DIRS["exp"], language, f"eval-{os.path.basename(checkpoint)}.json")
    cmd = [sys.executable, "-m", "thelip_train.evaluate", language, checkpoint, "--out", out, "--beam", str(beam)]
    if manifest:
        cmd += ["--manifest", manifest]
    if limit:
        cmd += ["--limit", str(limit)]
    if unit:
        cmd += ["--unit", unit]
    env = dict(os.environ, THELIP_TRAIN_HOME=os.path.dirname(DIRS["exp"]), PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0 or not os.path.isfile(out):
        raise SystemExit(f"evaluation of {checkpoint} failed (exit {proc.returncode})")
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def evaluate_here(language: str, checkpoint: str, manifest: Optional[str] = None, beam: int = 20, limit: Optional[int] = None,
                  unit: Optional[str] = None) -> dict:
    """The evaluation itself, in this process (see evaluate)."""
    import torch
    import torchvision

    manifest = manifest or os.path.join(DIRS["manifests"], language, "test.csv")
    rows = read_manifest(manifest)[:limit]
    module, device = _module(language, checkpoint, beam)
    from datamodule.transforms import VideoTransform  # noqa: E402  (auto_avsr's)

    transform = VideoTransform("test")
    unit = unit or ("char" if language == "zh" else "word")
    pairs: Dict[str, list] = {}
    say(f"{language}: evaluating {os.path.basename(checkpoint)} on {len(rows)} clips of {os.path.basename(manifest)}")
    with torch.no_grad():
        for i, r in enumerate(rows):
            path = os.path.join(DIRS["clips"], r["source"], r["rel"])
            video = torchvision.io.read_video(path, pts_unit="sec", output_format="THWC")[0].permute(0, 3, 1, 2)
            video = transform(video).to(device)
            hyp = module(video)
            ref = module.text_transform.post_process(torch.tensor(r["ids"]))
            pairs.setdefault(r["source"], []).append((ref, hyp))
            if i < 5 or i % 200 == 0:
                say(f"    {r['source']}/{r['rel']}: ref {ref!r} -> {hyp!r}")
    result = {"checkpoint": checkpoint, "manifest": manifest, "unit": unit, "n": len(rows),
              "overall": round(error_rate([p for ps in pairs.values() for p in ps], unit), 4),
              "by_source": {s: {"n": len(ps), "rate": round(error_rate(ps, unit), 4)} for s, ps in pairs.items()}}
    say(f"{language}: {unit} error rate {result['overall'] * 100:.1f}% overall; " + ", ".join(f"{s} {v['rate'] * 100:.1f}% (n={v['n']})" for s, v in result["by_source"].items()))
    return result


def record(language: str, name: str, result: dict) -> None:
    path = os.path.join(DIRS["exp"], language, "results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data[name] = result
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("language")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--unit", default=None)
    a = ap.parse_args()
    result = evaluate_here(a.language, a.checkpoint, a.manifest, a.beam, a.limit, a.unit)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
