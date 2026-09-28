"""Label lists in Auto-AVSR's format, split without leakage, per language.

    manifests/<lang>/{train,val,test}.csv    dataset,rel_path,frames,token ids
    manifests/<lang>/split.json              which videos went where, and why

Auto-AVSR reads `root_dir/labels/<file>` with rows `dataset_name,rel_path,
input_length,token_id` and the video at `root_dir/dataset_name/rel_path`;
here dataset_name is the source and rel_path the clip inside it, so
root_dir is the clips folder, and `labels` there is a link to the manifests.

Splits are by video (never by clip): every consented phone sample is test
(that is what matters), one whole source is held out for test when there
are several, val is a few percent of the rest by video.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional

from .common import DIRS, jsonl_read, say


class Tokenizer:
    """SentencePiece pieces -> Auto-AVSR unit ids (0 blank, 1 <unk>, pieces from 2)."""

    def __init__(self, model_path: str, units_path: str):
        import sentencepiece

        self.sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
        self.units: Dict[str, int] = {}
        with open(units_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    self.units[parts[0]] = int(parts[1])

    def ids(self, text: str) -> List[int]:
        return [self.units.get(piece, self.units.get("<unk>", 1)) for piece in self.sp.encode(text, out_type=str)]


def all_clips(language: str) -> List[dict]:
    rows = []
    for name in sorted(os.listdir(DIRS["clips"])):
        if name.endswith(".jsonl"):
            rows += [r for r in jsonl_read(os.path.join(DIRS["clips"], name)) if r.get("language") == language]
    return rows


def split(rows: List[dict], holdout_source: Optional[str] = None, val_fraction: float = 0.03, seed: int = 7) -> Dict[str, List[dict]]:
    sources = sorted({r["source"] for r in rows if r["source"] != "phone"})
    if holdout_source is None and len(sources) > 1:
        # The smallest of the non-phone sources is held out whole: a different
        # kind of video than the training ones, the honest test.
        by_seconds = sorted(sources, key=lambda s: sum(r["seconds"] for r in rows if r["source"] == s))
        holdout_source = by_seconds[0]
    rng = random.Random(seed)
    videos = sorted({(r["source"], r["video"]) for r in rows if r["source"] not in ("phone", holdout_source)})
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * val_fraction)) if videos else 0
    val_videos = set(videos[:n_val])
    parts = {"train": [], "val": [], "test": []}
    for r in rows:
        key = (r["source"], r["video"])
        if r["source"] == "phone" or r["source"] == holdout_source:
            parts["test"].append(r)
        elif key in val_videos:
            parts["val"].append(r)
        else:
            parts["train"].append(r)
    parts["_holdout"] = holdout_source
    return parts


def write_manifests(language: str, tokenizer: Tokenizer, holdout_source: Optional[str] = None) -> dict:
    rows = all_clips(language)
    if not rows:
        raise SystemExit(f"no clips for {language}; run the segment stage first")
    parts = split(rows, holdout_source)
    out_dir = os.path.join(DIRS["manifests"], language)
    os.makedirs(out_dir, exist_ok=True)
    summary = {"language": language, "holdout": parts["_holdout"], "counts": {}, "hours": {}}
    for name in ("train", "val", "test"):
        path = os.path.join(out_dir, f"{name}.csv")
        with open(path, "w", encoding="utf-8") as f:
            for r in parts[name]:
                ids = tokenizer.ids(r["text"])
                if not ids:
                    continue
                source, rel = r["path"].split("/", 1)
                f.write(f"{source},{rel},{r['frames']},{' '.join(str(i) for i in ids)}\n")
        summary["counts"][name] = len(parts[name])
        summary["hours"][name] = round(sum(r["seconds"] for r in parts[name]) / 3600, 2)
    summary["by_source"] = {s: round(sum(r["seconds"] for r in rows if r["source"] == s) / 3600, 2) for s in sorted({r["source"] for r in rows})}
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    # Auto-AVSR wants root_dir/labels/<file>: link the manifests in.
    labels_link = os.path.join(DIRS["clips"], "labels")
    if not os.path.exists(labels_link):
        os.symlink(DIRS["manifests"], labels_link)
    say(f"{language}: {summary['counts']} clips, {summary['hours']} hours; test holds out {parts['_holdout']} and every phone sample")
    return summary


def texts_for_tokenizer(language: str) -> str:
    """One line per training clip text: the corpus a tokenizer is trained on."""
    rows = all_clips(language)
    parts = split(rows)
    path = os.path.join(DIRS["spm"], language, "input.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in parts["train"]:
            f.write(r["text"] + "\n")
    return path
