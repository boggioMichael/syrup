"""The whole run, under a budget, resumable.

    python run_all.py --languages en,he --hours 40 --hourly-cost 3.5 \
        --video-hours en=150,he=80 --phone-data /path/to/thelip-server/data

Stages, each skipped when state.json says it is done:
  tools      auto_avsr and Chaplin (pinned), the base checkpoint, the face model
  fetch      video per source, to the hours asked for, licences kept
  label      transcripts with word times (captions, else Whisper)
  segment    stable faces, utterances, 96x96 mouth crops at 25 fps
  phone      thelip-server's consented, corrected samples as test clips
  manifest   tokenizer (a new one for a new language), train/val/test lists
  baseline   the base model's error rate on the test set (English only: no base exists for a new language)
  train      fine-tuning, wall clock capped to what the budget leaves
  evaluate   the fine-tuned model on the same test set
  export     the model folder thelip-server loads (+ Hugging Face with --push)

Every message carries the hours used and the money they cost at the rate
given. When the hours run out, the run stops between stages and says where
it got to; run again to continue.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from thelip_train.common import DIRS, WORK, Budget, ensure_dirs, load_state, save_state, say  # noqa: E402
from thelip_train.sources import SOURCES  # noqa: E402

AUTO_AVSR = ("https://github.com/mpc001/auto_avsr.git", "182b62837773ab01052d4ac21ef1d2203ea7d267")
CHAPLIN = ("https://github.com/amanvirparhar/chaplin.git", "7aee1f8fca776ce4f63690063310b53573b7d804")

PLAN = {   # sources per language, and the share of the language's video hours each gets
    "en": [("nasa", 0.25), ("whitehouse", 0.25), ("youtube-cc-en", 0.2), ("mit-ocw", 0.15), ("wikimedia-en", 0.15)],
    "he": [("my-videos-he", 0.0), ("youtube-cc-he", 0.8), ("wikimedia-he", 0.2)],
    "es": [("youtube-cc-es", 1.0)], "fr": [("youtube-cc-fr", 1.0)], "de": [("youtube-cc-de", 1.0)], "ar": [("youtube-cc-ar", 1.0)],
}
TRAINING = {  # epochs, learning rate: adapt the whole English model gently; a new language learns a new decoder
    "en": {"epochs": 8, "lr": 2e-4}, "*": {"epochs": 30, "lr": 5e-4},
}
ESTIMATE_HOURS = {"fetch": 0.5, "label": 0.15, "segment": 0.25, "train_min": 2.0, "evaluate": 0.3}   # per language-hour of video where it applies


def clone(url: str, commit: str, dest: str) -> None:
    """A pinned commit of a GitHub repository: git when there is one, else the
    commit's zip archive (a Windows PC often has no git on its PATH)."""
    import shutil

    if os.path.isdir(os.path.join(dest, ".git")) or os.path.isfile(os.path.join(dest, ".pinned")):
        return
    if shutil.which("git"):
        subprocess.check_call(["git", "clone", "-q", url, dest])
        subprocess.check_call(["git", "-C", dest, "checkout", "-q", commit])
        return
    import io
    import urllib.request
    import zipfile

    archive = url[: -len(".git")] + f"/archive/{commit}.zip"
    say(f"fetching {archive}")
    req = urllib.request.Request(archive, headers={"User-Agent": "thelip-train/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    parent = os.path.dirname(dest)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = z.namelist()[0].split("/")[0]
        z.extractall(parent)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.move(os.path.join(parent, root), dest)
    with open(os.path.join(dest, ".pinned"), "w") as f:
        f.write(commit + "\n")


def stage(state: dict, key: str) -> bool:
    return bool(state.get("done", {}).get(key))


def mark(state: dict, key: str, value=True) -> None:
    state.setdefault("done", {})[key] = value
    save_state(state)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--languages", default="en,he")
    ap.add_argument("--hours", type=float, default=40, help="wall-clock budget for this machine")
    ap.add_argument("--hourly-cost", type=float, default=3.5, help="what the machine costs, for the messages")
    ap.add_argument("--video-hours", default="en=150,he=80", help="hours of video to fetch per language")
    ap.add_argument("--phone-data", default=None, help="thelip-server's data/ folder (consented samples)")
    ap.add_argument("--push", default=None, help="Hugging Face repo prefix to publish to, e.g. boggioMichael/thelip")
    ap.add_argument("--only", default=None, help="run one stage only (tools, fetch, label, segment, phone, manifest, baseline, train, evaluate, export)")
    ap.add_argument("--smoke", action="store_true", help="a few videos, one epoch, a small evaluation: every stage once, quickly, before the real run")
    ap.add_argument("--max-frames", type=int, default=None, help="frames per training batch (default THELIP_MAX_FRAMES or 1600; about 640 on a 12 GB card)")
    ap.add_argument("--serve-dir", default=None, help="thelip-server's models/ folder: the exported model is copied there")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.smoke:
        args.video_hours = ",".join(f"{p.split('=')[0]}=0.3" for p in args.video_hours.split(",") if p)
        say("smoke run: 0.3 h of video per language, at most 12 items per source, one epoch, 40 clips evaluated")

    ensure_dirs()
    state = load_state()
    budget = Budget(args.hours, args.hourly_cost, started=state.get("started"))
    state["started"] = budget.started
    save_state(state)
    languages = [l.strip() for l in args.languages.split(",") if l.strip()]
    video_hours = {k: float(v) for k, v in (p.split("=") for p in args.video_hours.split(",") if p)}
    say(f"thelip-train: languages {languages}, budget {args.hours} h at ${args.hourly_cost}/h, work in {WORK} {budget.line()}")

    def want(name: str) -> bool:
        return args.only is None or args.only == name

    # ---- tools ----------------------------------------------------------------------------------
    if want("tools") and not stage(state, "tools"):
        clone(*AUTO_AVSR, os.path.join(DIRS["tools"], "auto_avsr"))
        clone(*CHAPLIN, os.path.join(DIRS["tools"], "chaplin"))
        from thelip_train.train import base_checkpoint

        base_checkpoint()
        from thelip_train.segment import make_landmarker

        make_landmarker()
        mark(state, "tools")
        say(f"tools ready {budget.line()}")

    # ---- fetch, label, segment per language -----------------------------------------------------
    from thelip_train.fetch import fetch_source
    from thelip_train.label import label_source
    from thelip_train.segment import make_landmarker, phone_samples, segment_source

    landmarker = None
    for lang in languages:
        hours_wanted = video_hours.get(lang, 50)
        plan = PLAN.get(lang, [])
        if want("fetch") and not stage(state, f"fetch:{lang}"):
            if not args.smoke and not budget.allows(ESTIMATE_HOURS["fetch"] * hours_wanted / 10):
                say(f"stopping before fetch:{lang}: budget {budget.line()}")
                return
            got = {name: fetch_source(name, hours_wanted * share, max_items=12 if args.smoke else 400) for name, share in plan}
            state.setdefault("hours", {})[lang] = got
            mark(state, f"fetch:{lang}")
            say(f"fetch:{lang} done {got} {budget.line()}")
        if want("label") and not stage(state, f"label:{lang}"):
            for name, _ in plan:
                label_source(name, lang)
            mark(state, f"label:{lang}")
            say(f"label:{lang} done {budget.line()}")
        if want("segment") and not stage(state, f"segment:{lang}"):
            landmarker = landmarker or make_landmarker()
            clips = {name: segment_source(name, landmarker) for name, _ in plan}
            state.setdefault("clips", {})[lang] = clips
            mark(state, f"segment:{lang}")
            say(f"segment:{lang} done {clips} {budget.line()}")
        if want("phone") and args.phone_data and not stage(state, f"phone:{lang}"):
            phone_samples(args.phone_data, lang)
            mark(state, f"phone:{lang}")

    # ---- manifests, baseline, training, evaluation, export ---------------------------------------
    from thelip_train.evaluate import evaluate, record
    from thelip_train.export import export
    from thelip_train.manifest import write_manifests
    from thelip_train.tokenizer import tokenizer_for
    from thelip_train.train import base_checkpoint, train

    for lang in languages:
        if want("manifest") and not stage(state, f"manifest:{lang}"):
            summary = write_manifests(lang, tokenizer_for(lang))
            state.setdefault("split", {})[lang] = summary
            mark(state, f"manifest:{lang}")
        split = state.get("split", {}).get(lang, {})
        if want("baseline") and lang == "en" and not stage(state, f"baseline:{lang}"):
            result = evaluate(lang, base_checkpoint(), limit=40 if args.smoke else 2000)
            record(lang, "base", result)
            mark(state, f"baseline:{lang}", result["overall"])
            say(f"baseline:{lang}: {result['overall'] * 100:.1f}% {result['unit']} error rate before any training {budget.line()}")
        if want("train") and not stage(state, f"train:{lang}"):
            left = budget.left_hours - ESTIMATE_HOURS["evaluate"] - 0.5
            if left < ESTIMATE_HOURS["train_min"] and not args.smoke:
                say(f"stopping before train:{lang}: only {left:.1f} h left {budget.line()}")
                return
            cfg = TRAINING.get(lang, TRAINING["*"])
            per_lang = left / max(1, len([l for l in languages if not stage(state, f'train:{l}')]))
            averaged = train(lang, run="v1", epochs=1 if args.smoke else cfg["epochs"], lr=cfg["lr"], max_hours=per_lang, max_frames=args.max_frames)
            mark(state, f"train:{lang}", averaged)
            say(f"train:{lang} done -> {averaged} {budget.line()}")
        averaged = state.get("done", {}).get(f"train:{lang}")
        if want("evaluate") and averaged and not stage(state, f"evaluate:{lang}"):
            result = evaluate(lang, averaged, limit=40 if args.smoke else None)
            record(lang, "v1", result)
            mark(state, f"evaluate:{lang}", result["overall"])
            say(f"evaluate:{lang}: {result['overall'] * 100:.1f}% {result['unit']} error rate after training {budget.line()}")
        if want("export") and averaged and not stage(state, f"export:{lang}"):
            results_path = os.path.join(DIRS["exp"], lang, "results.json")
            results = json.load(open(results_path, encoding="utf-8")) if os.path.isfile(results_path) else {}
            out = export(lang, "v1", averaged, results, split, push_to=f"{args.push}-{lang}" if args.push else None)
            mark(state, f"export:{lang}", out)
            if args.serve_dir and not args.smoke:
                import shutil

                dest = os.path.join(args.serve_dir, os.path.basename(out))
                if os.path.isdir(dest):
                    shutil.rmtree(dest)
                shutil.copytree(out, dest)
                say(f"copied {out} to {dest}: thelip-server serves it from its next start")

    say(f"done {budget.line()}")
    for lang in languages:
        base = state.get("done", {}).get(f"baseline:{lang}")
        after = state.get("done", {}).get(f"evaluate:{lang}")
        say(f"  {lang}: " + (f"before {base * 100:.1f}% -> " if isinstance(base, float) else "") + (f"after {after * 100:.1f}%" if isinstance(after, float) else "no model trained yet"))


if __name__ == "__main__":
    main()
