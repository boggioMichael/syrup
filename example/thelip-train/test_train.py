"""What can be checked without a GPU, a download or the model: the text
rules, the error rates, caption parsing, utterance cutting, the split, the
manifest format, the Auto-AVSR patch, and the mouth crops cut from GRID's
sample clips with syrup's own face points and Chaplin's alignment.

    THELIP_CHAPLIN=<chaplin checkout> THELIP_AUTO_AVSR=<auto_avsr checkout> python3 test_train.py
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = tempfile.mkdtemp(prefix="thelip-train-test-")
os.environ["THELIP_TRAIN_HOME"] = WORK
sys.path.insert(0, HERE)
from thelip_train import common, label, manifest, segment  # noqa: E402

failures = 0


def check(name, cond, detail=""):
    global failures
    print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if detail and not cond else ""))
    failures += 0 if cond else 1


def grid_clips():
    """GRID sample clips shipped with LipNet: (path, sentence) pairs, from their names."""
    C = {"b": "bin", "l": "lay", "p": "place", "s": "set"}; K = {"b": "blue", "g": "green", "r": "red", "w": "white"}
    P = {"a": "at", "b": "by", "i": "in", "w": "with"}; A = {"a": "again", "n": "now", "p": "please", "s": "soon"}
    D = {"1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", "z": "zero"}
    roots = [os.environ.get("LIPNET_DIR"), os.path.join(HERE, "..", "..", "python", "thelip", "LipNet"), "/home/claude/rizkiarm/lipnet"]
    for root in roots:
        if root and os.path.isdir(os.path.join(root, "evaluation", "samples", "GRID")):
            out = []
            for mpg in sorted(glob.glob(os.path.join(root, "evaluation", "samples", "GRID", "*.mpg")))[:3]:
                stem = os.path.basename(mpg)[:-4]
                out.append((mpg, " ".join([C[stem[0]], K[stem[1]], P[stem[2]], stem[3], D[stem[4]], A[stem[5]]])))
            return out
    return []


def main() -> int:
    common.ensure_dirs()
    # text
    check("English text follows LRS3", common.normalise("Hello, there! It's me.", "en") == "HELLO THERE IT'S ME")
    check("Hebrew text drops niqqud and keeps quotes inside words", common.normalise('שָׁלוֹם, מַה שְׁלוֹמְךָ? צה"ל ו-"ככה"', "he") == 'שלום מה שלומך צה"ל ו ככה')
    check("WER counts substitutions, insertions and deletions", abs(common.error_rate([("SET BLUE WITH E FIVE NOW", "THE TEMPLE WITHIN FIVE NOW")]) - 4 / 6) < 1e-9)
    check("CER on Chinese", abs(common.error_rate([("你好世界", "你好界")], "char") - 0.25) < 1e-9)
    # captions
    vtt = os.path.join(WORK, "x.vtt")
    open(vtt, "w", encoding="utf-8").write("WEBVTT\n\n1\n00:00:01.000 --> 00:00:03.000\nHello there\n\n2\n00:00:03.000 --> 00:00:05.500\nHello there\n\n3\n00:00:05.500 --> 00:00:07.000\n<c>General</c> Kenobi.\n")
    cues = label._parse_cues(vtt)
    check("VTT cues parsed, repeats of rolling captions dropped, tags stripped", [c["text"] for c in cues] == ["Hello there", "General Kenobi."] and cues[1]["start"] == 5.5, cues)
    seg = label._spread_words(cues[0])
    check("words get times spread over the cue", len(seg["words"]) == 2 and abs(seg["words"][1]["s"] - 2.0) < 1e-9, seg)
    # utterances
    words = [{"w": w, "s": i * 0.4, "e": i * 0.4 + 0.3, "p": 1} for i, w in enumerate("this is a test. and this is another one that goes on".split())]
    words[6]["s"] += 2.0; words[6]["e"] += 2.0  # a long gap before "is"
    for w in words[7:]:
        w["s"] += 2.0; w["e"] += 2.0
    utts = segment.utterances({"segments": [{"words": words}]})
    check("utterances cut at sentence ends and long gaps; a 0.7 s scrap is dropped", [u["text"] for u in utts] == ["this is a test.", "is another one that goes on"], [u["text"] for u in utts])
    # split and manifests
    rows = []
    for source, videos in (("a", 3), ("b", 2), ("phone", 2)):
        for v in range(videos):
            for c in range(2):
                rows.append({"path": f"{source}/v{v}/{c:04d}.mp4", "frames": 50, "text": "SET BLUE", "language": "en", "source": source, "video": f"v{v}", "seconds": 2.0})
    parts = manifest.split(rows, val_fraction=0.34)
    check("phone samples and the held-out source are test; nothing of a video straddles splits",
          all(r["source"] in ("phone", parts["_holdout"]) for r in parts["test"]) and parts["_holdout"] == "b"
          and not ({(r["source"], r["video"]) for r in parts["train"]} & {(r["source"], r["video"]) for r in parts["val"]}), parts["_holdout"])

    class Stub:
        def ids(self, text):
            return [2 + (ord(ch) % 50) for ch in text.replace(" ", "▁")]
    for r in rows:
        common.jsonl_append(os.path.join(common.DIRS["clips"], r["source"] + ".jsonl"), r)
    summary = manifest.write_manifests("en", Stub())
    line = open(os.path.join(common.DIRS["manifests"], "en", "train.csv"), encoding="utf-8").readline().rstrip("\n")
    check("manifest rows are dataset,rel_path,frames,ids as Auto-AVSR reads them", line.count(",") == 3 and line.startswith("a,v") and line.split(",")[2] == "50", line)
    check("root_dir/labels points at the manifests", os.path.islink(os.path.join(common.DIRS["clips"], "labels")))
    # the Auto-AVSR patch
    auto = os.environ.get("THELIP_AUTO_AVSR") or "/home/claude/auto_avsr"
    if os.path.isdir(os.path.join(auto, "spm")):
        os.environ["THELIP_AUTO_AVSR"] = auto
        from thelip_train import train

        copy = train.prepare_copy("en")
        src = open(os.path.join(copy, "train.py"), encoding="utf-8").read()
        check("the training copy logs to CSV, needs no SLURM, obeys a wall-clock limit",
              "CSVLogger" in src and "WandbLogger" not in src and 'environ.get("SLURM_JOB_ID"' in src and "THELIP_MAX_TIME" in src)
    else:
        print("skip the Auto-AVSR patch (no checkout; set THELIP_AUTO_AVSR)")
    # crops from GRID with syrup's points and Chaplin's alignment
    chaplin = os.environ.get("THELIP_CHAPLIN") or "/home/claude/chaplin"
    clips = grid_clips()
    if os.path.isdir(os.path.join(chaplin, "pipelines")) and clips and shutil.which("ffmpeg"):
        os.environ["THELIP_CHAPLIN"] = chaplin
        raw, labels = os.path.join(common.DIRS["raw"], "grid"), os.path.join(common.DIRS["labels"], "grid")
        os.makedirs(raw, exist_ok=True); os.makedirs(labels, exist_ok=True)
        for mpg, sentence in clips:
            stem = os.path.basename(mpg)[:-4]
            subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", mpg, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-an", os.path.join(raw, stem + ".mp4")])
            json.dump({"id": stem, "source": "grid", "language": "en", "licence": "CC BY 4.0 (GRID)"}, open(os.path.join(raw, stem + ".info.json"), "w"))
            ws = sentence.split(); step = 2.1 / len(ws)
            json.dump({"language": "en", "how": "captions", "segments": [{"start": 0.45, "end": 2.55, "text": sentence,
                       "words": [{"w": w, "s": 0.45 + i * step, "e": 0.45 + (i + 1) * step, "p": 1.0} for i, w in enumerate(ws)]}]},
                      open(os.path.join(labels, stem + ".json"), "w"))
        n = segment.segment_source("grid", segment.SyrupLandmarker())
        rows = common.jsonl_read(os.path.join(common.DIRS["clips"], "grid.jsonl"))
        check("every GRID clip became one training clip with its sentence", n == len(clips) and all(r["text"] == s.upper() for r, (_, s) in zip(rows, clips)), rows)
        import cv2

        cap = cv2.VideoCapture(os.path.join(common.DIRS["clips"], rows[0]["path"]))
        frames = 0
        while cap.read()[0]:
            frames += 1
        check("the clip is 96x96 at 25 fps, one frame per 40 ms of the utterance", frames == rows[0]["frames"] and abs(frames - 2.34 * 25) <= 2 and cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 96, (frames, rows[0]["frames"]))
    else:
        print("skip the GRID crops (needs Chaplin, LipNet's GRID samples and ffmpeg)")
    shutil.rmtree(WORK, ignore_errors=True)
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
