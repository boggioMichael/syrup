"""Paths, state, budget, text normalisation and error rates shared by the
stages. Everything the pipeline writes lives under WORK (THELIP_TRAIN_HOME,
default ./work): raw/ downloads, labels/ transcripts, clips/ mouth crops with
texts, manifests/ label lists, spm/ tokenizers, exp/ training runs, models/
exports, state.json progress, log.txt."""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from typing import Dict, Iterable, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.environ.get("THELIP_TRAIN_HOME") or os.path.join(os.path.dirname(HERE), "work"))
DIRS = {k: os.path.join(WORK, k) for k in ("raw", "labels", "clips", "manifests", "spm", "exp", "models", "tools")}
STATE = os.path.join(WORK, "state.json")
LOG = os.path.join(WORK, "log.txt")


def ensure_dirs() -> None:
    for d in DIRS.values():
        os.makedirs(d, exist_ok=True)


def say(*parts) -> None:
    line = " ".join(str(p) for p in parts)
    print(line, flush=True)
    try:
        os.makedirs(WORK, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")
    except OSError:
        pass


def load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    os.makedirs(WORK, exist_ok=True)
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE)


class Budget:
    """Wall-clock and money: the run stops starting stages once the hours are
    spent, and every message says what the machine has cost so far."""

    def __init__(self, hours: float, hourly_cost: float, started: Optional[float] = None):
        self.hours, self.hourly_cost = hours, hourly_cost
        self.started = started or time.time()

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.started) / 3600

    @property
    def left_hours(self) -> float:
        return self.hours - self.elapsed_hours

    @property
    def spent(self) -> float:
        return self.elapsed_hours * self.hourly_cost

    def allows(self, hours_needed: float) -> bool:
        return self.left_hours >= hours_needed

    def line(self) -> str:
        return f"[{self.elapsed_hours:.1f} h of {self.hours:.0f}, about ${self.spent:.0f}]"


# ---- text -------------------------------------------------------------------------------------

_HEBREW_MARKS = re.compile(r"[֑-ׇ]")          # niqqud and cantillation
_APOSTROPHES = "'’ʼ"


def normalise(text: str, language: str) -> str:
    """The text a model is trained to write: one case, one spelling of the
    apostrophe, letters and digits of the language, single spaces. English
    follows LRS3 (upper case, apostrophes kept); Hebrew drops niqqud and
    keeps geresh/gershayim as apostrophes and double quotes inside words."""
    text = unicodedata.normalize("NFKC", text or "")
    for a in _APOSTROPHES:
        text = text.replace(a, "'")
    if language == "he":
        text = _HEBREW_MARKS.sub("", text)
        text = text.replace("״", '"').replace("׳", "'")
        text = re.sub(r"[^א-ת0-9'\" ]+", " ", text)
        text = re.sub(r"(?<!\w)[\"']|[\"'](?!\w)", " ", text)  # quotes only inside words
    elif language == "zh":
        text = re.sub(r"[^一-鿿0-9]+", "", text)
        return text
    else:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: List[str], b: List[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rate(pairs: Iterable, unit: str = "word") -> float:
    """WER (unit="word") or CER (unit="char") over (reference, hypothesis) pairs."""
    errors = total = 0
    for ref, hyp in pairs:
        r = ref.split() if unit == "word" else list(ref.replace(" ", ""))
        h = hyp.split() if unit == "word" else list(hyp.replace(" ", ""))
        errors += edit_distance(r, h)
        total += len(r)
    return errors / total if total else 0.0


def jsonl_read(path: str) -> List[dict]:
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def jsonl_append(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h"


# ---- tools that differ between a Linux GPU box and a Windows PC ----------------------------------

def ffmpeg_exe() -> str:
    """ffmpeg on the PATH (a Linux box), else the one imageio-ffmpeg ships (Windows has none)."""
    import shutil

    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


def video_seconds(path: str) -> Optional[float]:
    """A video's length, read by PyAV or OpenCV (no ffprobe needed)."""
    try:
        import av

        with av.open(path) as c:
            if c.duration:
                return c.duration / 1_000_000
            s = next((s for s in c.streams if s.type == "video"), None)
            if s is not None and s.duration and s.time_base:
                return float(s.duration * s.time_base)
    except Exception:  # noqa: BLE001
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return frames / fps if frames and fps else None
    except Exception:  # noqa: BLE001
        return None


def link_dir(target: str, link: str) -> None:
    """link -> target: a symlink, or on Windows (no symlinks without developer mode) a junction, else a copy."""
    import shutil
    import subprocess

    if os.path.exists(link):
        return
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True)
    if not os.path.exists(link):
        shutil.copytree(target, link)
