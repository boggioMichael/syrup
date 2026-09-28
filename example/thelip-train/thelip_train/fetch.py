"""Download the candidates a source enumerates, up to a number of hours,
keeping each item's licence and captions next to the video.

    raw/<source>/<id>.mp4          720p at most, H.264 (yt-dlp merges; direct files as they are)
    raw/<source>/<id>.info.json    id, url, title, licence, language, source, duration, captions
    raw/<source>/<id>.<lang>.vtt   human captions when the source has them (never auto-generated)
    raw/<source>/<id>.srt          NASA captions

Everything is resumable: an item with its .info.json is not fetched again.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from typing import Iterable, Optional

from .common import DIRS, say
from .sources import SOURCES, enumerate_source


def probe_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], text=True)
        return float(out.strip())
    except (subprocess.CalledProcessError, ValueError, OSError):
        return None


def fetch_item(item: dict, out_dir: str) -> Optional[str]:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, item["id"])
    video = base + ".mp4"
    if os.path.isfile(base + ".info.json") and os.path.isfile(video):
        return video
    if item.get("direct"):
        ext = os.path.splitext(item["direct"].split("?")[0])[1].lower() or ".bin"
        tmp = base + ".download" + ext
        req = urllib.request.Request(item["direct"], headers={"User-Agent": "thelip-train/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        if ext == ".mp4":
            os.replace(tmp, video)
        else:  # Wikimedia's .webm / .ogv: one container for everything downstream
            subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", tmp, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-movflags", "+faststart", video])
            os.remove(tmp)
        if item.get("captions_url"):
            try:
                with urllib.request.urlopen(item["captions_url"], timeout=60) as r, open(base + ".srt", "wb") as f:
                    shutil.copyfileobj(r, f)
            except Exception as e:  # noqa: BLE001
                say(f"  captions for {item['id']}: {e}")
    else:
        import yt_dlp

        opts = {
            "quiet": True, "no_warnings": True, "outtmpl": base + ".%(ext)s",
            "format": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]",
            "merge_output_format": "mp4",
            "writesubtitles": bool(item.get("captions")), "writeautomaticsub": False,
            "subtitleslangs": [item["captions"]] if item.get("captions") else [], "subtitlesformat": "vtt",
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([item["url"]])
        if not os.path.isfile(video):
            for name in os.listdir(out_dir):
                if name.startswith(item["id"] + ".") and name.split(".")[-1] in ("mkv", "webm", "mp4"):
                    src = os.path.join(out_dir, name)
                    if src != video:
                        subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", src, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", video])
                        os.remove(src)
                    break
    if not os.path.isfile(video):
        return None
    item = dict(item, duration=item.get("duration") or probe_duration(video))
    with open(base + ".info.json", "w", encoding="utf-8") as f:
        json.dump(item, f, ensure_ascii=False, indent=1)
    return video


def fetch_source(name: str, max_hours: float, max_items: int = 400) -> float:
    """Downloads until `max_hours` of video are on disk for the source; returns the hours."""
    spec = SOURCES[name]
    if spec["kind"] == "thelip-samples":
        return 0.0
    out_dir = os.path.join(DIRS["raw"], name)
    have = sum((json.load(open(os.path.join(out_dir, n), encoding="utf-8")).get("duration") or 0)
               for n in os.listdir(out_dir) if n.endswith(".info.json")) / 3600 if os.path.isdir(out_dir) else 0.0
    say(f"{name}: {have:.1f} h on disk, target {max_hours:.1f} h ({spec['licence']})")
    if have >= max_hours:
        return have
    for item in enumerate_source(name, max_items=max_items):
        if have >= max_hours:
            break
        if os.path.isfile(os.path.join(out_dir, item["id"] + ".info.json")):
            continue
        try:
            video = fetch_item(item, out_dir)
        except Exception as e:  # noqa: BLE001
            say(f"  {item['id']}: {e}")
            continue
        if not video:
            continue
        seconds = probe_duration(video) or item.get("duration") or 0
        have += seconds / 3600
        say(f"  + {item['id']} ({seconds / 60:.0f} min, {item['licence']}) -> {have:.1f} h")
    return have


def fetch_all(plan: Iterable[tuple]) -> dict:
    """plan: (source name, max hours) pairs -> hours per source."""
    return {name: fetch_source(name, hours) for name, hours in plan}
