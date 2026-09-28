"""Where training video comes from, and under what licence.

Only material whose licence allows it: works of the US federal government
(public domain), Creative Commons BY / BY-SA / CC0 uploads, and MIT
OpenCourseWare (CC BY-NC-SA, compatible with the non-commercial models
this trains). Every downloaded item keeps its licence in its info.json; the
manifests carry it per clip. Nothing here touches copyrighted films or
broadcasts, and a YouTube result is kept only when YouTube's own licence
field says Creative Commons (or the channel is a US government channel).

Each source enumerates candidates as dicts: id, url, title, duration,
licence, language, source, plus `direct` for a plain file URL and
`captions` when the source publishes human captions.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Dict, Iterator, List, Optional

from .common import say

CC_OK = ("Creative Commons Attribution", "Creative Commons Attribution-ShareAlike", "CC BY", "CC0", "Public domain")
YT_CC_FILTER = "EgIwAQ%253D%253D"   # YouTube's "Creative Commons" licence filter

SOURCES: Dict[str, dict] = {
    # ---- English -------------------------------------------------------------------------
    "nasa": {"language": "en", "kind": "nasa", "licence": "public domain (NASA, US government work)",
             "queries": ["interview", "press briefing", "astronaut", "science briefing", "spacewalk briefing"],
             "note": "images.nasa.gov API; briefings and interviews, often with .srt captions"},
    "whitehouse": {"language": "en", "kind": "youtube-channel", "licence": "public domain (US government work)",
                   "url": "https://www.youtube.com/@WhiteHouse/videos", "government": True,
                   "note": "press briefings and remarks; a speaker at a podium, transcripts at whitehouse.gov"},
    "statedept": {"language": "en", "kind": "youtube-channel", "licence": "public domain (US government work)",
                  "url": "https://www.youtube.com/@statedept/videos", "government": True},
    "youtube-cc-en": {"language": "en", "kind": "youtube-cc-search", "licence": "CC BY (YouTube licence field)",
                      "queries": ["interview", "vlog talking to camera", "podcast video", "storytime", "q and a"]},
    "mit-ocw": {"language": "en", "kind": "youtube-channel", "licence": "CC BY-NC-SA 4.0 (MIT OpenCourseWare)",
                "url": "https://www.youtube.com/@mitocw/videos", "expect_cc": True,
                "note": "lectures; wide shots are dropped by the face-size test in segment"},
    "wikimedia-en": {"language": "en", "kind": "wikimedia", "licence": "per file (CC BY, CC BY-SA, CC0, public domain)",
                     "queries": ["interview", "speech", "talk", "testimony"]},
    # ---- Hebrew: no corpus exists; this is where the first one comes from ---------------
    "youtube-cc-he": {"language": "he", "kind": "youtube-cc-search", "licence": "CC BY (YouTube licence field)",
                      "queries": ["ראיון", "הרצאה", "שיחה", "פודקאסט", "ולוג", "סיפור אישי", "הסבר"]},
    "wikimedia-he": {"language": "he", "kind": "wikimedia", "licence": "per file (CC BY, CC BY-SA, CC0, public domain)",
                     "queries": ["ראיון", "נאום", "הרצאה"]},
    # ---- Spanish / French / German / Arabic, for later adaptation --------------------------
    "youtube-cc-es": {"language": "es", "kind": "youtube-cc-search", "licence": "CC BY (YouTube licence field)", "queries": ["entrevista", "vlog", "charla"]},
    "youtube-cc-fr": {"language": "fr", "kind": "youtube-cc-search", "licence": "CC BY (YouTube licence field)", "queries": ["interview", "vlog", "conférence"]},
    "youtube-cc-de": {"language": "de", "kind": "youtube-cc-search", "licence": "CC BY (YouTube licence field)", "queries": ["interview", "vlog", "vortrag"]},
    "youtube-cc-ar": {"language": "ar", "kind": "youtube-cc-search", "licence": "CC BY (YouTube licence field)", "queries": ["مقابلة", "محاضرة", "فلوق"]},
    # ---- thelip.ai's own consented samples: mouth crops already ---------------------------
    "phone": {"language": "*", "kind": "thelip-samples", "licence": "consented users of thelip.ai (asked at the start; the improve switch before 2026-09-28)"},
    # ---- the owner's own recordings, dropped in a folder (my-videos/<language>/ next to the
    #      work folder): long clips of the reader speaking, the best material for their face --
    "my-videos-he": {"language": "he", "kind": "local-folder", "licence": "the owner's own recordings", "folder": "he"},
    "my-videos-en": {"language": "en", "kind": "local-folder", "licence": "the owner's own recordings", "folder": "en"},
}

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".3gp")


def my_videos_folder(language_folder: str) -> str:
    import os

    from .common import WORK

    root = os.environ.get("THELIP_TRAIN_ROOT") or os.path.dirname(WORK)
    return os.path.join(root, "my-videos", language_folder)


def _get_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "thelip-train/1.0 (research; contact via github.com/boggioMichael/syrup)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _ytdlp():
    import yt_dlp

    return yt_dlp


def _licence_ok(text: Optional[str]) -> bool:
    return bool(text) and any(k.lower() in text.lower() for k in CC_OK)


def youtube_entries(url: str, max_items: int) -> List[dict]:
    """Flat listing of a channel or search page, ids and titles only."""
    yt = _ytdlp()
    opts = {"quiet": True, "extract_flat": "in_playlist", "playlistend": max_items, "skip_download": True, "no_warnings": True}
    with yt.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return list(info.get("entries") or [])


def youtube_details(video_id: str) -> Optional[dict]:
    yt = _ytdlp()
    opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    try:
        with yt.YoutubeDL(opts) as ydl:
            return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    except Exception as e:  # noqa: BLE001
        say(f"  {video_id}: {e}")
        return None


def enumerate_source(name: str, max_items: int = 200, min_seconds: int = 60, max_seconds: int = 7200) -> Iterator[dict]:
    spec = SOURCES[name]
    kind = spec["kind"]
    if kind in ("youtube-channel", "youtube-cc-search"):
        urls = [spec["url"]] if kind == "youtube-channel" else [
            f"https://www.youtube.com/results?search_query={urllib.parse.quote(q)}&sp={YT_CC_FILTER}" for q in spec["queries"]]
        seen = set()
        for url in urls:
            for entry in youtube_entries(url, max_items):
                vid = entry.get("id")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                details = youtube_details(vid)
                if not details:
                    continue
                duration = details.get("duration") or 0
                if not (min_seconds <= duration <= max_seconds):
                    continue
                licence = details.get("license") or ""
                if spec.get("government"):
                    licence_text = spec["licence"]
                elif _licence_ok(licence):
                    licence_text = licence
                else:
                    continue  # not free to use: skipped, whatever the channel says elsewhere
                subs = details.get("subtitles") or {}
                lang = spec["language"]
                caption = None
                for key in (lang, f"{lang}-orig", "en" if lang == "en" else None):
                    if key and key in subs:
                        caption = key
                        break
                yield {"id": f"yt_{vid}", "url": f"https://www.youtube.com/watch?v={vid}", "title": details.get("title"),
                       "duration": duration, "licence": licence_text, "language": lang, "source": name,
                       "channel": details.get("channel"), "captions": caption}
                if len(seen) >= max_items:
                    return
    elif kind == "nasa":
        seen = set()
        for q in spec["queries"]:
            page = 1
            while len(seen) < max_items:
                data = _get_json(f"https://images-api.nasa.gov/search?q={urllib.parse.quote(q)}&media_type=video&page={page}")
                items = data.get("collection", {}).get("items", [])
                if not items:
                    break
                for item in items:
                    meta = (item.get("data") or [{}])[0]
                    nid = meta.get("nasa_id")
                    if not nid or nid in seen:
                        continue
                    seen.add(nid)
                    try:
                        assets = _get_json(item["href"])
                    except Exception:  # noqa: BLE001
                        continue
                    files = [a for a in assets if a.lower().endswith(".mp4")]
                    if not files:
                        continue
                    pick = next((a for a in files if "~medium" in a), None) or next((a for a in files if "~orig" in a), files[0])
                    srt = next((a for a in assets if a.lower().endswith(".srt")), None)
                    yield {"id": f"nasa_{re.sub(r'[^A-Za-z0-9_-]', '_', nid)}", "url": pick, "direct": pick, "title": meta.get("title"),
                           "duration": None, "licence": spec["licence"], "language": "en", "source": name,
                           "captions_url": srt, "description": (meta.get("description") or "")[:2000]}
                    if len(seen) >= max_items:
                        break
                page += 1
    elif kind == "wikimedia":
        seen = set()
        for q in spec["queries"]:
            params = {"action": "query", "format": "json", "generator": "search", "gsrnamespace": "6",
                      "gsrsearch": f"{q} filetype:video", "gsrlimit": "50", "prop": "imageinfo",
                      "iiprop": "url|extmetadata|size", "iiextmetadatafilter": "LicenseShortName|License|Artist|ImageDescription"}
            data = _get_json("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params))
            for page in (data.get("query", {}).get("pages") or {}).values():
                info = (page.get("imageinfo") or [{}])[0]
                meta = info.get("extmetadata") or {}
                licence = (meta.get("LicenseShortName") or {}).get("value") or ""
                if not _licence_ok(licence) and not licence.lower().startswith("pd"):
                    continue
                url = info.get("url")
                title = page.get("title", "")
                if not url or title in seen:
                    continue
                seen.add(title)
                yield {"id": "wm_" + re.sub(r"[^A-Za-z0-9_-]", "_", title.replace("File:", ""))[:80], "url": url, "direct": url,
                       "title": title, "duration": None, "licence": licence, "language": spec["language"], "source": name,
                       "author": re.sub(r"<[^>]+>", "", (meta.get("Artist") or {}).get("value") or "")[:200]}
                if len(seen) >= max_items:
                    return
    elif kind == "local-folder":
        import os

        folder = my_videos_folder(spec["folder"])
        if not os.path.isdir(folder):
            return
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(VIDEO_EXTENSIONS):
                continue
            import zlib

            # ASCII ids (the paths go through PyTorch), unique even for names in Hebrew
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(fn)[0])[:40].strip("_") or "clip"
            yield {"id": f"my_{stem}_{zlib.crc32(fn.encode('utf-8')):08x}", "url": os.path.join(folder, fn), "local": os.path.join(folder, fn), "title": fn,
                   "duration": None, "licence": spec["licence"], "language": spec["language"], "source": name}
    elif kind == "thelip-samples":
        return
    else:
        raise ValueError(f"unknown source kind {kind}")
