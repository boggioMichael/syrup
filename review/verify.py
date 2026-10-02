"""Independent checks of Eitan's syrup (0xGh0stAn0n/syrup @ 2ba28c2), run in CI.

Every section is isolated: a failure is printed and the next section runs.
Numbers are printed as `RESULT <key> <value>` lines so they can be collected
from the job log.
"""

import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "crates" / "syrup-runtime" / "tests" / "fixtures"
QUICK = "--quick" in sys.argv


def result(key, value):
    print(f"RESULT {key} {value}", flush=True)


def section(title):
    def wrap(fn):
        def run():
            print(f"\n===== {title} =====", flush=True)
            try:
                fn()
            except Exception:
                print(f"SECTION FAILED: {title}")
                traceback.print_exc()
                result(f"section_failed.{fn.__name__}", 1)

        return run

    return wrap


def child(code, env=None):
    """Runs `code` in a fresh interpreter, returns (stdout, stderr, seconds)."""
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=900,
    )
    return proc.stdout, proc.stderr, time.perf_counter() - started, proc.returncode


import syrup  # noqa: E402

print(f"python {sys.version.split()[0]} on {platform.platform()}")
print(f"syrup imported from {syrup.__file__}")


@section("resolution and refusals")
def refusals():
    accepted = [
        "find_face",
        "find_largest_face_in_top_half",
        "find_2_largest_faces_in_center",
        "find_words_left_to_right",
        "find_leftmost_word",
        "detect_qr_codes",
        "find_largest_panel_in_bottom_half",
        "find_red_regions",
        "measure_fill_of_largest_red_bar_in_bottom_third",
        "track_moving_regions",
        "measure_sharpness_of_words",
        "find_faces_larger_than_5pct_left_to_right",
        "locate_three_most_confident_human_faces_in_centre",
    ]
    refused = [
        "find_best_face",
        "find_main_face",
        "find_large_faces",
        "find_two_faces",
        "find_largest_faces",
        "find_smiling_faces",
        "find_cars",
        "find_people",
        "find_white_regions",
        "find_red_faces",
        "measure_fill",
        "find_faces_in_top_half_in_left_half",
        "count_faces",
        "read_text",
        "find_eyes",
        "find_text",
        "find_icons",
        "find_faces_top",
        "find_face_please",
        "find_faces_in_the_top_half",
    ]
    ok = 0
    for name in accepted:
        try:
            op = syrup.resolve(name)
            print(f"  ok      {name:55s} -> {op.intent}")
            ok += 1
        except syrup.SyrupError as e:
            print(f"  REFUSED {name:55s} -> {e}")
    result("names_accepted", f"{ok}/{len(accepted)}")
    refused_ok = 0
    for name in refused:
        try:
            op = syrup.resolve(name)
            print(f"  ACCEPTED {name:54s} -> {op.intent}")
        except syrup.IntentError as e:
            print(f"  refused {name:55s} [{e.kind}] {e.reason}" + (f" (hint: {e.hint})" if e.hint else ""))
            refused_ok += 1
    result("names_refused", f"{refused_ok}/{len(refused)}")
    # Synonyms share one artifact.
    keys = {syrup.resolve(n).plan_hash for n in ["find_face", "detect_faces", "locate_human_faces", "find_all_faces"]}
    result("synonyms_share_plan", len(keys) == 1)
    # hasattr on an unknown/refused name: does it return False or raise?
    try:
        value = hasattr(syrup.ops, "find_best_face")
        result("hasattr_refused_name", f"returned {value}")
    except Exception as e:
        result("hasattr_refused_name", f"raised {type(e).__name__}")


@section("first use: generate + compile + validate, then reuse")
def compile_times():
    names = ["find_face", "find_red_bars_in_bottom_third", "measure_fill_of_largest_red_bar", "find_2_largest_faces_in_center"]
    if QUICK:
        names = names[:2]
    for name in names:
        cache = tempfile.mkdtemp(prefix="syrup-verify-")
        code = (
            "import json, time, syrup\n"
            f"op = syrup.resolve({name!r})\n"
            "t = time.perf_counter(); info = op.prepare(); first = time.perf_counter() - t\n"
            "t = time.perf_counter(); again = op.prepare(); second = time.perf_counter() - t\n"
            "import os\n"
            "print(json.dumps({'first_s': first, 'status': info['status'], 'second_ms': second * 1000,"
            " 'second_status': again['status'], 'library_bytes': os.path.getsize(info['library']),"
            " 'build_ms': info['manifest']['build_ms'], 'cases': info['manifest']['validation_cases'],"
            " 'rustc': info['manifest']['rustc']}))\n"
        )
        out, err, secs, rc = child(code, {"SYRUP_CACHE_DIR": cache})
        if rc != 0:
            print(f"  {name}: child failed rc={rc}\n{err[-2000:]}")
            continue
        info = json.loads(out.strip().splitlines()[-1])
        # A new process with the same cache loads from disk.
        code2 = (
            "import json, time, syrup\n"
            f"op = syrup.resolve({name!r})\n"
            "t = time.perf_counter(); info = op.prepare(); dt = time.perf_counter() - t\n"
            "print(json.dumps({'disk_ms': dt * 1000, 'status': info['status']}))\n"
        )
        out2, err2, _, rc2 = child(code2, {"SYRUP_CACHE_DIR": cache})
        disk = json.loads(out2.strip().splitlines()[-1]) if rc2 == 0 else {"disk_ms": None, "status": err2[-300:]}
        print(f"  {name}: {info}  disk: {disk}")
        result(f"compile_first_s.{name}", f"{info['first_s']:.3f}")
        result(f"compile_build_ms.{name}", info["build_ms"])
        result(f"reuse_memory_ms.{name}", f"{info['second_ms']:.3f}")
        result(f"reuse_disk_ms.{name}", f"{disk['disk_ms']:.3f}" if disk["disk_ms"] is not None else "failed")
        result(f"library_kib.{name}", info["library_bytes"] // 1024)
        result(f"validation_cases.{name}", info["cases"])
        shutil.rmtree(cache, ignore_errors=True)


def timed(op, image, repeat=10, **kwargs):
    op(image, **kwargs)  # warm: compile/load and any per-size model preparation
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        found = op(image, **kwargs)
        times.append((time.perf_counter() - t) * 1000)
    return found, statistics.median(times)


@section("faces on the fixtures")
def faces():
    find = syrup.resolve("find_faces")
    for name in ["astronaut", "chelsea", "coffee"]:
        found, ms = timed(find, FIX / f"{name}.jpg", repeat=3)
        print(f"  {name}: {len(found)} face(s) {[(round(f.box.x), round(f.box.y), round(f.box.w), round(f.box.h), round(f.confidence, 3)) for f in found]} median {ms:.1f} ms")
        result(f"faces.{name}", len(found))
    # 1280x720 frame with the astronaut pasted in.
    frame = np.full((720, 1280, 3), 30, np.uint8)
    frame[40:552, 40:552] = np.asarray(PILImage.open(FIX / "astronaut.jpg").convert("RGB"))
    found, ms = timed(find, frame, repeat=10 if not QUICK else 3)
    result("find_face_1280x720_ms", f"{ms:.1f}")
    print(f"  1280x720: {len(found)} face(s), median {ms:.1f} ms")


@section("small faces: recall when the frame is much larger than the 640 px canvas")
def small_faces():
    find = syrup.resolve("find_faces")
    face = PILImage.open(FIX / "astronaut.jpg").convert("RGB").crop((150, 40, 300, 200))  # head and shoulders
    for size in [120, 80, 60, 40]:
        canvas = np.full((1080, 1920, 3), 90, np.uint8)
        placed = 0
        tile = np.asarray(face.resize((size, int(size * 160 / 150))))
        th, tw = tile.shape[:2]
        for row in range(3):
            for col in range(6):
                y, x = 60 + row * 340, 60 + col * 310
                canvas[y : y + th, x : x + tw] = tile
                placed += 1
        found = find(canvas)
        # The same picture cut into 640-wide tiles, which is what the detector sees at full resolution.
        tiled = 0
        for ty in range(0, 1080, 540):
            for tx in range(0, 1920, 640):
                tiled += len(find(canvas[ty : ty + 540, tx : tx + 640]))
        print(f"  {placed} heads {size}px wide in 1920x1080: found {len(found)}; in 640x540 tiles: {tiled}")
        result(f"small_faces_found.{size}px", f"{len(found)}/{placed}")
        result(f"small_faces_found_tiled.{size}px", f"{tiled}/{placed}")


@section("words")
def words():
    found, ms = timed(syrup.resolve("find_words_left_to_right"), FIX / "words.png", repeat=3)
    print(f"  {[(f.text, round(f.confidence, 2)) for f in found]} median {ms:.1f} ms")
    result("words_text", "/".join(f.text or "" for f in found))
    one = syrup.resolve("find_leftmost_word")(FIX / "words.png")
    result("leftmost_word", "/".join(f.text or "" for f in one))


@section("colour, bars, fill")
def colours():
    frame = np.full((720, 1280, 3), 40, np.uint8)
    frame[660:672, 100:1100] = (70, 70, 78)  # track
    frame[660:672, 100:500] = (210, 40, 40)  # 40% full red bar
    (bar,) = syrup.resolve("measure_fill_of_largest_red_bar_in_bottom_third")(frame)
    result("fill_40pct_bar", f"{bar.value:.3f}")
    regions = syrup.resolve("find_red_regions")(FIX / "coffee.jpg")
    result("red_regions_on_coffee", len(regions))
    _, ms = timed(syrup.resolve("find_red_regions"), frame, repeat=10)
    result("find_red_regions_1280x720_ms", f"{ms:.1f}")


@section("tracking: moving regions on OpenCV's vtest.avi")
def tracking():
    if QUICK:
        print("  skipped in quick mode")
        return
    import cv2

    url = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
    path = Path(tempfile.gettempdir()) / "vtest.avi"
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    cap = cv2.VideoCapture(str(path))
    session = syrup.ops.track_moving_regions.session()
    ids, per_frame, times, lifetimes = set(), [], [], {}
    frames = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        t = time.perf_counter()
        found = session(rgb)
        times.append((time.perf_counter() - t) * 1000)
        frames += 1
        per_frame.append(len(found))
        for f in found:
            ids.add(f.track.id)
            lifetimes[f.track.id] = lifetimes.get(f.track.id, 0) + 1
    long_lived = sum(1 for n in lifetimes.values() if n >= 25)
    result("vtest_frames", frames)
    result("vtest_tracks", len(ids))
    result("vtest_objects_per_frame", f"{statistics.mean(per_frame):.1f}")
    result("vtest_tracks_seen_25_frames_or_more", long_lived)
    result("vtest_median_track_life_frames", statistics.median(lifetimes.values()) if lifetimes else 0)
    result("vtest_ms_per_frame", f"{statistics.median(times):.1f}")


@section("bundles and frozen mode")
def frozen():
    bundle = tempfile.mkdtemp(prefix="syrup-bundle-")
    build_cache = tempfile.mkdtemp(prefix="syrup-build-")
    out, err, _, rc = child(
        f"import syrup, json; print(json.dumps(syrup.bundle({bundle!r}, 'find_face')))",
        {"SYRUP_CACHE_DIR": build_cache},
    )
    print("  bundle:", out.strip()[-300:], err[-300:])
    code = (
        "import syrup\n"
        f"faces = syrup.resolve('find_face')({str(FIX / 'astronaut.jpg')!r})\n"
        "print('frozen find_face', len(faces), faces.provenance.artifact_status)\n"
        "try:\n"
        "    syrup.resolve('find_largest_face')(" + repr(str(FIX / "astronaut.jpg")) + ")\n"
        "    print('frozen find_largest_face: ran (unexpected)')\n"
        "except syrup.SyrupError as e:\n"
        "    print('frozen find_largest_face:', e.stage, e.kind)\n"
    )
    out, err, _, rc = child(code, {"SYRUP_CACHE_DIR": bundle, "SYRUP_MODE": "frozen", "SYRUP_RUSTC": "/nonexistent/rustc"})
    print("  " + out.strip().replace("\n", "\n  "), err[-500:])
    result("frozen_mode", "ok" if "frozen find_face 1" in out and "not_prepared" in out else "unexpected")
    # No compiler at all, in development mode.
    code = (
        "import syrup\n"
        "try:\n"
        "    syrup.resolve('find_face').prepare(); print('compiled without rustc?')\n"
        "except syrup.SyrupError as e:\n"
        "    print(type(e).__name__, e.stage, e.kind)\n"
    )
    out, err, _, rc = child(code, {"SYRUP_CACHE_DIR": tempfile.mkdtemp(), "SYRUP_RUSTC": "/nonexistent/rustc"})
    result("no_rustc", out.strip())


@section("damaged artifact")
def damaged():
    cache = tempfile.mkdtemp(prefix="syrup-damage-")
    code = (
        "import syrup, json\n"
        "info = syrup.resolve('find_red_regions').prepare()\n"
        "print(info['library'])\n"
    )
    out, err, _, rc = child(code, {"SYRUP_CACHE_DIR": cache})
    library = out.strip().splitlines()[-1]
    with open(library, "r+b") as f:
        f.seek(100)
        f.write(b"\x00\x01\x02\x03")
    code = (
        "import syrup\n"
        "info = syrup.resolve('find_red_regions').prepare()\n"
        "print('after damage:', info['status'])\n"
    )
    out, err, _, rc = child(code, {"SYRUP_CACHE_DIR": cache})
    result("damaged_artifact_dev_mode", out.strip() or err.strip()[-200:])


for check in [refusals, compile_times, faces, small_faces, words, colours, tracking, frozen, damaged]:
    check()
print("\nverify.py done")
