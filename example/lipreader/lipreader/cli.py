"""Command line: analyse a video, write transcripts.

    python -m lipreader analyze clip.mp4 --mode visual --language auto --out out.json --srt out.srt
    python -m lipreader languages
    python -m lipreader evaluate tests/fixtures --markdown docs/benchmarks.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import evaluation, export
from .schema import Options
from .vsr import registry


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lipreader")
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="read the lips in a video")
    a.add_argument("video")
    a.add_argument("--mode", default="visual", choices=["visual", "audiovisual", "audio-attributed"])
    a.add_argument("--language", default="auto")
    a.add_argument("--speakers", default="all", help="all | current | comma-separated track ids")
    a.add_argument("--max-faces", type=int, default=8)
    a.add_argument("--start", type=float, default=0.0)
    a.add_argument("--end", type=float, default=None)
    a.add_argument("--uncertain-below", type=float, default=0.5)
    a.add_argument("--detector", choices=["syrup", "opencv"], default=None)
    a.add_argument("--out", help="JSON result")
    a.add_argument("--srt")
    a.add_argument("--vtt")
    a.add_argument("--txt")

    sub.add_parser("languages", help="which languages can be read here, and why not")

    e = sub.add_parser("evaluate", help="score the fixtures with ground truth")
    e.add_argument("fixtures")
    e.add_argument("--only", nargs="*")
    e.add_argument("--json")
    e.add_argument("--markdown")

    args = ap.parse_args(argv)
    if args.command == "languages":
        import signal

        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        for s in registry.languages():
            print(f"{s.code:3s} {s.name:18s} {'yes' if s.visual_available else 'no ':3s} {s.visual_model or '-'}")
            print(f"    {s.visual_note}")
        return 0
    if args.command == "evaluate":
        from .pipeline import analyze

        scores = evaluation.run_suite(args.fixtures, lambda p: analyze(p, Options.from_dict({"mode": "visual", "language": "en"})), args.only)
        table = evaluation.markdown_table(scores)
        print(table)
        if args.json:
            with open(args.json, "w") as f:
                json.dump([s.to_dict() for s in scores], f, indent=1)
        if args.markdown:
            with open(args.markdown, "w") as f:
                f.write(table + "\n")
        return 0

    from .pipeline import analyze

    if args.detector:
        os.environ["LIPREADER_DETECTOR"] = args.detector
    speakers = args.speakers
    if speakers not in ("all", "current"):
        speakers = [int(v) for v in speakers.split(",")]
    options = Options.from_dict({
        "mode": args.mode, "language": args.language, "speakers": speakers, "maxFaces": args.max_faces,
        "startTime": args.start, "endTime": args.end, "uncertainBelow": args.uncertain_below,
    })
    result = analyze(args.video, options, progress=lambda p: print(f"\r{100 * p:5.1f}%", end="", file=sys.stderr))
    print(file=sys.stderr)
    for s in result.segments:
        print(f"[{s.start:6.2f}-{s.end:6.2f}] Person {s.track_id}: {s.text}   ({s.confidence:.2f}, {s.mode})")
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"{len(result.tracks)} people, {len(result.segments)} segments, {result.processing_seconds:.1f}s "
          f"(x{result.realtime_factor:.2f} realtime)", file=sys.stderr)
    for path, fmt in ((args.out, "json"), (args.srt, "srt"), (args.vtt, "vtt"), (args.txt, "txt")):
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(export.export(result, fmt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
