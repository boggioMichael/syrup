"""Measuring the pipeline: word and character error rates, whether words went
to the right person, whether one person stayed one track, and speed.

Ground truth is a fixture's JSON (tests/make_fixtures.py): people with a
frame region, the words they said and when. A track is matched to a person
by the region its boxes fall in.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import AnalysisResult, Box, PersonTrack


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    return edit_distance(ref, hypothesis.split()) / len(ref) if ref else (0.0 if not hypothesis.split() else 1.0)


def cer(reference: str, hypothesis: str) -> float:
    ref = reference.replace(" ", "")
    hyp = hypothesis.replace(" ", "")
    return edit_distance(ref, hyp) / len(ref) if ref else (0.0 if not hyp else 1.0)


def _plain(text: str) -> str:
    """Transcript text without the uncertainty marks."""
    return " ".join(w.strip("[]?") for w in text.split())


def match_tracks(result: AnalysisResult, people: Sequence[dict]) -> Dict[int, int]:
    """track id -> index of the person whose region holds most of the
    track's boxes (-1 when none does). When several people share a region
    (shots of a cut sequence) the one whose time span overlaps the track
    most wins."""
    regions = [Box(*p["region"]) for p in people]
    out: Dict[int, int] = {}
    for track in result.tracks:
        votes = [0.0] * len(regions)
        for k in track.keyframes:
            cx, cy = k.box.centre
            for i, r in enumerate(regions):
                if r.x <= cx <= r.x2 and r.y <= cy <= r.y2:
                    votes[i] += 1
        for i, p in enumerate(people):
            if votes[i] > 0 and p.get("speaking"):
                overlap = sum(_spans_overlap((track.first_seen, track.last_seen), tuple(sp)) for sp in p["speaking"])
                votes[i] += overlap / max(track.last_seen - track.first_seen, 0.04)
        out[track.track_id] = int(max(range(len(regions)), key=lambda i: votes[i])) if regions and max(votes) > 0 else -1
    return out


@dataclass
class Scores:
    name: str
    wer: Optional[float]
    cer: Optional[float]
    words_ref: int
    words_hyp: int
    attribution_accuracy: Optional[float]   # share of words placed on the right person
    track_consistency: Optional[float]      # 1 = every person was exactly one track
    tracks: int
    people: int
    speaking_precision: Optional[float]
    speaking_recall: Optional[float]
    cut_recall: Optional[float]            # share of known cuts at which a new track starts
    latency_ms_per_frame: float
    realtime_factor: float
    detail: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _spans_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def score(result: AnalysisResult, truth: dict, name: str = "") -> Scores:
    people = truth["people"]
    assignment = match_tracks(result, people)
    # Words per person: reference and hypothesis (all segments of the tracks matched to them).
    ref_total = hyp_total = errors = char_errors = ref_chars = 0
    right_place = words_placed = 0
    per_person = []
    known = [p for p in people if p["words"] is not None]  # None: the words are not known (a cut fixture)
    for i, p in enumerate(people):
        if p["words"] is None:
            continue
        segments = [s for s in result.segments if assignment.get(s.track_id, -1) == i]
        hyp = " ".join(_plain(s.text) for s in sorted(segments, key=lambda s: s.start))
        ref = p["words"]
        e = edit_distance(ref.split(), hyp.split())
        errors += e
        char_errors += edit_distance(ref.replace(" ", ""), hyp.replace(" ", ""))
        ref_total += len(ref.split())
        ref_chars += len(ref.replace(" ", ""))
        hyp_total += len(hyp.split())
        per_person.append({"person": i, "reference": ref, "hypothesis": hyp, "word_errors": e})
    # Attribution: a hypothesis word is well placed when the track it sits on
    # belongs to a person who did say something; words on a still face or on
    # no known person are misplaced.
    for s in result.segments:
        person = assignment.get(s.track_id, -1)
        if person >= 0 and people[person]["words"] is None:
            continue
        n = len(s.words)
        words_placed += n
        if person >= 0 and people[person]["words"]:
            right_place += n
    attribution = right_place / words_placed if words_placed else None
    cuts = truth.get("cuts", [])
    cut_recall = None
    if cuts:
        cut_recall = sum(1 for c in cuts if any(abs(t.first_seen - c) < 0.1 for t in result.tracks)) / len(cuts)
    # Track consistency: a person should be one track; extra tracks or id switches lower it.
    tracks_per_person = [sum(1 for t, p in assignment.items() if p == i) for i in range(len(people))]
    expected = truth.get("expected_tracks_per_person", 1)
    consistency = None
    if people:
        consistency = sum(1.0 / max(1, abs(n - expected) + 1) if n > 0 else 0.0 for n in tracks_per_person) / len(people)
    # Speaking spans: overlap of predicted spans with the truth spans.
    overlap = pred_total = true_total = 0.0
    for i, p in enumerate(people):
        tracks = [t for t in result.tracks if assignment.get(t.track_id, -1) == i]
        pred = [(s.start, s.end) for t in tracks for s in t.speaking]
        true = [tuple(x) for x in p["speaking"]]
        pred_total += sum(e - s for s, e in pred)
        true_total += sum(e - s for s, e in true)
        for a in pred:
            for b in true:
                overlap += _spans_overlap(a, b)
    precision = overlap / pred_total if pred_total > 0 else None
    recall = overlap / true_total if true_total > 0 else None
    frames = max(1.0, result.video.duration * (result.video.sampled_fps or result.video.fps or 25.0))
    return Scores(
        name=name, wer=errors / ref_total if ref_total else (0.0 if known else None),
        cer=char_errors / ref_chars if ref_chars else (0.0 if known else None),
        words_ref=ref_total, words_hyp=hyp_total, attribution_accuracy=attribution, track_consistency=consistency,
        tracks=len(result.tracks), people=len(people), speaking_precision=precision, speaking_recall=recall, cut_recall=cut_recall,
        latency_ms_per_frame=1000.0 * result.processing_seconds / frames, realtime_factor=result.realtime_factor,
        detail={"per_person": per_person, "assignment": assignment, "stages": result.stages, "cuts_expected": cuts},
    )


def markdown_table(scores: Sequence[Scores]) -> str:
    def fmt(v):
        return "-" if v is None else f"{v:.2f}"

    lines = ["| fixture | people | tracks | WER | CER | attribution | track consistency | cuts found | speaking P/R | ms/frame | RTF |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in scores:
        lines.append(f"| {s.name} | {s.people} | {s.tracks} | {fmt(s.wer)} | {fmt(s.cer)} | {fmt(s.attribution_accuracy)} | "
                     f"{fmt(s.track_consistency)} | {fmt(s.cut_recall)} | {fmt(s.speaking_precision)}/{fmt(s.speaking_recall)} | "
                     f"{s.latency_ms_per_frame:.0f} | {s.realtime_factor:.2f} |")
    return "\n".join(lines)


def run_suite(fixtures_dir: str, analyze, names: Optional[Sequence[str]] = None) -> List[Scores]:
    """analyze(path) -> AnalysisResult; scores every fixture with a JSON truth."""
    out = []
    for file in sorted(os.listdir(fixtures_dir)):
        if not file.endswith(".json"):
            continue
        name = file[:-5]
        if names and name not in names:
            continue
        video = os.path.join(fixtures_dir, name + ".mp4")
        if not os.path.exists(video):
            continue
        with open(os.path.join(fixtures_dir, file)) as f:
            truth = json.load(f)
        t0 = time.time()
        result = analyze(video)
        result.processing_seconds = time.time() - t0
        out.append(score(result, truth, name))
    return out
