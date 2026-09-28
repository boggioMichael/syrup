"""The pipeline against fixtures with exact ground truth (tests/make_fixtures.py;
run it first — it needs the LipNet checkout). pytest or plain python.

Slow tests (they run the whole pipeline) are marked so a quick run can skip
them: LIPREADER_QUICK=1 runs only the unit tests.
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from lipreader import activity, audio, decode, evaluation, export  # noqa: E402
from lipreader import pipeline  # noqa: E402
from lipreader.cuts import CutDetector  # noqa: E402
from lipreader.detect import Detection  # noqa: E402
from lipreader.schema import NOTICE, Box, Options, PersonTrack, Word  # noqa: E402
from lipreader.track import Tracker  # noqa: E402
from lipreader.vsr import LanguageUnavailable, registry  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")
SCHEMA_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "shared-types", "schema"))
QUICK = os.environ.get("LIPREADER_QUICK") == "1"
HAVE_LIPNET = registry.candidates("en")[0].available
_cache = {}


def fixture(name):
    return os.path.join(FIXTURES, name + ".mp4"), json.load(open(os.path.join(FIXTURES, name + ".json")))


def analyze(name, **options):
    """Cached: several tests look at the same result."""
    key = (name, json.dumps(options, sort_keys=True))
    if key not in _cache:
        path, _ = fixture(name)
        _cache[key] = pipeline.analyze(path, Options.from_dict(options))
    return _cache[key]


def words_of(result, track_id):
    return " ".join(w.text for s in result.segments if s.track_id == track_id for w in s.words)


def track_in(result, region):
    """The track whose first box centre lies in region (x, y, w, h)."""
    r = Box(*region)
    for t in result.tracks:
        cx, cy = t.keyframes[0].box.centre
        if r.x <= cx <= r.x2 and r.y <= cy <= r.y2:
            return t
    return None


def right_words(truth_words, hyp_words):
    return sum(1 for a, b in zip(truth_words.split(), hyp_words.split()) if a == b)


needs_fixtures = unittest.skipUnless(os.path.exists(os.path.join(FIXTURES, "one_speaker.mp4")) and HAVE_LIPNET and not QUICK,
                                     "fixtures or LipNet weights missing, or LIPREADER_QUICK=1")


# --- unit ------------------------------------------------------------------------

class UnitTests(unittest.TestCase):
    def test_edit_distance_wer_cer(self):
        self.assertEqual(evaluation.edit_distance("kitten", "sitting"), 3)
        self.assertAlmostEqual(evaluation.wer("set blue with e five now", "set blue with e five now"), 0.0)
        self.assertAlmostEqual(evaluation.wer("set blue with e five now", "set green with e five"), 2 / 6)
        self.assertAlmostEqual(evaluation.cer("abc", "abd"), 1 / 3)

    def test_ctc_best_path_keeps_timing_and_confidence(self):
        alphabet = list("abcdefghijklmnopqrstuvwxyz") + [" ", ""]
        a, b, space, blank = 0, 1, 26, 27
        path = [a, a, blank, b, blank, space, space, b, a]
        probs = np.full((len(path), 28), 0.01)
        for t, c in enumerate(path):
            probs[t, c] = 0.9 if t < 5 else 0.6
        decoded = decode.best_path(probs, alphabet, blank, space)
        self.assertEqual([w.text for w in decoded.words], ["ab", "ba"])
        self.assertEqual((decoded.words[0].start_frame, decoded.words[0].end_frame), (0, 3))
        self.assertAlmostEqual(decoded.words[0].confidence, 0.9)
        self.assertAlmostEqual(decoded.words[1].confidence, 0.6)
        fixed = decode.correct(decoded, lambda w: {"ab": "abe"}.get(w, w), known=lambda w: w in ("abe",))
        self.assertEqual(fixed.words[0].text, "abe")
        self.assertEqual(fixed.words[0].raw, "ab")
        self.assertAlmostEqual(fixed.words[0].confidence, 0.9 * 0.8)
        self.assertAlmostEqual(fixed.words[1].confidence, 0.6 * 0.5)  # unknown word: doubly doubtful

    def test_uncertain_words_are_marked_never_asserted(self):
        w = Word("five", 0.0, 0.2, 0.3, uncertain=True)
        self.assertEqual(w.rendered(), "[five?]")
        self.assertEqual(Word("five", 0.0, 0.2, 0.9, uncertain=False).rendered(), "five")

    def test_tracker_keeps_ids_and_cuts_end_tracks(self):
        tracker = Tracker(grace_frames=2)
        a, b = Box(10, 10, 50, 50), Box(200, 10, 50, 50)
        for i in range(5):
            live = tracker.update(i, i / 25, [Detection(Box(a.x + i, a.y, a.w, a.h), 1.0), Detection(b, 1.0)])
        self.assertEqual(sorted(t.track_id for t in live), [1, 2])
        # A disappears for longer than the grace: its track ends, B keeps its id.
        for i in range(5, 10):
            live = tracker.update(i, i / 25, [Detection(b, 1.0)])
        self.assertEqual([t.track_id for t in live], [2])
        tracker.cut(10 / 25)
        live = tracker.update(10, 10 / 25, [Detection(b, 1.0)])
        self.assertEqual([t.track_id for t in live], [3], "after a cut nothing is the same person")
        finished = tracker.finish()
        self.assertEqual([t.track_id for t in finished], [1, 2, 3])
        self.assertLessEqual(finished[0].last_t, 4 / 25 + 1e-9, "held frames after the last detection are trimmed")

    def test_speaking_probability_is_zero_for_a_still_face(self):
        still = np.zeros(50)
        p = activity.speaking_probability(still, still, still)
        self.assertTrue(np.all(p == 0))
        moving = 0.1 + 0.08 * np.sin(np.linspace(0, 12, 50))
        p = activity.speaking_probability(moving, np.zeros(50), np.zeros(50))
        self.assertGreater(p[25], 0.5)
        spans = activity.spans(p, np.arange(50) / 25.0)
        self.assertEqual(len(spans), 1)

    def test_cut_detector_fires_on_a_different_picture_only(self):
        a = np.zeros((72, 96, 3), dtype=np.uint8)
        a[..., 2] = 200  # a blue room
        a[20:50, 30:60] = (180, 150, 120)  # with a face
        b = np.zeros((72, 96, 3), dtype=np.uint8)
        b[..., 1] = 160  # a green room
        d = CutDetector()
        self.assertFalse(d.update(a))
        self.assertFalse(d.update(np.roll(a, 3, axis=1)))  # the same picture, moved a little
        self.assertTrue(d.update(b))

    def test_voice_activity_finds_the_loud_part(self):
        t = np.arange(0, 3.0, 1 / audio.RATE)
        signal = np.where((t > 1.0) & (t < 2.0), np.sin(2 * np.pi * 220 * t) * 0.5, 0.001 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
        spans = audio.voice_activity(signal)
        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0].start, 1.0, delta=0.05)
        self.assertAlmostEqual(spans[0].end, 2.0, delta=0.05)

    def test_languages_are_reported_honestly(self):
        table = {s.code: s for s in registry.languages()}
        self.assertFalse(table["he"].visual_available)
        self.assertIn("no public visual speech model", table["he"].visual_note)
        for code in ("ar", "de", "it", "es", "fr"):
            self.assertFalse(table[code].visual_available, code)
            self.assertTrue(table[code].visual_model, code)   # a model is named...
            self.assertTrue(table[code].visual_license, code)  # ...with its licence...
            self.assertIn("not runnable here", table[code].visual_note)  # ...and why it does not run
        with self.assertRaises(LanguageUnavailable):
            registry.resolve("he")
        with self.assertRaises(LanguageUnavailable) as caught:
            registry.resolve("ar")
        self.assertIn("CC BY-NC", str(caught.exception))

    def test_options_validate(self):
        with self.assertRaises(ValueError):
            Options.from_dict({"mode": "psychic"})
        with self.assertRaises(ValueError):
            Options.from_dict({"speakers": "someone"})
        self.assertEqual(Options.from_dict({"speakers": [1, 2]}).speakers, [1, 2])

    def test_export_formats(self):
        from lipreader.schema import AnalysisResult, TranscriptSegment, VideoInfo

        seg = TranscriptSegment("s1", 1, 1.0, 2.5, [Word("set", 1.0, 1.3, 0.9, False), Word("blue", 1.4, 2.5, 0.3, True)], "en", "visual", "lipnet-grid")
        res = AnalysisResult(VideoInfo(3.0, 25.0, 360, 288), "visual", "en", "en", "requested", [PersonTrack(1, 0.0, 3.0)], [seg], {}, 1.0)
        srt = export.to_srt(res)
        self.assertIn("00:00:01,000 --> 00:00:02,500", srt)
        self.assertIn("Person 1: set [blue?]", srt)
        vtt = export.to_vtt(res)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("<v Person 1>set [blue?]", vtt)
        txt = export.to_txt(res)
        self.assertIn(NOTICE, txt)
        data = json.loads(export.to_json(res))
        self.assertEqual(data["segments"][0]["words"][1]["uncertain"], True)
        self.assertEqual(export.to_srt(res, tracks=[2]).strip(), "")


# --- end to end -------------------------------------------------------------------

@needs_fixtures
class PipelineTests(unittest.TestCase):
    def test_one_speaker_is_read(self):
        result = analyze("one_speaker", mode="visual", language="en")
        _, truth = fixture("one_speaker")
        self.assertEqual(len(result.tracks), 1)
        self.assertGreaterEqual(right_words(truth["people"][0]["words"], words_of(result, 1)), 5)
        self.assertEqual(result.language_used, "en")
        self.assertEqual(result.mode, "visual")
        span = result.tracks[0].speaking[0]
        self.assertLess(abs(span.end - truth["people"][0]["speaking"][0][1]), 0.6)
        self.assertEqual(result.segments[0].mode, "visual")
        self.assertEqual(result.stages["asr"], 0.0)

    def test_visual_mode_never_opens_the_audio(self):
        path, _ = fixture("one_speaker")
        original = audio.samples
        calls = []
        audio.samples = lambda *a, **k: calls.append(a) or original(*a, **k)
        try:
            pipeline.analyze(path, Options.from_dict({"mode": "visual", "language": "en"}))
        finally:
            audio.samples = original
        self.assertEqual(calls, [])

    def test_two_alternating_speakers_get_their_own_words(self):
        result = analyze("two_alternating", mode="visual", language="en")
        _, truth = fixture("two_alternating")
        self.assertEqual(len(result.tracks), 2)
        left, right = truth["people"]
        lt, rt = track_in(result, left["region"]), track_in(result, right["region"])
        self.assertIsNotNone(lt)
        self.assertIsNotNone(rt)
        self.assertGreaterEqual(right_words(left["words"], words_of(result, lt.track_id)), 5)
        self.assertGreaterEqual(right_words(right["words"], words_of(result, rt.track_id)), 5)
        # Time: the left person's words come before the right person's.
        ls = [s for s in result.segments if s.track_id == lt.track_id]
        rs = [s for s in result.segments if s.track_id == rt.track_id]
        self.assertLess(max(s.end for s in ls), 3.2)
        self.assertGreater(min(s.start for s in rs), 2.8)

    def test_two_simultaneous_speakers_are_both_read(self):
        result = analyze("two_simultaneous", mode="visual", language="en")
        _, truth = fixture("two_simultaneous")
        self.assertEqual(len(result.tracks), 2)
        for person in truth["people"]:
            t = track_in(result, person["region"])
            self.assertIsNotNone(t)
            self.assertGreaterEqual(right_words(person["words"], words_of(result, t.track_id)), 5)
            self.assertTrue(t.speaking)

    def test_five_faces_three_speaking(self):
        result = analyze("five_faces", mode="visual", language="en")
        _, truth = fixture("five_faces")
        self.assertEqual(len(result.tracks), 5)
        for person in truth["people"]:
            t = track_in(result, person["region"])
            self.assertIsNotNone(t)
            if person["words"]:
                self.assertGreaterEqual(right_words(person["words"], words_of(result, t.track_id)), 5)
            else:
                self.assertEqual(words_of(result, t.track_id), "", "a still face says nothing")
                self.assertEqual(t.speaking, [])

    def test_low_resolution_degrades_but_does_not_break(self):
        result = analyze("low_res", mode="visual", language="en")
        self.assertEqual(len(result.tracks), 2)
        self.assertTrue(result.segments)
        scores = evaluation.score(result, fixture("low_res")[1], "low_res")
        self.assertLess(scores.wer, 1.0)  # measured, not promised: see docs/benchmarks.md

    def test_rapid_cuts_start_new_tracks_and_no_segment_crosses_a_cut(self):
        result = analyze("rapid_cuts", mode="visual", language="en")
        _, truth = fixture("rapid_cuts")
        self.assertEqual(len(result.tracks), len(truth["cuts"]) + 1)
        for cut in truth["cuts"]:
            for s in result.segments:
                self.assertFalse(s.start < cut - 0.02 and s.end > cut + 0.02, f"segment {s.id} crosses the cut at {cut}")
            self.assertTrue(any(abs(t.first_seen - cut) < 0.1 for t in result.tracks), f"a track starts at the cut {cut}")

    def test_muted_video_reads_the_same_and_audio_modes_say_so(self):
        muted = analyze("muted", mode="visual", language="en")
        with_sound = analyze("one_speaker", mode="visual", language="en")
        self.assertEqual(words_of(muted, 1), words_of(with_sound, 1))
        attributed = analyze("muted", mode="audio-attributed", language="en")
        self.assertEqual(attributed.segments, [])
        self.assertTrue(any("no audio track" in w for w in attributed.warnings))
        self.assertEqual(attributed.language_detection, "unavailable")
        av = analyze("muted", mode="audiovisual", language="en")
        self.assertEqual(words_of(av, 1), words_of(muted, 1), "audiovisual without audio falls back to the visual reading")
        self.assertTrue(any("visual-only" in w for w in av.warnings))

    def test_unavailable_language_produces_no_text(self):
        for language in ("he", "ar"):
            result = analyze("one_speaker", mode="visual", language=language)
            self.assertEqual(result.segments, [])
            self.assertEqual(result.language_detection, "unavailable")
            self.assertTrue(result.warnings and "visual reading disabled" in result.warnings[0])
            self.assertEqual(len(result.tracks), 1, "faces are still tracked")

    def test_auto_language_is_assumed_not_detected(self):
        result = analyze("one_speaker", mode="visual", language="auto")
        self.assertEqual(result.language_used, "en")
        self.assertEqual(result.language_detection, "assumed")
        self.assertIn("no visual language-identification", result.language_note)

    def test_current_speaker_only(self):
        result = analyze("two_simultaneous", mode="visual", language="en", speakers="current")
        read = {s.track_id for s in result.segments}
        self.assertEqual(len(read), 1)
        self.assertEqual(len(result.tracks), 2, "the other person is still tracked")

    def test_selected_tracks_only(self):
        result = analyze("two_simultaneous", mode="visual", language="en", speakers=[2])
        self.assertEqual({s.track_id for s in result.segments}, {2})

    def test_audio_attributed_mode_puts_words_on_the_moving_mouth(self):
        path, truth = fixture("two_alternating")
        words = []
        for person in truth["people"]:
            start, end = person["speaking"][0]
            tokens = person["words"].split()
            step = (end - start) / len(tokens)
            for i, tok in enumerate(tokens):
                words.append(audio.AsrWord(tok, start + i * step, start + (i + 1) * step, 0.95))
        fake = audio.FakeRecogniser(words)
        result = pipeline.analyze(path, Options.from_dict({"mode": "audio-attributed", "language": "en"}), recogniser=fake)
        self.assertTrue(result.segments)
        self.assertTrue(all(s.mode == "audio-attributed" for s in result.segments))
        self.assertTrue(all(s.model == "fake-asr" for s in result.segments))
        left, right = truth["people"]
        lt, rt = track_in(result, left["region"]), track_in(result, right["region"])
        self.assertEqual(words_of(result, lt.track_id), left["words"])
        self.assertEqual(words_of(result, rt.track_id), right["words"])
        self.assertEqual(result.stages["vsr"], 0.0, "audio-attributed mode does not lip-read")

    def test_audiovisual_mode_prefers_sound_where_it_exists(self):
        path, truth = fixture("two_alternating")
        left, right = truth["people"]
        start, end = left["speaking"][0]
        tokens = left["words"].split()
        step = (end - start) / len(tokens)
        words = [audio.AsrWord(t, start + i * step, start + (i + 1) * step, 0.95) for i, t in enumerate(tokens)]
        result = pipeline.analyze(path, Options.from_dict({"mode": "audiovisual", "language": "en"}), recogniser=audio.FakeRecogniser(words))
        lt, rt = track_in(result, left["region"]), track_in(result, right["region"])
        modes = {s.track_id: s.mode for s in result.segments}
        self.assertEqual(modes[lt.track_id], "audiovisual", "sound covered this person")
        self.assertEqual(modes[rt.track_id], "visual", "no sound for this person: the lips were read")
        self.assertEqual(words_of(result, lt.track_id), left["words"])

    def test_result_matches_the_shared_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        with open(os.path.join(SCHEMA_DIR, "analysis.schema.json")) as f:
            schema = json.load(f)
        result = analyze("two_simultaneous", mode="visual", language="en")
        jsonschema.validate(result.to_dict(), schema)
        self.assertEqual(result.to_dict()["notice"], NOTICE)

    def test_side_profile_faces(self):
        self.skipTest("no profile-view fixture with ground truth exists; the profile cascade keeps such a face tracked "
                      "but no model here reads a profile mouth (documented limitation)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
