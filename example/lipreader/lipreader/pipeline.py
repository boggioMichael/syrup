"""The pipeline, end to end:

    frames -> face detection -> tracking -> mouth -> speaking activity
           -> visual speech recognition (batched per model)
           -> words with time and confidence -> segments per person
           -> (audio modes) ASR words attributed to the person whose mouth moved

`Analyzer` takes frames one at a time (a file, or frames streamed from a
browser) and produces an AnalysisResult; `analyze(path, options)` wraps it
for a file. Visual mode never touches the audio track.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import activity as act
from . import audio as aud
from .cuts import CutDetector
from .decode import Decoded
from .detect import Detection, FaceDetector, default_detector
from .mouth import MouthTracker
from .schema import AnalysisResult, Box, Keyframe, ModelInfo, Options, PersonTrack, SpeakingSpan, TranscriptSegment, VideoInfo, Word
from .track import Tracker, TrackState
from .video import Frame, FrameSource, probe
from .vsr import LanguageUnavailable, ModelSpec, VisualSpeechModel, registry

FLUSH_FRAMES = 1500        # per track: read closed speech spans and drop their crops beyond this
MIN_CHUNK_FRAMES = 8


@dataclass
class PersonState:
    """Everything remembered about one track while it is alive."""

    mouth: MouthTracker = field(default_factory=MouthTracker)
    indices: List[int] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    openness: List[float] = field(default_factory=list)
    mouth_motion: List[float] = field(default_factory=list)
    face_motion: List[float] = field(default_factory=list)
    crops: List[np.ndarray] = field(default_factory=list)   # uint8 W x H x C
    mouth_boxes: Dict[int, Box] = field(default_factory=dict)
    last_crop_gray: Optional[np.ndarray] = None
    last_face_gray: Optional[np.ndarray] = None
    probability: Optional[np.ndarray] = None
    spans: List[SpeakingSpan] = field(default_factory=list)
    read_until: int = 0    # frames before this position have been read already


@dataclass
class Utterance:
    track_id: int
    frame_indices: List[int]
    times: List[float]
    clip: np.ndarray  # (T, W, H, C) float32
    span_probability: float


class Analyzer:
    def __init__(self, options: Options, video: VideoInfo, detector: Optional[FaceDetector] = None,
                 vsr_spec: Optional[ModelSpec] = None, recogniser: Optional[aud.SpeechRecogniser] = None,
                 progress: Optional[Callable[[float], None]] = None, activity_config: Optional[act.ActivityConfig] = None,
                 detect_every: int = 2, live: bool = False):
        """live: frames arrive as they happen; speech is read as soon as a
        span has closed (about 0.6 s after it ends) instead of at the end."""
        self.options = options
        self.detect_every = max(1, detect_every)
        self.live = live
        self.video = video
        self.detector = detector or default_detector()
        self.progress = progress
        self.activity_config = activity_config or act.ActivityConfig()
        self.cuts = CutDetector()
        self.tracker = Tracker()
        self.people: Dict[int, PersonState] = {}
        self.ended: Dict[int, PersonState] = {}
        self.frames_seen = 0
        self.warnings: List[str] = []
        self.stages: Dict[str, float] = {"detect": 0.0, "track": 0.0, "mouth": 0.0, "vsr": 0.0, "asr": 0.0}
        self.started = time.time()
        self.segments: List[TranscriptSegment] = []
        self.segment_count = 0
        self.language_used: Optional[str] = None
        self.language_detection = "unavailable"
        self.language_note = ""
        self.models: Dict[str, object] = {"faceDetector": self.detector.name}
        self.vsr: Optional[VisualSpeechModel] = None
        self.vsr_spec = vsr_spec
        self.recogniser = recogniser
        self.current_track: Optional[int] = None
        self._resolve_language()

    # --- language -------------------------------------------------------------

    def _resolve_language(self) -> None:
        wants_visual = self.options.mode in ("visual", "audiovisual")
        requested = self.options.language
        if not wants_visual:
            self.language_detection = "requested" if requested != "auto" else "audio-detected"
            self.language_used = None if requested == "auto" else requested
            return
        try:
            if self.vsr_spec is None:
                if requested == "auto":
                    ready = registry.available_languages()
                    if not ready:
                        raise LanguageUnavailable("auto", [])
                    self.vsr_spec = registry.resolve(ready[0])
                    self.language_used = ready[0]
                    self.language_detection = "assumed"
                    self.language_note = ("no visual language-identification model exists publicly; the only visual model "
                                          f"available here reads {ready[0]}, so that is what was used")
                else:
                    self.vsr_spec = registry.resolve(requested)
                    self.language_used = requested
                    self.language_detection = "requested"
            else:
                self.language_used = self.vsr_spec.info.languages[0] if self.vsr_spec.info.languages else requested
                self.language_detection = "requested" if requested != "auto" else "assumed"
            self.models["vsr"] = self.vsr_spec.info
        except LanguageUnavailable as error:
            self.vsr_spec = None
            self.language_used = None
            self.language_detection = "unavailable"
            self.warnings.append(f"visual reading disabled: {error}")

    # --- frames --------------------------------------------------------------------

    def push(self, frame: Frame) -> None:
        t = frame.t
        rgb = frame.rgb
        if self.cuts.update(rgb):
            self._end_tracks(self.tracker.live, cut=True)
            self.tracker.cut(t)
        t0 = time.time()
        if self.frames_seen % self.detect_every == 0 or not self.tracker.live:
            detections = self.detector.detect(rgb)
            detections.sort(key=lambda d: d.box.area, reverse=True)
            detections = detections[: self.options.max_faces]
        else:
            detections = [Detection(tr.box, 1.0, None, tr.profile, held=True) for tr in self.tracker.live]
        self.stages["detect"] += time.time() - t0
        t0 = time.time()
        before = {tr.track_id for tr in self.tracker.live}
        live = self.tracker.update(frame.index, t, detections)
        alive = {tr.track_id for tr in live}
        for track in list(self.tracker.finished):
            if track.track_id in self.people and track.track_id not in alive:
                self._end_track(track)
        self.stages["track"] += time.time() - t0
        t0 = time.time()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for track in live:
            if frame.index not in track.boxes:
                continue
            if track.track_id not in self.people:
                self.people[track.track_id] = PersonState()
            self._observe(self.people[track.track_id], track, frame, gray)
        if self.options.speakers == "current" and self.current_track is None and live:
            self.current_track = max(live, key=lambda tr: tr.box.area).track_id
        self.stages["mouth"] += time.time() - t0
        self.frames_seen += 1
        if self.progress and self.video.duration > 0:
            self.progress(min(0.95, t / self.video.duration))
        for track_id, person in self.people.items():
            if len(person.indices) - person.read_until > FLUSH_FRAMES:
                self._read_closed(person, track_id, keep_tail=40)
            elif self.live and self.frames_seen % 25 == 0 and len(person.indices) - person.read_until > 15:
                self._read_closed(person, track_id, keep_tail=15)

    def _observe(self, person: PersonState, track: TrackState, frame: Frame, gray: np.ndarray) -> None:
        eyes = track.eyes_by_frame.get(frame.index)
        estimate = person.mouth.update(gray, track.box, eyes)
        crop = person.mouth.crop(frame.rgb)  # W x H x C float
        crop_gray = cv2.cvtColor((crop.swapaxes(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        b = track.box
        x0, y0, x1, y1 = int(max(b.x, 0)), int(max(b.y, 0)), int(min(b.x2, gray.shape[1])), int(min(b.y2, gray.shape[0]))
        face = gray[y0:y1, x0:x1]
        face_small = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 if face.size else np.zeros((32, 32), np.float32)
        mouth_motion = float(np.abs(crop_gray - person.last_crop_gray).mean()) if person.last_crop_gray is not None else 0.0
        face_motion = float(np.abs(face_small - person.last_face_gray).mean()) if person.last_face_gray is not None else 0.0
        person.last_crop_gray = crop_gray
        person.last_face_gray = face_small
        person.indices.append(frame.index)
        person.times.append(frame.t)
        person.openness.append(estimate.openness)
        person.mouth_motion.append(mouth_motion)
        person.face_motion.append(face_motion)
        person.crops.append((crop * 255).astype(np.uint8))
        mouth_box = person.mouth.box()
        if mouth_box is not None:
            person.mouth_boxes[frame.index] = mouth_box
            if track.keyframes and track.keyframes[-1].t == frame.t:
                track.keyframes[-1] = Keyframe(frame.t, track.keyframes[-1].box, mouth_box)

    # --- reading -------------------------------------------------------------------

    def _selected(self, track_id: int) -> bool:
        speakers = self.options.speakers
        if speakers == "all":
            return True
        if speakers == "current":
            return track_id == self.current_track
        return track_id in speakers

    def _probability(self, person: PersonState) -> np.ndarray:
        return act.speaking_probability(np.array(person.openness), np.array(person.mouth_motion), np.array(person.face_motion), self.activity_config)

    def _read_closed(self, person: PersonState, track_id: int, keep_tail: int) -> None:
        """Read the speech spans that are safely over and forget their crops."""
        probability = self._probability(person)
        times = np.array(person.times)
        spans = act.spans(probability, times, self.activity_config)
        limit = len(person.indices) - keep_tail
        closed = [s for s in spans if self._pos(person, s.end) < limit]
        self._read_spans(person, track_id, closed, probability)
        if closed:
            last = self._pos(person, closed[-1].end)
            for i in range(person.read_until, last + 1):
                person.crops[i] = None  # type: ignore[call-overload]
            person.read_until = last + 1

    @staticmethod
    def _pos(person: PersonState, t: float) -> int:
        return int(np.searchsorted(np.array(person.times), t, side="left"))

    def _read_spans(self, person: PersonState, track_id: int, spans: Sequence[SpeakingSpan], probability: np.ndarray) -> None:
        person.spans.extend(spans)
        if self.vsr_spec is None or self.options.mode == "audio-attributed" or not self._selected(track_id):
            return
        utterances: List[Utterance] = []
        for c0, c1, probability in self._windows(person, spans):
            crops = [c for c in person.crops[c0 : c1 + 1] if c is not None]
            if len(crops) != c1 - c0 + 1:
                continue
            clip = np.stack(crops).astype(np.float32) / 255.0
            utterances.append(Utterance(track_id, person.indices[c0 : c1 + 1], person.times[c0 : c1 + 1], clip, probability))
        if utterances:
            self._run_vsr(utterances)

    def _windows(self, person: PersonState, spans: Sequence[SpeakingSpan]) -> List[Tuple[int, int, float]]:
        """Frame windows to read, as (first, last, probability). A short span
        is widened with the silence around it up to the model's clip length,
        because the model was trained on utterances with their silences and
        reads a bare fragment worse; it never takes another span's speech. A
        long span is cut into clips of the model's length."""
        max_frames = self.vsr_spec.max_frames
        limit = len(person.times) - 1
        fps = self.video.sampled_fps or self.video.fps or 25.0
        bounds = [(max(self._pos(person, sp.start), person.read_until), min(self._pos(person, sp.end), limit)) for sp in spans]
        # Neighbouring spans that fit in one clip are read together: a pause
        # of up to a second is still inside one sentence for the model.
        merged_bounds: List[Tuple[int, int]] = []
        merged_spans: List[SpeakingSpan] = []
        for (s, e), span in zip(bounds, spans):
            if merged_bounds and (s - merged_bounds[-1][1]) / fps <= 1.0 and e - merged_bounds[-1][0] + 1 <= max_frames:
                ps, pe = merged_bounds[-1]
                merged_bounds[-1] = (ps, e)
                prev = merged_spans[-1]
                merged_spans[-1] = SpeakingSpan(prev.start, span.end, max(prev.probability, span.probability))
            else:
                merged_bounds.append((s, e))
                merged_spans.append(span)
        bounds, spans = merged_bounds, merged_spans
        out = []
        for k, ((s, e), span) in enumerate(zip(bounds, spans)):
            if e - s + 1 < MIN_CHUNK_FRAMES:
                continue
            if e - s + 1 < max_frames:
                lo = bounds[k - 1][1] + 1 if k > 0 else person.read_until
                hi = bounds[k + 1][0] - 1 if k + 1 < len(bounds) else limit
                room = max_frames - (e - s + 1)
                before = min(room // 2, s - lo)
                after = min(room - before, hi - e)
                before = min(room - after, s - lo)
                out.append((s - before, e + after, span.probability))
            else:
                for c0 in range(s, e + 1, max_frames):
                    c1 = min(c0 + max_frames - 1, e)
                    if c1 - c0 + 1 >= MIN_CHUNK_FRAMES:
                        out.append((c0, c1, span.probability))
        return out

    def _run_vsr(self, utterances: List[Utterance]) -> None:
        if self.vsr is None:
            self.vsr = registry.load(self.vsr_spec)
        t0 = time.time()
        decoded = self.vsr.predict_batch([u.clip for u in utterances])
        self.stages["vsr"] += time.time() - t0
        for u, d in zip(utterances, decoded):
            self._add_segment(u, d)

    def _add_segment(self, u: Utterance, decoded: Decoded) -> None:
        if not decoded.words:
            return
        fps = self.video.sampled_fps or self.video.fps or 25.0
        words = []
        for w in decoded.words:
            start = u.times[min(w.start_frame, len(u.times) - 1)]
            end = u.times[min(w.end_frame, len(u.times) - 1)] + 1.0 / fps
            confidence = max(0.0, min(1.0, w.confidence))
            words.append(Word(w.text, start, end, confidence, confidence < self.options.uncertain_below, raw=w.raw))
        self.segment_count += 1
        self.segments.append(TranscriptSegment(
            id=f"s{self.segment_count}", track_id=u.track_id, start=words[0].start, end=words[-1].end, words=words,
            language=self.language_used or "?", mode="visual", model=self.vsr_spec.key,
        ))

    def _end_tracks(self, tracks: Sequence[TrackState], cut: bool = False) -> None:
        for track in list(tracks):
            if track.track_id in self.people:
                self._end_track(track)

    def _end_track(self, track: TrackState) -> None:
        person = self.people.pop(track.track_id, None)
        if person is None:
            return
        probability = self._probability(person)
        person.probability = probability
        spans = [s for s in act.spans(probability, np.array(person.times), self.activity_config)
                 if self._pos(person, s.end) >= person.read_until]
        self._read_spans(person, track.track_id, spans, probability)
        person.crops = []
        self.ended[track.track_id] = person

    # --- finish ---------------------------------------------------------------------

    def finish(self, audio_samples: Optional[np.ndarray] = None, audio_offset: float = 0.0) -> AnalysisResult:
        """audio_offset: the video time at which the audio samples begin."""
        tracks = self.tracker.finish()
        self._end_tracks(tracks)
        people = self.ended
        person_tracks: List[PersonTrack] = []
        for track in tracks:
            pt = track.to_person_track()
            person = people.get(track.track_id)
            if person is not None:
                pt.speaking = sorted(person.spans, key=lambda s: s.start)
            pt.detected_language = self.language_used if any(s.track_id == track.track_id for s in self.segments) else None
            person_tracks.append(pt)
        if self.options.mode in ("audiovisual", "audio-attributed"):
            self._audio_stage(audio_samples, person_tracks, audio_offset)
        for pt in person_tracks:
            pt.segment_ids = [s.id for s in self.segments if s.track_id == pt.track_id]
        self.segments.sort(key=lambda s: (s.start, s.track_id))
        if self.vsr is not None:
            self.vsr.close()
        if self.progress:
            self.progress(1.0)
        return AnalysisResult(
            video=self.video, mode=self.options.mode, language_requested=self.options.language,
            language_used=self.language_used, language_detection=self.language_detection, language_note=self.language_note,
            tracks=person_tracks, segments=self.segments, models=self.models,
            processing_seconds=time.time() - self.started, stages=dict(self.stages), warnings=list(self.warnings),
        )

    # --- audio modes ----------------------------------------------------------------

    def _audio_stage(self, audio_samples: Optional[np.ndarray], person_tracks: List[PersonTrack], offset: float = 0.0) -> None:
        def skipped(reason: str) -> None:
            self.warnings.append(reason + ("; the result is visual-only" if self.options.mode == "audiovisual" else ""))
            if self.options.mode == "audio-attributed":
                self.language_used, self.language_detection = None, "unavailable"

        if audio_samples is None:
            skipped("no audio track: the audio stage had nothing to read")
            return
        recogniser = self.recogniser if self.recogniser is not None else aud.default_recogniser()
        if recogniser is None:
            skipped("no speech recogniser installed (pip install faster-whisper); the audio stage was skipped")
            return
        t0 = time.time()
        requested = None if self.options.language == "auto" else self.options.language
        asr = recogniser.transcribe(audio_samples, requested)
        if offset:
            for w in asr.words:
                w.start += offset
                w.end += offset
        self.stages["asr"] += time.time() - t0
        self.models["asr"] = recogniser.info
        if asr.language:
            if self.options.language == "auto":
                self.language_used = asr.language
                self.language_detection = "audio-detected"
                self.language_note = f"language identified from the audio ({asr.language_probability:.2f})"
        attributed = self._attribute(asr.words)
        mode = self.options.mode
        new_segments: List[TranscriptSegment] = []
        for track_id, words in attributed:
            self.segment_count += 1
            uncertain = self.options.uncertain_below
            ws = [Word(w.text, w.start, w.end, w.confidence, w.confidence < uncertain) for w in words]
            new_segments.append(TranscriptSegment(
                id=f"s{self.segment_count}", track_id=track_id, start=ws[0].start, end=ws[-1].end, words=ws,
                language=asr.language or self.language_used or "?", mode=mode, model=recogniser.info.name,
            ))
        if mode == "audio-attributed":
            self.segments = new_segments
        else:
            # Fusion: sound is the more reliable reading where it exists; the
            # visual reading stays where the audio said nothing for that person.
            kept = []
            for s in self.segments:
                overlap = any(n.track_id == s.track_id and n.start < s.end and s.start < n.end for n in new_segments)
                if not overlap:
                    kept.append(s)
            self.segments = kept + new_segments
        for pt in person_tracks:
            if any(s.track_id == pt.track_id for s in self.segments):
                pt.detected_language = asr.language or pt.detected_language

    def _attribute(self, words: Sequence[aud.AsrWord]) -> List[Tuple[int, List[aud.AsrWord]]]:
        """Each spoken word goes to the person whose mouth was moving most
        while it was said; nobody moving enough means unattributed (-1).
        Consecutive words of one person form a segment."""
        groups: List[Tuple[int, List[aud.AsrWord]]] = []
        for w in words:
            best, best_p = -1, 0.0
            for track_id, person in self.ended.items():
                if person.probability is None or not person.times:
                    continue
                times = np.array(person.times)
                lo = np.searchsorted(times, w.start - 0.1)
                hi = np.searchsorted(times, w.end + 0.1)
                if hi <= lo:
                    continue
                p = float(person.probability[lo:hi].mean())
                if p > best_p:
                    best, best_p = track_id, p
            if best_p < 0.35:
                best = -1
            if groups and groups[-1][0] == best and w.start - groups[-1][1][-1].end <= 0.6:
                groups[-1][1].append(w)
            else:
                groups.append((best, [w]))
        return groups


def analyze(path: str, options: Optional[Options] = None, detector: Optional[FaceDetector] = None,
            recogniser: Optional[aud.SpeechRecogniser] = None, progress: Optional[Callable[[float], None]] = None,
            vsr_spec: Optional[ModelSpec] = None) -> AnalysisResult:
    options = options or Options()
    info = probe(path)
    native = 25.0
    if vsr_spec is None and options.mode != "audio-attributed":
        try:
            spec = registry.resolve(options.language) if options.language != "auto" else (
                registry.resolve(registry.available_languages()[0]) if registry.available_languages() else None)
            if spec:
                native = spec.native_fps
        except LanguageUnavailable:
            pass
    sample_fps = options.sample_fps or native
    source = FrameSource(path, sample_fps=sample_fps, start=options.start_time, end=options.end_time)
    info = source.info
    analyzer = Analyzer(options, info, detector=detector, vsr_spec=vsr_spec, recogniser=recogniser, progress=progress)
    for frame in source:
        analyzer.push(frame)
    audio_samples = None
    if options.mode in ("audiovisual", "audio-attributed"):
        audio_samples = aud.samples(path, options.start_time, options.end_time) if info.has_audio else None
    return analyzer.finish(audio_samples, audio_offset=options.start_time)
