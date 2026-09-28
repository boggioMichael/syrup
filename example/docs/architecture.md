# Architecture

```
                       video / frames
                            │
                 ┌──────────▼──────────┐
                 │  face detection     │  detect.py — syrup find_face + find_eyes (default), or OpenCV Haar
                 └──────────┬──────────┘  every 2nd frame; boxes held in between
                            │ detections (box, eyes)
                 ┌──────────▼──────────┐
                 │  tracking           │  track.py — IoU + Hungarian, grace frames; cuts.py ends every track at a shot change
                 └──────────┬──────────┘
                            │ PersonTrack (anonymous number, boxes over time)
                 ┌──────────▼──────────┐
                 │  mouth ROI          │  mouth.py — lip line and corners from the eyes; LipNet's 100x50 crop, scale fixed per track
                 └──────────┬──────────┘
                            │ crops, openness, mouth/face motion
                 ┌──────────▼──────────┐
                 │  speaking activity  │  activity.py — probability per frame, spans, merged pauses
                 └──────────┬──────────┘
                            │ utterance windows (widened with silence, ≤ 75 frames)
                 ┌──────────▼──────────┐
                 │  VSR (batched)      │  vsr/ — VisualSpeechModel.predict_batch; registry picks the model per language
                 └──────────┬──────────┘
                            │ per-frame symbol probabilities
                 ┌──────────▼──────────┐
                 │  decoding           │  decode.py — CTC best path with frame timing, confidence, dictionary correction
                 └──────────┬──────────┘
                            │ words with time and confidence
                 ┌──────────▼──────────┐
                 │  segments           │  pipeline.py — TranscriptSegment per utterance and person; [word?] below threshold
                 └──────────┬──────────┘
        audio modes only ───┤ audio.py — VAD, ASR backend (Whisper), words attributed to the mouth that moved; fusion
                            │
                 ┌──────────▼──────────┐
                 │  AnalysisResult     │  schema.py = shared-types/schema/analysis.schema.json; export.py -> TXT/JSON/SRT/VTT
                 └─────────────────────┘
```

The same pipeline is exposed three ways:

- **`lipreader.analyze(path, options)`** and `python -m lipreader` for files.
- **`Analyzer.push(frame)` / `finish()`** for frames that arrive one by one
  (the service's `/sessions`, a browser capturing a tab); `live=True` reads
  each speech span about 0.6 s after it closes.
- **The inference API** (`example/inference-api`, [api.md](api.md)) — jobs
  for whole videos, sessions for streams, exports, deletion, optional bearer
  token, rate limit. The Chrome extension, the web page and the iOS app's
  remote backend are its clients.

## Interfaces that let models be swapped

| interface | file | what a new implementation must provide |
|---|---|---|
| `FaceDetector.detect(rgb) -> [Detection(box, score, eyes?, profile?)]` | lipreader/detect.py | boxes and, when it can, the two eye centres |
| `VisualSpeechModel.predict_batch(clips) -> [Decoded]` + `ModelSpec` | lipreader/vsr/base.py | native fps, max clip length, the crop layout it expects, licence, availability |
| `SpeechRecogniser.transcribe(audio, language) -> AsrResult` | lipreader/audio.py | words with time and confidence, detected language |
| `registry.candidates(language)` | lipreader/vsr/registry.py | the models for a language, best first, with `available` decided at runtime |

Adding a language is adding a `ModelSpec` (and a backend class) to the
registry; everything downstream — windows, decoding, segments, exports,
the API's `/languages`, the extension's panel — follows.

## Multi-person processing

Each visible person is a `PersonState` while alive (mouth tracker, per-frame
signals, crops) and a `PersonTrack` in the result. Utterances from all
people are collected first and the model is called once per batch
(`_run_vsr`), so a backend with a real batch dimension (PyTorch, Core ML)
runs the faces together. Memory is bounded: crops are read and dropped
when a person's track ends, at a cut, or after 1500 frames.

Handled and tested: people entering and leaving (tracks start and end with
grace), shot cuts (every track ends; the fixture with three hard cuts
yields exactly four tracks), occlusion for up to 12 frames (boxes held),
overlapping faces (Hungarian assignment on IoU), simultaneous speakers
(both read), resolution changes (scale is fixed per track, not per video),
variable frame rates (frames are resampled to the model's 25 fps by
nearest frame). Profile views are tracked with the profile cascade when
enabled but not read — no model here reads a profile mouth.

## Where things run

Local by default: the service on the user's machine, the extension and the
web page talking to `127.0.0.1`. The same service deployed elsewhere is
`processing: "server"` in every response and the clients show it. Uploads
are deleted when their frames have been read; results are memory-only with
a TTL and a DELETE; no embedding or identity of a face is ever stored.

## The three modes

| mode | audio track | visual reading | segments say |
|---|---|---|---|
| `visual` | never opened | yes | `visual` |
| `audiovisual` | ASR words attributed to the moving mouth; sound wins where it exists, visual fills the rest | yes | `audiovisual` / `visual` |
| `audio-attributed` | ASR words attributed to the moving mouth | no | `audio-attributed` |

## Repository map

| | Python | TypeScript | Swift |
|---|---|---|---|
| contract | `lipreader/schema.py` | `shared-types/src/index.ts` | `ios/Sources/LipReaderCore/Schema.swift` |
| pipeline | `lipreader/pipeline.py` | — | `ios/Sources/LipReaderCore/Pipeline.swift` |
| detection | `lipreader/detect.py` (syrup / OpenCV) | — | `FaceTracker.swift` (Vision) |
| service / client | `inference-api/lipreader_api/app.py` | `chrome-extension/src/worker.ts`, `web/index.html` | `VisualSpeechModel.swift` (`RemoteLipReader`) |
| UI | — | `chrome-extension/src/{content,panel}.ts`, `web/index.html` | `ios/App/*.swift` |
