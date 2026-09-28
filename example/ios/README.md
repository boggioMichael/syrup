# LipReader for iOS

A native iOS app that finds the people in a video, watches their mouths,
and writes down what their lips say — with the same pipeline, the same JSON
and the same honesty rules as the Python reference in
`example/lipreader`. It is one of the example projects of the `syrup`
computer-vision library.

```
Photos / Files / camera / share sheet
        │
        ▼
FrameSource (AVAssetReader, video track only, nearest-frame sampling)
        │
        ▼
FaceTracker (Vision rectangles + landmarks every 2nd frame, IoU + Hungarian tracking, shot cuts)
        │
        ▼
MouthROI (100x50 LipNet crop at a per-track scale, openness, mouth/face motion)
        │
        ▼
Activity (speaking probability and spans, activity.py constants)
        │
        ├─ visual ──────────► LipNetCoreML (Core ML) ─► CTC best path + GRID dictionary
        ├─ audiovisual ─────► the above + SFSpeechRecognizer, fused: sound wins where it exists
        └─ audio-attributed ► SFSpeechRecognizer, words attributed to the mouth that moved
        │
        ▼
AnalysisResult (shared-types/schema/analysis.schema.json) ─► player, timeline, transcript, TXT/JSON/SRT/VTT
```

Where everything runs is always on screen: on this device by default; on
the inference API (`docs/api.md`) only when it is configured and the
device cannot read the chosen language, in which case the job's
`processing` ("local" or "server") is shown.

## Layout

| Path | What |
|---|---|
| `Package.swift` | SwiftPM manifest for `LipReaderCore` (iOS 17, no dependencies) |
| `Sources/LipReaderCore/Schema.swift` | Codable mirror of `analysis.schema.json` / `job.schema.json`, `boxAt` |
| `Sources/LipReaderCore/FrameSource.swift` | AVAssetReader frames at a target fps; `probe` |
| `Sources/LipReaderCore/FaceTracker.swift` | Vision detection, tracks, cuts (track.py, cuts.py) |
| `Sources/LipReaderCore/MouthROI.swift` | mouth geometry and the LipNet crop (mouth.py) |
| `Sources/LipReaderCore/Activity.swift` | speaking probability, spans, read windows (activity.py, `_windows`) |
| `Sources/LipReaderCore/VisualSpeechModel.swift` | model protocol, CTC decoder, LipNet via Core ML, remote API client |
| `Sources/LipReaderCore/LanguageRegistry.swift` | the honest language table; SFSpeechRecognizer availability and transcriber |
| `Sources/LipReaderCore/Pipeline.swift` | `Analyzer` actor and `LipReaderPipeline.analyze` (pipeline.py) |
| `Sources/LipReaderCore/Exporters.swift` | TXT/JSON/SRT/VTT (export.py) |
| `App/` | SwiftUI app: import, analyse, player with overlays, timeline, transcript, export, settings, privacy |
| `ShareExtension/` | "Share to LipReader": copies the video into the App Group inbox |
| `Tests/LipReaderCoreTests/` | XCTest unit tests that need no device or model |

## Building (Xcode 15 or newer, iOS 17 SDK)

There is no `.xcodeproj` checked in; create one so that signing, the App
Group and the extension are yours:

1. **New project** — iOS App, SwiftUI, product name `LipReader`, bundle id
   of your choice. Delete the generated `ContentView.swift` and
   `LipReaderApp.swift`, then add every file under `App/` to the app target
   and replace the target's `Info.plist` with `App/Info.plist` (or merge its
   keys: the four usage descriptions, `UIBackgroundModes`,
   `BGTaskSchedulerPermittedIdentifiers`, the `lipreader` URL scheme).
2. **Add the package** — File ▸ Add Package Dependencies ▸ Add Local…,
   choose this directory (`example/ios`), and link `LipReaderCore` to the
   app target. The package builds `Sources/LipReaderCore` and its tests.
3. **Capabilities** — on the app target add *Background Modes ▸ Background
   processing* and *App Groups* with the identifier in
   `AppGroup.identifier` (`LipReaderApp.swift`; the placeholder is
   `group.dev.lipreader.shared`).
4. **Share extension** — File ▸ New ▸ Target ▸ Share Extension, name it
   `LipReaderShare`, delete its generated view controller and storyboard,
   add `ShareExtension/ShareViewController.swift`, replace its `Info.plist`
   with `ShareExtension/Info.plist` (it sets the principal class and limits
   activation to one movie), and give it the same App Group. Keep
   `ShareAppGroup.identifier` equal to the app's.
5. **The model** — see below; without it the app still builds and runs,
   and reports English as "not available" with the reason.

## Getting the Core ML model

The only visual speech model that can run here is LipNet (Assael et al.
2016; weights from `rizkiarm/LipNet`, MIT; trained on GRID, CC BY 4.0),
converted from the Keras weights:

```sh
python3 ../ml/conversion/lipnet_to_coreml.py      # writes LipNetGRID.mlpackage
```

Drag `LipNetGRID.mlpackage` into the app target (Xcode compiles it to
`LipNetGRID.mlmodelc` in the bundle). `LipNetCoreML` expects one input of
shape `[1, T, 100, 50, 3]` float32 in `[0, 1]`, width-major mouth crops as
the original code stores them, with `T` a flexible dimension of 2…75, and
one output of shape `[1, T, 28]` (26 letters, space, CTC blank). Feature
names are read from the model description, so the script may name them
freely. `LanguageRegistry` reports English as available exactly when the
bundle contains the model.

Every other public visual model (Auto-AVSR, VSR-for-Multiple-Languages,
MuAViC/AV-HuBERT) is listed in the registry with its licence and the reason
it cannot run on iOS: PyTorch/fairseq research code, non-commercial terms,
downloads from Google Drive or `dl.fbaipublicfiles.com`, no conversion.
Hebrew has no public visual model or corpus at all; it is reachable only
through the audio modes, which the language picker says.

## Honesty rules (docs/decisions.md D4, D5)

- Every word carries a confidence; below `uncertainBelow` (0.5) it is
  rendered `[word?]` in the player, the transcript and every export.
- A language without a model produces no text and a warning, never a guess.
  `auto` in a lip-reading mode means "the one visual model available" and is
  reported as `detection: "assumed"` with the reason; Apple's speech
  recogniser does not identify languages either, so `auto` in the audio
  modes uses the device locale and is also `assumed`.
- Every screen that shows a transcript shows the notice from `schema.py`
  (`NOTICE`), and TXT/VTT exports carry it.
- Three modes, strictly separated. `visual` never opens the audio track:
  `FrameSource` adds only the video output to its reader in every mode, and
  nothing on the visual code path imports `AVAudio*` or touches
  `SFSpeechRecognizer` (`Pipeline.audioStage` is reached only through
  `Mode.usesAudio`). `audiovisual` fuses by "sound wins where it exists, the
  visual reading fills the rest". `audio-attributed` transcribes the sound
  and uses the faces only to decide who said each word.
- No identity: people are `Person 1`, `Person 2`… within one video. No
  embedding is computed; a result holds boxes, speaking times and words.
- Where processing happened is shown: on this device, or the server's
  `processing` field. "Delete now" removes the local copy, the saved result
  and the remote job (`DELETE /jobs/{id}`).

## Status

**Written without a compiler.** This directory was authored in an
environment with no Xcode, no Swift toolchain and no macOS. Nothing here has
been built, run or measured; no accuracy, speed or memory figure is claimed
for the iOS port. The code targets the documented iOS 17 / Swift 5.9 APIs
and the places where an API's exact shape was uncertain are marked
`// VERIFY:` in the source. Expect a first build to surface small fixes.

### Test plan for the first Mac build

1. `xcodebuild -scheme LipReaderCore -destination 'platform=iOS Simulator,name=iPhone 15' build`
   from `example/ios` (or open the package in Xcode) and fix whatever the
   compiler reports, starting with the `// VERIFY:` sites.
2. Run the unit tests in `Tests/LipReaderCoreTests` on the simulator
   (`xcodebuild test` with the same scheme). They cover the CTC decoder and
   the GRID corrector, speaking spans and read windows, the exporters'
   formats, the language table's honesty in both bundle states, the schema
   round-trip (including `used: null` and `raw`), `boxAt`, the tracker, the
   Hungarian assignment and the cut detector — none needs a device, a model
   or a video.
3. Build the app target and run it on a device (Vision face landmarks and
   Core ML are far slower on the simulator).
4. First manual test: the GRID sample clips in
   `python/thelip/LipNet/evaluation/samples` (`bbaf2n.mpg`, `sbwe5n.mpg`,
   …; the file name encodes the sentence, e.g. `sbwe5n` = "set blue with e
   five now"). Convert them to a format iOS decodes
   (`ffmpeg -i sbwe5n.mpg -c:v libx264 -pix_fmt yuv420p sbwe5n.mp4`), import
   in visual mode with language English, and compare the transcript with the
   sentence. The multi-person fixtures in
   `example/lipreader/tests/fixtures/*.mp4` come with their expected outputs
   in the `.json` next to each (people's regions, words and speaking times);
   `two_alternating`, `two_simultaneous`, `rapid_cuts` (tracks must end at
   the cuts) and `muted` (no audio track: the audio modes must warn and stay
   visual-only) are the ones to try after `one_speaker`.
5. Then the audio modes with a clip that has sound, in a language Apple's
   recogniser supports on device, and with Hebrew (audio-attributed) to see
   the visual stage skipped with its warning and the words attributed.
6. Check the overlay: the boxes must sit on the faces in the player for a
   portrait phone recording (the frame source renders rotated tracks upright
   through a video composition so that boxes, frames and AVPlayer agree).
