# Decisions

Engineering decisions, with the reason, in the order they were made. Each
entry is short; the code is the long version.

## D0 — Where this lives

These are example projects of the syrup library and live in `example/` of
its repository (one directory per project, `example/docs` for what they
share) rather than in a repository of their own: they exist to show the
library carrying a real product, and they version together with it.

## D1 — Build on The Lip

The repository already has a working muted-video lip reader
(`python/thelip`): mouth localisation from the eyes, LipNet in numpy from
published weights, a streaming CTC decode, 64/66 words on the GRID sample
clips. `example/lipreader` starts from that code rather than from zero,
generalised to many faces, many tracks and many languages behind interfaces.

## D2 — What can actually run here (2026-09-28)

The build environment has: Python 3.11 with numpy, OpenCV 4.13 (+contrib),
scipy, scikit-learn, Pillow, onnxruntime, starlette + uvicorn, pydantic,
httpx; Node 22 with TypeScript 6 and Playwright 1.56 (Chromium); ffmpeg. It
has no PyTorch, no TensorFlow, no coremltools, no pytest; `pip install` and
`npm install` are refused by the network policy, and so are Google Drive,
Hugging Face and `dl.fbaipublicfiles.com`, where every open visual speech
model except LipNet is hosted. GitHub repositories can be cloned.

Consequences: the only visual speech model that can be downloaded and run
here is LipNet (weights committed to `rizkiarm/LipNet`). Everything else is
implemented behind interfaces with download scripts and licence gates, and
documented as not run here rather than pretended.

## D3 — Model survey (see docs/models.md)

| model | languages | visual-only WER | licence | runs here |
|---|---|---|---|---|
| LipNet (Assael et al. 2016; rizkiarm weights) | en, GRID 51-word vocabulary | 3.4% (overlapped speakers) | MIT code+weights; GRID CC BY 4.0 | yes |
| Auto-AVSR (mpc001/auto_avsr) | en | 20.3% LRS3 | Apache-2.0 code; weights "may have their own licenses derived from the training data" (LRS3: non-commercial research) | no (Google Drive, PyTorch) |
| VSR for Multiple Languages (mpc001) | en, es, fr, pt, zh | 32.3 / 44.5 / 58.6 / 51.4% | non-commercial ("comparative or benchmarking purposes") | no (Google Drive, PyTorch) |
| AV-HuBERT + MuAViC (Meta) | en, ar, de, el, es, fr, it, pt, ru (AVSR checkpoints) | not reported for video-only | CC BY-NC 4.0 | no (dl.fbaipublicfiles, fairseq) |
| Hebrew | — | — | — | no public model or corpus found |

So: English is real today (closed vocabulary). Spanish, French, Italian,
Arabic, German exist only as non-commercial research checkpoints that need
PyTorch/fairseq and a download the build machine cannot make; adapters and
scripts are provided, marked "not run here". Hebrew visual-only does not
exist publicly; Hebrew is reachable only through the audio modes (Whisper
supports it), which is stated in the UI.

## D4 — Never invent text

Every word carries a confidence; below `uncertainBelow` (0.5) it is rendered
`[word?]`. A language with no model produces no segments and a `warnings`
entry, never a guess. "auto" language in visual-only mode means "the one
visual model available" and is reported as `detection: "assumed"`, because
no visual language-identification model exists publicly.

## D5 — Three modes, strictly separated

`visual` never opens the audio track (the frame source is opened video-only).
`audiovisual` runs both and fuses by confidence. `audio-attributed` runs ASR
on the audio and uses the visual pipeline only to decide who said it. The
mode is stamped on every segment.

## D6 — Local first

Everything runs on the user's machine: the API is a local service and the
extension talks to `127.0.0.1`. The same service can be deployed; the
response then says `processing: "server"` and the UI shows it. Uploads are
deleted when their frames have been read; results are memory-only with a TTL
and a DELETE.

## D7 — YouTube frames

A content script cannot read YouTube's frames: the media is cross-origin and
taints any canvas. Two paths, both implemented: the service fetches the video
by URL with `yt-dlp` (whole video, native resolution; the default), or the
extension captures the tab with `chrome.tabCapture` in an offscreen document
and streams JPEG frames to `/sessions` (live, what is on screen). The
tabCapture path could not be exercised in this environment (no YouTube, no
headed Chrome with tab capture) and is marked as such.

## D8 — Tests without a lab

Ground truth comes from GRID: the file name encodes the sentence. Multi-person
fixtures are composited from GRID clips (side by side, tiled, concatenated
with hard cuts, downscaled), so every fixture has exact words, exact speaker
regions and exact speaking times. Side-profile faces have no fixture with
ground truth; that test is skipped with the reason and the limitation is
documented.

## D9 — Batched inference

The VSR interface takes a batch of mouth clips; the numpy LipNet runs them one
after another (numpy has no batch dimension to exploit here), a PyTorch or
Core ML backend batches for real. The pipeline collects utterances from all
tracks first and calls the model once per batch.
