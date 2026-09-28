# Benchmarks

Measured on 2026-09-28 in the build environment: 2 vCPU Intel Xeon @ 2.1
GHz, no GPU, Python 3.11, numpy 2.4, OpenCV 4.13, syrup built with
`--release`. Everything below is the output of
`python -m lipreader evaluate tests/fixtures` (`docs/benchmarks-*.json`
hold the raw numbers) and of `python/thelip/evaluate.py`. Nothing here is
an estimate.

## Fixtures (composited from GRID clips; exact ground truth)

Detector: **syrup** (`find_face` + `find_eyes`, the default)

| fixture | people | tracks | WER | CER | attribution | track consistency | cuts found | speaking P/R | ms/frame | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| five_faces (1080x576, 3 speaking, 2 still) | 5 | 5 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.66/0.92 | 152 | 3.81 |
| low_res (360x144, two people) | 2 | 2 | 0.50 | 0.47 | 1.00 | 1.00 | - | 0.72/1.00 | 34 | 0.85 |
| muted | 1 | 1 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.70/0.74 | 57 | 1.42 |
| one_speaker | 1 | 1 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.70/0.74 | 57 | 1.41 |
| rapid_cuts (4 shots, 1.2 s each) | 4 | 4 | - | - | - | 1.00 | 1.00 | 1.00/0.47 | 55 | 1.38 |
| two_alternating | 2 | 2 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.74/0.89 | 70 | 1.74 |
| two_simultaneous | 2 | 2 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.70/0.67 | 71 | 1.77 |

Detector: **OpenCV Haar** (`LIPREADER_DETECTOR=opencv`)

| fixture | people | tracks | WER | CER | attribution | track consistency | cuts found | speaking P/R | ms/frame | RTF |
|---|---|---|---|---|---|---|---|---|---|---|
| five_faces | 5 | 5 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.66/0.92 | 74 | 1.85 |
| low_res | 2 | 2 | 0.50 | 0.44 | 1.00 | 1.00 | - | 0.77/0.99 | 15 | 0.37 |
| muted | 1 | 1 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.70/0.74 | 17 | 0.43 |
| one_speaker | 1 | 1 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.70/0.74 | 16 | 0.39 |
| rapid_cuts | 4 | 4 | - | - | - | 1.00 | 1.00 | 1.00/0.57 | 16 | 0.40 |
| two_alternating | 2 | 2 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.72/0.87 | 24 | 0.60 |
| two_simultaneous | 2 | 2 | 0.00 | 0.00 | 1.00 | 1.00 | - | 0.70/0.67 | 28 | 0.71 |

Columns: WER/CER over the people whose words are known; *attribution* =
share of transcribed words placed on a person who was speaking; *track
consistency* = each person one track; *cuts found* = share of hard cuts at
which a new track starts; *speaking P/R* = overlap of predicted speaking
spans with the audio's voice activity (the spans are padded on purpose, so
precision is < 1 by design); *RTF* = processing seconds per video second
(< 1 is faster than real time).

Reading: the transcript is exact on every fixture with normal-sized faces,
including three people speaking at once among five faces and two people
taking turns, with words on the right person every time. At half
resolution (faces ~67 px wide) half the words are wrong: LipNet wants a
mouth about 100 px wide and the crop is being upsampled. The fixtures
have clean, frontal, well-lit faces from a lab corpus: this is the model's
home ground, not the wild.

## Where the time goes (seconds per fixture, syrup / OpenCV)

| fixture | detect | mouth | VSR (LipNet, numpy) |
|---|---|---|---|
| one_speaker (75 frames) | 3.60 / 0.66 | 0.05 / 0.04 | 0.30 / 0.25 |
| two_simultaneous | 4.41 / 1.23 | 0.07 / 0.07 | 0.52 / 0.51 |
| five_faces (1080x576) | 9.91 / 4.09 | 0.17 / 0.17 | 0.77 / 0.76 |

Face detection is the cost; the network is 4 ms per frame per person.
syrup's cascade currently takes ~60 ms per 360x288 frame against OpenCV's
~20 ms on this machine (the detector runs on every second frame), and the
eye search inside each face another ~28 ms. That is the library's next
performance item; until then `LIPREADER_DETECTOR=opencv` is the fast
setting and the results above show the two detectors read the same words.

## GRID sample clips (python/thelip)

| | words right | sentences exact |
|---|---|---|
| The Lip (single centred face, eyes every frame) | 64/66 | 9/11 |
| lipreader pipeline (tracking, detection every 2nd frame, activity windows) | 63/66 | 9/11 |

The one-word difference is the letter in `swiz3n` (`z` read as `g`), the
hardest word class in GRID; LipNet is sensitive to sub-pixel differences in
the mouth crop, which the held frames introduce.

## Not measured

- The iOS app: never compiled here (no Xcode).
- Real youtube.com pages and the tab-capture source: no network to YouTube
  and no capturable tab in headless Chromium; the extension was measured
  against a fake YouTube page and the real service.
- Open-vocabulary models and the other languages: not runnable here (see
  models.md); no accuracy is claimed for them.
- GPU batching: no GPU here; the batch interface exists, the numpy backend
  runs clips one after another.
