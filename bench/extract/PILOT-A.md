# Pilot A: extraction on real HUD lines (exploratory)

**Exploratory. Decides no gate.** Written on 14 October 2026, before any
reader in it was measured on its dev-test frames. It tries the tools of P2
(a teacher of our own), P3 (black-box distillation) and P4 (reading the
teacher from the inside) on task A, with dev data only, so that the real
test sets, when they are recorded, meet tools that already work. Nothing
here can confirm or refute H1 or H2: the data are one session of a
compressed video, the labels are not a person's, and both extraction arms
are written by the same agent, who knows both.

## Data

Dev data of [DATA.md](DATA.md): the public Zakum recording
(`chaos-zakum-solo-lvl230.mp4`, 1152 × 720, 30 fps, 5,813 frames).

- **Lines.** The HP, MP and EXP text lines, cut at the boxes a layout gives
  on this recording (the game window does not move in it;
  [a/video_lines.py](a/video_lines.py)). Frames where the HUD is not drawn
  (a map change) are left out.
- **Runs.** Consecutive frames that show one value form a run: a frame
  starts a new run when more than 3 of its values differ by more than 50
  from the run's first frame. A run is the unit labelled; its frames are the
  items read. Splitting a value into two runs costs nothing; the thresholds
  sit just above the codec's noise so that two values are not joined.
- **Labels.** A run's label is read by Claude from the run's mean picture,
  enlarged and contrast-stretched, as DATA.md's protocol asks of a person; a
  reader may suggest, never decide. The visible characters, left to right,
  without spaces (amendment 5): `25896/29906`, `22,496,313,951[26.888%]`.
  Checks a label must pass where they apply: an MP or HP value no larger
  than its maximum; an EXP value and its percentage that agree with one
  total for the level; EXP that never falls. A run whose label cannot be
  read with certainty is marked unconfirmed: it is left out of every
  accuracy figure and counted.
- **Splits, by time.** Frames 0–3,499 train, 3,500–4,599 validation,
  4,600–5,812 dev-test. One session, so the splits are not independent (the
  maximum HP and MP, and the start of the EXP line, are the same in all
  three); confidence intervals resample blocks of 10 consecutive frames.

The crops are game graphics and stay outside the repository; the script,
the boxes, the run boundaries and the labels are in it.

## Readers

1. **Teacher:** the line reader of amendment 1 ([a/linenet.py](a/linenet.py)),
   a CNN over the whole line with CTC, at most 100,000 parameters, trained
   on the train split. For timing, INT8 in ONNX Runtime, one thread.
2. **White box:** explicit code built from the teacher's insides. The
   teacher's per-column posteriors place each character it reads (where its
   class peaks); the input windows at those places, averaged per class
   (several per class where the windows fall into clusters), are the
   templates; the gaps between placed characters give the spacing. The
   reader correlates every template along the line and decodes the best
   non-overlapping placement by dynamic programming; it refuses when the
   weakest character of the best reading, or the margin to the second-best
   reading, is under thresholds fitted on the validation split to make no
   wrong read there.
3. **Black box:** the same family of explicit code, built from the
   teacher's inputs and outputs only. Placements come from expectation and
   maximisation: the line's text extent split evenly among the characters
   the teacher read, templates averaged, placements found again by dynamic
   programming against the known string, and again until they stop
   changing. Thresholds are fitted the same way.
4. **Syrup today:** the glyph reader of `syrup::glyphs`, as MapleSyrup uses
   it, at the commit the readers are measured at.
5. **OpenCV:** Otsu, connected components and one nearest neighbour, as in
   the shakedown.
6. **Tesseract** 5, line mode, digits and the line's punctuation only.

The explicit readers learn from the train split only, and fit their
thresholds on the validation split only.

## Measures, on the dev-test frames

- Exact-line accuracy: correct, wrong and refused, as shares of all items.
- Fidelity: how often each explicit reader agrees with the teacher on the
  lines it answers.
- Cost: p50 and p95 time per line, one thread, five rounds in separate
  processes in rotated order (the shakedown's method), on this container;
  and on the owner's PC if it is reachable. Labelled exploratory: a shared
  container cannot pass G1 (amendment 2).
- Coverage of each explicit reader at zero wrong reads on validation, and
  its wrong rate on dev-test.

## What would be encouraging, what would not

- Encouraging for H1: an explicit reader that refuses few lines and reads
  none wrong, at a small fraction of the teacher's cost.
- Encouraging for H2: the white-box reader answering more lines than the
  black-box one at the same wrong rate. If the two are equal, that is the
  finding: on a task whose alignment can be recovered from outputs alone,
  the weights add nothing an EM loop cannot find.
- Either way the result is reported with the same prominence.
