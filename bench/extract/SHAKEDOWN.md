# P1 shakedown: glyph reading

**Exploratory.** This run decides no gate. It tries the P1 tools (the
tiny-network baseline, the explicit-first measures and the G1 check) on the
synthetic glyph lines of the OpenCV benchmark. Those lines are Hershey-font
digits, not MapleStory's, and their 400-line split was already used by that
benchmark, so here they are dev data ([DATA.md](DATA.md)).

Every number below is in [results/shakedown](results/shakedown).

## What ran

| Reader | What it is |
| --- | --- |
| Syrup | the `glyphs` reader of the binary measured in the OpenCV benchmark (sha256 `b8c20bb3…`) |
| OpenCV | Otsu threshold, connected components, one nearest neighbour ([opencv_side.py](../vs-opencv/opencv_side.py)) |
| Tiny network | the same segmentation, then an MLP with one hidden layer of 16 units (4,333 parameters), INT8, in ONNX Runtime 1.29.0 ([tinynet.py](tinynet.py), model sha256 `c7d49d8e…`, 5,223 bytes) |

The tiny network learned from the same two training lines as the others, with
generic augmentation: of the 2 lines and their 600 augmented copies, 434
segmented cleanly, giving 5,642 cells. Hidden sizes of 16, 32, 64 and 128 all read 95.0% of the dev
lines, so the smallest was kept. Its INT8 version reads the same 95.0%, and
ONNX Runtime's answers equal a NumPy re-computation of the integer
arithmetic.

The machine was an Intel Xeon at 2.10 GHz, with two shared vCPUs in a cloud
container running Ubuntu 24.04. Each reader used one thread, with 5 rounds of
3 warm-up and 5 timed calls per line, in rotated order. The whole measurement
ran twice (runs A and B).

## Accuracy and speed

The 400-line split, run A:

| Reader | Correct | Wrong | Refused | p50 ms | p95 ms | OpenCV ÷ this, p50 [95% CI] | Peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OpenCV | 374 (93.5%) | 26 | 0 | 0.082 | 0.174 | – | 65.1 |
| Tiny network | 374 (93.5%) | 26 | 0 | 0.084 | 0.175 | 0.97 [0.94–1.00] | 94.5 |
| Syrup | 193 (48.2%) | 0 | 207 | 0.125 | 0.547 | 0.65 [0.59–0.75] | 5.6 |

Peak RSS is the whole process. For the two Python readers it includes the
interpreter, NumPy and OpenCV, and for the tiny network ONNX Runtime too.

- The tiny network gave OpenCV's exact output on 387 of 400 lines. The 13 it
  read differently, both read wrong.
- All 26 wrong reads of both learned readers are on the small size (cap
  height about 11 pixels), where touching glyphs merge in the shared
  segmentation. On the larger size both read all 200 lines right.
- Syrup refused 164 of the 200 small lines and 43 of the 200 larger ones.

## Explicit first, a fallback behind it

The same 400 lines, run A ([cascade.py](cascade.py)):

| Explicit → fallback | Coverage | System correct | System wrong | Fallback alone, correct | Gain, points (paired) | System ms per line | Fallback alone ms per line | Fallback ÷ system cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Syrup → OpenCV | 48.2% | 95.5% | 4.5% | 93.5% | +2.0 [+0.8, +3.5] | 0.202 | 0.085 | 0.42 [0.39–0.46] |
| Syrup → tiny network | 48.2% | 95.5% | 4.5% | 93.5% | +2.0 [+0.8, +3.5] | 0.202 | 0.085 | 0.42 [0.39–0.46] |

Of OpenCV's 26 wrong lines, Syrup read 8 correctly and refused 18. That is the
whole gain: 18 wrong lines are left, not 26. The cost is 2.4 times the fallback
alone, because Syrup refuses half the lines and is slow on the small size.
Run B gives the same accuracy and a ratio of 0.43 [0.39–0.46].

## Run to run: the G1 check

[repeat_check.py](repeat_check.py) compares runs A and B:

| Reader | Same outputs | p50 change, 400 lines | p50 change, 120 dev lines | G1 |
| --- | --- | --- | --- | --- |
| OpenCV | yes | −7.9% | −26.7% | fails |
| Tiny network | yes | −6.5% | +33.0% | fails |
| Syrup | yes | −4.9% | +9.5% | passes on 400 lines only |

The outputs repeat exactly. The timings do not: with no code change, the
OpenCV reader's p50 moved by 27% between two runs on this shared machine. A
second pair of runs, from the reproduce script, gave a different verdict: on
400 lines OpenCV (+0.4%) and the tiny network (−1.4%) passed and Syrup (+5.2%)
failed ([repro-test-g1.md](results/shakedown/repro-test-g1.md)). G1 asks for
under 5%; here, passing it depends on which run you look at, so this machine
cannot decide it.

## What this changes

1. **H1's cost condition fails on this task today.** Explicit first is more
   accurate, but costs 2.4 times more per line. Before H1 can pass, Syrup's
   reader needs higher coverage and speed on small text: T6 and P3.
2. **The bottleneck is segmentation, not the classifier.** A learned
   classifier added nothing over one nearest neighbour. So for task A, the
   best learned baseline should read the whole line without cutting glyphs
   first. This is proposed in [AMENDMENTS.md](AMENDMENTS.md) for G0.
3. **G1 needs a dedicated machine.** Also proposed in AMENDMENTS.md.

## Limits

- Synthetic lines in OpenCV's Hershey font, not MapleStory's fonts.
- The Python readers' times include Python's per-call overhead (about 1 µs
  per OpenCV call, measured in the benchmark); Syrup's are the Rust call alone.
- One shared machine, one thread.

## Reproduce

[shakedown.sh](shakedown.sh) trains the network, measures twice, and writes
every file above.
