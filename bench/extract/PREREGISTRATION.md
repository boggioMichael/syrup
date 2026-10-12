# Pre-registration: from weights to explicit algorithms

**Status:** proposed on 12 October 2026, before any data for this research was
collected or measured. It locks at gate G0, when the owner signs it off
together with [LICENSES.md](LICENSES.md) and [DATA.md](DATA.md). After that,
every change goes to [AMENDMENTS.md](AMENDMENTS.md) with its reason. No test
data exists yet, so nothing here can have been tuned on it.

The plan, phases and dates are in the roadmap:
https://claude.ai/code/artifact/1132d900-351a-43cd-89be-a3d05c8080fd

## Hypotheses

- **H1, replacement.** On tasks A and B, explicit code answers almost every
  item by itself, makes no wrong reads within the budget below, and with a
  fallback model behind it costs far less per item than the model alone.
- **H2, the weights help.** With the same teacher, data and budget, an
  extraction agent that can read the teacher's activations and weights
  (white box) produces explicit code with higher coverage than one that sees
  only the teacher's inputs and outputs (black box).
- **H3, a decompiler.** A model trained on pairs of a Syrup program and the
  network compiled from it recovers an equivalent program from networks that
  were trained, not compiled.

H1 is the product. H2 is the research question and is expected to be the
hardest to confirm. H3 runs only if G4 passes.

## Tasks

| Task | Unit | Output | Wrong means |
| --- | --- | --- | --- |
| A, HUD text | one HUD field's text line (HP, MP, EXP, level, mesos), as the layout locates it | the exact string, or a refusal | an accepted string that differs from the label in any character |
| B, icons | one icon crop from the skill bar or the buff area | the icon's label, or a refusal | an accepted label that differs |
| C, objects | one tile of the change map (T3) in a frame where it changed | every object in the tile (box and class), or a refusal | an accepted tile with any missed, extra or misclassified object (IoU below 0.5 or wrong class) |

Task C's classes, written before labelling: monster, own character, other
character, NPC, dropped item, portal.

## Implementations compared

1. **Explicit code**: what P3 (black box), P4 (white box) and P5 (search)
   produce, each version identified by its commit.
2. **Teacher**: our own model from P2, trained from an architecture cleared in
   LICENSES.md.
3. **Tiny network**: the smallest network in a fixed family that reaches the
   teacher's dev accuracy within 1 point, trained on the teacher's labels.
   The family is an MLP with one hidden layer of 32 to 512 units, or a CNN with
   2 to 4 convolutions of 8 to 64 channels. It is quantised to INT8 and runs
   single-threaded in ONNX Runtime, or in Syrup's runtime once K7 lands.
4. **OpenCV**: the best pipeline found on dev. For A, Otsu thresholding,
   connected components and k-NN, as in the OpenCV benchmark (F5). For B,
   normalised template matching. For C there is no comparable OpenCV pipeline,
   and this is reported as such.
5. **Syrup today**: the commit at the start of P1.

## Measures

**Accuracy** (as shares of all items): correct, wrong and refused; for C also
recall and precision at IoU 0.5, per class and pooled.

**Cost**, per item:

- Wall time p50, p95 and p99: one thread, five rounds in separate processes,
  rotated order, as in the OpenCV benchmark.
- CPU time and peak memory.
- Instructions per item, from callgrind on a fixed subsample of 100 items.
- Energy per item where RAPL counters exist; otherwise "not measured: no RAPL",
  never an estimate.

**System measures** for explicit code with a fallback F behind it:

- coverage = items the explicit code answers ÷ all items
- system accuracy = (explicit correct + F correct on the items explicit refused) ÷ all items
- system wrong rate = (explicit wrong + F wrong on the refused items) ÷ all items
- expected cost per item = mean over items of (explicit cost + F's cost when explicit refused)

Costs are paired per item, measured on the same machine in the same run.

## Statistics

- 95% confidence intervals by bootstrap with 10,000 resamples. Items are
  resampled by session when a set has 8 or more sessions, and otherwise in
  blocks of 10 consecutive frames.
- Speed comparisons are ratios of p50 and of p95, with bootstrap intervals, as
  in the OpenCV benchmark.
- A zero-error claim uses the rule of three: no errors in n items bounds the
  error rate below 3/n at 95% confidence.
- Only the claims listed under the gates are confirmatory. Everything else,
  including the P1 shakedown, is exploratory and labelled so.

## Data

Dev data, sealed sets and labelling are in [DATA.md](DATA.md). Four sealed
sets are used, each for one decision and never for tuning:

| Set | Decides | Size |
| --- | --- | --- |
| S1 | G2, the teacher | 1,000 lines (A), 1,000 icons (B), 200 frames (C) |
| S2 | G3, H1; G5 reuses it for timing only, with outputs that must equal the G3 outputs | 3,000 lines, 3,000 icons, 1,000 frames |
| S3 | G4, H2 | 1,000 frames (C) |
| S4 | G8, the final run | 3,000 lines, 3,000 icons, 1,000 frames |

## Gates

| Gate | Passes when |
| --- | --- |
| G0 | The owner signs off this file, LICENSES.md and DATA.md. |
| G1 | On dev, every baseline gives the same outputs in two separate runs, its p50 differs by less than 5% between them, and its p95 interval is narrower than ±10% of the estimate. |
| G2 | On S1, the teacher reaches 99.5% exact strings on A, 99.5% on B, and recall and precision of at least 95% on C. |
| G3 (H1) | On S2, for A and for B, each on its own: explicit code makes 0 wrong reads in 3,000; refuses at most 2%; with the tiny network as fallback, system accuracy is at least 99.5%; and explicit p95 is below both the tiny network's and OpenCV's, with the ratio's interval excluding 1. |
| G4 (H2) | On S3, white-box coverage minus black-box coverage on C has a 95% interval above 0, while the white-box wrong rate stays at or below 0.5% of tiles. |
| G5 | On S2 timing, P5's code is at least 1.5 times faster at p95 than P3's on A or on B (interval above 1.2), not slower on the other, and its outputs equal P3's on every item. |
| G6 (H3) | At least 30% of 50 held-out trained networks decompile into programs that match the network on at least 99% of 10,000 held-out inputs. |
| G7 | In shadow mode, with players who opted in: coverage at least 95% on A and B and 80% on C; wrong rate, as judged by a person on every disagreement, at most 0.5%; and expected cost on C at most 10% of the model alone. |
| G8 | On S4, H1 to H3 are each marked confirmed, refuted or not tested, with the numbers. |

A gate that fails stops its line of work; the reason is written in the
results, at the same prominence as a pass. A failed G4 ends P6. A failed G3
sends P3 back once, with an amendment that names what changes; a second
failure refutes H1 for that task.

## How H2 is run

- Both arms start from the same commit, teacher, dev data and student
  families, and get the same budget: 8 agent-hours and the same token cap per
  run.
- The white-box arm also gets activations, weights and ablation tools. Nothing
  else differs.
- Each arm runs three times, as independent sessions that share no memory or
  notes. A person may fix infrastructure but does not touch the code either
  arm writes.
- The primary measure is the mean coverage over the three runs. All six runs
  are reported.

## How H3 is run

- A compiler turns programs written with Syrup's explicit primitives
  (thresholds, templates, glyph tables, cascades) into small convolutional
  networks. At least 100,000 random programs give the training pairs.
- The decompiler is tested twice: on held-out compiled networks (the published
  result to compare with), and on the 50 networks trained by gradient descent
  on dev tasks, sized like the compiled ones. Only the second decides G6.
