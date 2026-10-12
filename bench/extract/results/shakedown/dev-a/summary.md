# Syrup vs OpenCV — dev data, 1 thread(s)

2026-10-12T20:50:25+03:00 · Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPUs) · Ubuntu 24.04.5 LTS · rustc 1.97.0 (2d8144b78 2026-07-07) · OpenCV 5.0.0 · 5 rounds × (3 warm-up + 5 timed calls per image; videos once per round)

Times in ms per call (library call only, warm). `OpenCV/this` > 1: this implementation is faster than OpenCV; the interval is a 95% bootstrap over inputs (photographs, source images, or blocks of 10 consecutive frame pairs).

| task | source | impl | calls | p50 | p95 | p99 | p95 range over rounds | calls/s | OpenCV/this p50 [95% CI] | OpenCV/this p95 [95% CI] | p95 vs before [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glyphs | all | opencv | 3000 | 0.100 | 0.193 | 0.234 | 0.128–0.218 | 9646.0 | – | – | – |
| glyphs | all | syrup-after | 3000 | 0.122 | 0.464 | 0.695 | 0.444–0.496 | 6102.1 | 0.81 [0.68–1.03] | 0.42 [0.30–0.58] | – |
| glyphs | all | tinynet | 3000 | 0.076 | 0.143 | 0.201 | 0.129–0.161 | 11236.5 | 1.32 [1.19–1.35] | 1.35 [1.28–1.39] | – |
| glyphs | s050/bar | opencv | 500 | 0.104 | 0.194 | 0.228 | 0.116–0.218 | 9761.9 | – | – | – |
| glyphs | s050/bar | syrup-after | 500 | 0.241 | 0.692 | 0.749 | 0.650–0.704 | 3495.4 | 0.43 [0.30–0.51] | 0.28 [0.26–0.41] | – |
| glyphs | s050/bar | tinynet | 500 | 0.079 | 0.139 | 0.175 | 0.115–0.173 | 11366.2 | 1.31 [1.05–1.35] | 1.39 [1.28–1.48] | – |
| glyphs | s050/flat | opencv | 500 | 0.082 | 0.173 | 0.224 | 0.121–0.214 | 10195.0 | – | – | – |
| glyphs | s050/flat | syrup-after | 500 | 0.166 | 0.616 | 0.691 | 0.503–0.633 | 4408.9 | 0.49 [0.30–0.79] | 0.28 [0.23–0.45] | – |
| glyphs | s050/flat | tinynet | 500 | 0.072 | 0.126 | 0.182 | 0.112–0.155 | 11988.4 | 1.13 [0.98–1.29] | 1.38 [1.16–1.44] | – |
| glyphs | s050/noise | opencv | 500 | 0.074 | 0.151 | 0.194 | 0.107–0.195 | 11415.6 | – | – | – |
| glyphs | s050/noise | syrup-after | 500 | 0.137 | 0.619 | 0.692 | 0.384–0.637 | 5488.2 | 0.54 [0.39–0.78] | 0.24 [0.21–0.55] | – |
| glyphs | s050/noise | tinynet | 500 | 0.067 | 0.130 | 0.167 | 0.107–0.136 | 12740.2 | 1.11 [1.02–1.29] | 1.16 [1.06–1.32] | – |
| glyphs | s075/bar | opencv | 500 | 0.114 | 0.205 | 0.242 | 0.128–0.236 | 8554.7 | – | – | – |
| glyphs | s075/bar | syrup-after | 500 | 0.122 | 0.228 | 0.291 | 0.163–0.289 | 8919.5 | 0.93 [0.83–1.55] | 0.90 [0.80–1.20] | – |
| glyphs | s075/bar | tinynet | 500 | 0.082 | 0.151 | 0.212 | 0.141–0.164 | 10313.2 | 1.38 [1.09–1.47] | 1.36 [1.21–1.50] | – |
| glyphs | s075/flat | opencv | 500 | 0.110 | 0.202 | 0.240 | 0.131–0.234 | 8883.6 | – | – | – |
| glyphs | s075/flat | syrup-after | 500 | 0.063 | 0.162 | 0.250 | 0.149–0.185 | 12740.6 | 1.74 [1.38–2.33] | 1.25 [1.12–1.51] | – |
| glyphs | s075/flat | tinynet | 500 | 0.081 | 0.170 | 0.217 | 0.133–0.211 | 10335.7 | 1.36 [1.10–1.42] | 1.19 [1.04–1.34] | – |
| glyphs | s075/noise | opencv | 500 | 0.107 | 0.187 | 0.237 | 0.122–0.215 | 9575.5 | – | – | – |
| glyphs | s075/noise | syrup-after | 500 | 0.081 | 0.192 | 0.316 | 0.172–0.316 | 10250.4 | 1.32 [0.98–1.77] | 0.97 [0.64–1.16] | – |
| glyphs | s075/noise | tinynet | 500 | 0.077 | 0.140 | 0.177 | 0.117–0.158 | 11065.1 | 1.39 [1.10–1.43] | 1.33 [1.10–1.41] | – |

## Process (median over rounds)

| task/impl | CPU time / wall | peak RSS (MiB) | after imports (MiB) |
|---|---|---|---|
| glyphs/opencv | 1.00 | 65.1 | 65.1 |
| glyphs/syrup-after | 0.98 | 5.5 | – |
| glyphs/tinynet | 1.00 | 94.2 | 77.1 |

## Initialisation (median over rounds, ms)

| task/impl | load | first call |
|---|---|---|
| glyphs/opencv | 0.901 | 0.106 |
| glyphs/syrup-after | 1.726 | 0.321 |
| glyphs/tinynet | 4.034 | 0.298 |

## Quality

```json
{
 "glyphs": {
  "same_output_every_round": {
   "opencv": true,
   "syrup-after": true,
   "tinynet": true
  },
  "readings": {
   "opencv": {
    "all": {
     "correct": 114,
     "wrong": 6,
     "refused": 0,
     "cases": 120,
     "accuracy": 0.95,
     "wrong_rate": 0.05,
     "refusal_rate": 0.0
    },
    "s050": {
     "correct": 54,
     "wrong": 6,
     "refused": 0,
     "cases": 60,
     "accuracy": 0.9,
     "wrong_rate": 0.1,
     "refusal_rate": 0.0
    },
    "flat": {
     "correct": 37,
     "wrong": 3,
     "refused": 0,
     "cases": 40,
     "accuracy": 0.925,
     "wrong_rate": 0.075,
     "refusal_rate": 0.0
    },
    "bar": {
     "correct": 39,
     "wrong": 1,
     "refused": 0,
     "cases": 40,
     "accuracy": 0.975,
     "wrong_rate": 0.025,
     "refusal_rate": 0.0
    },
    "noise": {
     "correct": 38,
     "wrong": 2,
     "refused": 0,
     "cases": 40,
     "accuracy": 0.95,
     "wrong_rate": 0.05,
     "refusal_rate": 0.0
    },
    "s075": {
     "correct": 60,
     "wrong": 0,
     "refused": 0,
     "cases": 60,
     "accuracy": 1.0,
     "wrong_rate": 0.0,
     "refusal_rate": 0.0
    }
   },
   "syrup-after": {
    "all": {
     "correct": 58,
     "wrong": 0,
     "refused": 62,
     "cases": 120,
     "accuracy": 0.48333333333333334,
     "wrong_rate": 0.0,
     "refusal_rate": 0.5166666666666667
    },
    "s050": {
     "correct": 14,
     "wrong": 0,
     "refused": 46,
     "cases": 60,
     "accuracy": 0.23333333333333334,
     "wrong_rate": 0.0,
     "refusal_rate": 0.7666666666666667
    },
    "flat": {
     "correct": 21,
     "wrong": 0,
     "refused": 19,
     "cases": 40,
     "accuracy": 0.525,
     "wrong_rate": 0.0,
     "refusal_rate": 0.475
    },
    "bar": {
     "correct": 20,
     "wrong": 0,
     "refused": 20,
     "cases": 40,
     "accuracy": 0.5,
     "wrong_rate": 0.0,
     "refusal_rate": 0.5
    },
    "noise": {
     "correct": 17,
     "wrong": 0,
     "refused": 23,
     "cases": 40,
     "accuracy": 0.425,
     "wrong_rate": 0.0,
     "refusal_rate": 0.575
    },
    "s075": {
     "correct": 44,
     "wrong": 0,
     "refused": 16,
     "cases": 60,
     "accuracy": 0.7333333333333333,
     "wrong_rate": 0.0,
     "refusal_rate": 0.26666666666666666
    }
   },
   "tinynet": {
    "all": {
     "correct": 114,
     "wrong": 6,
     "refused": 0,
     "cases": 120,
     "accuracy": 0.95,
     "wrong_rate": 0.05,
     "refusal_rate": 0.0
    },
    "s050": {
     "correct": 54,
     "wrong": 6,
     "refused": 0,
     "cases": 60,
     "accuracy": 0.9,
     "wrong_rate": 0.1,
     "refusal_rate": 0.0
    },
    "flat": {
     "correct": 37,
     "wrong": 3,
     "refused": 0,
     "cases": 40,
     "accuracy": 0.925,
     "wrong_rate": 0.075,
     "refusal_rate": 0.0
    },
    "bar": {
     "correct": 39,
     "wrong": 1,
     "refused": 0,
     "cases": 40,
     "accuracy": 0.975,
     "wrong_rate": 0.025,
     "refusal_rate": 0.0
    },
    "noise": {
     "correct": 38,
     "wrong": 2,
     "refused": 0,
     "cases": 40,
     "accuracy": 0.95,
     "wrong_rate": 0.05,
     "refusal_rate": 0.0
    },
    "s075": {
     "correct": 60,
     "wrong": 0,
     "refused": 0,
     "cases": 60,
     "accuracy": 1.0,
     "wrong_rate": 0.0,
     "refusal_rate": 0.0
    }
   }
  },
  "learned": {
   "opencv": [
    {
     "file": "glyphs/dev/train_s050.png",
     "learned": 13
    },
    {
     "file": "glyphs/dev/train_s075.png",
     "learned": 13
    }
   ],
   "syrup-after": [
    {
     "file": "glyphs/dev/train_s050.png",
     "error": "learned on trial, the font reads the region as \"0123456719%. (not surely)\", not the label"
    },
    {
     "file": "glyphs/dev/train_s075.png",
     "learned": 13
    }
   ],
   "tinynet": [
    {
     "file": "glyphs-int8.onnx",
     "learned": 5642
    }
   ]
  }
 }
}
```
