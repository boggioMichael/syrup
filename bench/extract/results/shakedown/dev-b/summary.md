# Syrup vs OpenCV — dev data, 1 thread(s)

2026-10-12T20:51:37+03:00 · Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPUs) · Ubuntu 24.04.5 LTS · rustc 1.97.0 (2d8144b78 2026-07-07) · OpenCV 5.0.0 · 5 rounds × (3 warm-up + 5 timed calls per image; videos once per round)

Times in ms per call (library call only, warm). `OpenCV/this` > 1: this implementation is faster than OpenCV; the interval is a 95% bootstrap over inputs (photographs, source images, or blocks of 10 consecutive frame pairs).

| task | source | impl | calls | p50 | p95 | p99 | p95 range over rounds | calls/s | OpenCV/this p50 [95% CI] | OpenCV/this p95 [95% CI] | p95 vs before [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glyphs | all | opencv | 3000 | 0.073 | 0.141 | 0.175 | 0.129–0.162 | 11262.9 | – | – | – |
| glyphs | all | syrup-after | 3000 | 0.134 | 0.499 | 0.716 | 0.446–0.553 | 5571.5 | 0.55 [0.48–0.67] | 0.28 [0.22–0.39] | – |
| glyphs | all | tinynet | 3000 | 0.101 | 0.196 | 0.243 | 0.133–0.229 | 9404.2 | 0.72 [0.71–0.78] | 0.72 [0.69–0.76] | – |
| glyphs | s050/bar | opencv | 500 | 0.084 | 0.153 | 0.189 | 0.117–0.173 | 11004.4 | – | – | – |
| glyphs | s050/bar | syrup-after | 500 | 0.251 | 0.693 | 1.075 | 0.627–0.714 | 3253.5 | 0.33 [0.26–0.44] | 0.22 [0.17–0.31] | – |
| glyphs | s050/bar | tinynet | 500 | 0.100 | 0.196 | 0.254 | 0.117–0.236 | 9533.5 | 0.84 [0.70–1.04] | 0.78 [0.67–0.86] | – |
| glyphs | s050/flat | opencv | 500 | 0.066 | 0.138 | 0.168 | 0.114–0.165 | 11925.9 | – | – | – |
| glyphs | s050/flat | syrup-after | 500 | 0.175 | 0.625 | 0.716 | 0.478–0.659 | 4187.2 | 0.38 [0.24–0.63] | 0.22 [0.20–0.33] | – |
| glyphs | s050/flat | tinynet | 500 | 0.094 | 0.186 | 0.222 | 0.113–0.213 | 10119.5 | 0.70 [0.68–0.98] | 0.74 [0.67–0.83] | – |
| glyphs | s050/noise | opencv | 500 | 0.066 | 0.126 | 0.152 | 0.112–0.133 | 11924.2 | – | – | – |
| glyphs | s050/noise | syrup-after | 500 | 0.143 | 0.632 | 0.829 | 0.368–0.753 | 4968.5 | 0.46 [0.32–0.64] | 0.20 [0.17–0.42] | – |
| glyphs | s050/noise | tinynet | 500 | 0.078 | 0.175 | 0.202 | 0.111–0.202 | 10682.4 | 0.85 [0.73–0.94] | 0.72 [0.64–0.77] | – |
| glyphs | s075/bar | opencv | 500 | 0.077 | 0.152 | 0.199 | 0.132–0.171 | 10463.5 | – | – | – |
| glyphs | s075/bar | syrup-after | 500 | 0.126 | 0.277 | 0.330 | 0.168–0.314 | 7676.5 | 0.61 [0.53–1.12] | 0.55 [0.46–0.63] | – |
| glyphs | s075/bar | tinynet | 500 | 0.114 | 0.203 | 0.239 | 0.143–0.236 | 8485.0 | 0.68 [0.64–0.99] | 0.75 [0.66–0.83] | – |
| glyphs | s075/flat | opencv | 500 | 0.075 | 0.150 | 0.173 | 0.130–0.162 | 10769.0 | – | – | – |
| glyphs | s075/flat | syrup-after | 500 | 0.075 | 0.246 | 0.313 | 0.155–0.313 | 11029.9 | 1.00 [0.92–1.50] | 0.61 [0.51–0.97] | – |
| glyphs | s075/flat | tinynet | 500 | 0.110 | 0.211 | 0.270 | 0.139–0.263 | 8677.2 | 0.68 [0.66–1.02] | 0.71 [0.62–0.74] | – |
| glyphs | s075/noise | opencv | 500 | 0.073 | 0.134 | 0.154 | 0.120–0.141 | 11668.0 | – | – | – |
| glyphs | s075/noise | syrup-after | 500 | 0.087 | 0.203 | 0.338 | 0.181–0.338 | 9214.0 | 0.83 [0.59–0.99] | 0.66 [0.41–0.78] | – |
| glyphs | s075/noise | tinynet | 500 | 0.105 | 0.192 | 0.243 | 0.122–0.239 | 9294.9 | 0.69 [0.66–0.84] | 0.70 [0.60–0.81] | – |

## Process (median over rounds)

| task/impl | CPU time / wall | peak RSS (MiB) | after imports (MiB) |
|---|---|---|---|
| glyphs/opencv | 1.00 | 65.1 | 65.1 |
| glyphs/syrup-after | 0.97 | 5.3 | – |
| glyphs/tinynet | 1.00 | 94.1 | 77.1 |

## Initialisation (median over rounds, ms)

| task/impl | load | first call |
|---|---|---|
| glyphs/opencv | 0.980 | 0.127 |
| glyphs/syrup-after | 1.752 | 0.333 |
| glyphs/tinynet | 4.101 | 0.322 |

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
