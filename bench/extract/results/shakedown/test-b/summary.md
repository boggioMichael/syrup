# Syrup vs OpenCV — test data, 1 thread(s)

2026-10-12T20:51:44+03:00 · Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPUs) · Ubuntu 24.04.5 LTS · rustc 1.97.0 (2d8144b78 2026-07-07) · OpenCV 5.0.0 · 5 rounds × (3 warm-up + 5 timed calls per image; videos once per round)

Times in ms per call (library call only, warm). `OpenCV/this` > 1: this implementation is faster than OpenCV; the interval is a 95% bootstrap over inputs (photographs, source images, or blocks of 10 consecutive frame pairs).

| task | source | impl | calls | p50 | p95 | p99 | p95 range over rounds | calls/s | OpenCV/this p50 [95% CI] | OpenCV/this p95 [95% CI] | p95 vs before [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glyphs | all | opencv | 10000 | 0.075 | 0.159 | 0.226 | 0.128–0.193 | 10697.1 | – | – | – |
| glyphs | all | syrup-after | 10000 | 0.119 | 0.537 | 0.810 | 0.462–0.597 | 5572.4 | 0.63 [0.57–0.74] | 0.30 [0.27–0.34] | – |
| glyphs | all | tinynet | 10000 | 0.079 | 0.159 | 0.226 | 0.139–0.190 | 10668.0 | 0.96 [0.93–0.98] | 1.00 [0.96–1.05] | – |
| glyphs | s050/bar | opencv | 1675 | 0.069 | 0.138 | 0.176 | 0.117–0.157 | 11579.4 | – | – | – |
| glyphs | s050/bar | syrup-after | 1675 | 0.284 | 0.701 | 0.937 | 0.570–0.788 | 3101.4 | 0.24 [0.22–0.29] | 0.20 [0.17–0.23] | – |
| glyphs | s050/bar | tinynet | 1675 | 0.074 | 0.159 | 0.212 | 0.133–0.205 | 11288.5 | 0.94 [0.87–1.00] | 0.87 [0.77–0.99] | – |
| glyphs | s050/flat | opencv | 1675 | 0.067 | 0.146 | 0.212 | 0.115–0.176 | 11758.5 | – | – | – |
| glyphs | s050/flat | syrup-after | 1675 | 0.167 | 0.578 | 0.993 | 0.505–0.837 | 4315.3 | 0.40 [0.30–0.48] | 0.25 [0.21–0.29] | – |
| glyphs | s050/flat | tinynet | 1675 | 0.068 | 0.152 | 0.210 | 0.112–0.187 | 11736.7 | 0.99 [0.95–1.01] | 0.96 [0.84–1.09] | – |
| glyphs | s050/noise | opencv | 1650 | 0.066 | 0.152 | 0.220 | 0.124–0.172 | 11583.2 | – | – | – |
| glyphs | s050/noise | syrup-after | 1650 | 0.194 | 0.582 | 0.851 | 0.539–0.719 | 4083.7 | 0.34 [0.27–0.44] | 0.26 [0.22–0.33] | – |
| glyphs | s050/noise | tinynet | 1650 | 0.078 | 0.150 | 0.223 | 0.119–0.190 | 11057.5 | 0.85 [0.75–0.94] | 1.01 [0.88–1.15] | – |
| glyphs | s075/bar | opencv | 1675 | 0.083 | 0.174 | 0.254 | 0.136–0.213 | 9932.4 | – | – | – |
| glyphs | s075/bar | syrup-after | 1675 | 0.079 | 0.212 | 0.653 | 0.174–0.288 | 9755.5 | 1.05 [0.97–1.49] | 0.82 [0.50–1.02] | – |
| glyphs | s075/bar | tinynet | 1675 | 0.086 | 0.169 | 0.244 | 0.148–0.185 | 10019.8 | 0.96 [0.90–1.02] | 1.03 [0.94–1.13] | – |
| glyphs | s075/flat | opencv | 1675 | 0.078 | 0.165 | 0.223 | 0.132–0.197 | 10227.3 | – | – | – |
| glyphs | s075/flat | syrup-after | 1675 | 0.071 | 0.181 | 0.319 | 0.167–0.209 | 11471.7 | 1.11 [0.99–1.56] | 0.91 [0.67–1.03] | – |
| glyphs | s075/flat | tinynet | 1675 | 0.079 | 0.157 | 0.217 | 0.138–0.177 | 10431.1 | 0.99 [0.94–1.02] | 1.05 [0.98–1.12] | – |
| glyphs | s075/noise | opencv | 1650 | 0.099 | 0.169 | 0.234 | 0.135–0.218 | 9545.3 | – | – | – |
| glyphs | s075/noise | syrup-after | 1650 | 0.081 | 0.171 | 0.303 | 0.162–0.228 | 11413.6 | 1.23 [1.05–1.39] | 0.99 [0.90–1.14] | – |
| glyphs | s075/noise | tinynet | 1650 | 0.093 | 0.169 | 0.237 | 0.151–0.195 | 9744.6 | 1.06 [1.00–1.11] | 1.00 [0.93–1.11] | – |

## Process (median over rounds)

| task/impl | CPU time / wall | peak RSS (MiB) | after imports (MiB) |
|---|---|---|---|
| glyphs/opencv | 0.97 | 65.0 | 65.0 |
| glyphs/syrup-after | 0.97 | 5.5 | – |
| glyphs/tinynet | 0.99 | 94.6 | 77.2 |

## Initialisation (median over rounds, ms)

| task/impl | load | first call |
|---|---|---|
| glyphs/opencv | 0.888 | 0.091 |
| glyphs/syrup-after | 2.276 | 0.239 |
| glyphs/tinynet | 3.972 | 0.297 |

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
     "correct": 374,
     "wrong": 26,
     "refused": 0,
     "cases": 400,
     "accuracy": 0.935,
     "wrong_rate": 0.065,
     "refusal_rate": 0.0
    },
    "s050": {
     "correct": 174,
     "wrong": 26,
     "refused": 0,
     "cases": 200,
     "accuracy": 0.87,
     "wrong_rate": 0.13,
     "refusal_rate": 0.0
    },
    "flat": {
     "correct": 123,
     "wrong": 11,
     "refused": 0,
     "cases": 134,
     "accuracy": 0.917910447761194,
     "wrong_rate": 0.08208955223880597,
     "refusal_rate": 0.0
    },
    "bar": {
     "correct": 122,
     "wrong": 12,
     "refused": 0,
     "cases": 134,
     "accuracy": 0.9104477611940298,
     "wrong_rate": 0.08955223880597014,
     "refusal_rate": 0.0
    },
    "noise": {
     "correct": 129,
     "wrong": 3,
     "refused": 0,
     "cases": 132,
     "accuracy": 0.9772727272727273,
     "wrong_rate": 0.022727272727272728,
     "refusal_rate": 0.0
    },
    "s075": {
     "correct": 200,
     "wrong": 0,
     "refused": 0,
     "cases": 200,
     "accuracy": 1.0,
     "wrong_rate": 0.0,
     "refusal_rate": 0.0
    }
   },
   "syrup-after": {
    "all": {
     "correct": 193,
     "wrong": 0,
     "refused": 207,
     "cases": 400,
     "accuracy": 0.4825,
     "wrong_rate": 0.0,
     "refusal_rate": 0.5175
    },
    "s050": {
     "correct": 36,
     "wrong": 0,
     "refused": 164,
     "cases": 200,
     "accuracy": 0.18,
     "wrong_rate": 0.0,
     "refusal_rate": 0.82
    },
    "flat": {
     "correct": 61,
     "wrong": 0,
     "refused": 73,
     "cases": 134,
     "accuracy": 0.4552238805970149,
     "wrong_rate": 0.0,
     "refusal_rate": 0.5447761194029851
    },
    "bar": {
     "correct": 68,
     "wrong": 0,
     "refused": 66,
     "cases": 134,
     "accuracy": 0.5074626865671642,
     "wrong_rate": 0.0,
     "refusal_rate": 0.4925373134328358
    },
    "noise": {
     "correct": 64,
     "wrong": 0,
     "refused": 68,
     "cases": 132,
     "accuracy": 0.48484848484848486,
     "wrong_rate": 0.0,
     "refusal_rate": 0.5151515151515151
    },
    "s075": {
     "correct": 157,
     "wrong": 0,
     "refused": 43,
     "cases": 200,
     "accuracy": 0.785,
     "wrong_rate": 0.0,
     "refusal_rate": 0.215
    }
   },
   "tinynet": {
    "all": {
     "correct": 374,
     "wrong": 26,
     "refused": 0,
     "cases": 400,
     "accuracy": 0.935,
     "wrong_rate": 0.065,
     "refusal_rate": 0.0
    },
    "s050": {
     "correct": 174,
     "wrong": 26,
     "refused": 0,
     "cases": 200,
     "accuracy": 0.87,
     "wrong_rate": 0.13,
     "refusal_rate": 0.0
    },
    "flat": {
     "correct": 123,
     "wrong": 11,
     "refused": 0,
     "cases": 134,
     "accuracy": 0.917910447761194,
     "wrong_rate": 0.08208955223880597,
     "refusal_rate": 0.0
    },
    "bar": {
     "correct": 122,
     "wrong": 12,
     "refused": 0,
     "cases": 134,
     "accuracy": 0.9104477611940298,
     "wrong_rate": 0.08955223880597014,
     "refusal_rate": 0.0
    },
    "noise": {
     "correct": 129,
     "wrong": 3,
     "refused": 0,
     "cases": 132,
     "accuracy": 0.9772727272727273,
     "wrong_rate": 0.022727272727272728,
     "refusal_rate": 0.0
    },
    "s075": {
     "correct": 200,
     "wrong": 0,
     "refused": 0,
     "cases": 200,
     "accuracy": 1.0,
     "wrong_rate": 0.0,
     "refusal_rate": 0.0
    }
   }
  },
  "learned": {
   "opencv": [
    {
     "file": "glyphs/test/train_s050.png",
     "learned": 13
    },
    {
     "file": "glyphs/test/train_s075.png",
     "learned": 13
    }
   ],
   "syrup-after": [
    {
     "file": "glyphs/test/train_s050.png",
     "error": "learned on trial, the font reads the region as \"0123456719%. (not surely)\", not the label"
    },
    {
     "file": "glyphs/test/train_s075.png",
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
