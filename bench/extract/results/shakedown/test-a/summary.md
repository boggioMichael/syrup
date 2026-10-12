# Syrup vs OpenCV — test data, 1 thread(s)

2026-10-12T20:50:36+03:00 · Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPUs) · Ubuntu 24.04.5 LTS · rustc 1.97.0 (2d8144b78 2026-07-07) · OpenCV 5.0.0 · 5 rounds × (3 warm-up + 5 timed calls per image; videos once per round)

Times in ms per call (library call only, warm). `OpenCV/this` > 1: this implementation is faster than OpenCV; the interval is a 95% bootstrap over inputs (photographs, source images, or blocks of 10 consecutive frame pairs).

| task | source | impl | calls | p50 | p95 | p99 | p95 range over rounds | calls/s | OpenCV/this p50 [95% CI] | OpenCV/this p95 [95% CI] | p95 vs before [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| glyphs | all | opencv | 10000 | 0.082 | 0.174 | 0.230 | 0.139–0.203 | 10309.6 | – | – | – |
| glyphs | all | syrup-after | 10000 | 0.125 | 0.547 | 0.886 | 0.479–0.618 | 5298.4 | 0.65 [0.59–0.75] | 0.32 [0.29–0.35] | – |
| glyphs | all | tinynet | 10000 | 0.084 | 0.175 | 0.246 | 0.143–0.221 | 10238.2 | 0.97 [0.94–1.00] | 0.99 [0.95–1.04] | – |
| glyphs | s050/bar | opencv | 1675 | 0.083 | 0.180 | 0.230 | 0.127–0.216 | 10634.9 | – | – | – |
| glyphs | s050/bar | syrup-after | 1675 | 0.300 | 0.722 | 0.999 | 0.643–0.761 | 3015.8 | 0.28 [0.24–0.33] | 0.25 [0.21–0.31] | – |
| glyphs | s050/bar | tinynet | 1675 | 0.078 | 0.171 | 0.233 | 0.128–0.196 | 10686.9 | 1.07 [0.96–1.15] | 1.06 [0.97–1.13] | – |
| glyphs | s050/flat | opencv | 1675 | 0.073 | 0.164 | 0.209 | 0.126–0.201 | 11214.6 | – | – | – |
| glyphs | s050/flat | syrup-after | 1675 | 0.172 | 0.567 | 0.963 | 0.527–0.822 | 4219.3 | 0.42 [0.33–0.53] | 0.29 [0.25–0.33] | – |
| glyphs | s050/flat | tinynet | 1675 | 0.073 | 0.157 | 0.200 | 0.117–0.179 | 11332.3 | 1.00 [0.93–1.07] | 1.05 [0.93–1.17] | – |
| glyphs | s050/noise | opencv | 1650 | 0.083 | 0.179 | 0.230 | 0.124–0.209 | 10607.7 | – | – | – |
| glyphs | s050/noise | syrup-after | 1650 | 0.198 | 0.655 | 1.109 | 0.521–0.718 | 3743.9 | 0.42 [0.33–0.52] | 0.27 [0.24–0.33] | – |
| glyphs | s050/noise | tinynet | 1650 | 0.079 | 0.168 | 0.247 | 0.131–0.188 | 10761.4 | 1.06 [0.97–1.14] | 1.07 [0.97–1.16] | – |
| glyphs | s075/bar | opencv | 1675 | 0.083 | 0.174 | 0.250 | 0.148–0.206 | 9773.7 | – | – | – |
| glyphs | s075/bar | syrup-after | 1675 | 0.081 | 0.231 | 0.638 | 0.185–0.263 | 8998.3 | 1.03 [0.96–1.26] | 0.75 [0.44–0.98] | – |
| glyphs | s075/bar | tinynet | 1675 | 0.089 | 0.197 | 0.255 | 0.141–0.242 | 9571.5 | 0.94 [0.87–1.02] | 0.88 [0.81–1.02] | – |
| glyphs | s075/flat | opencv | 1675 | 0.079 | 0.169 | 0.235 | 0.135–0.200 | 10249.3 | – | – | – |
| glyphs | s075/flat | syrup-after | 1675 | 0.072 | 0.246 | 0.316 | 0.165–0.296 | 10750.9 | 1.09 [0.98–1.44] | 0.69 [0.62–0.93] | – |
| glyphs | s075/flat | tinynet | 1675 | 0.085 | 0.184 | 0.240 | 0.150–0.224 | 9868.5 | 0.92 [0.86–0.97] | 0.92 [0.84–1.03] | – |
| glyphs | s075/noise | opencv | 1650 | 0.100 | 0.171 | 0.225 | 0.156–0.184 | 9551.0 | – | – | – |
| glyphs | s075/noise | syrup-after | 1650 | 0.081 | 0.186 | 0.293 | 0.155–0.235 | 10834.8 | 1.24 [1.02–1.38] | 0.92 [0.76–1.07] | – |
| glyphs | s075/noise | tinynet | 1650 | 0.103 | 0.184 | 0.257 | 0.144–0.241 | 9475.3 | 0.98 [0.88–1.05] | 0.93 [0.83–1.07] | – |

## Process (median over rounds)

| task/impl | CPU time / wall | peak RSS (MiB) | after imports (MiB) |
|---|---|---|---|
| glyphs/opencv | 0.99 | 65.1 | 65.1 |
| glyphs/syrup-after | 0.98 | 5.6 | – |
| glyphs/tinynet | 0.99 | 94.5 | 77.2 |

## Initialisation (median over rounds, ms)

| task/impl | load | first call |
|---|---|---|
| glyphs/opencv | 0.889 | 0.093 |
| glyphs/syrup-after | 1.844 | 0.236 |
| glyphs/tinynet | 3.963 | 0.307 |

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
