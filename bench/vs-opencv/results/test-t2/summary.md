# Syrup vs OpenCV — test data, 2 thread(s)

2026-10-12T14:34:48+03:00 · Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPUs) · Ubuntu 24.04.5 LTS · rustc 1.97.0 (2d8144b78 2026-07-07) · OpenCV 5.0.0 · 5 rounds × (3 warm-up + 5 timed calls per image; videos once per round)

Times in ms per call (library call only, warm). `OpenCV/this` > 1: this implementation is faster than OpenCV; the interval is a 95% bootstrap over inputs (photographs, source images, or blocks of 10 consecutive frame pairs).

| task | source | impl | calls | p50 | p95 | p99 | p95 range over rounds | calls/s | OpenCV/this p50 [95% CI] | OpenCV/this p95 [95% CI] | p95 vs before [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| faces | all | opencv | 375 | 109.957 | 587.288 | 662.414 | 569.836–645.496 | 6.2 | – | – | – |
| faces | all | syrup-after | 375 | 112.942 | 656.678 | 723.630 | 599.787–695.990 | 5.7 | 0.97 [0.80–1.13] | 0.89 [0.82–1.00] | -53.2% [-53.5%, -44.0%] |
| faces | all | syrup-after-gate | 375 | 84.676 | 443.417 | 518.053 | 395.716–472.939 | 7.9 | 1.30 [1.18–1.33] | 1.32 [1.18–1.45] | -68.4% [-68.4%, -56.9%] |
| faces | all | syrup-before | 375 | 208.425 | 1404.124 | 1562.155 | 1330.309–1462.180 | 3.0 | 0.53 [0.43–0.61] | 0.42 [0.42–0.53] | – |
| faces | opencv_extra | opencv | 325 | 113.176 | 590.796 | 667.820 | 570.091–655.572 | 5.7 | – | – | – |
| faces | opencv_extra | syrup-after | 325 | 115.452 | 664.407 | 724.651 | 611.667–704.039 | 5.3 | 0.98 [0.74–1.14] | 0.89 [0.82–1.00] | -53.6% [-53.6%, -43.8%] |
| faces | opencv_extra | syrup-after-gate | 325 | 91.043 | 462.656 | 519.412 | 398.695–488.022 | 7.3 | 1.24 [1.15–1.30] | 1.28 [1.18–1.45] | -67.7% [-68.4%, -56.6%] |
| faces | opencv_extra | syrup-before | 325 | 213.092 | 1430.901 | 1583.902 | 1370.351–1470.814 | 2.7 | 0.53 [0.39–0.63] | 0.41 [0.41–0.53] | – |
| faces | opencv_samples | opencv | 25 | 111.006 | 131.956 | 155.428 | 106.103–153.116 | 8.8 | – | – | – |
| faces | opencv_samples | syrup-after | 25 | 107.801 | 147.745 | 205.056 | 106.818–207.898 | 8.8 | 1.03 | 0.89 | -37.9% |
| faces | opencv_samples | syrup-after-gate | 25 | 82.143 | 92.170 | 94.105 | 75.096–92.718 | 12.4 | 1.35 | 1.43 | -61.3% |
| faces | opencv_samples | syrup-before | 25 | 212.296 | 237.982 | 245.944 | 209.419–246.360 | 4.7 | 0.52 | 0.55 | – |
| faces | syrup | opencv | 25 | 39.604 | 43.572 | 44.641 | 37.279–44.141 | 25.5 | – | – | – |
| faces | syrup | syrup-after | 25 | 37.139 | 42.455 | 54.379 | 34.794–54.073 | 26.4 | 1.07 | 1.03 | -61.4% |
| faces | syrup | syrup-after-gate | 25 | 33.668 | 45.678 | 64.338 | 30.991–65.288 | 28.3 | 1.18 | 0.95 | -58.5% |
| faces | syrup | syrup-before | 25 | 79.442 | 110.098 | 117.782 | 78.981–118.172 | 12.4 | 0.50 | 0.40 | – |
| glyphs | all | opencv | 10000 | 0.079 | 0.147 | 0.205 | 0.140–0.155 | 11149.4 | – | – | – |
| glyphs | all | syrup-after | 10000 | 0.108 | 0.501 | 0.834 | 0.406–0.596 | 5858.0 | 0.73 [0.66–0.83] | 0.29 [0.26–0.32] | -2.1% [-9.7%, +5.7%] |
| glyphs | all | syrup-before | 10000 | 0.118 | 0.511 | 0.853 | 0.433–0.612 | 5659.0 | 0.67 [0.61–0.73] | 0.29 [0.26–0.32] | – |
| glyphs | s050/bar | opencv | 1675 | 0.076 | 0.152 | 0.226 | 0.113–0.170 | 11456.6 | – | – | – |
| glyphs | s050/bar | syrup-after | 1675 | 0.263 | 0.737 | 1.215 | 0.512–1.024 | 3097.2 | 0.29 [0.26–0.33] | 0.21 [0.16–0.24] | +10.3% [-9.9%, +44.8%] |
| glyphs | s050/bar | syrup-before | 1675 | 0.275 | 0.668 | 0.867 | 0.594–0.793 | 3202.6 | 0.28 [0.25–0.31] | 0.23 [0.20–0.26] | – |
| glyphs | s050/flat | opencv | 1675 | 0.068 | 0.141 | 0.215 | 0.107–0.176 | 12033.9 | – | – | – |
| glyphs | s050/flat | syrup-after | 1675 | 0.171 | 0.521 | 0.837 | 0.451–0.646 | 4596.5 | 0.40 [0.33–0.50] | 0.27 [0.22–0.32] | -10.0% [-25.0%, +11.9%] |
| glyphs | s050/flat | syrup-before | 1675 | 0.165 | 0.579 | 0.877 | 0.484–0.795 | 4470.6 | 0.41 [0.35–0.50] | 0.24 [0.19–0.32] | – |
| glyphs | s050/noise | opencv | 1650 | 0.073 | 0.149 | 0.201 | 0.111–0.173 | 11549.2 | – | – | – |
| glyphs | s050/noise | syrup-after | 1650 | 0.178 | 0.570 | 0.836 | 0.462–0.700 | 4067.8 | 0.41 [0.33–0.51] | 0.26 [0.23–0.32] | -13.2% [-32.6%, +8.0%] |
| glyphs | s050/noise | syrup-before | 1650 | 0.189 | 0.656 | 0.998 | 0.445–0.855 | 3997.5 | 0.39 [0.32–0.48] | 0.23 [0.17–0.30] | – |
| glyphs | s075/bar | opencv | 1675 | 0.083 | 0.150 | 0.193 | 0.104–0.169 | 10711.6 | – | – | – |
| glyphs | s075/bar | syrup-after | 1675 | 0.063 | 0.178 | 0.544 | 0.154–0.222 | 11783.8 | 1.31 [1.12–1.50] | 0.84 [0.57–1.02] | -26.9% [-31.8%, -17.8%] |
| glyphs | s075/bar | syrup-before | 1675 | 0.071 | 0.243 | 0.617 | 0.170–0.298 | 9931.0 | 1.16 [1.06–1.27] | 0.62 [0.47–0.78] | – |
| glyphs | s075/flat | opencv | 1675 | 0.080 | 0.142 | 0.181 | 0.104–0.158 | 11005.1 | – | – | – |
| glyphs | s075/flat | syrup-after | 1675 | 0.058 | 0.197 | 0.246 | 0.147–0.227 | 13142.8 | 1.40 [1.18–1.75] | 0.72 [0.65–0.97] | -7.0% [-20.8%, +1.9%] |
| glyphs | s075/flat | syrup-before | 1675 | 0.066 | 0.212 | 0.281 | 0.155–0.248 | 11456.2 | 1.22 [1.10–1.31] | 0.67 [0.59–0.84] | – |
| glyphs | s075/noise | opencv | 1650 | 0.086 | 0.149 | 0.202 | 0.112–0.167 | 10307.8 | – | – | – |
| glyphs | s075/noise | syrup-after | 1650 | 0.068 | 0.153 | 0.225 | 0.138–0.189 | 13029.5 | 1.27 [1.18–1.45] | 0.97 [0.86–1.17] | -17.0% [-23.6%, -10.4%] |
| glyphs | s075/noise | syrup-before | 1650 | 0.072 | 0.185 | 0.267 | 0.129–0.230 | 11637.8 | 1.20 [1.10–1.27] | 0.81 [0.69–0.98] | – |
| motion | all | opencv | 1290 | 2.226 | 2.976 | 3.611 | 2.510–3.243 | 505.2 | – | – | – |
| motion | all | syrup-after | 1290 | 0.566 | 0.847 | 0.994 | 0.728–0.923 | 1907.1 | 3.94 [3.70–4.34] | 3.52 [3.25–3.89] | +3.0% [-2.2%, +8.1%] |
| motion | all | syrup-before | 1290 | 0.596 | 0.822 | 1.106 | 0.780–0.837 | 1836.8 | 3.73 [3.51–4.11] | 3.62 [3.46–3.92] | – |
| motion | Megamind | opencv | 485 | 2.418 | 2.956 | 3.626 | 2.748–3.246 | 401.0 | – | – | – |
| motion | Megamind | syrup-after | 485 | 0.697 | 0.925 | 1.114 | 0.749–0.953 | 1410.8 | 3.47 [3.22–3.75] | 3.19 [3.05–3.50] | +2.8% [-4.7%, +11.0%] |
| motion | Megamind | syrup-before | 485 | 0.697 | 0.900 | 1.362 | 0.795–0.932 | 1389.5 | 3.47 [3.21–3.69] | 3.29 [2.96–3.73] | – |
| motion | tree | opencv | 320 | 0.480 | 0.710 | 0.896 | 0.520–0.763 | 1890.3 | – | – | – |
| motion | tree | syrup-after | 320 | 0.164 | 0.300 | 0.444 | 0.254–0.341 | 5194.5 | 2.93 [2.25–3.41] | 2.37 [2.14–3.53] | +12.7% [-4.9%, +20.8%] |
| motion | tree | syrup-before | 320 | 0.172 | 0.266 | 0.315 | 0.257–0.296 | 5444.3 | 2.79 [2.29–3.22] | 2.67 [2.39–3.40] | – |
| motion | vtest | opencv | 485 | 2.376 | 3.158 | 3.810 | 2.186–3.345 | 412.9 | – | – | – |
| motion | vtest | syrup-after | 485 | 0.553 | 0.698 | 0.744 | 0.530–0.734 | 1789.3 | 4.30 [4.07–4.54] | 4.53 [4.29–4.83] | -12.8% [-15.4%, -8.8%] |
| motion | vtest | syrup-before | 485 | 0.596 | 0.800 | 0.845 | 0.597–0.835 | 1647.0 | 3.99 [3.79–4.36] | 3.95 [3.71–4.28] | – |
| templates | all | opencv | 12000 | 3.838 | 16.957 | 28.065 | 15.442–18.614 | 173.3 | – | – | – |
| templates | all | syrup-after | 12000 | 2.632 | 11.694 | 22.868 | 10.084–14.474 | 255.5 | 1.46 [1.34–1.61] | 1.45 [0.90–2.34] | -87.6% [-98.5%, -78.6%] |
| templates | all | syrup-before | 12000 | 3.509 | 94.276 | 1078.448 | 87.841–102.375 | 19.4 | 1.09 [0.89–1.25] | 0.18 [0.02–0.39] | – |
| templates | opencv_extra | opencv | 6500 | 3.839 | 24.408 | 32.654 | 23.858–25.092 | 144.8 | – | – | – |
| templates | opencv_extra | syrup-after | 6500 | 2.866 | 12.534 | 22.871 | 10.796–16.879 | 239.1 | 1.34 [1.22–1.70] | 1.95 [1.07–3.14] | -95.9% [-98.8%, -82.3%] |
| templates | opencv_extra | syrup-before | 6500 | 4.140 | 306.068 | 2059.234 | 259.834–313.115 | 11.8 | 0.93 [0.73–1.21] | 0.08 [0.02–0.45] | – |
| templates | opencv_samples | opencv | 5500 | 3.838 | 9.745 | 13.693 | 7.074–11.517 | 225.8 | – | – | – |
| templates | opencv_samples | syrup-after | 5500 | 2.422 | 10.084 | 22.808 | 8.495–13.103 | 278.0 | 1.58 [1.38–1.70] | 0.97 [0.49–1.49] | -79.3% [-84.6%, -73.8%] |
| templates | opencv_samples | syrup-before | 5500 | 3.068 | 48.665 | 93.507 | 44.125–52.956 | 83.8 | 1.25 [1.03–1.45] | 0.20 [0.10–0.34] | – |
| templates | side 16 | opencv | 3600 | 3.764 | 16.360 | 28.742 | 14.975–17.425 | 174.8 | – | – | – |
| templates | side 16 | syrup-after | 3600 | 3.779 | 16.483 | 22.647 | 12.038–17.444 | 183.2 | 1.00 [0.90–1.09] | 0.99 [0.91–1.37] | -98.2% [-99.4%, -83.6%] |
| templates | side 16 | syrup-before | 3600 | 21.748 | 895.378 | 3406.755 | 866.659–910.591 | 7.0 | 0.17 [0.13–0.23] | 0.02 [0.01–0.17] | – |
| templates | side 32 | opencv | 3600 | 3.731 | 17.571 | 30.175 | 15.890–20.682 | 174.5 | – | – | – |
| templates | side 32 | syrup-after | 3600 | 2.446 | 10.290 | 22.876 | 9.626–11.609 | 254.3 | 1.53 [1.29–1.65] | 1.71 [0.75–2.75] | -76.4% [-85.2%, -35.7%] |
| templates | side 32 | syrup-before | 3600 | 3.175 | 43.582 | 976.004 | 38.289–46.466 | 45.2 | 1.18 [1.01–1.31] | 0.40 [0.12–1.48] | – |
| templates | side 48 | opencv | 2400 | 3.932 | 16.957 | 26.010 | 15.403–20.500 | 173.1 | – | – | – |
| templates | side 48 | syrup-after | 2400 | 1.854 | 8.576 | 26.895 | 7.789–11.861 | 311.2 | 2.12 [1.87–2.37] | 1.98 [0.45–4.58] | -42.2% [-76.2%, +3.1%] |
| templates | side 48 | syrup-before | 2400 | 1.958 | 14.849 | 102.326 | 10.548–20.543 | 161.1 | 2.01 [1.80–2.31] | 1.14 [0.11–4.36] | – |
| templates | side 64 | opencv | 2400 | 3.845 | 18.025 | 27.243 | 14.850–21.766 | 169.3 | – | – | – |
| templates | side 64 | syrup-after | 2400 | 1.964 | 4.881 | 6.297 | 4.493–5.466 | 441.2 | 1.96 [1.82–2.31] | 3.69 [2.03–4.79] | -14.4% [-23.9%, -0.0%] |
| templates | side 64 | syrup-before | 2400 | 2.120 | 5.705 | 8.438 | 5.434–6.056 | 402.2 | 1.81 [1.68–2.19] | 3.16 [1.66–4.21] | – |
| yunet | all | opencv | 375 | 13.997 | 21.164 | 23.264 | 18.525–22.267 | 68.9 | – | – | – |
| yunet | all | syrup | 375 | 106.307 | 178.042 | 228.198 | 158.966–206.961 | 9.0 | 0.13 [0.12–0.14] | 0.12 [0.09–0.14] | – |
| yunet | opencv_extra | opencv | 325 | 13.398 | 20.511 | 22.743 | 17.578–22.081 | 71.6 | – | – | – |
| yunet | opencv_extra | syrup | 325 | 98.538 | 183.434 | 236.267 | 157.463–207.164 | 9.1 | 0.14 [0.12–0.15] | 0.11 [0.09–0.14] | – |
| yunet | opencv_samples | opencv | 25 | 18.149 | 23.335 | 27.264 | 15.510–26.616 | 54.0 | – | – | – |
| yunet | opencv_samples | syrup | 25 | 120.239 | 163.330 | 169.914 | 110.895–170.132 | 8.1 | 0.15 | 0.14 | – |
| yunet | syrup | opencv | 25 | 17.447 | 21.076 | 21.849 | 16.561–21.883 | 56.5 | – | – | – |
| yunet | syrup | syrup | 25 | 113.613 | 158.479 | 164.263 | 113.981–160.338 | 8.2 | 0.15 | 0.13 | – |

## Process (median over rounds)

| task/impl | CPU time / wall | peak RSS (MiB) | after imports (MiB) |
|---|---|---|---|
| faces/opencv | 1.90 | 135.9 | 65.3 |
| faces/syrup-after | 1.96 | 41.8 | – |
| faces/syrup-after-gate | 1.95 | 39.1 | – |
| faces/syrup-before | 1.48 | 40.5 | – |
| glyphs/opencv | 1.96 | 65.0 | 65.0 |
| glyphs/syrup-after | 0.97 | 4.4 | – |
| glyphs/syrup-before | 0.97 | 4.4 | – |
| motion/opencv | 1.16 | 207.7 | 65.0 |
| motion/syrup-after | 0.99 | 177.0 | – |
| motion/syrup-before | 0.98 | 177.1 | – |
| templates/opencv | 0.99 | 107.4 | 65.0 |
| templates/syrup-after | 0.99 | 73.3 | – |
| templates/syrup-before | 0.99 | 79.2 | – |
| yunet/opencv | 1.82 | 98.3 | 93.2 |
| yunet/syrup | 0.99 | 58.2 | – |

## Initialisation (median over rounds, ms)

| task/impl | load | first call |
|---|---|---|
| faces/opencv | 6.126 | 299.404 |
| faces/syrup-after | 0.127 | 376.930 |
| faces/syrup-after-gate | 0.124 | 230.210 |
| faces/syrup-before | 0.123 | 660.564 |
| glyphs/opencv | 1.204 | 0.131 |
| glyphs/syrup-after | 2.538 | 0.331 |
| glyphs/syrup-before | 1.764 | 0.210 |
| motion/opencv | – | 0.000 |
| motion/syrup-after | – | 0.000 |
| motion/syrup-before | – | 0.000 |
| templates/opencv | – | 7.390 |
| templates/syrup-after | – | 4.845 |
| templates/syrup-before | – | 25.185 |
| yunet/opencv | 3.608 | 40.880 |
| yunet/syrup | 0.001 | 164.418 |

## Quality

```json
{
 "faces": {
  "same_output_every_round": {
   "opencv": false,
   "syrup-after": true,
   "syrup-after-gate": true,
   "syrup-before": true
  },
  "boxes": {
   "opencv": 102,
   "syrup-after": 101,
   "syrup-after-gate": 101,
   "syrup-before": 101
  },
  "syrup-after_vs_opencv": {
   "opencv_extra": {
    "precision": 0.98989898989899,
    "recall": 0.98,
    "f1": 0.9849246231155778,
    "mean_iou": 0.9457588983185299,
    "found": 99,
    "reference": 100
   },
   "all": {
    "precision": 0.9900990099009901,
    "recall": 0.9803921568627451,
    "f1": 0.9852216748768472,
    "mean_iou": 0.945530824210627,
    "found": 101,
    "reference": 102
   },
   "syrup": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "mean_iou": 0.9365244536940687,
    "found": 1,
    "reference": 1
   },
   "opencv_samples": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "mean_iou": 0.9321859321527063,
    "found": 1,
    "reference": 1
   }
  },
  "syrup-after-gate_vs_opencv": {
   "opencv_extra": {
    "precision": 0.98989898989899,
    "recall": 0.98,
    "f1": 0.9849246231155778,
    "mean_iou": 0.9457588983185299,
    "found": 99,
    "reference": 100
   },
   "all": {
    "precision": 0.9900990099009901,
    "recall": 0.9803921568627451,
    "f1": 0.9852216748768472,
    "mean_iou": 0.945530824210627,
    "found": 101,
    "reference": 102
   },
   "syrup": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "mean_iou": 0.9365244536940687,
    "found": 1,
    "reference": 1
   },
   "opencv_samples": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "mean_iou": 0.9321859321527063,
    "found": 1,
    "reference": 1
   }
  },
  "syrup-before_vs_opencv": {
   "opencv_extra": {
    "precision": 0.98989898989899,
    "recall": 0.98,
    "f1": 0.9849246231155778,
    "mean_iou": 0.9457588983185299,
    "found": 99,
    "reference": 100
   },
   "all": {
    "precision": 0.9900990099009901,
    "recall": 0.9803921568627451,
    "f1": 0.9852216748768472,
    "mean_iou": 0.945530824210627,
    "found": 101,
    "reference": 102
   },
   "syrup": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "mean_iou": 0.9365244536940687,
    "found": 1,
    "reference": 1
   },
   "opencv_samples": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "mean_iou": 0.9321859321527063,
    "found": 1,
    "reference": 1
   }
  },
  "syrup-after_vs_before": {
   "precision": 1.0,
   "recall": 1.0,
   "f1": 1.0,
   "identical_inputs": 15,
   "inputs": 15,
   "box_count_change": 0.0
  },
  "syrup-after-gate_vs_before": {
   "precision": 1.0,
   "recall": 1.0,
   "f1": 1.0,
   "identical_inputs": 15,
   "inputs": 15,
   "box_count_change": 0.0
  }
 },
 "glyphs": {
  "same_output_every_round": {
   "opencv": true,
   "syrup-after": true,
   "syrup-before": true
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
   "syrup-before": {
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
   "syrup-before": [
    {
     "file": "glyphs/test/train_s050.png",
     "error": "learned on trial, the font reads the region as \"0123456719%. (not surely)\", not the label"
    },
    {
     "file": "glyphs/test/train_s075.png",
     "learned": 13
    }
   ]
  }
 },
 "motion": {
  "same_output_every_round": {
   "opencv": true,
   "syrup-after": true,
   "syrup-before": true
  },
  "syrup-after_mask_vs_opencv": {
   "Megamind": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 100,
    "pairs": 100
   },
   "all": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 267,
    "pairs": 267
   },
   "tree": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 67,
    "pairs": 67
   },
   "vtest": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 100,
    "pairs": 100
   }
  },
  "syrup-after_regions_vs_opencv": {
   "Megamind": {
    "precision": 0.37277928334838906,
    "recall": 0.5399040558220671,
    "f1": 0.4410402565016031,
    "found": 3321,
    "reference": 2293
   },
   "all": {
    "precision": 0.38996763754045305,
    "recall": 0.6043129388164493,
    "f1": 0.47403619197482294,
    "found": 6180,
    "reference": 3988
   },
   "tree": {
    "precision": 0.22541966426858512,
    "recall": 0.4973544973544973,
    "f1": 0.3102310231023102,
    "found": 417,
    "reference": 189
   },
   "vtest": {
    "precision": 0.44144144144144143,
    "recall": 0.7158034528552457,
    "f1": 0.5460992907801419,
    "found": 2442,
    "reference": 1506
   }
  },
  "syrup-before_mask_vs_opencv": {
   "Megamind": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 100,
    "pairs": 100
   },
   "all": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 267,
    "pairs": 267
   },
   "tree": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 67,
    "pairs": 67
   },
   "vtest": {
    "pooled_iou": 1.0,
    "min_iou": 1.0,
    "identical_pairs": 100,
    "pairs": 100
   }
  },
  "syrup-before_regions_vs_opencv": {
   "Megamind": {
    "precision": 0.37277928334838906,
    "recall": 0.5399040558220671,
    "f1": 0.4410402565016031,
    "found": 3321,
    "reference": 2293
   },
   "all": {
    "precision": 0.38996763754045305,
    "recall": 0.6043129388164493,
    "f1": 0.47403619197482294,
    "found": 6180,
    "reference": 3988
   },
   "tree": {
    "precision": 0.22541966426858512,
    "recall": 0.4973544973544973,
    "f1": 0.3102310231023102,
    "found": 417,
    "reference": 189
   },
   "vtest": {
    "precision": 0.44144144144144143,
    "recall": 0.7158034528552457,
    "f1": 0.5460992907801419,
    "found": 2442,
    "reference": 1506
   }
  },
  "after_vs_before_identical_outputs": [
   258,
   258
  ]
 },
 "templates": {
  "same_output_every_round": {
   "opencv": true,
   "syrup-after": true,
   "syrup-before": true
  },
  "hit_rate": {
   "opencv": {
    "all": {
     "hits": 480,
     "cases": 480,
     "rate": 1.0,
     "refused": 0
    },
    "clean": {
     "hits": 240,
     "cases": 240,
     "rate": 1.0,
     "refused": 0
    },
    "side 16": {
     "hits": 144,
     "cases": 144,
     "rate": 1.0,
     "refused": 0
    },
    "opencv_samples": {
     "hits": 220,
     "cases": 220,
     "rate": 1.0,
     "refused": 0
    },
    "perturbed": {
     "hits": 240,
     "cases": 240,
     "rate": 1.0,
     "refused": 0
    },
    "side 32": {
     "hits": 144,
     "cases": 144,
     "rate": 1.0,
     "refused": 0
    },
    "side 48": {
     "hits": 96,
     "cases": 96,
     "rate": 1.0,
     "refused": 0
    },
    "side 64": {
     "hits": 96,
     "cases": 96,
     "rate": 1.0,
     "refused": 0
    },
    "opencv_extra": {
     "hits": 260,
     "cases": 260,
     "rate": 1.0,
     "refused": 0
    }
   },
   "syrup-after": {
    "all": {
     "hits": 474,
     "cases": 480,
     "rate": 0.9875,
     "refused": 0
    },
    "clean": {
     "hits": 237,
     "cases": 240,
     "rate": 0.9875,
     "refused": 0
    },
    "side 16": {
     "hits": 144,
     "cases": 144,
     "rate": 1.0,
     "refused": 0
    },
    "opencv_samples": {
     "hits": 218,
     "cases": 220,
     "rate": 0.990909090909091,
     "refused": 0
    },
    "perturbed": {
     "hits": 237,
     "cases": 240,
     "rate": 0.9875,
     "refused": 0
    },
    "side 32": {
     "hits": 140,
     "cases": 144,
     "rate": 0.9722222222222222,
     "refused": 0
    },
    "side 48": {
     "hits": 94,
     "cases": 96,
     "rate": 0.9791666666666666,
     "refused": 0
    },
    "side 64": {
     "hits": 96,
     "cases": 96,
     "rate": 1.0,
     "refused": 0
    },
    "opencv_extra": {
     "hits": 256,
     "cases": 260,
     "rate": 0.9846153846153847,
     "refused": 0
    }
   },
   "syrup-before": {
    "all": {
     "hits": 474,
     "cases": 480,
     "rate": 0.9875,
     "refused": 0
    },
    "clean": {
     "hits": 237,
     "cases": 240,
     "rate": 0.9875,
     "refused": 0
    },
    "side 16": {
     "hits": 144,
     "cases": 144,
     "rate": 1.0,
     "refused": 0
    },
    "opencv_samples": {
     "hits": 218,
     "cases": 220,
     "rate": 0.990909090909091,
     "refused": 0
    },
    "perturbed": {
     "hits": 237,
     "cases": 240,
     "rate": 0.9875,
     "refused": 0
    },
    "side 32": {
     "hits": 140,
     "cases": 144,
     "rate": 0.9722222222222222,
     "refused": 0
    },
    "side 48": {
     "hits": 94,
     "cases": 96,
     "rate": 0.9791666666666666,
     "refused": 0
    },
    "side 64": {
     "hits": 96,
     "cases": 96,
     "rate": 1.0,
     "refused": 0
    },
    "opencv_extra": {
     "hits": 256,
     "cases": 260,
     "rate": 0.9846153846153847,
     "refused": 0
    }
   }
  },
  "vs_opencv": {
   "syrup-after": {
    "same_position": 474,
    "cases": 480,
    "score_abs_diff_mean": 0.000502943635833421,
    "score_abs_diff_max": 0.05580630197677616
   },
   "syrup-before": {
    "same_position": 474,
    "cases": 480,
    "score_abs_diff_mean": 0.000502943635833421,
    "score_abs_diff_max": 0.05580630197677616
   }
  },
  "after_vs_before_identical_outputs": [
   480,
   480
  ]
 },
 "yunet": {
  "same_output_every_round": {
   "opencv": true,
   "syrup": true
  },
  "syrup_vs_opencv": {
   "precision": 1.0,
   "recall": 1.0,
   "f1": 1.0,
   "mean_iou": 0.9999999999999999,
   "mean_landmark_error_px": 0.0,
   "max_score_abs_diff": 0.0,
   "found": 102,
   "reference": 102
  }
 }
}
```
