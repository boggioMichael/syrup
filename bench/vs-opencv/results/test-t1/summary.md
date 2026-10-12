# Syrup vs OpenCV — test data, 1 thread(s)

2026-10-12T13:58:59+03:00 · Intel(R) Xeon(R) Processor @ 2.10GHz (2 CPUs) · Ubuntu 24.04.5 LTS · rustc 1.97.0 (2d8144b78 2026-07-07) · OpenCV 5.0.0 · 5 rounds × (3 warm-up + 5 timed calls per image; videos once per round)

Times in ms per call (library call only, warm). `OpenCV/this` > 1: this implementation is faster than OpenCV; the interval is a 95% bootstrap over inputs (photographs, source images, or blocks of 10 consecutive frame pairs).

| task | source | impl | calls | p50 | p95 | p99 | p95 range over rounds | calls/s | OpenCV/this p50 [95% CI] | OpenCV/this p95 [95% CI] | p95 vs before [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| faces | all | opencv | 375 | 199.790 | 1150.340 | 1227.478 | 1117.307–1179.750 | 3.3 | – | – | – |
| faces | all | syrup-after | 375 | 229.697 | 1281.112 | 1604.297 | 1184.728–1424.897 | 2.9 | 0.87 [0.73–1.06] | 0.90 [0.78–1.00] | -31.3% [-32.9%, -25.9%] |
| faces | all | syrup-after-gate | 375 | 167.397 | 892.701 | 1059.307 | 844.642–914.972 | 4.0 | 1.19 [1.14–1.28] | 1.29 [1.13–1.36] | -52.1% [-57.2%, -39.9%] |
| faces | all | syrup-before | 375 | 301.933 | 1864.495 | 2142.658 | 1660.917–2116.585 | 2.1 | 0.66 [0.55–0.83] | 0.62 [0.57–0.69] | – |
| faces | opencv_extra | opencv | 325 | 208.790 | 1157.993 | 1229.586 | 1126.014–1180.302 | 3.0 | – | – | – |
| faces | opencv_extra | syrup-after | 325 | 244.186 | 1318.319 | 1651.705 | 1227.072–1430.604 | 2.6 | 0.86 [0.67–1.10] | 0.88 [0.78–1.00] | -31.3% [-31.3%, -26.0%] |
| faces | opencv_extra | syrup-after-gate | 325 | 175.020 | 909.388 | 1072.736 | 851.766–953.981 | 3.7 | 1.19 [1.14–1.32] | 1.27 [1.11–1.36] | -52.6% [-57.9%, -39.1%] |
| faces | opencv_extra | syrup-before | 325 | 305.501 | 1919.595 | 2160.612 | 1761.409–2128.166 | 1.9 | 0.68 [0.53–0.86] | 0.60 [0.57–0.69] | – |
| faces | opencv_samples | opencv | 25 | 197.637 | 219.318 | 245.868 | 203.439–246.717 | 5.0 | – | – | – |
| faces | opencv_samples | syrup-after | 25 | 228.130 | 260.282 | 277.201 | 229.089–277.555 | 4.4 | 0.87 | 0.84 | -29.5% |
| faces | opencv_samples | syrup-after-gate | 25 | 163.897 | 207.143 | 213.303 | 137.100–206.253 | 6.2 | 1.21 | 1.06 | -43.9% |
| faces | opencv_samples | syrup-before | 25 | 325.048 | 369.175 | 383.797 | 337.278–384.509 | 3.1 | 0.61 | 0.59 | – |
| faces | syrup | opencv | 25 | 78.510 | 90.183 | 92.215 | 78.133–90.893 | 12.9 | – | – | – |
| faces | syrup | syrup-after | 25 | 73.372 | 97.528 | 99.377 | 69.169–99.474 | 13.2 | 1.07 | 0.92 | -11.5% |
| faces | syrup | syrup-after-gate | 25 | 68.094 | 100.189 | 102.766 | 66.205–100.543 | 14.4 | 1.15 | 0.90 | -9.1% |
| faces | syrup | syrup-before | 25 | 96.573 | 110.234 | 119.586 | 77.529–118.422 | 10.9 | 0.81 | 0.82 | – |
| glyphs | all | opencv | 10000 | 0.062 | 0.123 | 0.162 | 0.088–0.141 | 13607.9 | – | – | – |
| glyphs | all | syrup-after | 10000 | 0.113 | 0.481 | 0.788 | 0.395–0.678 | 5999.6 | 0.55 [0.50–0.59] | 0.26 [0.24–0.28] | +10.3% [+4.6%, +17.3%] |
| glyphs | all | syrup-before | 10000 | 0.100 | 0.436 | 0.669 | 0.307–0.529 | 6601.1 | 0.62 [0.55–0.67] | 0.28 [0.25–0.33] | – |
| glyphs | s050/bar | opencv | 1675 | 0.055 | 0.117 | 0.149 | 0.080–0.139 | 14328.3 | – | – | – |
| glyphs | s050/bar | syrup-after | 1675 | 0.260 | 0.675 | 0.937 | 0.505–0.896 | 3330.9 | 0.21 [0.20–0.25] | 0.17 [0.15–0.21] | +16.7% [+7.3%, +22.3%] |
| glyphs | s050/bar | syrup-before | 1675 | 0.237 | 0.578 | 0.916 | 0.399–0.671 | 3637.4 | 0.23 [0.21–0.27] | 0.20 [0.17–0.24] | – |
| glyphs | s050/flat | opencv | 1675 | 0.055 | 0.109 | 0.137 | 0.084–0.126 | 15471.3 | – | – | – |
| glyphs | s050/flat | syrup-after | 1675 | 0.153 | 0.493 | 0.846 | 0.388–0.832 | 4871.0 | 0.36 [0.28–0.43] | 0.22 [0.20–0.25] | +3.0% [-1.7%, +21.3%] |
| glyphs | s050/flat | syrup-before | 1675 | 0.139 | 0.479 | 0.643 | 0.358–0.601 | 5350.9 | 0.39 [0.30–0.49] | 0.23 [0.21–0.29] | – |
| glyphs | s050/noise | opencv | 1650 | 0.055 | 0.121 | 0.167 | 0.079–0.145 | 14353.7 | – | – | – |
| glyphs | s050/noise | syrup-after | 1650 | 0.180 | 0.542 | 0.924 | 0.420–0.905 | 4364.4 | 0.31 [0.25–0.38] | 0.22 [0.20–0.26] | +11.0% [-0.1%, +22.5%] |
| glyphs | s050/noise | syrup-before | 1650 | 0.161 | 0.488 | 0.690 | 0.368–0.652 | 4840.9 | 0.34 [0.29–0.43] | 0.25 [0.20–0.30] | – |
| glyphs | s075/bar | opencv | 1675 | 0.066 | 0.132 | 0.180 | 0.096–0.164 | 12467.6 | – | – | – |
| glyphs | s075/bar | syrup-after | 1675 | 0.073 | 0.226 | 0.555 | 0.127–0.281 | 10316.5 | 0.91 [0.84–1.06] | 0.58 [0.39–0.75] | +22.8% [+8.3%, +36.9%] |
| glyphs | s075/bar | syrup-before | 1675 | 0.067 | 0.184 | 0.600 | 0.145–0.202 | 11275.9 | 0.98 [0.87–1.20] | 0.72 [0.52–0.86] | – |
| glyphs | s075/flat | opencv | 1675 | 0.063 | 0.122 | 0.158 | 0.093–0.135 | 13083.9 | – | – | – |
| glyphs | s075/flat | syrup-after | 1675 | 0.065 | 0.184 | 0.289 | 0.148–0.265 | 11956.9 | 0.97 [0.88–1.07] | 0.66 [0.55–0.78] | +1.7% [-6.4%, +16.1%] |
| glyphs | s075/flat | syrup-before | 1675 | 0.055 | 0.181 | 0.224 | 0.150–0.198 | 13079.6 | 1.14 [0.92–1.25] | 0.67 [0.60–0.80] | – |
| glyphs | s075/noise | opencv | 1650 | 0.068 | 0.135 | 0.162 | 0.095–0.152 | 12464.1 | – | – | – |
| glyphs | s075/noise | syrup-after | 1650 | 0.072 | 0.183 | 0.270 | 0.125–0.258 | 11825.2 | 0.94 [0.86–1.10] | 0.74 [0.61–0.90] | +22.0% [+7.9%, +33.8%] |
| glyphs | s075/noise | syrup-before | 1650 | 0.070 | 0.150 | 0.197 | 0.141–0.187 | 13314.9 | 0.97 [0.87–1.18] | 0.90 [0.77–1.09] | – |
| motion | all | opencv | 1290 | 1.857 | 2.457 | 2.798 | 2.283–2.596 | 621.8 | – | – | – |
| motion | all | syrup-after | 1290 | 0.577 | 0.876 | 0.939 | 0.757–0.902 | 1838.7 | 3.22 [2.79–3.49] | 2.81 [2.69–3.08] | -4.0% [-11.8%, -1.9%] |
| motion | all | syrup-before | 1290 | 0.589 | 0.912 | 1.117 | 0.773–1.066 | 1796.2 | 3.16 [2.78–3.42] | 2.69 [2.43–2.91] | – |
| motion | Megamind | opencv | 485 | 1.745 | 2.330 | 2.661 | 1.747–2.644 | 548.9 | – | – | – |
| motion | Megamind | syrup-after | 485 | 0.719 | 0.904 | 1.211 | 0.787–0.931 | 1372.2 | 2.43 [2.25–2.65] | 2.58 [2.38–2.74] | -11.3% [-18.0%, -3.2%] |
| motion | Megamind | syrup-before | 485 | 0.719 | 1.019 | 1.173 | 0.813–1.140 | 1352.2 | 2.43 [2.25–2.64] | 2.29 [2.10–2.54] | – |
| motion | tree | opencv | 320 | 0.380 | 0.540 | 0.718 | 0.302–0.706 | 2534.0 | – | – | – |
| motion | tree | syrup-after | 320 | 0.181 | 0.301 | 0.444 | 0.253–0.325 | 4949.0 | 2.10 [1.79–2.36] | 1.79 [1.67–2.24] | +3.7% [-7.8%, +8.5%] |
| motion | tree | syrup-before | 320 | 0.188 | 0.290 | 0.375 | 0.245–0.351 | 5137.2 | 2.03 [1.82–2.44] | 1.86 [1.59–2.13] | – |
| motion | vtest | opencv | 485 | 2.220 | 2.550 | 2.968 | 2.337–2.702 | 455.4 | – | – | – |
| motion | vtest | syrup-after | 485 | 0.566 | 0.778 | 0.872 | 0.602–0.873 | 1710.9 | 3.92 [3.74–4.12] | 3.28 [3.00–3.73] | -7.8% [-16.4%, -3.3%] |
| motion | vtest | syrup-before | 485 | 0.587 | 0.843 | 0.899 | 0.602–0.885 | 1631.7 | 3.78 [3.65–3.98] | 3.02 [2.85–3.20] | – |
| templates | all | opencv | 12000 | 3.816 | 19.252 | 28.766 | 15.564–22.947 | 171.0 | – | – | – |
| templates | all | syrup-after | 12000 | 2.521 | 11.154 | 22.542 | 10.648–11.928 | 264.8 | 1.51 [1.33–1.70] | 1.73 [0.98–2.54] | -88.1% [-98.6%, -79.2%] |
| templates | all | syrup-before | 12000 | 3.539 | 93.529 | 1056.303 | 91.166–95.223 | 19.8 | 1.08 [0.85–1.25] | 0.21 [0.02–0.42] | – |
| templates | opencv_extra | opencv | 6500 | 3.718 | 24.578 | 33.755 | 18.737–25.638 | 145.1 | – | – | – |
| templates | opencv_extra | syrup-after | 6500 | 2.819 | 12.004 | 22.542 | 11.434–14.727 | 244.7 | 1.32 [1.15–1.69] | 2.05 [1.07–3.32] | -95.6% [-98.9%, -81.9%] |
| templates | opencv_extra | syrup-before | 6500 | 4.039 | 274.658 | 2090.031 | 240.315–277.809 | 12.0 | 0.92 [0.69–1.24] | 0.09 [0.02–0.49] | – |
| templates | opencv_samples | opencv | 5500 | 3.881 | 9.536 | 12.996 | 8.685–10.932 | 216.8 | – | – | – |
| templates | opencv_samples | syrup-after | 5500 | 2.352 | 8.915 | 22.218 | 8.581–9.273 | 293.2 | 1.65 [1.47–1.98] | 1.07 [0.55–1.61] | -81.7% [-84.8%, -75.0%] |
| templates | opencv_samples | syrup-before | 5500 | 3.130 | 48.755 | 89.260 | 46.782–52.006 | 83.6 | 1.24 [1.02–1.51] | 0.20 [0.11–0.34] | – |
| templates | side 16 | opencv | 3600 | 3.775 | 19.261 | 28.706 | 14.551–23.365 | 170.1 | – | – | – |
| templates | side 16 | syrup-after | 3600 | 3.749 | 15.838 | 21.535 | 12.977–17.328 | 191.8 | 1.01 [0.91–1.13] | 1.22 [0.98–1.55] | -98.2% [-99.4%, -85.2%] |
| templates | side 16 | syrup-before | 3600 | 21.258 | 899.251 | 3364.302 | 848.236–981.421 | 7.1 | 0.18 [0.14–0.23] | 0.02 [0.01–0.19] | – |
| templates | side 32 | opencv | 3600 | 3.859 | 19.115 | 30.880 | 16.441–23.827 | 172.3 | – | – | – |
| templates | side 32 | syrup-after | 3600 | 2.459 | 9.564 | 22.544 | 8.888–10.223 | 264.2 | 1.57 [1.32–1.74] | 2.00 [0.85–2.88] | -78.5% [-84.3%, -35.8%] |
| templates | side 32 | syrup-before | 3600 | 3.073 | 44.399 | 980.631 | 41.350–49.424 | 45.9 | 1.26 [1.04–1.39] | 0.43 [0.14–1.62] | – |
| templates | side 48 | opencv | 2400 | 3.870 | 18.810 | 28.985 | 16.532–22.512 | 170.7 | – | – | – |
| templates | side 48 | syrup-after | 2400 | 1.868 | 9.877 | 28.218 | 8.295–12.498 | 314.0 | 2.07 [1.79–2.34] | 1.90 [0.43–5.22] | -29.6% [-76.2%, +6.9%] |
| templates | side 48 | syrup-before | 2400 | 2.002 | 14.036 | 99.732 | 13.632–15.427 | 163.0 | 1.93 [1.72–2.23] | 1.34 [0.11–4.33] | – |
| templates | side 64 | opencv | 2400 | 3.820 | 19.744 | 26.758 | 15.878–24.522 | 170.8 | – | – | – |
| templates | side 64 | syrup-after | 2400 | 1.991 | 4.721 | 5.604 | 4.336–4.889 | 454.6 | 1.92 [1.70–2.27] | 4.18 [2.12–5.26] | -20.5% [-36.0%, -6.5%] |
| templates | side 64 | syrup-before | 2400 | 2.095 | 5.938 | 7.806 | 5.490–6.716 | 393.8 | 1.82 [1.63–2.11] | 3.32 [1.57–4.19] | – |
| yunet | all | opencv | 375 | 19.007 | 25.737 | 28.120 | 23.375–27.462 | 51.9 | – | – | – |
| yunet | all | syrup | 375 | 101.534 | 167.906 | 215.347 | 150.624–185.395 | 9.4 | 0.19 [0.17–0.21] | 0.15 [0.13–0.18] | – |
| yunet | opencv_extra | opencv | 325 | 18.813 | 25.720 | 27.935 | 23.512–27.583 | 53.2 | – | – | – |
| yunet | opencv_extra | syrup | 325 | 98.941 | 169.359 | 221.688 | 151.880–195.193 | 9.5 | 0.19 [0.17–0.22] | 0.15 [0.13–0.19] | – |
| yunet | opencv_samples | opencv | 25 | 23.425 | 26.906 | 28.549 | 19.387–28.335 | 43.9 | – | – | – |
| yunet | opencv_samples | syrup | 25 | 122.228 | 138.169 | 155.286 | 112.283–154.130 | 8.3 | 0.19 | 0.19 | – |
| yunet | syrup | opencv | 25 | 23.095 | 24.242 | 24.652 | 18.683–24.668 | 45.9 | – | – | – |
| yunet | syrup | syrup | 25 | 111.049 | 161.652 | 171.692 | 92.161–172.182 | 8.8 | 0.21 | 0.15 | – |

## Process (median over rounds)

| task/impl | CPU time / wall | peak RSS (MiB) | after imports (MiB) |
|---|---|---|---|
| faces/opencv | 0.98 | 135.4 | 63.5 |
| faces/syrup-after | 0.98 | 23.9 | – |
| faces/syrup-after-gate | 0.98 | 23.9 | – |
| faces/syrup-before | 0.99 | 40.5 | – |
| glyphs/opencv | 0.99 | 64.8 | 64.8 |
| glyphs/syrup-after | 0.97 | 4.4 | – |
| glyphs/syrup-before | 0.96 | 4.5 | – |
| motion/opencv | 0.99 | 206.9 | 64.8 |
| motion/syrup-after | 0.99 | 177.0 | – |
| motion/syrup-before | 0.97 | 177.1 | – |
| templates/opencv | 0.99 | 107.3 | 64.8 |
| templates/syrup-after | 0.99 | 73.3 | – |
| templates/syrup-before | 0.99 | 79.1 | – |
| yunet/opencv | 0.99 | 98.2 | 93.1 |
| yunet/syrup | 0.99 | 57.7 | – |

## Initialisation (median over rounds, ms)

| task/impl | load | first call |
|---|---|---|
| faces/opencv | 6.114 | 568.129 |
| faces/syrup-after | 0.121 | 704.215 |
| faces/syrup-after-gate | 0.123 | 411.235 |
| faces/syrup-before | 0.121 | 970.330 |
| glyphs/opencv | 1.014 | 0.080 |
| glyphs/syrup-after | 1.719 | 0.263 |
| glyphs/syrup-before | 1.619 | 0.248 |
| motion/opencv | – | 0.000 |
| motion/syrup-after | – | 0.000 |
| motion/syrup-before | – | 0.000 |
| templates/opencv | – | 6.335 |
| templates/syrup-after | – | 4.177 |
| templates/syrup-before | – | 26.224 |
| yunet/opencv | 3.284 | 40.905 |
| yunet/syrup | 0.001 | 169.546 |

## Quality

```json
{
 "faces": {
  "same_output_every_round": {
   "opencv": true,
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
