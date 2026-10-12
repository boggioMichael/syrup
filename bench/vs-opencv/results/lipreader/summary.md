# Lip-reading pipeline (F1-pipeline)

3 runs per detector, rotated order; ms per frame of the whole pipeline (detection, tracking, mouth crops, LipNet), median [range]; × = over OpenCV's median.

| fixture | opencv | syrup-after | syrup-before | quality columns as with syrup-before |
|---|---|---|---|---|
| five_faces | 83.2 [82.5–83.9] | 93.8 [92.1–94.0] (1.13×) | 174.9 [165.5–176.1] (2.10×) | syrup-after: yes |
| low_res | 18.3 [17.8–21.0] | 23.9 [22.7–25.8] (1.31×) | 40.3 [39.3–41.5] (2.21×) | syrup-after: yes |
| muted | 18.1 [17.8–19.5] | 34.1 [30.6–35.1] (1.89×) | 65.5 [60.5–67.1] (3.62×) | syrup-after: yes |
| one_speaker | 16.7 [16.2–17.1] | 33.8 [31.0–35.8] (2.02×) | 64.6 [64.2–70.1] (3.87×) | syrup-after: yes |
| rapid_cuts | 18.1 [18.0–18.3] | 32.2 [30.9–32.2] (1.78×) | 61.6 [60.6–61.9] (3.41×) | syrup-after: yes |
| two_alternating | 26.5 [25.8–26.9] | 39.9 [36.8–40.1] (1.50×) | 76.7 [75.0–77.0] (2.89×) | syrup-after: yes |
| two_simultaneous | 31.3 [30.6–33.2] | 44.5 [40.9–44.8] (1.42×) | 83.9 [81.7–85.9] (2.68×) | syrup-after: yes |

Quality columns (first run; every run gave the same: opencv yes, syrup-after yes, syrup-before yes)

| fixture | detector | WER | CER | attribution | track consistency | speaking P | speaking R |
|---|---|---|---|---|---|---|---|
| five_faces | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.6613 | 0.9152 |
| five_faces | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.6613 | 0.9152 |
| five_faces | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.6613 | 0.9152 |
| low_res | opencv | 0.5 | 0.4444 | 1.0 | 1.0 | 0.7663 | 0.993 |
| low_res | syrup-after | 0.5 | 0.4722 | 1.0 | 1.0 | 0.7172 | 1.0 |
| low_res | syrup-before | 0.5 | 0.4722 | 1.0 | 1.0 | 0.7172 | 1.0 |
| muted | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| muted | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| muted | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| one_speaker | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| one_speaker | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| one_speaker | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| rapid_cuts | opencv | None | None | None | 1.0 | 1.0 | 0.575 |
| rapid_cuts | syrup-after | None | None | None | 1.0 | 1.0 | 0.475 |
| rapid_cuts | syrup-before | None | None | None | 1.0 | 1.0 | 0.475 |
| two_alternating | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.7209 | 0.8732 |
| two_alternating | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.7412 | 0.8873 |
| two_alternating | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.7412 | 0.8873 |
| two_simultaneous | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.6985 | 0.669 |
| two_simultaneous | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.6985 | 0.669 |
| two_simultaneous | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.6985 | 0.669 |
