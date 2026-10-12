Task `glyphs`: run A `shakedown-dev`, run B `repeat-dev`.

| implementation | same outputs | p50 A → B (ms) | p50 change | p95 A → B (ms) | p95 interval in A | ± of p95 | G1 |
|---|---|---|---|---|---|---|---|
| opencv | yes | 0.100 → 0.073 | -26.7% | 0.193 → 0.141 | 0.182–0.200 | 4.8% | fails |
| syrup-after | yes | 0.122 → 0.134 | +9.5% | 0.464 → 0.499 | 0.331–0.637 | 33.0% | fails |
| tinynet | yes | 0.076 → 0.101 | +33.0% | 0.143 → 0.196 | 0.136–0.153 | 5.8% | fails |
