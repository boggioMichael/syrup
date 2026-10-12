Task `glyphs`: run A `shakedown-test`, run B `repeat-test`.

| implementation | same outputs | p50 A → B (ms) | p50 change | p95 A → B (ms) | p95 interval in A | ± of p95 | G1 |
|---|---|---|---|---|---|---|---|
| opencv | yes | 0.082 → 0.075 | -7.9% | 0.174 → 0.159 | 0.166–0.181 | 4.5% | fails |
| syrup-after | yes | 0.125 → 0.119 | -4.9% | 0.547 → 0.537 | 0.500–0.592 | 8.4% | passes |
| tinynet | yes | 0.084 → 0.079 | -6.5% | 0.175 → 0.159 | 0.167–0.183 | 4.4% | fails |
