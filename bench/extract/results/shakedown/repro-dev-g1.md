Task `glyphs`: run A `repro/dev-a`, run B `repro/dev-b`.

| implementation | same outputs | p50 A → B (ms) | p50 change | p95 A → B (ms) | p95 interval in A | ± of p95 | G1 |
|---|---|---|---|---|---|---|---|
| opencv | yes | 0.081 → 0.073 | -10.0% | 0.183 → 0.144 | 0.161–0.195 | 9.2% | fails |
| syrup-after | yes | 0.126 → 0.124 | -1.5% | 0.499 → 0.500 | 0.337–0.658 | 32.2% | fails |
| tinynet | yes | 0.077 → 0.077 | +0.5% | 0.150 → 0.151 | 0.141–0.161 | 6.6% | passes |
