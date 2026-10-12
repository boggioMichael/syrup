Task `glyphs`: run A `repro/test-a`, run B `repro/test-b`.

| implementation | same outputs | p50 A → B (ms) | p50 change | p95 A → B (ms) | p95 interval in A | ± of p95 | G1 |
|---|---|---|---|---|---|---|---|
| opencv | yes | 0.074 → 0.074 | +0.4% | 0.151 → 0.152 | 0.146–0.156 | 3.4% | passes |
| syrup-after | yes | 0.109 → 0.115 | +5.2% | 0.477 → 0.506 | 0.408–0.532 | 13.0% | fails |
| tinynet | yes | 0.078 → 0.077 | -1.4% | 0.151 → 0.158 | 0.146–0.155 | 2.9% | passes |
