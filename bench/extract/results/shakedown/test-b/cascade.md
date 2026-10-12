Task `glyphs`, split `test`, 400 items. 95% bootstrap intervals over items.

| explicit → fallback | coverage | system correct | system wrong | fallback alone: correct | fallback alone: wrong | correct gain, points (paired) | system ms/item | fallback alone ms/item | fallback ÷ system cost |
|---|---|---|---|---|---|---|---|---|---|
| syrup-after → opencv | 48.2% [43.5–53.0] | 95.5% [93.5–97.2] | 4.5% [2.5–6.5] | 93.5% [91.0–95.8] | 6.5% [4.2–9.0] | +2.0 [+0.8, +3.5] | 0.194 [0.179–0.209] | 0.083 | 0.43 [0.39–0.46] |
| syrup-after → tinynet | 48.2% [43.2–53.0] | 95.5% [93.5–97.5] | 4.5% [2.5–6.8] | 93.5% [91.0–95.8] | 6.5% [4.2–9.0] | +2.0 [+0.8, +3.5] | 0.195 [0.180–0.210] | 0.082 | 0.42 [0.39–0.46] |
