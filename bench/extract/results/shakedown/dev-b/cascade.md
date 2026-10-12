Task `glyphs`, split `dev`, 120 items. 95% bootstrap intervals over items.

| explicit → fallback | coverage | system correct | system wrong | fallback alone: correct | fallback alone: wrong | correct gain, points (paired) | system ms/item | fallback alone ms/item | fallback ÷ system cost |
|---|---|---|---|---|---|---|---|---|---|
| syrup-after → opencv | 48.3% [39.2–57.5] | 97.5% [94.2–100.0] | 2.5% [0.0–5.8] | 95.0% [90.8–98.3] | 5.0% [1.7–9.2] | +2.5 [+0.0, +5.8] | 0.196 [0.170–0.224] | 0.081 | 0.41 [0.36–0.48] |
| syrup-after → tinynet | 48.3% [39.2–57.5] | 97.5% [94.2–100.0] | 2.5% [0.0–5.8] | 95.0% [90.8–98.3] | 5.0% [1.7–9.2] | +2.5 [+0.0, +5.8] | 0.201 [0.175–0.230] | 0.093 | 0.46 [0.40–0.53] |
