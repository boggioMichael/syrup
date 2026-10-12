# Amendments

Every change to [PREREGISTRATION.md](PREREGISTRATION.md) or [DATA.md](DATA.md)
after gate G0 is recorded here: the date, what changed, why, and what data had
been seen when it changed.

None yet.

## Proposed for G0, not applied

Both come from the P1 shakedown ([SHAKEDOWN.md](SHAKEDOWN.md)), which used only
synthetic dev data. The owner accepts or rejects them at G0; until then
PREREGISTRATION.md stays as committed.

1. **A line reader in the tiny-network family for task A** (12 October 2026).
   Add to the family of implementation 3: a small CNN over the whole text line
   with CTC decoding, of at most 100,000 parameters. *Why:* in the shakedown,
   every wrong read of both learned readers came from the shared per-glyph
   segmentation (26 of 26, all on the small size). A learned classifier added
   nothing over one nearest neighbour, so a baseline that cuts glyphs first is
   not the best learned reader.
2. **G1 on a dedicated machine** (12 October 2026). Measure G1 on a machine
   that runs nothing else, with CPU frequency scaling fixed, and not on a shared
   cloud container. The rule itself (under 5% between runs) stays. *Why:* on a
   two-vCPU shared container, the unchanged OpenCV reader's p50 moved by 27%
   between two runs.
