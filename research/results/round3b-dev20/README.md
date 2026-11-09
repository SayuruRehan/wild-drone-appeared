# Round 3b (20×20): QMIX + overlap, and second seeds — every candidate now has 2 seeds

Same protocol as `../pilot-dev20`. New runs: `qmix-overlap` seeds 0 and 1, second seeds of
`qmix-structured` and `structured-n`. `stats.md` pools all 20×20 learner results so far
(2 training seeds × 20 development worlds per learner).

IQM of weighted freshness:

| N | Greedy | VDN | Overlap only | Structured-n | QMIX | QMIX + overlap | QMIX-structured |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.330 | 0.317 | 0.321 | 0.317 | 0.330 | 0.329 | **0.343** |
| 4 | 0.385 | 0.365 | 0.380 | 0.373 | 0.385 | 0.380 | **0.389** |
| 5 | 0.405 | 0.391 | 0.416 | 0.397 | 0.423 | 0.408 | **0.425** |
| 6 | 0.449 | 0.404 | 0.439 | 0.423 | **0.455** | 0.426 | 0.443 |
| 8 | **0.512** | 0.425 | 0.482 | 0.474 | 0.496 | 0.473 | 0.447 |

## Reading (2 seeds each, 100k steps; development worlds only)

1. **QMIX + overlap does not beat plain QMIX**: tied at N = 3, worse at N ≥ 5
   (−0.028 at N = 6 and −0.020 at N = 8, both Holm p ≤ 0.003).
2. **QMIX-structured** is the best learner at N = 3–4 (+0.013 vs Greedy at N = 3,
   CI [+0.004, +0.023]; +0.009 vs QMIX, CI [+0.001, +0.015]) but falls behind QMIX at N ≥ 6.
   None of its gains survive Holm correction with 2 seeds.
3. **Plain QMIX transfers best**: trained on 3 drones it stays within 0.016 of Greedy at
   N = 8 and is level with or above Greedy at N = 3–6.
4. **Relative to additive factorisation the structured terms clearly help**: overlap-only
   and structured-n beat VDN by about 0.05 at N = 8 (Holm p = 0.0015 for qmix-overlap − vdn;
   similar CIs for the additive variants), and they inspect far fewer duplicate cells.
5. **No learner beats Greedy at N = 8 on 20×20.** Learner budgets are only 100k steps.

## What this means for the claims

- C1 (fleet-size transfer) is supported, most strongly for QMIX-style mixing with
  per-drone hypernetwork weights and exact feasible joint selection.
- C2 (domain-structured interactions) holds against additive factorisation (VDN) but not
  against QMIX: adding pairwise terms to the mixer improves coordination near the training
  size and hurts extrapolation. The defensible framing is a characterised trade-off, not
  "structured beats QMIX".
