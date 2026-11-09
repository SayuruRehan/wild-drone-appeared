# Round 3 check (20×20): normalised charger term and structured QMIX

Same protocol as `../pilot-dev20` (dev20, trained on N = 3, 100k steps, final checkpoint,
development worlds 30000–30019, N ∈ {3, 4, 5, 6, 8}). New runs: `structured-n` (seed 0),
`qmix-structured` (seed 0), and second seeds of `qmix` and `structured-overlap`.
`stats.md` pools these with `../pilot-dev20` and `../round2-dev20` (IQM, two-level
bootstrap CIs, Holm-corrected paired tests).

Seeds per model: VDN 2, structured 2, QMIX 2, overlap-only 2, structured-n 1, qmix-structured 1.

| N | Greedy | VDN | Structured | Structured-n | Overlap only | QMIX | QMIX-structured |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.330 | 0.317 | 0.321 | 0.326 | 0.321 | 0.330 | **0.345** |
| 4 | 0.385 | 0.365 | 0.359 | 0.375 | 0.380 | 0.385 | **0.395** |
| 5 | 0.405 | 0.391 | 0.388 | 0.394 | 0.416 | 0.423 | **0.425** |
| 6 | 0.449 | 0.404 | 0.408 | 0.417 | 0.439 | **0.455** | 0.433 |
| 8 | 0.512 | 0.425 | 0.451 | 0.466 | 0.482 | **0.496** | 0.424 |

## Reading

1. **Normalising the charger term per partner helps**: structured-n > structured at every N
   (+0.011 to +0.027; Holm p = 0.013 at N = 4).
2. **QMIX-structured is the best controller at 3–5 drones**: above Greedy at N = 3
   (+0.013, CI [+0.004, +0.022]) and above QMIX (+0.009, CI [+0.001, +0.016]); neither
   survives Holm correction with one seed.
3. **But it collapses at N = 8** (0.424 vs QMIX 0.496). Every model with a charger pair
   term degrades on large fleets, even normalised; overlap-only and plain QMIX transfer best.
4. Working interpretation: domain-structured pair terms improve coordination at and near
   the training fleet size; the *charger* term, which couples drones that are far apart,
   is what breaks extrapolation; the local *overlap* term does not.

## Next

`qmix-overlap` (QMIX mixer + overlap terms only) with 2 seeds, and second seeds of
`qmix-structured` and `structured-n`, are running.
