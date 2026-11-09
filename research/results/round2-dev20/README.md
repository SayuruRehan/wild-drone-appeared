# Round 2 check (20×20): QMIX, paper-faithful DCG and single-feature ablations

Same protocol as `../pilot-dev20` (dev20, trained on N = 3, 100k steps, final checkpoint,
development worlds 30000–30019, N ∈ {3, 4, 5, 6, 8}). **One seed per model**; the VDN and
structured rows in `stats.md` come from the 2-seed pilot. `stats.md` gives IQMs with
two-level bootstrap CIs and Holm-corrected paired tests (`scripts/stats.py`).

| N | Greedy | VDN | Structured | Overlap only | Charger only | DCG (v2) | QMIX |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.330 | 0.317 | 0.321 | 0.310 | 0.310 | 0.324 | 0.327 |
| 4 | 0.385 | 0.365 | 0.359 | 0.366 | 0.345 | 0.365 | 0.381 |
| 5 | 0.405 | 0.391 | 0.388 | 0.406 | 0.364 | 0.381 | 0.414 |
| 6 | 0.449 | 0.404 | 0.408 | 0.428 | 0.362 | 0.386 | 0.450 |
| 8 | 0.512 | 0.425 | 0.451 | 0.481 | 0.386 | 0.394 | 0.503 |

(IQM of weighted freshness.)

## Reading

1. **All new models learn**, and the paper-faithful DCG no longer collapses (the pilot DCG
   did on 40×40). At N = 3 every learner is statistically tied with Greedy.
2. **QMIX transfers best**: trained on 3 drones it stays level with Greedy up to 8 drones
   (−0.011 at N = 8, CI includes 0) and beats structured at N ≥ 4 (Holm p ≤ 0.009).
3. **The overlap term helps transfer, the charger term hurts it.** Overlap-only is the best
   additive model at N ≥ 5 (0.481 at N = 8 vs 0.451 structured, 0.425 VDN); charger-only is
   the worst model at every N ≥ 4. Likely cause: the charger feature couples *every* pair of
   drones heading home, so the number of active charger terms grows as N(N−1)/2, while the
   overlap terms stay local (only nearby pairs are non-zero). A gain learned on 3 drones
   (3 pairs) is applied to up to 28 pairs at N = 8.
4. DCG-v2 also degrades at N ≥ 6. Its 1/|E| averaging changes the balance between unary
   and pairwise terms as the fleet grows, so it too is mis-scaled for transfer.

## Consequences for the method (to decide before protocol-v1)

- The shared-charger interaction is a single shared resource, not a set of independent
  pairs. Candidate fix: normalise the charger term per partner (divide by N − 1), so the
  total charger penalty a drone sees is scale-free.
- QMIX is now the strongest learned baseline and must be beaten or matched. Candidate
  method: domain-structured terms (overlap, normalised charger) added to a QMIX-style
  monotonic mixer, instead of to a purely additive VDN.
