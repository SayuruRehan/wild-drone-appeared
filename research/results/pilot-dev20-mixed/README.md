# Mixed-size training check: trained on N ∈ {2, 3, 4}, evaluated on 3–8 drones

Same settings as `../pilot-dev20` (100k steps, dev20, development worlds 30000–30019), but
episodes cycle through 2, 3 and 4 drones (one replay buffer per size). One seed per model,
so this is a direction check only.

| N | Greedy | VDN (N=3) | VDN (N=2–4) | Structured (N=3) | Structured (N=2–4) |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.329 | 0.316 | 0.312 | 0.315 | 0.308 |
| 5 | 0.395 | 0.386 | 0.386 | 0.383 | 0.360 |
| 6 | 0.449 | 0.401 | 0.400 | 0.405 | 0.382 |
| 8 | 0.513 | 0.420 | 0.420 | 0.435 | 0.409 |

(N=3-trained columns: mean of 2 seeds from `../pilot-dev20`.)

**Reading.** Training on 2–4 drones does not close the gap to Greedy at 6–8 drones. VDN is
unchanged, and structured is slightly worse, though each training size got only a third of
the budget. So the drop at large N is not explained by "never saw more than 3 drones". The
next suspects are budget and the per-drone representation for crowded states. The mixed
structured model cut charger queuing sharply (37 vs 277 drone-steps for mixed VDN at N = 8)
without a freshness gain; a single seed can't separate that from noise. The main study keeps
N = 3 training as the primary condition.
