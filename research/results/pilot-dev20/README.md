# Pilot (gate G3): 20×20, trained on 3 drones, evaluated on 3–8 drones

**Setup.** Environment `dev20` (frozen development settings). Models VDN, DCG and
structured; 2 training seeds each; 100,000 environment steps; trained on N = 3 only;
Double DQN, 3-step returns, 100k replay, batch 64, update every 4 steps, reward ×10,
ε 1→0.05 over 30k steps (`runs/*/config.json`). Final checkpoint (no checkpoint selection).
Evaluation: development worlds 30000–30019, N ∈ {3, 4, 5, 6, 8}, exact feasible argmax.
Baselines are the same worlds from `../headroom/episodes.csv`.
Checkpoints are not committed; their SHA-256 sums are in `SHA256SUMS`.

The joint dueling DDQN runs (gate G2) are reported separately once they finish.

```bash
PYTHONPATH=research:src python research/scripts/pilot.py --kinds vdn dcg structured \
  --seeds 0 1 --steps 100000 --env dev20 --parallel 4 --root research/runs/pilot-dev20 \
  -- --eval-every 25000 --eval-worlds 10 --eval-drones 3 5
PYTHONPATH=research:src python research/scripts/evaluate_runs.py \
  --root research/runs/pilot-dev20 --drones 3 4 5 6 8 --worlds 20 --out research/results/pilot-dev20
```

## Results (`table.md`, `episodes.csv`)

Weighted freshness. Learners: mean over worlds per seed, then mean (SD) over 2 seeds.
"Wins" = (seed, world) pairs beating Greedy on the same world. Dup = cells inspected by
more than one drone per step. Wait = drone-steps queuing at the charger per mission.

| N | Masked random | Greedy | Assign | VDN | DCG | Structured | Dup VDN / Struct | Wait Greedy / Struct |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.147 | 0.329 | 0.350 | 0.316 (0.007) | 0.312 (0.024) | 0.315 (0.017) | 1.70 / 1.12 | 31 / 7 |
| 4 | 0.176 | 0.362 | 0.399 | 0.358 (0.010) | 0.350 (0.025) | 0.356 (0.001) | 3.83 / 2.50 | 51 / 32 |
| 5 | 0.193 | 0.395 | 0.432 | 0.386 (0.012) | 0.354 (0.030) | 0.383 (0.002) | 6.34 / 3.84 | 101 / 64 |
| 6 | 0.213 | 0.449 | 0.462 | 0.401 (0.008) | 0.351 (0.030) | 0.405 (0.003) | 8.38 / 4.72 | 205 / 86 |
| 8 | 0.264 | 0.513 | 0.510 | 0.420 (0.003) | 0.345 (0.049) | 0.435 (0.034) | 10.03 / 5.65 | 351 / 129 |

No drone was lost and no collision was attempted in any evaluated mission.

## Reading

1. **G3 passes.** At N = 3 the factored learners close 92% of the masked-random → Greedy
   gap (0.315 vs 0.147 → 0.329) after 100k steps. The assignment-study DQN stayed at
   masked-random level after 90k steps. All stay far above masked random at N = 8.
2. **No learner beats Greedy yet.** At N = 3 they win 12–16 of 40 paired worlds. Learning
   curves (`runs/*/log.jsonl`) were still rising at 100k steps, so the budget is the first
   thing to increase. Assign remains the strongest controller at 3–6 drones.
3. **Transfer across fleet size (C1).** Trained on 3 drones, VDN and structured stay
   within about 0.01 of Greedy at N = 4–5 without retraining. They fall behind Greedy at
   N = 6–8 (−0.04 to −0.09), but keep about 60% of the masked-random → Greedy gap at N = 8.
4. **Learned pairwise tables transfer worst.** DCG matches the others at N = 3 but drops
   to 0.345 at N = 8, below its own N = 3–4 level (0/40 wins against Greedy at N ≥ 6).
   Free 5×5 interaction tables learned on 3 drones do not carry over to crowded fleets.
5. **Structured interactions (C2): coordination improves, freshness barely does yet.**
   Structured inspects about 40–45% fewer duplicate cells than VDN at every N and keeps
   charger queues well below Greedy's. On freshness it is indistinguishable from VDN at
   N ≤ 6 (paired differences −0.003 to +0.005) and ahead at N = 8 (+0.015, 24/40 paired
   wins, one seed driving most of it). With 2 seeds this is a hypothesis, not a result.

## Decisions for the next round

- Increase the budget (300k steps) and seeds (≥ 3) before judging H1/H3.
- Add a mixed-size training condition (N ∈ {2, 3, 4}) to test whether the drop at
  N ≥ 6 comes from never seeing crowded states.
- Run the same matrix on 40×40 (gate G4) on the Colab GPU.
- Keep DCG as the learned-interaction baseline. Its poor transfer supports C2 if it holds
  with more seeds.
