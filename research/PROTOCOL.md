# Research protocol — fleet-size generalisation with domain-structured interactions

Status: **draft v0.2 (development)**. The protocol is frozen (tagged `protocol-v1`) only
after gates G1–G3 below. Until then, everything here may change, and every change is
recorded in the changelog at the bottom. Final-test worlds are not flown before the tag.

The frozen assignment study (`docs/experiments/rl-main-2026-10-11`, the 90,000-step
runs and the 20 final worlds 22000–22019) stays unchanged as historical evidence. Nothing in
`research/` modifies `src/wild_drone` or `configs/`.

## 1. Problem

Three or more battery-limited drones share **one** charging pad and must keep a stochastic
wildfire map freshly inspected (priority-weighted freshness), on a fully observable grid,
with a central controller choosing one joint move per step. The environment is the frozen
assignment simulator, generalised to N drones (`research/fleetrl/env.py`; identical to the
frozen environment at N = 3, verified by `research/tests/test_equivalence.py`).

## 2. Claims under test

**C1 — Fleet-size generalisation.** A parameter-shared, drone-centric value function with
exact feasibility-constrained joint action selection, trained on a small fleet (N = 3),
controls larger fleets (N = 4…8) without retraining, while a joint-action network cannot
be applied at all and its output layer grows as 5^N.

**C2 — Domain-structured interactions.** Pairwise interaction terms built from quantities
the simulator defines exactly, i.e.

    P_ij(a_i, a_j) = alpha_ij(s) * Overlap_ij(a_i, a_j) + beta_ij(s) * Charger_ij(a_i, a_j)

improve coordination over VDN (no interactions) and over a learned, unstructured
coordination graph (DCG), especially as the fleet grows. `Overlap` is the urgency covered by
both drones' footprints after their moves, which is exactly the pairwise inclusion–exclusion
term of the next reward (tested in `test_learning.py`); `Charger` marks both drones ending on
or approaching the single pad. alpha and beta are state-conditioned (from the pair of drone
embeddings), so the network learns *how much* each known interaction matters in context.

### Hypotheses (development wording; thresholds frozen at `protocol-v1`)

- **H0 (repaired baseline).** A dueling Double-DQN joint-action learner with a large replay
  buffer and n-step returns beats masked random at N = 3. *If false, no comparison against
  "joint DQN" is reported, because the baseline would be uninformative.*
- **H1 (factorisation).** At N = 3 and equal environment steps, VDN reaches at least the
  joint learner's development freshness (area under the development curve and final value).
- **H2 (generalisation).** Factored models trained at N = 3 keep their freshness ratio to
  Greedy at N ∈ {4, 5, 6, 8} within a margin to be frozen at `protocol-v1`, and stay far
  above masked random, with zero drone losses.
- **H3 (structured interactions).** Structured > VDN at N ≥ 5 on freshness, with lower
  duplicate inspection and charger waiting; structured ≥ DCG.
- **H4 (ablation).** Removing either domain feature (overlap-only, charger-only) or adding a
  free residual table (`structured+`) shows which part carries the effect.

## 3. Controllers

| Controller | Role | Code |
|---|---|---|
| Masked random | safety-only floor | `baselines.MaskedRandomPolicy` |
| Greedy (one-step) | frozen strong reference; exact for N ≤ 5 | `baselines.GreedyPolicy` |
| Assign (target assignment) | non-myopic heuristic; headroom probe | `baselines.AssignPolicy` |
| Joint dueling DDQN | repaired joint-action RL, N = 3 only | `models.FleetQ('joint')` |
| VDN | factorisation without interactions | `models.FleetQ('vdn')` |
| DCG | learned pairwise tables (Böhmer et al., 2020) | `models.FleetQ('dcg')` |
| **Structured** | proposed | `models.FleetQ('structured')` |
| Structured+ | proposed + learned residual (ablation) | `models.FleetQ('structured+')` |
| QMIX | established monotonic mixer | *to add before protocol-v1* |

All learners share the same drone encoder (17×17 egocentric crop, 8×8 coarse global map,
per-drone scalars, per-command features), optimiser, Double-DQN target, n-step returns,
reward scale, replay and exploration schedule; only the value combination differs. All
controllers act through the same factored feasibility mask (battery return, pad charging,
no collisions or swaps), and joint actions are chosen by **exact** dense maximisation over
all 5^N joint actions (N ≤ 8), so no result depends on an approximate action search.

## 4. Data splits (world seeds)

| Split | Seeds | Use |
|---|---|---|
| Training | 100000 + 100000·seed + episode | training episodes only |
| Development | 30000–30049 | all tuning, checkpoint choice, gates |
| Final test | 40000–40099 | flown once, after `protocol-v1`, by the final runs only |

All are disjoint from the assignment seeds (0–299, 5000–5004, 22000–22019, 23000–23019).
`evaluate_runs.py` refuses final-test worlds unless `--final` is passed.

## 5. Gates and stopping rules

| Gate | Question | Rule | Status |
|---|---|---|---|
| G1 | Is there headroom above Greedy? | some controller beats Greedy on ≥ 70% of 50 dev worlds at N = 3 (40×40) | **passed**: Assign +0.033, 48/50 (`results/headroom`) |
| G2 | Does the repaired joint baseline learn? (H0) | joint DDQN > masked random at N = 3, dev20 | pilot running |
| G3 | Do factored models learn and transfer on 20×20? | best factored model ≥ 50% of the masked-random→Greedy gap at N = 3 and stays above masked random at N = 8 | **passed**: 92% of the gap at N = 3; 0.435 vs 0.264 at N = 8 (`results/pilot-dev20`) |
| G4 | 40×40 pilot | same as G3 on main40 with one seed per model | after G3 |

If G3 fails after one diagnosis round (budget, representation), the 40×40 study is not
launched and the representation is revisited first. No reward shaping or demonstrations are
added in the main comparison.

## 6. Final study design (fixed at `protocol-v1`)

- Environments: main40 (primary), dev20 (secondary).
- Training: N = 3 for C1/C2 (optional: mixed N ∈ {2, 3, 4} as a secondary condition).
- Budget: one common environment-step budget for every learner, chosen from measured
  throughput and the G4 learning curves (not from results on test worlds).
- Seeds: 10 training seeds per learner.
- Evaluation: 100 final-test worlds × N ∈ {3, 4, 5, 6, 8}; checkpoint = last (primary) and
  best-on-development (secondary), declared before running.
- Primary metric: weighted freshness (full-horizon accounting).
  Secondary: overdue fraction, return, drone losses, energy, duplicate cells per step,
  charger waiting steps, decision latency.
- Statistics: interquartile mean with stratified-bootstrap 95% CIs across seeds
  (Agarwal et al., 2021), paired per-world differences against Greedy and between learners,
  Holm correction over the primary comparisons (H2, H3). Effect sizes are reported with CIs,
  not only p-values.

## 7. Artifacts and reproducibility

Every run writes `config.json`, `log.jsonl` (development curve), `final.pt`, `best_dev.pt`
and `done.json`. Checkpoints are not committed (`*.pt` is ignored); final checkpoints are
published as a GitHub release with SHA-256 sums. Each results folder contains the per-episode
CSV, the summary table and the exact command used. `pytest research/tests` must pass on the
commit that produced a result.

## 8. Related work and novelty (to be completed with a systematic search)

| Work | Relation |
|---|---|
| VDN (Sunehag et al., 2018), QMIX (Rashid et al., 2018), QPLEX (Wang et al., 2021) | value factorisation; no pairwise terms (VDN, QMIX) or no domain structure |
| DCG (Böhmer et al., 2020); DICG (Li et al., 2021); CASEC (Wang et al., 2022); SOPCG (Yang et al., 2022) | learned coordination graphs; the pairwise *form* is theirs, so DCG is a baseline here |
| UPDeT (Hu et al., 2021); REFIL (Iqbal et al., 2021) | transfer across agent counts via attention/entities; candidates for related work, possibly a baseline |
| "Energy-Aware Persistent Multi-UAV Coverage via RL Guided by User Priority and Outage" (*Drones* 10:688, POGS-QMIX) | same application family (persistent priority coverage, energy); centralised greedy/Hungarian assignment coordinator + QMIX; no pairwise value terms, no fleet-size transfer. Must be a cited comparison; its coordinator is close in spirit to our `Assign` heuristic |
| Masked deep-RL multi-UAV task scheduling (*Drones* 9(6):418) and other action-masking UAV papers (arXiv 2407.19657, 2509.23905) | feasibility via action masks; masking alone is therefore **not** a contribution here |
| Zero-shot scalable resilience in UAV swarms with physics-informed graph interactions (arXiv 2604.15762) | zero-shot transfer across swarm sizes with domain-informed interaction graphs (imitation learning, resilience task); closest framing to C1+C2, must be cited and distinguished |
| Graph-based MARL coverage with overlap-rate metric (2026, found in search; to read) | reports a coverage-overlap rate as a metric; we use overlap inside the value function |

Working novelty statement: (i) pairwise interaction terms derived from the reward's
inclusion–exclusion structure and from a shared resource, with state-conditioned gains,
(ii) exact feasibility-constrained joint selection under coupled battery/collision/charger
constraints, and (iii) evidence of zero-shot transfer across fleet sizes in persistent
inspection with one shared charger. This statement must be re-checked against the systematic
search before `protocol-v1`.

## Changelog

- v0.1: initial draft; G1 headroom result added.
- v0.2: learners, trainer and 20×20 pilot launched (joint, VDN, DCG, structured; 2 seeds,
  100k steps, N = 3).
- v0.3: G3 passed (factored learners). Joint learners must be evaluated only at their training
  fleet size (bug fixed before their runs). Next: 300k steps, ≥ 3 seeds, mixed-size training
  condition, 40×40 pilot (G4).
