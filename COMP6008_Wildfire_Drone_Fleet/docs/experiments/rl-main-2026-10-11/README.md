# Frozen 40×40 RL main study

Source: `855e5510ff46b16082ab7d551e7c228acab6f572`. Exactly nine RL fits, three per algorithm,
300 episodes each, independently initialized without supervised warm starts.
No post-result tuning or best checkpoint selection. Original negative evidence is preserved.

| Policy | Freshness ↑ | Overdue ↓ | Return ↑ | Losses ↓ |
|---|---:|---:|---:|---:|
| dqn | 0.084383 | 0.868689 | -24.785684 | 0.000 |
| reinforce | 0.069014 | 0.894019 | -25.810541 | 0.000 |
| a2c | 0.084347 | 0.869489 | -24.827722 | 0.000 |
| greedy | 0.220769 | 0.655626 | -16.100083 | 0.000 |
| sweep | 0.064858 | 0.901210 | -26.071031 | 0.000 |
| random | 0.059562 | 0.920978 | -42.871080 | 3.000 |
| masked_random | 0.083794 | 0.886005 | -25.461477 | 0.000 |

All-seed learned-contribution gates: `{'dqn': False, 'reinforce': False, 'a2c': False}`.
All-seed Greedy capability gates: `{'dqn': False, 'reinforce': False, 'a2c': False}`.
Comparison completion and performance gates are separate; a negative result does
not trigger more fitting or establish an RL advantage. See `completion.json` for
every seed and each criterion. `summary.csv` preserves fitted-seed sample SD.

The complete environment and all settings were frozen before fitting in
`configs/rl-main-frozen.json`; the prospective protocol is `docs/scale-rl-protocol.md`.
Training worlds 0–299, new final worlds 22000–22019, no validation selection.
All nine fits finished before final evaluation. Each final model runs 20 missions;
Greedy, sweep, unmasked random and masked random run the same worlds (260 total).
Full-horizon accounting includes early fleet failure. Masked methods receive a
rule-based battery/collision feasibility mask. Greedy/sweep retain different
native proposal preferences; safe flight is not entirely a learned result.
Actual steps, updates and training time appear in `audit.json` and each fit's
metadata/history. A fixed episode budget can yield different real interactions
if fleets fail early. This is a fixed-budget comparison, not proof of convergence.

Means first average worlds within each fitted seed; SD is across three fitted
seed means. Baseline SD is across 20 worlds. Repeated seed/world rows are not
independent trained replicates; there are no significance claims. `paired.csv`
contains all 720 matched fitted-world/baseline comparisons. Neural outputs are
RL Q/logit scores, not supervised inspection gains. No 60×60 results are claimed.

Independent audit verified initial tensors, checkpoint bytes, source/settings,
all training episode/update counts, every mission action/hash/mask/reward and
all summaries/gates. All 75911 mission steps replayed, maximum neural score
drift 0, with fixed atol 1e-5 and rtol 1e-6 and exact selected actions.
Numeric evidence, complete compressed scores/traces, learning curves and checksums
are archived here. Raw initial/final weights remain in nine per-fit Actions
artifacts and `frozen-40x40-rl-results-855e5510ff46b16082ab7d551e7c228acab6f572` for 90 days.
Download the results artifact, extract its run folder, and independently replay:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python scripts/audit_scale_rl.py --run runs/rl-main-2026-10-11 --out runs/replayed-rl-main
```
