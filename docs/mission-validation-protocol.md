# Prospective fixed-scorer mission validation at 20×20

Follow PR #28's successful inspection-ranking comparison with complete natural
missions. Keep its six final supervised checkpoints fixed: small/expanded
training corpora, seeds 0/1/2, centered input transform, 2,000 completed updates.
No new fitting, Q interpretation, RL update, reward/mask change, policy tuning
or 40×40 run is included. Freeze this protocol and runner on `sr` before launch.

Use exactly the parent environment: 20×20, horizon 160, battery 80, recharge 5,
radius 1, deadline 25, max fuel 100 and remaining fields from the parent launch.
Evaluate every policy on new development worlds 6600–6619. These are disjoint
from training 0–199, reused 6200–6215 and previously examined 6400–6415.
Confirmation 5800–5819 and final-test 10000+ remain unopened. No excluded worlds,
checkpoint selection, early stopping of the comparison or recipe extensions
after reading results. Same scenario seed gives the same exogenous fire draws.

Deploy frozen network scores with deterministic first-index argmax under the
unchanged `action_mask`, including its emergency-return fallback. Scores are
one-step inspection predictions, not Q values. Deployment uses only current
observation plus the existing feasibility rule, without action labels or
simulator lookahead. Two training-action-mean controls use this same selection
rule. Report native Greedy and Sweep controllers and world-seeded masked-random
as contextual baselines. Native Greedy/Sweep use their existing candidate rule;
unlike `action_mask`, it lacks the emergency fallback. Log any action outside
`action_mask`. Primary small/expanded comparisons share exactly the same rule.

There are 11 policy/seed conditions × 20 worlds = 220 episodes. Run all missions
through termination or horizon 160. Failed fleets contribute zero freshness
and full overdue for the remaining horizon, matching existing `run_episode`.
Report original undiscounted return, horizon-averaged weighted freshness and
overdue fraction, energy used, losses, intact fleets, all-fleet failures, safe
returns, charge completions, collisions, steps and decision time. Energy use and
charging are descriptive; lower values alone do not establish better service.

Within each fitted seed, average over all 20 worlds, then report mean/sample SD
over three seed means. Baselines/control SDs use worlds and are labeled as such.
Archive paired expanded-minus-small differences per matched seed/world. These
are descriptive development results; make no statistical significance claim.

Keep two decisions separate, requiring every fitted seed to pass:

1. **Useful mission data effect:** expanded weighted freshness improves by ≥.01
   absolute over matched small mean, overdue fraction does not increase, and
   neither mean drone losses nor fleet-failure count increases.
2. **Greedy-relative mission capability:** expanded weighted freshness ≥90% of
   native Greedy mean, overdue fraction ≤Greedy+.05, mean losses ≤Greedy+.1,
   and fleet-failure count ≤Greedy.

A one-step ranking improvement without these mission checks supports further
trajectory diagnosis, not promotion to 40×40. Passing both supports planning a
separate RL/40×40 validation, not claiming trained RL or 40×40 success.

Verify source, model identities and seed/update/format/configuration metadata
before any rollout. Archive each action and pre/post observation hash, per-step
metrics/reward components, all 125 scores and masks for every network/control
decision, all episode/fit/aggregate rows and declared conditions. Numeric scores
remain complete lossless float32 arrays. Audit every selected argmax and replay
all traces against the real simulator; independently recompute full-horizon
metrics, paired differences, aggregates and decisions without runner helpers.
Timing measurements are descriptive and excluded from exact replay equality.
Verify original parent checkpoints are unchanged after evaluation. Tests must
cover masking/ties/configuration, invalid scores, trace/legacy-metric parity,
per-seed criteria, checkpoint identity/metadata and an end-to-end miniature run.
Review frozen source and final evidence, pass repository lint/tests/demo and
both final CI jobs, merge the authorized sr→master PR and synchronize `sr`.
