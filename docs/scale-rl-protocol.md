# Frozen main-grid RL comparison — 11 October 2026

This completes the approved 40×40 assignment comparison. Preserve all earlier
negative supervised and RL results. No more loss tuning, warm starts, reward
shaping, architecture changes, best-seed selection, or outcome-driven extra fits.
A fair negative comparison is a valid outcome; passing a performance gate is not
a requirement for completing the experiment or reporting it.

Use the unchanged `configs/main.json` environment (40×40, horizon 300, three drones,
battery 160, recharge 8, radius 2, deadline 30). Full frozen environment and every
TrainConfig field are in `configs/rl-main-frozen.json`. Use corrected standard
DQN, REINFORCE with learned Monte Carlo baseline, and synchronous n-step A2C.
All use the existing full-grid spatial encoding and baseline-candidate feasibility
mask. This rule supplies battery/collision heuristics; masked safety must not be
credited entirely to learned coordination. Greedy and sweep retain their native
candidate/movement rules, a disclosed contextual difference.

Exactly nine main fits: each algorithm with seeds 0, 1, 2, independently initialized
from scratch. Fixed 300 episodes per fit on training worlds 0–299: up to 90,000 real
environment interactions per fit. Report actual steps, updates and elapsed time;
early fleet termination can reduce interactions. Adam, batch 32, replay 2,000,
gamma .99, gradient clipping .5; no auxiliary reward or supervision. DQN learning
rate .0001 uses the previously explored lower-rate stability setting; standard TD
targets remain explicit. REINFORCE rate .001 / entropy 0 and A2C rate .0001 /
entropy .05 / n_steps 32 reuse historical frozen development choices. No 40×40
hyperparameter search is performed. Temperature or loss adjustments are excluded.

No periodic validation or best checkpoint selection: retain initial and final
weights, every episode row and complete metadata. All nine fits must finish and
pass identity/budget checks before any final-world evaluation begins. Use exactly
20 new final scenarios 22000–22019, absent from archived scenario columns, for each
of the nine final checkpoints (180 missions). These are separate from all earlier
20×20 final worlds and 40×40 development diagnostics. Baselines Greedy, sweep,
uniform unmasked random and uniformly feasible masked random use the same 20
worlds (80 missions). No transfer of 20×20 policies or scored checkpoint selection.

Evaluation uses deterministic first-index masked argmax for RL. Random baseline
action RNGs are independently seeded with the scenario ID. Save every state hash,
mask feasibility, action, reward component and termination event, and all 125
neural outputs/masks. Quality metrics use the full mission horizon, padding an
early failed fleet with zero freshness and full overdue status. Record return,
freshness, overdue fraction, energy, losses, returns/charges, collisions, intact
fleet/failure rate, steps and inference time. Pair every fitted seed/world with
each baseline. Average over worlds within each fitted seed, then report mean and
sample SD across the three fitted seeds; baseline SD is across 20 worlds. Repeated
world/seed rows are not independent trained replicates; no significance claims.

Keep descriptive capability gates separate from completion: per seed, freshness
>= masked random +.01, return >= masked random +.5, losses/failures <= masked
random. Separately require >=90% Greedy freshness, overdue <=Greedy+.05, losses
<=Greedy+.1 and failures <=Greedy. Every seed must pass for an algorithm-level
gate. Freeze these before results; failures do not trigger additional training.

Publish/review protocol, settings, runner, audit, tests and workflow before launch.
Actions runs nine isolated CPU fits, one Torch thread each, at the frozen source.
No retraining on rerun, no stale-branch fits and no silent reuse. A failed/partial
fit remains evidence and is not silently replaced; any recovery must preserve
the original attempts. Retain raw checkpoints/artifacts for 90 days. The final
job downloads all nine artifacts, evaluates once, then independently replays all
260 complete missions and recomputes all tables/gates. Fixed neural reconstruction
tolerance atol 1e-5, rtol 1e-6; actions, hashes, masks, rewards and decisions remain
strict. Verify source, checkpoint bytes, initial tensors, full training budgets,
metadata and unchanged negative archives. Publish audited numeric rows, compressed
scores/traces, learning curves and checksums; raw weights stay in Actions artifacts.
Review final evidence, pass full tests/lint/demo and final CI, merge sr→master under
the owner's standing instruction, and synchronize sr. No 60×60 study or report
writing is included in this run request.
