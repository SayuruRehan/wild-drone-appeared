# Loss-only supervised comparison at 40×40

Purpose: test whether a decision-focused objective improves the failed supervised
40×40 pilot after an exact-ranking reference reached near-Greedy mission freshness.
This is a separate, prospective development experiment, not trained RL or final testing.

Reuse the original audited training/fresh packets and all three regression fits
from Actions run 38111928600, source 7c9562aa6064f46d96cc9a163de3deb2ecaa3e19.
Verify complete parent source, archive inputs, dataset/checkpoint byte counts and
SHA-256 identities before fitting. Use only training worlds 0–199, fresh ranking
worlds 6800–6815 and complete mission worlds 7000–7019. No collection, new worlds,
checkpoint selection or opening confirmation 5800–5819/final 10000+.

The only fitting intervention is the loss. Keep the original centered network,
434,877 parameters, environment, observation transform, initial tensors, seeds
0/1/2, informative-state indices, uniform sample sequence, 2,000 updates, batch 64,
Adam .001, clip .5, one CPU thread and deterministic Torch algorithms. Reuse
audited archived regression fits as the control instead of repeating them. Check
new initial tensors against each archived initial checkpoint and sample digests
against each original regression fit. Export initial/final checkpoints and losses.

The new objective is the negative log probability of the tied feasible best set:
`logsumexp(scores[feasible]) - logsumexp(scores[tied_best])`, averaged across
sampled states. Tied best means original float64 gain within absolute 1e-12 of
the maximum feasible gain, matching the established ranking metric. Temperature
is fixed at 1; no auxiliary regression term, smoothing, margin or tuned weight.
All probability mass on any tied optimum is sufficient; no uniform target over
ties is required. Infeasible actions contribute neither loss nor gradient. Exclude
only the original zero-range/non-informative states from fitting. Model outputs
are unconstrained ranking logits, not calibrated gains or RL Q values.

Finish all three fits before reading evaluation predictions. Measure initial
fresh and final training/fresh regret and tied-best match; keep full float32 score
vectors and exact gain/mask packet identities. Masked MSE may be recorded by the
existing metric helper but is not comparable calibration evidence for logits.
Use the original first-index masked argmax and full-horizon mission accounting.
Run 60 new missions, 20 per fitted seed, plus archived matched regression/Greedy/
masked-random controls and the exact-inspection reference as context. Report all
original mission metrics, per-world paired rows, every action/hash/reward component,
and fitted-seed means/sample SD. Do not treat 60 world/seed rows as independent
replicates or make significance claims. Native Greedy's mask/movement preference
remains a disclosed contextual difference.

For each of the three seeds, define objective-effect checks: fresh normalized
regret no higher than its matching regression seed, freshness improvement >=.01,
overdue fraction no higher, mean losses no higher and fleet failures no higher.
All seeds must pass to call this an effective objective intervention. Separately
retain the original ranking capability checks (training regret <=.05, fresh <=.20,
fresh best match >=25%) and all original mission gates (freshness >=90% Greedy,
overdue <=Greedy+.05, losses <=Greedy+.1, failures <=Greedy, freshness >=random+.01,
return >=random+.5). Do not promote a best seed or modify these after results.
If negative, publish it and stop this objective direction without more fits.
Passing supports designing actual RL comparison; it does not establish RL success.

Freeze/review/publish runner, tests, audit and this protocol before launching.
Retain the complete raw new run/checkpoints as a GitHub Actions artifact; publish
numeric rows, full compressed scores/traces, audit and checksums in the repository.
Independently replay all 18,000 planned mission steps, rebuild observations/masks,
recompute neural outputs, require every selected action exactly, and independently
recompute all ranking/mission/paired summaries and decisions. Float32 reconstruction
tolerance is fixed at atol 1e-5, rtol 1e-6; actions, hashes and gates remain strict.
Parent exact gains have already passed simulator-clone audit; verify those packet
identities rather than recollecting/relabeling them. Audit initial tensors, full
sample-index digest/counts, all checkpoint bytes and unchanged parent identities.
Pass full tests/lint/demo and final CI, review the evidence, merge sr→master under
the owner's standing instruction, then synchronize sr.
