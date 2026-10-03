# Phase 2: learning pilot and resumable Colab experiments

The foundation PR is merged. Continue the approved fully observable fleet MDP without
changing its physical or reward rules merely to improve an algorithm's score. The next
goal is a reproducible learning pilot, followed by a user-run experiment on Colab when
larger compute is useful. A100 runtime access is held by the user; this workspace has CPU.

## Design

1. Run 80-episode original 20x20 pilots for all three methods. Evaluate deterministic policies
   against their untrained networks and conventional controllers on validation seeds 5000–5004.
   This is one training seed, sufficient to diagnose behaviour but not superiority. Keep
   final test seeds 10000+ unused. Preserve original checkpoints and metrics.
2. Add explicit CPU/CUDA/auto selection, portable policy loading and a CUDA smoke test
   skipped on CPU. Set deterministic CUDA workspace configuration before model operations;
   record device/runtime. Do not claim an A100 speedup without a benchmark.
3. Save an atomic episode-boundary training state: model/target, optimiser, replay, all
   sampling RNGs, progress and completed rows. Restore it with weights-only loading and
   strict config checks. Resume permits a larger total episode budget and a device change,
   but cannot change algorithm, environment or learning settings silently. Periodic CSV,
   summary and policy snapshots survive a disconnected Colab session.
4. Add progress and validation checkpoint logging. Evaluate every fixed episode interval on
   declared validation seeds; always preserve the last policy, with best-validation policy
   saved separately. Keep final test results outside model/hyperparameter selection. Write
   learning curves and per-seed validation results, not just one final cumulative reward.
5. If the pilot shows weak behaviour, make justified learning-only improvements, record the
   decision and compare under identical environment/seed/episode budgets. A shared full-grid
   spatial representation may explicitly mark drone/base locations; this adds a feature
   transform, not hidden information or a safety shield. REINFORCE may use a learned
   Monte Carlo baseline (no TD bootstrap); its family remains distinct from n-step A2C.
   These choices must be independently ablatable and versioned in checkpoints.
6. Build a private-repo-compatible Colab notebook. Use transient token entry, never put
   credentials into a URL, command output or saved cell. Store run/checkpoint data in the
   user's Drive; include interrupt/resume commands, an A100 smoke check, a bounded benchmark,
   a 20x20 repeated-seed pilot, 40x40 opt-in experiments and a downloadable results archive.

## Tests and execution

- Test device validation, checkpoint portability and a real CUDA step when available.
- Compare uninterrupted vs episode-boundary resumed training for each algorithm: equal
  returns, update counts, model tensors and replay/target sampling on the same runtime.
- Reject mismatched resume settings and validation/training overlap before writing outputs.
- Test spatial channels and Monte Carlo baseline targets, preserving complete-state inputs.
- Test persisted intermediate metrics and learning-curve plots; compile notebook code cells.
- Run the full suite/lint, repeat bounded learning pilots and examine failures.
- Request independent whole-branch review, fix material findings with regression tests,
  publish a new PR, wait for CI, and merge if ready under the user's existing instruction.

Success means ready, reviewable code and an honest pilot analysis. A whole assignment study
or publication claim still requires repeated tuning/held-out experiments and simulator
validation. The Colab handoff must identify exactly which cells to run and which files to
return; the user's available GPU does not imply this agent can operate their Colab runtime.
