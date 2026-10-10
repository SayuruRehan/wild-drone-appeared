# Matched inspection-input comparison at 20×20

Drone-centered inspection planes improved fresh-world action rankings under
the same model size, training corpus and fitting budget. They beat both matched
controls in all three seeds, but **none passed the full prospective capability
criterion**. These are supervised one-step inspection results; there is no
RL, long-term mission or 40×40 success claim.

Protocol and source were published before fresh collection/fitting at
`2cebf052f6fc6f0a4a3b57c87c41591ac4a421db` on `sr`.
See [the prospective protocol](../../centered-inspection-protocol.md).
The parent diagnostic source is `2ff35318b67ceaa8a7bd186b85650095771e083d`.
Collection, nine fits and evaluation took 154.99 seconds on one CPU thread.
No Colab GPU run was needed.

## Primary fresh-development result

Means ± sample SD across three fitted seeds, with 125 informative states per
fit. All 128 planned natural snapshots were observed; three zero-gain-range
states are excluded from informative means. Fresh worlds 6200–6215 were fixed
before fitting. Their results are development evidence, not a final test.

| Input arm | Normalized gain regret ↓ | Tied-best match ↑ | Masked regression MSE ↓ |
|---|---:|---:|---:|
| Current | 0.448607 ± 0.016381 | 11.20% ± 2.88 pp | 0.118404 ± 0.004457 |
| Absolute field | 0.437609 ± 0.009952 | 10.40% ± 1.60 pp | 0.120934 ± 0.005700 |
| Drone-centered field | **0.292269 ± 0.023170** | **17.87% ± 0.92 pp** | **0.079665 ± 0.005417** |

The shared training-action-mean control has regret 0.496501 and best match 5.60%.
Uniform feasible random has expected regret 0.608362 and best match 5.06%.
An exact inspection reference has zero regret; it is a label-based diagnostic
reference, not a deployable learned policy or a mission baseline.

| Seed | Current regret | Absolute regret | Centered regret | Centered − current | Centered − absolute | Centered best match |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0.449037 | 0.438660 | 0.273671 | −0.175365 | −0.164989 | 18.40% |
| 1 | 0.432016 | 0.427173 | 0.318224 | −0.113792 | −0.108949 | 16.80% |
| 2 | 0.464769 | 0.446993 | 0.284912 | −0.179857 | −0.162081 | 18.40% |

Centered inputs reduce mean regret by about 34.85% relative to current inputs.
The absolute-field arm alone yields little primary improvement. This supports
the specific translation/cropping treatment in this experiment, beyond simply
making the cell inspection field explicit. Paired differences are descriptive;
three fits do not support a general significance or architecture superiority claim.

## Criterion and remaining gap

Every centered seed passed training regret ≤0.05 and improvement ≥0.10 over
both matched controls. Every seed failed fresh regret ≤0.20 and best match ≥25%.
See `completion.json` for each condition, without rounding.

All arms fit the same 303 informative training states closely. Mean training
regrets are approximately 0.001392 (current), 0.001665 (absolute), and 0.001623
(centered). The remaining train-to-fresh gap is substantial. More fitting on
these exact states is not established as a solution by this comparison.

Reused worlds 6000–6015 remain secondary development evidence: mean final
regrets 0.462413, 0.431406 and 0.304933 for current, absolute and centered.
They were previously examined and are not fresh tests. Full denominators,
initial controls, losses and per-state metrics are preserved.

Secondary common-mask preferred-action changes on 32 constructed pairs:

| Arm | Urgency changes, seeds 0 / 1 / 2 | Position changes, seeds 0 / 1 / 2 |
|---|---|---|
| Current | 5 / 4 / 8 | 9 / 14 / 10 |
| Absolute | 21 / 17 / 12 | 0 / 0 / 1 |
| Centered | 19 / 22 / 23 | 10 / 10 / 10 |

Every no-op pair has exactly identical score vectors. A changed preferred
action does not itself establish correct responsiveness: the full before/after
gain regret, order changes and deployment-mask effects are in the pair archive.
These artificial urgency/position maps are separate from natural outcomes.

The next useful experiment is broader training-state coverage while holding
centered inputs, model and update budget fixed, with newly declared evaluation
worlds. Check whether held-out ranking quality improves before changing RL or
scaling the mission. Confirmation worlds 5800–5819 and final-test worlds 10000+
remain unopened. No additional fitting, RL default promotion or 40×40 run is
part of this comparison.

## Matching and verification

- Exact parent dataset SHA-256/bytes and case/gain/mask identities verified.
- Each arm has 127,677 parameters; per-seed initial tensors are identical.
- All arms use identical sampled batch sequences, recorded in `matching.json`.
- Every current-arm final tensor and all 6,000 losses reproduce the parent fits.
- No action labels, footprints, feasible masks or hidden simulator state enter
  the input transform. Both derived arms use observation-only cell urgency.
- Independent field-formula calculation checked all 734 dataset states with
  maximum tolerance 1e-7. Translation, cropping and zero padding checked exactly.
- All 18 checkpoints verified; 13,212 complete score vectors reproduced exactly.
- All 128 fresh states replayed with original collector seeds; all 16,000 actual
  simulator action rewards agree with archived labels to tolerance 1e-13.
- Separate scalar/action-pair recomputation checks all 11,480 natural rows,
  1,920 pair rows, 60 fit/control rows, summaries, paired differences and criterion.
- Local validation: 198 tests passed, five CUDA-only skips; Ruff passed.

The numerical audit uses separate implementations in this repository; it is
not an independent human review or statistical replication.

## Archive and reproduction

`natural.csv.gz`, `pairs.csv.gz` and `inputs.json.gz` use lossless compression.
The 18 `*-scores.npz` files retain complete float32 vectors for all four datasets;
open with `numpy.load(path, allow_pickle=False)`. `per-fit.csv`, `summary.csv`,
`pair-summary.csv`, `completion.json` and all nine `training.csv` files preserve
unrounded numeric results. Source, dataset, input, sampler and checkpoint
identities and audit outputs are included. `checksums.json` covers every archive
file other than itself.

Run numerical verification from a checkout with NumPy; no model binaries needed:

```bash
python docs/experiments/centered-inspection-2026-10-10/recompute.py
```

Binary datasets/weights remain ignored under `runs/`; their identities are
archived. Generate the parent diagnostic from its frozen source/protocol first,
then run this frozen comparison in a fresh output directory:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_centered_inspection.py \
  --parent-root runs/inspection-targets-2026-10-10 \
  --out runs/centered-inspection-2026-10-10 \
  --source-revision 2cebf052f6fc6f0a4a3b57c87c41591ac4a421db
PYTHONPATH=src python docs/experiments/centered-inspection-2026-10-10/replay.py \
  --run runs/centered-inspection-2026-10-10 \
  --parent-root runs/inspection-targets-2026-10-10
```

Exact parent binary identities are enforced; a different Python/Torch/NumPy
environment or serializer can differ. This run used Python 3.12.14,
Torch 2.5.1+cpu and NumPy 2.3.5. Keep the archived identities rather than replacing
them to make a reproduction pass.
