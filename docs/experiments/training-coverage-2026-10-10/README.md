# Broader training coverage at 20×20

Expanding natural training coverage from 40 to 200 worlds improved the fixed
drone-centered inspection scorer on fresh development worlds. All three seeds
passed both prospectively declared data-effect and capability criteria. This
supports a subsequent 20×20 mission/RL validation; it does not establish improved
long-term coordination, mission performance or 40×40 performance.

## Fixed comparison

Source and protocol were published before collection/fitting at
`a2a6b4e7fa68ff71d735542d3e0633eb7a09886a` (PR #28). The preceding centered
experiment used source `2cebf052f6fc6f0a4a3b57c87c41591ac4a421db` (PR #27).
See [the prospective protocol](../../training-coverage-protocol.md).

Only the training corpus changed. Both arms used the unchanged centered
representation, 127,677-parameter network, normalized inspection targets,
masked-random and Greedy collectors, snapshots at steps 16/32/64/96, seeds 0/1/2,
and 2,000 Adam updates per fit at .001, batch 64 and gradient clip .5. The six
fits ran on one CPU thread and completed before evaluation predictions were
read. Collection and fitting took 331.1 seconds; no GPU was needed.

| Dataset | Worlds | Planned states | Observed | Informative |
| --- | --- | ---: | ---: | ---: |
| Small training | 0–39 | 320 | 319 | 303 |
| Expanded training | 0–199 | 1,600 | 1,591 | 1,521 |
| Primary fresh development | 6400–6415 | 128 | 127 | 121 |
| Reused development | 6200–6215 | 128 | 128 | 125 |

Expanded training retains every original tensor and record unchanged. Missing
states are logged early Greedy terminations: original world 20, appended worlds
98/134/146/147/158/160/174/192, and fresh world 6412, each at snapshot 96.
Zero feasible-gain-range states remain archived and are excluded from fitting
and informative-state means. No favorable-world filtering was performed.
Confirmation worlds 5800–5819 and final-test worlds 10000+ remain unopened.

## Results

Primary results are means ± sample SD over three fitted seeds, each evaluated
on the same 121 informative fresh states. Regret is normalized one-step
inspection gain regret; best match includes all gain-tied optimal actions.

| Fresh metric | Small training | Expanded training |
| --- | ---: | ---: |
| Inspection regret ↓ | 0.285042 ± 0.029209 | 0.064422 ± 0.013474 |
| Best-action match ↑ | 17.91% ± 2.52 pp | 51.79% ± 5.98 pp |
| Masked regression MSE ↓ | 0.081153 ± 0.007284 | 0.032550 ± 0.002959 |

Mean regret fell 77.4%; mean best match rose 33.88 percentage points. These are
descriptive comparisons, without a significance claim.

| Seed | Small fresh regret | Expanded fresh regret | Expanded − small | Small best match | Expanded best match | Expanded own-training regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.255138 | 0.050602 | −0.204536 | 20.66% | 58.68% | 0.011972 |
| 1 | 0.286486 | 0.077521 | −0.208964 | 15.70% | 48.76% | 0.009193 |
| 2 | 0.313502 | 0.065144 | −0.248358 | 17.36% | 47.93% | 0.009841 |

Every seed passed both separate decisions: regret improved by at least .05 and
best match did not fall; expanded own-training regret was ≤.05, fresh regret
≤.20 and fresh best match ≥25%. See `completion.json` for every condition.

The secondary, previously examined development set also improved: regret
0.292269 ± 0.023170 → 0.082814 ± 0.013536 and best match
17.87% ± 0.92 pp → 48.53% ± 5.68 pp. It is not a fresh test.
Own-training means use different populations and should not be treated as a
held-out comparison.

Fresh action-mean controls had regret 0.486514 (small) and 0.451717 (expanded);
uniform feasible-random expected regret was 0.595390. Their best-match values
were 6.61%, 8.26% and 5.23%, respectively. The exact inspection-label reference
has zero regret by construction and is a diagnostic reference.

Each fit made 128,000 sample draws. Initial weights matched exactly per seed,
but sampler IDs differ with corpus size. Mean exposure fell from 422.44 draws
per informative small state to 84.16 per expanded state. `sampling.json`
preserves per-state counts, unique exposure and sample-sequence digests.

## Evidence and reproduction

The archive includes all six training histories, 12,865 natural metric rows,
36 per-fit summaries, complete lossless float32 scores for 8,025 network-state
vectors, gain/mask/input records, controls, source/dataset/checkpoint identities,
and separate numerical and simulator replay audits. Gzip and NPZ compression do
not round scores or labels. Initial fresh/reused scores share one file per seed;
initial training scores and small-model scores on appended training states were
not measured. Raw dataset and checkpoint binaries remain under ignored `runs/`;
their hashes are archived, rather than uploading model binaries to Git.

`recompute.py` independently recalculates argmax/ties, normalization, every
natural row, aggregates and both criteria without production metric helpers.
`replay.py` verifies 28 source hashes, four dataset files, 12 checkpoints,
2,165 transformation inputs, and exact original tensor/record prefixes. It
replays 1,399 newly collected states and all 174,875 actual simulator action
rewards, and reproduces all 8,025 exported score vectors exactly. All six small
checkpoint tensor comparisons and 6,000 small-fit losses match their parent.
Both verification JSON files report PASS. Supervised exports are rejected by
RL checkpoint loaders. Repository checks: 202 tests passed, five CUDA-only
tests skipped; Ruff and the CLI demo passed.

From the frozen source checkout with dependencies installed and the preceding
raw runs available:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_training_coverage.py \
  --parent-root runs/inspection-targets-2026-10-10 \
  --centered-root runs/centered-inspection-2026-10-10 \
  --out runs/training-coverage-2026-10-10 \
  --source-revision a2a6b4e7fa68ff71d735542d3e0633eb7a09886a
python docs/experiments/training-coverage-2026-10-10/recompute.py
PYTHONPATH=src python docs/experiments/training-coverage-2026-10-10/replay.py \
  --run runs/training-coverage-2026-10-10 \
  --parent-root runs/inspection-targets-2026-10-10 \
  --centered-root runs/centered-inspection-2026-10-10
```

The runner requires a fresh output directory. `checksums.json` covers all other
archive files. Next, validate the learned scorer in full 20×20 trajectories,
checking inspection freshness, fleet survival, energy/charging and mission
performance before RL fine-tuning or promotion to the proposed 40×40 scope.
