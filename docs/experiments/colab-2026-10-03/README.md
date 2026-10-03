# T4 validation experiment — 2026-10-03

This directory contains the derived analysis of the completed 1,000-episode, three-fit
validation study. See [the report](../../colab1000-results.md). The original simulator and
training revision is `93716b264ad568c0a7c9c754f140f4969880d8ba`; analysis changes do not
retroactively change that provenance.

## Upload the original results

Place these files in the repository and commit them:

| Artifact | Repository path |
|---|---|
| Original archive | `docs/experiments/colab-2026-10-03/wild-drone-results.zip` |
| Extracted ZIP contents | `docs/experiments/colab-2026-10-03/raw/` |
| Executed notebook, renamed | `notebooks/executed/colab-2026-10-03.ipynb` |

Extract **the contents** of the ZIP directly into `raw/`. For example,
`raw/runtime-benchmark.csv`, `raw/validation20-v2/protocol.json` and
`raw/validation20-v2/dqn/seed-0/training.csv`. Preserve all seed folders and original filenames.
GitHub keeps a ZIP as one file; it does not automatically extract it.

Keep the reusable `notebooks/colab_experiments.ipynb` as the current template. The executed
copy records the older revision and the manual import fix used for this completed run.
`runs/` is intentionally ignored, so it is not the upload destination for committed evidence.
Model and replay checkpoints remain in Drive; the original archive contains metrics and figures.

`archive-manifest.json` records SHA-256 and byte size for each of the 43 original archive
entries, plus the original ZIP checksum. These paths are relative to `raw/`. Derived files:

- `summary.csv`: scenario means per fitted model, then mean and sample SD across fits;
  baseline SD is across scenarios.
- `selected-checkpoints.csv`: selected episode and final score for each fit, with measured runtime.
- `checkpoint-diagnostics.csv`, `diagnostic-summary.csv`, `diagnostic-manifest.json`:
  CPU replay of the Drive policies, plus separately declared categorical actor sampling.

The raw archive and executed notebook are uploaded manually by the project owner. Their
absence here does not mean the derived analysis was generated from a different run.
Selected scores reuse validation seeds 5000–5004 and cannot establish final generalisation.
