# COMP6008 group report

IEEE two-column (A4) report for the COMP6008 group project, *Coordinated Wildfire Inspection
with a Battery-Constrained Drone Fleet*. The compiled report is `main.pdf`.

## Structure

| File | Contents |
| --- | --- |
| `main.tex` | IEEEtran entry point: title, authors, abstract, keywords |
| `body.tex` | Order of the three parts, references and appendix |
| `sections/part1.tex` | Part I: introduction, MDP and environment, agents, 20×20 development, tuning, main study, sensitivity, critical review |
| `sections/part2.tex` | Part II: limitations, three improvements, three future-work directions |
| `sections/part3.tex` | Part III: statement of contributions and section authorship |
| `sections/appendices.tex` | Appendix: update equations and fire-spread derivation |
| `references.tex` | IEEE reference list, numbered in order of first citation |
| `figures/learning_curves.pdf` | Training returns of the nine main-study fits |

Every section starts with a note naming its author, as the unit guidelines require. The
page limit is eight pages before the references and appendix.

## Sources of the numbers

All results come from the archived experiments in the submission folder:

- 20×20 development study: `docs/experiments/final-2026-10-05/`
- 40×40 main study: `docs/experiments/rl-main-2026-10-11/` (summary, training logs)
- Learning-rate sweep, planner and sensitivity: `docs/experiments/rl-followups-2026-10-11/results/`
- Settings: `configs/rl-main-frozen.json` and `configs/development.json`

`figures/learning_curves.pdf` plots the 20-episode moving mean of `return` from
`docs/experiments/rl-main-2026-10-11/fits/<agent>/seed-<n>/training.csv`.

## Build

With a TeX distribution that includes IEEEtran:

```bash
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The workflow `.github/workflows/build-report.yml` builds the PDF on pull requests that
change `report/`, uploads it as an artifact, and commits the rebuilt `main.pdf` to `master`
after a merge. It can also be run manually from the Actions tab.
