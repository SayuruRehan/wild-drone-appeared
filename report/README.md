# COMP6008 Report Research Dossier — Word-Equivalent LaTeX Edition

This directory reproduces the complete **Wild_Drone_COMP6008_Report(1).docx** research dossier provided by the group on 18 October 2026, rather than the earlier condensed rewrite.

**Academic integrity:** The COMP6008 guidelines explicitly prohibit generative AI in submitted reports, code documentation and presentation materials. These AI-assisted research materials must not be submitted as the group's assessed work. Each member must independently prepare, verify and attribute the actual submission.

## What is preserved from the Word research dossier

- All three parts: project/MDP/reward/agents/results/critical review; three limitations-linked improvements and three distinct research expansions; individual contribution placeholders.
- The frozen 40x40 main-study performance table, conclusions, descriptive 20x20 vs 40x40 caveats, and two editable figures.
- All **19** references in the original numbering, plus an editable `references.bib` resource.
- Appendices A–C: assignment checklist, code/evidence pointers, and oral-defence questions.
- Original research-dossier warnings, so no failed pilot is represented as a successful learned policy.

## Repository layout

| Resource | Purpose |
| --- | --- |
| `main.tex` | IEEEtran A4 two-column entrypoint |
| `body.tex` | Ordered includes for the complete document |
| `sections/preface.tex` | Source abstract and index terms |
| `sections/part1.tex` | Problem, MDP, algorithms, training, results and critical review |
| `sections/part2.tex` | Three proposed improvements and three future expansions |
| `sections/part3.tex` | Individual contribution table and responsibilities |
| `sections/references.tex` | Word report's reference-section heading |
| `sections/appendices.tex` | All three source appendices |
| `references.tex` | Numbered bibliography as it appears in the report |
| `references.bib` | Companion BibTeX catalogue containing the 19 sources |
| `figures/main_freshness.tex` | Seven-policy frozen-grid bar chart |
| `figures/historical_comparison.tex` | Historical 20x20 versus 40x40 comparison |
| `main.pdf` | Compiled, GitHub Actions-generated output |

`references.tex` is rendered directly to keep the Word report's original citations, order, and wording. `references.bib` is a separate editable metadata resource; changes to its fields do not automatically alter the rendered reference list.

## Build locally

Install a TeX distribution with IEEEtran, PGFPlots, AMS, booktabs, enumitem and XeLaTeX. In the repository root:

```bash
cd report
latexmk -xelatex -interaction=nonstopmode -halt-on-error -pdfxe main.tex
```

The expected output is `report/main.pdf`, seven A4 pages in the verified local build. The GitHub workflow `.github/workflows/build-report.yml` builds on pull requests and pushes to `master`, attaches the PDF as an Actions artifact, and commits the rebuilt PDF back to `master` when the post-merge workflow succeeds.

## Verification

Check for both graphs, 19 references, the three improvements, the three expansions, the contribution table, and appendices A–C. The content was transferred from the uploaded Word dossier; LaTeX floating table/figure positions and page breaks may differ from the six-page Word rendering. The research findings and text are not replaced by new claims.

The frozen-study research evidence remains in `docs/experiments/rl-main-2026-10-11`. This is a research reference for independent assessed preparation, not a submission-ready manuscript.
