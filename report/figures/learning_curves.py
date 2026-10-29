"""Figure 1: training return of the nine main-study fits (10-episode moving mean).

Uses the same training logs and the same 10-episode window as the learning curves in
Section 10 of the notebook. Run from the repository root:
    python report/figures/learning_curves.py
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FITS = ROOT/'docs/experiments/rl-main-2026-10-11/fits'
OUT = Path(__file__).with_name('learning_curves.pdf')
WINDOW = 10
SHADES = ['#000000', '#555555', '#999999']

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(7, 1.75), sharey=True)
for ax, (algorithm, title) in zip(axes, [('dqn', 'DQN'), ('reinforce', 'REINFORCE'),
                                         ('a2c', 'A2C')]):
    for seed, shade in enumerate(SHADES):
        log = pd.read_csv(FITS/algorithm/f'seed-{seed}'/'training.csv')
        ax.plot(log['episode'], log['return'].rolling(WINDOW, min_periods=WINDOW).mean(),
                color=shade, linewidth=0.9, label=f'seed {seed}')
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('Training episode')
    ax.grid(axis='y', color='#dddddd', linewidth=0.5)
axes[0].set_ylabel(f'Return ({WINDOW}-episode mean)')
axes[0].set_ylim(-27.0, -22.7)  # leaves room for the legend below the curves
axes[1].legend(loc='lower center', ncol=3, frameon=False, fontsize=6.5, handlelength=2,
               borderaxespad=0.2, bbox_to_anchor=(0.56, 0))
fig.tight_layout(pad=0.3)
fig.savefig(OUT)
print(f'wrote {OUT.relative_to(ROOT)}')
