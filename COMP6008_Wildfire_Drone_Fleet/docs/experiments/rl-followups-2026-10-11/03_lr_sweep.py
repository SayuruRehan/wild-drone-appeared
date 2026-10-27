"""Task 3 - one-seed learning-rate sweep at 40x40.

The frozen study used a single learning rate per algorithm with no tuning at 40x40.  This
script asks whether that choice mattered: each algorithm is trained from scratch at three
learning rates (frozen value / 3, frozen value, frozen value x 3) with everything else
exactly as frozen (300 episodes, training worlds 0-299, masked actions, training seed 0).

    dqn        3e-5   1e-4 (frozen)   3e-4
    reinforce  3e-4   1e-3 (frozen)   3e-3
    a2c        3e-5   1e-4 (frozen)   3e-4

The final (episode-300) weights are then evaluated by masked argmax on 20 VALIDATION worlds,
23000-23019.  These are distinct from the frozen final worlds 22000-22019 and from the
training worlds, and the script refuses to run if they appear in any archived CSV.  The
final worlds are never touched here, so the frozen final comparison stays untuned.  Greedy
and masked random are flown on the same validation worlds as references.

This is ONE training seed per setting.  The frozen study's spread across three seeds at a
fixed learning rate (see results/planner_vs_frozen_table.md) is the yardstick for whether a
difference between learning rates is larger than seed noise.

The frozen-rate fit repeats the archived seed-0 fit; ``reproduces_frozen_training`` in the
summary records whether its training curve matches the archive on this machine.

A fit whose ``policy.pt`` already exists is not retrained, but only after the file is checked
to be the finished fit this script would train (same algorithm, learning rate, seed,
environment and 300 completed episodes).  Each checkpoint's SHA-256 is then compared with
``results/lr_sweep/SHA256SUMS``, the hashes of the checkpoints behind the committed results.

Outputs (in ``results/``):
    lr_sweep/<algorithm>/lr-<rate>/   policy.pt, training.csv, training_summary.json, ...
    lr_sweep/SHA256SUMS               hashes of the published checkpoints (read, not written)
    lr_sweep_episodes.csv             validation missions
    lr_sweep_summary.csv              one row per (algorithm, learning rate) + references
    lr_sweep.png                      validation freshness and training curves

Run (9 fits in parallel, one CPU thread each; DQN dominates at roughly 20-25 minutes):
    .venv/bin/python 03_lr_sweep.py --workers 9
"""

import argparse
import csv
import time
from dataclasses import replace
from multiprocessing import get_context

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from common import (  # noqa: E402
    ALGORITHMS,
    EPISODE_COLUMNS,
    FROZEN_ARCHIVE,
    METRICS,
    RESULTS,
    ROOT,
    frozen_setup,
    read_csv,
    run_jobs,
    sha256,
    summarise,
    write_csv,
)

FACTORS = (1/3, 1.0, 3.0)                      # multiples of the frozen learning rate
VALIDATION_WORLDS = tuple(range(23000, 23020))
TRAINING_SEED = 0
OUT = RESULTS/'lr_sweep'
MANIFEST = OUT/'SHA256SUMS'                    # `shasum -a 256` format, paths relative to OUT
COLOURS = {'low': '#2a78d6', 'frozen': '#eb6834', 'high': '#1baf7a'}
LEVELS = ('low', 'frozen', 'high')


def rates(settings, algorithm):
    """Three learning rates, rounded to one significant figure (3e-5, 1e-4, 3e-4, ...)."""
    return [float(f'{settings[algorithm]["learning_rate"]*f:.0e}') for f in FACTORS]


def fit_dir(algorithm, rate):
    return OUT/algorithm/f'lr-{rate:g}'


def check_validation_worlds_are_fresh(final_worlds):
    """Validation worlds must not be final worlds, training worlds or archived evidence."""
    assert not set(VALIDATION_WORLDS) & set(final_worlds)
    assert not set(VALIDATION_WORLDS) & set(range(300))
    for path in (ROOT/'docs'/'experiments').rglob('*.csv'):
        # The sweep's own saved validation outputs are not earlier independent evidence.
        if path.is_relative_to(RESULTS):
            continue
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                for key in ('scenario_seed', 'world', 'world_seed'):
                    value = row.get(key)
                    if value and value.isdigit() and int(value) in VALIDATION_WORLDS:
                        raise ValueError(f'validation world {value} already used in {path}')


def check_reusable(directory, algorithm, config, train_config):
    """Refuse an existing ``policy.pt`` unless it is the finished fit this job would train."""
    import torch
    saved = torch.load(directory/'policy.pt', map_location='cpu', weights_only=True)
    expected = {'algorithm': algorithm, 'env_config': config.to_dict(),
                'train_config': train_config.to_dict(),
                'completed_episodes': train_config.episodes}
    wrong = [key for key, value in expected.items() if saved.get(key) != value]
    log = directory/'training.csv'
    if not log.is_file() or len(read_csv(log)) != train_config.episodes:
        wrong.append('training.csv')
    if wrong:
        raise ValueError(f'{directory/"policy.pt"} is not the expected fit (mismatch in '
                         f'{", ".join(wrong)}); delete {directory} to retrain it')


def train(job):
    """Train one (algorithm, learning rate) fit with the repo's own masked trainer."""
    algorithm, rate = job
    import torch

    from wild_drone.learning import TrainConfig
    from wild_drone.masked_learning import train_masked
    config, settings, _ = frozen_setup()
    directory = fit_dir(algorithm, rate)
    # Only learning_rate differs from the frozen recipe; progress printing is silenced.
    train_config = replace(TrainConfig(**settings[algorithm], seed=TRAINING_SEED),
                           learning_rate=rate, progress_interval=0)
    if (directory/'policy.pt').exists():        # finished earlier: reuse only if it matches
        check_reusable(directory, algorithm, config, train_config)
        return algorithm, rate, None
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    summary = train_masked(algorithm, config, train_config, directory)
    return algorithm, rate, summary['elapsed_seconds']


def report_published(fits):
    """Say which checkpoints are byte-identical to the published ones in the manifest."""
    published = {name: digest for digest, name in
                 (line.split() for line in MANIFEST.read_text().splitlines())}
    different = [str(fit_dir(a, r).relative_to(OUT)) for a, r in fits
                 if sha256(fit_dir(a, r)/'policy.pt')
                 != published.get(f'{fit_dir(a, r).relative_to(OUT)}/policy.pt')]
    print(f'{len(fits)-len(different)}/{len(fits)} checkpoints match {MANIFEST.name}')
    if different:
        print('  retrained here, so results will differ from the committed ones: '
              + ', '.join(different))


def reproduces_frozen(algorithm, rate):
    """Does the frozen-rate fit retrace the archived seed-0 training returns?"""
    new = [float(r['return']) for r in read_csv(fit_dir(algorithm, rate)/'training.csv')]
    old = [float(r['return']) for r in read_csv(
        FROZEN_ARCHIVE/'fits'/algorithm/f'seed-{TRAINING_SEED}'/'training.csv')]
    return bool(np.allclose(new, old, rtol=0, atol=1e-9))


def plot(table, settings):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    reference = {r['policy']: r for r in table if r['policy'] in ('greedy', 'masked_random')}
    for column, algorithm in enumerate(ALGORITHMS):
        rows = [r for r in table if r['policy'] == algorithm]
        ax = axes[0, column]
        for i, (row, level) in enumerate(zip(rows, LEVELS, strict=True)):
            ax.bar(i, row['weighted_freshness_mean'], yerr=row['weighted_freshness_sd'],
                   color=COLOURS[level], width=0.6, capsize=4, edgecolor='white')
            ax.text(i, row['weighted_freshness_mean']+row['weighted_freshness_sd']+0.004,
                    f"{row['weighted_freshness_mean']:.3f}", ha='center', fontsize=9)
        for name, style in (('greedy', '-'), ('masked_random', ':')):
            ax.axhline(reference[name]['weighted_freshness_mean'], color='#52514e',
                       linestyle=style, linewidth=1.2, label=name)
        ax.set_xticks(range(3), [f"{r['learning_rate']:g}" + ('\n(frozen)' if lv == 'frozen'
                                 else '') for r, lv in zip(rows, LEVELS, strict=True)])
        ax.set_title(f'{algorithm}: validation freshness')
        ax.set_xlabel('learning rate')
        ax.set_ylim(0, reference['greedy']['weighted_freshness_mean']*1.15)
        if column == 0:
            ax.set_ylabel('Weighted freshness, 20 validation worlds')
            ax.legend(frameon=False, fontsize=9, loc='lower left',
                      bbox_to_anchor=(0.0, 0.55))
        ax = axes[1, column]
        for row, level in zip(rows, LEVELS, strict=True):
            returns = np.array([float(r['return']) for r in read_csv(
                fit_dir(algorithm, row['learning_rate'])/'training.csv')])
            smooth = np.convolve(returns, np.ones(20)/20, mode='valid')
            ax.plot(np.arange(20, len(returns)+1), smooth, color=COLOURS[level], linewidth=2,
                    label=f"lr {row['learning_rate']:g}")
        ax.set_title(f'{algorithm}: training return (20-episode mean)')
        ax.set_xlabel('training episode')
        ax.legend(frameon=False, fontsize=9)
        if column == 0:
            ax.set_ylabel('Return while exploring')
    for ax in axes.flat:
        ax.grid(axis='y', color='#e6e5e0', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('40x40 learning-rate sweep, one training seed, 300 episodes, validation '
                 'worlds 23000-23019\nbars: sample SD across the 20 validation worlds',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS/'lr_sweep.png', dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workers', type=int, default=9)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    _, settings, final_worlds = frozen_setup()
    check_validation_worlds_are_fresh(final_worlds)
    fits = [(a, r) for a in ALGORITHMS for r in rates(settings, a)]

    if not args.plot_only:
        started = time.perf_counter()
        seconds = {}
        with get_context('spawn').Pool(args.workers) as pool:
            for algorithm, rate, elapsed in pool.imap_unordered(train, fits):
                seconds[algorithm, rate] = elapsed
                print(f'trained {algorithm} lr={rate:g}: '
                      + (f'{elapsed:.0f}s' if elapsed else 'already present'), flush=True)
        print(f'all fits ready after {time.perf_counter()-started:.0f}s wall')
        report_published(fits)

        # Validation missions: every final checkpoint, plus two references.
        jobs = [({'learning_rate': rate}, {}, algorithm, TRAINING_SEED, world,
                 str(fit_dir(algorithm, rate)/'policy.pt'))
                for algorithm, rate in fits for world in VALIDATION_WORLDS]
        jobs += [({'learning_rate': ''}, {}, name, None, world, None)
                 for name in ('greedy', 'masked_random') for world in VALIDATION_WORLDS]
        rows = run_jobs(jobs, min(args.workers, 8), label='validation')
        write_csv(RESULTS/'lr_sweep_episodes.csv', rows, ('learning_rate',)+EPISODE_COLUMNS)

    rows = read_csv(RESULTS/'lr_sweep_episodes.csv')
    table = []
    for algorithm, rate in fits:
        group = [dict(r, training_seed='') for r in rows
                 if r['policy'] == algorithm and float(r['learning_rate'] or 0) == rate]
        stats = summarise(group)                 # SD across the 20 validation worlds
        training = read_csv(fit_dir(algorithm, rate)/'training.csv')
        frozen_rate = rate == rates(settings, algorithm)[1]
        table.append({'policy': algorithm, 'learning_rate': rate, 'frozen_rate': frozen_rate,
                      'training_seed': TRAINING_SEED, 'worlds': len(group),
                      **{f'{m}_mean': stats[m][0] for m in METRICS},
                      **{f'{m}_sd': stats[m][1] for m in METRICS},
                      'train_return_first50': np.mean([float(r['return'])
                                                       for r in training[:50]]),
                      'train_return_last50': np.mean([float(r['return'])
                                                      for r in training[-50:]]),
                      'reproduces_frozen_training': reproduces_frozen(algorithm, rate)
                      if frozen_rate else ''})
    for name in ('greedy', 'masked_random'):
        stats = summarise([r for r in rows if r['policy'] == name])
        table.append({'policy': name, 'learning_rate': '', 'frozen_rate': '',
                      'training_seed': '', 'worlds': stats['units'],
                      **{f'{m}_mean': stats[m][0] for m in METRICS},
                      **{f'{m}_sd': stats[m][1] for m in METRICS}})
    write_csv(RESULTS/'lr_sweep_summary.csv', table)
    plot(table, settings)
    print(f"{'policy':14s}{'lr':>8s}{'freshness':>11s}{'overdue':>9s}{'return':>9s}"
          f"{'losses':>8s}{'train last50':>14s}  matches archive")
    for r in table:
        print(f"{r['policy']:14s}{str(r['learning_rate']):>8s}"
              f"{r['weighted_freshness_mean']:11.4f}{r['overdue_fraction_mean']:9.4f}"
              f"{r['return_mean']:9.3f}{r['total_losses_mean']:8.2f}"
              f"{r.get('train_return_last50', float('nan')):14.3f}  "
              f"{r.get('reproduces_frozen_training', '')}")


if __name__ == '__main__':
    main()
