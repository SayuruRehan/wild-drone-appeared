"""Reproducible episode evaluation and provenance, independent of training code."""

import csv
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from wild_drone.environment import WildfireFleetEnv


def validate_seed_split(train_start, episodes, evaluation_seeds):
    overlap = sorted(set(range(train_start, train_start+episodes)) & set(evaluation_seeds))
    if overlap:
        raise ValueError(f'training and evaluation scenario seeds overlap: {overlap}')


def write_metadata(out, env_config, **details):
    versions = {}
    for package in ['numpy', 'gymnasium', 'torch', 'matplotlib', 'pillow']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True))
    except (subprocess.SubprocessError, FileNotFoundError):
        revision, dirty = None, None
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    metadata = {'env_config': env_config.to_dict(), 'python': platform.python_version(),
                'packages': versions, 'git_revision': revision, 'git_dirty': dirty, **details}
    (path/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(config, policy, scenario_seed, *, name, training_seed=None, gif_path=None):
    env = WildfireFleetEnv(config, render_mode='rgb_array' if gif_path else None)
    observation, _ = env.reset(seed=scenario_seed)
    frames = [Image.fromarray(env.render())] if gif_path else []
    fresh, overdue, decisions = [], [], []
    done, total = False, 0.0
    while not done:
        started = time.perf_counter()
        action = policy.act(env, observation)
        decisions.append((time.perf_counter()-started)*1000)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total += reward
        fresh.append(info['weighted_freshness'])
        overdue.append(info['overdue_fraction'])
        if gif_path and (env.elapsed % 2 == 0 or done):
            frames.append(Image.fromarray(env.render()))
    if gif_path:
        path = Path(gif_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=250, loop=0, optimize=False)
    # Failed fleets receive zero freshness and full overdue for the remaining mission.
    # This avoids favouring policies that end early when reporting average quality.
    missing = config.horizon-len(fresh)
    return {'policy': name, 'training_seed': training_seed, 'scenario_seed': scenario_seed,
            'grid_size': config.grid_size, 'horizon': config.horizon, 'steps': env.elapsed,
            'return': total, 'weighted_freshness': float(sum(fresh)/config.horizon),
            'overdue_fraction': float((sum(overdue)+missing)/config.horizon),
            'energy_used': env.totals['energy_used'], 'total_losses': int(env.lost.sum()),
            'safe_returns': env.totals['safe_returns'],
            'charge_completions': env.totals['charge_completions'],
            'collision_attempts': env.totals['collision_attempts'],
            'decision_ms': float(np.mean(decisions))}


def plot_results(csv_path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('no experiment rows to plot')
    policies = list(dict.fromkeys(row['policy'] for row in rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, metric, title in zip(axes, ['weighted_freshness', 'overdue_fraction', 'total_losses'],
                                  ['Inspection freshness (higher)', 'Overdue (lower)',
                                   'Drone losses (lower)'], strict=True):
        means, stds = [], []
        for name in policies:
            selected = [r for r in rows if r['policy'] == name]
            # Aggregate each fitted training seed first; do not treat its scenarios
            # as independent fitted models. Baselines aggregate over scenarios.
            seeds = list(dict.fromkeys(r['training_seed'] for r in selected))
            if seeds == ['']:
                units = [float(r[metric]) for r in selected]
            else:
                units = [np.mean([float(r[metric]) for r in selected
                                  if r['training_seed'] == seed]) for seed in seeds]
            means.append(np.mean(units))
            stds.append(np.std(units, ddof=1) if len(units) > 1 else 0)
        ax.bar(policies, means, yerr=stds, capsize=4, color='#168aab')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle('Pilot comparison — mean ± sample SD; see CSV for seeds and budgets')
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning(csv_paths, out):
    """Per-run curves; validation mean is over scenarios, not independent fitted models."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for index, csv_path in enumerate(csv_paths):
        path = Path(csv_path)
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f'no training rows in {path}')
        episodes = [int(r['episode']) for r in rows]
        returns = [float(r['return']) for r in rows]
        label = path.parent.name
        color = f'C{index % 10}'
        axes[0].plot(episodes, returns, alpha=0.2, color=color)
        window = min(10, len(rows))
        axes[0].plot(episodes[window-1:], np.convolve(returns, np.ones(window)/window,
                                                   mode='valid'), label=label, color=color)
        validation = path.parent/'validation.csv'
        if validation.exists():
            with validation.open() as f:
                records = list(csv.DictReader(f))
            checkpoints = sorted({int(r['episode']) for r in records})
            means = [np.mean([float(r['return']) for r in records if int(r['episode']) == ep])
                     for ep in checkpoints]
            axes[1].plot(checkpoints, means, marker='o', label=label, color=color)
    axes[0].set_title('Training return (raw and up to 10-episode moving mean)', fontsize=11)
    axes[1].set_title('Deterministic validation return (scenario mean)', fontsize=11)
    for ax in axes:
        ax.set_xlabel('Completed episodes')
        ax.set_ylabel('Undiscounted mission return')
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
