"""Shared helpers for the reference scripts in this folder.

Nothing here changes the repository: the package is imported straight from ``src/`` and the
frozen 40x40 study (``configs/rl-main-frozen.json`` and
``docs/experiments/rl-main-2026-10-11``) is only ever read.

Contents
--------
* ``frozen_setup``        - the frozen environment config, training settings and final worlds.
* ``run_episode``         - one full mission, returning the same columns as the frozen
                            ``episodes.csv`` (same formulas as ``scripts/run_mission_validation.rollout``).
* ``CheckpointPolicy``    - a trained DQN/REINFORCE/A2C checkpoint acting by masked argmax,
                            exactly as in the frozen evaluation.
* ``make_policy``         - one place that builds every controller by name.
* ``run_jobs``            - run many (config, policy, world) missions across CPU cores.
* ``summarise``           - mean / sample SD tables using the frozen study's convention.
"""

import csv
import hashlib
import json
import sys
import time
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))          # use the repo's package without installing it

from wild_drone.baselines import (  # noqa: E402
    GreedyPolicy,
    RandomPolicy,
    RollingHorizonPolicy,
    SweepPolicy,
)
from wild_drone.config import EnvConfig  # noqa: E402
from wild_drone.environment import WildfireFleetEnv  # noqa: E402

FROZEN_CONFIG = ROOT/'configs'/'rl-main-frozen.json'
FROZEN_ARCHIVE = ROOT/'docs'/'experiments'/'rl-main-2026-10-11'
RESULTS = HERE/'results'
CHECKPOINTS = HERE/'checkpoints'            # filled by `gh run download` (see README)

ALGORITHMS = ('dqn', 'reinforce', 'a2c')
TRAINING_SEEDS = (0, 1, 2)

# Column order of the frozen episodes.csv.
EPISODE_COLUMNS = ('policy', 'training_seed', 'scenario_seed', 'grid_size', 'horizon', 'steps',
                   'return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
                   'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts',
                   'decision_ms', 'fleet_failure', 'intact_fleet')
# Metric order of the frozen summary.csv / comparison.csv.
METRICS = ('return', 'weighted_freshness', 'overdue_fraction', 'energy_used', 'total_losses',
           'safe_returns', 'charge_completions', 'collision_attempts', 'steps',
           'fleet_failure', 'intact_fleet', 'decision_ms')


def frozen_setup():
    """Return (EnvConfig, per-algorithm training settings, the 20 final evaluation worlds)."""
    frozen = json.loads(FROZEN_CONFIG.read_text())
    return (EnvConfig(**frozen['environment']), frozen['training'],
            tuple(frozen['evaluation_worlds']))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------------------

class CheckpointPolicy:
    """A trained learner: argmax of its Q-values / logits over the feasible (masked) actions.

    This is the deployment rule of the frozen study (``RLScorePolicy`` in
    ``scripts/run_scale_rl.py``).  Unlike ``MaskedNeuralPolicy`` it does not refuse an
    environment whose config differs from the training config, because the sensitivity
    study deliberately evaluates the unchanged weights under shifted settings.
    """

    def __init__(self, path):
        import torch

        from wild_drone.learning import FleetNetwork
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.algorithm = data['algorithm']
        self.trained_config = EnvConfig(**data['env_config'])
        self.model = FleetNetwork(self.trained_config.grid_size, **data['model_config'])
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()

    def act(self, env, observation):
        import torch

        from wild_drone.learning import tensor_observations
        from wild_drone.masked_learning import action_mask
        with torch.no_grad():
            scores = self.model(*tensor_observations(
                observation, spatial_encoding=self.model.spatial_encoding))[0][0].numpy()
        return int(np.argmax(np.where(action_mask(env), scores, -np.inf)))


def checkpoint_path(algorithm, seed, name='policy.pt'):
    """Where `gh run download` puts the weights of one frozen fit."""
    return CHECKPOINTS/f'{algorithm}-{seed}'/'fits'/algorithm/f'seed-{seed}'/name


def verify_checkpoint(algorithm, seed):
    """Check the downloaded weights are byte-identical to the ones the frozen study audited."""
    record = json.loads((FROZEN_ARCHIVE/f'fit-{algorithm}-{seed}.json').read_text())
    path = checkpoint_path(algorithm, seed)
    return path.is_file() and sha256(path) == record['files']['policy.pt']['sha256']


def make_policy(name, world, training_seed=None, checkpoint=None):
    """Build a controller.  Stochastic controllers are seeded with the world number, which is
    what the frozen evaluation did for random and masked random."""
    if checkpoint is not None:
        return CheckpointPolicy(checkpoint)
    if name in ALGORITHMS:
        return CheckpointPolicy(checkpoint_path(name, training_seed))
    if name == 'greedy':
        return GreedyPolicy()
    if name == 'sweep':
        return SweepPolicy()
    if name == 'random':
        return RandomPolicy(world)
    if name == 'masked_random':
        from wild_drone.masked_learning import MaskedRandomPolicy
        return MaskedRandomPolicy(world)
    if name == 'planner':
        # Default planner: horizon 4, 8 first actions, 2 sampled futures each.
        return RollingHorizonPolicy(world)
    raise ValueError(f'unknown policy: {name}')


# --------------------------------------------------------------------------------------
# One mission
# --------------------------------------------------------------------------------------

def run_episode(config, policy, world, *, name, training_seed=None):
    """Fly one complete mission and return one frozen-format episode row.

    Full-horizon accounting, as in the frozen study: if the whole fleet is lost early the
    missing steps count as zero freshness and fully overdue.
    """
    env = WildfireFleetEnv(config)
    observation, _ = env.reset(seed=world)
    total, freshness, overdue, durations = 0.0, 0.0, 0.0, []
    while True:
        started = time.perf_counter()
        action = policy.act(env, observation)
        durations.append((time.perf_counter()-started)*1000)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        freshness += info['weighted_freshness']
        overdue += info['overdue_fraction']
        if terminated or truncated:
            break
    missing = config.horizon-env.elapsed
    return {'policy': name, 'training_seed': training_seed, 'scenario_seed': world,
            'grid_size': config.grid_size, 'horizon': config.horizon, 'steps': env.elapsed,
            'return': total, 'weighted_freshness': freshness/config.horizon,
            'overdue_fraction': (overdue+missing)/config.horizon,
            'energy_used': env.totals['energy_used'], 'total_losses': int(env.lost.sum()),
            'safe_returns': env.totals['safe_returns'],
            'charge_completions': env.totals['charge_completions'],
            'collision_attempts': env.totals['collision_attempts'],
            'decision_ms': float(np.mean(durations)), 'fleet_failure': int(env.lost.all()),
            'intact_fleet': int(not env.lost.any())}


# --------------------------------------------------------------------------------------
# Many missions in parallel
# --------------------------------------------------------------------------------------

def _worker(job):
    """job = (tag dict, env overrides dict, policy name, training seed, world, checkpoint)."""
    tag, overrides, name, training_seed, world, checkpoint = job
    if name in ALGORITHMS or checkpoint is not None:
        import torch
        torch.set_num_threads(1)
    config = replace(frozen_setup()[0], **overrides)
    policy = make_policy(name, world, training_seed, checkpoint)
    started = time.perf_counter()
    row = run_episode(config, policy, world, name=name, training_seed=training_seed)
    return {**tag, **row, 'wall_seconds': time.perf_counter()-started}


def run_jobs(jobs, workers=1, label=''):
    """Run missions; results come back in job order regardless of which worker finished first."""
    started = time.perf_counter()
    if workers <= 1:
        rows = []
        for i, job in enumerate(jobs, 1):
            rows.append(_worker(job))
            print(f'{label} {i}/{len(jobs)} done ({time.perf_counter()-started:.0f}s)', flush=True)
        return rows
    with get_context('spawn').Pool(workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap(_worker, jobs), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                print(f'{label} {i}/{len(jobs)} done ({time.perf_counter()-started:.0f}s)',
                      flush=True)
    return rows


# --------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------

def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns or rows[0])
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def summarise(rows, metrics=METRICS):
    """Mean and sample SD per metric for one policy's rows.

    Frozen-study convention: a learner is first averaged over worlds within each training
    seed and the SD is across those seed means; a baseline's SD is across worlds.
    """
    seeds = sorted({r['training_seed'] for r in rows if r['training_seed'] not in (None, '')})
    out = {}
    for metric in metrics:
        if seeds:
            values = [np.mean([float(r[metric]) for r in rows if r['training_seed'] == s])
                      for s in seeds]
        else:
            values = [float(r[metric]) for r in rows]
        out[metric] = (float(np.mean(values)),
                       float(np.std(values, ddof=1)) if len(values) > 1 else float('nan'))
    out['unit'] = 'seed_means' if seeds else 'worlds'
    out['units'] = len(seeds) if seeds else len(rows)
    return out
