"""Replay every mission against real transitions and frozen score decisions."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import GreedyPolicy, SweepPolicy
from wild_drone.centered_inspection import load_centered_model
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import tensor_observations
from wild_drone.masked_learning import MaskedRandomPolicy, action_mask

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_hash(obs):
    payload = {key: value.tolist() for key, value in obs.items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    launch = json.loads((ROOT/'launch.json').read_text())
    config = EnvConfig(**launch['env_config'])
    for path, sha in launch['source_sha256'].items():
        assert digest(REPO/path) == sha, path
    parent = REPO/'docs'/'experiments'/'training-coverage-2026-10-10'
    assert digest(parent/'action-mean-controls.json') == launch['controls_sha256']
    controls = json.loads((parent/'action-mean-controls.json').read_text())
    for item in launch['checkpoints'].values():
        path = args.parent_root/item['relative_path']
        assert path.stat().st_size == item['bytes'] and digest(path) == item['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert saved['completed_updates'] == 2000 and saved['arm'] == 'centered'
        assert saved['env_config'] == config.to_dict()
    with (ROOT/'episodes.csv').open(newline='') as f:
        episodes = list(csv.DictReader(f))
    with gzip.open(ROOT/'steps.csv.gz', 'rt', newline='') as f:
        trace = list(csv.DictReader(f))
    groups = {}
    for row in trace:
        key = row['policy'], row['training_seed'], int(row['scenario_seed'])
        groups.setdefault(key, []).append(row)
    blocks, offsets, models = {}, {}, {}
    for name, seed in dict.fromkeys((r['policy'], r['training_seed']) for r in episodes):
        if name not in ('small', 'expanded') and not name.endswith('_action_mean'):
            continue
        suffix = seed if seed else 'None'
        with np.load(ROOT/f'{name}-seed-{suffix}-scores.npz', allow_pickle=False) as arrays:
            blocks[name, seed] = {key: arrays[key].copy() for key in ('scores', 'masks')}
        block = blocks[name, seed]
        assert block['scores'].dtype == np.float32 and block['masks'].dtype == bool
        assert block['scores'].ndim == 2 and block['scores'].shape[1] == 125
        assert block['masks'].shape == block['scores'].shape
        assert np.isfinite(block['scores']).all() and block['masks'].any(1).all()
        offsets[name, seed] = 0
        if name in ('small', 'expanded'):
            item = launch['checkpoints'][f'{name}/{seed}']
            models[name, seed] = load_centered_model(args.parent_root/item['relative_path'],
                                                    config, 'centered')
    diagnostics, total, network_vectors, control_vectors, outside = [], 0, 0, 0, 0
    for ep in episodes:
        name, seed, world = ep['policy'], ep['training_seed'], int(ep['scenario_seed'])
        env = WildfireFleetEnv(config)
        obs, _ = env.reset(seed=world)
        baseline = {'greedy': GreedyPolicy, 'sweep': SweepPolicy,
                    'masked_random': lambda: MaskedRandomPolicy(world)}.get(name)
        baseline = baseline() if baseline else None
        visited, positions, grounded, unchanged, max_distance = set(), set(), 0, 0, 0
        rows = groups[name, seed, world]
        for row in rows:
            assert observation_hash(obs) == row['pre_sha256']
            mask = action_mask(env)
            action = int(row['action'])
            assert bool(mask[action]) == (row['selected_mask_feasible'] == 'True')
            outside += int(not mask[action])
            if (name, seed) in blocks:
                block, offset = blocks[name, seed], offsets[name, seed]
                assert np.array_equal(mask, block['masks'][offset])
                if name in ('small', 'expanded'):
                    with torch.no_grad():
                        scores = models[name, seed](*tensor_observations(obs))[0][0].numpy()
                    network_vectors += 1
                else:
                    scores = np.asarray(controls[name.split('_')[0]]['scores'], dtype=np.float32)
                    control_vectors += 1
                assert np.array_equal(scores, block['scores'][offset])
                assert action == int(np.where(mask, scores, -np.inf).argmax())
                offsets[name, seed] += 1
            else:
                assert action == baseline.act(env, obs)
            before = env.positions.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            assert observation_hash(obs) == row['post_sha256']
            assert int(row['step']) == env.elapsed
            assert row['terminated'] == str(terminated) and row['truncated'] == str(truncated)
            expected = {'reward': reward, 'weighted_freshness': info['weighted_freshness'],
                        'overdue_fraction': info['overdue_fraction'],
                        'energy_used': info['energy_used'], 'new_losses': info['new_losses'],
                        'total_losses': int(env.lost.sum()), 'safe_returns': info['safe_returns'],
                        'charge_completions': info['charge_completions'],
                        'collision_attempts': info['collision_attempts'],
                        **{f'reward_{k}': v for k, v in info['reward_components'].items()}}
            for key, value in expected.items():
                assert math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-12), key
            total += 1
            visited.update(np.flatnonzero(env._inspection_mask()).tolist())
            positions.add(tuple(env.positions.flatten()))
            grounded += int(np.all(env.positions == env.base))
            unchanged += int(np.array_equal(before, env.positions))
            max_distance = max(max_distance, int(np.abs(env.positions-env.base).sum(1).max()))
        assert env._done and env.elapsed == int(ep['steps'])
        assert int(env.lost.sum()) == int(ep['total_losses'])
        for metric in ('energy_used', 'safe_returns', 'charge_completions', 'collision_attempts'):
            assert math.isclose(env.totals[metric], float(ep[metric]), abs_tol=1e-12)
        diagnostics.append({'policy': name, 'training_seed': seed, 'scenario_seed': world,
                            'unique_joint_positions': len(positions),
                            'fully_grounded_steps': grounded,
                            'unchanged_positions_steps': unchanged,
                            'unique_inspected_cells': len(visited),
                            'max_distance_from_base': max_distance})
        if world == launch['worlds'][-1]:
            print(f'{name}/seed-{seed}: replay PASS', flush=True)
    for key, block in blocks.items():
        assert offsets[key] == len(block['scores'])
    for item in launch['checkpoints'].values():
        assert digest(args.parent_root/item['relative_path']) == item['sha256']
    with (ROOT/'trajectory-diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    result = {'status': 'PASS', 'episodes_replayed': len(episodes), 'steps_replayed': total,
              'source_files_verified': len(launch['source_sha256']),
              'frozen_checkpoints_verified_before_and_after': 6,
              'network_score_vectors_exact': network_vectors,
              'control_score_vectors_exact': control_vectors,
              'all_score_masks_and_selected_argmax_verified': True,
              'all_observation_hashes_and_reward_components_verified': True,
              'native_baseline_actions_outside_action_mask': outside,
              'trajectory_diagnostics': 'Exploratory descriptions derived from replayed '
              'states; not additional tuning or promotion criteria.',
              'decision_timing_exactness': 'Timing excluded from replay equality.'}
    (ROOT/'replay-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
