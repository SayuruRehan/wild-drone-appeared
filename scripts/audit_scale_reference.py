"""Independently replay every controlled reference mission and cloned action reward."""

import argparse
import copy
import csv
import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.masked_learning import action_mask

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
           'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts',
           'steps', 'fleet_failure', 'intact_fleet')
PARENT = 'docs/experiments/scale-pilot-2026-10-11/'


def validate_metric_schema(record, *, paired):
    identity = {'comparison_policy', 'training_seed', 'scenario_seed'} if paired else \
        {'policy', 'training_seed', 'worlds'}
    assert set(record) == identity | set(METRICS), 'incomplete or unexpected metric schema'


def validate_launch(launch):
    expected = {PARENT + name for name in ('launch.json', 'missions/episodes.csv',
                                         'checkpoints.json', 'completion.json')}
    assert set(launch['parent_input_sha256']) == expected, \
        'complete parent input identities required'
    parent = json.loads((ROOT / (PARENT + 'launch.json')).read_text())
    assert launch['env_config'] == parent['env_config'], 'full frozen environment identity required'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def obs_sha(obs):
    return hashlib.sha256(json.dumps({k: v.tolist() for k, v in obs.items()},
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def equal(actual, expected):
    np.testing.assert_allclose(float(actual), float(expected), atol=1e-12, rtol=0)


def audit_world(config, episode, trace, scores, masks):
    assert scores.shape == masks.shape == (len(trace), 125)
    offset, reward_drift = 0, 0.0
    world = int(episode['scenario_seed'])
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=world)
    rows = []
    while not env._done:
        record = trace[offset]
        assert record['policy'] == 'exact_inspection' and record['training_seed'] == ''
        assert int(record['scenario_seed']) == world and int(record['step']) == env.elapsed + 1
        assert record['pre_sha256'] == obs_sha(obs)
        mask = action_mask(env)
        np.testing.assert_array_equal(mask, masks[offset])
        footprints, gains = [], []
        for action in range(125):
            clone = copy.deepcopy(env)
            _, _, _, _, info = clone.step(action)
            footprints.append(clone._inspection_mask())
            gains.append(info['reward_components']['freshness_gain'])
        # Independently sum actual post-transition footprints; also check
        # against every clone's actual reward, whose summation order differs.
        weights = env.priority_weights()
        urgency = weights * np.minimum((env.ages + 1) / config.inspection_deadline, 1)
        reconstructed = (np.stack(footprints) * urgency).sum(axis=(1, 2)) / \
            weights.sum() * config.freshness_reward
        np.testing.assert_array_equal(reconstructed, scores[offset])
        np.testing.assert_allclose(scores[offset], gains, atol=1e-12, rtol=0)
        reward_drift = max(reward_drift, float(np.max(np.abs(scores[offset] - gains))))
        action = int(record['action'])
        assert action == int(np.argmax(np.where(mask, reconstructed, -np.inf)))
        assert mask[action] and record['selected_mask_feasible'] == 'True'
        obs, reward, terminated, truncated, info = env.step(action)
        assert record['post_sha256'] == obs_sha(obs)
        equal(record['reward'], reward)
        for key, value in info['reward_components'].items():
            equal(record[f'reward_{key}'], value)
        for key in ('weighted_freshness', 'overdue_fraction', 'energy_used', 'new_losses',
                    'safe_returns', 'charge_completions', 'collision_attempts'):
            equal(record[key], info[key])
        equal(record['total_losses'], env.lost.sum())
        assert record['terminated'] == str(terminated)
        assert record['truncated'] == str(truncated)
        rows.append(record)
        offset += 1
    missing = config.horizon - len(rows)
    assert episode['policy'] == 'exact_inspection' and episode['training_seed'] == ''
    values = {'grid_size': config.grid_size, 'horizon': config.horizon,
              'steps': len(rows), 'return': sum(float(r['reward']) for r in rows),
              'weighted_freshness': sum(float(r['weighted_freshness']) for r in rows) /
              config.horizon,
              'overdue_fraction': (sum(float(r['overdue_fraction']) for r in rows) + missing) /
              config.horizon,
              **{k: sum(float(r[k]) for r in rows) for k in
                 ('energy_used', 'safe_returns', 'charge_completions', 'collision_attempts')},
              'total_losses': int(env.lost.sum()), 'fleet_failure': int(env.lost.all()),
              'intact_fleet': int(not env.lost.any())}
    for key, value in values.items():
        equal(episode[key], value)
    assert np.isfinite(float(episode['decision_ms'])) and float(episode['decision_ms']) >= 0
    verified = {'policy': 'exact_inspection', 'training_seed': '',
                'scenario_seed': world, **values}
    assert offset == len(trace)
    return verified, reward_drift


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    launch = json.loads((args.run / 'launch.json').read_text())
    validate_launch(launch)
    source = {str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')} | {
        'scripts/run_scale_reference.py', 'scripts/run_mission_validation.py',
        'docs/scale-reference-protocol.md'}
    assert set(launch['source_sha256']) == source, 'complete frozen source identities required'
    for key in ('source_sha256', 'parent_input_sha256'):
        for path, expected in launch[key].items():
            assert sha(ROOT / path) == expected, f'input/source identity differs: {path}'
    assert launch['worlds'] == list(range(7000, 7020))
    assert launch['new_fits'] == launch['rl_updates'] == launch['new_worlds'] == 0
    config = EnvConfig(**launch['env_config'])
    assert config.grid_size == 40 and config.horizon == 300
    episodes = read(args.run / 'episodes.csv')
    assert len(episodes) == 20
    assert [int(r['scenario_seed']) for r in episodes] == launch['worlds']
    with gzip.open(args.run / 'steps.csv.gz', 'rt') as f:
        trace = list(csv.DictReader(f))
    with np.load(args.run / 'scores.npz', allow_pickle=False) as f:
        scores, masks = f['scores'], f['masks']
    assert scores.shape == masks.shape == (len(trace), 125)
    assert scores.dtype == np.float64 and masks.dtype == bool
    assert np.isfinite(scores).all() and masks.any(1).all()
    packets, offset = [], 0
    for episode in episodes:
        count = int(episode['steps'])
        end = offset + count
        packets.append((config, episode, trace[offset:end], scores[offset:end], masks[offset:end]))
        offset = end
    assert offset == len(trace)
    replayed, reward_drift = [], 0.0
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(audit_world, *packet) for packet in packets]
        for future in as_completed(futures):
            verified, drift = future.result()
            replayed.append(verified)
            reward_drift = max(reward_drift, drift)
            print(f"audited world {verified['scenario_seed']} ({len(replayed)}/20)", flush=True)
    replayed.sort(key=lambda r: r['scenario_seed'])
    parent = read(ROOT / 'docs/experiments/scale-pilot-2026-10-11/missions/episodes.csv')
    comparisons = read(args.run / 'comparison.csv')
    keys = [('exact_inspection', ''), ('centered', '0'), ('centered', '1'),
            ('centered', '2'), ('greedy', ''), ('masked_random', '')]
    assert sorted((r['policy'], r['training_seed']) for r in comparisons) == sorted(keys)
    metric_keys = METRICS
    for record in comparisons:
        validate_metric_schema(record, paired=False)
        group = [r for r in replayed + parent if r['policy'] == record['policy'] and
                 r['training_seed'] == record['training_seed']]
        assert len(group) == 20 and sorted(int(r['scenario_seed']) for r in group) == \
            launch['worlds']
        equal(record['worlds'], len(group))
        for key in metric_keys:
            equal(record[key], np.mean([float(r[key]) for r in group]))
    pairs = read(args.run / 'paired.csv')
    expected_pairs = {(name, seed, world) for name, seed in keys if name != 'exact_inspection'
                      for world in launch['worlds']}
    actual_pairs = [(r['comparison_policy'], r['training_seed'], int(r['scenario_seed']))
                    for r in pairs]
    assert len(actual_pairs) == len(expected_pairs) and set(actual_pairs) == expected_pairs
    for record in pairs:
        validate_metric_schema(record, paired=True)
        world = int(record['scenario_seed'])
        exact = next(r for r in replayed if r['scenario_seed'] == world)
        baseline = next(r for r in parent if r['policy'] == record['comparison_policy'] and
                        r['training_seed'] == record['training_seed'] and
                        int(r['scenario_seed']) == world)
        for key in metric_keys:
            equal(record[key], exact[key] - float(baseline[key]))
    completion = json.loads((args.run / 'completion.json').read_text())
    assert completion['episodes'] == 20 and completion['steps'] == len(trace)
    assert completion['cloned_action_transitions'] == 125 * len(trace)
    assert completion['new_fits'] == completion['rl_updates'] == completion['new_worlds'] == 0
    assert np.isfinite(completion['elapsed_seconds']) and completion['elapsed_seconds'] >= 0
    assert len(completion['comparisons']) == len(comparisons)
    for actual, expected in zip(completion['comparisons'], comparisons, strict=True):
        validate_metric_schema(actual, paired=False)
        assert actual['policy'] == expected['policy']
        assert str(actual['training_seed'] or '') == expected['training_seed']
        for key in ['worlds', *metric_keys]:
            equal(actual[key], expected[key])
    result = {'passed': True, 'episodes': len(episodes), 'steps': len(trace),
              'audit_workers': 4,
              'cloned_action_rewards_checked': 125 * len(trace),
              'full_reference_vectors_exact': True, 'feasible_first_argmax_exact': True,
              'max_actual_reward_summation_drift': reward_drift, 'paired_rows': len(pairs),
              'audit_script_sha256': sha(Path(__file__)),
              'input_sha256': {p.name: sha(p) for p in sorted(args.run.iterdir())
                               if p.is_file() and p.resolve() != args.out.resolve()}}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
