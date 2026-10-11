"""Independent, read-only audit and compact publication of the frozen 40x40 pilot."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import make_policy
from wild_drone.centered_inspection import load_centered_model
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import tensor_observations
from wild_drone.masked_learning import MaskedRandomPolicy, action_mask

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, name):
    if not math.isclose(float(actual), float(expected), abs_tol=1e-7, rel_tol=1e-6):
        raise AssertionError(f'{name}: {actual} != {expected}')


def observation_hash(observation):
    text = json.dumps({k: v.tolist() for k, v in observation.items()},
                      sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def numeric_ranking(scores, gains, mask):
    ids = np.flatnonzero(mask)
    chosen = ids[np.argmax(scores[ids])]
    values = gains[ids]
    low, high = float(values.min()), float(values.max())
    span = high - low
    best = np.isclose(values, high, rtol=0, atol=1e-12)
    return {
        'selected_action': int(chosen), 'reference_best_count': int(best.sum()),
        'one_step_optimal': int(chosen in ids[best]),
        'normalised_gain_regret': (high - float(gains[chosen])) / span if span > 1e-12
        else None,
        'random_expected_normalised_regret': (high - float(values.mean())) / span
        if span > 1e-12 else None,
    }


def audit_rankings(run, config, index, data):
    rows = read_csv(run / 'rankings/natural.csv')
    expected = {}
    for seed in range(3):
        for phase in ('initial', 'final'):
            path = run / f'rankings/seed-{seed}-{phase}.npz'
            with np.load(path, allow_pickle=False) as stored:
                scores = {key: stored[key].copy() for key in stored.files}
            model = load_centered_model(run / f'seed-{seed}/{phase}.pt', config, 'centered')
            for split, block in scores.items():
                dataset = data[split]
                with torch.no_grad():
                    predicted = np.concatenate([
                        model(dataset['maps'][i:i + 32],
                              dataset['vectors'][i:i + 32])[0].numpy()
                        for i in range(0, len(dataset['maps']), 32)
                    ])
                np.testing.assert_array_equal(predicted, block)
                expected.update({(split, str(seed), phase, 'network', i): s
                                 for i, s in enumerate(block)})
    action_mean = json.loads((run / 'action-mean.json').read_text())['scores']
    for split, dataset in data.items():
        for i in range(len(dataset['maps'])):
            expected[split, '', 'control', 'action_mean', i] = np.array(action_mean)
            expected[split, '', 'control', 'inspection_reference', i] = (
                dataset['gains'][i].numpy())
    seen = set()
    case_position = {split: {r['case']: i for i, r in enumerate(records)}
                     for split, records in index.items()}
    for row in rows:
        key = (row['split'], row['seed'], row['phase'], row['policy'],
               case_position[row['split']][row['case']])
        if key in seen:
            raise AssertionError('duplicate ranking row')
        seen.add(key)
        split, _, _, _, i = key
        scores = np.asarray(expected[key], dtype=np.float64)
        gains = data[split]['gains'][i].numpy()
        mask = data[split]['masks'][i].numpy()
        raw = numeric_ranking(scores, gains, mask)
        if int(row['selected_action']) != raw['selected_action']:
            raise AssertionError('ranking argmax mismatch')
        if int(row['reference_best_count']) != raw['reference_best_count']:
            raise AssertionError('ranking best-set mismatch')
        if (row['one_step_optimal'].lower() == 'true') != bool(raw['one_step_optimal']):
            raise AssertionError('ranking tied-best mismatch')
        for field in ('normalised_gain_regret', 'random_expected_normalised_regret'):
            if raw[field] is None:
                if row[field] not in ('', 'None'):
                    raise AssertionError('uninformative ranking state not preserved')
            else:
                close(row[field], raw[field], field)
    if seen != set(expected):
        raise AssertionError('incomplete lossless ranking record coverage')
    return len(seen)


def audit_data(run, launch):
    datasets = json.loads((run / 'datasets.json').read_text())
    with gzip.open(run / 'inputs.json.gz', 'rt') as stream:
        index = json.load(stream)
    data = {}
    for split in ('training', 'fresh'):
        path = run / f'{split}.pt'
        if (sha(path) != datasets[split]['sha256'] or
                path.stat().st_size != datasets[split]['bytes']):
            raise AssertionError('dataset bytes changed')
        data[split] = torch.load(path, map_location='cpu', weights_only=True)
        packet = data[split]
        assert len(index[split]) == len(packet['maps']) == datasets[split]['states']
        assert len(set(r['case'] for r in index[split])) == len(index[split])
        for i, record in enumerate(index[split]):
            np.testing.assert_array_equal(record['gains'], packet['gains'][i].numpy())
            np.testing.assert_array_equal(record['mask'], packet['masks'][i].numpy())
        gains = packet['gains'].numpy()
        masks = packet['masks'].numpy()
        low = np.where(masks, gains, np.inf).min(1)
        high = np.where(masks, gains, -np.inf).max(1)
        informative = high - low > 1e-12
        normalised = np.where(masks & informative[:, None],
                              (gains - low[:, None]) /
                              np.where(informative, high - low, 1)[:, None], 0)
        np.testing.assert_array_equal(informative, packet['informative'].numpy())
        np.testing.assert_allclose(normalised, packet['targets'].numpy(), atol=1e-6)
        assert int(informative.sum()) == datasets[split]['informative']
        assert not ({r['scenario_seed'] for r in index[split]} &
                    (set(range(5800, 5820)) | set(range(10000, 10020))))
    assert {r['scenario_seed'] for r in index['training']} == set(launch['training_worlds'])
    assert {r['scenario_seed'] for r in index['fresh']} == set(launch['ranking_worlds'])
    return index, data


def audit_fits(run, launch):
    checkpoints = json.loads((run / 'checkpoints.json').read_text())
    sampling = json.loads((run / 'sampling.json').read_text())
    dataset_sha = sha(run / 'training.pt')
    for seed in range(3):
        record = checkpoints[str(seed)]
        assert record['completed_updates'] == launch['updates_per_fit'] == 2000
        assert record['parameters'] == 434877
        assert record['training_dataset_sha256'] == dataset_sha
        assert record['sample_sequence_sha256'] == sampling[str(seed)]['sample_sequence_sha256']
        assert sampling[str(seed)]['sample_draws'] == 128000
        history = read_csv(run / f'seed-{seed}/training.csv')
        assert len(history) == 2000 and [int(r['update']) for r in history] == list(
            range(1, 2001))
        assert all(np.isfinite(float(r['training_batch_loss'])) for r in history)
        for phase, updates in (('initial', 0), ('final', 2000)):
            path = run / f'seed-{seed}/{phase}.pt'
            expected = record['checkpoints'][f'{phase}.pt']
            assert sha(path) == expected['sha256']
            assert path.stat().st_size == expected['bytes']
            stored = torch.load(path, map_location='cpu', weights_only=True)
            assert stored['seed'] == seed and stored['completed_updates'] == updates
            assert stored['arm'] == 'centered' and stored['env_config'] == launch['env_config']
        before = torch.load(run / f'seed-{seed}/initial.pt', weights_only=True)['state_dict']
        after = torch.load(run / f'seed-{seed}/final.pt', weights_only=True)['state_dict']
        assert any(not torch.equal(before[k], after[k]) for k in before)


def audit_missions(run, config):
    episodes = read_csv(run / 'missions/episodes.csv')
    with gzip.open(run / 'missions/steps.csv.gz', 'rt', newline='') as stream:
        traces = list(csv.DictReader(stream))
    grouped = defaultdict(list)
    for row in traces:
        grouped[row['policy'], row['training_seed'], int(row['scenario_seed'])].append(row)
    assert len(episodes) == 140 and len(grouped) == 140
    models = {str(seed): load_centered_model(
        run / f'seed-{seed}/final.pt', config, 'centered') for seed in range(3)}
    action_mean = np.asarray(json.loads((run / 'action-mean.json').read_text())['scores'])
    stores = {}
    for policy in ('centered', 'action_mean'):
        for seed in (('0', '1', '2') if policy == 'centered' else ('',)):
            filename = f'{policy}-seed-{seed if seed else "None"}-scores.npz'
            with np.load(run / 'missions' / filename, allow_pickle=False) as arrays:
                stores[policy, seed] = (arrays['scores'].copy(), arrays['masks'].copy(), 0)
    for row in episodes:
        key = row['policy'], row['training_seed'], int(row['scenario_seed'])
        steps = grouped.pop(key)
        env = WildfireFleetEnv(config)
        observation, _ = env.reset(seed=key[2])
        rewards, freshness, overdue = [], [], []
        for n, record in enumerate(steps):
            assert int(record['step']) == n + 1
            assert observation_hash(observation) == record['pre_sha256']
            action = int(record['action'])
            mask = action_mask(env)
            assert (record['selected_mask_feasible'].lower() == 'true') == bool(mask[action])
            if key[0] in ('centered', 'action_mean'):
                scores, saved_masks, offset = stores[key[0], key[1]]
                np.testing.assert_array_equal(mask, saved_masks[offset])
                if key[0] == 'centered':
                    with torch.no_grad():
                        predicted = models[key[1]](*tensor_observations(observation))[0][0]
                    np.testing.assert_array_equal(predicted.numpy(), scores[offset])
                else:
                    np.testing.assert_array_equal(action_mean.astype(np.float32),
                                                  scores[offset])
                assert action == int(np.argmax(np.where(mask, scores[offset], -np.inf)))
                stores[key[0], key[1]] = (scores, saved_masks, offset + 1)
            observation, reward, term, trunc, info = env.step(action)
            assert observation_hash(observation) == record['post_sha256']
            close(reward, record['reward'], 'step return')
            assert (str(term) == record['terminated'] and
                    str(trunc) == record['truncated'])
            for field in ('weighted_freshness', 'overdue_fraction', 'energy_used',
                          'new_losses', 'safe_returns', 'charge_completions',
                          'collision_attempts'):
                close(info[field], record[field], 'step ' + field)
            for name, value in info['reward_components'].items():
                close(value, record['reward_' + name], name)
            rewards.append(reward)
            freshness.append(info['weighted_freshness'])
            overdue.append(info['overdue_fraction'])
        assert env._done and int(row['steps']) == env.elapsed
        assert (str(key[2]) in [str(w) for w in range(7000, 7020)])
        close(sum(rewards), row['return'], 'episode return')
        close(sum(freshness) / config.horizon, row['weighted_freshness'], 'freshness')
        close((sum(overdue) + config.horizon - len(steps)) / config.horizon,
              row['overdue_fraction'], 'overdue')
        for name in ('energy_used', 'safe_returns', 'charge_completions',
                     'collision_attempts'):
            close(env.totals[name], row[name], 'episode ' + name)
        assert int(row['total_losses']) == int(env.lost.sum())
        assert int(row['fleet_failure']) == int(env.lost.all())
        assert int(row['intact_fleet']) == int(not env.lost.any())
    assert not grouped
    assert all(offset == len(scores) == len(masks)
               for scores, masks, offset in stores.values())
    return episodes, len(traces)


def audit_natural_simulator_actions(config, index, data):
    count = 0
    for split, records in index.items():
        by_state = {(r['scenario_seed'], r['collector'], r['step']): i
                    for i, r in enumerate(records)}
        worlds = sorted({r['scenario_seed'] for r in records})
        snapshots = sorted({r['step'] for r in records})
        for world in worlds:
            for collector in ('masked_random', 'greedy'):
                env = WildfireFleetEnv(config)
                observation, _ = env.reset(seed=world)
                policy = MaskedRandomPolicy(world) if collector == 'masked_random' else (
                    make_policy('greedy'))
                for t in range(max(snapshots) + 1):
                    key = world, collector, t
                    if key in by_state:
                        i = by_state[key]
                        assert observation_hash(observation) == records[i]['observation_sha256']
                        for action in range(125):
                            # One-step counterfactuals are cloned so neither the real
                            # world nor its policy's RNG is advanced.
                            import copy
                            candidate = copy.deepcopy(env)
                            _, _, _, _, info = candidate.step(action)
                            close(info['reward_components']['freshness_gain'],
                                  data[split]['gains'][i, action], 'all-action gain')
                            count += 1
                    if t == max(snapshots):
                        break
                    observation, _, done, truncated, _ = env.step(policy.act(env, observation))
                    if done or truncated:
                        break
    assert count == sum(len(records) for records in index.values()) * 125
    return count


def publish(run, out, audit):
    out.mkdir(parents=True, exist_ok=False)
    small = ['launch.json', 'datasets.json', 'checkpoints.json', 'sampling.json',
             'missing-snapshots.json', 'action-mean.json', 'completion.json', 'inputs.json.gz']
    for name in small:
        shutil.copy2(run / name, out / name)
    for sub, names in {
        'rankings': ['natural.csv', 'per-fit.csv'],
        'missions': ['episodes.csv', 'per-fit.csv', 'summary.csv', 'steps.csv.gz'],
    }.items():
        (out / sub).mkdir()
        for name in names:
            shutil.copy2(run / sub / name, out / sub / name)
    for path in (run / 'rankings').glob('*.npz'):
        shutil.copy2(path, out / 'rankings' / path.name)
    for path in (run / 'missions').glob('*.npz'):
        shutil.copy2(path, out / 'missions' / path.name)
    for seed in range(3):
        (out / f'seed-{seed}').mkdir()
        shutil.copy2(run / f'seed-{seed}/training.csv', out / f'seed-{seed}/training.csv')
    (out / 'audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    hashes = {str(p.relative_to(out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
              for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'checksums.json').write_text(json.dumps(hashes, indent=2, sort_keys=True) + '\n')
    completion = json.loads((run / 'completion.json').read_text())
    (out / 'README.md').write_text(
        '# 40×40 prospective supervised inspection pilot\n\n'
        'Frozen source: ' + json.loads((run / 'launch.json').read_text())['source_revision']
        + '\n\n- Ranking criterion all seeds: **' + str(completion['ranking_passed'])
        + '**\n- Full-mission criterion all seeds: **' + str(completion['mission_passed'])
        + '**\n- Complete episodes: ' + str(completion['mission_episodes'])
        + '; steps: ' + str(completion['mission_steps'])
        + '\n- Independent audit: ' + str(audit) + '\n\n'
        'This is supervised inspection scoring, **not trained reinforcement learning**. '
        'A failed predeclared criterion is a negative pilot, not an invitation to retune. '
        'Training data and checkpoint binaries are retained in the run artifact, '
        'while the exact binary hashes are tracked here. Confirmation and final '
        'test worlds were not used.\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    launch = json.loads((run / 'launch.json').read_text())
    config = EnvConfig(**launch['env_config'])
    assert (config.grid_size, config.horizon) == (40, 300)
    assert launch['rl_updates'] == 0
    for path, expected in launch['source_sha256'].items():
        assert sha(ROOT / path) == expected, 'frozen source identity changed: ' + path
    audit_fits(run, launch)
    index, data = audit_data(run, launch)
    rank_count = audit_rankings(run, config, index, data)
    print('ranking records audited:', rank_count, flush=True)
    episodes, steps = audit_missions(run, config)
    print('mission episodes/steps replayed:', len(episodes), steps, flush=True)
    counterfactuals = audit_natural_simulator_actions(config, index, data)
    print('independent actual action rewards:', counterfactuals, flush=True)
    completion = json.loads((run / 'completion.json').read_text())
    assert completion['mission_episodes'] == len(episodes) == 140
    assert completion['mission_steps'] == steps
    assert completion['ranking_passed'] == all(x['ranking_passed']
                                               for x in completion['per_seed'])
    assert completion['mission_passed'] == all(x['mission_passed']
                                               for x in completion['per_seed'])
    assert not any(x['scenario_seed'] in range(5800, 5820) for x in episodes)
    audit = {'passed': True, 'ranking_rows': rank_count, 'mission_episodes': len(episodes),
             'mission_steps': steps, 'all_action_rewards_checked': counterfactuals,
             'checkpoint_source_hashes_verified': True, 'simulator_replay': True,
             'full_score_vector_reconstruction': True}
    publish(run, args.out, audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
