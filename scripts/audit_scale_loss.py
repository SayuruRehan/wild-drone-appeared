"""Independent replay, numeric recomputation and publication of the loss comparison."""

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from audit_scale_pilot import audit_data, observation_hash

from wild_drone.centered_inspection import load_centered_model
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import tensor_observations
from wild_drone.masked_learning import action_mask

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'docs/experiments/scale-pilot-2026-10-11'
EXACT = ROOT / 'docs/experiments/scale-reference-2026-10-11'
METRICS = ('return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
           'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts',
           'steps', 'fleet_failure', 'intact_fleet', 'decision_ms')
WORLDS = set(range(7000, 7020))
SCORE_ATOL, SCORE_RTOL = 1e-5, 1e-6


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def equal_tree(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), 'schema differs'
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal_tree(a, b)
    elif isinstance(expected, bool):
        assert isinstance(actual, bool) and actual == expected
    elif expected is None or isinstance(expected, str):
        assert actual == expected
    else:
        np.testing.assert_allclose(float(actual), float(expected), atol=1e-12, rtol=0)


def check_scores(predicted, saved, masks):
    assert predicted.shape == saved.shape == masks.shape and masks.dtype == bool
    assert saved.dtype == np.float32 and np.isfinite(saved).all() and masks.any(-1).all()
    np.testing.assert_allclose(predicted, saved, atol=SCORE_ATOL, rtol=SCORE_RTOL)
    np.testing.assert_array_equal(np.where(masks, predicted, -np.inf).argmax(-1),
                                  np.where(masks, saved, -np.inf).argmax(-1))
    return float(np.max(np.abs(predicted.astype(np.float64) - saved.astype(np.float64))))


def audit_world(config, model, episode, trace, scores, masks):
    assert scores.shape == masks.shape == (len(trace), 125)
    assert episode['policy'] == 'ranking' and episode['training_seed'] in ('0', '1', '2')
    world, seed = int(episode['scenario_seed']), episode['training_seed']
    env = WildfireFleetEnv(config)
    obs, _ = env.reset(seed=world)
    actual, drift = [], 0.
    for i, row in enumerate(trace):
        assert not env._done
        assert (row['policy'], row['training_seed'], int(row['scenario_seed'])) == \
            ('ranking', seed, world)
        assert int(row['step']) == i + 1 and row['pre_sha256'] == observation_hash(obs)
        mask = action_mask(env)
        np.testing.assert_array_equal(mask, masks[i])
        with torch.no_grad():
            predicted = model(*tensor_observations(obs))[0][0].numpy()
        drift = max(drift, check_scores(predicted, scores[i], mask))
        action = int(row['action'])
        assert action == int(np.argmax(np.where(mask, predicted, -np.inf)))
        assert mask[action] and row['selected_mask_feasible'] == 'True'
        obs, reward, term, trunc, info = env.step(action)
        assert row['post_sha256'] == observation_hash(obs)
        equal_tree(row['reward'], reward)
        for name, value in info['reward_components'].items():
            equal_tree(row['reward_' + name], value)
        for name in ('weighted_freshness', 'overdue_fraction', 'energy_used', 'new_losses',
                     'safe_returns', 'charge_completions', 'collision_attempts'):
            equal_tree(row[name], info[name])
        equal_tree(row['total_losses'], int(env.lost.sum()))
        assert row['terminated'] == str(term) and row['truncated'] == str(trunc)
        actual.append({'reward': reward, **info})
    assert env._done
    values = {'return': sum(r['reward'] for r in actual), 'steps': len(actual),
              'weighted_freshness': sum(r['weighted_freshness'] for r in actual)/config.horizon,
              'overdue_fraction': (sum(r['overdue_fraction'] for r in actual) +
                                   config.horizon - len(actual))/config.horizon,
              **{k: sum(r[k] for r in actual) for k in
                 ('energy_used', 'safe_returns', 'charge_completions', 'collision_attempts')},
              'total_losses': int(env.lost.sum()), 'fleet_failure': int(env.lost.all()),
              'intact_fleet': int(not env.lost.any())}
    for key, value in values.items():
        equal_tree(episode[key], value)
    equal_tree(episode['grid_size'], config.grid_size)
    equal_tree(episode['horizon'], config.horizon)
    assert np.isfinite(float(episode['decision_ms'])) and float(episode['decision_ms']) >= 0
    return {'policy': 'ranking', 'training_seed': seed, 'scenario_seed': world,
            **values, 'decision_ms': float(episode['decision_ms'])}, drift


def ranking_values(scores, gains, mask, targets):
    ids = np.flatnonzero(mask)
    selected = int(ids[np.argmax(scores[ids])])
    high, low = float(gains[ids].max()), float(gains[ids].min())
    span, regret = high - low, high - float(gains[selected])
    best = ids[np.isclose(gains[ids], high, atol=1e-12, rtol=0)]
    return {'selected_action': selected, 'selected_gain': float(gains[selected]),
            'feasible_actions': len(ids), 'reference_best_count': len(best), 'gain_range': span,
            'gain_regret': regret, 'normalised_gain_regret': regret/span if span > 1e-12 else None,
            'one_step_optimal': selected in best, 'random_best_probability': len(best)/len(ids),
            'random_expected_normalised_regret': (high-float(gains[ids].mean()))/span
            if span > 1e-12 else None,
            # Match float32 arithmetic of the export; this is diagnostic MSE for logits.
            'masked_regression_error': float(np.mean((scores[mask]-targets[mask])**2))}


def audit_rankings(run, config, index, data):
    all_expected, expected_means, drift = {}, [], 0.
    for seed in range(3):
        for phase in ('initial', 'final'):
            model = load_centered_model(run / f'seed-{seed}/{phase}.pt', config, 'centered')
            with np.load(run / f'rankings/seed-{seed}-{phase}.npz', allow_pickle=False) as stored:
                splits = ('fresh',) if phase == 'initial' else ('training', 'fresh')
                assert set(stored.files) == set(splits)
                for split in splits:
                    packet, block = data[split], stored[split]
                    with torch.no_grad():
                        predicted = np.concatenate([model(packet['maps'][i:i+64],
                                                           packet['vectors'][i:i+64])[0].numpy()
                                                    for i in range(0, len(block), 64)])
                    drift = max(drift, check_scores(predicted, block, packet['masks'].numpy()))
                    values = []
                    for i, score in enumerate(block):
                        numeric = ranking_values(score, packet['gains'][i].numpy(),
                                                 packet['masks'][i].numpy(),
                                                 packet['targets'][i].numpy())
                        key = (split, str(seed), phase, index[split][i]['case'])
                        metadata = {k: v for k, v in index[split][i].items()
                                    if k not in ('gains', 'mask')}
                        all_expected[key] = {**metadata, 'split': split, 'seed': str(seed),
                                             'phase': phase, 'policy': 'network', **numeric}
                        values.append(numeric)
                    informative = [r for r in values if r['normalised_gain_regret'] is not None]
                    fields = ('normalised_gain_regret', 'one_step_optimal',
                              'masked_regression_error', 'random_expected_normalised_regret',
                              'random_best_probability')
                    expected_means.append({'split': split, 'seed': str(seed), 'phase': phase,
                                           'policy': 'network', 'states': len(values),
                                           'informative_states': len(informative),
                                           **{k: float(np.mean([r[k] for r in informative]))
                                              for k in fields}})
    rows = read(run / 'rankings/natural.csv')
    assert len(rows) == len(all_expected)
    for row in rows:
        key = (row['split'], row['seed'], row['phase'], row['case'])
        expected = all_expected.pop(key)
        assert set(row) == set(expected)
        for field, value in expected.items():
            if value is None:
                assert row[field] == ''
            elif field == 'one_step_optimal':
                assert row[field] == str(bool(value))
            else:
                equal_tree(row[field], value)
    assert not all_expected
    means = read(run / 'rankings/per-fit.csv')
    assert len(means) == len(expected_means)
    for row, expected in zip(means, expected_means, strict=True):
        equal_tree(row, expected)
    return expected_means, len(rows), drift


def audit_fits(run, parent_run, launch, data):
    identities = json.loads((run / 'checkpoints.json').read_text())
    parent = json.loads((PARENT / 'checkpoints.json').read_text())
    stored_sampling = json.loads((run / 'sampling.json').read_text())
    parent_sampling = json.loads((PARENT / 'sampling.json').read_text())
    ids = torch.where(data['training']['informative'])[0]
    for seed in range(3):
        key, generator = str(seed), torch.Generator().manual_seed(seed+1000)
        counts = torch.zeros(len(data['training']['maps']), dtype=torch.int64)
        sequence = hashlib.sha256()
        for _ in range(2000):
            batch = ids[torch.randint(len(ids), (64,), generator=generator)]
            sequence.update(batch.numpy().astype('<i8').tobytes())
            counts += torch.bincount(batch, minlength=len(counts))
        expected_sampling = {'sample_sequence_sha256': sequence.hexdigest(),
                             'per_state_counts': counts.tolist(), 'sample_draws': 128000,
                             'informative_states': len(ids),
                             'sampled_distinct_states': int((counts > 0).sum()),
                             'draws_per_informative_state': 128000/len(ids)}
        equal_tree(stored_sampling[key], expected_sampling)
        equal_tree(stored_sampling[key], parent_sampling[key])
        assert sequence.hexdigest() == parent[key]['sample_sequence_sha256']
        history = read(run / f'seed-{seed}/training.csv')
        assert len(history) == 2000 and [int(r['update']) for r in history] == list(range(1, 2001))
        assert all(np.isfinite(float(r['training_batch_loss'])) for r in history)
        record = identities[key]
        assert record['parameters'] == 434877 and record['completed_updates'] == 2000
        assert record['method'] == 'supervised-tied-best-set-probability'
        assert record['seed'] == seed and record['initial_tensors_match_parent'] is True
        assert record['training_dataset_sha256'] == sha(parent_run / 'training.pt')
        assert record['sample_sequence_sha256'] == sequence.hexdigest()
        states = {}
        for phase, updates in [('initial', 0), ('final', 2000)]:
            path = run / f'seed-{seed}/{phase}.pt'
            assert sha(path) == record['checkpoints'][path.name]['sha256']
            assert path.stat().st_size == record['checkpoints'][path.name]['bytes']
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert checkpoint['method'] == record['method'] and checkpoint['seed'] == seed
            assert checkpoint['env_config'] == launch['env_config']
            assert checkpoint['completed_updates'] == updates and checkpoint['arm'] == 'centered'
            states[phase] = checkpoint['state_dict']
        old = torch.load(parent_run / f'seed-{seed}/initial.pt', map_location='cpu',
                         weights_only=True)['state_dict']
        assert set(states['initial']) == set(old)
        assert all(torch.equal(v, old[k]) for k, v in states['initial'].items())
        assert any(not torch.equal(v, states['final'][k]) for k, v in states['initial'].items())


def mean(group, key):
    return float(np.mean([float(r[key]) for r in group]))


def audit_summaries(run, replayed, rank_means):
    parent = [r for r in read(PARENT / 'missions/episodes.csv')
              if r['policy'] in ('centered', 'greedy', 'masked_random')]
    parent += read(EXACT / 'episodes.csv')
    all_rows = replayed + parent
    groups = {}
    for row in all_rows:
        groups.setdefault((row['policy'], str(row['training_seed'])), []).append(row)
    keys = [('ranking', str(s)) for s in range(3)] + [('centered', str(s)) for s in range(3)] + \
        [(p, '') for p in ('greedy', 'masked_random', 'exact_inspection')]
    assert set(groups) == set(keys)
    for group in groups.values():
        assert len(group) == 20 and {int(r['scenario_seed']) for r in group} == WORLDS
    comparison = read(run / 'comparison.csv')
    assert len(comparison) == len(keys)
    seen = set()
    for row in comparison:
        key = row['policy'], row['training_seed']
        assert key not in seen
        seen.add(key)
        equal_tree(row, {'policy': key[0], 'training_seed': key[1], 'worlds': 20,
                         **{k: mean(groups[key], k) for k in METRICS}})
    assert seen == set(keys)
    aggregate = read(run / 'summary.csv')
    assert len(aggregate) == 5*len(METRICS)
    seen = set()
    for row in aggregate:
        policy, metric = row['policy'], row['metric']
        assert (policy, metric) not in seen
        seen.add((policy, metric))
        network = policy in ('ranking', 'centered')
        values = [mean(groups[policy, str(s)], metric) for s in range(3)] if network else \
            [float(r[metric]) for r in groups[policy, '']]
        equal_tree(row, {'policy': policy, 'metric': metric,
                         'unit': 'fitted_seed_means' if network else 'worlds',
                         'units': len(values), 'mean': float(np.mean(values)),
                         'sample_sd': float(np.std(values, ddof=1))})
    assert seen == {(p, k) for p in ('ranking', 'centered', 'greedy', 'masked_random',
                                    'exact_inspection') for k in METRICS}
    pairs = read(run / 'paired.csv')
    assert len(pairs) == 240
    seen = set()
    for row in pairs:
        seed, world, policy = row['training_seed'], int(row['scenario_seed']), \
            row['comparison_policy']
        key = seed, world, policy
        assert key not in seen
        seen.add(key)
        new = next(r for r in groups['ranking', seed] if int(r['scenario_seed']) == world)
        old = next(r for r in groups[policy, seed if policy == 'centered' else '']
                   if int(r['scenario_seed']) == world)
        equal_tree(row, {'training_seed': seed, 'scenario_seed': world,
                         'comparison_policy': policy,
                         **{k: float(new[k])-float(old[k]) for k in METRICS}})
    assert seen == {(str(s), w, p) for s in range(3) for w in WORLDS
                    for p in ('centered', 'greedy', 'masked_random', 'exact_inspection')}
    parent_ranks = read(PARENT / 'rankings/per-fit.csv')
    criteria = []
    for seed in range(3):
        def ranks(split):
            return next(r for r in rank_means if r['seed'] == str(seed) and
                        r['phase'] == 'final' and r['split'] == split)

        fresh, training = ranks('fresh'), ranks('training')
        old_rank = next(r for r in parent_ranks if r['seed'] == str(seed) and
                        r['phase'] == 'final' and r['split'] == 'fresh' and
                        r['policy'] == 'network')
        fields = ('weighted_freshness', 'overdue_fraction', 'total_losses',
                  'fleet_failure', 'return')
        values = {p: {k: mean(groups[p, str(seed) if p in ('ranking', 'centered') else ''], k)
                       for k in fields} for p in ('ranking', 'centered', 'greedy', 'masked_random')}
        new, old, greedy, random = (values[p] for p in
                                    ('ranking', 'centered', 'greedy', 'masked_random'))
        rank_checks = {'training_regret_le_005': training['normalised_gain_regret'] <= .05,
                       'fresh_regret_le_020': fresh['normalised_gain_regret'] <= .20,
                       'fresh_best_match_ge_025': fresh['one_step_optimal'] >= .25}
        mission = {'freshness_ge_90pct_greedy': new['weighted_freshness'] >=
                   .9*greedy['weighted_freshness'],
                   'overdue_le_greedy_plus_005': new['overdue_fraction'] <=
                   greedy['overdue_fraction']+.05,
                   'losses_le_greedy_plus_01': new['total_losses'] <= greedy['total_losses']+.1,
                   'fleet_failures_le_greedy': new['fleet_failure'] <= greedy['fleet_failure'],
                   'freshness_ge_random_plus_001': new['weighted_freshness'] >=
                   random['weighted_freshness']+.01,
                   'return_ge_random_plus_05': new['return'] >= random['return']+.5}
        effect = {'fresh_regret_not_higher': fresh['normalised_gain_regret'] <=
                  float(old_rank['normalised_gain_regret']),
                  'freshness_improves_ge_001': new['weighted_freshness'] >=
                  old['weighted_freshness']+.01,
                  **{k+'_not_higher': new[k] <= old[k] for k in
                     ('overdue_fraction', 'total_losses', 'fleet_failure')}}
        criteria.append({'seed': seed, 'ranking_checks': rank_checks,
                          'ranking_passed': all(rank_checks.values()), 'mission_checks': mission,
                          'mission_passed': all(mission.values()),
                          'mission_means': {p: values[p] for p in
                                            ('ranking', 'greedy', 'masked_random')},
                          'effect_checks': effect, 'effect_passed': all(effect.values()),
                          'objective_means': {p: {k: values[p][k] for k in fields if k != 'return'}
                                              for p in ('ranking', 'centered')}})
    completion = json.loads((run / 'completion.json').read_text())
    elapsed = completion['elapsed_seconds']
    assert np.isfinite(elapsed) and elapsed >= 0
    equal_tree(completion, {'new_fits': 3, 'new_worlds': 0, 'rl_updates': 0,
                            'new_mission_episodes': 60,
                            'new_mission_steps': sum(int(r['steps']) for r in replayed),
                            'effect_passed': all(r['effect_passed'] for r in criteria),
                            'ranking_passed': all(r['ranking_passed'] for r in criteria),
                            'mission_passed': all(r['mission_passed'] for r in criteria),
                            'per_seed': criteria, 'elapsed_seconds': elapsed})
    return completion, comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    launch = json.loads((args.run / 'launch.json').read_text())
    parent_launch = json.loads((PARENT / 'launch.json').read_text())
    assert parent_launch['source_revision'] == '7c9562aa6064f46d96cc9a163de3deb2ecaa3e19'
    assert launch['env_config'] == parent_launch['env_config']
    assert (launch['training_seeds'], launch['updates_per_fit'], launch['batch_size'],
            launch['learning_rate'], launch['clip'], launch['cpu_threads'], launch['temperature'],
            launch['new_fits'], launch['new_worlds'], launch['rl_updates']) == \
        ([0, 1, 2], 2000, 64, .001, .5, 1, 1, 3, 0, 0)
    assert launch['method'] == 'supervised-tied-best-set-probability'
    assert launch['mission_worlds'] == list(range(7000, 7020))
    expected_source = {str(p.relative_to(ROOT)) for p in (ROOT / 'src').rglob('*.py')} | \
        {'scripts/'+n for n in ('run_scale_loss.py', 'audit_scale_loss.py', 'run_scale_pilot.py',
                               'run_centered_inspection.py', 'run_mission_validation.py',
                               'audit_scale_pilot.py')} | {'docs/scale-loss-protocol.md',
                               'tests/test_scale_loss.py', '.github/workflows/scale-loss.yml'}
    expected_parent = {str((PARENT / n).relative_to(ROOT)) for n in
                       ('launch.json', 'checkpoints.json', 'datasets.json', 'sampling.json',
                        'rankings/per-fit.csv', 'missions/episodes.csv', 'inputs.json.gz',
                        'completion.json', 'audit.json', 'checksums.json')} | \
        {str((EXACT / n).relative_to(ROOT)) for n in
         ('episodes.csv', 'audit.json', 'launch.json', 'checksums.json')}
    assert set(launch['source_sha256']) == expected_source
    assert set(launch['parent_input_sha256']) == expected_parent
    for key in ('source_sha256', 'parent_input_sha256'):
        for path, expected in launch[key].items():
            assert sha(ROOT/path) == expected, path
    for path, expected in parent_launch['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    checks = json.loads((PARENT / 'checksums.json').read_text())
    for name in ('launch.json', 'checkpoints.json', 'datasets.json', 'sampling.json',
                 'inputs.json.gz', 'rankings/per-fit.csv', 'missions/episodes.csv',
                 'completion.json', 'audit.json'):
        assert sha(PARENT/name) == checks[name]['sha256']
        if name != 'audit.json':
            assert sha(args.parent_root/name) == sha(PARENT/name)
    exact_checks = json.loads((EXACT/'checksums.json').read_text())
    for name in ('episodes.csv', 'audit.json', 'launch.json'):
        assert sha(EXACT/name) == exact_checks[name]['sha256']
    assert json.loads((EXACT/'audit.json').read_text())['passed']
    assert json.loads((PARENT/'audit.json').read_text())['passed']
    index, data = audit_data(args.parent_root, parent_launch)
    parent_checkpoints = json.loads((PARENT/'checkpoints.json').read_text())
    for seed in range(3):
        for phase in ('initial', 'final'):
            path = args.parent_root/f'seed-{seed}/{phase}.pt'
            assert sha(path) == parent_checkpoints[str(seed)]['checkpoints'][path.name]['sha256']
    audit_fits(args.run, args.parent_root, launch, data)
    config = EnvConfig(**launch['env_config'])
    rank_means, rank_rows, rank_drift = audit_rankings(args.run, config, index, data)
    episodes = read(args.run/'missions/episodes.csv')
    assert len(episodes) == 60
    assert {(r['training_seed'], int(r['scenario_seed'])) for r in episodes} == \
        {(str(s), w) for s in range(3) for w in WORLDS}
    with gzip.open(args.run/'missions/steps.csv.gz', 'rt', newline='') as stream:
        traces = list(csv.DictReader(stream))
    grouped = {}
    for row in traces:
        grouped.setdefault((row['training_seed'], int(row['scenario_seed'])), []).append(row)
    assert len(grouped) == 60
    stores = {}
    for seed in range(3):
        with np.load(args.run/f'missions/ranking-seed-{seed}-scores.npz',
                     allow_pickle=False) as stored:
            assert set(stored.files) == {'scores', 'masks'}
            stores[str(seed)] = [stored['scores'], stored['masks'], 0]
    replayed, mission_drift = [], 0.
    models = {str(s): load_centered_model(args.run/f'seed-{s}/final.pt', config, 'centered')
              for s in range(3)}
    for row in episodes:
        seed, world = row['training_seed'], int(row['scenario_seed'])
        trace = grouped.pop((seed, world))
        scores, masks, offset = stores[seed]
        end = offset + len(trace)
        verified, drift = audit_world(config, models[seed], row, trace,
                                      scores[offset:end], masks[offset:end])
        stores[seed][2] = end
        replayed.append(verified)
        mission_drift = max(mission_drift, drift)
        print(f'audited ranking seed {seed} world {world} ({len(replayed)}/60)', flush=True)
    assert not grouped and sum(int(r['steps']) for r in replayed) == len(traces)
    for scores, masks, offset in stores.values():
        assert offset == len(scores) == len(masks)
    completion, comparison = audit_summaries(args.run, replayed, rank_means)
    args.out.mkdir(parents=True, exist_ok=False)
    for path in args.run.iterdir():
        if path.is_dir():
            shutil.copytree(path, args.out/path.name, ignore=shutil.ignore_patterns('*.pt'))
        elif path.suffix != '.pt':
            shutil.copy2(path, args.out/path.name)
    audit = {'passed': True, 'ranking_rows': rank_rows, 'new_mission_episodes': 60,
             'new_mission_steps': len(traces), 'paired_rows': 240,
             'score_atol': SCORE_ATOL, 'score_rtol': SCORE_RTOL,
             'ranking_max_score_drift': rank_drift, 'mission_max_score_drift': mission_drift,
             'every_feasible_argmax_exact': True, 'parent_inputs_unchanged': True,
             'initial_tensors_and_sampling_match_control': True,
             'audit_script_sha256': sha(Path(__file__)),
             'workflow_run_id': os.environ.get('GITHUB_RUN_ID'),
             'source_revision': launch['source_revision']}
    (args.out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    def avg(policy, key):
        return mean([r for r in comparison if r['policy'] == policy], key)

    table = '\n'.join(f"| {p} | {avg(p, 'weighted_freshness'):.6f} | "
                       f"{avg(p, 'overdue_fraction'):.6f} | {avg(p, 'return'):.6f} |"
                       for p in ('ranking', 'centered', 'greedy', 'masked_random',
                                 'exact_inspection'))
    report = f'''# Frozen loss-only 40×40 comparison

Source: `{launch['source_revision']}`. Three new supervised fits, zero RL updates
and zero new worlds. All control fits/data are from the audited original pilot.
The original failed regression pilot and exact-reference diagnostic remain intact.

| Policy | Freshness ↑ | Overdue fraction ↓ | Return ↑ |
|---|---:|---:|---:|
{table}

Every-seed objective-effect decision: **{completion['effect_passed']}**.
Original every-seed ranking capability: **{completion['ranking_passed']}**.
Original every-seed mission capability: **{completion['mission_passed']}**.
See `completion.json` for all separate checks; no best seed is selected.

The only fitting intervention is a tied-best-set log-probability loss, replacing
masked gain MSE. Architecture, initial tensors, exact sample-index sequence,
training packets, optimizer, clipping and 2,000-update budget match the control.
All 60 complete development missions are evaluated under the same feasible mask
and first-index argmax. Greedy's different native candidate/movement rule remains
contextual. Confirmation and final worlds remain unused. Logits are not calibrated
gains; their MSE is a diagnostic export, not comparable calibration evidence.
These descriptive seed/world means establish no significance or unique causal
mechanism. This is not trained DQN, REINFORCE or A2C or assignment completion.

Independent replay passed {rank_rows} ranking rows, all {len(traces)} new mission
steps and 240 paired comparisons. Every action matches exactly; checkpoint/data/
source bytes, initial tensors and sampling counts/digests were verified. Fixed
float32 tolerance: atol={SCORE_ATOL}, rtol={SCORE_RTOL}; gates/hashes/actions stay strict.
`comparison.csv` has every seed mean; `paired.csv` has every matched-world delta.
`summary.csv` reports means/sample SD across fitted seed means separately from
baseline world SD; repeated seed/world rows are not independent replicates.
`rankings/` and `missions/` preserve full compressed scores and action/reward/hash traces.

Full raw checkpoints are retained in Actions artifact
`frozen-40x40-loss-{launch['source_revision']}`; original training/control inputs
are in `frozen-40x40-pilot-7c9562aa6064f46d96cc9a163de3deb2ecaa3e19`
from run 38111928600. Artifact retention is 90 days. Download both artifacts and
extract their `runs/` folders into the repository, then replay to a fresh output:

```bash
python -m pip install -e '.[rl]'
python scripts/audit_scale_loss.py --run runs/scale-loss-2026-10-11 \\
  --parent-root runs/scale-pilot-2026-10-11 --out runs/replayed-scale-loss
```

The prospective recipe is in `docs/scale-loss-protocol.md`. A negative result
stops this objective direction without extra fits. A positive result supports
designing actual RL comparison and separately reserved evaluation, not RL success.
'''
    (args.out/'README.md').write_text(report)
    identities = {str(p.relative_to(args.out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                  for p in sorted(args.out.rglob('*')) if p.is_file()}
    (args.out/'checksums.json').write_text(json.dumps(identities, indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
