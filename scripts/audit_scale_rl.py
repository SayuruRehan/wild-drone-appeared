"""Independent budget/identity checks, complete mission replay and numeric recomputation."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from audit_scale_pilot import observation_hash

from wild_drone.baselines import GreedyPolicy, RandomPolicy, SweepPolicy
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.learning import FleetNetwork, TrainConfig, tensor_observations
from wild_drone.masked_learning import (
    MASK_RULE,
    MaskedNeuralPolicy,
    MaskedRandomPolicy,
    action_mask,
    load_masked_policy,
)

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ('dqn', 'reinforce', 'a2c')
BASELINES = ('greedy', 'sweep', 'random', 'masked_random')
SEEDS = (0, 1, 2)
WORLDS = tuple(range(22000, 22020))
METRICS = ('return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
           'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts',
           'steps', 'fleet_failure', 'intact_fleet', 'decision_ms')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def equal_tree(actual, expected):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected)
        for key in expected:
            equal_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            equal_tree(a, b)
    elif isinstance(expected, bool):
        assert isinstance(actual, bool) and actual == expected
    elif expected is None or isinstance(expected, str):
        assert actual == expected
    else:
        assert not isinstance(actual, bool)
        assert math.isfinite(float(actual)) and math.isfinite(float(expected))
        assert abs(float(actual)-float(expected)) <= 1e-12


def verify_fit(directory, algorithm, seed, config, settings, source_revision):
    saved = torch.load(directory/'policy.pt', weights_only=True, map_location='cpu')
    summary = json.loads((directory/'training_summary.json').read_text())
    metadata = json.loads((directory/'metadata.json').read_text())
    rows = read(directory/'training.csv')
    model_config = {'critic': algorithm != 'dqn', 'spatial_encoding': settings.spatial_encoding}
    for packet in (saved, summary, metadata):
        assert packet['algorithm'] == algorithm and packet['action_mask'] == MASK_RULE
        assert packet['env_config'] == config.to_dict()
        assert packet['train_config'] == settings.to_dict()
        assert packet['learning_signal'] == 'original-reward'
    for packet in (saved, summary):
        assert packet['completed_episodes'] == settings.episodes
        assert packet['model_config'] == model_config
        expected_rule = 'masked-standard-dqn' if algorithm == 'dqn' else None
        assert packet['dqn_target_rule'] == expected_rule
    if source_revision is not None:
        assert saved['git_revision'] == metadata['git_revision'] == source_revision
        assert metadata['git_dirty'] is False
    assert len(rows) == settings.episodes and not (directory/'best_policy.pt').exists()
    steps, updates = 0, 0
    for episode, row in enumerate(rows, 1):
        assert int(row['episode']) == episode and int(row['training_seed']) == seed
        assert int(row['scenario_seed']) == settings.scenario_seed_start+episode-1
        length = int(row['steps'])
        assert 0 < length <= config.horizon
        steps += length
        if algorithm == 'dqn':
            updates = max(0, steps-max(settings.warmup_steps, settings.batch_size)+1)
        elif algorithm == 'reinforce':
            updates = episode
        else:
            updates += math.ceil(length/settings.n_steps)
        assert int(row['updates']) == updates
        assert float(row['shaping_return']) == float(row['discounted_shaping_return']) == 0
        equal_tree(row['return'], float(row['learning_return']))
        for key in ('return', 'weighted_freshness', 'overdue_fraction'):
            assert math.isfinite(float(row[key]))
        if updates:
            assert math.isfinite(float(row['loss']))
    assert summary['steps'] == steps and summary['updates'] == updates and updates > 0
    assert summary['best_episode'] is None and summary['best_validation_return'] is None
    assert math.isfinite(summary['elapsed_seconds']) and summary['elapsed_seconds'] > 0
    for value in saved['state_dict'].values():
        assert torch.isfinite(value).all()
    load_masked_policy(directory/'policy.pt')
    initial_path = directory/'initial.pt'
    if source_revision is not None:
        assert initial_path.is_file()
        initial = torch.load(initial_path, weights_only=True, map_location='cpu')
        assert initial['completed_episodes'] == 0 and initial['algorithm'] == algorithm
        assert initial['train_config'] == settings.to_dict()
        assert initial['env_config'] == config.to_dict()
        assert initial['git_revision'] == source_revision
        assert initial['model_config'] == model_config
        torch.manual_seed(seed)
        reconstructed = FleetNetwork(config.grid_size, **model_config).state_dict()
        assert set(initial['state_dict']) == set(reconstructed) == set(saved['state_dict'])
        assert all(torch.equal(v, initial['state_dict'][k]) for k, v in reconstructed.items())
        assert any(not torch.equal(v, saved['state_dict'][k]) for k, v in reconstructed.items())
    return {'algorithm': algorithm, 'seed': seed, 'episodes': len(rows), 'steps': steps,
            'updates': updates, 'training_seconds': summary['elapsed_seconds']}


def replay_world(config, policy, episode, trace, vectors):
    env = WildfireFleetEnv(config)
    world = int(episode['scenario_seed'])
    obs, _ = env.reset(seed=world)
    neural = isinstance(policy, MaskedNeuralPolicy)
    if neural:
        assert set(vectors) == {'scores', 'masks'}
        assert vectors['scores'].shape == vectors['masks'].shape == (len(trace), 125)
        assert vectors['scores'].dtype == np.float32 and vectors['masks'].dtype == bool
        assert np.isfinite(vectors['scores']).all()
    else:
        assert vectors is None
    metrics, drift = [], 0.
    for i, row in enumerate(trace):
        assert not env._done and int(row['step']) == i+1
        assert (row['policy'], row['training_seed'], int(row['scenario_seed'])) == \
            (episode['policy'], episode['training_seed'], world)
        assert observation_hash(obs) == row['pre_sha256']
        mask = action_mask(env)
        expected_action = policy.act(env, obs)
        if neural:
            with torch.no_grad():
                predicted = policy.model(*tensor_observations(
                    obs, spatial_encoding=policy.model.spatial_encoding))[0][0].numpy()
            np.testing.assert_array_equal(mask, vectors['masks'][i])
            scores = vectors['scores'][i]
            np.testing.assert_allclose(predicted, scores, atol=1e-5, rtol=1e-6)
            drift = max(drift, float(np.max(np.abs(predicted.astype(float)-scores.astype(float)))))
            assert expected_action == int(np.argmax(np.where(mask, scores, -np.inf)))
        action = int(row['action'])
        assert action == expected_action
        assert row['selected_mask_feasible'] == str(bool(mask[action]))
        if neural or episode['policy'] == 'masked_random':
            assert mask[action]
        obs, reward, term, trunc, info = env.step(action)
        assert observation_hash(obs) == row['post_sha256']
        assert row['terminated'] == str(term) and row['truncated'] == str(trunc)
        equal_tree(row['reward'], reward)
        for key, value in info['reward_components'].items():
            equal_tree(row['reward_'+key], value)
        for key in ('weighted_freshness', 'overdue_fraction', 'energy_used', 'new_losses',
                    'safe_returns', 'charge_completions', 'collision_attempts'):
            equal_tree(row[key], info[key])
        equal_tree(row['total_losses'], int(env.lost.sum()))
        metrics.append({'reward': reward, **info})
    assert env._done and len(metrics) == int(episode['steps'])
    values = {'policy': episode['policy'], 'training_seed': episode['training_seed'],
              'scenario_seed': str(world), 'grid_size': config.grid_size, 'horizon': config.horizon,
              'return': sum(r['reward'] for r in metrics), 'steps': len(metrics),
              'weighted_freshness': sum(r['weighted_freshness'] for r in metrics)/config.horizon,
              'overdue_fraction': (sum(r['overdue_fraction'] for r in metrics)+
                                   config.horizon-len(metrics))/config.horizon,
              **{k: sum(r[k] for r in metrics) for k in
                 ('energy_used', 'safe_returns', 'charge_completions', 'collision_attempts')},
              'total_losses': int(env.lost.sum()), 'fleet_failure': int(env.lost.all()),
              'intact_fleet': int(not env.lost.any()), 'decision_ms': float(episode['decision_ms'])}
    for key, value in values.items():
        equal_tree(episode[key], value)
    assert math.isfinite(values['decision_ms']) and values['decision_ms'] >= 0
    return values, drift


def independent_tables(rows, worlds=WORLDS):
    numeric = [{**r, 'training_seed': int(r['training_seed']) if r['training_seed'] != '' else None,
                'scenario_seed': int(r['scenario_seed']),
                **{m: float(r[m]) for m in METRICS}} for r in rows]
    keys = [(a, s) for a in ALGORITHMS for s in SEEDS] + [(b, None) for b in BASELINES]
    assert set((r['policy'], r['training_seed']) for r in numeric) == set(keys)
    comparison = []
    for policy, seed in keys:
        selected = [r for r in numeric if r['policy'] == policy and r['training_seed'] == seed]
        assert len(selected) == len(worlds)
        assert {r['scenario_seed'] for r in selected} == set(worlds)
        comparison.append({'policy': policy, 'training_seed': seed, 'worlds': len(worlds),
                           **{m: sum(r[m] for r in selected)/len(selected) for m in METRICS}})
    summary = []
    for policy in ALGORITHMS+BASELINES:
        units = [r for r in (comparison if policy in ALGORITHMS else numeric)
                 if r['policy'] == policy]
        for metric in METRICS:
            values = [r[metric] for r in units]
            avg = sum(values)/len(values)
            sd = math.sqrt(sum((x-avg)**2 for x in values)/(len(values)-1))
            summary.append({'policy': policy, 'metric': metric,
                            'unit': 'seed_means' if policy in ALGORITHMS else 'worlds',
                            'units': len(values), 'mean': avg, 'sample_sd': sd})
    paired = []
    for row in numeric:
        if row['policy'] in ALGORITHMS:
            for baseline in BASELINES:
                old = next(r for r in numeric if r['policy'] == baseline and
                           r['scenario_seed'] == row['scenario_seed'])
                paired.append({'policy': row['policy'], 'training_seed': row['training_seed'],
                               'scenario_seed': row['scenario_seed'], 'baseline': baseline,
                               **{m+'_delta': row[m]-old[m] for m in METRICS}})
    masked = next(r for r in comparison if r['policy']=='masked_random')
    greedy = next(r for r in comparison if r['policy']=='greedy')
    decisions = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            row = next(r for r in comparison if r['policy']==algorithm and r['training_seed']==seed)
            checks = {'freshness_ge_masked_random_plus_001':
                      row['weighted_freshness'] >= masked['weighted_freshness']+.01,
                      'return_ge_masked_random_plus_05': row['return'] >= masked['return']+.5,
                      'losses_le_masked_random': row['total_losses'] <= masked['total_losses'],
                      'failures_le_masked_random': row['fleet_failure'] <= masked['fleet_failure']}
            gc = {'freshness_ge_90pct_greedy': row['weighted_freshness'] >=
                  greedy['weighted_freshness']*.9,
                  'overdue_le_greedy_plus_005': row['overdue_fraction'] <=
                  greedy['overdue_fraction']+.05,
                  'losses_le_greedy_plus_01': row['total_losses'] <= greedy['total_losses']+.1,
                  'failures_le_greedy': row['fleet_failure'] <= greedy['fleet_failure']}
            decisions.append({'algorithm': algorithm, 'seed': seed, 'masked_random_checks': checks,
                              'greedy_checks': gc, 'learning_gate_passed': all(checks.values()),
                              'greedy_gate_passed': all(gc.values())})
    return comparison, summary, paired, decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    launch = json.loads((args.run/'launch.json').read_text())
    frozen = json.loads((ROOT/'configs/rl-main-frozen.json').read_text())
    config = EnvConfig(**frozen['environment'])
    assert launch['env_config'] == config.to_dict() and config.grid_size == 40
    assert launch['training'] == frozen['training']
    assert launch['training_seeds'] == list(SEEDS) == frozen['training_seeds']
    assert launch['evaluation_worlds'] == list(WORLDS) == frozen['evaluation_worlds']
    assert launch['new_rl_fits'] == 9 and launch['action_mask'] == MASK_RULE
    assert launch['checkpoint_selection'] == 'final-only'
    expected = {str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')} | {
        'configs/main.json', 'configs/rl-main-frozen.json', 'docs/scale-rl-protocol.md',
        'scripts/run_scale_rl.py', 'scripts/audit_scale_rl.py', 'scripts/run_mission_validation.py',
        'scripts/audit_scale_pilot.py',
        'tests/test_scale_rl.py', '.github/workflows/scale-rl.yml'}
    assert set(launch['source_sha256']) == expected
    for path, identity in launch['source_sha256'].items():
        assert sha(ROOT/path) == identity, path
    expected_fits = {(a, s) for a in ALGORITHMS for s in SEEDS}
    assert len(launch['fits']) == 9 and {(r['algorithm'], r['seed']) for r in launch['fits']} == \
        expected_fits
    fit_audits = []
    for record in launch['fits']:
        algorithm, seed = record['algorithm'], record['seed']
        settings = TrainConfig(**frozen['training'][algorithm], seed=seed)
        assert settings.episodes == 300 and settings.scenario_seed_start == 0
        assert not settings.validation_seeds and not settings.coordination_shaping_weight
        assert not settings.double_dqn and settings.device == 'cpu'
        assert record == json.loads((args.run/f'fit-{algorithm}-{seed}.json').read_text())
        assert record['source_revision'] == launch['source_revision']
        assert record['source_sha256'] == launch['source_sha256']
        assert record['train_config'] == settings.to_dict()
        assert record['env_config'] == config.to_dict()
        negative = {f'docs/experiments/{n}/checksums.json' for n in
                    ('scale-pilot-2026-10-11', 'scale-reference-2026-10-11',
                     'scale-loss-2026-10-11')}
        assert set(record['negative_archive_sha256']) == negative
        for path, identity in record['negative_archive_sha256'].items():
            assert sha(ROOT/path) == identity
            manifest = ROOT/path
            for relative, previous in json.loads(manifest.read_text()).items():
                evidence = manifest.parent/relative
                assert evidence.is_file(), str(evidence)
                assert evidence.stat().st_size == previous['bytes'], str(evidence)
                assert sha(evidence) == previous['sha256'], str(evidence)
        directory = args.run/'fits'/algorithm/f'seed-{seed}'
        assert set(record['files']) == {'policy.pt', 'initial.pt', 'metadata.json',
                                        'training.csv', 'training_summary.json'}
        assert {p.name for p in directory.iterdir() if p.is_file()} == set(record['files'])
        for name, identity in record['files'].items():
            assert sha(directory/name) == identity['sha256']
            assert (directory/name).stat().st_size == identity['bytes']
        fit_audits.append(verify_fit(directory, algorithm, seed, config, settings,
                                     launch['source_revision']))
    episodes = read(args.run/'episodes.csv')
    assert len(episodes) == 260
    with gzip.open(args.run/'steps.csv.gz', 'rt', newline='') as stream:
        traces = list(csv.DictReader(stream))
    grouped = {}
    for row in traces:
        key = (row['policy'], row['training_seed'], row['scenario_seed'])
        grouped.setdefault(key, []).append(row)
    assert len(grouped) == 260
    stores = {}
    for algorithm, seed in expected_fits:
        with np.load(args.run/f'{algorithm}-{seed}-scores.npz', allow_pickle=False) as packet:
            assert set(packet.files) == {'scores', 'masks'}
            stores[algorithm, str(seed)] = [packet['scores'], packet['masks'], 0]
    makers = {'greedy': lambda w: GreedyPolicy(), 'sweep': lambda w: SweepPolicy(),
              'random': RandomPolicy, 'masked_random': MaskedRandomPolicy}
    verified, max_drift = [], 0.
    for i, episode in enumerate(episodes):
        policy, seed, world = episode['policy'], episode['training_seed'], episode['scenario_seed']
        trace = grouped.pop((policy, seed, world))
        vectors = None
        if policy in ALGORITHMS:
            loaded = load_masked_policy(args.run/'fits'/policy/f'seed-{seed}/policy.pt')
            scores, masks, offset = stores[policy, seed]
            end = offset+len(trace)
            vectors = {'scores': scores[offset:end], 'masks': masks[offset:end]}
            stores[policy, seed][2] = end
        else:
            assert policy in makers and seed == ''
            loaded = makers[policy](int(world))
        row, drift = replay_world(config, loaded, episode, trace, vectors)
        verified.append(row)
        max_drift = max(max_drift, drift)
        print(f'audited {policy} seed {seed} world {world} ({i+1}/260)', flush=True)
    assert not grouped and sum(int(r['steps']) for r in verified) == len(traces)
    for scores, masks, offset in stores.values():
        assert len(scores) == len(masks) == offset
    comparison, summary, paired, decisions = independent_tables(verified)
    for name, rows in [('comparison', comparison), ('summary', summary), ('paired', paired)]:
        saved = read(args.run/f'{name}.csv')
        # CSV empty seed fields represent None, while all numeric fields retain exact decimals.
        if name == 'comparison':
            for r in saved:
                r['training_seed'] = int(r['training_seed']) if r['training_seed'] else None
        equal_tree(saved, rows)
    completion = {'comparison_complete': True, 'new_rl_fits': 9, 'rl_missions': 180,
                  'baseline_missions': 80, 'mission_steps': len(traces), 'per_fit': decisions,
                  'every_seed_learning_gate': {a: all(r['learning_gate_passed'] for r in decisions
                                                     if r['algorithm']==a) for a in ALGORITHMS},
                  'every_seed_greedy_gate': {a: all(r['greedy_gate_passed'] for r in decisions
                                                   if r['algorithm']==a) for a in ALGORITHMS}}
    equal_tree(json.loads((args.run/'completion.json').read_text()), completion)
    args.out.mkdir(parents=True, exist_ok=False)
    for path in args.run.iterdir():
        if path.is_dir():
            shutil.copytree(path, args.out/path.name, ignore=shutil.ignore_patterns('*.pt'))
        elif path.suffix != '.pt':
            shutil.copy2(path, args.out/path.name)
    audit = {'passed': True, 'source_revision': launch['source_revision'], 'fits': fit_audits,
             'mission_episodes': 260, 'mission_steps': len(traces), 'paired_rows': 720,
             'every_action_mask_hash_reward_exact': True, 'negative_archives_unchanged': True,
             'initial_tensors_verified': True, 'max_score_drift': max_drift,
             'score_atol': 1e-5, 'score_rtol': 1e-6, 'audit_script_sha256': sha(Path(__file__)),
             'workflow_run_id': os.environ.get('GITHUB_RUN_ID')}
    (args.out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    stats = {(r['policy'], r['metric']): r for r in summary}
    table = '\n'.join(f"| {p} | {stats[p, 'weighted_freshness']['mean']:.6f} | "
                      f"{stats[p, 'overdue_fraction']['mean']:.6f} | "
                      f"{stats[p, 'return']['mean']:.6f} | "
                      f"{stats[p, 'total_losses']['mean']:.3f} |" for p in ALGORITHMS+BASELINES)
    report = f'''# Frozen 40×40 RL main study

Source: `{launch['source_revision']}`. Exactly nine RL fits, three per algorithm,
300 episodes each, independently initialized without supervised warm starts.
No post-result tuning or best checkpoint selection. Original negative evidence is preserved.

| Policy | Freshness ↑ | Overdue ↓ | Return ↑ | Losses ↓ |
|---|---:|---:|---:|---:|
{table}

All-seed learned-contribution gates: `{completion['every_seed_learning_gate']}`.
All-seed Greedy capability gates: `{completion['every_seed_greedy_gate']}`.
Comparison completion and performance gates are separate; a negative result does
not trigger more fitting or establish an RL advantage. See `completion.json` for
every seed and each criterion. `summary.csv` preserves fitted-seed sample SD.

The complete environment and all settings were frozen before fitting in
`configs/rl-main-frozen.json`; the prospective protocol is `docs/scale-rl-protocol.md`.
Training worlds 0–299, new final worlds 22000–22019, no validation selection.
All nine fits finished before final evaluation. Each final model runs 20 missions;
Greedy, sweep, unmasked random and masked random run the same worlds (260 total).
Full-horizon accounting includes early fleet failure. Masked methods receive a
rule-based battery/collision feasibility mask. Greedy/sweep retain different
native proposal preferences; safe flight is not entirely a learned result.
Actual steps, updates and training time appear in `audit.json` and each fit's
metadata/history. A fixed episode budget can yield different real interactions
if fleets fail early. This is a fixed-budget comparison, not proof of convergence.

Means first average worlds within each fitted seed; SD is across three fitted
seed means. Baseline SD is across 20 worlds. Repeated seed/world rows are not
independent trained replicates; there are no significance claims. `paired.csv`
contains all 720 matched fitted-world/baseline comparisons. Neural outputs are
RL Q/logit scores, not supervised inspection gains. No 60×60 results are claimed.

Independent audit verified initial tensors, checkpoint bytes, source/settings,
all training episode/update counts, every mission action/hash/mask/reward and
all summaries/gates. All {len(traces)} mission steps replayed, maximum neural score
drift {max_drift:.9g}, with fixed atol 1e-5 and rtol 1e-6 and exact selected actions.
Numeric evidence, complete compressed scores/traces, learning curves and checksums
are archived here. Raw initial/final weights remain in nine per-fit Actions
artifacts and `frozen-40x40-rl-results-{launch['source_revision']}` for 90 days.
Download the results artifact, extract its run folder, and independently replay:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python scripts/audit_scale_rl.py --run runs/rl-main-2026-10-11 --out runs/replayed-rl-main
```
'''
    (args.out/'README.md').write_text(report)
    identities = {str(p.relative_to(args.out)): {'sha256': sha(p), 'bytes': p.stat().st_size}
                  for p in sorted(args.out.rglob('*')) if p.is_file()}
    (args.out/'checksums.json').write_text(json.dumps(identities, indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == '__main__':
    main()
