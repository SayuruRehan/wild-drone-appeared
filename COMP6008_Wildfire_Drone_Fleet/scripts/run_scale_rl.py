"""Frozen nine-fit 40x40 RL study; evaluation starts only after every fit completes."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from run_mission_validation import METRICS, ScorePolicy, per_fit, rollout

from wild_drone.baselines import GreedyPolicy, RandomPolicy, SweepPolicy
from wild_drone.config import EnvConfig
from wild_drone.experiments import plot_learning, plot_results, write_rows
from wild_drone.learning import FleetNetwork, TrainConfig, tensor_observations
from wild_drone.masked_learning import (
    MASK_RULE,
    MaskedRandomPolicy,
    load_masked_policy,
    train_masked,
)
from wild_drone.training_io import atomic_json_save, atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ('dqn', 'reinforce', 'a2c')
BASELINES = ('greedy', 'sweep', 'random', 'masked_random')
SEEDS = (0, 1, 2)
WORLDS = tuple(range(22000, 22020))
NEGATIVE = tuple(f'docs/experiments/{n}/checksums.json' for n in
                 ('scale-pilot-2026-10-11', 'scale-reference-2026-10-11', 'scale-loss-2026-10-11'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    files = list((ROOT/'src').rglob('*.py')) + [ROOT/p for p in
             ('configs/main.json', 'configs/rl-main-frozen.json', 'docs/scale-rl-protocol.md',
              'scripts/run_scale_rl.py', 'scripts/audit_scale_rl.py',
              'scripts/run_mission_validation.py', 'scripts/audit_scale_pilot.py',
              'tests/test_scale_rl.py',
              '.github/workflows/scale-rl.yml')]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def verify_negative_archives():
    for name in NEGATIVE:
        manifest = ROOT/name
        for relative, identity in json.loads(manifest.read_text()).items():
            evidence = manifest.parent/relative
            assert evidence.is_file(), str(evidence)
            assert evidence.stat().st_size == identity['bytes'], str(evidence)
            assert digest(evidence) == identity['sha256'], str(evidence)


def recipe():
    frozen = json.loads((ROOT/'configs/rl-main-frozen.json').read_text())
    config = EnvConfig(**frozen['environment'])
    assert frozen['training_seeds'] == list(SEEDS) and frozen['evaluation_worlds'] == list(WORLDS)
    assert config.grid_size == 40 and config.horizon == 300
    assert set(frozen['training']) == set(ALGORITHMS)
    for algorithm in ALGORITHMS:
        t = training_config(frozen['training'], algorithm, 0)
        assert t.episodes == 300 and t.scenario_seed_start == 0 and not t.validation_seeds
        assert not t.double_dqn and t.coordination_shaping_weight == 0 and t.device == 'cpu'
        assert t.spatial_encoding and t.monte_carlo_baseline
    return config, frozen['training']


def training_config(settings, algorithm, seed):
    if algorithm not in ALGORITHMS or seed not in SEEDS:
        raise ValueError('one of the nine frozen fits required')
    return TrainConfig(**settings[algorithm], seed=seed)


def check_fresh_worlds(archive, worlds):
    seen = set()
    for path in Path(archive).rglob('*.csv'):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                for key in ('scenario_seed', 'world', 'world_seed'):
                    value = row.get(key)
                    if value and str(value).isdigit():
                        seen.add(int(value))
    if set(worlds) & seen:
        raise ValueError('evaluation scenarios already appear in archived evidence')


class RLScorePolicy(ScorePolicy):
    """Capture full genuine RL outputs under the deployed first-index masked argmax."""

    def act(self, env, observation):
        from wild_drone.masked_learning import action_mask

        assert env.config == self.config
        with torch.no_grad():
            scores = self.model(*tensor_observations(
                observation, spatial_encoding=self.model.spatial_encoding))[0][0].numpy()
        if scores.shape != (125,) or not np.isfinite(scores).all():
            raise ValueError('finite full RL action outputs required')
        self.last_scores, self.last_mask = scores.copy(), action_mask(env)
        return int(np.argmax(np.where(self.last_mask, scores, -np.inf)))


def tables(rows, worlds=WORLDS):
    keys = {(p, s) for p in ALGORITHMS for s in SEEDS} | {(p, None) for p in BASELINES}
    if set((r['policy'], r['training_seed']) for r in rows) != keys:
        raise ValueError('all nine fits and four baselines required')
    for p, s in keys:
        group = [r for r in rows if (r['policy'], r['training_seed']) == (p, s)]
        if len(group) != len(worlds) or {r['scenario_seed'] for r in group} != set(worlds):
            raise ValueError('distinct matched evaluation worlds required')
    comparison = per_fit(rows)
    summary, paired, decisions = [], [], []
    for p in ALGORITHMS + BASELINES:
        group = [r for r in (comparison if p in ALGORITHMS else rows) if r['policy'] == p]
        for metric in METRICS:
            values = [float(r[metric]) for r in group]
            if not np.isfinite(values).all():
                raise ValueError('finite evaluation metrics required')
            summary.append({'policy': p, 'metric': metric,
                            'unit': 'seed_means' if p in ALGORITHMS else 'worlds',
                            'units': len(values), 'mean': float(np.mean(values)),
                            'sample_sd': float(np.std(values, ddof=1))})
    for row in rows:
        if row['policy'] not in ALGORITHMS:
            continue
        for baseline in BASELINES:
            old = next(r for r in rows if r['policy'] == baseline and
                       r['scenario_seed'] == row['scenario_seed'])
            paired.append({'policy': row['policy'], 'training_seed': row['training_seed'],
                           'scenario_seed': row['scenario_seed'], 'baseline': baseline,
                           **{m+'_delta': row[m]-old[m] for m in METRICS}})
    means = {(r['policy'], r['training_seed']): r for r in comparison}
    random, greedy = means['masked_random', None], means['greedy', None]
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            row = means[algorithm, seed]
            checks = {'freshness_ge_masked_random_plus_001':
                      row['weighted_freshness'] >= random['weighted_freshness']+.01,
                      'return_ge_masked_random_plus_05': row['return'] >= random['return']+.5,
                      'losses_le_masked_random': row['total_losses'] <= random['total_losses'],
                      'failures_le_masked_random': row['fleet_failure'] <= random['fleet_failure']}
            greedy_checks = {'freshness_ge_90pct_greedy':
                             row['weighted_freshness'] >= .9*greedy['weighted_freshness'],
                             'overdue_le_greedy_plus_005':
                             row['overdue_fraction'] <= greedy['overdue_fraction']+.05,
                             'losses_le_greedy_plus_01':
                             row['total_losses'] <= greedy['total_losses']+.1,
                             'failures_le_greedy': row['fleet_failure'] <= greedy['fleet_failure']}
            decisions.append({'algorithm': algorithm, 'seed': seed,
                              'masked_random_checks': checks, 'greedy_checks': greedy_checks,
                              'learning_gate_passed': all(checks.values()),
                              'greedy_gate_passed': all(greedy_checks.values())})
    return comparison, summary, paired, decisions


def fit(args):
    config, settings = recipe()
    verify_negative_archives()
    check_fresh_worlds(ROOT/'docs/experiments', WORLDS)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == \
        args.source_revision
    t = training_config(settings, args.algorithm, args.seed)
    directory = args.out/'fits'/args.algorithm/f'seed-{args.seed}'
    directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    model_config = {'critic': args.algorithm != 'dqn', 'spatial_encoding': True}
    torch.manual_seed(args.seed)
    initial = FleetNetwork(config.grid_size, **model_config)
    atomic_torch_save({'state_dict': initial.state_dict(), 'model_config': model_config,
                       'env_config': config.to_dict(), 'train_config': t.to_dict(),
                       'algorithm': args.algorithm, 'completed_episodes': 0,
                       'git_revision': args.source_revision}, directory/'initial.pt')
    train_masked(args.algorithm, config, t, directory)
    identities = {p.name: {'sha256': digest(p), 'bytes': p.stat().st_size}
                  for p in directory.iterdir() if p.is_file()}
    atomic_json_save({'source_revision': args.source_revision, 'algorithm': args.algorithm,
                      'seed': args.seed, 'env_config': config.to_dict(),
                      'train_config': t.to_dict(), 'action_mask': MASK_RULE,
                      'source_sha256': source_manifest(),
                      'negative_archive_sha256': {p: digest(ROOT/p) for p in NEGATIVE},
                      'started_utc': started,
                      'finished_utc': datetime.now(timezone.utc).isoformat(),
                      'files': identities}, args.out/f'fit-{args.algorithm}-{args.seed}.json')


def evaluate(args):
    from audit_scale_rl import verify_fit

    config, settings = recipe()
    assert not (args.out/'launch.json').exists(), 'evaluation already started'
    verify_negative_archives()
    check_fresh_worlds(ROOT/'docs/experiments', WORLDS)
    assert {str(p.relative_to(args.out/'fits')) for p in (args.out/'fits').glob('*/seed-*')} == \
        {f'{a}/seed-{s}' for a in ALGORITHMS for s in SEEDS}
    records = []
    # Verify all nine complete final checkpoints before opening any final-world state.
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            record = json.loads((args.out/f'fit-{algorithm}-{seed}.json').read_text())
            assert record['source_revision'] == args.source_revision
            assert (record['algorithm'], record['seed']) == (algorithm, seed)
            assert record['env_config'] == config.to_dict()
            assert record['train_config'] == training_config(settings, algorithm, seed).to_dict()
            assert record['source_sha256'] == source_manifest()
            assert record['negative_archive_sha256'] == {p: digest(ROOT/p) for p in NEGATIVE}
            directory = args.out/'fits'/algorithm/f'seed-{seed}'
            assert set(record['files']) == {'policy.pt', 'initial.pt', 'metadata.json',
                                           'training.csv', 'training_summary.json'}
            assert {p.name for p in directory.iterdir() if p.is_file()} == set(record['files'])
            for name, identity in record['files'].items():
                assert digest(directory/name) == identity['sha256']
                assert (directory/name).stat().st_size == identity['bytes']
            saved = torch.load(directory/'policy.pt', weights_only=True, map_location='cpu')
            assert saved['completed_episodes'] == 300
            assert saved['git_revision'] == args.source_revision
            assert saved['train_config'] == record['train_config']
            verify_fit(directory, algorithm, seed, config,
                       training_config(settings, algorithm, seed), args.source_revision)
            records.append(record)
    atomic_json_save({'source_revision': args.source_revision, 'env_config': config.to_dict(),
                      'training': settings, 'training_seeds': list(SEEDS),
                      'evaluation_worlds': list(WORLDS), 'fits': records,
                      'source_sha256': source_manifest(), 'new_rl_fits': 9,
                      'action_mask': MASK_RULE, 'checkpoint_selection': 'final-only',
                      'started_utc': datetime.now(timezone.utc).isoformat()},
                     args.out/'launch.json')
    episodes, traces = [], []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            loaded = load_masked_policy(args.out/'fits'/algorithm/f'seed-{seed}/policy.pt')
            policy = RLScorePolicy(config, model=loaded.model)
            vectors = []
            for world in WORLDS:
                episode, trace, packet = rollout(config, policy, world, name=algorithm,
                                                  training_seed=seed)
                episodes.append(episode)
                traces.extend(trace)
                vectors.append(packet)
            np.savez_compressed(args.out/f'{algorithm}-{seed}-scores.npz',
                                **{k: np.concatenate([p[k] for p in vectors])
                                   for k in ('scores', 'masks')})
            print(f'evaluated {algorithm} seed {seed}: 20 final missions', flush=True)
    makers = {'greedy': lambda w: GreedyPolicy(), 'sweep': lambda w: SweepPolicy(),
              'random': RandomPolicy, 'masked_random': MaskedRandomPolicy}
    for name, make in makers.items():
        for world in WORLDS:
            episode, trace, _ = rollout(config, make(world), world, name=name)
            episodes.append(episode)
            traces.extend(trace)
    write_rows(args.out/'episodes.csv', episodes)
    with gzip.open(args.out/'steps.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    comparison, summary, paired, decisions = tables(episodes)
    for name, rows in [('comparison', comparison), ('summary', summary), ('paired', paired)]:
        write_rows(args.out/f'{name}.csv', rows)
    atomic_json_save({'comparison_complete': True, 'new_rl_fits': 9, 'rl_missions': 180,
                      'baseline_missions': 80, 'mission_steps': len(traces), 'per_fit': decisions,
                      'every_seed_learning_gate': {a: all(r['learning_gate_passed'] for r in
                                                         decisions if r['algorithm'] == a)
                                                   for a in ALGORITHMS},
                      'every_seed_greedy_gate': {a: all(r['greedy_gate_passed'] for r in
                                                       decisions if r['algorithm'] == a)
                                                 for a in ALGORITHMS}}, args.out/'completion.json')
    for algorithm in ALGORITHMS:
        plot_learning([args.out/'fits'/algorithm/f'seed-{s}/training.csv' for s in SEEDS],
                      args.out/f'learning-{algorithm}.png')
    plot_results(args.out/'episodes.csv', args.out/'comparison.png')
    print('Nine fits and 260 final missions complete; independent audit remains required.',
          flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'evaluate'])
    parser.add_argument('--algorithm', choices=ALGORITHMS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        parser.error('full frozen source revision required')
    if args.stage == 'fit' and (args.algorithm is None or args.seed is None):
        parser.error('algorithm and seed required for one of the nine fits')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    (fit if args.stage == 'fit' else evaluate)(args)


if __name__ == '__main__':
    main()
