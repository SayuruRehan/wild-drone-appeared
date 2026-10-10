"""Prospective supervised inspection ranking and mission pilot in the 40x40 domain."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from run_centered_inspection import digest, metrics, summary, zipped_json
from run_mission_validation import METRICS, ScorePolicy, per_fit, rollout

from wild_drone.baselines import GreedyPolicy, SweepPolicy
from wild_drone.centered_inspection import fit_centered, load_centered_model
from wild_drone.config import EnvConfig
from wild_drone.experiments import write_rows
from wild_drone.inspection_targets import collect_states, score_model, validate_worlds
from wild_drone.masked_learning import MaskedRandomPolicy
from wild_drone.training_io import atomic_json_save, atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
TRAIN_WORLDS = tuple(range(200))
RANKING_WORLDS = tuple(range(6800, 6816))
MISSION_WORLDS = tuple(range(7000, 7020))
STEPS = (30, 60, 120, 240)
UPDATES, BATCH_SIZE = 2000, 64


def pilot_config():
    return EnvConfig(**json.loads((ROOT/'configs'/'main.json').read_text())['environment'])


def validate_recipe(training, ranking, missions, steps, config):
    validate_worlds(training, ranking)
    validate_worlds(training, missions)
    validate_worlds(ranking, missions)
    if (not steps or tuple(sorted(set(steps))) != tuple(steps) or
            any(not isinstance(s, int) or not 0 <= s < config.horizon for s in steps)):
        raise ValueError('distinct increasing active snapshot steps required')
    if any(5800 <= world < 5820 or world >= 10000
           for world in (*training, *ranking, *missions)):
        raise ValueError('confirmation and final worlds remain unopened')


def pilot_decisions(ranking, missions):
    results = []
    baselines = {name: [r for r in missions if r['policy'] == name]
                 for name in ('greedy', 'masked_random')}
    worlds = {r['scenario_seed'] for r in baselines['greedy']}
    if not worlds or any(len(group) != len(worlds) or
                         {r['scenario_seed'] for r in group} != worlds
                         for group in baselines.values()):
        raise ValueError('matched baseline mission worlds required')
    measured = ('weighted_freshness', 'overdue_fraction', 'total_losses',
                'fleet_failure', 'return')
    means = {name: {k: float(np.mean([r[k] for r in group])) for k in measured}
             for name, group in baselines.items()}
    for seed in (0, 1, 2):
        selected = [r for r in ranking if r['seed'] == seed and r['phase'] == 'final' and
                    r['policy'] == 'network']
        if len(selected) != 2 or {r['split'] for r in selected} != {'training', 'fresh'}:
            raise ValueError('matched complete ranking splits required')
        natural = {r['split']: r for r in selected}
        group = [r for r in missions if r['policy'] == 'centered' and r['training_seed'] == seed]
        if len(group) != len(worlds) or {r['scenario_seed'] for r in group} != worlds:
            raise ValueError('matched scorer mission worlds required')
        model = {k: float(np.mean([r[k] for r in group])) for k in measured}
        greedy, random = means['greedy'], means['masked_random']
        ranks = {'training_regret_le_005': natural['training']['normalised_gain_regret'] <= .05,
                 'fresh_regret_le_020': natural['fresh']['normalised_gain_regret'] <= .20,
                 'fresh_best_match_ge_025': natural['fresh']['one_step_optimal'] >= .25}
        mission = {'freshness_ge_90pct_greedy': model['weighted_freshness'] >=
                   .9*greedy['weighted_freshness'],
                   'overdue_le_greedy_plus_005': model['overdue_fraction'] <=
                   greedy['overdue_fraction']+.05,
                   'losses_le_greedy_plus_01': model['total_losses'] <= greedy['total_losses']+.1,
                   'fleet_failures_le_greedy': model['fleet_failure'] <= greedy['fleet_failure'],
                   'freshness_ge_random_plus_001': model['weighted_freshness'] >=
                   random['weighted_freshness']+.01,
                   'return_ge_random_plus_05': model['return'] >= random['return']+.5}
        results.append({'seed': seed, 'ranking_checks': ranks,
                        'ranking_passed': all(ranks.values()),
                        'mission_checks': mission, 'mission_passed': all(mission.values()),
                        'mission_means': {'centered': model, **means}})
    return results


def sample_audit(data, seed):
    ids = torch.where(data['informative'])[0]
    rng = torch.Generator().manual_seed(seed+1000)
    sequence, counts = hashlib.sha256(), torch.zeros(len(data['maps']), dtype=torch.int64)
    for _ in range(UPDATES):
        batch = ids[torch.randint(len(ids), (BATCH_SIZE,), generator=rng)]
        sequence.update(batch.numpy().astype('<i8').tobytes())
        counts += torch.bincount(batch, minlength=len(counts))
    return {'sample_sequence_sha256': sequence.hexdigest(), 'per_state_counts': counts.tolist(),
            'sample_draws': UPDATES*BATCH_SIZE, 'informative_states': len(ids),
            'sampled_distinct_states': int((counts > 0).sum()),
            'draws_per_informative_state': UPDATES*BATCH_SIZE/len(ids)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('full published source revision required')
    config = pilot_config()
    validate_recipe(TRAIN_WORLDS, RANKING_WORLDS, MISSION_WORLDS, STEPS, config)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve(),
             ROOT/'scripts'/'run_centered_inspection.py',
             ROOT/'scripts'/'run_mission_validation.py',
             ROOT/'docs'/'scale-pilot-protocol.md', ROOT/'configs'/'main.json']
    launch = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'source_revision': args.source_revision,
              'source_revision_kind': 'operator-verified published source snapshot',
              'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
              'env_config': config.to_dict(), 'training_worlds': list(TRAIN_WORLDS),
              'ranking_worlds': list(RANKING_WORLDS), 'mission_worlds': list(MISSION_WORLDS),
              'snapshot_steps': list(STEPS), 'training_seeds': [0, 1, 2],
              'updates_per_fit': UPDATES, 'batch_size': BATCH_SIZE,
              'learning_rate': .001, 'clip': .5, 'cpu_threads': 1,
              'transfer_from_20x20': False, 'rl_updates': 0}
    atomic_json_save(launch, args.out/'launch.json')
    started, data, records, identities, missing = time.perf_counter(), {}, {}, {}, {}
    for split, worlds in [('training', TRAIN_WORLDS), ('fresh', RANKING_WORLDS)]:
        data[split], records[split], missing[split] = collect_states(config, worlds, STEPS)
        path = args.out/f'{split}.pt'
        atomic_torch_save(data[split], path)
        restored = torch.load(path, weights_only=True, map_location='cpu')
        if any(not torch.equal(v, restored[k]) for k, v in data[split].items()):
            raise ValueError('dataset export differs')
        identities[split] = {'sha256': digest(path), 'bytes': path.stat().st_size,
                             'states': len(records[split]),
                             'informative': int(data[split]['informative'].sum())}
        print(f'{split}: {len(records[split])} states collected', flush=True)
    zipped_json({split: [r | {'gains': data[split]['gains'][i].tolist(),
                             'mask': data[split]['masks'][i].tolist()}
                        for i, r in enumerate(index)] for split, index in records.items()},
                args.out/'inputs.json.gz')
    atomic_json_save(identities, args.out/'datasets.json')
    atomic_json_save(missing, args.out/'missing-snapshots.json')
    checkpoints, sampling = {}, {}
    for seed in (0, 1, 2):
        _, identity, history, _ = fit_centered(
            config, data['training'], arm='centered', seed=seed, updates=UPDATES,
            batch_size=BATCH_SIZE, output=args.out/f'seed-{seed}')
        sampling[str(seed)] = sample_audit(data['training'], seed)
        if sampling[str(seed)]['sample_sequence_sha256'] != identity['sample_sequence_sha256']:
            raise ValueError('sample sequence differs')
        identity['training_dataset_sha256'] = identities['training']['sha256']
        checkpoints[str(seed)] = identity
        write_rows(args.out/f'seed-{seed}'/'training.csv', history)
        print(f'seed-{seed}: fitting complete', flush=True)
    atomic_json_save(checkpoints, args.out/'checkpoints.json')
    atomic_json_save(sampling, args.out/'sampling.json')
    rank_dir, mission_dir = args.out/'rankings', args.out/'missions'
    rank_dir.mkdir()
    mission_dir.mkdir()
    natural, ranked = [], []

    def measure(scores, split, seed, phase, policy):
        rows = metrics(scores, data[split], records[split], split=split, seed=seed,
                       phase=phase, policy=policy)
        natural.extend(rows)
        ranked.append({'split': split, 'seed': seed, 'phase': phase, 'policy': policy,
                       **summary(rows)})

    for seed in (0, 1, 2):
        for phase in ('initial', 'final'):
            model = load_centered_model(args.out/f'seed-{seed}'/f'{phase}.pt', config, 'centered')
            scores = {split: score_model(model, data[split]) for split in
                      (('fresh',) if phase == 'initial' else ('training', 'fresh'))}
            np.savez_compressed(rank_dir/f'seed-{seed}-{phase}.npz', **scores)
            for split, values in scores.items():
                measure(values, split, seed, phase, 'network')
    train = data['training']
    masked = train['masks'] & train['informative'][:, None]
    means = (train['targets']*masked).sum(0)/masked.sum(0).clamp(min=1)
    atomic_json_save({'scores': means.tolist(), 'counts': masked.sum(0).tolist()},
                    args.out/'action-mean.json')
    for policy in ('action_mean', 'inspection_reference'):
        for split in ('training', 'fresh'):
            scores = np.tile(means.numpy(), (len(data[split]['maps']), 1)) if \
                policy == 'action_mean' else data[split]['gains'].numpy()
            measure(scores, split, None, 'control', policy)
    write_rows(rank_dir/'natural.csv', natural)
    write_rows(rank_dir/'per-fit.csv', ranked)
    episodes, traces = [], []
    conditions = [('centered', seed) for seed in (0, 1, 2)] + \
                 [(name, None) for name in ('action_mean', 'greedy', 'sweep', 'masked_random')]
    for name, seed in conditions:
        model = None if seed is None else load_centered_model(
            args.out/f'seed-{seed}'/'final.pt', config, 'centered')
        values, masks = [], []
        for world in MISSION_WORLDS:
            if model is not None:
                policy = ScorePolicy(config, model=model)
            elif name == 'action_mean':
                policy = ScorePolicy(config, scores=means.numpy())
            else:
                policy = {'greedy': GreedyPolicy, 'sweep': SweepPolicy,
                          'masked_random': lambda: MaskedRandomPolicy(world)}[name]()
            episode, trace, vectors = rollout(config, policy, world, name=name, training_seed=seed)
            episodes.append(episode)
            traces.extend(trace)
            if vectors is not None:
                values.append(vectors['scores'])
                masks.append(vectors['masks'])
        if values:
            np.savez_compressed(mission_dir/f'{name}-seed-{seed}-scores.npz',
                                scores=np.concatenate(values), masks=np.concatenate(masks))
        write_rows(mission_dir/'episodes.csv', episodes)
        print(f'{name}/seed-{seed}: {len(MISSION_WORLDS)} missions complete', flush=True)
    with gzip.open(mission_dir/'steps.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    fitted = per_fit(episodes)
    write_rows(mission_dir/'per-fit.csv', fitted)
    aggregate = []
    for name in dict.fromkeys(r['policy'] for r in episodes):
        group = [r for r in (fitted if name == 'centered' else episodes) if r['policy'] == name]
        for metric in METRICS:
            values = [r[metric] for r in group]
            aggregate.append({'policy': name, 'metric': metric,
                              'unit': 'fitted_seed_means' if name == 'centered' else 'worlds',
                              'units': len(values), 'mean': float(np.mean(values)),
                              'sample_sd': float(np.std(values, ddof=1))
                              if len(values) > 1 else 0.0})
    write_rows(mission_dir/'summary.csv', aggregate)
    criterion = pilot_decisions(ranked, episodes)
    completion = {'ranking_passed': all(r['ranking_passed'] for r in criterion),
                  'mission_passed': all(r['mission_passed'] for r in criterion),
                  'per_seed': criterion, 'mission_episodes': len(episodes),
                  'mission_steps': len(traces), 'elapsed_seconds': time.perf_counter()-started,
                  'rl_updates': 0}
    atomic_json_save(completion, args.out/'completion.json')
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    main()
