"""Frozen loss-only comparison against the audited 40x40 regression fits."""

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
from audit_scale_pilot import audit_data
from run_centered_inspection import metrics, summary
from run_mission_validation import METRICS, ScorePolicy, digest, per_fit, rollout
from run_scale_pilot import pilot_decisions, sample_audit

from wild_drone.centered_inspection import FORMAT, CenteredInspectionNetwork, load_centered_model
from wild_drone.config import EnvConfig
from wild_drone.experiments import write_rows
from wild_drone.inspection_targets import score_model
from wild_drone.training_io import atomic_json_save, atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'docs/experiments/scale-pilot-2026-10-11'
EXACT = ROOT / 'docs/experiments/scale-reference-2026-10-11'
UPDATES, BATCH_SIZE = 2000, 64
WORLDS = tuple(range(7000, 7020))
METHOD = 'supervised-tied-best-set-probability'


def read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def ranking_loss(scores, gains, masks):
    """Negative log total probability of any tied feasible optimum."""
    if (scores.ndim != 2 or scores.shape != gains.shape or masks.shape != scores.shape or
            masks.dtype != torch.bool or not masks.any(1).all() or
            not torch.isfinite(scores).all() or not torch.isfinite(gains).all()):
        raise ValueError('finite matching scores/gains and nonempty boolean masks required')
    best = gains.masked_fill(~masks, -torch.inf).max(1).values
    optimum = masks & torch.isclose(gains, best[:, None], rtol=0, atol=1e-12)
    return (torch.logsumexp(scores.masked_fill(~masks, -torch.inf), 1) -
            torch.logsumexp(scores.masked_fill(~optimum, -torch.inf), 1)).mean()


def fit_ranking(config, data, *, seed, updates, batch_size, output, parent_initial,
                expected_sequence):
    if (not isinstance(seed, int) or seed < 0 or not isinstance(updates, int) or updates < 1 or
            not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('nonnegative seed and positive fitting budget required')
    ids = torch.where(data['informative'])[0]
    if not len(ids):
        raise ValueError('no informative training states')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = CenteredInspectionNetwork(config, 'centered')
    parent = torch.load(parent_initial, map_location='cpu', weights_only=True)
    if (parent['seed'] != seed or parent['completed_updates'] != 0 or
            parent['env_config'] != config.to_dict() or parent['arm'] != 'centered' or
            set(parent['state_dict']) != set(model.state_dict()) or
            any(not torch.equal(v, parent['state_dict'][k])
                for k, v in model.state_dict().items())):
        raise ValueError('parent initial tensors or metadata differ')
    sampler = torch.Generator().manual_seed(seed + 1000)
    optimiser = torch.optim.Adam(model.parameters(), lr=.001)
    sequence, history, identities = hashlib.sha256(), [], {}

    def export(phase, completed):
        path = output / f'{phase}.pt'
        atomic_torch_save({'format': FORMAT, 'method': METHOD, 'env_config': config.to_dict(),
                           'arm': 'centered', 'seed': seed, 'completed_updates': completed,
                           'state_dict': model.state_dict()}, path)
        loaded = load_centered_model(path, config, 'centered')
        if any(not torch.equal(v, loaded.state_dict()[k])
               for k, v in model.state_dict().items()):
            raise ValueError('exported ranking tensors differ')
        identities[path.name] = {'sha256': digest(path), 'bytes': path.stat().st_size,
                                 'completed_updates': completed}

    export('initial', 0)
    for update in range(1, updates + 1):
        batch = ids[torch.randint(len(ids), (batch_size,), generator=sampler)]
        sequence.update(batch.numpy().astype('<i8').tobytes())
        scores = model(data['maps'][batch], data['vectors'][batch])[0]
        loss = ranking_loss(scores, data['gains'][batch], data['masks'][batch])
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite ranking loss')
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimiser.step()
        history.append({'update': update, 'training_batch_loss': float(loss.detach())})
        if update % 250 == 0:
            print(f'seed {seed}: {update}/{updates} updates', flush=True)
    if sequence.hexdigest() != expected_sequence:
        raise ValueError('parent sampling sequence differs')
    export('final', updates)
    return model.eval(), {'seed': seed, 'method': METHOD, 'completed_updates': updates,
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'initial_tensors_match_parent': True,
                          'sample_sequence_sha256': sequence.hexdigest(),
                          'checkpoints': identities}, history


def load_parent(parent_run):
    launch = json.loads((PARENT / 'launch.json').read_text())
    if (launch['source_revision'] != '7c9562aa6064f46d96cc9a163de3deb2ecaa3e19' or
            launch['training_seeds'] != [0, 1, 2] or launch['updates_per_fit'] != UPDATES or
            launch['batch_size'] != BATCH_SIZE or launch['mission_worlds'] != list(WORLDS) or
            launch['training_worlds'] != list(range(200)) or
            launch['ranking_worlds'] != list(range(6800, 6816))):
        raise ValueError('frozen parent recipe differs')
    for path, expected in launch['source_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'frozen parent source differs: {path}')
    names = ('launch.json', 'datasets.json', 'checkpoints.json', 'sampling.json',
             'inputs.json.gz', 'rankings/per-fit.csv', 'missions/episodes.csv', 'completion.json',
             'audit.json')
    checks = json.loads((PARENT / 'checksums.json').read_text())
    for name in names:
        if digest(PARENT / name) != checks[name]['sha256'] or \
                (name != 'audit.json' and digest(parent_run / name) != digest(PARENT / name)):
            raise ValueError(f'parent archive/input differs: {name}')
    if not json.loads((PARENT / 'audit.json').read_text())['passed']:
        raise ValueError('audited parent required')
    exact_checks = json.loads((EXACT / 'checksums.json').read_text())
    for name in ('episodes.csv', 'audit.json', 'launch.json'):
        if digest(EXACT / name) != exact_checks[name]['sha256']:
            raise ValueError('exact reference archive differs')
    if not json.loads((EXACT / 'audit.json').read_text())['passed']:
        raise ValueError('audited exact reference required')
    index, data = audit_data(parent_run, launch)
    checkpoints = json.loads((PARENT / 'checkpoints.json').read_text())
    for seed in range(3):
        for phase in ('initial', 'final'):
            path = parent_run / f'seed-{seed}/{phase}.pt'
            identity = checkpoints[str(seed)]['checkpoints'][path.name]
            if digest(path) != identity['sha256'] or path.stat().st_size != identity['bytes']:
                raise ValueError('parent checkpoint bytes differ')
    return EnvConfig(**launch['env_config']), index, data, checkpoints


def loss_decisions(ranking, parent_ranking, episodes):
    # Preserve the original numerical gates while naming the new policy separately.
    gate_rows = [dict(r, policy='centered') if r['policy'] == 'ranking' else r
                 for r in episodes if r['policy'] != 'centered']
    result = pilot_decisions(ranking, gate_rows)
    for record in result:
        seed = record['seed']
        groups = {p: [r for r in episodes if r['policy'] == p and
                      str(r['training_seed']) == str(seed)] for p in ('ranking', 'centered')}
        if (not groups['ranking'] or len(groups['ranking']) != len(groups['centered']) or
                any(len({r['scenario_seed'] for r in group}) != len(group)
                    for group in groups.values()) or
                {r['scenario_seed'] for r in groups['ranking']} !=
                {r['scenario_seed'] for r in groups['centered']}):
            raise ValueError('matched objective comparison worlds required')
        means = {p: {k: float(np.mean([float(r[k]) for r in group])) for k in
                     ('weighted_freshness', 'overdue_fraction', 'total_losses', 'fleet_failure')}
                 for p, group in groups.items()}
        old_rank = [r for r in parent_ranking if str(r['seed']) == str(seed) and
                    r['split'] == 'fresh' and r['phase'] == 'final' and r['policy'] == 'network']
        new_rank = [r for r in ranking if r['seed'] == seed and r['split'] == 'fresh' and
                    r['phase'] == 'final' and r['policy'] == 'network']
        if len(old_rank) != 1 or len(new_rank) != 1:
            raise ValueError('unique fresh ranking means required')
        new, old = means['ranking'], means['centered']
        checks = {'fresh_regret_not_higher': new_rank[0]['normalised_gain_regret'] <=
                  float(old_rank[0]['normalised_gain_regret']),
                  'freshness_improves_ge_001': new['weighted_freshness'] >=
                  old['weighted_freshness'] + .01,
                  **{k + '_not_higher': new[k] <= old[k] for k in
                     ('overdue_fraction', 'total_losses', 'fleet_failure')}}
        record.update(effect_checks=checks, effect_passed=all(checks.values()),
                      objective_means=means)
        record['mission_means']['ranking'] = record['mission_means'].pop('centered')
    return result


def comparisons(episodes):
    return per_fit(episodes)


def aggregate(episodes):
    fitted, result = per_fit(episodes), []
    for policy in dict.fromkeys(r['policy'] for r in episodes):
        network = policy in ('ranking', 'centered')
        group = [r for r in (fitted if network else episodes) if r['policy'] == policy]
        for key in METRICS:
            values = [float(r[key]) for r in group]
            result.append({'policy': policy, 'metric': key,
                           'unit': 'fitted_seed_means' if network else 'worlds',
                           'units': len(values), 'mean': float(np.mean(values)),
                           'sample_sd': float(np.std(values, ddof=1)) if len(values)>1 else 0.})
    return result


def paired(episodes):
    result = []
    for row in episodes:
        if row['policy'] != 'ranking':
            continue
        for name in ('centered', 'greedy', 'masked_random', 'exact_inspection'):
            matches = [r for r in episodes if r['policy'] == name and
                       int(r['scenario_seed']) == int(row['scenario_seed']) and
                       (name != 'centered' or str(r['training_seed']) == str(row['training_seed']))]
            if len(matches) != 1:
                raise ValueError('unique paired baseline required')
            result.append({'training_seed': row['training_seed'],
                           'scenario_seed': row['scenario_seed'], 'comparison_policy': name,
                           **{k: float(row[k]) - float(matches[0][k]) for k in METRICS}})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('full published source revision required')
    config, index, data, parent_checks = load_parent(args.parent_root)
    args.out.mkdir(parents=True, exist_ok=False)
    source = sorted((ROOT / 'src').rglob('*.py')) + [ROOT / 'scripts' / name for name in
              ('run_scale_loss.py', 'audit_scale_loss.py', 'run_scale_pilot.py',
               'run_centered_inspection.py', 'run_mission_validation.py', 'audit_scale_pilot.py')]
    source.append(ROOT / 'docs/scale-loss-protocol.md')
    source += [ROOT / 'tests/test_scale_loss.py', ROOT / '.github/workflows/scale-loss.yml']
    parent_files = [PARENT / n for n in ('launch.json', 'checkpoints.json', 'datasets.json',
                    'sampling.json', 'rankings/per-fit.csv', 'missions/episodes.csv',
                    'inputs.json.gz', 'completion.json', 'audit.json', 'checksums.json')]
    parent_files += [EXACT / n for n in
                     ('episodes.csv', 'audit.json', 'launch.json', 'checksums.json')]
    atomic_json_save({'started_utc': datetime.now(timezone.utc).isoformat(),
                      'source_revision': args.source_revision, 'env_config': config.to_dict(),
                      'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in source},
                      'parent_input_sha256': {str(p.relative_to(ROOT)): digest(p)
                                              for p in parent_files},
                      'training_seeds': [0, 1, 2], 'updates_per_fit': UPDATES,
                      'batch_size': BATCH_SIZE, 'learning_rate': .001, 'clip': .5,
                      'cpu_threads': 1, 'method': METHOD, 'temperature': 1,
                      'mission_worlds': list(WORLDS), 'new_fits': 3, 'new_worlds': 0,
                      'rl_updates': 0, 'regression_control': 'audited archived matched fits'},
                     args.out / 'launch.json')
    started, checkpoints, sampling = time.perf_counter(), {}, {}
    for seed in range(3):
        _, identity, history = fit_ranking(
            config, data['training'], seed=seed, updates=UPDATES, batch_size=BATCH_SIZE,
            output=args.out / f'seed-{seed}',
            parent_initial=args.parent_root / f'seed-{seed}/initial.pt',
            expected_sequence=parent_checks[str(seed)]['sample_sequence_sha256'])
        identity['training_dataset_sha256'] = digest(args.parent_root / 'training.pt')
        checkpoints[str(seed)] = identity
        sampling[str(seed)] = sample_audit(data['training'], seed)
        write_rows(args.out / f'seed-{seed}/training.csv', history)
    atomic_json_save(checkpoints, args.out / 'checkpoints.json')
    atomic_json_save(sampling, args.out / 'sampling.json')
    rank_dir, mission_dir = args.out / 'rankings', args.out / 'missions'
    rank_dir.mkdir()
    mission_dir.mkdir()
    natural, ranked = [], []
    for seed in range(3):
        for phase in ('initial', 'final'):
            model = load_centered_model(args.out / f'seed-{seed}/{phase}.pt', config, 'centered')
            scores = {}
            for split in (('fresh',) if phase == 'initial' else ('training', 'fresh')):
                scores[split] = score_model(model, data[split])
                rows = metrics(scores[split], data[split], index[split], split=split, seed=seed,
                               phase=phase, policy='network')
                natural.extend(rows)
                ranked.append({'split': split, 'seed': seed, 'phase': phase, 'policy': 'network',
                               **summary(rows)})
            np.savez_compressed(rank_dir / f'seed-{seed}-{phase}.npz', **scores)
    write_rows(rank_dir / 'natural.csv', natural)
    write_rows(rank_dir / 'per-fit.csv', ranked)
    episodes, traces = [], []
    for seed in range(3):
        model = load_centered_model(args.out / f'seed-{seed}/final.pt', config, 'centered')
        scores, masks = [], []
        for world in WORLDS:
            row, trace, vectors = rollout(config, ScorePolicy(config, model=model), world,
                                          name='ranking', training_seed=seed)
            episodes.append(row)
            traces.extend(trace)
            scores.append(vectors['scores'])
            masks.append(vectors['masks'])
            print(f'seed {seed}: mission {world} complete', flush=True)
        np.savez_compressed(mission_dir / f'ranking-seed-{seed}-scores.npz',
                            scores=np.concatenate(scores), masks=np.concatenate(masks))
        write_rows(mission_dir / 'episodes.csv', episodes)
    with gzip.open(mission_dir / 'steps.csv.gz', 'wt', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    controls = [r for r in read(PARENT / 'missions/episodes.csv')
                if r['policy'] in ('centered', 'greedy', 'masked_random')]
    # Normalize archive identities/numerics before production gate comparisons.
    controls += read(EXACT / 'episodes.csv')
    for row in controls:
        row['scenario_seed'] = int(row['scenario_seed'])
        row['training_seed'] = int(row['training_seed']) if row['training_seed'] else None
        for k in METRICS:
            row[k] = float(row[k])
    all_episodes = episodes + controls
    write_rows(args.out / 'comparison.csv', comparisons(all_episodes))
    write_rows(args.out / 'summary.csv', aggregate(all_episodes))
    write_rows(args.out / 'paired.csv', paired(all_episodes))
    criteria = loss_decisions(ranked, read(PARENT / 'rankings/per-fit.csv'), all_episodes)
    completion = {'new_fits': 3, 'new_worlds': 0, 'rl_updates': 0,
                  'new_mission_episodes': len(episodes), 'new_mission_steps': len(traces),
                  'effect_passed': all(r['effect_passed'] for r in criteria),
                  'ranking_passed': all(r['ranking_passed'] for r in criteria),
                  'mission_passed': all(r['mission_passed'] for r in criteria),
                  'per_seed': criteria, 'elapsed_seconds': time.perf_counter() - started}
    atomic_json_save(completion, args.out / 'completion.json')
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    main()
