"""Frozen matched current/absolute/drone-centered inspection capability probe."""

import argparse
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wild_drone.centered_inspection import ARMS, fit_centered, load_centered_model
from wild_drone.config import EnvConfig
from wild_drone.experiments import write_rows
from wild_drone.inspection_targets import collect_states, score_model, validate_worlds
from wild_drone.ranking_probe import pair_metrics, ranking_metrics
from wild_drone.training_io import atomic_json_save, atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
FRESH_WORLDS = tuple(range(6200, 6216))
STEPS = (16, 32, 64, 96)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zipped_json(value, path):
    path.write_bytes(gzip.compress(json.dumps(value, sort_keys=True,
                                            separators=(',', ':')).encode(), mtime=0))


def load_inputs(parent):
    archive = ROOT/'docs'/'experiments'/'inspection-targets-2026-10-10'
    launch = json.loads((archive/'launch.json').read_text())
    config = EnvConfig(grid_size=20, horizon=160, battery_capacity=80, recharge_rate=5,
                       sensor_radius=1, inspection_deadline=25, max_fuel=100)
    if launch['env_config'] != config.to_dict():
        raise ValueError('parent environment differs from frozen recipe')
    identities = json.loads((archive/'datasets.json').read_text())
    index = json.loads(gzip.decompress((archive/'inputs.json.gz').read_bytes()))
    data = {}
    for split, identity in identities.items():
        path = parent/f'{split}.pt'
        if path.stat().st_size != identity['bytes'] or digest(path) != identity['sha256']:
            raise ValueError('parent dataset differs from archived identity')
        data[split] = torch.load(path, map_location='cpu', weights_only=True)
        if len(index[split]) != len(data[split]['maps']):
            raise ValueError('parent case count differs')
        for i, record in enumerate(index[split]):
            if (record['gains'] != data[split]['gains'][i].tolist() or
                    record['mask'] != data[split]['masks'][i].tolist()):
                raise ValueError('parent case/gain/mask identities differ')
    checkpoints = json.loads((archive/'checkpoints.json').read_text())
    for seed in (0, 1, 2):
        for label in ('initial', 'final'):
            path = parent/f'seed-{seed}'/f'{label}.pt'
            identity = checkpoints[str(seed)]['checkpoints'][path.name]
            if path.stat().st_size != identity['bytes'] or digest(path) != identity['sha256']:
                raise ValueError('parent current-arm checkpoint differs')
    validate_worlds(launch['training_worlds'], FRESH_WORLDS)
    if set(FRESH_WORLDS) & set(launch['heldout_worlds']):
        raise ValueError('fresh and reused development worlds overlap')
    return config, data, index, identities, json.loads((archive/'probe-plan.json').read_text())


def metrics(scores, data, records, **identity):
    rows = []
    for i, record in enumerate(records):
        gains, mask = data['gains'][i].numpy(), data['masks'][i].numpy()
        regression = None if identity.get('policy') == 'inspection_reference' else float(
            np.mean((scores[i][mask]-data['targets'][i].numpy()[mask])**2))
        metadata = {k: v for k, v in record.items() if k not in ('gains', 'mask')}
        rows.append(metadata | identity | {'masked_regression_error': regression,
                    **ranking_metrics(scores[i], gains, np.flatnonzero(mask))})
    return rows


def summary(rows):
    informative = [r for r in rows if r['normalised_gain_regret'] is not None]
    fields = ('normalised_gain_regret', 'one_step_optimal', 'masked_regression_error',
              'random_expected_normalised_regret', 'random_best_probability')
    return {'states': len(rows), 'informative_states': len(informative),
            **{key: float(np.mean([r[key] for r in informative if r[key] is not None]))
               if any(r[key] is not None for r in informative) else None for key in fields}}


def measure_pairs(scores, data, plan, **identity):
    rows = []
    for pair in plan['pairs']:
        a, b = pair['before'], pair['after']
        values = pair_metrics(scores[a], scores[b], data['gains'][a].numpy(),
                              data['gains'][b].numpy(), data['masks'][a].numpy(),
                              data['masks'][b].numpy())
        if pair['kind'] == 'noop' and values['score_delta_max'] != 0:
            raise ValueError('identical-state predictions changed')
        rows.append(pair | identity | values)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    config, data, index, identities, plan = load_inputs(args.parent_root)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('full published source revision required')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve(),
            ROOT/'docs'/'centered-inspection-protocol.md']
    atomic_json_save({'started_utc': datetime.now(timezone.utc).isoformat(),
                      'source_revision': args.source_revision,
                      'source_revision_kind': 'operator-verified published source snapshot',
                      'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
                      'parent_source_revision': '2ff35318b67ceaa8a7bd186b85650095771e083d',
                      'env_config': config.to_dict(), 'arms': list(ARMS),
                      'training_seeds': [0, 1, 2],
                      'updates_per_fit': 2000, 'batch_size': 64, 'learning_rate': .001,
                      'fresh_worlds': list(FRESH_WORLDS), 'snapshot_steps': list(STEPS),
                      'cpu_threads': 1, 'rl_updates': False}, args.out/'launch.json')
    started = time.perf_counter()
    data['fresh'], records, missing = collect_states(config, FRESH_WORLDS, STEPS)
    index['fresh'] = [record | {'gains': data['fresh']['gains'][i].tolist(),
                               'mask': data['fresh']['masks'][i].tolist()}
                      for i, record in enumerate(records)]
    fresh_path = args.out/'fresh.pt'
    atomic_torch_save(data['fresh'], fresh_path)
    restored = torch.load(fresh_path, weights_only=True)
    assert all(torch.equal(value, restored[k]) for k, value in data['fresh'].items())
    identities['fresh'] = {'sha256': digest(fresh_path), 'bytes': fresh_path.stat().st_size,
                           'states': len(records),
                           'informative': int(data['fresh']['informative'].sum())}
    atomic_json_save(identities, args.out/'datasets.json')
    zipped_json(index, args.out/'inputs.json.gz')
    atomic_json_save(plan, args.out/'probe-plan.json')
    print(f'fresh: {len(records)} states; fitting fixed nine arms/seeds', flush=True)
    checkpoints, matching = {}, []
    for seed in (0, 1, 2):
        initial, samplers, parameters = {}, set(), set()
        for arm in ARMS:
            model, identity, history, tensors = fit_centered(
                config, data['training'], arm=arm, seed=seed, updates=2000, batch_size=64,
                output=args.out/arm/f'seed-{seed}')
            initial[arm] = tensors
            samplers.add(identity['sample_sequence_sha256'])
            parameters.add(identity['parameters'])
            checkpoints[f'{arm}/{seed}'] = identity
            write_rows(args.out/arm/f'seed-{seed}'/'training.csv', history)
            if arm == 'current':
                saved = torch.load(args.parent_root/f'seed-{seed}'/'final.pt', weights_only=True)
                if any(not torch.equal(v, saved['state_dict'][k])
                       for k, v in model.state_dict().items()):
                    raise ValueError('matched current tensors differ from prior fit')
                import csv
                previous = list(csv.DictReader((args.parent_root/f'seed-{seed}'/
                                                'training.csv').open()))
                if any(int(r['update']) != h['update'] or float(r['training_batch_loss']) !=
                       h['training_batch_loss'] for r, h in zip(previous, history, strict=True)):
                    raise ValueError('matched current training history differs')
            print(f'{arm}/seed-{seed}: fitting complete', flush=True)
        if len(samplers) != 1 or len(parameters) != 1 or any(
                not torch.equal(v, initial[arm][k]) for arm in ARMS
                for k, v in initial['current'].items()):
            raise ValueError('arm initialization/size/sampling differs')
        matching.append({'seed': seed, 'initial_tensors_exactly_equal': True,
                         'sample_sequence_sha256': next(iter(samplers)),
                         'parameters_per_arm': next(iter(parameters)),
                         'current_final_and_training_history_match_parent': True})
    atomic_json_save(checkpoints, args.out/'checkpoints.json')
    atomic_json_save(matching, args.out/'matching.json')
    rows, paired, per_fit = [], [], []
    for arm in ARMS:
        for seed in (0, 1, 2):
            for phase in ('initial', 'final'):
                model = load_centered_model(args.out/arm/f'seed-{seed}'/f'{phase}.pt', config, arm)
                scores = {split: score_model(model, dataset) for split, dataset in data.items()}
                np.savez_compressed(args.out/arm/f'seed-{seed}'/f'{phase}-scores.npz', **scores)
                for split in ('training', 'heldout', 'fresh'):
                    measured = metrics(scores[split], data[split], index[split], arm=arm,
                                       seed=seed, phase=phase, split=split, policy='network')
                    rows.extend(measured)
                    per_fit.append({'arm': arm, 'seed': seed, 'phase': phase, 'split': split,
                                    'policy': 'network', **summary(measured)})
                paired.extend(measure_pairs(scores['probes'], data['probes'], plan,
                                            arm=arm, seed=seed, phase=phase, policy='network'))
    train = data['training']
    masks = train['masks'] & train['informative'][:, None]
    action_mean = (train['targets']*masks).sum(0)/masks.sum(0).clamp(min=1)
    atomic_json_save({'scores': action_mean.tolist(),
                     'training_action_counts': masks.sum(0).tolist()},
                     args.out/'action-mean-control.json')
    for name in ('action_mean', 'inspection_reference'):
        for split in ('training', 'heldout', 'fresh'):
            scores = np.tile(action_mean.numpy(), (len(data[split]['maps']), 1)) if \
                name == 'action_mean' else data[split]['gains'].numpy()
            measured = metrics(scores, data[split], index[split], arm='shared', seed=None,
                               phase='control', split=split, policy=name)
            rows.extend(measured)
            per_fit.append({'arm': 'shared', 'seed': None, 'phase': 'control', 'split': split,
                            'policy': name, **summary(measured)})
        scores = np.tile(action_mean.numpy(), (len(data['probes']['maps']), 1)) if \
            name == 'action_mean' else data['probes']['gains'].numpy()
        paired.extend(measure_pairs(scores, data['probes'], plan, arm='shared', seed=None,
                                    phase='control', policy=name))
    write_rows(args.out/'natural.csv', rows)
    write_rows(args.out/'pairs.csv', paired)
    write_rows(args.out/'per-fit.csv', per_fit)
    comparisons, criterion = [], []
    for seed in (0, 1, 2):
        found = {(r['arm'], r['split']): r for r in per_fit
                 if r['seed'] == seed and r['phase'] == 'final'}
        centered = found['centered', 'fresh']
        for control in ('current', 'absolute'):
            comparisons.append({'seed': seed, 'control': control,
                                'fresh_regret_difference_centered_minus_control':
                                centered['normalised_gain_regret'] -
                                found[control, 'fresh']['normalised_gain_regret'],
                                'fresh_best_match_difference_centered_minus_control':
                                centered['one_step_optimal'] -
                                found[control, 'fresh']['one_step_optimal']})
        checks = {'training_regret_le_005': found['centered', 'training'][
                  'normalised_gain_regret'] <= .05,
                  'fresh_regret_le_020': centered['normalised_gain_regret'] <= .20,
                  'fresh_best_match_ge_025': centered['one_step_optimal'] >= .25,
                  **{f'improves_{control}_ge_010':
                     found[control, 'fresh']['normalised_gain_regret'] -
                     centered['normalised_gain_regret'] >= .10
                     for control in ('current', 'absolute')}}
        criterion.append({'seed': seed, 'checks': checks, 'passed': all(checks.values())})
    aggregate = {}
    for arm in ARMS:
        selected = [r for r in per_fit if r['arm'] == arm and r['split'] == 'fresh'
                    and r['phase'] == 'final']
        aggregate[arm] = {key: {'mean': float(np.mean([r[key] for r in selected])),
                               'sample_sd': float(np.std([r[key] for r in selected], ddof=1))}
                          for key in ('normalised_gain_regret', 'one_step_optimal',
                                      'masked_regression_error')}
    atomic_json_save({'capability_criterion_passed': all(r['passed'] for r in criterion),
                      'per_seed_criterion': criterion, 'fresh_fit_mean_sd': aggregate,
                      'paired_comparisons': comparisons, 'planned_fresh_states': 128,
                      'observed_fresh_states': len(records), 'missing_fresh_snapshots': missing,
                      'elapsed_seconds': time.perf_counter()-started,
                      'rl_or_mission_improvement_claim': False}, args.out/'completion.json')
    print(json.dumps({'criterion': criterion, 'fresh': aggregate}, indent=2), flush=True)


if __name__ == '__main__':
    main()
