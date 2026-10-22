"""Execute the frozen all-action inspection capability protocol, without RL."""

import argparse
import copy
import gzip
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.experiments import write_rows
from wild_drone.inspection_targets import (
    collect_states,
    fit_targets,
    load_target_model,
    pack_states,
    score_model,
    state_data,
    validate_worlds,
)
from wild_drone.masked_learning import MaskedRandomPolicy
from wild_drone.ranking_probe import (
    pair_metrics,
    position_pair,
    ranking_metrics,
    transition_coverage,
    urgency_pair,
)
from wild_drone.training_io import atomic_json_save, atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
TRAIN_WORLDS = tuple(range(40))
HELDOUT_WORLDS = tuple(range(6000, 6016))
STEPS = (16, 32, 64, 96)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zipped_json(value, path):
    path.write_bytes(gzip.compress(json.dumps(value, sort_keys=True,
                                            separators=(',', ':')).encode(), mtime=0))


def collect_pairs(config):
    states, records, pairs, coverage = [], [], [], []
    for world in range(6000, 6008):
        for collector in ('masked_random', 'greedy'):
            env = WildfireFleetEnv(config)
            observation, _ = env.reset(seed=world)
            controller = MaskedRandomPolicy(world) if collector == 'masked_random' else \
                make_policy('greedy')
            for step in range(65):
                if step in (16, 64):
                    case = f'{world}-{collector}-{step}'
                    cov = transition_coverage(env)
                    urgency, position = urgency_pair(env, cov), position_pair(env, cov)
                    variants = {'natural': (env, cov), 'noop': (copy.deepcopy(env), cov)}
                    details = {}
                    if urgency:
                        variants['urgency_a'] = (urgency[0], cov)
                        variants['urgency_b'] = (urgency[1], cov)
                        details['urgency'] = urgency[2]
                    if position:
                        variants['position'] = (position[0], position[1])
                        details['position'] = position[2]
                    offsets = {}
                    for label, (state, footprint) in variants.items():
                        entry = state_data(state, footprint)
                        offsets[label] = len(states)
                        states.append(entry)
                        records.append({'case': f'{case}:{label}',
                                        'scenario_seed': world, 'collector': collector,
                                        'step': step, 'observation_sha256': entry[4]})
                    kinds = [('noop', 'natural', 'noop')]
                    if urgency:
                        kinds.append(('urgency', 'urgency_a', 'urgency_b'))
                    if position:
                        kinds.append(('position', 'natural', 'position'))
                    for kind, before, after in kinds:
                        ai, bi = offsets[before], offsets[after]
                        if np.count_nonzero(states[ai][3] & states[bi][3]) >= 2:
                            pairs.append({'case': case, 'kind': kind, 'before': ai, 'after': bi})
                    feasible = int(states[offsets['natural']][3].sum())
                    coverage.append({'case': case, 'natural_feasible_actions': feasible,
                                     'urgency_eligible': urgency is not None,
                                     'position_eligible': position is not None,
                                     'noop_eligible': feasible >= 2, 'details': details,
                                     'urgency_skip_reason': '' if urgency else
                                     'fewer than two feasible actions' if feasible < 2 else
                                     'no balanced disjoint optimum in top64 proposal pairs',
                                     'position_skip_reason': '' if position else
                                     'fewer than two feasible actions' if feasible < 2 else
                                     'no valid one-cell move with shared disjoint optimum'})
                if step == 64:
                    break
                observation, _, terminal, truncated, _ = env.step(controller.act(env, observation))
                if terminal or truncated:
                    coverage.extend({'case': f'{world}-{collector}-{s}',
                                     'missing_reason': 'collector terminated before snapshot'}
                                    for s in (16, 64) if s > step)
                    break
    return pack_states(states), records, pairs, coverage


def metrics(scores, data, records, **identity):
    rows = []
    for i, record in enumerate(records):
        mask, gains = data['masks'][i].numpy(), data['gains'][i].numpy()
        regression = None if identity.get('policy') == 'inspection_reference' else float(
            np.mean((scores[i][mask]-data['targets'][i].numpy()[mask])**2))
        rows.append(record | identity | {'masked_regression_error': regression,
                    **ranking_metrics(scores[i], gains, np.flatnonzero(mask))})
    return rows


def summary(rows):
    informative = [r for r in rows if r['normalised_gain_regret'] is not None]
    fields = ('normalised_gain_regret', 'one_step_optimal', 'masked_regression_error',
              'random_expected_normalised_regret', 'random_best_probability')
    return {'states': len(rows), 'informative_states': len(informative),
            **{key: float(np.mean([r[key] for r in informative if r[key] is not None]))
               if any(r[key] is not None for r in informative) else None
               for key in fields}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    validate_worlds(TRAIN_WORLDS, HELDOUT_WORLDS)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('full published source revision required')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = EnvConfig(grid_size=20, horizon=160, battery_capacity=80, recharge_rate=5,
                       sensor_radius=1, inspection_deadline=25, max_fuel=100)
    sources = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve(),
                                                               ROOT/'docs'/
                                                               'inspection-target-protocol.md']
    atomic_json_save({'started_utc': datetime.now(timezone.utc).isoformat(),
                      'source_revision': args.source_revision,
                      'source_revision_kind': 'operator-verified published source snapshot',
                      'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                      'env_config': config.to_dict(), 'training_worlds': list(TRAIN_WORLDS),
                      'heldout_worlds': list(HELDOUT_WORLDS), 'snapshot_steps': list(STEPS),
                      'training_seeds': [0, 1, 2], 'updates_per_fit': 2000,
                      'batch_size': 64, 'learning_rate': .001, 'cpu_threads': 1,
                      'rl_updates': False}, args.out/'launch.json')
    started = time.perf_counter()
    datasets, records, missing = {}, {}, {}
    for split, worlds in [('training', TRAIN_WORLDS), ('heldout', HELDOUT_WORLDS)]:
        datasets[split], records[split], missing[split] = collect_states(config, worlds, STEPS)
        print(f'{split}: {len(records[split])} states', flush=True)
    datasets['probes'], records['probes'], pairs, coverage = collect_pairs(config)
    observed = {r['case']: r['observation_sha256'] for r in records['heldout']}
    assert all(observed[r['case'].removesuffix(':natural')] == r['observation_sha256']
               for r in records['probes'] if r['case'].endswith(':natural'))
    index = {split: [record | {'gains': data['gains'][i].tolist(),
                               'mask': data['masks'][i].tolist()}
                    for i, record in enumerate(records[split])]
             for split, data in datasets.items()}
    zipped_json(index, args.out/'inputs.json.gz')
    atomic_json_save({'pairs': pairs, 'coverage': coverage}, args.out/'probe-plan.json')
    data_identities = {}
    for split, data in datasets.items():
        path = args.out/f'{split}.pt'
        atomic_torch_save(data, path)
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(value, loaded[key]) for key, value in data.items())
        data_identities[split] = {'sha256': digest(path), 'bytes': path.stat().st_size,
                                 'states': len(data['maps']),
                                 'informative': int(data['informative'].sum())}
    atomic_json_save(data_identities, args.out/'datasets.json')
    models, checkpoints = {}, {}
    for seed in (0, 1, 2):
        model, identity, history, _ = fit_targets(
            config, datasets['training'], seed=seed, updates=2000, batch_size=64,
            output=args.out/f'seed-{seed}')
        models[seed] = model
        checkpoints[str(seed)] = identity
        write_rows(args.out/f'seed-{seed}'/'training.csv', history)
        print(f'seed {seed}: fitting complete', flush=True)
    atomic_json_save(checkpoints, args.out/'checkpoints.json')
    train = datasets['training']
    masks = train['masks'] & train['informative'][:, None]
    action_mean = (train['targets']*masks).sum(0)/masks.sum(0).clamp(min=1)
    atomic_json_save({'scores': action_mean.tolist(),
                      'training_action_counts': masks.sum(0).tolist()},
                     args.out/'action-mean-control.json')
    rows, paired, per_fit = [], [], []
    for seed in (0, 1, 2):
        for phase in ('initial', 'final'):
            model = load_target_model(args.out/f'seed-{seed}'/f'{phase}.pt', config)
            predictions = {split: score_model(model, data) for split, data in datasets.items()}
            zipped_json({split: values.tolist() for split, values in predictions.items()},
                        args.out/f'seed-{seed}'/f'{phase}-scores.json.gz')
            for split in ('training', 'heldout'):
                measured = metrics(predictions[split], datasets[split], records[split],
                                   seed=seed, phase=phase, split=split, policy='network')
                rows.extend(measured)
                per_fit.append({'seed': seed, 'phase': phase, 'split': split,
                                'policy': 'network', **summary(measured)})
            probe = datasets['probes']
            for pair in pairs:
                a, b = pair['before'], pair['after']
                result = pair_metrics(predictions['probes'][a], predictions['probes'][b],
                                      probe['gains'][a].numpy(), probe['gains'][b].numpy(),
                                      probe['masks'][a].numpy(), probe['masks'][b].numpy())
                if pair['kind'] == 'noop' and result['score_delta_max'] != 0:
                    raise ValueError('identical-state scores changed')
                paired.append(pair | {'seed': seed, 'phase': phase, 'policy': 'network', **result})
    for split in ('training', 'heldout'):
        data = datasets[split]
        for name, scores in [('action_mean', np.tile(action_mean.numpy(), (len(data['maps']), 1))),
                             ('inspection_reference', data['gains'].numpy())]:
            measured = metrics(scores, data, records[split], seed=None, phase='control',
                               split=split, policy=name)
            rows.extend(measured)
            per_fit.append({'seed': None, 'phase': 'control', 'split': split,
                            'policy': name, **summary(measured)})
    probe = datasets['probes']
    for name, scores in [('action_mean', np.tile(action_mean.numpy(), (len(probe['maps']), 1))),
                         ('inspection_reference', probe['gains'].numpy())]:
        for pair in pairs:
            a, b = pair['before'], pair['after']
            result = pair_metrics(scores[a], scores[b], probe['gains'][a].numpy(),
                                  probe['gains'][b].numpy(), probe['masks'][a].numpy(),
                                  probe['masks'][b].numpy())
            paired.append(pair | {'seed': None, 'phase': 'control', 'policy': name, **result})
    write_rows(args.out/'natural.csv', rows)
    write_rows(args.out/'pairs.csv', paired)
    write_rows(args.out/'per-fit.csv', per_fit)
    control = next(r for r in per_fit if r['policy'] == 'action_mean' and r['split'] == 'heldout')
    decisions = []
    for seed in (0, 1, 2):
        result = {r['phase']+'-'+r['split']: r for r in per_fit if r['seed'] == seed}
        train, heldout = result['final-training'], result['final-heldout']
        initial = result['initial-heldout']
        checks = {'training_regret_le_005': train['normalised_gain_regret'] <= .05,
                  'heldout_regret_le_020': heldout['normalised_gain_regret'] <= .20,
                  'improves_initial_ge_015': initial['normalised_gain_regret'] -
                  heldout['normalised_gain_regret'] >= .15,
                  'improves_action_mean_ge_015': control['normalised_gain_regret'] -
                  heldout['normalised_gain_regret'] >= .15,
                  'heldout_best_match_ge_025': heldout['one_step_optimal'] >= .25}
        decisions.append({'seed': seed, 'checks': checks, 'passed': all(checks.values())})
    final = [r for r in per_fit if r['phase'] == 'final' and r['split'] == 'heldout']
    report = {'capability_criterion_passed': all(r['passed'] for r in decisions),
              'per_seed_criterion': decisions,
              'heldout_fit_mean_sd': {key: {'mean': float(np.mean([r[key] for r in final])),
                                           'sample_sd': float(np.std(
                                               [r[key] for r in final], ddof=1))}
                                     for key in ('normalised_gain_regret', 'one_step_optimal',
                                                 'masked_regression_error')},
              'missing_snapshots': missing, 'planned_training_states': 320,
              'planned_heldout_states': 128, 'observed_states':
              {k: len(v) for k, v in records.items()},
              'probe_coverage_cases': len(coverage),
              'probe_pair_counts': {kind: sum(p['kind'] == kind for p in pairs)
                                    for kind in ('noop', 'urgency', 'position')},
              'elapsed_seconds': time.perf_counter()-started,
              'representation_changed': False, 'rl_or_mission_improvement_claim': False}
    atomic_json_save(report, args.out/'completion.json')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
