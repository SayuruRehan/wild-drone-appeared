"""Frozen nested 40-versus-200-world centered inspection fitting comparison."""

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from run_centered_inspection import digest, load_inputs, metrics, summary, zipped_json

from wild_drone.centered_inspection import fit_centered, load_centered_model
from wild_drone.experiments import write_rows
from wild_drone.inspection_targets import collect_states, score_model, validate_worlds
from wild_drone.training_io import atomic_json_save, atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
SMALL_WORLDS = tuple(range(40))
EXPANDED_WORLDS = tuple(range(200))
FRESH_WORLDS = tuple(range(6400, 6416))
REUSED_WORLDS = tuple(range(6200, 6216))
STEPS = (16, 32, 64, 96)
UPDATES, BATCH_SIZE = 2000, 64


def validate_recipe(small, expanded, fresh, reused):
    if tuple(expanded[:len(small)]) != tuple(small) or len(expanded) <= len(small):
        raise ValueError('expanded training worlds must strictly extend the small prefix')
    validate_worlds(small, fresh)
    validate_worlds(expanded, fresh)
    validate_worlds(expanded, reused)
    validate_worlds(fresh, reused)


def extend_data(small, more, records, other):
    if set(r['scenario_seed'] for r in records) & set(r['scenario_seed'] for r in other):
        raise ValueError('appended training worlds overlap the original corpus')
    if small.keys() != more.keys():
        raise ValueError('appended tensor fields differ')
    expanded = {key: torch.cat((tensor, more[key])) for key, tensor in small.items()}
    return expanded, records+other


def criteria(rows):
    result = []
    for seed in (0, 1, 2):
        found = {(r['corpus'], r['split']): r for r in rows
                 if r['seed'] == seed and r['phase'] == 'final' and r['policy'] == 'network'}
        small, expanded = found['small', 'fresh'], found['expanded', 'fresh']
        effect = {'fresh_regret_improves_ge_005': small['normalised_gain_regret']-
                  expanded['normalised_gain_regret'] >= .05,
                  'fresh_best_match_not_lower': expanded['one_step_optimal'] >=
                  small['one_step_optimal']}
        capability = {'training_regret_le_005': found['expanded', 'training'][
                      'normalised_gain_regret'] <= .05,
                      'fresh_regret_le_020': expanded['normalised_gain_regret'] <= .20,
                      'fresh_best_match_ge_025': expanded['one_step_optimal'] >= .25}
        result.append({'seed': seed, 'data_effect_checks': effect,
                       'data_effect_passed': all(effect.values()),
                       'capability_checks': capability,
                       'capability_passed': all(capability.values())})
    return result


def load_prior(parent, centered):
    config, data, index, _, _ = load_inputs(parent)
    archive = ROOT/'docs'/'experiments'/'centered-inspection-2026-10-10'
    launch = json.loads((archive/'launch.json').read_text())
    if config.to_dict() != launch['env_config'] or launch['fresh_worlds'] != list(REUSED_WORLDS):
        raise ValueError('centered parent configuration/worlds differ')
    for path, expected in launch['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError('fixed parent production source differs')
    identities = json.loads((archive/'datasets.json').read_text())
    path = centered/'fresh.pt'
    if digest(path) != identities['fresh']['sha256'] or path.stat().st_size != \
            identities['fresh']['bytes']:
        raise ValueError('reused evaluation dataset differs from parent identity')
    reused = torch.load(path, map_location='cpu', weights_only=True)
    import gzip
    records = json.loads(gzip.decompress((archive/'inputs.json.gz').read_bytes()))['fresh']
    if len(records) != len(reused['maps']):
        raise ValueError('reused state count differs')
    for i, record in enumerate(records):
        if record['gains'] != reused['gains'][i].tolist() or \
                record['mask'] != reused['masks'][i].tolist():
            raise ValueError('reused state labels/masks differ')
    checkpoints = json.loads((archive/'checkpoints.json').read_text())
    checksums = json.loads((archive/'checksums.json').read_text())
    for seed in (0, 1, 2):
        directory = centered/'centered'/f'seed-{seed}'
        for phase in ('initial', 'final'):
            p = directory/f'{phase}.pt'
            identity = checkpoints[f'centered/{seed}']['checkpoints'][p.name]
            if digest(p) != identity['sha256'] or p.stat().st_size != identity['bytes']:
                raise ValueError('parent centered checkpoint differs')
        relative = f'centered/seed-{seed}/training.csv'
        if digest(directory/'training.csv') != checksums[relative]['sha256']:
            raise ValueError('parent centered training history differs')
    return config, data['training'], reused, index['training'], records


def sample_audit(data, seed):
    ids = torch.where(data['informative'])[0]
    generator = torch.Generator().manual_seed(seed+1000)
    counts, sequence = torch.zeros(len(data['maps']), dtype=torch.int64), hashlib.sha256()
    for _ in range(UPDATES):
        batch = ids[torch.randint(len(ids), (BATCH_SIZE,), generator=generator)]
        sequence.update(batch.numpy().astype('<i8').tobytes())
        counts += torch.bincount(batch, minlength=len(counts))
    return {'sample_sequence_sha256': sequence.hexdigest(),
            'sampled_distinct_states': int((counts > 0).sum()),
            'informative_states': len(ids), 'sample_draws': UPDATES*BATCH_SIZE,
            'draws_per_informative_state': UPDATES*BATCH_SIZE/len(ids),
            'per_state_counts': counts.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--centered-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('full published source revision required')
    validate_recipe(SMALL_WORLDS, EXPANDED_WORLDS, FRESH_WORLDS, REUSED_WORLDS)
    config, small, reused, small_index, reused_index = load_prior(
        args.parent_root, args.centered_root)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = sorted((ROOT/'src').rglob('*.py')) + [Path(__file__).resolve(),
             ROOT/'scripts'/'run_centered_inspection.py',
             ROOT/'docs'/'training-coverage-protocol.md']
    launch = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'source_revision': args.source_revision,
              'source_revision_kind': 'operator-verified published source snapshot',
              'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
              'parent_source_revision': '2cebf052f6fc6f0a4a3b57c87c41591ac4a421db',
              'env_config': config.to_dict(), 'representation': 'centered',
              'small_worlds': list(SMALL_WORLDS), 'expanded_worlds': list(EXPANDED_WORLDS),
              'fresh_worlds': list(FRESH_WORLDS), 'reused_worlds': list(REUSED_WORLDS),
              'snapshot_steps': list(STEPS), 'training_seeds': [0, 1, 2],
              'updates_per_fit': UPDATES, 'batch_size': BATCH_SIZE, 'learning_rate': .001,
              'cpu_threads': 1, 'rl_updates': False}
    atomic_json_save(launch, args.out/'launch.json')
    started = time.perf_counter()
    more, more_index, missing_more = collect_states(config, EXPANDED_WORLDS[len(SMALL_WORLDS):],
                                                   STEPS)
    expanded, expanded_index = extend_data(small, more, small_index, more_index)
    print(f'expanded: {len(expanded_index)} states; collecting primary fresh worlds', flush=True)
    fresh, fresh_index, missing_fresh = collect_states(config, FRESH_WORLDS, STEPS)
    datasets = {'small': small, 'expanded': expanded, 'fresh': fresh, 'reused': reused}
    records = {'small': small_index, 'expanded': expanded_index,
               'fresh': fresh_index, 'reused': reused_index}
    identities, index = {}, {}
    for split, data in datasets.items():
        path = args.out/f'{split}.pt'
        atomic_torch_save(data, path)
        restored = torch.load(path, map_location='cpu', weights_only=True)
        assert all(torch.equal(v, restored[k]) for k, v in data.items())
        identities[split] = {'sha256': digest(path), 'bytes': path.stat().st_size,
                             'states': len(data['maps']),
                             'informative': int(data['informative'].sum())}
        index[split] = [record | {'gains': data['gains'][i].tolist(),
                                  'mask': data['masks'][i].tolist()}
                        for i, record in enumerate(records[split])]
    atomic_json_save(identities, args.out/'datasets.json')
    zipped_json(index, args.out/'inputs.json.gz')
    observed_small = {r['case'] for r in small_index}
    missing_small = [{'scenario_seed': world, 'collector': collector, 'step': step,
                      'reason': 'collector terminated before snapshot'}
                     for world in SMALL_WORLDS for collector in ('masked_random', 'greedy')
                     for step in STEPS if f'{world}-{collector}-{step}' not in observed_small]
    atomic_json_save({'original_training': missing_small,
                     'appended_training': missing_more, 'fresh': missing_fresh},
                     args.out/'missing-snapshots.json')
    print(f'fresh: {len(fresh_index)} states; fitting six corpus/seed combinations', flush=True)
    checkpoints, matching, exposures = {}, [], {}
    for seed in (0, 1, 2):
        initials, sizes = {}, set()
        for corpus in ('small', 'expanded'):
            directory = args.out/corpus/f'seed-{seed}'
            model, identity, history, initial = fit_centered(
                config, datasets[corpus], arm='centered', seed=seed, updates=UPDATES,
                batch_size=BATCH_SIZE, output=directory)
            initials[corpus] = initial
            sizes.add(identity['parameters'])
            identity.update(corpus=corpus, training_dataset_sha256=identities[corpus]['sha256'])
            checkpoints[f'{corpus}/{seed}'] = identity
            audit = sample_audit(datasets[corpus], seed)
            if audit['sample_sequence_sha256'] != identity['sample_sequence_sha256']:
                raise ValueError('fitting sample sequence differs from independent reconstruction')
            exposures[f'{corpus}/{seed}'] = audit
            write_rows(directory/'training.csv', history)
            if corpus == 'small':
                previous = args.centered_root/'centered'/f'seed-{seed}'
                saved = torch.load(previous/'final.pt', weights_only=True)
                if any(not torch.equal(v, saved['state_dict'][k])
                       for k, v in model.state_dict().items()):
                    raise ValueError('small final tensors differ from parent centered fit')
                old = list(csv.DictReader((previous/'training.csv').open()))
                new = list(csv.DictReader((directory/'training.csv').open()))
                if old != new:
                    raise ValueError('small training history differs from parent centered fit')
            print(f'{corpus}/seed-{seed}: fitting complete', flush=True)
        if len(sizes) != 1 or any(not torch.equal(v, initials['expanded'][k])
                                  for k, v in initials['small'].items()):
            raise ValueError('corpus initialization/model size differs')
        matching.append({'seed': seed, 'initial_tensors_exactly_equal': True,
                         'parameters_per_corpus': next(iter(sizes)),
                         'small_final_and_training_history_match_parent': True})
    atomic_json_save(checkpoints, args.out/'checkpoints.json')
    atomic_json_save(matching, args.out/'matching.json')
    atomic_json_save(exposures, args.out/'sampling.json')
    rows, per_fit = [], []

    def measure(scores, data_key, corpus, seed, phase, policy, split):
        measured = metrics(scores, datasets[data_key], index[data_key], corpus=corpus,
                           seed=seed, phase=phase, policy=policy, split=split, dataset=data_key)
        rows.extend(measured)
        per_fit.append({'corpus': corpus, 'seed': seed, 'phase': phase, 'split': split,
                        'dataset': data_key, 'policy': policy, **summary(measured)})

    (args.out/'initial').mkdir()
    for seed in (0, 1, 2):
        model = load_centered_model(args.out/'small'/f'seed-{seed}'/'initial.pt',
                                    config, 'centered')
        scores = {split: score_model(model, datasets[split]) for split in ('fresh', 'reused')}
        np.savez_compressed(args.out/'initial'/f'seed-{seed}-scores.npz', **scores)
        for split, predictions in scores.items():
            measure(predictions, split, 'shared', seed, 'initial', 'network', split)
        for corpus in ('small', 'expanded'):
            directory = args.out/corpus/f'seed-{seed}'
            model = load_centered_model(directory/'final.pt', config, 'centered')
            scores = {split: score_model(model, datasets[corpus if split == 'training' else split])
                      for split in ('training', 'fresh', 'reused')}
            np.savez_compressed(directory/'final-scores.npz', **scores)
            for split, predictions in scores.items():
                measure(predictions, corpus if split == 'training' else split, corpus,
                        seed, 'final', 'network', split)
    controls = {}
    for corpus in ('small', 'expanded'):
        train = datasets[corpus]
        masks = train['masks'] & train['informative'][:, None]
        means = (train['targets']*masks).sum(0)/masks.sum(0).clamp(min=1)
        controls[corpus] = {'scores': means.tolist(),
                            'training_action_counts': masks.sum(0).tolist()}
        for name in ('action_mean', 'inspection_reference'):
            for split in ('training', 'fresh', 'reused'):
                key = corpus if split == 'training' else split
                scores = np.tile(means.numpy(), (len(datasets[key]['maps']), 1)) if \
                    name == 'action_mean' else datasets[key]['gains'].numpy()
                measure(scores, key, corpus, None, 'control', name, split)
    atomic_json_save(controls, args.out/'action-mean-controls.json')
    write_rows(args.out/'natural.csv', rows)
    write_rows(args.out/'per-fit.csv', per_fit)
    criterion, aggregate, comparisons = criteria(per_fit), {}, []
    for corpus in ('small', 'expanded'):
        selected = [r for r in per_fit if r['corpus'] == corpus and r['split'] == 'fresh'
                    and r['phase'] == 'final']
        aggregate[corpus] = {key: {'mean': float(np.mean([r[key] for r in selected])),
                                  'sample_sd': float(np.std([r[key] for r in selected], ddof=1))}
                             for key in ('normalised_gain_regret', 'one_step_optimal',
                                         'masked_regression_error')}
    for seed in (0, 1, 2):
        found = {r['corpus']: r for r in per_fit if r['seed'] == seed and
                 r['phase'] == 'final' and r['split'] == 'fresh'}
        comparisons.append({'seed': seed, **{f'fresh_{key}_expanded_minus_small':
                            found['expanded'][key]-found['small'][key] for key in
                            ('normalised_gain_regret', 'one_step_optimal')}})
    completion = {'data_effect_passed': all(r['data_effect_passed'] for r in criterion),
                  'capability_criterion_passed': all(r['capability_passed'] for r in criterion),
                  'per_seed_criterion': criterion, 'fresh_fit_mean_sd': aggregate,
                  'paired_comparisons': comparisons,
                  'planned_expanded_states': len(EXPANDED_WORLDS)*2*len(STEPS),
                  'observed_expanded_states': len(expanded_index),
                  'planned_fresh_states': len(FRESH_WORLDS)*2*len(STEPS),
                  'observed_fresh_states': len(fresh_index),
                  'elapsed_seconds': time.perf_counter()-started,
                  'rl_or_mission_improvement_claim': False}
    atomic_json_save(completion, args.out/'completion.json')
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == '__main__':
    main()
