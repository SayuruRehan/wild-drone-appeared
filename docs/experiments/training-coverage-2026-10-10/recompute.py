"""Recompute all archived rankings without production metric/fitting functions."""

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def readzip(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def rank(scores, gains, ids):
    chosen = max(ids, key=lambda i: (scores[i], -i))
    high, low = max(gains[i] for i in ids), min(gains[i] for i in ids)
    ties = sum(abs(gains[i]-high) <= 1e-12 for i in ids)
    span = high-low
    regret = high-gains[chosen]
    return {'selected_action': chosen, 'selected_gain': gains[chosen],
            'feasible_actions': len(ids), 'reference_best_count': ties, 'gain_range': span,
            'gain_regret': regret, 'normalised_gain_regret': regret/span if span > 1e-12 else None,
            'one_step_optimal': abs(gains[chosen]-high) <= 1e-12,
            'random_best_probability': ties/len(ids), 'random_expected_normalised_regret':
            (high-sum(gains[i] for i in ids)/len(ids))/span if span > 1e-12 else None}


def compare(expected, actual):
    if expected is None:
        assert actual == ''
    elif isinstance(expected, bool):
        assert str(expected) == actual
    else:
        assert math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-11), (
            expected, actual)


def main():
    inputs = readzip(ROOT/'inputs.json.gz')
    raw = {('shared', str(seed), 'initial'): np.load(
        ROOT/'initial'/f'seed-{seed}-scores.npz', allow_pickle=False) for seed in (0, 1, 2)}
    raw.update({(corpus, str(seed), 'final'): np.load(
        ROOT/corpus/f'seed-{seed}'/'final-scores.npz', allow_pickle=False)
        for corpus in ('small', 'expanded') for seed in (0, 1, 2)})
    controls = json.loads((ROOT/'action-mean-controls.json').read_text())
    worlds = {key: {r['scenario_seed'] for r in records} for key, records in inputs.items()}
    assert worlds['small'] < worlds['expanded']
    assert not worlds['expanded'] & (worlds['fresh'] | worlds['reused'])
    assert not worlds['fresh'] & worlds['reused']
    assert inputs['expanded'][:len(inputs['small'])] == inputs['small']
    for corpus, control in controls.items():
        totals, counts = np.zeros(125), np.zeros(125, dtype=int)
        for record in inputs[corpus]:
            ids = [i for i in range(125) if record['mask'][i]]
            low, high = min(record['gains'][i] for i in ids), max(record['gains'][i] for i in ids)
            if high-low <= 1e-12:
                continue
            for i in ids:
                totals[i] += np.float32((record['gains'][i]-low)/(high-low))
                counts[i] += 1
        np.testing.assert_array_equal(counts, control['training_action_counts'])
        np.testing.assert_allclose(totals/np.maximum(counts, 1), control['scores'],
                                   rtol=0, atol=1e-6)
    indices = {key: {r['case']: i for i, r in enumerate(records)}
               for key, records in inputs.items()}
    natural = list(csv.DictReader(gzip.open(ROOT/'natural.csv.gz', 'rt')))
    computed = []
    for row in natural:
        dataset, split = row['dataset'], row['split']
        i = indices[dataset][row['case']]
        record = inputs[dataset][i]
        gains, mask = record['gains'], record['mask']
        ids = [j for j in range(125) if mask[j]]
        if row['policy'] == 'inspection_reference':
            scores = gains
        elif row['policy'] == 'action_mean':
            scores = controls[row['corpus']]['scores']
        else:
            scores = raw[row['corpus'], row['seed'], row['phase']][split][i].tolist()
        expected = rank(scores, gains, ids)
        low, high = min(gains[j] for j in ids), max(gains[j] for j in ids)
        targets = np.asarray([(gains[j]-low)/(high-low) if high-low > 1e-12 else 0
                              for j in ids], dtype=np.float32)
        expected['masked_regression_error'] = None if row['policy'] == 'inspection_reference' else \
            float(np.mean((np.asarray([scores[j] for j in ids], dtype=np.float32)-targets)**2))
        for key, value in expected.items():
            compare(value, row[key])
        computed.append(row | expected)
    per_fit = list(csv.DictReader((ROOT/'per-fit.csv').open()))
    for row in per_fit:
        selected = [r for r in computed if all(r[k] == row[k]
                    for k in ('corpus', 'seed', 'phase', 'split', 'policy', 'dataset'))]
        informative = [r for r in selected if r['normalised_gain_regret'] is not None]
        compare(len(selected), row['states'])
        compare(len(informative), row['informative_states'])
        for key in ('normalised_gain_regret', 'one_step_optimal', 'masked_regression_error',
                    'random_expected_normalised_regret', 'random_best_probability'):
            values = [r[key] for r in informative if r[key] is not None]
            compare(sum(values)/len(values) if values else None, row[key])
    completion = json.loads((ROOT/'completion.json').read_text())
    criteria, comparisons = [], []
    for seed in ('0', '1', '2'):
        found = {(r['corpus'], r['split']): r for r in per_fit if r['seed'] == seed
                 and r['phase'] == 'final' and r['policy'] == 'network'}
        small, expanded = found['small', 'fresh'], found['expanded', 'fresh']
        small_regret, regret = [float(r['normalised_gain_regret']) for r in (small, expanded)]
        small_best, best = [float(r['one_step_optimal']) for r in (small, expanded)]
        effect = {'fresh_regret_improves_ge_005': small_regret-regret >= .05,
                  'fresh_best_match_not_lower': best >= small_best}
        capability = {'training_regret_le_005': float(found['expanded', 'training'][
                      'normalised_gain_regret']) <= .05,
                      'fresh_regret_le_020': regret <= .20, 'fresh_best_match_ge_025': best >= .25}
        criteria.append({'seed': int(seed), 'data_effect_checks': effect,
                         'data_effect_passed': all(effect.values()),
                         'capability_checks': capability,
                         'capability_passed': all(capability.values())})
        comparisons.append({'seed': int(seed),
                            'fresh_normalised_gain_regret_expanded_minus_small':
                            regret-small_regret,
                            'fresh_one_step_optimal_expanded_minus_small': best-small_best})
    assert criteria == completion['per_seed_criterion']
    assert comparisons == completion['paired_comparisons']
    assert all(r['data_effect_passed'] for r in criteria) == completion['data_effect_passed']
    assert all(r['capability_passed'] for r in criteria) == \
        completion['capability_criterion_passed']
    for corpus, metrics in completion['fresh_fit_mean_sd'].items():
        for key, stats in metrics.items():
            values = [float(r[key]) for r in per_fit if r['phase'] == 'final'
                      and r['split'] == 'fresh' and r['corpus'] == corpus]
            assert len(values) == 3
            assert math.isclose(float(np.mean(values)), stats['mean'], abs_tol=1e-12)
            assert math.isclose(float(np.std(values, ddof=1)), stats['sample_sd'], abs_tol=1e-12)
    for row in csv.DictReader((ROOT/'summary.csv').open()):
        selected = [r for r in per_fit if r['corpus'] == row['corpus'] and
                    r['split'] == row['split'] and r['phase'] == 'final']
        values = [float(r[row['metric']]) for r in selected]
        compare(np.mean(values), row['mean'])
        compare(np.std(values, ddof=1), row['sample_sd'])
        compare(len(selected), row['fits'])
        compare(int(selected[0]['informative_states']), row['informative_states_per_fit'])
    print(json.dumps({'status': 'PASS', 'natural_rows': len(natural), 'per_fit_rows': len(per_fit),
                      'nested_original_records_exact': True, 'world_splits_disjoint': True,
                      'training_action_means_independently_checked': True,
                      'separate_effect_and_capability_criteria_verified': True,
                      'aggregate_mean_sd_and_paired_differences_verified': True,
                      'implementation': 'Separate scalar argmax/ties/normalisation/means; '
                      'no production metric or fitting functions.'}, indent=2))


if __name__ == '__main__':
    main()

