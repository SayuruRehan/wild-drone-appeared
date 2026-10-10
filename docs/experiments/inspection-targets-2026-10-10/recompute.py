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
    raw = {(str(seed), phase): readzip(ROOT/f'seed-{seed}'/f'{phase}-scores.json.gz')
           for seed in (0, 1, 2) for phase in ('initial', 'final')}
    action_mean = json.loads((ROOT/'action-mean-control.json').read_text())['scores']
    natural = list(csv.DictReader((ROOT/'natural.csv').open()))
    case_indices = {split: {r['case']: i for i, r in enumerate(records)}
                    for split, records in inputs.items()}
    train_worlds = {r['scenario_seed'] for r in inputs['training']}
    heldout_worlds = {r['scenario_seed'] for r in inputs['heldout']}
    assert not train_worlds & heldout_worlds
    computed = []
    for row in natural:
        split = row['split']
        i = case_indices[split][row['case']]
        record = inputs[split][i]
        gains, mask = record['gains'], record['mask']
        ids = [j for j in range(125) if mask[j]]
        scores = gains if row['policy'] == 'inspection_reference' else action_mean if \
            row['policy'] == 'action_mean' else raw[row['seed'], row['phase']][split][i]
        expected = rank(scores, gains, ids)
        low, high = min(gains[j] for j in ids), max(gains[j] for j in ids)
        targets = np.asarray([(gains[j]-low)/(high-low) if high-low > 1e-12 else 0
                              for j in ids], dtype=np.float32)
        regression = None if row['policy'] == 'inspection_reference' else float(np.mean(
            (np.asarray([scores[j] for j in ids], dtype=np.float32)-targets)**2))
        expected['masked_regression_error'] = regression
        for key, value in expected.items():
            compare(value, row[key])
        computed.append(row | expected)
    per_fit = list(csv.DictReader((ROOT/'per-fit.csv').open()))
    for row in per_fit:
        selected = [r for r in computed if all(r[k] == row[k]
                    for k in ('seed', 'phase', 'split', 'policy'))]
        informative = [r for r in selected if r['normalised_gain_regret'] is not None]
        compare(len(selected), row['states'])
        compare(len(informative), row['informative_states'])
        for key in ('normalised_gain_regret', 'one_step_optimal', 'masked_regression_error',
                    'random_expected_normalised_regret', 'random_best_probability'):
            values = [r[key] for r in informative if r[key] is not None]
            compare(sum(values)/len(values) if values else None, row[key])
    pairs = list(csv.DictReader((ROOT/'pairs.csv').open()))
    for row in pairs:
        a, b = int(row['before']), int(row['after'])
        first, second = inputs['probes'][a], inputs['probes'][b]
        ga, gb, ma, mb = first['gains'], second['gains'], first['mask'], second['mask']
        if row['policy'] == 'network':
            sa, sb = raw[row['seed'], row['phase']]['probes'][a], \
                raw[row['seed'], row['phase']]['probes'][b]
        elif row['policy'] == 'action_mean':
            sa = sb = action_mean
        else:
            sa, sb = ga, gb
        ids = [j for j in range(125) if ma[j] and mb[j]]
        ra, rb = rank(sa, ga, ids), rank(sb, gb, ids)
        orders, strict = [], []
        for x, i in enumerate(ids):
            for j in ids[x+1:]:
                aa, bb = np.sign(sa[i]-sa[j]), np.sign(sb[i]-sb[j])
                orders.append(aa != bb)
                if aa and bb:
                    strict.append(aa != bb)
        am, bm = sum(sa[j] for j in ids)/len(ids), sum(sb[j] for j in ids)/len(ids)
        da = rank(sa, ga, [j for j in range(125) if ma[j]])['selected_action']
        db = rank(sb, gb, [j for j in range(125) if mb[j]])['selected_action']
        expected = {'common_actions': len(ids), 'mask_changed': ma != mb,
                    'score_delta_max': max(abs(sb[j]-sa[j]) for j in ids),
                    'centred_score_delta_max': max(abs((sb[j]-bm)-(sa[j]-am)) for j in ids),
                    'order_change_fraction': sum(orders)/len(orders),
                    'strict_flip_fraction': sum(strict)/len(strict) if strict else None,
                    'strict_comparable_pairs': len(strict),
                    'common_top_before': ra['selected_action'],
                    'common_top_after': rb['selected_action'],
                    'common_top_changed': ra['selected_action'] != rb['selected_action'],
                    'deployment_top_before': da, 'deployment_top_after': db,
                    'deployment_top_changed': da != db}
        expected.update({f'{side}_{k}': v for side, r in [('before', ra), ('after', rb)]
                         for k, v in r.items() if k != 'selected_action'})
        for key, value in expected.items():
            compare(value, row[key])
        if row['kind'] == 'noop':
            assert sa == sb
        if row['policy'] == 'action_mean':
            assert expected['order_change_fraction'] == 0
    completion = json.loads((ROOT/'completion.json').read_text())
    criterion = []
    control = next(r for r in per_fit if r['policy'] == 'action_mean'
                   and r['split'] == 'heldout')
    for seed in ('0', '1', '2'):
        found = {r['phase']+'-'+r['split']: r for r in per_fit if r['seed'] == seed}
        training, heldout = found['final-training'], found['final-heldout']
        initial = found['initial-heldout']
        checks = {'training_regret_le_005': float(training['normalised_gain_regret']) <= .05,
                  'heldout_regret_le_020': float(heldout['normalised_gain_regret']) <= .20,
                  'improves_initial_ge_015': float(initial['normalised_gain_regret']) -
                  float(heldout['normalised_gain_regret']) >= .15,
                  'improves_action_mean_ge_015': float(control['normalised_gain_regret']) -
                  float(heldout['normalised_gain_regret']) >= .15,
                  'heldout_best_match_ge_025': float(heldout['one_step_optimal']) >= .25}
        criterion.append({'seed': int(seed), 'checks': checks, 'passed': all(checks.values())})
    assert criterion == completion['per_seed_criterion']
    assert all(r['passed'] for r in criterion) == completion['capability_criterion_passed']
    for key, stats in completion['heldout_fit_mean_sd'].items():
        values = [float(r[key]) for r in per_fit if r['phase'] == 'final'
                  and r['split'] == 'heldout']
        assert len(values) == 3
        assert math.isclose(float(np.mean(values)), stats['mean'], abs_tol=1e-12)
        assert math.isclose(float(np.std(values, ddof=1)), stats['sample_sd'], abs_tol=1e-12)
    result = {'status': 'PASS', 'natural_rows': len(natural), 'pair_rows': len(pairs),
              'per_fit_rows': len(per_fit), 'world_split_disjoint': True,
              'no_op_vectors_exact': True, 'state_blind_pair_orders_unchanged': True,
              'aggregate_mean_sd_verified': True,
              'predeclared_capability_criterion_verified': True,
              'implementation': 'Independent scalar argmax/ties and explicit action-pair loops; '
              'no production metric or fitting functions.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
