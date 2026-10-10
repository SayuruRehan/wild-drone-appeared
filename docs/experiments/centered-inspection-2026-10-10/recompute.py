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
    raw = {(arm, str(seed), phase): np.load(
        ROOT/arm/f'seed-{seed}'/f'{phase}-scores.npz', allow_pickle=False)
        for arm in ('current', 'absolute', 'centered')
        for seed in (0, 1, 2) for phase in ('initial', 'final')}
    action_mean = json.loads((ROOT/'action-mean-control.json').read_text())['scores']
    natural = list(csv.DictReader(gzip.open(ROOT/'natural.csv.gz', 'rt')))
    case_indices = {split: {r['case']: i for i, r in enumerate(records)}
                    for split, records in inputs.items()}
    train_worlds = {r['scenario_seed'] for r in inputs['training']}
    heldout_worlds = {r['scenario_seed'] for r in inputs['heldout']}
    fresh_worlds = {r['scenario_seed'] for r in inputs['fresh']}
    assert not train_worlds & heldout_worlds
    assert not train_worlds & fresh_worlds
    assert not heldout_worlds & fresh_worlds
    computed = []
    for row in natural:
        split = row['split']
        i = case_indices[split][row['case']]
        record = inputs[split][i]
        gains, mask = record['gains'], record['mask']
        ids = [j for j in range(125) if mask[j]]
        scores = gains if row['policy'] == 'inspection_reference' else action_mean if \
            row['policy'] == 'action_mean' else raw[
                row['arm'], row['seed'], row['phase']][split][i].tolist()
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
                    for k in ('arm', 'seed', 'phase', 'split', 'policy'))]
        informative = [r for r in selected if r['normalised_gain_regret'] is not None]
        compare(len(selected), row['states'])
        compare(len(informative), row['informative_states'])
        for key in ('normalised_gain_regret', 'one_step_optimal', 'masked_regression_error',
                    'random_expected_normalised_regret', 'random_best_probability'):
            values = [r[key] for r in informative if r[key] is not None]
            compare(sum(values)/len(values) if values else None, row[key])
    pairs = list(csv.DictReader(gzip.open(ROOT/'pairs.csv.gz', 'rt')))
    computed_pairs = []
    for row in pairs:
        a, b = int(row['before']), int(row['after'])
        first, second = inputs['probes'][a], inputs['probes'][b]
        ga, gb, ma, mb = first['gains'], second['gains'], first['mask'], second['mask']
        if row['policy'] == 'network':
            sa, sb = raw[row['arm'], row['seed'], row['phase']]['probes'][a].tolist(), \
                raw[row['arm'], row['seed'], row['phase']]['probes'][b].tolist()
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
            np.testing.assert_array_equal(sa, sb)
        if row['policy'] == 'action_mean':
            assert expected['order_change_fraction'] == 0
        computed_pairs.append(row | expected)
    for row in csv.DictReader((ROOT/'summary.csv').open()):
        selected = [r for r in per_fit if r['arm'] == row['arm'] and
                    r['split'] == row['split'] and r['phase'] == 'final']
        values = [float(r[row['metric']]) for r in selected]
        compare(np.mean(values), row['mean'])
        compare(np.std(values, ddof=1), row['sample_sd'])
        compare(len(selected), row['fits'])
        compare(int(selected[0]['informative_states']), row['informative_states_per_fit'])
    for row in csv.DictReader((ROOT/'pair-summary.csv').open()):
        selected = [r for r in computed_pairs if r['arm'] == row['arm'] and
                    r['seed'] == row['seed'] and r['kind'] == row['kind'] and
                    r['phase'] == 'final']
        compare(len(selected), row['pairs'])
        for key in ('common_top_changed', 'mask_changed'):
            compare(sum(r[key] for r in selected), row[key])
        for key in ('order_change_fraction', 'before_normalised_gain_regret',
                    'after_normalised_gain_regret'):
            output_key = 'mean_'+key if key == 'order_change_fraction' else \
                'mean_normalised_regret_'+('before' if key.startswith('before') else 'after')
            compare(np.mean([r[key] for r in selected]), row[output_key])
    completion = json.loads((ROOT/'completion.json').read_text())
    criterion, comparisons = [], []
    for seed in ('0', '1', '2'):
        found = {(r['arm'], r['split']): r for r in per_fit
                 if r['seed'] == seed and r['phase'] == 'final'}
        centered = found['centered', 'fresh']
        regret = float(centered['normalised_gain_regret'])
        checks = {'training_regret_le_005': float(found['centered', 'training'][
                  'normalised_gain_regret']) <= .05,
                  'fresh_regret_le_020': regret <= .20,
                  'fresh_best_match_ge_025': float(centered['one_step_optimal']) >= .25}
        for control in ('current', 'absolute'):
            other = found[control, 'fresh']
            difference = regret-float(other['normalised_gain_regret'])
            checks[f'improves_{control}_ge_010'] = -difference >= .10
            comparisons.append({'seed': int(seed), 'control': control,
                                'fresh_regret_difference_centered_minus_control': difference,
                                'fresh_best_match_difference_centered_minus_control':
                                float(centered['one_step_optimal'])-float(other['one_step_optimal'])})
        criterion.append({'seed': int(seed), 'checks': checks, 'passed': all(checks.values())})
    assert criterion == completion['per_seed_criterion']
    assert comparisons == completion['paired_comparisons']
    assert all(r['passed'] for r in criterion) == completion['capability_criterion_passed']
    for arm, metrics in completion['fresh_fit_mean_sd'].items():
        for key, stats in metrics.items():
            values = [float(r[key]) for r in per_fit if r['phase'] == 'final'
                      and r['split'] == 'fresh' and r['arm'] == arm]
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
