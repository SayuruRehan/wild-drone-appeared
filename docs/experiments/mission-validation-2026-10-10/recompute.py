"""Independent scalar full-horizon mission metrics and declared decisions."""

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ('return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
           'total_losses', 'safe_returns', 'charge_completions', 'collision_attempts',
           'steps', 'fleet_failure', 'intact_fleet', 'decision_ms')


def read(name):
    opener = gzip.open if name.endswith('.gz') else open
    with opener(ROOT/name, 'rt', newline='') as f:
        return list(csv.DictReader(f))


def key(row):
    return row['policy'], row['training_seed'], int(row['scenario_seed'])


def close(left, right):
    assert math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12), (left, right)


def main():
    episodes, trace = read('episodes.csv'), read('steps.csv.gz')
    launch, completion = [json.loads((ROOT/name).read_text()) for name in
                          ('launch.json', 'completion.json')]
    assert launch['worlds'] == list(range(6600, 6620))
    assert launch['training_updates'] == completion['rl_updates'] == 0
    assert not completion['scale_to_40x40']
    groups = {}
    for row in trace:
        groups.setdefault(key(row), []).append(row)
    assert len(episodes) == len(groups) == completion['episodes'] == 220
    assert len(trace) == completion['steps']
    expected = {(name, str(seed), world) for name in ('small', 'expanded')
                for seed in range(3) for world in launch['worlds']} | {
                (name, '', world) for name in ('small_action_mean', 'expanded_action_mean',
                'greedy', 'sweep', 'masked_random') for world in launch['worlds']}
    assert set(groups) == expected == {key(r) for r in episodes}
    for ep in episodes:
        rows = groups[key(ep)]
        horizon, steps = int(ep['horizon']), int(ep['steps'])
        assert horizon == 160 and steps == len(rows)
        assert [int(r['step']) for r in rows] == list(range(1, steps+1))
        assert rows[-1]['terminated'] == 'True'
        assert all(r['terminated'] == r['truncated'] == 'False' for r in rows[:-1])
        for before, after in zip(rows, rows[1:]):
            assert before['post_sha256'] == after['pre_sha256']
        for row in rows:
            close(row['reward'], sum(float(v) for k, v in row.items() if k.startswith('reward_')))
        close(ep['return'], sum(float(r['reward']) for r in rows))
        close(ep['weighted_freshness'], sum(float(r['weighted_freshness']) for r in rows)/horizon)
        close(ep['overdue_fraction'], (sum(float(r['overdue_fraction']) for r in rows)+
                                      horizon-steps)/horizon)
        for metric in ('energy_used', 'safe_returns', 'charge_completions', 'collision_attempts'):
            close(ep[metric], sum(float(r[metric]) for r in rows))
        losses = sum(int(r['new_losses']) for r in rows)
        assert losses == int(ep['total_losses']) == int(rows[-1]['total_losses'])
        assert int(ep['fleet_failure']) == int(losses == 3)
        assert int(ep['intact_fleet']) == int(losses == 0)
        assert steps == horizon or losses == 3
    fits, summary = read('per-fit.csv'), read('summary.csv')
    metrics = [k for k in fits[0] if k not in ('policy', 'training_seed', 'worlds')]
    assert metrics == list(METRICS)
    assert len(fits) == 11
    fit_keys = {(name, seed) for name, seed, _ in expected}
    assert {(r['policy'], r['training_seed']) for r in fits} == fit_keys
    for fit in fits:
        rows = [r for r in episodes if (r['policy'], r['training_seed']) ==
                (fit['policy'], fit['training_seed'])]
        assert len(rows) == int(fit['worlds']) == 20
        for metric in metrics:
            close(fit[metric], sum(float(r[metric]) for r in rows)/len(rows))
    for row in summary:
        fitted = row['policy'] in ('small', 'expanded')
        assert row['unit'] == ('fitted_seed_means' if fitted else 'worlds')
        values = [float(r[row['metric']]) for r in (fits if fitted else episodes)
                  if r['policy'] == row['policy']]
        assert len(values) == int(row['units']) == (3 if fitted else 20)
        close(row['mean'], np.mean(values))
        close(row['sample_sd'], np.std(values, ddof=1))
    assert len(summary) == 7*len(metrics)
    assert {(r['policy'], r['metric']) for r in summary} == {
        (name, metric) for name, _ in fit_keys for metric in metrics}
    index = {key(r): r for r in episodes}
    paired = read('paired-expanded-minus-small.csv')
    assert len(paired) == 60
    assert {(r['training_seed'], int(r['scenario_seed'])) for r in paired} == {
        (str(seed), world) for seed in range(3) for world in launch['worlds']}
    for row in paired:
        seed, world = row['training_seed'], int(row['scenario_seed'])
        for metric in metrics:
            close(row[metric], float(index['expanded', seed, world][metric])-
                  float(index['small', seed, world][metric]))
    criteria = []
    for seed in range(3):
        means = {}
        for name in ('small', 'expanded', 'greedy'):
            rows = [r for r in episodes if r['policy'] == name and
                    r['training_seed'] == ('' if name == 'greedy' else str(seed))]
            means[name] = {metric: sum(float(r[metric]) for r in rows)/20 for metric in
                           ('weighted_freshness', 'overdue_fraction', 'total_losses',
                            'fleet_failure')}
        small, big, greedy = [means[name] for name in ('small', 'expanded', 'greedy')]
        effect = [big['weighted_freshness']-small['weighted_freshness'] >= .01,
                  big['overdue_fraction'] <= small['overdue_fraction'],
                  big['total_losses'] <= small['total_losses'],
                  big['fleet_failure'] <= small['fleet_failure']]
        capability = [big['weighted_freshness'] >= .9*greedy['weighted_freshness'],
                      big['overdue_fraction'] <= greedy['overdue_fraction']+.05,
                      big['total_losses'] <= greedy['total_losses']+.1,
                      big['fleet_failure'] <= greedy['fleet_failure']]
        archived = completion['per_seed'][seed]
        assert archived['seed'] == seed
        assert list(archived['data_effect_checks'].values()) == effect
        assert list(archived['greedy_capability_checks'].values()) == capability
        assert archived['data_effect_passed'] == all(effect)
        assert archived['greedy_capability_passed'] == all(capability)
        for name in means:
            for metric in means[name]:
                close(means[name][metric], archived['means'][name][metric])
        criteria.append((all(effect), all(capability)))
    assert completion['data_effect_passed'] == all(c[0] for c in criteria)
    assert completion['greedy_capability_passed'] == all(c[1] for c in criteria)
    result = {'status': 'PASS', 'episodes_verified': len(episodes),
              'steps_verified': len(trace), 'per_fit_rows_verified': len(fits),
              'aggregate_rows_verified': len(summary), 'paired_rows_verified': len(paired),
              'failure_padding_and_reward_components_verified': True,
              'separate_per_seed_decisions_verified': True,
              'implementation': 'Independent scalar sums/criteria and NumPy mean/sample SD; '
              'no runner metric helpers.'}
    (ROOT/'recompute-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
