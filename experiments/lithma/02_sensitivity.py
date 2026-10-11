"""Task 2 - 40x40 sensitivity to battery, charging and fire spread (no retraining).

One setting is changed at a time around the frozen default; everything else stays frozen:

    battery_capacity     120 | 160 (default) | 200
    recharge_rate          5 |   8 (default) |  12     (battery units per charging step, so
                                                        a lower rate means a longer charge)
    spread_probability  0.02 | 0.04 (default) | 0.08

That is 7 conditions (the default is shared).  Each condition is flown on the same 20 final
worlds as the frozen study.  A world seed fixes terrain, fuel, priority zones, ignitions and
the fire's random draws, and none of the three settings changes how many random numbers the
environment consumes, so every condition sees the same world apart from the changed setting.

Policies: greedy, rolling-horizon planner, masked random, sweep, and the nine frozen learner
checkpoints (DQN / REINFORCE / A2C x seeds 0-2).  The learners are NOT retrained: the weights
trained at the default setting are evaluated unchanged, so their rows measure robustness of
a fixed policy, not what the algorithm could learn under the new setting.  Their battery
input is normalised by the current capacity, and the feasibility mask (shared with the
baselines) is recomputed from the current setting.

Outputs (in ``results/``):
    sensitivity_episodes.csv   one row per (condition, policy, training seed, world)
    sensitivity_summary.csv    mean and sample SD per (condition, policy), plus the change
                               from the default condition
    sensitivity.png            freshness, return and drone losses against each setting

Run:
    .venv/bin/python 02_sensitivity.py --workers 8
    .venv/bin/python 02_sensitivity.py --plot-only     # redraw from the saved CSV
"""

import argparse
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from common import (  # noqa: E402
    ALGORITHMS,
    EPISODE_COLUMNS,
    METRICS,
    RESULTS,
    TRAINING_SEEDS,
    frozen_setup,
    read_csv,
    run_jobs,
    summarise,
    verify_checkpoint,
    write_csv,
)

# parameter -> (low, default, high); the default must equal the frozen config.
SWEEPS = {'battery_capacity': (120, 160, 200),
          'recharge_rate': (5, 8, 12),
          'spread_probability': (0.02, 0.04, 0.08)}
BASELINES = ('greedy', 'planner', 'masked_random', 'sweep')
PLOT_METRICS = (('weighted_freshness', 'Weighted freshness (higher is better)'),
                ('return', 'Episode return (higher is better)'),
                ('total_losses', 'Drones lost per mission (lower is better)'))
# Fixed colour per policy (validated categorical order); marker and line style repeat the
# identity so the figure does not rely on colour alone.  Learners are dashed.
STYLE = {'greedy': ('#2a78d6', 'o', '-'), 'planner': ('#eb6834', 's', '-'),
         'masked_random': ('#1baf7a', '^', '-'), 'sweep': ('#eda100', 'D', '-'),
         'dqn': ('#e87ba4', 'v', '--'), 'reinforce': ('#008300', 'P', '--'),
         'a2c': ('#4a3aa7', 'X', '--')}


def conditions():
    """[(condition name, parameter, value, overrides)], default first."""
    result = [('default', 'default', '', {})]
    for parameter, (low, default, high) in SWEEPS.items():
        assert getattr(frozen_setup()[0], parameter) == default, parameter
        for value in (low, high):
            result.append((f'{parameter}={value}', parameter, value, {parameter: value}))
    return result


def simulate(workers):
    _, _, worlds = frozen_setup()
    learners = [(a, s) for a in ALGORITHMS for s in TRAINING_SEEDS if verify_checkpoint(a, s)]
    missing = 9-len(learners)
    print(f'{len(learners)}/9 learner checkpoints present and byte-identical to the frozen '
          f'archive' + (f'; {missing} missing -> those learners are skipped' if missing else ''))
    jobs = []
    for name, parameter, value, overrides in conditions():
        tag = {'condition': name, 'parameter': parameter, 'value': value}
        for policy in BASELINES:
            jobs += [(tag, overrides, policy, None, w, None) for w in worlds]
        for algorithm, seed in learners:
            jobs += [(tag, overrides, algorithm, seed, w, None) for w in worlds]
    started = time.perf_counter()
    rows = run_jobs(jobs, workers, label='sensitivity')
    print(f'{len(rows)} missions in {time.perf_counter()-started:.0f}s wall, {workers} workers')
    write_csv(RESULTS/'sensitivity_episodes.csv', rows,
              ('condition', 'parameter', 'value')+EPISODE_COLUMNS+('wall_seconds',))


def load():
    rows = read_csv(RESULTS/'sensitivity_episodes.csv')
    for row in rows:
        row['training_seed'] = int(row['training_seed']) if row['training_seed'] else None
    return rows


def summary(rows):
    policies = [p for p in BASELINES+ALGORITHMS if any(r['policy'] == p for r in rows)]
    stats = {(c, p): summarise([r for r in rows if r['condition'] == c and r['policy'] == p])
             for c in dict.fromkeys(r['condition'] for r in rows) for p in policies}
    table = []
    for (condition, policy), s in stats.items():
        base = stats['default', policy]
        parameter, _, value = condition.partition('=')
        table.append({'condition': condition, 'parameter': parameter, 'value': value,
                      'policy': policy, 'unit': s['unit'], 'units': s['units'],
                      **{f'{m}_mean': s[m][0] for m in METRICS},
                      **{f'{m}_sd': s[m][1] for m in METRICS},
                      **{f'{m}_change_from_default': s[m][0]-base[m][0]
                         for m in ('return', 'weighted_freshness', 'overdue_fraction',
                                   'total_losses')}})
    write_csv(RESULTS/'sensitivity_summary.csv', table)
    return stats, policies


def plot(stats, policies):
    fig, axes = plt.subplots(len(PLOT_METRICS), len(SWEEPS), figsize=(13, 10.5), sharey='row')
    for column, (parameter, values) in enumerate(SWEEPS.items()):
        names = [f'{parameter}={v}' if v != values[1] else 'default' for v in values]
        for row, (metric, label) in enumerate(PLOT_METRICS):
            ax = axes[row, column]
            for policy in policies:
                colour, marker, line = STYLE[policy]
                means = [stats[n, policy][metric][0] for n in names]
                sds = [stats[n, policy][metric][1] for n in names]
                if metric == 'total_losses':    # a count cannot go below zero: no SD bars
                    sds = None
                ax.errorbar(range(3), means, yerr=sds, color=colour, marker=marker,
                            linestyle=line, linewidth=2, markersize=7, capsize=3,
                            elinewidth=1, label=policy)
            ax.set_xticks(range(3), [f'{v}\n(default)' if v == values[1] else str(v)
                                     for v in values])
            ax.grid(axis='y', color='#e6e5e0', linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            if column == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(parameter.replace('_', ' '))
            if row == len(PLOT_METRICS)-1:
                ax.set_xlabel(parameter)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(policies), frameon=False)
    fig.suptitle('40x40 sensitivity on the 20 frozen final worlds, one setting changed at a '
                 'time, no retraining\nbars: sample SD across 20 worlds (baselines, solid) or '
                 'across 3 training-seed means (learners, dashed)', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(RESULTS/'sensitivity.png', dpi=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if not args.plot_only:
        simulate(args.workers)
    stats, policies = summary(load())
    plot(stats, policies)
    print(f"{'condition':26s}" + ''.join(f'{p:>15s}' for p in policies))
    for metric in ('weighted_freshness', 'return', 'total_losses'):
        print(metric)
        for condition in dict.fromkeys(c for c, _ in stats):
            print(f'  {condition:24s}' + ''.join(f'{stats[condition, p][metric][0]:15.4f}'
                                                 for p in policies))


if __name__ == '__main__':
    main()
