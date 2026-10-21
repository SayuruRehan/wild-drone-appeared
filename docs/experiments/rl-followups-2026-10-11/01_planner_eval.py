"""Task 1 - rolling-horizon planner on the frozen 40x40 final worlds.

The frozen main study compared DQN / REINFORCE / A2C with greedy, sweep, random and masked
random, but never ran ``RollingHorizonPolicy`` (``src/wild_drone/baselines.py``), which the
proposal promised.  This script fills that gap without touching the frozen archive:

1. ``--check``  : time the planner on 2 worlds, serially, to estimate the cost of the rest.
2. (default)    : a) re-run greedy, sweep, random and masked random on the 20 frozen worlds and
                     confirm this harness reproduces the archived ``episodes.csv`` numbers,
                     so the new planner rows are directly comparable with the frozen rows;
                  b) run the planner on the same 20 worlds (planner RNG seed = world number);
                  c) write the planner rows and a table next to the frozen baselines/learners.

Planner settings are the class defaults: lookahead 4 steps, 8 candidate first actions (the
greedy action plus 7 random feasible ones), 2 sampled fire futures per candidate, rollouts
continue with 75% greedy / 25% random feasible actions, and the final state's weighted
freshness is added as a small terminal value.

Outputs (in ``results/``):
    planner_check.json            runtime check on 2 worlds
    planner_episodes.csv          20 planner missions, frozen episodes.csv columns
    harness_reproduction.csv      max |difference| from the frozen archive, per baseline
    planner_vs_frozen_table.csv   mean and sample SD for every policy
    planner_vs_frozen_table.md    the same as a readable table

Run:
    .venv/bin/python 01_planner_eval.py --check
    .venv/bin/python 01_planner_eval.py --workers 5
"""

import argparse
import json
import time

import numpy as np
from common import (
    EPISODE_COLUMNS,
    FROZEN_ARCHIVE,
    METRICS,
    RESULTS,
    frozen_setup,
    make_policy,
    read_csv,
    run_episode,
    run_jobs,
    summarise,
    write_csv,
)

FROZEN_BASELINES = ('greedy', 'sweep', 'random', 'masked_random')
TABLE_ORDER = ('dqn', 'reinforce', 'a2c', 'greedy', 'planner', 'sweep', 'random',
               'masked_random')
# decision_ms is wall-clock and machine dependent, so it is not expected to reproduce.
EXACT_METRICS = tuple(m for m in METRICS if m != 'decision_ms')


def check(worlds):
    """Serial timing on the first two worlds (planner and, for scale, greedy)."""
    config = frozen_setup()[0]
    report = {'worlds': list(worlds), 'note': 'serial, one process, this machine'}
    for name in ('greedy', 'planner'):
        rows = []
        for world in worlds:
            started = time.perf_counter()
            row = run_episode(config, make_policy(name, world), world, name=name)
            row['wall_seconds'] = time.perf_counter()-started
            rows.append(row)
            print(f"{name} world {world}: {row['wall_seconds']:.1f}s, "
                  f"{row['decision_ms']:.1f} ms/decision, return {row['return']:.3f}, "
                  f"freshness {row['weighted_freshness']:.4f}", flush=True)
        report[name] = {'seconds_per_mission': float(np.mean([r['wall_seconds'] for r in rows])),
                        'decision_ms': float(np.mean([r['decision_ms'] for r in rows])),
                        'episodes': rows}
    per = report['planner']['seconds_per_mission']
    report['estimate_20_worlds_serial_minutes'] = per*20/60
    RESULTS.mkdir(exist_ok=True)
    (RESULTS/'planner_check.json').write_text(json.dumps(report, indent=2))
    print(f'planner: ~{per:.0f}s per mission -> ~{per*20/60:.1f} min for 20 worlds serially')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true', help='2-world runtime check only')
    parser.add_argument('--workers', type=int, default=5)
    args = parser.parse_args()
    _, _, worlds = frozen_setup()
    if args.check:
        return check(worlds[:2])

    def jobs(name):
        return [({}, {}, name, None, w, None) for w in worlds]

    # a) Reproduce the archived baselines with this harness.
    frozen = read_csv(FROZEN_ARCHIVE/'episodes.csv')
    reproduction = []
    for name in FROZEN_BASELINES:
        new = run_jobs(jobs(name), args.workers, label=name)
        old = {int(r['scenario_seed']): r for r in frozen if r['policy'] == name}
        worst = {m: max(abs(float(r[m])-float(old[r['scenario_seed']][m])) for r in new)
                 for m in EXACT_METRICS}
        reproduction.append({'policy': name, 'worlds': len(new),
                             'max_abs_difference': max(worst.values()),
                             **{f'max_diff_{m}': v for m, v in worst.items()}})
        print(f"{name}: max |difference| from frozen archive = {max(worst.values()):.3g}")
    write_csv(RESULTS/'harness_reproduction.csv', reproduction)

    # b) The planner on the same worlds.
    started = time.perf_counter()
    planner = run_jobs(jobs('planner'), args.workers, label='planner')
    print(f'planner: 20 missions in {time.perf_counter()-started:.0f}s wall '
          f'with {args.workers} workers')
    write_csv(RESULTS/'planner_episodes.csv', planner, EPISODE_COLUMNS+('wall_seconds',))

    # c) One table: frozen rows as archived, plus the new planner rows.
    for row in frozen:
        row['scenario_seed'] = int(row['scenario_seed'])
    everything = frozen + planner
    table = []
    for name in TABLE_ORDER:
        stats = summarise([r for r in everything if r['policy'] == name])
        table.append({'policy': name, 'source': 'new (this script)' if name == 'planner'
                      else 'frozen archive', 'unit': stats['unit'], 'units': stats['units'],
                      **{f'{m}_mean': stats[m][0] for m in METRICS},
                      **{f'{m}_sd': stats[m][1] for m in METRICS}})
    write_csv(RESULTS/'planner_vs_frozen_table.csv', table)
    lines = ['| Policy | Freshness ↑ | Overdue ↓ | Return ↑ | Energy | Losses ↓ | ms/decision |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['policy']} | {r['weighted_freshness_mean']:.4f} ± "
                     f"{r['weighted_freshness_sd']:.4f} | {r['overdue_fraction_mean']:.4f} ± "
                     f"{r['overdue_fraction_sd']:.4f} | {r['return_mean']:.2f} ± "
                     f"{r['return_sd']:.2f} | {r['energy_used_mean']:.0f} | "
                     f"{r['total_losses_mean']:.2f} | {r['decision_ms_mean']:.2f} |")
    lines += ['', 'Mean ± sample SD. Learners: SD across 3 training-seed means; baselines and '
              'planner: SD across 20 worlds.', 'Frozen rows are copied from the archive; '
              'their ms/decision was measured on a GitHub runner, the planner on this machine '
              f'with {args.workers} parallel workers, so timing columns are indicative only.']
    (RESULTS/'planner_vs_frozen_table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

    # Paired planner-minus-greedy differences on the same worlds.
    greedy = {r['scenario_seed']: r for r in frozen if r['policy'] == 'greedy'}
    for metric in ('return', 'weighted_freshness', 'overdue_fraction'):
        delta = [float(r[metric])-float(greedy[r['scenario_seed']][metric]) for r in planner]
        print(f'planner - greedy, {metric}: mean {np.mean(delta):+.4f}, '
              f'SD {np.std(delta, ddof=1):.4f}, planner better in '
              f"{sum(d < 0 if metric == 'overdue_fraction' else d > 0 for d in delta)}/20 worlds")


if __name__ == '__main__':
    main()
