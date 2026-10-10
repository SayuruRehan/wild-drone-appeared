"""Post hoc, nonintervening behavior replay of a completed coordination pilot."""

import argparse
import csv
import json

import torch

from wild_drone.baselines import make_policy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.experiments import write_rows
from wild_drone.masked_learning import MaskedRandomPolicy, load_masked_policy


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    root = parser.parse_args().run_root
    torch.set_num_threads(1)
    config = EnvConfig(**json.loads((root/'unshaped.json').read_text())['environment'])
    with (root/'screening.csv').open() as handle:
        original = {(r['policy'], int(r['scenario_seed'])): r for r in csv.DictReader(handle)}
    rows = []
    for name in ['shaped double dqn', 'unshaped double dqn', 'masked random', 'greedy']:
        policy = load_masked_policy(root/'raw'/name.split()[0]/'seed-0'/'policy.pt') \
            if 'double' in name else None
        for seed in range(5700, 5720):
            controller = policy or (MaskedRandomPolicy(seed) if name == 'masked random'
                                    else make_policy('greedy'))
            row, _ = behavior_episode(config, controller, seed, name=name,
                                      training_seed=0 if policy else None)
            previous = original[name, seed]
            for metric in ['return', 'weighted_freshness', 'overdue_fraction', 'total_losses',
                           'energy_used', 'safe_returns', 'charge_completions',
                           'collision_attempts', 'steps']:
                if float(row[metric]) != float(previous[metric]):
                    raise ValueError(f'Replay differs: {name}, world {seed}, {metric}')
            rows.append(row)
    write_rows(root/'behavior.csv', rows)
    print('All 80 replays match original screening metrics exactly.')


if __name__ == '__main__':
    main()
