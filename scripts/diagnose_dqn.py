"""Audit DQN exports and collect Q-bound and mission diagnostics on validation scenarios."""

import argparse
from pathlib import Path

import torch

from wild_drone.checkpoint_audit import audit_checkpoints
from wild_drone.diagnostics import DQNDiagnosticPolicy
from wild_drone.experiments import run_episode, validate_seed_split
from wild_drone.learning import load_policy
from wild_drone.training_io import atomic_csv_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--scenario-seeds', nargs='+', type=int,
                        default=list(range(5000, 5005)))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for directory in args.directories:
        report = audit_checkpoints(directory)
        for name, record in report['exports'].items():
            policy = load_policy(directory/name)
            validate_seed_split(policy.train_config.scenario_seed_start,
                                policy.train_config.episodes, args.scenario_seeds)
            for scenario in args.scenario_seeds:
                diagnostic = DQNDiagnosticPolicy(policy)
                row = run_episode(policy.env_config, diagnostic, scenario, name=policy.algorithm,
                                  training_seed=policy.train_config.seed)
                rows.append(row | {'double_dqn': policy.train_config.double_dqn,
                                   'learning_rate': policy.train_config.learning_rate,
                                   'checkpoint': name, 'checkpoint_sha256': record['sha256'],
                                   'checkpoint_episode': record['completed_episodes'],
                                   'q_max': diagnostic.max_q,
                                   'q_upper_bound_violations': diagnostic.violations,
                                   'q_decisions': diagnostic.decisions,
                                   'q_upper_bound_excess_max': diagnostic.max_excess})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv_save(rows, args.out)
    print(f'Diagnostics: {len(rows)} validation rows -> {args.out}')


if __name__ == '__main__':
    main()
