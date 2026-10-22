"""Audit and trace DQN command behavior and final replay coverage on validation missions."""

import argparse
from pathlib import Path

import torch

from wild_drone.baselines import make_policy
from wild_drone.behavior import behavior_episode, exploration_summary, replay_summary
from wild_drone.checkpoint_audit import audit_checkpoints
from wild_drone.diagnostics import DQNDiagnosticPolicy
from wild_drone.experiments import validate_seed_split
from wild_drone.learning import load_policy
from wild_drone.training_io import atomic_csv_save, atomic_json_save


class ParkPolicy:
    """Diagnostic all-stay control; no inspection coordination or learned policy."""

    def act(self, env, observation):
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+', type=Path)
    parser.add_argument('--scenario-seeds', nargs='+', type=int, default=list(range(5000, 5005)))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if (len(args.scenario_seeds) != len(set(args.scenario_seeds)) or
            any(seed < 0 for seed in args.scenario_seeds)):
        raise ValueError('scenario seeds must be distinct and nonnegative')
    torch.set_num_threads(1)
    jobs, replay_rows, manifests = [], [], {}
    config = None
    # Preflight all checkpoints before replacing diagnostics.
    for directory in args.directories:
        report = audit_checkpoints(directory)
        state = torch.load(directory/'training_state.pt', map_location='cpu', weights_only=True)
        if state['algorithm'] != 'dqn':
            raise ValueError('this diagnostic runner requires DQN training states')
        manifests[str(directory)] = report
        common = {'directory': str(directory), 'training_seed': state['train_config']['seed'],
                  'epsilon_decay_steps': state['train_config']['epsilon_decay_steps'],
                  'authority_sha256': report['authority_sha256']}
        replay_rows.append(common | replay_summary(state) |
                           exploration_summary(state['train_config'], state['steps']))
        for name, record in report['exports'].items():
            policy = load_policy(directory/name)
            if config is None:
                config = policy.env_config
            elif config != policy.env_config:
                raise ValueError('all diagnostic runs must use the same environment')
            validate_seed_split(policy.train_config.scenario_seed_start,
                                policy.train_config.episodes, args.scenario_seeds)
            jobs.append((common | {'checkpoint': name, 'checkpoint_sha256': record['sha256'],
                                   'checkpoint_episode': record['completed_episodes']}, policy))
    rows, example_trace = [], []
    for common, policy in jobs:
        for scenario in args.scenario_seeds:
            diagnostic = DQNDiagnosticPolicy(policy)
            keep_trace = not rows  # One fully attributable example; no large default trace dump.
            row, trace = behavior_episode(config, diagnostic, scenario, name='dqn',
                                           training_seed=common['training_seed'], trace=keep_trace)
            rows.append(row | common | {'q_max': diagnostic.max_q,
                                         'q_upper_bound_violations': diagnostic.violations,
                                         'q_decisions': diagnostic.decisions,
                                         'q_upper_bound_excess_max': diagnostic.max_excess})
            if keep_trace:
                example_trace = [record | common | {'scenario_seed': scenario} for record in trace]
    # Same schema as fitted policies, with inapplicable provenance/Q fields left empty.
    for name in ['random', 'sweep', 'greedy', 'park']:
        for scenario in args.scenario_seeds:
            policy = ParkPolicy() if name == 'park' else make_policy(name, scenario)
            row, _ = behavior_episode(config, policy, scenario, name=name)
            rows.append({key: row.get(key) for key in rows[0]})
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_csv_save(rows, args.out/'missions.csv')
    atomic_csv_save(replay_rows, args.out/'replay.csv')
    atomic_csv_save(example_trace, args.out/'example-trace.csv')
    atomic_json_save({'validation_scenarios': args.scenario_seeds,
                      'environment': config.to_dict(), 'checkpoint_audits': manifests,
                      'critical_margin': '0 through three move costs; battery units',
                      'return_bound': 'Manhattan distance * move cost; necessary, not sufficient',
                      'epsilon_counts': 'expected schedule selections, not observed branch counts',
                      'example_trace': 'first directory, final policy, first validation scenario'},
                     args.out/'protocol.json')
    print(f'Behavior diagnostics: {len(rows)} missions, {len(replay_rows)} buffers -> {args.out}')


if __name__ == '__main__':
    main()
