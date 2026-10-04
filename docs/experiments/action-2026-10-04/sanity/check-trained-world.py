"""Evaluate the predeclared trained-world competence gate; never validation success."""

import json
from pathlib import Path

import torch

from wild_drone.action_probe import load_action
from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.imitation_probe import label_metrics
from wild_drone.training_io import atomic_csv_save, atomic_json_save

torch.set_num_threads(1)
root = Path('runs/action-sanity-v11')
protocol = json.loads((root/'protocol.json').read_text())
assert protocol['training_scenarios'] == [0]
assert protocol['budgets'] == [2000] and protocol['fit_seeds'] == [0, 1, 2]
config = EnvConfig(**protocol['environment'])
training = torch.load(root/'training.pt', weights_only=True)
teacher, _ = behavior_episode(config, GreedyPolicy(), 0, name='greedy')
rows, agreements, matches = [], {}, {}
for arm in protocol['arms']:
    for seed in protocol['fit_seeds']:
        folder = root/arm/f'seed-{seed}'
        record = json.loads((folder/'checkpoint_manifest.json').read_text())['final_action.pt']
        policy = load_action(folder/'final_action.pt', record)
        agreement = label_metrics(policy.model, training, 64)['joint_accuracy']
        row, _ = behavior_episode(config, policy, 0, name='action-sanity', training_seed=seed)
        key = f'{arm}/seed-{seed}'
        agreements[key] = agreement
        matches[key] = all(v == teacher[k] for k, v in row.items()
                           if k not in {'policy', 'training_seed', 'decision_ms'})
        rows.append(row | {'arm': arm, 'checkpoint_sha256': record['sha256']})
rows.append(teacher | {'arm': None, 'checkpoint_sha256': None})
atomic_csv_save(rows, root/'training_missions.csv')
passed = all(v == 1. for v in agreements.values()) and all(matches.values())
atomic_json_save({'gate_passed': passed, 'joint_training_agreements': agreements,
                  'exact_teacher_mission_matches': matches, 'training_world': 0,
                  'training_missions': 7, 'meaning': 'known-trajectory competence only'},
                 root/'sanity-gate.json')
print('Predeclared competence gate:', passed, agreements, matches, flush=True)
if not passed:
    raise SystemExit('Do not launch the full comparison: investigate failed competence gate.')
