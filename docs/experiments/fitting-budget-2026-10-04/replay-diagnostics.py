"""Verify saved flights and record separate baseline/post-hoc familiar-world flights.

Run from the repository root with the original ignored checkpoints present.
These evaluations do not update any model or select a checkpoint.
"""

import csv
import hashlib
import json
from pathlib import Path

import torch

from wild_drone.baselines import GreedyPolicy
from wild_drone.behavior import behavior_episode
from wild_drone.config import EnvConfig
from wild_drone.coverage_probe import METHOD, load_coverage
from wild_drone.training_io import atomic_csv_save, atomic_json_save


def main():
    archive = Path(__file__).resolve().parent
    raw = archive/'raw'
    run = Path('runs/fitting-budget-v7')
    protocol = json.loads((raw/'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    config = EnvConfig(**protocol['environment'])
    torch.set_num_threads(1)
    original = list(csv.DictReader((raw/'validation_missions.csv').open()))
    learned = [r for r in original if r['policy'] == METHOD]
    assert len(learned) == 75
    policies = {}
    compared = 0
    for old in learned:
        seed, checkpoint = int(old['training_seed']), old['checkpoint']
        key = seed, checkpoint
        if key not in policies:
            manifest = json.loads((run/f'seed-{seed}/checkpoint_manifest.json').read_text())
            assert manifest[checkpoint]['sha256'] == old['checkpoint_sha256']
            policies[key] = load_coverage(run/f'seed-{seed}'/checkpoint, manifest[checkpoint])
        row, _ = behavior_episode(config, policies[key], int(old['scenario_seed']),
                                  name=METHOD, training_seed=seed)
        for name, value in row.items():
            if name == 'decision_ms':
                continue
            if value is None:
                assert old[name] == '', (key, name)
            elif isinstance(value, (int, float)):
                assert float(old[name]) == value, (key, old['scenario_seed'], name)
            else:
                assert old[name] == str(value), (key, name)
            compared += 1
    baseline = [behavior_episode(config, GreedyPolicy(), scenario, name='greedy')[0]
                for scenario in protocol['validation_scenarios']]
    atomic_csv_save(baseline, raw/'baseline-replay.csv')

    # Declared after the budget run: diagnostic overlap with five training worlds.
    scenarios = list(range(5))
    familiar = []
    for seed in protocol['fit_seeds']:
        checkpoint = 'round-20.pt'
        manifest = json.loads((run/f'seed-{seed}/checkpoint_manifest.json').read_text())
        policy = load_coverage(run/f'seed-{seed}'/checkpoint, manifest[checkpoint])
        for scenario in scenarios:
            row, _ = behavior_episode(config, policy, scenario, name=METHOD, training_seed=seed)
            familiar.append(row | {'checkpoint': checkpoint,
                                   'checkpoint_sha256': manifest[checkpoint]['sha256']})
    for scenario in scenarios:
        row, _ = behavior_episode(config, GreedyPolicy(), scenario, name='greedy')
        familiar.append({key: row.get(key) for key in familiar[0]})
    atomic_csv_save(familiar, raw/'posthoc-training-missions.csv')
    atomic_json_save({'original_learned_missions': len(learned),
                      'exact_non_timing_comparisons': compared,
                      'baseline_replay_missions': len(baseline),
                      'posthoc_training_scenarios': scenarios,
                      'posthoc_training_missions': len(familiar),
                      'original_csv_sha256': hashlib.sha256(
                          (raw/'validation_missions.csv').read_bytes()).hexdigest(),
                      'original_csv_note': 'Run log reports 80 missions, but persisted CSV has '
                      '75 learned missions. Preserve it unchanged; replay the five declared '
                      'Greedy baseline missions separately. Cause is unresolved.'},
                     raw/'replay-verification.json')
    print(f'Exactly verified {len(learned)} learned missions; recorded 5 baseline and '
          '20 post-hoc familiar-world missions.')


if __name__ == '__main__':
    main()
