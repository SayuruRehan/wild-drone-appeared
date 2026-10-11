"""Post-hoc diagnostic of saved development missions; no fitting or new worlds."""
import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.masked_learning import action_mask
from wild_drone.ranking_probe import gain_scores, ranking_metrics, transition_coverage

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
RUN, OUT = args.run, args.out
OUT.mkdir(exist_ok=False)
launch = json.loads((RUN / 'launch.json').read_text())
config = EnvConfig(**launch['env_config'])
with gzip.open(RUN / 'missions/steps.csv.gz', 'rt') as f:
    traces = list(csv.DictReader(f))
groups = defaultdict(list)
for row in traces:
    groups[row['policy'], row['training_seed'], int(row['scenario_seed'])].append(row)
rows = []
for seed in range(3):
    with np.load(RUN / f'missions/centered-seed-{seed}-scores.npz') as f:
        scores, masks = f['scores'], f['masks']
    offset = 0
    for world in launch['mission_worlds']:
        env = WildfireFleetEnv(config)
        observation, _ = env.reset(seed=world)
        group = groups['centered', str(seed), world]
        for record in group:
            assert env.elapsed + 1 == int(record['step'])
            if env.elapsed in launch['snapshot_steps']:
                text = json.dumps({k: v.tolist() for k, v in observation.items()},
                                  sort_keys=True, separators=(',', ':'))
                assert hashlib.sha256(text.encode()).hexdigest() == record['pre_sha256']
                mask = action_mask(env)
                np.testing.assert_array_equal(mask, masks[offset])
                gains = gain_scores(env, transition_coverage(env))
                metrics = ranking_metrics(scores[offset], gains, np.flatnonzero(mask))
                assert metrics['selected_action'] == int(record['action'])
                rows.append(dict(seed=seed, world=world, step=env.elapsed, **metrics))
            observation, reward, done, truncated, _ = env.step(int(record['action']))
            assert abs(reward - float(record['reward'])) < 1e-12
            offset += 1
        assert env._done
    assert offset == len(scores)
    print('diagnosed seed', seed, flush=True)
summary = []
for seed in range(3):
    for step in [None, *launch['snapshot_steps']]:
        subset = [r for r in rows if r['seed'] == seed and
                  (step is None or r['step'] == step)]
        informative = [r for r in subset if r['normalised_gain_regret'] is not None]
        summary.append(dict(seed=seed, step=step, states=len(subset),
                            informative_states=len(informative),
                            normalised_gain_regret=float(np.mean([
                                r['normalised_gain_regret'] for r in informative])),
                            one_step_optimal=float(np.mean([
                                r['one_step_optimal'] for r in informative]))))
with (OUT / 'on-policy-ranking.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
result = {'kind': 'post-hoc development diagnostic, not a new capability gate',
          'new_fits': 0, 'new_worlds': 0, 'snapshot_steps': launch['snapshot_steps'],
          'states': len(rows), 'per_seed_and_phase': summary,
          'source_revision': launch['source_revision'],
          'artifact_run_id': 38111928600,
          'input_sha256': {str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [RUN / 'launch.json', RUN / 'missions/steps.csv.gz',
                                     *sorted((RUN / 'missions').glob('centered-*.npz'))]}}
(OUT / 'on-policy-summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)

with (RUN / 'missions/episodes.csv').open() as f:
    episodes = list(csv.DictReader(f))
mission_means = []
for name in ('centered', 'greedy', 'sweep', 'masked_random', 'action_mean'):
    group = [r for r in episodes if r['policy'] == name]
    mission_means.append({'policy': name, 'episodes': len(group), **{
        k: float(np.mean([float(r[k]) for r in group])) for k in (
            'return', 'weighted_freshness', 'overdue_fraction', 'energy_used',
            'safe_returns', 'charge_completions', 'collision_attempts', 'total_losses')}})
phase_means = []
for name in ('centered', 'greedy', 'masked_random'):
    for lo, hi in ((1, 100), (101, 200), (201, 300)):
        group = [r for r in traces if r['policy'] == name and lo <= int(r['step']) <= hi]
        phase_means.append({'policy': name, 'first_step': lo, 'last_step': hi,
                            'states': len(group), **{
            k: float(np.mean([float(r[k]) for r in group])) for k in (
                'weighted_freshness', 'overdue_fraction', 'reward_freshness_gain')}})
(OUT / 'mission-trace-summary.json').write_text(json.dumps({
    'post_hoc': True, 'mission_means': mission_means, 'phase_means': phase_means},
    indent=2) + '\n')
