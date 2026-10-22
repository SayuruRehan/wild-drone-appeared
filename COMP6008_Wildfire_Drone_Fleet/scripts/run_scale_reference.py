"""Controlled exact-inspection missions on the frozen 40x40 development worlds."""

import argparse
import csv
import gzip
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from run_mission_validation import METRICS, ScorePolicy, digest, rollout

from wild_drone.config import EnvConfig
from wild_drone.experiments import write_rows
from wild_drone.masked_learning import action_mask
from wild_drone.ranking_probe import gain_scores, transition_coverage
from wild_drone.training_io import atomic_json_save

ROOT = Path(__file__).resolve().parents[1]
WORLDS = tuple(range(7000, 7020))
PARENT = ROOT / 'docs/experiments/scale-pilot-2026-10-11'
PROTOCOL = ROOT / 'docs/scale-reference-protocol.md'


class ExactInspectionPolicy(ScorePolicy):
    """Diagnostic simulator lookahead, not a deployable learned policy or RL."""

    def __init__(self, config):
        super().__init__(config, scores=np.zeros(125))

    def act(self, env, observation):
        if env.config != self.config:
            raise ValueError('matching environment configuration required')
        self.last_mask = action_mask(env)
        self.last_scores = gain_scores(env, transition_coverage(env))
        return int(np.argmax(np.where(self.last_mask, self.last_scores, -np.inf)))


def compare(episodes, parent):
    """Matched-world descriptive comparisons; no new promotion criterion."""
    rows = []
    for name in ('exact_inspection', 'centered', 'greedy', 'masked_random'):
        group = [r for r in (episodes if name == 'exact_inspection' else parent)
                 if r['policy'] == name]
        expected = 60 if name == 'centered' else 20
        if (len(group) != expected or
                sorted({int(r['scenario_seed']) for r in group}) != list(WORLDS)):
            raise ValueError('complete matched development mission worlds required')
        seeds = ('0', '1', '2') if name == 'centered' else (None,)
        for seed in seeds:
            subset = group if seed is None else [r for r in group
                                                if str(r['training_seed']) == seed]
            if (len(subset) != 20 or len({int(r['scenario_seed']) for r in subset}) != 20):
                raise ValueError('complete matched fitted seeds required')
            rows.append({'policy': name, 'training_seed': seed, 'worlds': len(subset),
                         **{k: float(np.mean([float(r[k]) for r in subset]))
                            for k in METRICS if k != 'decision_ms'}})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}', args.source_revision):
        raise ValueError('full published source revision required')
    parent_launch = json.loads((PARENT / 'launch.json').read_text())
    if tuple(parent_launch['mission_worlds']) != WORLDS:
        raise ValueError('original development worlds required')
    config = EnvConfig(**parent_launch['env_config'])
    if config.grid_size != 40 or config.horizon != 300:
        raise ValueError('frozen 40x40/300 environment required')
    for path, sha in parent_launch['source_sha256'].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f'frozen simulator/scoring source differs: {path}')
    parent_paths = [PARENT / name for name in ('launch.json', 'missions/episodes.csv',
                                             'checkpoints.json', 'completion.json')]
    checks = json.loads((PARENT / 'checksums.json').read_text())
    for path in parent_paths:
        if digest(path) != checks[str(path.relative_to(PARENT))]['sha256']:
            raise ValueError('parent comparison archive identity differs')
    with (PARENT / 'missions/episodes.csv').open() as f:
        parent = list(csv.DictReader(f))
    args.out.mkdir(parents=True, exist_ok=False)
    source = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__).resolve(), PROTOCOL,
              ROOT / 'scripts/run_mission_validation.py']
    atomic_json_save({
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'controlled development diagnostic; not a new fitted-policy capability gate',
        'source_revision': args.source_revision,
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in source},
        'parent_source_revision': parent_launch['source_revision'],
        'parent_input_sha256': {str(p.relative_to(ROOT)): digest(p) for p in parent_paths},
        'env_config': config.to_dict(), 'worlds': list(WORLDS),
        'new_fits': 0, 'rl_updates': 0, 'new_worlds': 0,
        'action_rule': 'exact inspection gain; common mask; first-index argmax',
        'lookahead': '125 isolated simulator clones per pre-action state; real RNG untouched',
    }, args.out / 'launch.json')
    started, episodes, traces, scores, masks = time.perf_counter(), [], [], [], []
    for world in WORLDS:
        episode, trace, vectors = rollout(config, ExactInspectionPolicy(config), world,
                                          name='exact_inspection')
        episodes.append(episode)
        traces.extend(trace)
        scores.append(vectors['scores'])
        masks.append(vectors['masks'])
        write_rows(args.out / 'episodes.csv', episodes)
        print(f'exact inspection: world {world} complete ({len(episodes)}/20)', flush=True)
    np.savez_compressed(args.out / 'scores.npz', scores=np.concatenate(scores),
                        masks=np.concatenate(masks))
    with gzip.open(args.out / 'steps.csv.gz', 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    comparisons = compare(episodes, parent)
    write_rows(args.out / 'comparison.csv', comparisons)
    pairs = []
    for policy, seed in [('greedy', None), ('masked_random', None),
                         *[('centered', str(s)) for s in (0, 1, 2)]]:
        for row in episodes:
            reference = [r for r in parent if r['policy'] == policy and
                         (seed is None or r['training_seed'] == seed) and
                         int(r['scenario_seed']) == row['scenario_seed']]
            if len(reference) != 1:
                raise ValueError('unique paired comparison required')
            pairs.append({'comparison_policy': policy, 'training_seed': seed,
                          'scenario_seed': row['scenario_seed'],
                          **{k: row[k] - float(reference[0][k])
                             for k in METRICS if k != 'decision_ms'}})
    write_rows(args.out / 'paired.csv', pairs)
    result = {'episodes': len(episodes), 'steps': len(traces),
              'cloned_action_transitions': len(traces) * 125,
              'new_fits': 0, 'rl_updates': 0, 'new_worlds': 0,
              'elapsed_seconds': time.perf_counter() - started,
              'comparisons': comparisons}
    atomic_json_save(result, args.out / 'completion.json')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
