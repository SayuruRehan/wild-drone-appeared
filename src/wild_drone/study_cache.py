"""Validate completed study fits before reusing any of their exports."""

import csv
import json
import math
import pickle
from pathlib import Path

import torch

from wild_drone.learning import load_policy
from wild_drone.masked_learning import MASK_RULE


def _metric_rows(path, required):
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(required) <= set(reader.fieldnames or ()):
            raise ValueError(f'{path.name} is missing report columns')
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path.name} has no metrics')
    if any(not math.isfinite(float(row[name])) for row in rows for name in required):
        raise ValueError(f'{path.name} has nonfinite metrics')
    return rows


def validate_cached_fit(masked, algorithm, config, settings, output_dir):
    """Return False for a fresh fit; reject stale or incomplete completed fits.

    These study runners do not resume training. A mismatch requires a new output directory;
    changing a request never silently overwrites or extends an old experiment.
    """
    out = Path(output_dir)
    summary = out/'training_summary.json'
    if not summary.exists():
        if any((out/name).exists() for name in ('policy.pt', 'training_state.pt', 'training.csv')):
            raise ValueError(f'incomplete cached fit in {out}; choose another output folder')
        return False
    expected = {'algorithm': algorithm, 'env_config': config.to_dict(),
                'train_config': settings.to_dict(), 'completed_episodes': settings.episodes,
                'action_mask': MASK_RULE if masked else None}
    if masked and algorithm == 'dqn' and settings.double_dqn:
        expected['dqn_target_rule'] = 'masked-double-dqn-v1'
    try:
        saved = json.loads(summary.read_text())
        policy = torch.load(out/'policy.pt', map_location='cpu', weights_only=True)
        for label, data in [('summary', saved), ('checkpoint', policy)]:
            for key, value in expected.items():
                if data.get(key) != value:
                    raise ValueError(f'{label} {key} differs from the requested fit')
        if saved['model_config'] != policy['model_config']:
            raise ValueError('checkpoint model differs from summary')
        # Validate tensor shapes and the loader contract, not just primitive metadata.
        load_policy(out/'policy.pt')
        metrics = ('return', 'weighted_freshness', 'overdue_fraction', 'total_losses',
                   'energy_used', 'safe_returns', 'charge_completions', 'collision_attempts',
                   'steps', 'training_seed', 'scenario_seed', 'episode')
        rows = _metric_rows(out/'training.csv', metrics + ('updates',))
        episodes = list(range(1, settings.episodes+1))
        if [int(r['episode']) for r in rows] != episodes:
            raise ValueError('training CSV is incomplete')
        if any(int(r['training_seed']) != settings.seed or
               int(r['scenario_seed']) != settings.scenario_seed_start+int(r['episode'])-1
               for r in rows):
            raise ValueError('training CSV seed identity differs')
        if settings.validation_seeds:
            validation = _metric_rows(out/'validation.csv', metrics + ('decision_ms',))
            checkpoints = [e for e in episodes if e % settings.validation_interval == 0 or
                           e == settings.episodes]
            expected_pairs = {(e, s) for e in checkpoints for s in settings.validation_seeds}
            pairs = [(int(r['episode']), int(r['scenario_seed'])) for r in validation]
            if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
                raise ValueError('validation CSV is incomplete')
            if any(int(r['training_seed']) != settings.seed for r in validation):
                raise ValueError('validation CSV training seed differs')
            selected = torch.load(out/'best_policy.pt', map_location='cpu', weights_only=True)
            for key in ('algorithm', 'env_config', 'train_config', 'action_mask', 'model_config'):
                if selected.get(key) != policy.get(key):
                    raise ValueError(f'selected checkpoint {key} differs')
            if selected['completed_episodes'] != saved.get('best_episode',
                                                          selected['completed_episodes']):
                raise ValueError('selected checkpoint episode differs')
            load_policy(out/'best_policy.pt')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError,
            pickle.UnpicklingError) as error:
        raise ValueError(f'invalid cached fit in {out}: {error}; choose another output folder') \
            from error
    return True
