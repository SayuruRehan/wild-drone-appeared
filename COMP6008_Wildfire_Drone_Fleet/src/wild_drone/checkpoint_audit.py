"""Verify portable exports against an authoritative episode-boundary training state."""

import hashlib
import io
import pickle
from pathlib import Path

import torch


def _load(path):
    try:
        data = path.read_bytes()
        checkpoint = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError('expected checkpoint dictionary')
        return checkpoint, hashlib.sha256(data).hexdigest()
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as error:
        raise ValueError(f'{path.name}: cannot read checkpoint: {error}') from error


def _equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        return (isinstance(actual, torch.Tensor) and actual.dtype == expected.dtype and
                actual.shape == expected.shape and torch.equal(actual, expected))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys() and
                all(_equal(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, (list, tuple)):
        return (type(actual) is type(expected) and len(actual) == len(expected) and
                all(_equal(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def audit_checkpoints(output_dir):
    """Read-only snapshot audit; fail on stale episodes, metadata or any changed tensor.

    A successful return certifies the bytes whose hashes appear in the report. To repair
    interrupted exports, explicitly resume training from training_state.pt at the same
    total episode budget. Policy files alone cannot certify their training provenance.
    """
    out = Path(output_dir)
    state, authority_hash = _load(out/'training_state.pt')
    if state.get('format_version') != 2 or 'optimizer' not in state:
        raise ValueError('training_state.pt: requires a version-2 authoritative training state')
    completed = state['completed_episodes']
    if len(state['rows']) != completed:
        raise ValueError('training_state.pt: episode row count differs from completed episodes')
    fields = ('format_version', 'algorithm', 'env_config', 'train_config', 'model_config',
              'state_dict', 'completed_episodes', 'source_history')
    expected = {'policy.pt': {key: state[key] for key in fields}}
    if state['best_checkpoint'] is not None:
        expected['best_policy.pt'] = state['best_checkpoint']
    elif (out/'best_policy.pt').exists():
        raise ValueError('best_policy.pt: unexpected export without an authoritative selection')
    exports = {}
    for name, reference in expected.items():
        actual, digest = _load(out/name)
        for key, value in reference.items():
            if key not in actual or not _equal(actual[key], value):
                raise ValueError(f'{name}: {key} differs from training_state.pt; '
                                 'resume from authority to regenerate exports')
        exports[name] = {'completed_episodes': actual['completed_episodes'], 'sha256': digest}
    return {'format_version': 1, 'algorithm': state['algorithm'],
            'completed_episodes': completed, 'authority_sha256': authority_hash,
            'exports': exports}
