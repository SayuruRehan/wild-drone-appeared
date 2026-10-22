import importlib.util
from pathlib import Path

import numpy as np
import pytest


def auditor():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_scale_pilot.py'
    spec = importlib.util.spec_from_file_location('scale_audit', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_reconstruction_accepts_float32_roundoff_with_same_action():
    module = auditor()
    saved = np.array([[-.008393138647079468, .25, .125]], dtype=np.float32)
    reconstructed = saved.copy()
    reconstructed[0, 0] = -.00839313492178917
    assert not np.array_equal(reconstructed, saved)
    drift = module.audit_score_vectors(reconstructed, saved, np.ones_like(saved, dtype=bool))
    assert 0 < drift < 1e-7


def test_score_reconstruction_rejects_changed_values_or_actions():
    module = auditor()
    saved = np.array([[.25, .25 - 1e-7, 1.0]], dtype=np.float32)
    mask = np.array([[True, True, False]])
    changed = saved.copy()
    changed[0, 0] -= 2e-7
    with pytest.raises(AssertionError, match='argmax'):
        module.audit_score_vectors(changed, saved, mask)
    changed = saved.copy()
    changed[0, 2] += 1e-3
    with pytest.raises(AssertionError):
        module.audit_score_vectors(changed, saved, mask)


def test_score_reconstruction_rejects_nonfinite_or_incomplete_vectors():
    module = auditor()
    saved = np.array([[.25, .125]], dtype=np.float32)
    for changed, mask in [(saved[:, :1], np.ones_like(saved, dtype=bool)),
                          (np.array([[np.nan, .125]]), np.ones_like(saved, dtype=bool)),
                          (saved, np.zeros_like(saved, dtype=bool))]:
        with pytest.raises(AssertionError):
            module.audit_score_vectors(changed, saved, mask)
