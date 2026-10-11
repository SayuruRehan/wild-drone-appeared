"""Exercise the main notebook's archive analysis without fitting new models."""

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
MAIN = ROOT/'docs/experiments/rl-main-2026-10-11'


def notebook_cells():
    notebook = json.loads((ROOT/'Wildfire_Drone_Fleet.ipynb').read_text())
    return {cell['id']: ''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code'}


def review_namespace(folder):
    return {'MAIN': folder, 'pd': pd, 'np': np, 'display': lambda value: None,
            'ALGORITHMS': ['dqn', 'reinforce', 'a2c'],
            'POLICIES': ['dqn', 'reinforce', 'a2c', 'greedy', 'sweep', 'random', 'masked_random'],
            'frozen': {'evaluation_worlds': list(range(22000, 22020))}}


def copy_analysis(folder):
    for name in ['episodes.csv', 'summary.csv', 'comparison.csv', 'paired.csv']:
        shutil.copyfile(MAIN/name, folder/name)


@pytest.mark.parametrize('offset', [0, 123])
def test_main_notebook_reads_and_recomputes_the_selected_archive(tmp_path, offset):
    copy_analysis(tmp_path)
    episodes = pd.read_csv(tmp_path/'episodes.csv')
    episodes['return'] += offset
    episodes.to_csv(tmp_path/'episodes.csv', index=False)
    summary = pd.read_csv(tmp_path/'summary.csv')
    summary.loc[summary['metric'] == 'return', 'mean'] += offset
    summary.to_csv(tmp_path/'summary.csv', index=False)
    ns = review_namespace(tmp_path)
    exec(compile(notebook_cells()['main-results'], 'main-results', 'exec'), ns)
    assert len(ns['episodes']) == 260
    assert ns['main_means'].loc['dqn', 'return'] == pytest.approx(-24.7856840524+offset)
    assert ns['main_sds'].loc['dqn', 'return'] == pytest.approx(0.249913624)
    assert ns['main_means'].loc['random', 'total_losses'] == 3
    assert ns['main_means'].loc['masked_random', 'total_losses'] == 0


def test_main_notebook_rejects_a_stale_summary(tmp_path):
    copy_analysis(tmp_path)
    summary = pd.read_csv(tmp_path/'summary.csv')
    summary.loc[0, 'mean'] += 1
    summary.to_csv(tmp_path/'summary.csv', index=False)
    with pytest.raises(AssertionError):
        exec(compile(notebook_cells()['main-results'], 'main-results', 'exec'),
             review_namespace(tmp_path))


def test_review_cells_cannot_launch_training():
    forbidden = {'train', 'train_masked', 'run_jobs', 'run_with_progress', 'system', 'Popen'}
    for ident, source in notebook_cells().items():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (node.func.id if isinstance(node.func, ast.Name) else
                        node.func.attr if isinstance(node.func, ast.Attribute) else '')
                assert name not in forbidden, (ident, name)
                assert not (isinstance(node.func, ast.Attribute) and
                            isinstance(node.func.value, ast.Name) and
                            node.func.value.id == 'subprocess'), ident


def test_review_manifest_rejects_changed_source(tmp_path):
    source = notebook_cells()['input-integrity']
    ns = {'REPO': ROOT, 'pd': pd, 'display': lambda value: None}
    import hashlib

    ns['hashlib'] = hashlib
    exec(compile(source, 'input-integrity', 'exec'), ns)
    for relative in ns['ANALYSIS_INPUT_SHA256']:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/relative, path)
    changed = tmp_path/'src/wild_drone/environment.py'
    changed.write_bytes(changed.read_bytes()+b'\n# changed source\n')
    ns['REPO'] = tmp_path
    with pytest.raises(AssertionError,
                       match='Analysis input changed: src/wild_drone/environment.py'):
        exec(compile(source, 'input-integrity', 'exec'), ns)
