"""Verify the submission notebook and its archived analysis without rerunning training."""

import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
MAIN = ROOT/'docs/experiments/rl-main-2026-10-11'
FOLLOW_UP = ROOT/'docs/experiments/rl-followups-2026-10-11'


def notebook_cells():
    notebook = json.loads((ROOT/'Wildfire_Drone_Fleet_clean.ipynb').read_text())
    return [''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code']


def code_starting(prefix):
    matches = [source for source in notebook_cells() if source.startswith(prefix)]
    assert len(matches) == 1, f'Expected one notebook code cell starting with {prefix!r}'
    return matches[0]


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
    exec(compile(code_starting("episodes = pd.read_csv(MAIN/'episodes.csv')"),
                 'main-results', 'exec'), ns)
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
        exec(compile(code_starting("episodes = pd.read_csv(MAIN/'episodes.csv')"),
                     'main-results', 'exec'), review_namespace(tmp_path))


def test_main_notebook_rejects_duplicate_mission_rows(tmp_path):
    copy_analysis(tmp_path)
    episodes = pd.read_csv(tmp_path/'episodes.csv')
    episodes = pd.concat([episodes, episodes.iloc[[0]]], ignore_index=True)
    episodes.to_csv(tmp_path/'episodes.csv', index=False)
    with pytest.raises(AssertionError):
        exec(compile(code_starting("episodes = pd.read_csv(MAIN/'episodes.csv')"),
                     'main-results', 'exec'), review_namespace(tmp_path))


def test_costly_studies_are_opt_in_and_final_archives_exist():
    setup = code_starting('import copy, hashlib, json, random, sys, tempfile, time')
    flags = {'RUN_TRAINING', 'RUN_SWEEP', 'RUN_MASK_COMPARISON',
             'RUN_PLANNER', 'RUN_SENSITIVITY'}
    values = {}
    for statement in ast.parse(setup).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in flags:
                values[target.id] = statement.value
    assert set(values) == flags
    assert all(isinstance(value, ast.Constant) and value.value is False
               for value in values.values())
    assert not (ROOT/'Wildfire_Drone_Fleet.ipynb').exists()
    assert (MAIN/'episodes.csv').is_file()
    assert (FOLLOW_UP/'results/planner_episodes.csv').is_file()
    assert (FOLLOW_UP/'results/sensitivity_summary.csv').is_file()
