import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize('rerun', [False, True])
def test_submission_displays_results_from_the_selected_run(tmp_path, monkeypatch, rerun):
    import matplotlib.pyplot as plt
    import pandas as pd

    notebook = json.loads((ROOT/'Wildfire_Drone_Fleet.ipynb').read_text())
    archive = ROOT/'docs/experiments/final-2026-10-05'
    dest = tmp_path/'docs/experiments/final-2026-10-05'
    dest.mkdir(parents=True)
    # Tiny, distinguishable datasets let us check what is displayed without retraining.
    for filename in ['tuning.csv', 'test_summary.csv', 'sensitivity_summary.csv']:
        shutil.copyfile(archive/filename, dest/filename)
    for stage, filename, metric, sentinel in [
        ('tune', 'tuning.csv', 'final_validation_return', 123),
        ('test', 'test_summary.csv', 'return_mean', 456),
        ('sensitivity', 'sensitivity_summary.csv', 'return_mean', 789),
    ]:
        fresh = tmp_path/'runs/final'/stage
        fresh.mkdir(parents=True)
        frame = pd.read_csv(dest/filename)
        frame[metric] = sentinel
        frame.to_csv(fresh/filename, index=False)
    commands = []
    monkeypatch.setattr('subprocess.run', lambda args, **kw: commands.append(args))
    monkeypatch.setattr(plt, 'show', lambda: plt.close('all'))
    source = ''.join(notebook['cells'][2]['source'])
    source = source.replace('REPO = Path.cwd()', f'REPO = Path({str(tmp_path)!r})')
    source = source.replace('RUN_EXPERIMENTS = False', f'RUN_EXPERIMENTS = {rerun}')
    ns = {}
    exec(compile(source, 'setup', 'exec'), ns)
    for index in [23, 26, 29]:
        exec(compile(''.join(notebook['cells'][index]['source']), str(index), 'exec'), ns)
    if rerun:
        assert set(ns['tuning']['final_validation_return']) == {123}
        assert set(ns['summary']['return_mean']) == {456}
        assert set(ns['sens']['return_mean']) == {789}
        for command in commands:
            assert '--out' in command
            assert Path(command[command.index('--out')+1]) == tmp_path/'runs/final'
    else:
        assert not commands
        assert ns['summary']['return_mean'].max() < 0
    plt.close('all')


def test_documented_sensitivity_command_reproduces_the_archived_algorithm():
    notebook = json.loads((ROOT/'Wildfire_Drone_Fleet.ipynb').read_text())
    setup = ''.join(notebook['cells'][1]['source'])
    command = next(line for line in setup.splitlines() if 'run_final_study.py sensitivity' in line)
    assert '--algorithm dqn' in command
