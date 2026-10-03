import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_progress_arrives_before_child_exits_and_failure_is_raised(tmp_path):
    """The first line must reach the notebook while the child is still waiting."""
    script = r'''
import subprocess, sys
from pathlib import Path
from wild_drone.notebook_runtime import run_with_progress
gate = Path(sys.argv[1])
def show(line):
    assert line == 'ready\n'
    gate.write_text('continue')
child = ("import sys,time;from pathlib import Path;print('ready',flush=True);"
         "p=Path(sys.argv[1]);\nwhile not p.exists():time.sleep(.01)\nsys.exit(7)")
try:
    run_with_progress([sys.executable, '-u', '-c', child, str(gate)], emit=show)
except subprocess.CalledProcessError as e:
    assert e.returncode == 7
else:
    raise AssertionError('nonzero exit was hidden')
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path/'gate')],
                   check=True, timeout=10)


@pytest.mark.parametrize('ignore_termination', [False, True])
def test_interrupted_progress_reaps_child_before_returning(tmp_path, ignore_termination):
    script = r'''
import os, signal, sys
from pathlib import Path
from wild_drone.notebook_runtime import run_with_progress
pid_file = Path(sys.argv[1])
def stop(line):
    pid_file.write_text(line.strip())
    os.kill(os.getpid(), signal.SIGINT)
try:
    child = 'import os,time,signal;'
    if sys.argv[2] == 'True':
        child += 'signal.signal(signal.SIGTERM, signal.SIG_IGN);'
    child += 'print(os.getpid(),flush=True);time.sleep(30)'
    run_with_progress([sys.executable, '-u', '-c', child], emit=stop)
except KeyboardInterrupt:
    pass
else:
    raise AssertionError('interrupt was hidden')
pid = int(pid_file.read_text())
try:
    os.kill(pid, 0)
except ProcessLookupError:
    pass
else:
    raise AssertionError('training child survived the interrupt')
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path/'pid'), str(ignore_termination)],
                   check=True, timeout=10)


def test_progress_returns_success_and_includes_stderr(capsys):
    from wild_drone.notebook_runtime import run_with_progress

    result = run_with_progress([sys.executable, '-u', '-c',
                               "import sys;print('out');print('err',file=sys.stderr)"])
    assert result.returncode == 0
    assert set(capsys.readouterr().out.splitlines()) == {'out', 'err'}


def test_colab_long_runs_use_unbuffered_progress():
    path = Path(__file__).parents[1]/'notebooks'/'colab_experiments.ipynb'
    notebook = json.loads(path.read_text())
    for index in [10, 14]:
        code = ''.join(notebook['cells'][index]['source'])
        assert 'run_with_progress(' in code
        assert "sys.executable, '-u'" in code
