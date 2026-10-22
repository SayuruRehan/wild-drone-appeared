"""Check the existing notebook kernel, independently of pytest's src path injection."""

import subprocess
import sys
from pathlib import Path


def test_install_cell_activates_sources_in_already_running_kernel(tmp_path):
    repository = Path(__file__).parents[1]
    script = r'''
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
site_path = Path(sys.argv[2])
site_path.mkdir()
sys.path = [p for p in sys.path if p and Path(p).resolve() not in [REPO, REPO/'src']]
sys.path.append(str(site_path))
assert importlib.util.find_spec('wild_drone') is None

# The editable installer writes this file AFTER kernel startup. Child verification
# sees it at its own startup; the running kernel does not process it automatically.
def install_or_check(command, **kwargs):
    if 'pip' in command:
        (site_path/'editable.pth').write_text(str(REPO/'src')+'\n')
    return subprocess.CompletedProcess(command, 0)
subprocess.run = install_or_check
notebook = json.loads((REPO/'notebooks/colab_experiments.ipynb').read_text())
exec(compile(''.join(notebook['cells'][4]['source']), 'colab-install-cell', 'exec'))
from wild_drone.config import EnvConfig
import wild_drone
assert Path(wild_drone.__file__).resolve() == REPO/'src/wild_drone/__init__.py'
assert EnvConfig(grid_size=8).grid_size == 8
'''
    result = subprocess.run([sys.executable, '-c', script, str(repository),
                             str(tmp_path/'site')], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
