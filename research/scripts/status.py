"""One-screen progress report for a run folder.

    python3 research/scripts/status.py /workspace/wild-drone-runs/main40-r2

Shows, per run: state (running / done / waiting / stopped), latest evaluated step, minutes,
development freshness at N = 3 and 5, drone losses and safety-guard trips, plus an ETA.
"""

import json
import subprocess
import sys
from pathlib import Path

REFERENCE = {  # development-world references (research/results/headroom)
    'main40': {'greedy': (0.222, 0.273), 'masked random': (0.088, 0.117)},
    'dev20': {'greedy': (0.330, 0.405), 'masked random': (0.143, 0.193)},
}


def running_dirs():
    try:
        out = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    except OSError:
        return set()
    dirs = set()
    for line in out.splitlines():
        if 'fleetrl.train' in line and '--out' in line:
            parts = line.split()
            dirs.add(Path(parts[parts.index('--out')+1]).resolve())
    return dirs


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    if not root.exists():
        sys.exit(f'{root} does not exist (nothing launched yet?)')
    live = running_dirs()
    env_name = None
    rows = []
    for run in sorted(p for p in root.iterdir() if (p/'config.json').exists()):
        cfg = json.load(open(run/'config.json'))
        env_name = cfg.get('env', env_name)
        lines = [json.loads(x) for x in open(run/'log.jsonl')] if (run/'log.jsonl').exists() else []
        last = lines[-1] if lines else {}
        state = ('done' if (run/'done.json').exists() else
                 'running' if run.resolve() in live else 'stopped')
        losses = sum(v for k, v in last.items() if k.startswith('dev_losses'))
        eta = ''
        if state == 'running' and last.get('steps'):
            per_step = last['minutes']/last['steps']
            eta = f"~{(cfg['steps']-last['steps'])*per_step/60:.1f} h left"
        rows.append((run.name, state, last.get('steps', 0), cfg['steps'], last.get('minutes', 0),
                     last.get('dev_freshness_n3'), last.get('dev_freshness_n5'), losses,
                     last.get('guard_trips', '-'), eta))
    print(f"{'run':24} {'state':8} {'steps':>15} {'min':>6} {'N3':>6} {'N5':>6} "
          f"{'losses':>6} {'guard':>5}  eta")
    for name, state, steps, total, minutes, n3, n5, losses, guard, eta in rows:
        f3 = f'{n3:.3f}' if n3 is not None else '  -  '
        f5 = f'{n5:.3f}' if n5 is not None else '  -  '
        print(f'{name:24} {state:8} {steps:>7}/{total:<7} {minutes:6.0f} {f3:>6} {f5:>6} '
              f'{losses:>6} {guard!s:>5}  {eta}')
    if not rows:
        print('(no runs started yet — see launcher.log)')
    ref = REFERENCE.get(env_name)
    if ref:
        print('\nreference (development worlds) N3 / N5: ' +
              ', '.join(f'{k} {a:.3f} / {b:.3f}' for k, (a, b) in ref.items()))
    waiting = [r for r in rows if r[1] == 'stopped' and r[2] == 0]
    print(f"\n{sum(r[1] == 'done' for r in rows)} done, "
          f"{sum(r[1] == 'running' for r in rows)} running, "
          f"{sum(r[1] == 'stopped' for r in rows)} not running "
          f"({len(waiting)} not started yet). Runs not yet created by the launcher are not listed.")
    log = root/'launcher.log'
    if log.exists():
        tail = log.read_text().strip().splitlines()[-3:]
        print('launcher.log (last lines):\n  ' + '\n  '.join(tail))
    if any(r[7] for r in rows):
        print('\nWARNING: drone losses recorded. Keep the run folders; snapshots/ hold the exact '
              'models for scripts/diagnose_losses.py.')


if __name__ == '__main__':
    main()
