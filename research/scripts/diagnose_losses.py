"""Find and explain drone losses of a trained policy (safety check of the feasibility mask).

    PYTHONPATH=research:src python research/scripts/diagnose_losses.py \
        --run /content/drive/MyDrive/wild-drone-research/runs/pilot-main40/vdn-0 \
        --drones 5 --worlds 10
    # a specific evaluated model (runs started after this script keep one per evaluation):
    ... --checkpoint snapshots/step-0200000.pt

For every mission it reports losses, steps where the feasible set was empty (the mask then
falls back to "everyone stays"), and steps where the environment rejected a move. For the
first lost drone it prints that drone's last steps: position, distance to base, battery,
allowed commands, the chosen command and what the other drones nearby were doing.
Development worlds only.
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'src')]

from fleetrl.env import FleetEnv  # noqa: E402
from fleetrl.feasibility import dense_feasible, pairwise_conflicts, unary_allowed  # noqa: E402
from fleetrl.train import load_policy  # noqa: E402

NAMES = ['stay', 'up', 'down', 'left', 'right']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--checkpoint', default='final.pt')
    p.add_argument('--drones', type=int, nargs='+', default=[5])
    p.add_argument('--worlds', type=int, default=10)
    p.add_argument('--history', type=int, default=12)
    p.add_argument('--device', default='cpu', help="'cuda' reproduces in-training GPU evaluation")
    p.add_argument('--raw', action='store_true',
                   help='reproduce the pre-guard policy: plain masked argmax, no NaN handling')
    a = p.parse_args()
    torch.set_num_threads(1)
    policy, cfg, config = load_policy(a.run, a.checkpoint, a.device)
    nonfinite = {'steps': 0}
    if a.raw:
        from fleetrl.features import build_features, collate, snapshot
        from fleetrl.feasibility import feasible_mask

        def raw_act(env, policy=policy):
            with torch.no_grad():
                f = build_features(collate([snapshot(env)], a.device), policy.spec)
                q = policy.model.dense_q(f)[0]
                if not torch.isfinite(q).all():
                    nonfinite['steps'] += 1
                mask = torch.as_tensor(feasible_mask(env), device=a.device)
                index = int(q.masked_fill(~mask, float('-inf')).argmax())
            return np.array(np.unravel_index(index, (5,)*env.n_drones), dtype=np.int64)
        policy.act_commands = raw_act
    split = json.load(open(HERE/'configs/splits.json'))['development']
    explained = False
    totals = {'missions': 0, 'losses': 0, 'empty_mask_steps': 0, 'rejected_moves': 0}
    for n in a.drones:
        for seed in range(split['start'], split['start']+a.worlds):
            env = FleetEnv(config, n)
            env.reset(seed)
            history = deque(maxlen=a.history)
            empty = rejected = 0
            while not env.done:
                allowed, conflicts = unary_allowed(env), pairwise_conflicts(env)
                feasible = dense_feasible(allowed, conflicts)
                if not feasible.any():
                    empty += 1
                commands = policy.act_commands(env)
                history.append({
                    'step': env.elapsed, 'positions': env.positions.copy(),
                    'batteries': env.batteries.copy(), 'allowed': allowed.copy(),
                    'commands': commands.copy(), 'feasible_count': int(feasible.sum()),
                    'chosen_feasible': bool(feasible[tuple(commands)]),
                    'charger_owner': env.charger_owner})
                lost_before = env.lost.copy()
                _, _, info = env.step(commands)
                rejected += info['collision_attempts']
                newly = np.flatnonzero(env.lost & ~lost_before)
                if len(newly) and not explained:
                    explained = True
                    i = int(newly[0])
                    print(f'\n=== first loss: N={n} world={seed} drone={i} at step {env.elapsed} ===')
                    for h in history:
                        pos = h['positions'][i]
                        dist = int(np.abs(pos-env.base).sum())
                        near = [j for j in range(n) if j != i and
                                np.abs(h['positions'][j]-pos).sum() <= 2]
                        print(f"t={h['step']:3d} pos={pos.tolist()} dist={dist:2d} "
                              f"battery={h['batteries'][i]:6.2f} "
                              f"allowed={[NAMES[c] for c in np.flatnonzero(h['allowed'][i])]} "
                              f"chose={NAMES[h['commands'][i]]} feasible_joint={h['feasible_count']} "
                              f"chosen_feasible={h['chosen_feasible']} charger={h['charger_owner']} "
                              f"near={[(j, h['positions'][j].tolist(), NAMES[h['commands'][j]]) for j in near]}")
            totals['missions'] += 1
            totals['losses'] += int(env.lost.sum())
            totals['empty_mask_steps'] += empty
            totals['rejected_moves'] += rejected
            print(f'N={n} world={seed}: losses={int(env.lost.sum())} empty-mask steps={empty} '
                  f'rejected moves={rejected} freshness={env._metrics()["weighted_freshness"]:.3f}',
                  flush=True)
    totals['nonfinite_q_steps'] = nonfinite['steps']
    print('\nTOTAL', json.dumps(totals))


if __name__ == '__main__':
    main()
