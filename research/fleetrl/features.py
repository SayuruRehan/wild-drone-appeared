"""Compact state snapshots (for replay) and batched torch features for any fleet size.

Replay stores only the raw state needed to rebuild features (an urgency map, fire and
priority maps and per-drone scalars), which is independent of the number of drones and
small enough for 100k+ transitions on 40x40. Features are built on the training device.

Per-drone inputs
  local : (6, L, L) egocentric crop, L = 2*local_radius+1 — urgency, burning, priority,
          other drones, base, inside-grid
  glob  : (6, G, G) coarse whole-grid map — urgency, burning, priority, all drones,
          own position, base
  vec   : (10,) battery, return-cost ratio, base offset, time left, lost, charger
          owner/free, queue length, on pad
  act   : (5, 4) per-command features — immediate own coverage of urgency, distance to
          base after the move, move indicator, stays-on-grid
Pairwise domain features (structured interaction model)
  overlap  : (P, 5, 5) urgency covered by both drones' footprints after (a_i, a_j)
             — exactly the pairwise inclusion-exclusion term of the next reward
  charger  : (P, 5, 5) both drones end on the charger pad or move toward it
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from fleetrl.env import MOVES
from fleetrl.selection import pairs

MAX_WEIGHT = 9.0  # priority (<=3) * (1 + 2 burning) — upper bound of priority_weights


@dataclass(frozen=True)
class FeatureSpec:
    grid_size: int
    sensor_radius: int
    inspection_deadline: int
    battery_capacity: float
    move_cost: float
    horizon: int
    local_radius: int = 8
    global_size: int = 8

    @classmethod
    def from_config(cls, config, **kw):
        return cls(config.grid_size, config.sensor_radius, config.inspection_deadline,
                   float(config.battery_capacity), float(config.move_cost), config.horizon,
                   **kw)


def snapshot(env):
    """Raw state needed by ``build_features``; numpy, compact dtypes."""
    c = env.config
    urgency = env.priority_weights()*np.minimum((env.ages+1)/c.inspection_deadline, 1)
    return {
        'urgency': (urgency/MAX_WEIGHT).astype(np.float16),
        'burning': env.burning.astype(np.uint8),
        'priority': env.priority.astype(np.uint8),
        'positions': env.positions.astype(np.int16),
        'batteries': env.batteries.astype(np.float32),
        'lost': env.lost.copy(),
        'owner': np.int16(env.charger_owner),
        'elapsed': np.int16(env.elapsed),
    }


def collate(snapshots, device):
    out = {}
    for key in snapshots[0]:
        arr = np.stack([s[key] for s in snapshots])
        out[key] = torch.as_tensor(arr, device=device)
    out['urgency'] = out['urgency'].float()
    out['burning'] = out['burning'].float()
    out['priority'] = out['priority'].float()/3
    out['positions'] = out['positions'].long()
    out['lost'] = out['lost'].bool()
    out['owner'] = out['owner'].long()
    out['elapsed'] = out['elapsed'].float()
    return out


def _box_sums(integral, lo_r, lo_c, hi_r, hi_c):
    """Sums of the map over inclusive rectangles; empty when lo > hi.
    integral: (B, n+1, n+1); bounds: (B, ...) long tensors."""
    b = integral.shape[0]
    shape = lo_r.shape
    flat = integral.reshape(b, -1)
    m = integral.shape[-1]
    empty = (lo_r > hi_r) | (lo_c > hi_c)
    hi_r1, hi_c1 = (hi_r+1).clamp(min=0), (hi_c+1).clamp(min=0)
    lo_r, lo_c = lo_r.clamp(min=0), lo_c.clamp(min=0)

    def at(r, c):
        return flat.gather(1, (r*m+c).reshape(b, -1)).reshape(shape)
    total = at(hi_r1, hi_c1) - at(lo_r, hi_c1) - at(hi_r1, lo_c) + at(lo_r, lo_c)
    return torch.where(empty, torch.zeros_like(total), total)


def build_features(state, spec: FeatureSpec):
    """Batched features from collated snapshots. Returns a dict of tensors."""
    urg, burn, prio = state['urgency'], state['burning'], state['priority']
    pos, bat, lost = state['positions'], state['batteries'], state['lost']
    b, n, _ = urg.shape
    k = pos.shape[1]
    dev = urg.device
    r = spec.sensor_radius
    base = torch.tensor([n // 2, n // 2], device=dev)
    alive = (~lost).float()

    occupancy = torch.zeros(b, k, n, n, device=dev)
    bidx = torch.arange(b, device=dev)[:, None].expand(b, k)
    didx = torch.arange(k, device=dev)[None].expand(b, k)
    occupancy[bidx, didx, pos[..., 0], pos[..., 1]] = alive
    all_occ = occupancy.sum(1)
    base_map = torch.zeros(b, n, n, device=dev)
    base_map[:, n // 2, n // 2] = 1

    # ---- local egocentric crops
    R = spec.local_radius
    L = 2*R+1
    shared = torch.stack([urg, burn, prio, base_map, torch.ones_like(urg)], 1)   # (B,5,n,n)
    padded = F.pad(shared, (R, R, R, R))
    occ_pad = F.pad(all_occ[:, None], (R, R, R, R))[:, 0]
    offs = torch.arange(L, device=dev)
    rows = pos[..., 0, None] + offs                                             # (B,k,L)
    cols = pos[..., 1, None] + offs
    bi = torch.arange(b, device=dev)[:, None, None, None]
    crop = padded.permute(0, 2, 3, 1)[bi, rows[..., :, None], cols[..., None, :]]  # B,k,L,L,5
    crop = crop.permute(0, 1, 4, 2, 3)
    others = occ_pad[bi, rows[..., :, None], cols[..., None, :]]                 # B,k,L,L
    centre = torch.zeros(L, L, device=dev)
    centre[R, R] = 1
    others = others - alive[..., None, None]*centre
    local = torch.cat([crop[:, :, :3], others[:, :, None], crop[:, :, 3:]], 2)   # B,k,6,L,L

    # ---- coarse global maps
    G = spec.global_size
    pooled = F.adaptive_avg_pool2d(torch.stack([urg, burn, prio, all_occ, base_map], 1), G)
    pooled[:, 3:] = pooled[:, 3:]*(n*n/(G*G))  # counts per coarse cell
    own = F.adaptive_avg_pool2d(occupancy, G)*(n*n/(G*G))                       # B,k,G,G
    glob = torch.cat([pooled[:, None, :3].expand(b, k, 3, G, G), pooled[:, None, 3:4].expand(
        b, k, 1, G, G), own[:, :, None], pooled[:, None, 4:5].expand(b, k, 1, G, G)], 2)

    # ---- per-drone vectors
    dist_home = (pos-base).abs().sum(-1).float()
    on_pad = dist_home == 0
    owner = state['owner']
    queue = (on_pad & (bat < spec.battery_capacity) & ~lost).float().sum(1, keepdim=True)
    vec = torch.stack([
        bat/spec.battery_capacity,
        (dist_home*spec.move_cost/bat.clamp(min=1e-3)).clamp(max=4)/4,
        (base[0]-pos[..., 0]).float()/n,
        (base[1]-pos[..., 1]).float()/n,
        (1-state['elapsed']/spec.horizon)[:, None].expand(b, k),
        lost.float(),
        (owner[:, None] == torch.arange(k, device=dev)[None]).float(),
        (owner < 0).float()[:, None].expand(b, k),
        (queue/k).expand(b, k),
        on_pad.float(),
    ], -1)

    # ---- per-command features
    moves = torch.as_tensor(MOVES, device=dev)
    proposed = pos[:, :, None, :] + moves[None, None]                           # B,k,5,2
    in_grid = ((proposed >= 0) & (proposed < n)).all(-1)
    proposed = torch.where(in_grid[..., None], proposed, pos[:, :, None, :])
    integral = F.pad(urg.cumsum(1).cumsum(2), (1, 0, 1, 0))
    area = float((2*r+1)**2)
    lo = (proposed-r).clamp(min=0)
    hi = (proposed+r).clamp(max=n-1)
    own_gain = _box_sums(integral, lo[..., 0], lo[..., 1], hi[..., 0], hi[..., 1])/area
    own_gain = own_gain*alive[..., None]
    new_home = (proposed-base).abs().sum(-1).float()
    act = torch.stack([own_gain, new_home/n, (moves.abs().sum(-1) > 0).float()[None, None].expand(
        b, k, 5), in_grid.float()], -1)                                         # B,k,5,4

    # ---- pairwise domain features
    pidx = pairs(k)
    if pidx:
        ii = torch.tensor([p[0] for p in pidx], device=dev)
        jj = torch.tensor([p[1] for p in pidx], device=dev)
        lo_i, hi_i = lo[:, ii][:, :, :, None], hi[:, ii][:, :, :, None]          # B,P,5,1,2
        lo_j, hi_j = lo[:, jj][:, :, None], hi[:, jj][:, :, None]                # B,P,1,5,2
        lo_ij = torch.maximum(lo_i, lo_j)
        hi_ij = torch.minimum(hi_i, hi_j)
        overlap = _box_sums(integral, lo_ij[..., 0], lo_ij[..., 1], hi_ij[..., 0],
                            hi_ij[..., 1])/area
        both_alive = (alive[:, ii]*alive[:, jj])[:, :, None, None]
        overlap = overlap*both_alive
        toward = ((new_home < dist_home[..., None]) | (new_home == 0)).float()  # B,k,5
        charger = toward[:, ii][:, :, :, None]*toward[:, jj][:, :, None]*both_alive
        delta = (pos[:, jj]-pos[:, ii]).float()
        cheb = delta.abs().max(-1).values
        pair_vec = torch.stack([delta[..., 0]/n, delta[..., 1]/n,
                                (cheb/(2*r+2)).clamp(max=3)/3,
                                (bat[:, ii]-bat[:, jj]).abs()/spec.battery_capacity], -1)
    else:
        overlap = charger = pair_vec = None
    return {'local': local, 'glob': glob, 'vec': vec, 'act': act, 'overlap': overlap,
            'charger': charger, 'pair_vec': pair_vec, 'pairs': pidx}
