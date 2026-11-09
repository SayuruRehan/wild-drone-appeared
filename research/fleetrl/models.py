"""Value networks. Every model maps features to

    Q(s, a) = V(s) + sum_i u_i(a_i) + sum_{i<j} P_ij(a_i, a_j)

and differs only in the pairwise term P (and the joint model, which has no factorisation):

  vdn          P = 0                                         (Sunehag et al., 2018)
  dcg          P = learned 5x5 table per pair                (Boehmer et al., 2020)
  structured   P = alpha_ij(s) * Overlap + beta_ij(s) * Charger      (this study)
  structured+  structured + learned residual table             (ablation)
  structured-overlap / structured-charger   only one domain feature (ablations)
  joint        dueling 5**N head on concatenated drone embeddings (N fixed; baseline)
  qmix         monotonic state-conditioned mixing of per-drone utilities
               (Rashid et al., 2018); per-drone hypernetwork weights, so any N

model_version 1 is the pilot architecture (DCG as an unnormalised full 5x5 table per
pair). model_version 2 (default) makes DCG follow Boehmer et al. (2020): utilities and
payoffs are averaged over drones and edges, and payoffs are low-rank (rank ``dcg_rank``).

V(s) is a state baseline from mean-pooled drone embeddings (dueling-style); it does not
change the greedy action but absorbs the large, mostly action-independent part of the
return (the overdue penalty), so the per-drone terms only need to model action differences.
All factored models are permutation-equivariant and accept any number of drones.
"""

import torch
from torch import nn

from fleetrl.selection import torch_dense_q, torch_gather_joint

KINDS = ('vdn', 'dcg', 'structured', 'structured+', 'structured-overlap', 'structured-charger',
         'joint', 'qmix')
STRUCTURED = ('structured', 'structured+', 'structured-overlap', 'structured-charger')
MIX_EMBED = 32


class DroneEncoder(nn.Module):
    def __init__(self, local_size, global_size, hidden=128):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        self.glob = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            nl = self.local(torch.zeros(1, 6, local_size, local_size)).shape[1]
            ng = self.glob(torch.zeros(1, 6, global_size, global_size)).shape[1]
        self.mlp = nn.Sequential(nn.Linear(nl+ng+10, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, local, glob, vec):
        b, k = vec.shape[:2]
        x = torch.cat([self.local(local.flatten(0, 1)), self.glob(glob.flatten(0, 1)),
                       vec.flatten(0, 1)], 1)
        return self.mlp(x).view(b, k, -1)


class FleetQ(nn.Module):
    def __init__(self, kind, local_size, global_size, hidden=128, n_drones=None,
                 model_version=2, dcg_rank=1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f'kind must be one of {KINDS}')
        if kind == 'joint' and not n_drones:
            raise ValueError('the joint model needs a fixed n_drones')
        if model_version not in (1, 2):
            raise ValueError('model_version must be 1 or 2')
        self.kind, self.hidden, self.n_drones = kind, hidden, n_drones
        self.faithful_dcg = kind == 'dcg' and model_version == 2
        self.dcg_rank = dcg_rank
        self.encoder = DroneEncoder(local_size, global_size, hidden)
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        if kind == 'joint':
            self.joint = nn.Sequential(nn.Linear(n_drones*(hidden+20), 256), nn.ReLU(),
                                       nn.Linear(256, 5**n_drones))
            return
        self.utility = nn.Sequential(nn.Linear(hidden+4+5, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 1))
        pair_in = 2*hidden+4
        if kind == 'qmix':
            # per-drone first-layer weights from (drone, state); state-conditioned rest
            self.hyper_w1 = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, MIX_EMBED))
            self.hyper_b1 = nn.Linear(hidden, MIX_EMBED)
            self.hyper_w2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                          nn.Linear(hidden, MIX_EMBED))
            return
        if self.faithful_dcg:
            self.residual = nn.Sequential(nn.Linear(pair_in, hidden), nn.ReLU(),
                                          nn.Linear(hidden, 10*dcg_rank))
        elif kind in ('dcg', 'structured+'):
            self.residual = nn.Sequential(nn.Linear(pair_in, hidden), nn.ReLU(),
                                          nn.Linear(hidden, 25))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)
        if kind in STRUCTURED:
            self.coeff = nn.Sequential(nn.Linear(pair_in, hidden), nn.ReLU(),
                                       nn.Linear(hidden, 2))
            # Zero-initialised like the dcg residual, so every factored model starts as VDN
            # and any pairwise effect has to be learned from data.
            nn.init.zeros_(self.coeff[-1].weight)
            nn.init.zeros_(self.coeff[-1].bias)

    # -------------------------------------------------------------- components
    def components(self, f):
        """Returns (V (B,), unary (B,N,5), pair_values (B,P,5,5) or None).
        For the joint model returns (V, joint advantages (B, 5**N), None)."""
        h = self.encoder(f['local'], f['glob'], f['vec'])                    # B,N,H
        b, k, _ = h.shape
        v = self.value(h.mean(1)).squeeze(-1)
        if self.kind == 'joint':
            if k != self.n_drones:
                raise ValueError(f'joint model trained for {self.n_drones} drones, got {k}')
            x = torch.cat([h, f['act'].flatten(2)], -1).flatten(1)
            adv = self.joint(x)
            return v, adv - adv.mean(1, keepdim=True), None
        eye = torch.eye(5, device=h.device).expand(b, k, 5, 5)
        u_in = torch.cat([h[:, :, None].expand(b, k, 5, self.hidden), f['act'], eye], -1)
        unary = self.utility(u_in).squeeze(-1)                               # B,N,5
        if self.kind == 'qmix':
            state = h.mean(1)
            w1 = self.hyper_w1(torch.cat([h, state[:, None].expand_as(h)], -1)).abs()
            mix = (w1, self.hyper_b1(state), self.hyper_w2(state).abs())
            return v, unary, mix                     # pair slot carries the mixer
        unary = unary - unary.mean(-1, keepdim=True)
        if self.faithful_dcg:
            unary = unary/k
        if self.kind == 'vdn' or not f['pairs']:
            return v, unary, None
        ii = [p[0] for p in f['pairs']]
        jj = [p[1] for p in f['pairs']]
        hi, hj, pv = h[:, ii], h[:, jj], f['pair_vec']
        flip = pv.clone()
        flip[..., :2] = -flip[..., :2]
        fwd = torch.cat([hi, hj, pv], -1)
        rev = torch.cat([hj, hi, flip], -1)
        pair = 0
        if self.kind in STRUCTURED:
            c = 0.5*(self.coeff(fwd)+self.coeff(rev))                         # B,P,2
            if self.kind != 'structured-charger':
                pair = pair + c[..., 0, None, None]*f['overlap']
            if self.kind != 'structured-overlap':
                pair = pair + c[..., 1, None, None]*f['charger']
        if self.faithful_dcg:
            r = self.dcg_rank

            def payoff(x):  # low-rank 5x5 table: sum_k p_k(a_i) q_k(a_j)
                pq = self.residual(x).view(*x.shape[:2], 2, r, 5)
                return torch.einsum('bpka,bpkc->bpac', pq[:, :, 0], pq[:, :, 1])
            pair = 0.5*(payoff(fwd)+payoff(rev).transpose(-1, -2))/len(f['pairs'])
        elif self.kind in ('dcg', 'structured+'):
            rf = self.residual(fwd).view(*fwd.shape[:2], 5, 5)
            rr = self.residual(rev).view(*fwd.shape[:2], 5, 5).transpose(-1, -2)
            pair = pair + 0.5*(rf+rr)
        return v, unary, pair

    @staticmethod
    def _mix(hidden, b1, w2):
        """QMIX output from first-layer pre-activations hidden (..., E)."""
        return (nn.functional.elu(hidden + b1)*w2).sum(-1)

    def dense_from(self, parts, pair_index):
        v, unary, pair = parts
        if self.kind == 'joint':
            return v[:, None] + unary
        if self.kind == 'qmix':
            w1, b1, w2 = pair
            b, k, _ = unary.shape
            z = unary[..., None]*w1[:, :, None, :]                              # B,N,5,E
            total = z.new_zeros((b,)+(5,)*k+(MIX_EMBED,))
            for i in range(k):
                shape = [b]+[1]*k+[MIX_EMBED]
                shape[1+i] = 5
                total = total + z[:, i].reshape(shape)
            total = total.reshape(b, -1, MIX_EMBED)
            return v[:, None] + self._mix(total, b1[:, None], w2[:, None])
        return v[:, None] + torch_dense_q(unary, pair, pair_index)

    def gather_from(self, parts, pair_index, commands):
        v, unary, pair = parts
        if self.kind == 'joint':
            k = commands.shape[1]
            weights = 5**torch.arange(k-1, -1, -1, device=commands.device)
            return v + unary.gather(1, (commands*weights).sum(1, keepdim=True)).squeeze(1)
        if self.kind == 'qmix':
            w1, b1, w2 = pair
            chosen = unary.gather(2, commands[:, :, None])                       # B,N,1
            return v + self._mix((chosen*w1).sum(1), b1, w2)
        return v + torch_gather_joint(unary, pair, pair_index, commands)

    def dense_q(self, f):
        return self.dense_from(self.components(f), f['pairs'])

    def q_of(self, f, commands):
        """Q of the given joint actions; commands (B, N) long."""
        return self.gather_from(self.components(f), f['pairs'], commands)
