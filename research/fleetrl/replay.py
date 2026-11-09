"""Preallocated replay buffer that stores each state once and can be saved to disk.

States go into a ring of snapshot arrays (one row per environment state, including the
reset state of every episode); transitions refer to their start and bootstrap states by a
global state id. An n-step transition (s_t, a_t, G, gamma^n, s_{t+n}) therefore costs one
state row, not two, and sampling is plain fancy indexing (no per-sample Python objects).
"""

import numpy as np


class Replay:
    def __init__(self, capacity, n_drones):
        self.capacity = int(capacity)
        self.n_drones = n_drones
        # each episode adds one state more than it adds transitions; leave room for that
        self.state_capacity = self.capacity + self.capacity//20 + 2_000
        self.states = None
        self.state_count = 0          # global ids handed out so far
        self.size = 0                 # valid transitions
        self.cursor = 0               # next transition slot
        self.start = np.zeros(self.capacity, dtype=np.int64)
        self.next = np.zeros(self.capacity, dtype=np.int64)
        self.actions = np.zeros((self.capacity, n_drones), dtype=np.int8)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.next_masks = np.zeros((self.capacity, 5**n_drones), dtype=bool)

    def __len__(self):
        return self.size

    def add_state(self, snap):
        if self.states is None:
            self.states = {k: np.zeros((self.state_capacity,)+np.shape(v), dtype=np.asarray(v).dtype)
                           for k, v in snap.items()}
        row = self.state_count % self.state_capacity
        for k, v in snap.items():
            self.states[k][row] = v
        self.state_count += 1
        return self.state_count - 1

    def add(self, start_id, commands, ret, discount, next_id, next_mask):
        i = self.cursor
        self.start[i], self.next[i] = start_id, next_id
        self.actions[i] = commands
        self.returns[i], self.discounts[i] = ret, discount
        self.next_masks[i] = next_mask
        self.cursor = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def _rows(self, ids):
        if np.any(self.state_count - ids > self.state_capacity):
            raise RuntimeError('replay state ring overwritten a referenced state')
        return {k: v[ids % self.state_capacity] for k, v in self.states.items()}

    def sample(self, rng, batch_size):
        idx = rng.integers(0, self.size, batch_size)
        return (self._rows(self.start[idx]), self.actions[idx].astype(np.int64),
                self.returns[idx], self.discounts[idx], self._rows(self.next[idx]),
                self.next_masks[idx])

    # ---------------------------------------------------------------- persistence
    def state_dict(self):
        out = {'capacity': self.capacity, 'n_drones': self.n_drones,
               'state_count': self.state_count, 'size': self.size, 'cursor': self.cursor,
               'start': self.start, 'next': self.next, 'actions': self.actions,
               'returns': self.returns, 'discounts': self.discounts,
               'next_masks': np.packbits(self.next_masks, axis=1)}
        if self.states is not None:
            out.update({'state_'+k: v for k, v in self.states.items()})
        return out

    @classmethod
    def from_state_dict(cls, d):
        r = cls(int(d['capacity']), int(d['n_drones']))
        for key in ('state_count', 'size', 'cursor'):
            setattr(r, key, int(d[key]))
        for key in ('start', 'next', 'actions', 'returns', 'discounts'):
            getattr(r, key)[:] = d[key]
        r.next_masks[:] = np.unpackbits(d['next_masks'], axis=1,
                                        count=5**r.n_drones).astype(bool)
        keys = [k for k in d if k.startswith('state_') and k != 'state_count']
        if keys:
            r.states = {k[len('state_'):]: np.array(d[k]) for k in keys}
        return r
