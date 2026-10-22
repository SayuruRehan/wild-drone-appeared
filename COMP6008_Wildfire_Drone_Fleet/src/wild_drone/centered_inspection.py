"""Matched supervised comparison of absolute and drone-centered inspection maps."""

import copy
import hashlib
from pathlib import Path

import torch

from wild_drone.inspection_targets import masked_target_loss
from wild_drone.learning import FleetNetwork
from wild_drone.training_io import atomic_torch_save

ARMS = ('current', 'absolute', 'centered')
FORMAT = 'wild-drone-centered-inspection-v1'


def transform_maps(maps, vector, config, arm):
    """Observation-only field, with no labels, action footprints or feasible mask.

    Retain the five world maps, base one-hot and full scalar vector. The two
    derived arms replace three drone one-hots with either the same absolute
    inspection field or one translated copy per drone. Out-of-grid and lost
    drone cells are zero. Translation crops the auxiliary plane; original world
    maps retain the full scene. There is no circular wrapping or action lookup.
    """
    n = config.grid_size
    if (arm not in ARMS or maps.ndim != 4 or maps.shape[1:] != (9, n, n) or
            vector.shape != (len(maps), 24)):
        raise ValueError('matching nine-map/24-vector inputs and valid arm required')
    if arm == 'current':
        return maps
    burning, fuel, priority, age = maps[:, 0], maps[:, 1], maps[:, 3], maps[:, 4]
    neighbours = torch.zeros_like(burning)
    neighbours[:, 1:] += burning[:, :-1]
    neighbours[:, :-1] += burning[:, 1:]
    neighbours[:, :, 1:] += burning[:, :, :-1]
    neighbours[:, :, :-1] += burning[:, :, 1:]
    frontier = (neighbours > 0) & (fuel > 0) & (burning < .5)
    age_limit = max(config.horizon, config.inspection_deadline+1)
    urgency = ((age*age_limit+1)/config.inspection_deadline).clamp(max=1)
    field = priority*(1+2*burning+frontier)/3*urgency
    drones = vector[:, :12].reshape(-1, 3, 4)
    positions = (drones[:, :, :2]*(n-1)).round().long()
    result = maps.clone()
    for drone in range(3):
        alive = (drones[:, drone, 3] < .5).reshape(-1, 1, 1)
        if arm == 'absolute':
            plane = field
        else:
            rows = torch.arange(n, device=maps.device).reshape(1, n, 1) + \
                positions[:, drone, 0].reshape(-1, 1, 1)-n//2
            cols = torch.arange(n, device=maps.device).reshape(1, 1, n) + \
                positions[:, drone, 1].reshape(-1, 1, 1)-n//2
            valid = (rows >= 0) & (rows < n) & (cols >= 0) & (cols < n)
            indices = rows.clamp(0, n-1)*n+cols.clamp(0, n-1)
            plane = field.flatten(1).gather(1, indices.flatten(1)).reshape(-1, n, n)*valid
        result[:, 5+drone] = plane*alive
    return result


class CenteredInspectionNetwork(FleetNetwork):
    def __init__(self, config, arm):
        if arm not in ARMS:
            raise ValueError('invalid inspection input arm')
        super().__init__(config.grid_size)
        self.env_config, self.arm = config, arm

    def forward(self, maps, vector):
        return super().forward(transform_maps(maps, vector, self.env_config, self.arm), vector)


def load_centered_model(path, config, arm):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('format') != FORMAT or saved.get('env_config') != config.to_dict() or
            saved.get('arm') != arm or arm not in ARMS):
        raise ValueError('matching supervised centered-inspection checkpoint required')
    model = CenteredInspectionNetwork(config, arm)
    model.load_state_dict(saved['state_dict'])
    return model.eval()


def fit_centered(config, data, *, arm, seed, updates, batch_size, output):
    if (arm not in ARMS or not isinstance(seed, int) or seed < 0 or
            not isinstance(updates, int) or updates < 1 or
            not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError('valid arm, nonnegative seed and positive fitting budget required')
    ids = torch.where(data['informative'])[0]
    if not len(ids):
        raise ValueError('no informative training states')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    model = CenteredInspectionNetwork(config, arm)
    initial = copy.deepcopy(model.state_dict())
    sampler = torch.Generator().manual_seed(seed+1000)
    optimiser = torch.optim.Adam(model.parameters(), lr=.001)
    history, identities = [], {}
    sequence = hashlib.sha256()

    def export(label, completed):
        path = output/f'{label}.pt'
        checkpoint = {'format': FORMAT, 'method': 'matched-supervised-inspection-inputs',
                      'env_config': config.to_dict(), 'arm': arm, 'seed': seed,
                      'completed_updates': completed, 'state_dict': model.state_dict()}
        atomic_torch_save(checkpoint, path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        assert {k: v for k, v in saved.items() if k != 'state_dict'} == {
            k: v for k, v in checkpoint.items() if k != 'state_dict'}
        loaded = load_centered_model(path, config, arm)
        if any(not torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()):
            raise ValueError('exported centered-inspection tensors differ')
        identities[path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'bytes': path.stat().st_size, 'completed_updates': completed}

    export('initial', 0)
    for update in range(1, updates+1):
        batch = ids[torch.randint(len(ids), (batch_size,), generator=sampler)]
        sequence.update(batch.numpy().astype('<i8').tobytes())
        scores = model(data['maps'][batch], data['vectors'][batch])[0]
        loss = masked_target_loss(scores, data['targets'][batch], data['masks'][batch])
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite centered-inspection loss')
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        optimiser.step()
        history.append({'update': update, 'training_batch_loss': float(loss.detach())})
    export('final', updates)
    return model.eval(), {'arm': arm, 'seed': seed, 'completed_updates': updates,
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'sample_sequence_sha256': sequence.hexdigest(),
                          'checkpoints': identities}, history, initial
