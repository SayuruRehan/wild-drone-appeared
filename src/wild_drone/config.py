"""Illustrative simulator settings; not a physical wildfire calibration."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EnvConfig:
    grid_size: int = 40
    horizon: int = 300
    battery_capacity: float = 160.0
    move_cost: float = 1.0
    hover_cost: float = 0.25
    recharge_rate: float = 8.0
    sensor_radius: int = 2
    inspection_deadline: int = 30
    max_fuel: int = 180
    initial_fires: int = 3
    spread_probability: float = 0.04
    wind_multiplier: float = 2.0
    wind_change_probability: float = 0.06
    freshness_reward: float = 1.0
    overdue_penalty: float = 0.1
    energy_penalty: float = 0.02
    collision_penalty: float = 0.2
    loss_penalty: float = 5.0

    def __post_init__(self):
        for name in ('grid_size', 'horizon', 'inspection_deadline', 'max_fuel', 'initial_fires'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.grid_size < 4:
            raise ValueError('grid_size must be at least 4')
        if not isinstance(self.sensor_radius, int) or self.sensor_radius < 0:
            raise ValueError('sensor_radius must be a nonnegative integer')
        if self.initial_fires > self.grid_size**2 - 1:
            raise ValueError('too many initial fires')
        for name in ('battery_capacity', 'move_cost', 'hover_cost', 'recharge_rate'):
            if not 0 < getattr(self, name) < float('inf'):
                raise ValueError(f'{name} must be finite and positive')
        for name in ('spread_probability', 'wind_change_probability'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        if not 1 <= self.wind_multiplier < float('inf'):
            raise ValueError('wind_multiplier must be finite and at least 1')
        for name in ('freshness_reward', 'overdue_penalty', 'energy_penalty',
                     'collision_penalty', 'loss_penalty'):
            if not 0 <= getattr(self, name) < float('inf'):
                raise ValueError(f'{name} must be finite and nonnegative')

    def to_dict(self):
        return asdict(self)
