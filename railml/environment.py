"""Synthetic single-machine tamping MDP; all parameters are project assumptions."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    segments: int = 10
    horizon: int = 60
    segment_km: float = 5.0
    threshold: float = 6.0
    tamp_minutes: float = 45.0
    speed_km_min: float = 0.6
    restoration: float = 0.35
    gamma_shape: float = 2.0
    deterioration_multiplier: float = 1.0
    closure_probability: float = 0.15

    def __post_init__(self):
        if not 2 <= self.segments <= 30 or not 2 <= self.horizon <= 300:
            raise ValueError("segments must be 2..30 and horizon 2..300")
        if not 0 <= self.closure_probability <= 1:
            raise ValueError("closure_probability must be 0..1")
        if not 0 < self.restoration < 1:
            raise ValueError("restoration must be between 0 and 1")
        if any(x <= 0 for x in (self.segment_km, self.threshold, self.tamp_minutes,
                               self.speed_km_min, self.gamma_shape,
                               self.deterioration_multiplier)):
            raise ValueError("physical parameters must be positive")


class MaintenanceEnv:
    """Action 0 waits; action i+1 tamps segment i. Invalid actions are rejected.

    Exogenous degradation, windows and possessions are sampled before actions,
    giving every policy the same scenario for a given seed. Future samples are
    not part of observations or baseline decisions.
    """
    def __init__(self, config=None):
        self.config = config or Config()
        self.n = self.config.segments
        self.actions = self.n + 1
        self.obs_size = 5 * self.n + 2

    def reset(self, seed=0):
        c = self.config
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.t = 0
        self.position = int(rng.integers(self.n))
        self.condition = rng.uniform(1.5, 5.8, self.n)
        self.rates = rng.uniform(.07, .20, self.n) * c.deterioration_multiplier
        self._increments = rng.gamma(c.gamma_shape, self.rates / c.gamma_shape,
                                     (c.horizon, self.n))
        self._windows = rng.choice([75., 90., 120.], c.horizon)
        self._closures = rng.random((c.horizon, self.n)) < c.closure_probability
        self.history = []
        self.initial = self.snapshot()
        return self.observation(), self.mask()

    @property
    def done(self):
        return self.t >= self.config.horizon

    def travel(self):
        return abs(np.arange(self.n) - self.position) * self.config.segment_km

    def mask(self):
        if self.done:
            return np.array([True] + [False] * self.n)
        feasible = ((self.travel() / self.config.speed_km_min
                     + self.config.tamp_minutes) <= self._windows[self.t])
        feasible &= ~self._closures[self.t]
        return np.r_[True, feasible]

    def observation(self):
        c = self.config
        # Five per-track channels plus remaining horizon and available time.
        channels = np.stack([
            self.condition / c.threshold,
            self.rates / .20,
            np.eye(self.n)[self.position],
            self.mask()[1:].astype(float),
            self.travel() / (self.n * c.segment_km),
        ])
        window = 0. if self.done else self._windows[self.t] / 120.
        return np.r_[channels.ravel(), 1 - self.t / c.horizon, window].astype(np.float32)

    def expected_after(self, action):
        after = self.condition.copy()
        if action:
            after[action - 1] *= self.config.restoration
        return after + self.rates

    def action_cost(self, action):
        if not action:
            return 0., 0., 0.
        km = float(self.travel()[action - 1])
        minutes = self.config.tamp_minutes + km / self.config.speed_km_min
        # Synthetic cost units, not currency or estimated railway costs.
        return 10. + .20 * km, km, minutes

    def penalty(self, after, cost):
        """Common objective: cost + condition burden + overdue severity."""
        return (cost / 10. + .50 * float(np.mean(after))
                + 2.0 * float(np.maximum(after - self.config.threshold, 0).sum()))

    def snapshot(self):
        return {
            "window": self.t, "condition": self.condition.tolist(),
            "position": self.position, "mask": self.mask().tolist(),
            "available_minutes": 0. if self.done else float(self._windows[self.t]),
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode finished; reset before stepping")
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool):
            raise ValueError("Action must be an integer")
        if not 0 <= action < self.actions or not self.mask()[action]:
            raise ValueError("Action violates possession or maintenance window")
        before = self.snapshot()
        cost, km, minutes = self.action_cost(action)
        if action:
            self.condition[action - 1] *= self.config.restoration
            self.position = action - 1
        self.condition += self._increments[self.t]
        reward = -self.penalty(self.condition, cost)
        self.t += 1
        row = {
            **before, "action": int(action), "cost": cost, "travel_km": km,
            "minutes_used": minutes, "reward": reward,
            "after": self.condition.tolist(), "next_position": self.position,
            "overdue": int((self.condition > self.config.threshold).sum()),
            "mean_condition": float(self.condition.mean()),
        }
        self.history.append(row)
        return self.observation(), reward, self.done, self.mask()

    def result(self):
        h = self.history
        return {
            "seed": self.seed, "config": asdict(self.config),
            "initial": self.initial, "trace": h,
            "metrics": {
                "return": sum(r["reward"] for r in h),
                "cost": sum(r["cost"] for r in h),
                "mean_condition": float(np.mean([r["mean_condition"] for r in h])) if h else 0.,
                "overdue_segment_windows": sum(r["overdue"] for r in h),
                "travel_km": sum(r["travel_km"] for r in h),
                "maintenance_actions": sum(r["action"] > 0 for r in h),
                "constraint_violations": 0,  # invalid actions cannot enter a trace
            },
        }
