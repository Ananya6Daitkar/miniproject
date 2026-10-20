import numpy as np
import pytest

from railml.environment import Config, MaintenanceEnv
from railml.baselines import cp_sat, run_episode


def test_same_scenario_uses_same_noise_even_for_different_actions():
    a, b = MaintenanceEnv(), MaintenanceEnv()
    a.reset(42)
    b.reset(42)
    before = a.condition.copy()
    action = int(np.flatnonzero(a.mask()[1:])[0] + 1)
    a.step(0)
    b.step(action)
    noise = a.condition - before
    expected = before.copy()
    expected[action - 1] *= a.config.restoration
    np.testing.assert_allclose(b.condition - expected, noise)


def test_closed_track_action_is_rejected_without_mutation():
    env = MaintenanceEnv(Config(closure_probability=1.))
    env.reset(1)
    initial = env.observation().copy()
    with pytest.raises(ValueError):
        env.step(1)
    np.testing.assert_array_equal(initial, env.observation())
    assert env.mask().tolist() == [True] + [False] * 10
    env.step(0)


def test_time_window_masks_far_track():
    env = MaintenanceEnv()
    env.reset(2)
    env.position = 0
    env._windows[0] = 75
    env._closures[0] = False
    assert env.mask()[1]
    assert not env.mask()[-1]
    with pytest.raises(ValueError):
        env.step(10)


def test_cp_sat_matches_enumerated_minimum():
    for seed in range(5):
        env = MaintenanceEnv()
        env.reset(seed)
        action = cp_sat(env)
        scores = {int(a): round(10000 * env.penalty(
            env.expected_after(a), env.action_cost(a)[0]))
            for a in np.flatnonzero(env.mask())}
        assert scores[action] == min(scores.values())


def test_rollout_constraints_and_metric_accounting():
    result = run_episode(Config(horizon=8), 32, "cp_sat")
    trace = result["trace"]
    assert len(trace) == 8
    assert result["metrics"]["cost"] == sum(x["cost"] for x in trace)
    assert result["metrics"]["return"] == sum(x["reward"] for x in trace)
    for x in trace:
        assert x["mask"][x["action"]]
        assert x["minutes_used"] <= x["available_minutes"]
        assert min(x["after"]) >= 0


def test_terminal_and_invalid_input_handling():
    env = MaintenanceEnv(Config(horizon=2))
    env.reset()
    for bad in (-1, 11, 1.5, True):
        with pytest.raises(ValueError):
            env.step(bad)
    env.step(0)
    _, _, done, mask = env.step(0)
    assert done and mask[0] and mask.sum() == 1
    assert np.isfinite(env.observation()).all()
    with pytest.raises(RuntimeError):
        env.step(0)
