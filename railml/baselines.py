"""Baselines share the SAC observation boundary and cannot access future noise."""
import numpy as np
from ortools.sat.python import cp_model


def threshold_rule(env):
    feasible = np.flatnonzero(env.mask()[1:])
    if not len(feasible):
        return 0
    target = feasible[np.argmax(env.condition[feasible])]
    return int(target + 1) if env.condition[target] >= 4.5 else 0


def cp_sat(env):
    """One-window expected-cost optimizer, not a global stochastic oracle.

    Enumerates feasible wait/service candidates with known mean deterioration;
    CP-SAT chooses exactly one. Simple by design, an honest myopic baseline.
    """
    candidates = np.flatnonzero(env.mask()).tolist()
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"action_{a}") for a in candidates]
    model.add_exactly_one(x)
    costs = [round(10000 * env.penalty(env.expected_after(a), env.action_cost(a)[0]))
             for a in candidates]
    model.minimize(sum(cost * choice for cost, choice in zip(costs, x)))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 1.
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT failed to find a feasible schedule")
    return next(a for a, choice in zip(candidates, x) if solver.value(choice))


def run_episode(config, seed, policy, model=None):
    from .environment import MaintenanceEnv
    env = MaintenanceEnv(config)
    obs, mask = env.reset(seed)
    while not env.done:
        probabilities = None
        if policy == "sac":
            if model is None:
                raise ValueError("SAC requires a trained checkpoint")
            probabilities = model.probabilities(obs, mask).tolist()
            action = int(np.argmax(probabilities))
        elif policy == "cp_sat":
            action = cp_sat(env)
        elif policy == "threshold":
            action = threshold_rule(env)
        elif policy == "no_maintenance":
            action = 0
        else:
            raise ValueError(f"Unknown policy: {policy}")
        obs, _, _, mask = env.step(action)
        if probabilities is not None:
            env.history[-1]["probabilities"] = probabilities
    return {"policy": policy, **env.result()}
