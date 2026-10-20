"""Run: python -m railml.train --steps 20000 --seeds 0 1 2"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .baselines import run_episode
from .environment import Config, MaintenanceEnv
from .sac import Replay, SAC

VALIDATION_SEEDS = list(range(100000, 100005))


def train(seed, steps, out, config=None):
    config = config or Config()
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = SAC(config.segments, seed)
    env = MaintenanceEnv(config)
    replay = Replay(env.obs_size, env.actions, seed=seed)
    directory = Path(out) / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    episode = 0
    # Each run uses a disjoint training scenario range, below validation/test.
    obs, mask = env.reset(seed * 20000)
    best = -float("inf")
    rows, validations = [], []
    loss = {}
    for step in range(1, steps + 1):
        action = (int(rng.choice(np.flatnonzero(mask))) if step <= 1000
                  else model.act(obs, mask, False, rng))
        next_obs, reward, done, next_mask = env.step(action)
        replay.add(obs, mask, action, reward, next_obs, next_mask, done)
        obs, mask = next_obs, next_mask
        if step >= 1000:
            loss = model.update(replay.sample())
        if done:
            rows.append({"step": step, "episode": episode, **env.result()["metrics"], **loss})
            episode += 1
            obs, mask = env.reset(seed * 20000 + episode)
        if step % 2000 == 0 or step == steps:
            scores = [run_episode(config, s, "sac", model)["metrics"]["return"]
                      for s in VALIDATION_SEEDS]
            score = float(np.mean(scores))
            validations.append({"step": step, "mean_return": score})
            metadata = {"training_seed": seed, "steps": step, "config": asdict(config),
                        "validation_seeds": VALIDATION_SEEDS, "validation_return": score,
                        "algorithm": "discrete SAC", "data": "synthetic",
                        "training_scenario_start": seed * 20000,
                        "training_scenario_end": seed * 20000 + episode,
                        "architecture": "3 Conv1d layers (16 channels), dense 64",
                        "alpha": .10, "gamma": .97, "tau": .01, "lr": .0003}
            if score > best:
                best = score
                model.save(directory / "best.pt", metadata)
            print(f"seed={seed} step={step}/{steps} validation={score:.2f} "
                  f"best={best:.2f} elapsed={time.monotonic()-start:.0f}s", flush=True)
    fields = sorted({k for row in rows for k in row})
    with (directory / "training.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (directory / "validation.json").write_text(json.dumps(validations, indent=2))
    checkpoint = directory / "best.pt"
    manifest = {"seed": seed, "requested_steps": steps, "episodes": episode,
                "seconds": time.monotonic() - start, "best_validation_return": best,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "config": asdict(config), "torch_version": torch.__version__,
                "numpy_version": np.__version__, "training_seed_range": [seed * 20000, seed * 20000 + episode]}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out", default="artifacts")
    args = p.parse_args()
    if not 1000 <= args.steps <= 1000000 or any(s not in (0, 1, 2) for s in args.seeds):
        p.error("steps must be 1000..1000000; seeds must be 0, 1 or 2")
    for seed in args.seeds:
        train(seed, args.steps, args.out)


if __name__ == "__main__":
    main()
