"""Discrete SAC: exact action expectations, twin Q critics and target networks.

Reference for algorithm: Christodoulou, arXiv:1910.07207.
This is an independent implementation, not the maintenance authors' source.
"""
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TrackNetwork(nn.Module):
    def __init__(self, segments):
        super().__init__()
        self.n = segments
        self.features = nn.Sequential(
            nn.Conv1d(5, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(16 * segments + 2, 64), nn.ReLU(),
                                  nn.Linear(64, segments + 1))

    def forward(self, x):
        z = self.features(x[:, :-2].reshape(-1, 5, self.n)).flatten(1)
        return self.head(torch.cat([z, x[:, -2:]], dim=1))


def distribution(logits, mask):
    logp = F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)
    return logp.exp(), logp


class Replay:
    def __init__(self, obs_size, actions, capacity=50000, seed=0):
        self.rng = np.random.default_rng(seed)
        self.size = self.index = 0
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_size), np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.mask = np.zeros((capacity, actions), bool)
        self.next_mask = np.zeros_like(self.mask)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)

    def add(self, obs, mask, action, reward, next_obs, next_mask, done):
        i = self.index
        for name, val in locals().copy().items():
            if name in ("obs", "mask", "action", "reward", "next_obs", "next_mask", "done"):
                getattr(self, name)[i] = val
        self.index = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch=128):
        idx = self.rng.integers(self.size, size=batch)
        return {k: torch.from_numpy(getattr(self, k)[idx])
                for k in ("obs", "mask", "action", "reward", "next_obs", "next_mask", "done")}


class SAC:
    def __init__(self, segments=10, seed=0, alpha=.10, gamma=.97, tau=.01, lr=3e-4):
        torch.manual_seed(seed)
        self.segments, self.seed = segments, seed
        self.alpha, self.gamma, self.tau = alpha, gamma, tau
        self.actor = TrackNetwork(segments)
        self.q1, self.q2 = TrackNetwork(segments), TrackNetwork(segments)
        self.target1, self.target2 = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for net in (self.target1, self.target2):
            net.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

    @torch.no_grad()
    def probabilities(self, obs, mask):
        p, _ = distribution(self.actor(torch.as_tensor(obs).unsqueeze(0)),
                            torch.as_tensor(mask).unsqueeze(0))
        return p[0].numpy()

    def act(self, obs, mask, deterministic=True, rng=None):
        p = self.probabilities(obs, mask)
        if deterministic:
            return int(p.argmax())
        return int(rng.choice(len(p), p=p / p.sum()))

    def update(self, batch):
        b = batch
        with torch.no_grad():
            p, logp = distribution(self.actor(b["next_obs"]), b["next_mask"])
            q = torch.minimum(self.target1(b["next_obs"]), self.target2(b["next_obs"]))
            value = (p * (q - self.alpha * logp)).sum(-1)
            target = b["reward"] + self.gamma * (1 - b["done"]) * value
        selected = b["action"].unsqueeze(1)
        q1 = self.q1(b["obs"]).gather(1, selected).squeeze(1)
        q2 = self.q2(b["obs"]).gather(1, selected).squeeze(1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.)
        self.critic_opt.step()
        p, logp = distribution(self.actor(b["obs"]), b["mask"])
        with torch.no_grad():
            q = torch.minimum(self.q1(b["obs"]), self.q2(b["obs"]))
        actor_loss = (p * (self.alpha * logp - q)).sum(-1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
        self.actor_opt.step()
        with torch.no_grad():
            for current, target_net in ((self.q1, self.target1), (self.q2, self.target2)):
                for a, b_ in zip(current.parameters(), target_net.parameters()):
                    b_.lerp_(a, self.tau)
        return {"critic_loss": float(critic_loss.detach()), "actor_loss": float(actor_loss.detach()),
                "entropy": float(-(p * logp).sum(-1).mean().detach())}

    def save(self, path, metadata=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "segments": self.segments,
                    "seed": self.seed, "metadata": metadata or {}}, path)

    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(state["segments"], state["seed"])
        model.actor.load_state_dict(state["actor"])
        model.actor.eval()
        model.metadata = state["metadata"]
        return model
