import numpy as np
import torch

from railml.environment import MaintenanceEnv
from railml.sac import SAC, Replay, distribution

torch.set_num_threads(1)


def test_mask_has_exactly_zero_probability_and_finite_entropy():
    logits = torch.tensor([[100., 2., 3.], [1., 9., 2.]])
    mask = torch.tensor([[False, True, True], [True, False, False]])
    p, logp = distribution(logits, mask)
    assert p[0, 0] == 0 and p[1, 0] == 1
    torch.testing.assert_close(p.sum(-1), torch.ones(2))
    assert torch.isfinite(p * logp).all()


def test_update_changes_actor_critics_targets_and_checkpoint_roundtrip(tmp_path):
    env = MaintenanceEnv()
    model = SAC(seed=7)
    replay = Replay(env.obs_size, env.actions)
    obs, mask = env.reset(3)
    for _ in range(32):
        action = int(np.flatnonzero(mask)[-1])
        nxt, reward, done, nxt_mask = env.step(action)
        replay.add(obs, mask, action, reward, nxt, nxt_mask, done)
        obs, mask = nxt, nxt_mask
    actor_before = [p.detach().clone() for p in model.actor.parameters()]
    q_before = [p.detach().clone() for p in model.q1.parameters()]
    target_before = [p.detach().clone() for p in model.target1.parameters()]
    loss = model.update(replay.sample(16))
    assert all(np.isfinite(x) for x in loss.values())
    for before, net in ((actor_before, model.actor), (q_before, model.q1),
                        (target_before, model.target1)):
        assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
    assert all(p.grad is None for p in model.target1.parameters())
    path = tmp_path / "model.pt"
    model.save(path)
    loaded = SAC.load(path)
    np.testing.assert_array_equal(model.probabilities(obs, mask),
                                  loaded.probabilities(obs, mask))


def test_terminal_bellman_target_does_not_bootstrap():
    # Identical networks/batches except terminal next-state: losses must match.
    env = MaintenanceEnv()
    obs, mask = env.reset(12)
    replay = Replay(env.obs_size, env.actions)
    replay.add(obs, mask, 0, -4., obs, mask, True)
    b = replay.sample(4)
    c = {k: v.clone() for k, v in b.items()}
    c["next_obs"] = c["next_obs"] * 100
    a, z = SAC(seed=9), SAC(seed=9)
    assert abs(a.update(b)["critic_loss"] - z.update(c)["critic_loss"]) < 1e-6
