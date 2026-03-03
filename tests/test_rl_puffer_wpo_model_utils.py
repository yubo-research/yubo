from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rl.pufferlib.wpo import model_utils
from rl.pufferlib.wpo.config import WPOConfig


class _ReplayStub:
    def __init__(self, *, obs_dim: int, act_dim: int):
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

    def sample(self, batch_size: int, device: torch.device):
        n = int(batch_size)
        obs = torch.randn((n, self.obs_dim), dtype=torch.float32, device=device)
        act = torch.randn((n, self.act_dim), dtype=torch.float32, device=device).tanh()
        rew = torch.randn((n,), dtype=torch.float32, device=device)
        nxt = torch.randn((n, self.obs_dim), dtype=torch.float32, device=device)
        done = torch.zeros((n,), dtype=torch.float32, device=device)
        return (obs, act, rew, nxt, done)


def test_wpo_model_utils_build_update_and_actor_state(monkeypatch):
    cfg = WPOConfig(
        backbone_hidden_sizes=(8,),
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        batch_size=4,
        per_dim_constraining=True,
    )
    env = SimpleNamespace(
        act_dim=2,
        obs_lb=np.asarray([-1.0, -1.0, -1.0], dtype=np.float32),
        obs_width=np.asarray([2.0, 2.0, 2.0], dtype=np.float32),
    )
    obs_spec = SimpleNamespace(mode="vector", vector_dim=3)

    modules = model_utils.build_modules(cfg, env, obs_spec, device=torch.device("cpu"))
    assert modules.log_alpha_mean.shape == (2,)
    assert modules.log_alpha_stddev.shape == (2,)
    assert modules.actor_target.training is False
    assert all(not p.requires_grad for p in modules.actor_target.parameters())

    cfg_scalar_dual = WPOConfig(
        backbone_hidden_sizes=(8,),
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        per_dim_constraining=False,
    )
    modules_scalar_dual = model_utils.build_modules(cfg_scalar_dual, env, obs_spec, device=torch.device("cpu"))
    assert modules_scalar_dual.log_alpha_mean.shape == (1,)
    assert modules_scalar_dual.log_alpha_stddev.shape == (1,)

    optimizers = model_utils.build_optimizers(cfg, modules)
    assert optimizers.actor_optimizer is not None
    assert optimizers.critic_optimizer is not None
    assert optimizers.dual_optimizer is not None

    replay = _ReplayStub(obs_dim=3, act_dim=2)
    monkeypatch.setattr(model_utils, "wpo_update_step", lambda **_kwargs: (1.0, 2.0, 3.0, 0.4, 0.5))
    losses = model_utils.wpo_update(cfg, modules, optimizers, replay, device=torch.device("cpu"))
    assert losses == (1.0, 2.0, 3.0, 0.4, 0.5)

    snapshot = model_utils.capture_actor_state(modules)
    assert "backbone" in snapshot and "head" in snapshot and "obs_scaler" in snapshot
    first_key = next(iter(modules.actor_backbone.state_dict().keys()))
    with torch.no_grad():
        modules.actor_backbone.state_dict()[first_key].add_(1.0)
    model_utils.restore_actor_state(modules, snapshot)
    with model_utils.use_actor_state(modules, snapshot):
        _ = modules.actor.act(torch.zeros((1, 3), dtype=torch.float32))
