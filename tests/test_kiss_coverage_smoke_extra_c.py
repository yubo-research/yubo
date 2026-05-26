from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rl.torchrl.sac import setup as sac_setup
from rl.torchrl.sac.config import SACConfig


def test_kiss_cov_sac_setup_build_and_update(monkeypatch, tmp_path):
    fake_env_conf = SimpleNamespace(
        from_pixels=False,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
    )
    shared = SimpleNamespace(
        env_conf=fake_env_conf,
        problem_seed=7,
        noise_seed_0=11,
        obs_dim=4,
        act_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        obs_lb=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        obs_width=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "rl.torchrl.sac.sac_setup_build.build_env_setup",
        lambda _config, **kwargs: shared,
    )

    cfg = SACConfig(exp_dir=str(tmp_path), env_tag="pend", batch_size=4, replay_size=32)
    env_setup = sac_setup.build_env_setup(cfg)
    assert env_setup.obs_dim == 4
    modules = sac_setup.build_modules(cfg, env_setup, device=torch.device("cpu"))
    training = sac_setup.build_training(cfg, modules)
    assert training.metrics_path.name == "metrics.jsonl"

    calls = {}

    def _fake_update_step(*, modules, optimizers, batch, hyper):
        calls["target_entropy"] = hyper.target_entropy
        assert batch.obs.shape[0] == 4
        return (1.0, 2.0, 3.0)

    monkeypatch.setattr("rl.core.sac_update.sac_update_step", _fake_update_step)
    out = sac_setup.sac_update_shared(
        cfg,
        modules,
        training,
        obs=torch.zeros((4, 4)),
        act=torch.zeros((4, 2)),
        rew=torch.zeros(4),
        nxt=torch.zeros((4, 4)),
        done=torch.zeros(4),
    )
    assert out == (1.0, 2.0, 3.0)
    assert isinstance(calls["target_entropy"], float)
