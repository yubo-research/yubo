from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def test_kiss_tidy_d_sac_puffer_torchrl_ppo_core(monkeypatch, tmp_path):
    from rl.pufferlib.sac import sac_puffer_engine_impl, sac_puffer_train_run
    from rl.pufferlib.sac.config import SACConfig
    from rl.pufferlib.sac.sac_puffer_train_run_orchestrate import (
        train_sac_puffer_impl as orch_impl,
    )

    sac_cfg = SACConfig(
        exp_dir=str(tmp_path),
        env_tag="pend",
        device="cpu",
        total_timesteps=0,
        replay_backend="auto",
    )

    import gymnasium as gym

    monkeypatch.setattr(
        "rl.pufferlib.sac.env_utils.build_env_setup",
        lambda _cfg: SimpleNamespace(
            env_conf=SimpleNamespace(
                make_gym_env=lambda seed=0, **kwargs: SimpleNamespace(
                    reset=lambda seed=None: (np.zeros(3, dtype=np.float32), {}),
                    step=lambda a: (np.zeros(3, dtype=np.float32), 0.0, False, False, {}),
                    close=lambda: None,
                    observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,)),
                    action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                    is_discrete=False,
                )
            ),
            problem_seed=1,
            noise_seed_0=2,
            obs_dim=3,
            act_dim=1,
            action_low=np.array([-1.0]),
            action_high=np.array([1.0]),
            obs_lb=None,
            obs_width=None,
        ),
    )

    assert sac_puffer_train_run.train_sac_puffer_impl(sac_cfg).num_steps == 0
    assert sac_puffer_train_run.train_sac_puffer(sac_cfg).num_steps == 0
    assert orch_impl(sac_cfg).num_steps == 0
    monkeypatch.setattr("rl.registry.register_algo", lambda *a, **k: None)
    sac_puffer_engine_impl.register()
    assert sac_puffer_engine_impl.config_proxy("SACConfig") is SACConfig
