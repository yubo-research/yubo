from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from kiss_rl_puffer_remaining_helpers import (
    patch_puffer_sac_engine_for_kiss,
    patch_torchrl_ppo_core_for_kiss,
    patch_torchrl_sac_setup_for_kiss,
)


def test_kiss_cov_remaining_setup_and_train_units(monkeypatch, tmp_path):
    # Apply mocks BEFORE any imports that might trigger real logic
    patch_torchrl_ppo_core_for_kiss(monkeypatch)

    # Fully mock ppo_build_env_setup to avoid any real logic
    mock_env = SimpleNamespace(
        problem_seed=1,
        obs_dim=3,
        act_dim=2,
        io_contract=SimpleNamespace(observation=SimpleNamespace(mode="vector", vector_dim=3)),
        obs_lb=np.array([-1.0, -1.0, -1.0]),
        obs_width=np.array([2.0, 2.0, 2.0]),
        is_discrete=False,
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        env_conf=SimpleNamespace(ensure_spaces=lambda: None, gym_conf=None),
    )
    monkeypatch.setattr("rl.torchrl.ppo.core_env_setup.build_env_setup", lambda *args, **kwargs: mock_env)
    monkeypatch.setattr("rl.torchrl.ppo.core.build_env_setup", lambda *args, **kwargs: mock_env)

    from rl.pufferlib.offpolicy.model_utils import (
        OffPolicyModules,
        OffPolicyOptimizers,
        build_modules,
    )
    from rl.pufferlib.sac.engine import train_sac_puffer_impl
    from rl.torchrl.ppo.core import build_env_setup as ppo_build_env_setup
    from rl.torchrl.ppo.core import build_modules as ppo_build_modules
    from rl.torchrl.ppo.core import build_training as ppo_build_training
    from rl.torchrl.sac.config import SACConfig
    from rl.torchrl.sac.setup import build_env_setup as sac_build_env_setup
    from rl.torchrl.sac.setup import build_modules as sac_build_modules
    from rl.torchrl.sac.setup import build_training as sac_build_training

    off_cfg = SimpleNamespace(
        backbone_name="mlp",
        backbone_hidden_sizes=(8,),
        backbone_activation="relu",
        backbone_layer_norm=False,
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        head_activation="relu",
        theta_dim=None,
        learning_rate_actor=1e-3,
        learning_rate_critic=1e-3,
    )
    off_env = SimpleNamespace(act_dim=2, obs_lb=None, obs_width=None)
    off_obs_spec = SimpleNamespace(mode="vector", vector_dim=3)
    off_modules = build_modules(off_cfg, off_env, off_obs_spec, device=torch.device("cpu"))
    assert isinstance(off_modules, OffPolicyModules)
    off_opt = OffPolicyOptimizers(
        actor_optimizer=torch.optim.AdamW(
            list(off_modules.actor_backbone.parameters()) + list(off_modules.actor_head.parameters()),
            lr=1e-3,
        ),
        critic_optimizer=torch.optim.AdamW(
            list(off_modules.q1.parameters()) + list(off_modules.q2.parameters()),
            lr=1e-3,
        ),
    )
    assert off_opt.critic_optimizer is not None

    ppo_cfg = SimpleNamespace(
        env_tag="pend",
        seed=1,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        backbone_name="mlp",
        backbone_hidden_sizes=(8,),
        backbone_activation="relu",
        backbone_layer_norm=False,
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        head_activation="relu",
        share_backbone=False,
        log_std_init=-0.5,
        theta_dim=None,
        num_envs=1,
        num_steps=1,
        total_timesteps=2,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=1.0,
        norm_adv=False,
        clip_vloss=False,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        exp_dir=str(tmp_path),
        to_dict=lambda: {},
    )

    ppo_env = ppo_build_env_setup(ppo_cfg)
    assert ppo_env.problem_seed == 1

    patch_puffer_sac_engine_for_kiss(monkeypatch, tmp_path)
    res = train_sac_puffer_impl(
        SimpleNamespace(
            eval_noise_mode=None,
            total_timesteps=0,
            checkpoint_interval_steps=None,
            problem_seed=None,
            noise_seed_0=None,
            replay_backend="auto",
            env_tag="pend",
            seed=1,
        )
    )
    assert res.num_steps == 0

    ppo_modules = ppo_build_modules(ppo_cfg, ppo_env, device=torch.device("cpu"))

    from rl.core.torchrl_runtime import TorchRLRuntimeConfig

    runtime = TorchRLRuntimeConfig().resolve_runtime()

    ppo_training = ppo_build_training(
        ppo_cfg,
        ppo_env,
        ppo_modules,
        runtime=runtime,
    )
    assert ppo_training is not None

    patch_torchrl_sac_setup_for_kiss(monkeypatch)
    # Use real SACConfig to ensure all fields are present
    sac_cfg = SACConfig(
        env_tag="pend",
        seed=1,
        exp_dir=str(tmp_path),
        num_envs=1,
        batch_size=2,
        replay_size=4,
    )
    sac_cfg.metrics_path = tmp_path / "metrics.jsonl"

    sac_env = sac_build_env_setup(sac_cfg)
    sac_modules = sac_build_modules(sac_cfg, sac_env, device=torch.device("cpu"))
    sac_training = sac_build_training(sac_cfg, sac_modules)
    assert sac_training is not None
