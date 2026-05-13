from __future__ import annotations

from types import SimpleNamespace

import torch
from kiss_rl_puffer_remaining_helpers import (
    patch_puffer_sac_engine_for_kiss,
    patch_torchrl_ppo_core_for_kiss,
    patch_torchrl_sac_setup_for_kiss,
)


def test_kiss_cov_remaining_setup_and_train_units(monkeypatch, tmp_path):
    from rl.pufferlib.offpolicy.model_utils import (
        OffPolicyModules,
        OffPolicyOptimizers,
        build_modules,
    )
    from rl.pufferlib.sac.engine import train_sac_puffer_impl
    from rl.torchrl.ppo.core import build_env_setup as ppo_build_env_setup
    from rl.torchrl.ppo.core import build_modules as ppo_build_modules
    from rl.torchrl.ppo.core import build_training as ppo_build_training
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

    patch_puffer_sac_engine_for_kiss(monkeypatch, tmp_path)
    res = train_sac_puffer_impl(
        SimpleNamespace(
            eval_noise_mode=None,
            total_timesteps=0,
            checkpoint_interval_steps=None,
            problem_seed=None,
            noise_seed_0=None,
            replay_backend="auto",
            env_tag="x",
            seed=1,
        )
    )
    assert res.num_steps == 0

    patch_torchrl_ppo_core_for_kiss(monkeypatch)
    ppo_cfg = SimpleNamespace(
        env_tag="x",
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
    ppo_modules = ppo_build_modules(ppo_cfg, ppo_env, device=torch.device("cpu"))
    ppo_training = ppo_build_training(
        ppo_cfg,
        ppo_env,
        ppo_modules,
        runtime=SimpleNamespace(collector_backend="multi", single_env_backend="serial"),
    )
    assert ppo_training.frames_per_batch == 1

    patch_torchrl_sac_setup_for_kiss(monkeypatch)
    sac_cfg = SimpleNamespace(
        env_tag="x",
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
        alpha_init=0.2,
        theta_dim=None,
        learning_rate_actor=1e-3,
        learning_rate_critic=1e-3,
        learning_rate_alpha=1e-3,
        replay_size=32,
        batch_size=4,
        exp_dir=str(tmp_path),
        gamma=0.99,
        tau=0.005,
        target_entropy=None,
        to_dict=lambda: {},
    )
    sac_env = sac_build_env_setup(sac_cfg)
    sac_modules = sac_build_modules(sac_cfg, sac_env, device=torch.device("cpu"))
    sac_training = sac_build_training(sac_cfg, sac_modules)
    assert sac_training.replay is not None
