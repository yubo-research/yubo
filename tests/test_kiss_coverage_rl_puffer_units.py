from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def test_kiss_cov_puffer_offpolicy_env_model_eval_units(monkeypatch, tmp_path):
    from rl.pufferlib.offpolicy.env_utils import build_env_setup, make_vector_env, resolve_backbone_name, seed_everything
    from rl.pufferlib.offpolicy.eval_utils import (
        TrainState,
        capture_actor_state,
        evaluate_actor,
        evaluate_heldout_if_enabled,
        log_if_due,
        maybe_eval,
        use_actor_state,
    )
    from rl.pufferlib.offpolicy.model_utils import ActorNet, build_optimizers
    from rl.pufferlib.offpolicy.model_utils import capture_actor_state as capture_offpolicy_actor_state
    from rl.pufferlib.offpolicy.model_utils import use_actor_state as use_offpolicy_actor_state

    seed_everything(1)
    obs_spec = SimpleNamespace(mode="pixels", channels=4)
    cfg = SimpleNamespace(backbone_name="mlp")
    assert resolve_backbone_name(cfg, obs_spec) == "nature_cnn_atari"

    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.env_utils.build_continuous_gym_env_setup",
        lambda **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(),
            problem_seed=1,
            noise_seed_0=2,
            obs_lb=None,
            obs_width=None,
            act_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        ),
    )
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace(get_env_conf=lambda *args, **kwargs: SimpleNamespace()))
    env_setup = build_env_setup(SimpleNamespace(env_tag="x", seed=1, problem_seed=None, noise_seed_0=None, from_pixels=False, pixels_only=True))
    assert env_setup.act_dim == 2

    monkeypatch.setattr("rl.pufferlib.offpolicy.env_utils._make_vector_env_shared", lambda config, **kwargs: SimpleNamespace(ok=True))
    assert make_vector_env(SimpleNamespace()) is not None

    obs_scaler = torch.nn.Identity()
    actor = ActorNet(torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), obs_scaler, act_dim=2)
    obs = torch.zeros((2, 3), dtype=torch.float32)
    action, _ = actor.sample(obs, deterministic=True)
    assert action.shape[-1] == 2

    modules = SimpleNamespace(
        actor_backbone=torch.nn.Linear(3, 4),
        actor_head=torch.nn.Linear(4, 2),
        log_std=None,
        actor=SimpleNamespace(act=lambda x: torch.zeros((x.shape[0], 2), dtype=torch.float32)),
        q1=torch.nn.Linear(5, 1),
        q2=torch.nn.Linear(5, 1),
        obs_scaler=torch.nn.Identity(),
    )
    snap = capture_actor_state(modules)
    with use_actor_state(modules, snap, device=torch.device("cpu")):
        pass
    snap2 = capture_offpolicy_actor_state(modules)
    with use_offpolicy_actor_state(modules, snap2):
        pass
    optimizers = build_optimizers(
        SimpleNamespace(learning_rate_actor=1e-3, learning_rate_critic=1e-3),
        SimpleNamespace(actor_backbone=modules.actor_backbone, actor_head=modules.actor_head, q1=modules.q1, q2=modules.q2),
    )
    assert optimizers.actor_optimizer is not None

    eval_cfg = SimpleNamespace(
        num_denoise=1,
        num_denoise_passive=1,
        eval_interval_steps=1,
        eval_seed_base=None,
        eval_noise_mode=None,
        log_interval_steps=1,
        seed=1,
    )
    env = SimpleNamespace(env_conf=SimpleNamespace(), problem_seed=1)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.collect_denoised_trajectory", lambda *args, **kwargs: (SimpleNamespace(rreturn=1.0), None))
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.evaluate_for_best", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.evaluate_heldout_with_best_actor", lambda **kwargs: 1.0)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.build_eval_plan", lambda **kwargs: SimpleNamespace(eval_seed=11, heldout_i_noise=12))
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.update_best_actor_if_improved", lambda **kwargs: (1.0, snap, True))
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.due_mark", lambda step, interval, prev: 1)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.rl_logger.append_metrics", lambda path, record: None)
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.rl_logger.log_eval_iteration", lambda **kwargs: None)

    state = TrainState(start_time=0.0)
    assert evaluate_actor(eval_cfg, env, modules, SimpleNamespace(mode="vector"), device=torch.device("cpu"), eval_seed=1) == 1.0
    assert (
        evaluate_heldout_if_enabled(
            eval_cfg,
            env,
            modules,
            SimpleNamespace(mode="vector"),
            device=torch.device("cpu"),
            heldout_i_noise=3,
            best_actor_state=snap,
            with_actor_state_fn=lambda s: use_actor_state(modules, s, device=torch.device("cpu")),
        )
        == 1.0
    )
    maybe_eval(eval_cfg, env, modules, SimpleNamespace(mode="vector"), state, device=torch.device("cpu"))
    log_if_due(eval_cfg, state, step=1, frames_per_batch=1)


def test_kiss_cov_puffer_ppo_and_sac_units(monkeypatch, tmp_path):
    from rl.pufferlib.ppo.checkpoint import build_checkpoint_payload, save_final_checkpoint
    from rl.pufferlib.ppo.config import PufferPPOConfig, TrainResult
    from rl.pufferlib.ppo.engine import build_eval_env_conf, make_vector_env, register
    from rl.pufferlib.ppo.eval import capture_actor_snapshot, maybe_eval_and_update_state, maybe_render_videos, restore_actor_snapshot, use_actor_snapshot
    from rl.pufferlib.ppo.specs import normalize_action_bounds
    from rl.pufferlib.sac.config import SACConfig
    from rl.pufferlib.sac.config import TrainResult as SACTrainResult
    from rl.pufferlib.sac.engine import register as sac_register

    model = SimpleNamespace(
        actor_backbone=torch.nn.Linear(3, 4),
        actor_head=torch.nn.Linear(4, 2),
        critic_backbone=torch.nn.Linear(3, 4),
        critic_head=torch.nn.Linear(4, 1),
    )
    opt = torch.optim.Adam(model.actor_backbone.parameters(), lr=1e-3)
    state = SimpleNamespace(global_step=1, best_actor_state=None, best_return=0.0, last_eval_return=0.0, last_heldout_return=None, last_episode_return=0.0)
    payload = build_checkpoint_payload(model, opt, state, iteration=1)
    assert payload["iteration"] == 1
    mgr = SimpleNamespace(save_both=lambda payload, iteration: None)
    save_final_checkpoint(SimpleNamespace(checkpoint_interval=1), mgr, model, opt, state, iteration=1)

    cfg = PufferPPOConfig(env_tag="pend")
    assert isinstance(cfg.to_dict(), dict)
    assert TrainResult(best_return=0.0, last_eval_return=0.0, last_heldout_return=None, num_iterations=1).num_iterations == 1
    assert normalize_action_bounds(np.array([-1.0]), np.array([1.0]), 1)[0].shape == (1,)
    assert SACConfig(env_tag="pend").env_tag == "pend"
    assert SACTrainResult(best_return=0.0, last_eval_return=0.0, last_heldout_return=None, num_steps=1).num_steps == 1

    monkeypatch.setattr("rl.pufferlib.ppo.engine._make_vector_env", lambda config: SimpleNamespace())
    monkeypatch.setattr("rl.pufferlib.ppo.engine._build_eval_env_conf", lambda config, obs_spec: SimpleNamespace())
    monkeypatch.setattr("importlib.import_module", lambda name: SimpleNamespace(register_algo=lambda *args, **kwargs: None))
    assert make_vector_env(cfg) is not None
    assert build_eval_env_conf(cfg, obs_spec=SimpleNamespace(mode="vector")) is not None
    register()
    sac_register()

    snap = capture_actor_snapshot(model)
    restore_actor_snapshot(model, snap, device=torch.device("cpu"))
    with use_actor_snapshot(model, snap, device=torch.device("cpu")):
        pass

    monkeypatch.setattr("rl.pufferlib.ppo.eval.is_due", lambda iteration, interval: True)
    monkeypatch.setattr("rl.pufferlib.ppo.eval.eval_noise.build_eval_plan", lambda **kwargs: SimpleNamespace(eval_seed=1, heldout_i_noise=2))
    monkeypatch.setattr("rl.pufferlib.ppo.eval.collect_denoised_trajectory", lambda *args, **kwargs: (SimpleNamespace(rreturn=1.0), None))
    monkeypatch.setattr("rl.pufferlib.ppo.eval.ppo_eval.update_best_actor_if_improved", lambda **kwargs: (1.0, snap, True))
    monkeypatch.setattr("rl.pufferlib.ppo.eval.ppo_eval.evaluate_heldout_with_best_actor", lambda **kwargs: 1.0)
    monkeypatch.setattr("rl.pufferlib.ppo.eval.common_video.render_policy_videos", lambda *args, **kwargs: None)
    ppo_state = SimpleNamespace(
        obs_spec=SimpleNamespace(mode="vector"),
        action_spec=SimpleNamespace(kind="continuous"),
        eval_env_conf=None,
        best_actor_state=None,
        best_return=0.0,
        last_eval_return=0.0,
        last_heldout_return=None,
    )
    ppo_cfg = SimpleNamespace(
        eval_interval=1,
        num_denoise=1,
        num_denoise_passive=1,
        problem_seed=1,
        seed=1,
        eval_seed_base=None,
        eval_noise_mode=None,
        video_enable=True,
        video_seed_base=None,
        video_prefix="p",
        video_num_episodes=1,
        video_num_video_episodes=1,
        video_episode_selection="best",
    )
    maybe_eval_and_update_state(
        ppo_cfg,
        model,
        ppo_state,
        iteration=1,
        device=torch.device("cpu"),
        build_eval_env_conf_fn=lambda config, obs_spec: SimpleNamespace(gym_conf=SimpleNamespace()),
        prepare_obs_fn=lambda obs, obs_spec, device: torch.as_tensor(obs, dtype=torch.float32, device=device),
    )
    maybe_render_videos(
        ppo_cfg,
        model,
        ppo_state,
        exp_dir=Path(tmp_path),
        device=torch.device("cpu"),
        build_eval_env_conf_fn=lambda config, obs_spec: SimpleNamespace(gym_conf=SimpleNamespace()),
        prepare_obs_fn=lambda obs, obs_spec, device: torch.as_tensor(obs, dtype=torch.float32, device=device),
    )


def test_kiss_cov_torchrl_offpolicy_and_sac_loop_units(monkeypatch):
    from rl.torchrl.offpolicy.models import ActorNet, QNet
    from rl.torchrl.sac.config import TrainResult as TorchSACTrainResult
    from rl.torchrl.sac.loop import evaluate_heldout_if_enabled, log_if_due
    from rl.torchrl.sac.trainer import register

    actor = ActorNet(torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), torch.nn.Identity(), 2)
    q = QNet(torch.nn.Linear(5, 4), torch.nn.Linear(4, 1), torch.nn.Identity())
    obs = torch.zeros((2, 3), dtype=torch.float32)
    act = torch.zeros((2, 2), dtype=torch.float32)
    _ = actor.forward(obs)
    _ = q.forward(obs, act)
    assert TorchSACTrainResult(best_return=0.0, last_eval_return=0.0, last_heldout_return=None, num_steps=1).num_steps == 1

    monkeypatch.setattr("rl.core.sac_eval.evaluate_heldout_with_best_actor", lambda **kwargs: 1.0)
    monkeypatch.setattr("rl.logger.log_eval_iteration", lambda **kwargs: None)
    monkeypatch.setattr("rl.core.sac_metrics.build_log_eval_iteration_kwargs", lambda **kwargs: kwargs)
    monkeypatch.setattr("rl.torchrl.sac.loop.is_due", lambda step, interval: True)
    env_setup = SimpleNamespace(problem_seed=1, noise_seed_0=2, env_conf=SimpleNamespace(from_pixels=False, pixels_only=True))
    train_state = SimpleNamespace(best_actor_state={}, last_eval_return=0.0, best_return=0.0, last_heldout_return=None)
    assert (
        evaluate_heldout_if_enabled(
            SimpleNamespace(env_tag="x", num_denoise_passive=1),
            env_setup,
            SimpleNamespace(),
            train_state,
            device=torch.device("cpu"),
            capture_actor_state=lambda modules: {},
            restore_actor_state=lambda modules, state: None,
            eval_policy_factory=lambda modules, env_setup, device: object(),
            get_env_conf=lambda *args, **kwargs: object(),
            evaluate_for_best=lambda *args, **kwargs: 1.0,
        )
        == 1.0
    )
    log_if_due(
        SimpleNamespace(log_interval_steps=1),
        train_state,
        step=1,
        start_time=0.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=1,
    )

    monkeypatch.setattr("rl.torchrl.sac.trainer.sac_deps.registry.register_algo", lambda *args, **kwargs: None)
    register()


def test_kiss_cov_remaining_setup_and_train_units(monkeypatch, tmp_path):
    from rl.pufferlib.offpolicy.model_utils import OffPolicyModules, OffPolicyOptimizers, build_modules
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
        actor_optimizer=torch.optim.AdamW(list(off_modules.actor_backbone.parameters()) + list(off_modules.actor_head.parameters()), lr=1e-3),
        critic_optimizer=torch.optim.AdamW(list(off_modules.q1.parameters()) + list(off_modules.q2.parameters()), lr=1e-3),
    )
    assert off_opt.critic_optimizer is not None

    # puffer SAC engine high-level path
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._init_run_artifacts",
        lambda config: (Path(tmp_path), Path(tmp_path) / "metrics.jsonl", SimpleNamespace(save_both=lambda payload, iteration: None)),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._init_runtime",
        lambda config: (
            SimpleNamespace(problem_seed=1, noise_seed_0=2, act_dim=2, action_low=np.array([-1.0, -1.0]), action_high=np.array([1.0, 1.0])),
            torch.device("cpu"),
        ),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine.make_vector_env",
        lambda config: SimpleNamespace(reset=lambda seed=None: (np.zeros((1, 3), dtype=np.float32), {}), close=lambda: None),
    )
    monkeypatch.setattr("rl.pufferlib.sac.engine.infer_observation_spec", lambda config, obs_np: SimpleNamespace(mode="vector", vector_dim=3, raw_shape=(3,)))
    monkeypatch.setattr("rl.pufferlib.sac.engine.prepare_obs_np", lambda obs_np, obs_spec: np.asarray(obs_np, dtype=np.float32))
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._build_training_components",
        lambda config, env_setup, obs_spec, obs_batch, device: (
            SimpleNamespace(
                actor=SimpleNamespace(sample=lambda obs_t, deterministic=False: (torch.zeros((obs_t.shape[0], 2)), None)),
                actor_backbone=torch.nn.Linear(3, 4),
                actor_head=torch.nn.Linear(4, 2),
                q1=torch.nn.Linear(5, 1),
                q2=torch.nn.Linear(5, 1),
                q1_target=torch.nn.Linear(5, 1),
                q2_target=torch.nn.Linear(5, 1),
                obs_scaler=torch.nn.Identity(),
                log_alpha=torch.nn.Parameter(torch.tensor(0.0)),
            ),
            SimpleNamespace(
                actor_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))], lr=1e-3),
                critic_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))], lr=1e-3),
                alpha_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))], lr=1e-3),
            ),
            SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None, size=0),
            SimpleNamespace(
                global_step=0,
                total_updates=0,
                best_return=float("nan"),
                best_actor_state=None,
                last_eval_return=0.0,
                last_heldout_return=None,
                start_time=0.0,
                eval_mark=0,
                log_mark=0,
                ckpt_mark=0,
            ),
        ),
    )
    monkeypatch.setattr("rl.pufferlib.sac.engine._log_header", lambda *args, **kwargs: None)
    monkeypatch.setattr("rl.pufferlib.sac.engine._train_loop", lambda *args, **kwargs: None)
    monkeypatch.setattr("rl.pufferlib.sac.engine.render_videos_if_enabled", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "importlib.import_module", lambda name: SimpleNamespace(normalize_eval_noise_mode=lambda mode: None, log_run_footer=lambda **kwargs: None)
    )
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

    # torchrl ppo core wrappers
    monkeypatch.setattr(
        "rl.torchrl.ppo.core.core_env_conf.build_seeded_env_conf_from_run",
        lambda **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(ensure_spaces=lambda: None),
            problem_seed=1,
            noise_seed_0=2,
        ),
    )
    monkeypatch.setattr(
        "rl.torchrl.ppo.core.torchrl_env_contract.resolve_env_io_contract",
        lambda env_conf, default_image_size=84: SimpleNamespace(
            observation=SimpleNamespace(mode="vector", vector_dim=3),
            action=SimpleNamespace(kind="continuous", dim=2, low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])),
        ),
    )
    monkeypatch.setattr("rl.torchrl.ppo.core.torchrl_common.obs_scale_from_env", lambda env_conf: (None, None))
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
    ppo_training = ppo_build_training(ppo_cfg, ppo_env, ppo_modules, runtime=SimpleNamespace(collector_backend="multi", single_env_backend="serial"))
    assert ppo_training.frames_per_batch == 1

    # torchrl sac setup wrappers
    monkeypatch.setattr(
        "rl.torchrl.sac.setup.build_continuous_gym_env_setup",
        lambda **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(
                from_pixels=False,
                make=lambda: SimpleNamespace(reset=lambda seed=None: (None, {}), action_space=SimpleNamespace(seed=lambda seed: None)),
                state_space=SimpleNamespace(shape=(3,)),
            ),
            problem_seed=1,
            noise_seed_0=2,
            act_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
            obs_lb=None,
            obs_width=None,
        ),
    )
    monkeypatch.setattr("rl.torchrl.sac.setup.sac_deps.torchrl_common.obs_scale_from_env", lambda env_conf: (None, None))
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
