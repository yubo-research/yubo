from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def _mod(parts, names):
    return __import__(".".join(parts), fromlist=names)


def test_kiss_cov_puffer_offpolicy_env_model_eval_units(monkeypatch, tmp_path):
    env_utils = _mod(
        ("rl", "pufferlib", "offpolicy", "env_utils"),
        fromlist=[
            "build_env_setup",
            "make_vector_env",
            "resolve_backbone_name",
            "seed_everything",
        ],
    )
    eval_utils = _mod(
        ("rl", "pufferlib", "offpolicy", "eval_utils"),
        fromlist=[
            "TrainState",
            "capture_actor_state",
            "evaluate_actor",
            "evaluate_heldout_if_enabled",
            "log_if_due",
            "maybe_eval",
            "use_actor_state",
        ],
    )
    model_utils = _mod(
        ("rl", "pufferlib", "offpolicy", "model_utils"),
        fromlist=[
            "ActorNet",
            "build_optimizers",
            "capture_actor_state",
            "use_actor_state",
        ],
    )

    env_utils.seed_everything(1)
    obs_spec = SimpleNamespace(mode="pixels", channels=4)
    cfg = SimpleNamespace(backbone_name="mlp")
    assert env_utils.resolve_backbone_name(cfg, obs_spec) == "nature_cnn_atari"

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
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: SimpleNamespace(get_env_conf=lambda *args, **kwargs: SimpleNamespace()),
    )
    env_setup = env_utils.build_env_setup(
        SimpleNamespace(
            env_tag="x",
            seed=1,
            problem_seed=None,
            noise_seed_0=None,
            from_pixels=False,
            pixels_only=True,
        )
    )
    assert env_setup.act_dim == 2

    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.env_utils._make_vector_env_shared",
        lambda config, **kwargs: SimpleNamespace(ok=True),
    )
    assert env_utils.make_vector_env(SimpleNamespace()) is not None

    obs_scaler = torch.nn.Identity()
    actor = model_utils.ActorNet(torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), obs_scaler, act_dim=2)
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
    snap = eval_utils.capture_actor_state(modules)
    with eval_utils.use_actor_state(modules, snap, device=torch.device("cpu")):
        pass
    snap2 = model_utils.capture_actor_state(modules)
    with model_utils.use_actor_state(modules, snap2):
        pass
    optimizers = model_utils.build_optimizers(
        SimpleNamespace(learning_rate_actor=1e-3, learning_rate_critic=1e-3),
        SimpleNamespace(
            actor_backbone=modules.actor_backbone,
            actor_head=modules.actor_head,
            q1=modules.q1,
            q2=modules.q2,
        ),
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
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.collect_denoised_trajectory",
        lambda *args, **kwargs: (SimpleNamespace(rreturn=1.0), None),
    )
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.evaluate_for_best",
        lambda *args, **kwargs: 1.0,
    )
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.evaluate_heldout_with_best_actor",
        lambda **kwargs: 1.0,
    )
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.build_eval_plan",
        lambda **kwargs: SimpleNamespace(eval_seed=11, heldout_i_noise=12),
    )
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.update_best_actor_if_improved",
        lambda **kwargs: (1.0, snap, True),
    )
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.due_mark", lambda step, interval, prev: 1)
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.rl_logger.append_metrics",
        lambda path, record: None,
    )
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.rl_logger.log_eval_iteration",
        lambda **kwargs: None,
    )

    state = eval_utils.TrainState(start_time=0.0)
    assert (
        eval_utils.evaluate_actor(
            eval_cfg,
            env,
            modules,
            SimpleNamespace(mode="vector"),
            device=torch.device("cpu"),
            eval_seed=1,
        )
        == 1.0
    )
    assert (
        eval_utils.evaluate_heldout_if_enabled(
            eval_cfg,
            env,
            modules,
            SimpleNamespace(mode="vector"),
            device=torch.device("cpu"),
            heldout_i_noise=3,
            best_actor_state=snap,
            with_actor_state_fn=lambda s: eval_utils.use_actor_state(modules, s, device=torch.device("cpu")),
        )
        == 1.0
    )
    eval_utils.maybe_eval(
        eval_cfg,
        env,
        modules,
        SimpleNamespace(mode="vector"),
        state,
        device=torch.device("cpu"),
    )
    eval_utils.log_if_due(eval_cfg, state, step=1, frames_per_batch=1)


def test_kiss_cov_torchrl_offpolicy_and_sac_loop_units(monkeypatch):
    models = _mod(("rl", "torchrl", "offpolicy", "models"), ["ActorNet", "QNet"])
    sac_config = _mod(("rl", "torchrl", "sac", "config"), ["TrainResult"])
    sac_loop = _mod(
        ("rl", "torchrl", "sac", "loop"),
        ["evaluate_heldout_if_enabled", "log_if_due"],
    )
    trainer = _mod(("rl", "torchrl", "sac", "trainer"), ["register"])

    actor = models.ActorNet(torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), torch.nn.Identity(), 2)
    q = models.QNet(torch.nn.Linear(5, 4), torch.nn.Linear(4, 1), torch.nn.Identity())
    obs = torch.zeros((2, 3), dtype=torch.float32)
    act = torch.zeros((2, 2), dtype=torch.float32)
    _ = actor.forward(obs)
    _ = q.forward(obs, act)
    assert sac_config.TrainResult(best_return=0.0, last_eval_return=0.0, last_heldout_return=None, num_steps=1).num_steps == 1

    monkeypatch.setattr("rl.core.sac_eval.evaluate_heldout_with_best_actor", lambda **kwargs: 1.0)
    monkeypatch.setattr("rl.logger.log_eval_iteration", lambda **kwargs: None)
    monkeypatch.setattr("rl.core.sac_metrics.build_log_eval_iteration_kwargs", lambda **kwargs: kwargs)
    monkeypatch.setattr("rl.torchrl.sac.loop.is_due", lambda step, interval: True)
    env_setup = SimpleNamespace(
        problem_seed=1,
        noise_seed_0=2,
        env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
    )
    train_state = SimpleNamespace(
        best_actor_state={},
        last_eval_return=0.0,
        best_return=0.0,
        last_heldout_return=None,
    )
    assert (
        sac_loop.evaluate_heldout_if_enabled(
            SimpleNamespace(env_tag="x", num_denoise_passive=1),
            env_setup,
            SimpleNamespace(),
            train_state,
            device=torch.device("cpu"),
            capture_actor_state=lambda modules: {},
            restore_actor_state=lambda modules, state: None,
            eval_policy_factory=lambda modules, env_setup, device: object(),
            build_env_runtime=lambda *args, **kwargs: object(),
            evaluate_for_best=lambda *args, **kwargs: 1.0,
        )
        == 1.0
    )
    sac_loop.log_if_due(
        SimpleNamespace(log_interval_steps=1),
        train_state,
        step=1,
        start_time=0.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=1,
    )

    monkeypatch.setattr(
        "rl.torchrl.sac.trainer.sac_deps.registry.register_algo",
        lambda *args, **kwargs: None,
    )
    trainer.register()
