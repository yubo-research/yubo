from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from rl import logger as rl_logger
from rl.backbone import BackboneSpec, HeadSpec
from rl.core.env_contract import ObservationContract
from rl.core.pixel_transform import ensure_pixel_obs_format
from rl.core.profiler import run_with_profiler
from rl.policy_backbone import (
    AtariMLP16DiscretePolicy,
    DiscreteActorPolicySpec,
    GaussianActorBackbonePolicyFactory,
)
from rl.shared_gaussian_actor import (
    build_shared_gaussian_actor,
    get_gaussian_actor_spec,
)
from rl.torchrl.ppo.models import (
    ActorNet,
    CriticNet,
    DiscreteActorNet,
    prepare_obs_for_backbone,
)


def _fake_atari_env_conf():
    gym_conf = SimpleNamespace(state_space=SimpleNamespace(shape=(4, 84, 84)))
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=gym_conf,
        action_space=SimpleNamespace(n=6),
        from_pixels=True,
        ensure_spaces=lambda: None,
    )


def _fake_continuous_env_conf():
    gym_conf = SimpleNamespace(state_space=SimpleNamespace(shape=(4,)))
    return SimpleNamespace(
        problem_seed=0,
        gym_conf=gym_conf,
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )


def test_logger_facade_functions(tmp_path: Path):
    metrics_path = tmp_path / "metrics.jsonl"
    rl_logger.append_metrics(metrics_path, {"x": 1, "y": 2})
    rows = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    parsed = json.loads(rows[0])
    assert parsed["x"] == 1
    assert parsed["y"] == 2

    config = SimpleNamespace(env_tag="pend", seed=1, backbone_name="mlp")
    env = SimpleNamespace(env_conf=SimpleNamespace(from_pixels=False), obs_dim=3, act_dim=1)
    training = SimpleNamespace(frames_per_batch=8, num_iterations=2)
    runtime = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    rl_logger.log_run_header("ppo", config, env, training, runtime)
    rl_logger.log_run_header_basic(
        algo_name="ppo",
        env_tag="pend",
        seed=1,
        backbone_name="mlp",
        from_pixels=False,
        obs_dim=3,
        act_dim=1,
        frames_per_batch=8,
        num_iterations=2,
        device_type="cpu",
    )
    rl_logger.log_eval_iteration(
        1,
        2,
        8,
        eval_return=1.0,
        heldout_return=0.5,
        best_return=1.0,
        algo_metrics={"kl": 0.01, "clipfrac": 0.1},
        algo_name="ppo",
        elapsed=0.1,
    )
    rl_logger.log_progress_iteration(1, 2, 8, elapsed=0.1, algo_name="ppo")
    rl_logger.log_run_footer(1.0, 2, 0.2, algo_name="ppo")


def test_logger_rl_iter_record(tmp_path: Path):
    metrics_path = tmp_path / "rl_iter.jsonl"
    record = {
        "iter": 1,
        "step": 32,
        "elapsed": 0.25,
        "fps": 128.0,
        "ret_rollout": 2.5,
        "ret_eval": None,
        "ret_best": 2.5,
        "rew": 0.1,
        "done_frac": 0.0,
        "kl": 0.001,
    }

    line = rl_logger.format_rl_iter_record(record)
    assert line == "ITER: iter = 1 step = 32 elapsed = 0.25s fps = 128 ret_rollout = 2.5 ret_best = 2.5 rew = 0.1 done_frac = 0 kl = 0.001"

    rl_logger.log_rl_iter(record, metrics_path=metrics_path)
    parsed = json.loads(metrics_path.read_text(encoding="utf-8").strip())
    assert parsed["ret_rollout"] == 2.5
    assert "ret_eval" not in parsed


def test_shared_gaussian_actor_factory_and_variant_validation():
    backbone, head = get_gaussian_actor_spec("rl-gauss-tanh")
    assert backbone.name == "mlp"
    assert isinstance(head.hidden_sizes, tuple)
    actor = build_shared_gaussian_actor(4, 2, variant="rl-gauss-tanh")
    out = actor(torch.zeros((1, 4), dtype=torch.float32))
    assert out.shape == (1, 2)
    with pytest.raises(ValueError, match="Unknown Gaussian actor variant"):
        get_gaussian_actor_spec("_unknown_variant_")


def test_policy_backbone_factories_and_variants():
    spec = DiscreteActorPolicySpec(
        backbone=BackboneSpec(name="mlp", hidden_sizes=(16, 16), activation="relu", layer_norm=False),
        head=HeadSpec(hidden_sizes=(16, 16), activation="relu"),
        param_scale=0.5,
    )
    assert spec.param_scale == 0.5

    atari_policy = AtariMLP16DiscretePolicy(_fake_atari_env_conf())
    action = atari_policy(np.zeros((4, 84, 84), dtype=np.float32))
    assert isinstance(action, int)

    factory = GaussianActorBackbonePolicyFactory(variant="rl-gauss-tanh")
    policy = factory(_fake_continuous_env_conf())
    action_vec = policy(np.zeros((4,), dtype=np.float32))
    assert action_vec.shape == (2,)


def test_torchrl_pixel_and_models_paths():
    img = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)
    formatted = ensure_pixel_obs_format(img, channels=3, size=84)
    assert formatted.shape == (3, 84, 84)
    assert formatted.dtype == torch.float32

    obs_contract = ObservationContract(mode="pixels", raw_shape=(84, 84, 3), model_channels=3, image_size=84)
    obs = torch.rand((84, 84, 3), dtype=torch.float32)
    prepared, batch_shape, squeeze = prepare_obs_for_backbone(obs, obs_contract)
    assert prepared.shape[-3:] == (3, 84, 84)
    assert batch_shape is None
    assert isinstance(squeeze, bool)

    backbone = nn.Sequential(nn.Flatten(), nn.Linear(3 * 84 * 84, 8))
    actor_head = nn.Linear(8, 2)
    critic_head = nn.Linear(8, 1)
    obs_scaler = nn.Identity()

    actor = ActorNet(
        backbone,
        actor_head,
        nn.Parameter(torch.zeros(2)),
        obs_scaler,
        obs_contract=obs_contract,
    )
    loc, scale = actor(obs)
    assert loc.shape == (2,)
    assert scale.shape == (2,)

    discrete_actor = DiscreteActorNet(backbone, nn.Linear(8, 4), obs_scaler, obs_contract=obs_contract)
    logits = discrete_actor(obs)
    assert logits.shape == (4,)

    critic = CriticNet(backbone, critic_head, obs_scaler, obs_contract=obs_contract)
    value = critic(obs)
    assert value.shape == (1,)


def test_torchrl_profiler_run_with_profiler(monkeypatch, tmp_path: Path):
    class _DummyProfiler:
        def __init__(self):
            self.steps = 0
            self.trace = None

        def step(self):
            self.steps += 1

        def export_chrome_trace(self, path: str):
            self.trace = path

    class _DummyProfileCtx:
        def __init__(self, on_trace_ready):
            self._on_trace_ready = on_trace_ready
            self._prof = _DummyProfiler()

        def __enter__(self):
            return self._prof

        def __exit__(self, exc_type, exc, tb):
            self._on_trace_ready(self._prof)
            return False

    monkeypatch.setattr(
        "rl.core.profiler.torch.profiler.schedule",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "rl.core.profiler.torch.profiler.profile",
        lambda **kwargs: _DummyProfileCtx(kwargs["on_trace_ready"]),
    )

    calls: list[tuple[int, int]] = []

    def _run_iteration(iteration: int, batch):
        calls.append((iteration, int(batch)))

    cfg = SimpleNamespace(exp_dir=str(tmp_path), profile_wait=0, profile_warmup=1, profile_active=1)
    collector = [1, 2, 3]
    run_with_profiler(
        cfg,
        collector,
        _run_iteration,
        device=torch.device("cpu"),
        num_iterations=2,
        start_iteration=0,
    )
    assert calls == [(1, 1), (2, 2)]


def test_mjx_runtime_eval_and_train_loop_helpers_are_addressable():
    import rl.mjx_eval as mjx_eval
    import rl.mjx_runtime as mjx_runtime
    import rl.mjx_train_loop as mjx_train_loop

    runtime = mjx_runtime.MJXRuntime(
        jax="jax",
        jnp="jnp",
        optax="optax",
        adapter="adapter",
        obs_dim=3,
        act_dim=2,
        low=-1.0,
        high=1.0,
    )

    assert runtime.obs_dim == 3
    assert callable(mjx_runtime.require_mjx_stack)
    assert callable(mjx_runtime.make_mjx_runtime)
    assert callable(mjx_eval.make_mjx_eval_step)
    assert callable(mjx_train_loop.run_mjx_training_loop)
    assert mjx_train_loop._last_rollout_return({"ep_ret": float("nan"), "rollout_return": 4.0}) == 4.0
    assert mjx_train_loop._is_finite(1.0)
