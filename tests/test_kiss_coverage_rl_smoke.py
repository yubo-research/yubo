from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from rl import logger as rl_logger
from rl.backbone import BackboneSpec, HeadSpec
from rl.policy_backbone import (
    AtariMLP16DiscretePolicy,
    DiscreteActorPolicySpec,
    GaussianActorBackbonePolicyFactory,
)
from rl.pufferlib.ppo.eval import (
    PufferEvalPolicy,
    resolve_eval_seeds,
    validate_eval_config,
)
from rl.pufferlib_compat import import_pufferlib_modules
from rl.registry import register_algo_backend, resolve_algo_name
from rl.shared_gaussian_actor import (
    build_shared_gaussian_actor,
    get_gaussian_actor_spec,
)
from rl.torchrl.common.env_contract import ObservationContract
from rl.torchrl.common.pixel_transform import ensure_pixel_obs_format
from rl.torchrl.common.profiler import run_with_profiler
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


def test_registry_resolve_algo_name_backend_paths():
    algo_name = "_kiss_cov_algo"
    backend = "torchrl"
    implementation = "_kiss_cov_algo_impl"
    register_algo_backend(algo_name, backend, implementation)
    assert resolve_algo_name(algo_name, backend=backend) == implementation
    assert resolve_algo_name(algo_name, backend=None) == algo_name
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_algo_name(algo_name, backend="pufferlib")


def test_import_pufferlib_modules_from_sys_modules(monkeypatch):
    puffer_pkg = ModuleType("pufferlib")
    puffer_pkg.__path__ = []  # mark as package
    puffer_vector = ModuleType("pufferlib.vector")
    puffer_env_pkg = ModuleType("pufferlib.environments")
    puffer_env_pkg.__path__ = []
    puffer_atari = ModuleType("pufferlib.environments.atari")

    monkeypatch.setitem(sys.modules, "pufferlib", puffer_pkg)
    monkeypatch.setitem(sys.modules, "pufferlib.vector", puffer_vector)
    monkeypatch.setitem(sys.modules, "pufferlib.environments", puffer_env_pkg)
    monkeypatch.setitem(sys.modules, "pufferlib.environments.atari", puffer_atari)
    monkeypatch.delitem(sys.modules, "gym", raising=False)

    pufferlib, puffer_vector_mod, puffer_atari_mod = import_pufferlib_modules()
    assert pufferlib is puffer_pkg
    assert puffer_vector_mod is puffer_vector
    assert puffer_atari_mod is puffer_atari
    assert "gym" in sys.modules


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
        "rl.torchrl.common.profiler.torch.profiler.schedule",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "rl.torchrl.common.profiler.torch.profiler.profile",
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


def test_puffer_eval_helpers_and_validation():
    config = SimpleNamespace(seed=7, problem_seed=None, noise_seed_0=None)
    problem_seed, noise_seed_0 = resolve_eval_seeds(config)
    assert isinstance(problem_seed, int)
    assert isinstance(noise_seed_0, int)

    model = SimpleNamespace(
        actor_backbone=nn.Identity(),
        actor_head=nn.Linear(4, 2),
    )
    obs_spec = SimpleNamespace(mode="vector")
    action_spec = SimpleNamespace(kind="continuous")
    policy = PufferEvalPolicy(
        model=model,
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=torch.device("cpu"),
        prepare_obs_fn=lambda state, **_kwargs: torch.as_tensor(state, dtype=torch.float32),
    )
    act = policy(np.zeros((4,), dtype=np.float32))
    assert act.shape == (2,)

    valid_cfg = SimpleNamespace(
        eval_interval=1,
        eval_noise_mode=None,
        num_denoise_eval=1,
        num_denoise_passive_eval=1,
        checkpoint_interval=1,
        video_num_episodes=1,
        video_num_video_episodes=0,
        video_episode_selection="best",
    )
    validate_eval_config(valid_cfg)

    invalid_cfg = SimpleNamespace(
        eval_interval=1,
        eval_noise_mode=None,
        num_denoise_eval=1,
        num_denoise_passive_eval=1,
        checkpoint_interval=1,
        video_num_episodes=1,
        video_num_video_episodes=0,
        video_episode_selection="invalid",
    )
    with pytest.raises(ValueError, match="video_episode_selection"):
        validate_eval_config(invalid_cfg)
