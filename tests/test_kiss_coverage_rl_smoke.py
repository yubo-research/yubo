from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from rl.backbone import BackboneSpec, HeadSpec
from rl.core.pixel_transform import ensure_pixel_obs_format
from rl.core.profiler import run_with_profiler
from rl.policy_backbone import (
    AtariMLP16DiscretePolicy,
    DiscreteActorPolicySpec,
    GaussianActorBackbonePolicyFactory,
)
from rl.ppo.eval import (
    PPOEvalPolicy,
    validate_eval_config,
)
from rl.pufferlib_compat import import_pufferlib_modules
from rl.shared_gaussian_actor import (
    build_shared_gaussian_actor,
    get_gaussian_actor_spec,
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
    assert hasattr(atari_policy, "backbone")
    assert hasattr(atari_policy, "head")

    factory = GaussianActorBackbonePolicyFactory(variant="rl-gauss-tanh")
    policy = factory(_fake_continuous_env_conf())
    action_vec = policy(np.zeros((4,), dtype=np.float32))
    assert action_vec.shape == (2,)


def test_pixel_obs_format():
    img = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)
    formatted = ensure_pixel_obs_format(img, channels=3, size=84)
    assert formatted.shape == (3, 84, 84)
    assert formatted.dtype == torch.float32


def test_rl_sac_runtime_utils(monkeypatch):
    from rl.sac import runtime_utils as sac_rt

    monkeypatch.setattr(sac_rt, "_mps_is_available", lambda: False)
    d = sac_rt.select_device("cpu")
    assert str(d) == "cpu"

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,)),
        ),
        ensure_spaces=lambda: None,
    )
    lb, width = sac_rt.obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.array([-1.0]))
    assert np.allclose(width, np.array([2.0]))


def test_rl_ppo_eval_helpers():
    from rl.core import ppo_eval

    out = ppo_eval.update_best_actor_if_improved(
        eval_return=2.0,
        best_return=1.0,
        best_actor_state=None,
        capture_actor_state=lambda: {"k": "v"},
    )
    assert out[0] == 2.0
    assert out[1] == {"k": "v"}
    assert out[2] is True

    out2 = ppo_eval.update_best_actor_if_improved(
        eval_return=0.5,
        best_return=1.0,
        best_actor_state={"old": 1},
        capture_actor_state=lambda: {"new": 2},
    )
    assert out2[0] == 1.0
    assert out2[1] == {"old": 1}
    assert out2[2] is False

    result = ppo_eval.evaluate_heldout_with_best_actor(
        best_actor_state=None,
        num_denoise_passive=1,
        heldout_i_noise=0,
        with_actor_state=lambda s: __import__("contextlib").nullcontext(),
        evaluate_for_best=lambda *a, **k: 3.0,
        eval_env_conf=None,
        eval_policy=None,
    )
    assert result is None

    calls = []

    class _Ctx:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *a):
            calls.append("exit")
            return False

    result2 = ppo_eval.evaluate_heldout_with_best_actor(
        best_actor_state={"snap": 1},
        num_denoise_passive=1,
        heldout_i_noise=0,
        with_actor_state=lambda s: _Ctx(),
        evaluate_for_best=lambda *a, **k: 2.5,
        eval_env_conf=None,
        eval_policy=None,
    )
    assert result2 == 2.5
    assert calls == ["enter", "exit"]


def test_rl_core_runtime_helpers(monkeypatch):
    from rl.core import runtime

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", None)
    d = runtime.select_device("cpu")
    assert str(d) == "cpu"
    d_auto = runtime.select_device("auto")
    assert str(d_auto) == "cpu"

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,)),
        ),
        ensure_spaces=lambda: None,
    )
    lb, width = runtime.obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.array([-1.0]))
    assert np.allclose(width, np.array([2.0]))

    runtime.seed_everything(42)
    kw = runtime.collector_device_kwargs(torch.device("cpu"))
    assert kw["env_device"].type == "cpu"
    with runtime.temporary_distribution_validate_args(False):
        pass


def test_rl_core_envs_dataclasses():
    from rl.core import envs

    rs = envs.ResolvedSeeds(problem_seed=7, noise_seed_0=11)
    assert rs.problem_seed == 7 and rs.noise_seed_0 == 11

    sec = envs.SeededEnvConf(env_conf=object(), problem_seed=1, noise_seed_0=2)
    assert sec.problem_seed == 1

    ces = envs.ContinuousEnvSetup(
        env_conf=object(),
        problem_seed=0,
        noise_seed_0=0,
        act_dim=2,
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        obs_lb=None,
        obs_width=None,
    )
    assert ces.act_dim == 2


def test_rl_core_pixel_transform_atari():
    from rl.core.pixel_transform import ensure_atari_obs_format

    img = torch.randint(0, 256, (84, 84, 4), dtype=torch.uint8)
    out = ensure_atari_obs_format(img, size=84)
    assert out.shape == (4, 84, 84)


def test_replay_core_numpy_buffer_and_make():
    from rl.core.replay import NumpyReplayBuffer, make_replay_buffer, resolve_replay_backend

    buf = make_replay_buffer(obs_shape=(4,), act_dim=2, capacity=16, backend="numpy")
    assert isinstance(buf, NumpyReplayBuffer)
    buf.add_batch(
        obs=np.ones((3, 4), dtype=np.float32),
        act=np.zeros((3, 2), dtype=np.float32),
        rew=np.array([1.0, 0.5, 0.0], dtype=np.float32),
        nxt=np.ones((3, 4), dtype=np.float32),
        done=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    assert buf.size == 3
    obs, act, rew, nxt, done = buf.sample(batch_size=2, device=torch.device("cpu"))
    assert obs.shape == (2, 4)
    assert act.shape == (2, 2)
    state = buf.state_dict()
    buf2 = make_replay_buffer(obs_shape=(4,), act_dim=2, capacity=16, backend="numpy")
    buf2.load_state_dict(state)
    assert buf2.size == buf.size
    assert resolve_replay_backend("auto", device=torch.device("cpu"), platform_name="darwin") == "numpy"


def test_profiler_run_with_profiler(monkeypatch, tmp_path: Path):
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


def test_puffer_eval_helpers_and_validation():
    model = SimpleNamespace(
        actor_backbone=nn.Identity(),
        actor_head=nn.Linear(4, 2),
    )
    obs_spec = SimpleNamespace(mode="vector")
    action_spec = SimpleNamespace(kind="continuous")
    policy = PPOEvalPolicy(
        model=model,
        obs_spec=obs_spec,
        action_spec=action_spec,
        device=torch.device("cpu"),
        prepare_obs_fn=lambda state, **_kwargs: torch.as_tensor(state, dtype=torch.float32),
    )
    act = policy(np.zeros((4,), dtype=np.float32))
    assert act.shape == (2,)

    invalid_cfg = SimpleNamespace(
        eval_interval=1,
        eval_noise_mode=None,
        num_denoise=1,
        num_denoise_passive=1,
        checkpoint_interval=1,
        video_num_episodes=1,
        video_num_video_episodes=0,
        video_episode_selection="invalid",
    )
    with pytest.raises(ValueError, match="video_episode_selection"):
        validate_eval_config(invalid_cfg)
