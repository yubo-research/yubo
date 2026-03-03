from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn


def test_core_offpolicy_eval_helpers():
    from rl.core.offpolicy_eval import evaluate_heldout_with_best_actor, update_best_actor_if_improved

    calls: list[tuple] = []

    @contextmanager
    def _with_actor_state(snapshot):
        calls.append(("enter", snapshot))
        try:
            yield
        finally:
            calls.append(("exit", snapshot))

    def _evaluate_for_best(env_conf, policy, num_denoise, i_noise):
        calls.append(("eval", env_conf, policy, num_denoise, i_noise))
        return 3.5

    best_return, best_actor_state, updated = update_best_actor_if_improved(
        eval_return=2.0,
        best_return=1.0,
        best_actor_state=None,
        capture_actor_state=lambda: {"weights": 1},
    )
    assert updated is True
    assert best_return == 2.0
    assert best_actor_state == {"weights": 1}

    heldout = evaluate_heldout_with_best_actor(
        best_actor_state=best_actor_state,
        num_denoise_passive_eval=4,
        heldout_i_noise=123,
        with_actor_state=_with_actor_state,
        evaluate_for_best=_evaluate_for_best,
        eval_env_conf="env",
        eval_policy="policy",
    )
    assert heldout == 3.5
    assert calls[0] == ("enter", {"weights": 1})
    assert calls[1] == ("eval", "env", "policy", 4, 123)
    assert calls[2] == ("exit", {"weights": 1})


def test_core_offpolicy_metrics_helpers():
    from rl.core.offpolicy_metrics import (
        build_eval_metric_record,
        build_log_eval_iteration_kwargs,
        normalize_returns_for_log,
    )

    rec = build_eval_metric_record(
        step=10,
        eval_return=1.2,
        heldout_return=0.8,
        best_return=1.2,
        loss_actor=0.1,
        loss_critic=0.2,
        loss_alpha=0.3,
        total_updates=7,
        started_at=10.0,
        now=12.0,
    )
    assert rec["step"] == 10
    assert rec["total_updates"] == 7
    assert rec["time_seconds"] == 2.0
    assert "steps_per_second" in rec

    eval_out, heldout_out, best_out = normalize_returns_for_log(eval_return=float("nan"), heldout_return=1.0, best_return=float("nan"))
    assert eval_out is None
    assert heldout_out == 1.0
    assert best_out == 0.0

    kwargs = build_log_eval_iteration_kwargs(
        step=20,
        frames_per_batch=1,
        started_at=10.0,
        now=14.0,
        eval_return=2.0,
        heldout_return=None,
        best_return=2.0,
        loss_actor=0.4,
        loss_critic=0.5,
        loss_alpha=0.6,
    )
    assert kwargs["step_override"] == 20
    assert kwargs["algo_metrics"]["actor"] == 0.4


def test_sac_wrappers_delegate_to_offpolicy():
    from rl.core import offpolicy_eval, offpolicy_metrics, sac_eval, sac_metrics

    assert sac_eval.update_best_actor_if_improved is offpolicy_eval.update_best_actor_if_improved
    assert sac_eval.evaluate_heldout_with_best_actor is offpolicy_eval.evaluate_heldout_with_best_actor
    assert sac_metrics.build_eval_metric_record is offpolicy_metrics.build_eval_metric_record
    assert sac_metrics.build_log_eval_iteration_kwargs is offpolicy_metrics.build_log_eval_iteration_kwargs


def test_offpolicy_runtime_and_env_utils(monkeypatch):
    from rl.pufferlib.offpolicy import env_utils, runtime_utils

    assert runtime_utils.select_device("cpu").type == "cpu"

    env_no_scale = SimpleNamespace(gym_conf=SimpleNamespace(transform_state=False), ensure_spaces=lambda: None)
    lb, width = runtime_utils.obs_scale_from_env(env_no_scale)
    assert lb is None and width is None

    env_utils.seed_everything(77)
    first = float(np.random.rand())
    env_utils.seed_everything(77)
    second = float(np.random.rand())
    assert first == second

    mapped = env_utils.to_env_action(
        np.asarray([[-1.0, 1.0]], dtype=np.float32),
        low=np.asarray([-2.0, -2.0], dtype=np.float32),
        high=np.asarray([2.0, 2.0], dtype=np.float32),
    )
    assert np.allclose(mapped, np.asarray([[-2.0, 2.0]], dtype=np.float32))

    cfg_vec = SimpleNamespace(from_pixels=False, backbone_name="mlp", framestack=1)
    spec_vec = env_utils.infer_observation_spec(cfg_vec, np.zeros((2, 5), dtype=np.float32))
    assert spec_vec.mode == "vector"
    vec = env_utils.prepare_obs_np(np.arange(10, dtype=np.float32).reshape(2, 5), obs_spec=spec_vec)
    assert vec.shape == (2, 5)

    cfg_px = SimpleNamespace(from_pixels=True, backbone_name="mlp", framestack=4)
    spec_px = env_utils.infer_observation_spec(cfg_px, np.zeros((2, 84, 84, 4), dtype=np.uint8))
    assert spec_px.mode == "pixels"
    assert env_utils.resolve_backbone_name(cfg_px, spec_px) == "nature_cnn_atari"
    px = env_utils.prepare_obs_np(np.zeros((84, 84, 3), dtype=np.uint8), obs_spec=env_utils.ObservationSpec("pixels", (84, 84, 3), channels=3, image_size=84))
    assert px.shape == (1, 3, 84, 84)

    monkeypatch.setattr(
        env_utils,
        "build_continuous_gym_env_setup",
        lambda **_kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
            problem_seed=11,
            noise_seed_0=22,
            obs_lb=None,
            obs_width=None,
            act_dim=2,
            action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            action_high=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    )
    built = env_utils.build_env_setup(SimpleNamespace(env_tag="pend", seed=0, problem_seed=None, noise_seed_0=None, from_pixels=False, pixels_only=True))
    assert built.problem_seed == 11
    assert built.noise_seed_0 == 22


def test_offpolicy_eval_utils_paths(monkeypatch, tmp_path):
    from rl.pufferlib.offpolicy import eval_utils

    class _Actor:
        def act(self, obs):
            return torch.zeros((obs.shape[0], 2), dtype=torch.float32, device=obs.device)

    modules = SimpleNamespace(
        actor_backbone=nn.Linear(3, 4),
        actor_head=nn.Linear(4, 2),
        log_std=nn.Parameter(torch.zeros(2)),
        actor=_Actor(),
    )
    env = SimpleNamespace(env_conf=SimpleNamespace(), problem_seed=5)
    obs_spec = SimpleNamespace(mode="vector", raw_shape=(3,), vector_dim=3)
    state = eval_utils.TrainState(global_step=20, start_time=10.0)
    cfg = SimpleNamespace(
        seed=0,
        eval_interval_steps=10,
        eval_seed_base=None,
        eval_noise_mode="natural",
        num_denoise_eval=1,
        num_denoise_passive_eval=1,
        log_interval_steps=10,
        video_enable=True,
        video_prefix="offpolicy",
        video_num_episodes=2,
        video_num_video_episodes=1,
        video_episode_selection="best",
        video_seed_base=100,
        exp_dir=str(tmp_path),
    )

    monkeypatch.setattr(eval_utils, "build_eval_plan", lambda **_kwargs: SimpleNamespace(eval_seed=9, heldout_i_noise=7))
    monkeypatch.setattr(eval_utils, "evaluate_actor", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(eval_utils, "evaluate_heldout_if_enabled", lambda *_args, **_kwargs: 1.5)

    metric_rows: list[dict] = []
    log_rows: list[dict] = []
    monkeypatch.setattr(eval_utils.rl_logger, "append_metrics", lambda _path, row: metric_rows.append(row))
    monkeypatch.setattr(eval_utils.rl_logger, "log_eval_iteration", lambda **kwargs: log_rows.append(kwargs))

    eval_utils.maybe_eval(cfg, env, modules, obs_spec, state, device=torch.device("cpu"))
    assert state.last_eval_return == 2.0
    assert state.last_heldout_return == 1.5
    assert state.best_return == 2.0
    assert state.best_actor_state is not None

    eval_utils.append_eval_metric(tmp_path / "metrics.jsonl", state, step=20)
    assert metric_rows and metric_rows[-1]["step"] == 20

    eval_utils.log_if_due(cfg, state, step=20, frames_per_batch=4)
    assert log_rows and log_rows[-1]["step_override"] == 20

    video_calls: list[dict] = []
    monkeypatch.setattr("common.video.render_policy_videos", lambda env_conf, policy, **kwargs: video_calls.append(kwargs))
    eval_utils.render_videos_if_enabled(cfg, env, modules, obs_spec, device=torch.device("cpu"))
    assert video_calls and video_calls[-1]["video_prefix"] == "offpolicy"


def test_torchrl_wpo_actor_eval_exports():
    from rl.torchrl.offpolicy.actor_eval import OffPolicyActorEvalPolicy, capture_actor_snapshot
    from rl.torchrl.wpo import actor_eval as wpo_actor_eval

    assert wpo_actor_eval.OffPolicyActorEvalPolicy is OffPolicyActorEvalPolicy
    assert wpo_actor_eval.capture_actor_snapshot is capture_actor_snapshot


def test_wpo_update_rejects_unknown_squashing_type():
    from rl.core.wpo_update import _squash_gradient

    with pytest.raises(ValueError, match="Unsupported WPO squashing_type"):
        _ = _squash_gradient(torch.ones((2, 3)), "bad")
