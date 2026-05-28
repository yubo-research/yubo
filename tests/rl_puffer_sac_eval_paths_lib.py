from __future__ import annotations

from types import SimpleNamespace


def run_eval_utils_policy_and_evaluate(monkeypatch, eval_utils, puffer_sac, env_setup, obs_spec, cfg):
    import numpy as np
    import torch

    class _Actor:
        @staticmethod
        def act(obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros((obs.shape[0], 2), dtype=torch.float32)

    policy = eval_utils.SacEvalPolicy(SimpleNamespace(actor=_Actor()), obs_spec, device=torch.device("cpu"))
    out = policy(np.zeros((3,), dtype=np.float32))
    assert out.shape == (2,)

    monkeypatch.setattr(
        eval_utils,
        "collect_denoised_trajectory",
        lambda _env_conf, _policy, **_kwargs: (SimpleNamespace(rreturn=3.5), 0),
    )
    eval_return = eval_utils.evaluate_actor(
        cfg,
        env_setup,
        SimpleNamespace(actor=_Actor()),
        obs_spec,
        device=torch.device("cpu"),
        eval_seed=0,
    )
    assert eval_return == 3.5
    return _Actor


def run_eval_utils_heldout_and_due(monkeypatch, eval_utils, puffer_sac, env_setup, obs_spec, _Actor):
    import torch

    cfg_no_heldout = puffer_sac.SACConfig(num_denoise_passive=None)
    assert (
        eval_utils.evaluate_heldout_if_enabled(
            cfg_no_heldout,
            env_setup,
            SimpleNamespace(actor=_Actor()),
            obs_spec,
            device=torch.device("cpu"),
            heldout_i_noise=0,
        )
        is None
    )
    monkeypatch.setattr(eval_utils, "evaluate_for_best", lambda *_args, **_kwargs: 4.2)
    heldout = eval_utils.evaluate_heldout_if_enabled(
        puffer_sac.SACConfig(num_denoise_passive=2),
        env_setup,
        SimpleNamespace(actor=_Actor()),
        obs_spec,
        device=torch.device("cpu"),
        heldout_i_noise=7,
    )
    assert heldout == 4.2
    assert eval_utils.due_mark(0, 10, 0) is None
    assert eval_utils.due_mark(10, 10, 0) == 1
    assert eval_utils.due_mark(10, 0, 0) is None


def run_eval_utils_metrics_and_maybe_eval(monkeypatch, eval_utils, puffer_sac, env_setup, obs_spec, modules, tmp_path):
    import time

    import torch

    metric_calls = []
    monkeypatch.setattr(
        eval_utils.rl_logger,
        "append_metrics",
        lambda path, record: metric_calls.append((path, record)),
    )
    state = eval_utils.TrainState(start_time=float(time.time()) - 1.0)
    state.last_eval_return = 1.25
    state.last_heldout_return = 0.75
    state.best_return = 1.25
    eval_utils.append_eval_metric(tmp_path / "metrics.jsonl", state, step=10)
    assert len(metric_calls) == 1
    assert metric_calls[0][1]["step"] == 10

    log_calls = []
    monkeypatch.setattr(
        eval_utils.rl_logger,
        "log_eval_iteration",
        lambda **kwargs: log_calls.append(kwargs),
    )
    log_cfg = puffer_sac.SACConfig(log_interval_steps=5)
    eval_utils.log_if_due(log_cfg, state, step=5, frames_per_batch=8)
    assert len(log_calls) == 1
    assert state.log_mark == 1

    monkeypatch.setattr(
        eval_utils,
        "build_eval_plan",
        lambda **_kwargs: SimpleNamespace(eval_seed=1, heldout_i_noise=2),
    )
    monkeypatch.setattr(eval_utils, "evaluate_actor", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(eval_utils, "evaluate_heldout_if_enabled", lambda *_args, **_kwargs: 1.5)
    eval_cfg = puffer_sac.SACConfig(eval_interval_steps=10, num_denoise_passive=1)
    state2 = eval_utils.TrainState(global_step=10, start_time=float(time.time()) - 1.0)
    eval_utils.maybe_eval(eval_cfg, env_setup, modules, obs_spec, state2, device=torch.device("cpu"))
    assert state2.eval_mark == 1
    assert state2.best_return == 2.0
    assert state2.best_actor_state is not None
    assert state2.last_heldout_return == 1.5
    state2.global_step = 20
    monkeypatch.setattr(eval_utils, "evaluate_actor", lambda *_args, **_kwargs: 1.0)
    eval_utils.maybe_eval(eval_cfg, env_setup, modules, obs_spec, state2, device=torch.device("cpu"))
    assert state2.best_return == 2.0


def run_eval_utils_video_if_enabled(monkeypatch, eval_utils, puffer_sac, env_setup, modules, obs_spec, tmp_path):
    import torch

    video_calls = []
    monkeypatch.setattr(
        "common.video.render_policy_videos",
        lambda *args, **kwargs: video_calls.append((args, kwargs)),
    )
    video_cfg = puffer_sac.SACConfig(
        exp_dir=str(tmp_path / "exp"),
        video_enable=True,
        video_num_episodes=3,
        video_num_video_episodes=1,
        video_seed_base=None,
    )
    eval_utils.render_videos_if_enabled(video_cfg, env_setup, modules, obs_spec, device=torch.device("cpu"))
    assert len(video_calls) == 1
    assert video_calls[0][1]["seed_base"] == env_setup.problem_seed + 10000
