from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch


def test_ppo_eval_noise_mode_natural_advances_eval_and_heldout(monkeypatch, tmp_path):
    from rl.torchrl.ppo import core as ppo_core

    eval_seeds: list[int] = []
    heldout_noise_indices: list[int] = []

    def _fake_run(**kwargs):
        ep = ppo_core.rl_eval.plan(
            current=kwargs["current"],
            interval=kwargs["interval"],
            seed=kwargs["seed"],
            eval_seed_base=kwargs["eval_seed_base"],
            eval_noise_mode=kwargs["eval_noise_mode"],
        )
        state = kwargs["state"]
        eval_seeds.append(int(ep.eval_seed))
        heldout_noise_indices.append(int(ep.heldout_i_noise))
        state.last_eval_return = float(ep.eval_seed)
        state.last_heldout_return = float(ep.heldout_i_noise)
        state.best_return = max(float(state.best_return), float(state.last_eval_return))
        state.best_actor_state = {"snapshot": 1}

    monkeypatch.setattr(ppo_core.rl_eval, "run", _fake_run)

    config = ppo_core.PPOConfig(
        eval_interval=1,
        eval_seed_base=100,
        eval_noise_mode="natural",
        num_denoise_passive=3,
    )
    env_setup = SimpleNamespace(problem_seed=7, noise_seed_0=70)
    modules = SimpleNamespace(actor_backbone=object(), actor_head=object(), obs_scaler=object())
    training_setup = SimpleNamespace(
        frames_per_batch=8,
        num_iterations=10,
        metrics_path=tmp_path / "metrics.jsonl",
    )
    train_state = ppo_core._TrainState()

    ppo_core._maybe_eval_and_log(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        iteration=1,
        approx_kls=[],
        clipfracs=[],
        device=torch.device("cpu"),
        start_time=time.time() - 1.0,
    )
    ppo_core._maybe_eval_and_log(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        iteration=2,
        approx_kls=[],
        clipfracs=[],
        device=torch.device("cpu"),
        start_time=time.time() - 1.0,
    )

    assert eval_seeds == [100, 101]
    assert heldout_noise_indices == [100, 101]


def test_ppo_eval_noise_mode_frozen_uses_fixed_seeds(monkeypatch, tmp_path):
    from rl.torchrl.ppo import core as ppo_core

    eval_seeds: list[int] = []
    heldout_noise_indices: list[int] = []

    def _fake_run(**kwargs):
        ep = ppo_core.rl_eval.plan(
            current=kwargs["current"],
            interval=kwargs["interval"],
            seed=kwargs["seed"],
            eval_seed_base=kwargs["eval_seed_base"],
            eval_noise_mode=kwargs["eval_noise_mode"],
        )
        state = kwargs["state"]
        eval_seeds.append(int(ep.eval_seed))
        heldout_noise_indices.append(int(ep.heldout_i_noise))
        state.last_eval_return = float(ep.eval_seed)
        state.last_heldout_return = float(ep.heldout_i_noise)
        state.best_return = max(float(state.best_return), float(state.last_eval_return))
        state.best_actor_state = {"snapshot": 1}

    monkeypatch.setattr(ppo_core.rl_eval, "run", _fake_run)

    config = ppo_core.PPOConfig(
        eval_interval=1,
        eval_seed_base=100,
        eval_noise_mode="frozen",
        num_denoise_passive=3,
    )
    env_setup = SimpleNamespace(problem_seed=7, noise_seed_0=70)
    modules = SimpleNamespace(actor_backbone=object(), actor_head=object(), obs_scaler=object())
    training_setup = SimpleNamespace(
        frames_per_batch=8,
        num_iterations=10,
        metrics_path=tmp_path / "metrics.jsonl",
    )
    train_state = ppo_core._TrainState()

    ppo_core._maybe_eval_and_log(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        iteration=1,
        approx_kls=[],
        clipfracs=[],
        device=torch.device("cpu"),
        start_time=time.time() - 1.0,
    )
    ppo_core._maybe_eval_and_log(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        iteration=2,
        approx_kls=[],
        clipfracs=[],
        device=torch.device("cpu"),
        start_time=time.time() - 1.0,
    )

    assert eval_seeds == [100, 100]
    assert heldout_noise_indices == [99999, 99999]


def test_ppo_eval_noise_mode_invalid_rejected_before_env_build(monkeypatch):
    from rl.torchrl.ppo import core as ppo_core

    monkeypatch.setattr(
        ppo_core,
        "build_env_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("env build should not run")),
    )
    with pytest.raises(ValueError, match="eval_noise_mode must be one of"):
        ppo_core.train_ppo(ppo_core.PPOConfig(eval_noise_mode="invalid-mode"))
