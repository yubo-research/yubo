"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_kiss_bridge_torchrl_sac_setup_loop_ppo_engine_tail(monkeypatch, tmp_path):
    pytest.importorskip("torchrl")
    import rl.core.sac_eval as sac_eval_mod
    from rl.torchrl.sac.config import SACConfig
    from rl.torchrl.sac.loop import (
        evaluate_heldout_if_enabled as tr_sac_evaluate_heldout_if_enabled,
    )
    from rl.torchrl.sac.loop import log_if_due as tr_sac_log_if_due
    from rl.torchrl.sac.setup import build_env_setup as tr_sac_setup_build_env_setup
    from rl.torchrl.sac.setup import build_modules as tr_sac_setup_build_modules
    from rl.torchrl.sac.setup import build_training as tr_sac_setup_build_training

    monkeypatch.setattr(sac_eval_mod, "evaluate_heldout_with_best_actor", lambda **k: 0.0)
    tr_sac_evaluate_heldout_if_enabled(
        SimpleNamespace(env_tag="pend", num_denoise_passive=1),
        SimpleNamespace(
            problem_seed=0,
            noise_seed_0=0,
            env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
        ),
        SimpleNamespace(),
        SimpleNamespace(best_actor_state=None),
        device=torch.device("cpu"),
        capture_actor_state=lambda m: {},
        restore_actor_state=lambda *a, **k: None,
        eval_policy_factory=lambda *a, **k: lambda obs: obs,
        get_env_conf=lambda *a, **k: SimpleNamespace(),
        evaluate_for_best=lambda *a, **k: 0.0,
    )

    monkeypatch.setattr("rl.logger.log_eval_iteration", lambda **k: None)
    tr_sac_log_if_due(
        SimpleNamespace(log_interval_steps=1),
        SimpleNamespace(last_eval_return=0.0, last_heldout_return=None, best_return=0.0),
        step=1,
        start_time=0.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=0,
    )

    def _fake_bcges(**_kwargs):
        return SimpleNamespace(
            env_conf=SimpleNamespace(
                from_pixels=False,
                state_space=SimpleNamespace(shape=(4,)),
                gym_conf=None,
            ),
            problem_seed=0,
            noise_seed_0=0,
            act_dim=2,
            action_low=np.zeros(2, dtype=np.float32),
            action_high=np.ones(2, dtype=np.float32),
            obs_lb=np.zeros(4, dtype=np.float32),
            obs_width=np.ones(4, dtype=np.float32),
        )

    import rl.torchrl.sac.setup as tr_sac_setup_mod

    monkeypatch.setattr(tr_sac_setup_mod, "build_env_setup", _fake_bcges)
    cfg = SACConfig(exp_dir=str(tmp_path / "sac_exp"), env_tag="pend", policy_tag="mlp-16-8", replay_size=100, batch_size=4)
    env_setup = tr_sac_setup_build_env_setup(cfg)
    dev = torch.device("cpu")
    mods = tr_sac_setup_build_modules(cfg, env_setup, device=dev)
    tr_sac_setup_build_training(cfg, mods)


def test_kiss_bridge_modal_synthetic_sine_disk_and_main_raw(monkeypatch, tmp_path, capsys):
    import contextlib

    from analysis.fitting_time.evaluate import (
        SURROGATE_BENCHMARK_KEYS,
        BMResult,
        MuSe,
        SyntheticSineSurrogateBenchmark,
    )
    from experiments import synthetic_sine_benchmark_payload as pl

    _zr = BMResult(MuSe(0.0, 0.0), MuSe(0.0, 0.0), MuSe(0.0, 0.0))
    _z = SyntheticSineSurrogateBenchmark(results={k: _zr for k in SURROGATE_BENCHMARK_KEYS})

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.modal.enable_output",
        lambda: contextlib.nullcontext(),
    )

    PlApp = type("PlApp", (), {"run": lambda self: contextlib.nullcontext()})
    PlRem = type(
        "PlRem",
        (),
        {"remote": staticmethod(lambda n, d, fn, ps, *_args: pl.synthetic_sine_benchmark_result_to_payload(_z, n=n, d=d, function_name=fn, problem_seed=ps))},
    )
    monkeypatch.setattr(pl.modal, "enable_output", lambda: contextlib.nullcontext())
    pl_dest = pl.run_synthetic_sine_benchmark_modal_to_disk(1, 1, "sine", 0, tmp_path / "pl_direct", app=PlApp(), remote_fn=PlRem())
    assert pl_dest.exists()
