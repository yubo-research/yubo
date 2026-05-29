from __future__ import annotations

from rl.mjx_metrics import build_iter_record


def _ppo_metrics() -> dict[str, float]:
    return {
        "rollout_return": 1.0,
        "ep_ret": 1.0,
        "ep_len": 2.0,
        "rollout_reward": 0.25,
        "loss": 0.0,
        "loss_objective": 0.1,
        "loss_critic": 0.2,
        "entropy": 0.3,
        "approx_kl": 0.4,
        "clipfrac": 0.5,
        "done_fraction": 0.6,
    }


def _sac_metrics() -> dict[str, float]:
    return {
        "rollout_return": 2.0,
        "ep_ret": 2.0,
        "ep_len": 3.0,
        "rollout_reward": 0.5,
        "done_fraction": 0.1,
        "loss_actor": 0.2,
        "loss_critic": 0.3,
        "alpha_value": 0.4,
        "loss_alpha": 0.05,
    }


def test_mjx_ppo_iter_record_matches_bo_style_fields() -> None:
    record = build_iter_record(
        algo_name="ppo",
        iteration=2,
        frames_per_iter=32,
        elapsed=4.0,
        iter_dt=0.5,
        metrics=_ppo_metrics(),
        ret_best=3.0,
        ret_eval=2.5,
        eval_dt=0.05,
    )
    assert record["step"] == 64
    assert record["fps"] == 64.0
    assert record["ret_eval"] == 2.5
    assert record["eval_dt"] == 0.05
    assert record["kl"] == 0.4


def test_mjx_sac_iter_record_uses_per_iter_fps() -> None:
    record = build_iter_record(
        algo_name="sac",
        iteration=3,
        frames_per_iter=16,
        elapsed=10.0,
        iter_dt=0.25,
        metrics=_sac_metrics(),
        ret_best=5.0,
    )
    assert record["fps"] == 64.0
    assert record["actor"] == 0.2
    assert record["alpha"] == 0.4


def test_mjx_iter_record_formats_like_bo_iter_line() -> None:
    from rl.logger import format_rl_iter_record

    record = build_iter_record(
        algo_name="ppo",
        iteration=1,
        frames_per_iter=32,
        elapsed=1.0,
        iter_dt=0.5,
        metrics=_ppo_metrics(),
        ret_best=1.0,
    )
    line = format_rl_iter_record(record)
    assert line.startswith("ITER:")
    assert "iter = 1" in line
    assert "fps = 64" in line
    assert "kl = 0.4" in line
