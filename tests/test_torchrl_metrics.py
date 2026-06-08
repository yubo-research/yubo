from __future__ import annotations

from rl.iter_record import IterInputs
from rl.logger import format_rl_iter_record, log_rl_iter
from rl.torchrl_metrics import build_ppo_iter_record, build_sac_iter_record


def test_torchrl_ppo_iter_record_format() -> None:
    record = build_ppo_iter_record(
        IterInputs(
            iteration=3,
            step=120,
            frames_per_iter=40,
            elapsed=12.0,
            iter_dt=0.5,
            metrics={
                "kl": 0.02,
                "clipfrac": 0.1,
                "ret_rollout": 100.0,
                "ret_eval": 110.0,
                "ret_best": 115.0,
                "eval_dt": 0.08,
            },
        )
    )
    assert format_rl_iter_record(record).startswith("ITER:")


def test_torchrl_ppo_log_rl_iter_uses_table_not_iter_line(capsys, tmp_path) -> None:
    record = build_ppo_iter_record(
        IterInputs(
            iteration=3,
            step=120,
            frames_per_iter=40,
            elapsed=12.0,
            iter_dt=0.5,
            metrics={"kl": 0.02, "clipfrac": 0.1, "ret_eval": 110.0, "ret_best": 115.0},
        )
    )
    log_rl_iter(record, metrics_path=tmp_path / "m.jsonl", algo_name="ppo")
    out = capsys.readouterr().out
    assert "ITER:" not in out
    assert "    3" in out
    assert "     120" in out


def test_torchrl_sac_iter_record_uses_batch_fps() -> None:
    record = build_sac_iter_record(
        IterInputs(
            iteration=4,
            step=64,
            frames_per_iter=16,
            elapsed=8.0,
            iter_dt=0.25,
            metrics={
                "actor": 0.1,
                "critic": 0.2,
                "alpha": 0.3,
                "ret_best": 5.0,
                "ret_eval": 4.5,
                "total_updates": 10,
            },
        )
    )
    assert record["fps"] == 64.0
    assert record["actor"] == 0.1
    assert record["frames_per_iter"] == 16
