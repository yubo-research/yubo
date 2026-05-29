from __future__ import annotations

from rl.iter_record import IterInputs
from rl.logger import format_rl_iter_record
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
    line = format_rl_iter_record(record)
    assert line.startswith("ITER:")
    assert "iter = 3" in line
    assert "step = 120" in line
    assert "fps = 80" in line
    assert "ret_eval = 110" in line
    assert "kl = 0.02" in line


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
    line = format_rl_iter_record(record)
    assert "actor = 0.1" in line
