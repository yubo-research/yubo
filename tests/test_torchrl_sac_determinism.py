import json
import math
from pathlib import Path

import pytest

from rl.algos.backends.torchrl.sac.trainer import SACConfig, train_sac

# Baseline with BO-style seeding: seed=7 → problem_seed=24, seed_all(24+27).
# Captured from current train_sac loop (no initial eval, eval at step 96 before break).
_BASELINE_ROWS = [
    {
        "step": 32,
        "eval_return": -1170.3912758206882,
        "heldout_return": -1467.798166303253,
        "best_return": -1170.3912758206882,
        "loss_actor": 0.5280730128288269,
        "loss_critic": 9.374556541442871,
        "loss_alpha": -2.4916810989379883,
        "total_updates": 14,
    },
    {
        "step": 64,
        "eval_return": -1287.348388671875,
        "heldout_return": -1467.798166303253,
        "best_return": -1170.3912758206882,
        "loss_actor": 1.4693553447723389,
        "loss_critic": 3.363049030303955,
        "loss_alpha": -3.243229389190674,
        "total_updates": 30,
    },
    {
        "step": 96,
        "eval_return": -1299.128662109375,
        "heldout_return": -1467.798166303253,
        "best_return": -1170.3912758206882,
        "loss_actor": 1.9300659894943237,
        "loss_critic": 8.443120002746582,
        "loss_alpha": -2.1574792861938477,
        "total_updates": 46,
    },
]


def _build_test_config(exp_dir: Path) -> SACConfig:
    return SACConfig(
        exp_dir=str(exp_dir),
        env_tag="pend",
        seed=7,
        device="cpu",
        total_timesteps=96,
        learning_starts=8,
        batch_size=8,
        replay_size=2000,
        update_every=2,
        updates_per_step=1,
        eval_interval_steps=32,
        log_interval_steps=32,
        num_denoise_eval=1,
        num_denoise_passive_eval=1,
    )


def _read_metric_rows(metrics_path: Path) -> list[dict]:
    keys = [
        "step",
        "eval_return",
        "heldout_return",
        "best_return",
        "loss_actor",
        "loss_critic",
        "loss_alpha",
        "total_updates",
    ]
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    return [{key: row.get(key) for key in keys} for row in rows]


def _assert_close(a: float, b: float, *, atol: float) -> None:
    """Compare floats; treat NaN == NaN as equal (pytest.approx does not)."""
    if math.isnan(a) and math.isnan(b):
        return
    assert a == pytest.approx(b, abs=atol)


def _assert_metric_rows_close(observed_rows: list[dict], expected_rows: list[dict], *, atol: float = 1e-6) -> None:
    assert len(observed_rows) == len(expected_rows)
    for observed, expected in zip(observed_rows, expected_rows):
        assert observed["step"] == expected["step"]
        assert observed["total_updates"] == expected["total_updates"]
        for key in (
            "eval_return",
            "heldout_return",
            "best_return",
            "loss_actor",
            "loss_critic",
            "loss_alpha",
        ):
            _assert_close(observed[key], expected[key], atol=atol)


def test_sac_metrics_match_baseline(tmp_path):
    exp_dir = Path(tmp_path) / "sac_regression"
    cfg = _build_test_config(exp_dir)
    result = train_sac(cfg)
    assert result.num_steps == 96
    observed_rows = _read_metric_rows(exp_dir / "metrics.jsonl")
    _assert_metric_rows_close(observed_rows, _BASELINE_ROWS, atol=1e-2)


def test_sac_repeatability_same_seed(tmp_path):
    exp_a = Path(tmp_path) / "sac_run_a"
    exp_b = Path(tmp_path) / "sac_run_b"
    train_sac(_build_test_config(exp_a))
    train_sac(_build_test_config(exp_b))
    rows_a = _read_metric_rows(exp_a / "metrics.jsonl")
    rows_b = _read_metric_rows(exp_b / "metrics.jsonl")
    _assert_metric_rows_close(rows_a, rows_b, atol=1e-10)
