import json
import math
from pathlib import Path

import pytest

from rl.torchrl.sac import train_sac
from rl.torchrl.sac.config import SACConfig

_BASELINE_ROWS = [
    {
        "step": 32,
        "eval_return": -1632.3920898797385,
        "heldout_return": -1260.0408725253917,
        "best_return": -1632.3920898797385,
        "loss_actor": 0.5643786787986755,
        "loss_critic": 110.67838287353516,
        "loss_alpha": -2.797122001647949,
        "total_updates": 14,
    },
    {
        "step": 64,
        "eval_return": -1704.2397681216287,
        "heldout_return": -1260.0408725253917,
        "best_return": -1632.3920898797385,
        "loss_actor": 1.5171889066696167,
        "loss_critic": 75.31757354736328,
        "loss_alpha": -2.8861546516418457,
        "total_updates": 30,
    },
    {
        "step": 96,
        "eval_return": -1632.4531731833333,
        "heldout_return": -1260.0408725253917,
        "best_return": -1632.3920898797385,
        "loss_actor": 2.718534469604492,
        "loss_critic": 89.66275787353516,
        "loss_alpha": -1.9424128532409668,
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
        num_denoise=1,
        num_denoise_passive=1,
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
