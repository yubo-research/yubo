import json
import math
from pathlib import Path

import pytest

from rl.torchrl.sac.trainer import SACConfig, train_sac

_BASELINE_ROWS = [
    {
        "step": 32,
        "eval_return": -1225.942996711321,
        "heldout_return": -1273.3617814039442,
        "best_return": -1225.942996711321,
        "loss_actor": 0.5929603576660156,
        "loss_critic": 83.40846252441406,
        "loss_alpha": -2.7948989868164062,
        "total_updates": 14,
    },
    {
        "step": 64,
        "eval_return": -1286.1293195500311,
        "heldout_return": -1273.3617814039442,
        "best_return": -1225.942996711321,
        "loss_actor": 1.3357847929000854,
        "loss_critic": 49.38249588012695,
        "loss_alpha": -2.438335418701172,
        "total_updates": 30,
    },
    {
        "step": 96,
        "eval_return": -1306.8451731020393,
        "heldout_return": -1273.3617814039442,
        "best_return": -1225.942996711321,
        "loss_actor": 2.8664050102233887,
        "loss_critic": 24.738079071044922,
        "loss_alpha": -2.4196784496307373,
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
