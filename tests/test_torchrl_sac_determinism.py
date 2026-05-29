import json
import math
from pathlib import Path

import pytest

from rl.torchrl.sac import train_sac
from rl.torchrl.sac.config import (
    SACCollectorConfig,
    SACConfig,
    SACEvalConfig,
    SACOptimConfig,
    SACReplayBufferConfig,
)

_EXPECTED_STEPS = [32, 64, 96]
_EXPECTED_TOTAL_UPDATES = [14, 30, 46]


def _build_test_config(exp_dir: Path) -> SACConfig:
    return SACConfig(
        exp_dir=str(exp_dir),
        env_tag="pend",
        policy_tag="mlp-16-8",
        seed=7,
        device="cpu",
        log_interval_steps=32,
        collector=SACCollectorConfig(total_frames=96, init_random_frames=8),
        replay_buffer=SACReplayBufferConfig(batch_size=8, size=2000),
        optim=SACOptimConfig(update_every=2, optim_steps_per_batch=1),
        eval=SACEvalConfig(interval_steps=32, num_denoise=1, num_denoise_passive=1),
    )


def _read_metric_rows(metrics_path: Path) -> list[dict]:
    keys = [
        "step",
        "ret_eval",
        "ret_heldout",
        "ret_best",
        "actor",
        "critic",
        "alpha",
        "total_updates",
    ]
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    return [{key: row.get(key) for key in keys} for row in rows]


def _assert_close(a: float | None, b: float | None, *, atol: float) -> None:
    """Compare floats; treat NaN == NaN as equal (pytest.approx does not)."""
    if a is None and b is None:
        return
    if math.isnan(a) and math.isnan(b):
        return
    assert a == pytest.approx(b, abs=atol)


def _assert_metric_rows_close(observed_rows: list[dict], expected_rows: list[dict], *, atol: float = 1e-6) -> None:
    assert len(observed_rows) == len(expected_rows)
    for observed, expected in zip(observed_rows, expected_rows):
        assert observed["step"] == expected["step"]
        assert observed["total_updates"] == expected["total_updates"]
        for key in (
            "ret_eval",
            "ret_heldout",
            "ret_best",
            "actor",
            "critic",
            "alpha",
        ):
            _assert_close(observed[key], expected[key], atol=atol)


def _assert_metric_rows_well_formed(rows: list[dict]) -> None:
    assert [row["step"] for row in rows] == _EXPECTED_STEPS
    assert [row["total_updates"] for row in rows] == _EXPECTED_TOTAL_UPDATES
    best_return = -float("inf")
    for row in rows:
        for key in ("ret_eval", "actor", "critic", "alpha"):
            assert math.isfinite(float(row[key]))
        if row.get("ret_heldout") is not None:
            assert math.isfinite(float(row["ret_heldout"]))
        best_return = max(best_return, float(row["ret_eval"]))
        assert row["ret_best"] == pytest.approx(best_return, abs=1e-6)


def test_sac_metrics_are_well_formed(tmp_path):
    exp_dir = Path(tmp_path) / "sac_regression"
    cfg = _build_test_config(exp_dir)
    result = train_sac(cfg)
    assert result.num_steps == 96
    observed_rows = _read_metric_rows(exp_dir / "metrics.jsonl")
    _assert_metric_rows_well_formed(observed_rows)


def test_sac_repeatability_same_seed(tmp_path):
    exp_a = Path(tmp_path) / "sac_run_a"
    exp_b = Path(tmp_path) / "sac_run_b"
    train_sac(_build_test_config(exp_a))
    train_sac(_build_test_config(exp_b))
    rows_a = _read_metric_rows(exp_a / "metrics.jsonl")
    rows_b = _read_metric_rows(exp_b / "metrics.jsonl")
    _assert_metric_rows_close(rows_a, rows_b, atol=1e-10)
