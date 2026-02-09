import json
from pathlib import Path

import pytest

from rl.algos.torchrl_sac import SACConfig, train_sac

_BASELINE_ROWS = [
    {
        "step": 32,
        "eval_return": -1274.38670952083,
        "heldout_return": -1431.1100294384146,
        "best_return": -1274.38670952083,
        "loss_actor": 0.6417927742004395,
        "loss_critic": 2.781093120574951,
        "loss_alpha": -2.753187656402588,
        "total_updates": 13,
    },
    {
        "step": 64,
        "eval_return": -1292.2235349132427,
        "heldout_return": -1431.1100294384146,
        "best_return": -1274.38670952083,
        "loss_actor": 1.7655994892120361,
        "loss_critic": 2.3881845474243164,
        "loss_alpha": -2.6435177326202393,
        "total_updates": 29,
    },
    {
        "step": 96,
        "eval_return": -1315.394701729028,
        "heldout_return": -1431.1100294384146,
        "best_return": -1274.38670952083,
        "loss_actor": 2.7381205558776855,
        "loss_critic": 2.8168418407440186,
        "loss_alpha": -2.9497852325439453,
        "total_updates": 45,
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


def _assert_metric_rows_close(observed_rows: list[dict], expected_rows: list[dict], *, atol: float = 1e-6) -> None:
    assert len(observed_rows) == len(expected_rows)
    for observed, expected in zip(observed_rows, expected_rows):
        assert observed["step"] == expected["step"]
        assert observed["total_updates"] == expected["total_updates"]
        for key in ("eval_return", "heldout_return", "best_return", "loss_actor", "loss_critic", "loss_alpha"):
            assert observed[key] == pytest.approx(expected[key], abs=atol)


def test_sac_metrics_match_baseline(tmp_path):
    exp_dir = Path(tmp_path) / "sac_regression"
    cfg = _build_test_config(exp_dir)
    result = train_sac(cfg)
    assert result.num_steps == 96
    observed_rows = _read_metric_rows(exp_dir / "metrics.jsonl")
    _assert_metric_rows_close(observed_rows, _BASELINE_ROWS, atol=1e-5)


def test_sac_repeatability_same_seed(tmp_path):
    exp_a = Path(tmp_path) / "sac_run_a"
    exp_b = Path(tmp_path) / "sac_run_b"
    train_sac(_build_test_config(exp_a))
    train_sac(_build_test_config(exp_b))
    rows_a = _read_metric_rows(exp_a / "metrics.jsonl")
    rows_b = _read_metric_rows(exp_b / "metrics.jsonl")
    _assert_metric_rows_close(rows_a, rows_b, atol=1e-10)
