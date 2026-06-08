from __future__ import annotations

from rl.iter_record import IterInputs
from rl.logger import infer_algo_name, log_rl_iter
from rl.torchrl_metrics import build_sac_iter_record


def test_infer_algo_name_from_record_fields() -> None:
    assert infer_algo_name({"kl": 0.1}) == "ppo"
    assert infer_algo_name({"actor": 0.2, "critic": 0.3}) == "sac"


def test_log_rl_iter_prints_aligned_table_row(capsys, tmp_path) -> None:
    record = build_sac_iter_record(
        IterInputs(
            iteration=4,
            step=4096,
            frames_per_iter=1024,
            elapsed=12.5,
            iter_dt=0.40,
            metrics={
                "actor": 0.11,
                "critic": 0.22,
                "alpha": 0.33,
                "ret_best": -120.0,
                "ret_eval": -115.0,
                "eval_dt": 0.08,
            },
        )
    )
    metrics_path = tmp_path / "metrics.jsonl"
    log_rl_iter(record, metrics_path=metrics_path, algo_name="sac")

    out = capsys.readouterr().out
    assert "ITER:" not in out
    assert "4,096" in out
    assert "-115.0" in out
    assert "0.40s" in out
    assert "0.08s" in out
    assert metrics_path.read_text(encoding="utf-8").strip()
