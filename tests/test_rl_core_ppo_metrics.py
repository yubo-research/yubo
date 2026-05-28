import logging

from rl.core import ppo_metrics
from rl.core.ppo_metrics import build_eval_record


def test_finite_mean_ignores_nonfinite_values():
    assert ppo_metrics.finite_mean([]) is None
    assert ppo_metrics.finite_mean([float("nan"), float("inf")]) is None
    assert ppo_metrics.finite_mean([1.0, float("nan"), 3.0]) == 2.0


def test_build_algo_metrics_uses_finite_means():
    metrics = ppo_metrics.build_algo_metrics(
        approx_kls=[0.1, float("nan"), 0.3],
        clipfracs=[0.0, 1.0],
    )
    assert metrics == {"kl": 0.2, "clipfrac": 0.5}


def test_update_and_log_record_diagnostics(caplog):
    record = {}
    ppo_metrics.update_record_diagnostics(
        record,
        rollout_metrics={"nonfinite_reward_fraction": 0.25},
        update_stats={
            "nonfinite_gae_fractions": [0.0, 0.5],
            "skipped_updates": [0.0, 1.0],
            "ess_values": [2.0, 4.0],
            "loss_objective": [1.0],
            "loss_critic": [2.0],
            "loss_entropy": [3.0],
            "grad_norm": [4.0],
        },
    )

    assert record["nonfinite_reward_fraction"] == 0.25
    assert record["nonfinite_gae_fraction"] == 0.25
    assert record["skipped_update_fraction"] == 0.5
    assert record["ess"] == 3.0
    assert record["loss_objective"] == 1.0
    assert record["loss_critic"] == 2.0
    assert record["loss_entropy"] == 3.0
    assert record["grad_norm"] == 4.0

    logger = logging.getLogger("rl.core.ppo_metrics")
    old_propagate = logger.propagate
    old_level = logger.level
    try:
        # In some environments the logging setup prevents propagation to root
        # (where caplog installs its handler). Attach caplog's handler directly.
        logger.addHandler(caplog.handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = True
        with caplog.at_level(logging.WARNING):
            ppo_metrics.log_record_diagnostics(record, iteration=7)
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate

    assert "ppo diagnostics iteration=7" in caplog.text


def test_build_eval_record_fields_and_sps():
    record = build_eval_record(
        iteration=3,
        global_step=120,
        eval_return=11.0,
        heldout_return=9.5,
        best_return=12.0,
        approx_kl=0.02,
        clipfrac=0.15,
        started_at=10.0,
        now=15.0,
    )
    assert record["iteration"] == 3
    assert record["global_step"] == 120
    assert record["eval_return"] == 11.0
    assert record["heldout_return"] == 9.5
    assert record["best_return"] == 12.0
    assert record["approx_kl"] == 0.02
    assert record["clipfrac"] == 0.15
    assert record["time_seconds"] == 5.0
    assert record["steps_per_second"] == 24.0


def test_build_eval_record_preserves_none_values():
    record = build_eval_record(
        iteration=1,
        global_step=0,
        eval_return=None,
        heldout_return=None,
        best_return=None,
        approx_kl=None,
        clipfrac=None,
        started_at=20.0,
        now=20.0,
    )
    assert record["eval_return"] is None
    assert record["heldout_return"] is None
    assert record["best_return"] is None
    assert record["approx_kl"] is None
    assert record["clipfrac"] is None
