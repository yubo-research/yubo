from rl.core.sac_metrics import (
    log,
    normalize_returns_for_log,
    record,
)


def test_build_eval_metric_record_fields():
    out = record(
        step=20,
        eval_return=3.5,
        heldout_return=2.0,
        best_return=4.0,
        loss_actor=1.1,
        loss_critic=2.2,
        loss_alpha=3.3,
        total_updates=7,
        started_at=10.0,
        now=14.0,
    )
    assert out["step"] == 20
    assert out["eval_return"] == 3.5
    assert out["heldout_return"] == 2.0
    assert out["best_return"] == 4.0
    assert out["loss_actor"] == 1.1
    assert out["loss_critic"] == 2.2
    assert out["loss_alpha"] == 3.3
    assert out["total_updates"] == 7
    assert out["time_seconds"] == 4.0
    assert out["steps_per_second"] == 5.0


def test_normalize_returns_for_log_handles_non_finite():
    eval_return, heldout_return, best_return = normalize_returns_for_log(
        eval_return=float("nan"),
        heldout_return=None,
        best_return=float("inf"),
    )
    assert eval_return is None
    assert heldout_return is None
    assert best_return == 0.0


def test_build_log_eval_iteration_kwargs_fields():
    kwargs = log(
        step=12,
        frames_per_batch=4,
        started_at=10.0,
        now=13.0,
        eval_return=2.5,
        heldout_return=1.5,
        best_return=3.0,
        loss_actor=0.1,
        loss_critic=0.2,
        loss_alpha=0.3,
    )
    assert kwargs["iteration"] == 0
    assert kwargs["num_iterations"] == 0
    assert kwargs["frames_per_batch"] == 4
    assert kwargs["eval_return"] == 2.5
    assert kwargs["heldout_return"] == 1.5
    assert kwargs["best_return"] == 3.0
    assert kwargs["algo_name"] == "sac"
    assert kwargs["elapsed"] == 3.0
    assert kwargs["step_override"] == 12
    assert kwargs["algo_metrics"]["actor"] == 0.1
    assert kwargs["algo_metrics"]["critic"] == 0.2
    assert kwargs["algo_metrics"]["alpha"] == 0.3
