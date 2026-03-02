from rl.core.ppo_metrics import build_eval_record


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
