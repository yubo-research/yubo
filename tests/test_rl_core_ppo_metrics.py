from rl.core.ppo_metrics import record


def test_build_eval_record_fields_and_sps():
    out = record(
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
    assert out["iteration"] == 3
    assert out["global_step"] == 120
    assert out["eval_return"] == 11.0
    assert out["heldout_return"] == 9.5
    assert out["best_return"] == 12.0
    assert out["approx_kl"] == 0.02
    assert out["clipfrac"] == 0.15
    assert out["time_seconds"] == 5.0
    assert out["steps_per_second"] == 24.0


def test_build_eval_record_preserves_none_values():
    out = record(
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
    assert out["eval_return"] is None
    assert out["heldout_return"] is None
    assert out["best_return"] is None
    assert out["approx_kl"] is None
    assert out["clipfrac"] is None
