import pytest

from rl.eval_noise import (
    EvalPlan,
    build_eval_plan,
    eval_index_for_due_step,
    normalize_eval_noise_mode,
    resolve_eval_seed,
    resolve_heldout_noise_index,
)


def test_eval_plan():
    p = EvalPlan(eval_seed=7, heldout_i_noise=99999)
    assert p.eval_seed == 7
    assert p.heldout_i_noise == 99999


def test_normalize_eval_noise_mode():
    assert normalize_eval_noise_mode(None) == "frozen"
    assert normalize_eval_noise_mode("frozen") == "frozen"
    assert normalize_eval_noise_mode("natural") == "natural"
    assert normalize_eval_noise_mode("  FROZEN  ") == "frozen"
    with pytest.raises(ValueError, match="must be one of"):
        normalize_eval_noise_mode("invalid")


def test_eval_index_for_due_step():
    assert eval_index_for_due_step(current=32, interval=32) == 1
    assert eval_index_for_due_step(current=64, interval=32) == 2
    with pytest.raises(ValueError, match="current must be > 0"):
        eval_index_for_due_step(current=0, interval=32)
    with pytest.raises(ValueError, match="interval must be > 0"):
        eval_index_for_due_step(current=32, interval=0)
    with pytest.raises(ValueError, match="not divisible"):
        eval_index_for_due_step(current=30, interval=32)


def test_resolve_eval_seed():
    assert resolve_eval_seed(seed=7, eval_seed_base=None, eval_noise_mode="frozen", eval_index=1) == 7
    assert resolve_eval_seed(seed=7, eval_seed_base=100, eval_noise_mode="frozen", eval_index=2) == 100
    assert resolve_eval_seed(seed=7, eval_seed_base=None, eval_noise_mode="natural", eval_index=1) == 7
    assert resolve_eval_seed(seed=7, eval_seed_base=None, eval_noise_mode="natural", eval_index=3) == 9


def test_resolve_heldout_noise_index():
    assert resolve_heldout_noise_index(eval_noise_mode="frozen", eval_seed=7) == 99999
    assert resolve_heldout_noise_index(eval_noise_mode="natural", eval_seed=42) == 42


def test_build_eval_plan():
    plan = build_eval_plan(
        current=32,
        interval=32,
        seed=7,
        eval_seed_base=None,
        eval_noise_mode="frozen",
    )
    assert plan.eval_seed == 7
    assert plan.heldout_i_noise == 99999
