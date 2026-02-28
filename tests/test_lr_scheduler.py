import pytest

from optimizer.lr_scheduler import (
    ConstantLR,
    LinearLRScheduler,
    LRScheduler,
    OneCycleLR,
)


def test_protocol_isinstance():
    assert isinstance(ConstantLR(0.1), LRScheduler)
    assert isinstance(LinearLRScheduler(0.1, 10), LRScheduler)
    assert isinstance(OneCycleLR(0.1, 10), LRScheduler)


# --- ConstantLR tests ---


def test_constant_lr():
    sched = ConstantLR(lr=0.05)
    assert sched.lr == pytest.approx(0.05)
    for _ in range(100):
        sched.step()
    assert sched.lr == pytest.approx(0.05)


def _assert_lr_after_steps(sched, n_steps, expected):
    for _ in range(n_steps):
        sched.step()
    assert sched.lr == pytest.approx(expected)


def test_starts_at_lr_0():
    sched = LinearLRScheduler(lr_0=0.1, num_steps=100)
    assert sched.lr == pytest.approx(0.1)


def test_decays_to_zero():
    _assert_lr_after_steps(LinearLRScheduler(lr_0=0.1, num_steps=100), 100, 0.0)


def test_linear_midpoint():
    _assert_lr_after_steps(LinearLRScheduler(lr_0=0.1, num_steps=100), 50, 0.05)


def test_warmup_starts_at_zero():
    sched = LinearLRScheduler(lr_0=0.1, num_steps=100, warmup_steps=10)
    assert sched.lr == pytest.approx(0.0)


def test_warmup_reaches_lr_0():
    _assert_lr_after_steps(LinearLRScheduler(lr_0=0.1, num_steps=100, warmup_steps=10), 10, 0.1)


def test_warmup_midpoint():
    _assert_lr_after_steps(LinearLRScheduler(lr_0=0.1, num_steps=100, warmup_steps=10), 5, 0.05)


def test_warmup_then_decay():
    sched = LinearLRScheduler(lr_0=0.1, num_steps=100, warmup_steps=10)
    # After warmup (step 10): lr_0 = 0.1
    # Decay over 90 steps from 0.1 to 0.
    # At step 55 (midpoint of decay): lr = 0.1 * (100 - 55) / 90 = 0.05
    for _ in range(55):
        sched.step()
    assert sched.lr == pytest.approx(0.05)


def test_past_end_stays_zero():
    _assert_lr_after_steps(LinearLRScheduler(lr_0=0.1, num_steps=10), 20, 0.0)


def test_monotonically_decreasing_no_warmup():
    sched = LinearLRScheduler(lr_0=0.1, num_steps=50)
    prev = sched.lr
    for _ in range(50):
        sched.step()
        assert sched.lr <= prev
        prev = sched.lr


# --- OneCycleLR tests ---


def test_one_cycle_starts_at_initial_lr():
    sched = OneCycleLR(max_lr=0.1, num_steps=100, div_factor=25.0)
    assert sched.lr == pytest.approx(0.1 / 25.0)


def test_one_cycle_reaches_max_lr():
    sched = OneCycleLR(max_lr=0.1, num_steps=100, pct_start=0.3)
    warmup_steps = int(100 * 0.3)
    for _ in range(warmup_steps):
        sched.step()
    assert sched.lr == pytest.approx(0.1)


def test_one_cycle_warmup_midpoint():
    sched = OneCycleLR(max_lr=0.1, num_steps=100, pct_start=0.5, div_factor=10.0)
    # initial_lr = 0.1 / 10 = 0.01
    # warmup_steps = 50
    # At step 25: frac = 25/50 = 0.5 → lr = 0.01 + 0.5 * (0.1 - 0.01) = 0.055
    for _ in range(25):
        sched.step()
    assert sched.lr == pytest.approx(0.055)


def test_one_cycle_ends_at_min_lr():
    sched = OneCycleLR(max_lr=0.1, num_steps=100, div_factor=25.0, final_div_factor=1e4)
    for _ in range(100):
        sched.step()
    min_lr = 0.1 / 25.0 / 1e4
    assert sched.lr == pytest.approx(min_lr)


def test_one_cycle_decay_is_cosine():
    """At the midpoint of the decay phase, cosine gives (max + min) / 2."""
    sched = OneCycleLR(max_lr=0.1, num_steps=100, pct_start=0.5, div_factor=25.0, final_div_factor=1e4)
    warmup_steps = 50
    decay_steps = 50
    for _ in range(warmup_steps + decay_steps // 2):
        sched.step()
    min_lr = 0.1 / 25.0 / 1e4
    expected = (0.1 + min_lr) / 2.0
    assert sched.lr == pytest.approx(expected, rel=1e-6)


def test_one_cycle_past_end_stays_at_min():
    sched = OneCycleLR(max_lr=0.1, num_steps=10, div_factor=25.0, final_div_factor=1e4)
    for _ in range(20):
        sched.step()
    assert sched.lr == pytest.approx(0.1 / 25.0 / 1e4)


def test_one_cycle_warmup_then_decay():
    """LR should rise during warmup and fall during decay."""
    sched = OneCycleLR(max_lr=0.1, num_steps=100, pct_start=0.3)
    warmup_steps = 30

    # Warmup: monotonically increasing.
    prev = sched.lr
    for _ in range(warmup_steps):
        sched.step()
        assert sched.lr >= prev
        prev = sched.lr

    # Decay: monotonically decreasing.
    for _ in range(70):
        sched.step()
        assert sched.lr <= prev
        prev = sched.lr
