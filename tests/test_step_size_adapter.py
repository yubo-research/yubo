import pytest

from optimizer.step_size_adapter import StepSizeAdapter


def test_initial_sigma():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10)
    assert adapter.sigma == 0.1


def test_expand_after_success_tolerance():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    adapter.update(accepted=True)  # 3rd consecutive success -> expand
    assert adapter.sigma > 0.1


def test_expand_doubles_sigma():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    assert adapter.sigma == 0.2


def test_expand_capped_at_sigma_max():
    adapter = StepSizeAdapter(sigma_0=0.15, dim=10, sigma_max=0.2, success_tolerance=3)
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    assert adapter.sigma == 0.2  # capped, not 0.3


def test_shrink_after_failure_tolerance():
    # failure_tolerance = max(10, 5*3) = 15
    adapter = StepSizeAdapter(sigma_0=0.1, dim=100)
    for _ in range(14):
        adapter.update(accepted=False)
    assert adapter.sigma == 0.1  # not yet
    adapter.update(accepted=False)  # 15th failure -> shrink
    assert adapter.sigma == 0.05


def test_shrink_floored_at_sigma_min():
    adapter = StepSizeAdapter(sigma_0=1e-5, dim=100, sigma_min=1e-5, restart_on_floor=False)
    for _ in range(15):
        adapter.update(accepted=False)
    assert adapter.sigma == 1e-5


def test_restart_on_floor_resets_to_sigma_init():
    adapter = StepSizeAdapter(sigma_0=0.01, dim=100, sigma_min=0.005, success_tolerance=3)
    for _ in range(15):
        adapter.update(accepted=False)
    assert adapter.sigma == 0.005
    for _ in range(15):
        adapter.update(accepted=False)
    assert adapter.sigma == 0.01


@pytest.mark.parametrize(
    ("dim", "success_tolerance", "steps_before_shrink"),
    [
        (10, 5, 24),  # failure_tolerance=max(10, 25)=25
        (4, 1, 9),  # failure_tolerance=max(10, 5)=10
    ],
)
def test_failure_tolerance_behavior(dim, success_tolerance, steps_before_shrink):
    adapter = StepSizeAdapter(sigma_0=0.1, dim=dim, success_tolerance=success_tolerance)
    for _ in range(steps_before_shrink):
        adapter.update(accepted=False)
    assert adapter.sigma == 0.1
    adapter.update(accepted=False)
    assert adapter.sigma == 0.05


def test_success_resets_failure_count():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=100)
    for _ in range(8):
        adapter.update(accepted=False)  # 8 failures
    adapter.update(accepted=True)  # resets failure count
    for _ in range(9):
        adapter.update(accepted=False)  # 9 more failures (not 10 total)
    assert adapter.sigma == 0.1  # no shrink yet


def test_clear_failure_streak_preserves_sigma():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=100)
    for _ in range(8):
        adapter.update(accepted=False)
    adapter.clear_failure_streak()
    for _ in range(14):
        adapter.update(accepted=False)
    assert adapter.sigma == 0.1


def test_failure_resets_success_count():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(accepted=True)  # success 1
    adapter.update(accepted=True)  # success 2
    adapter.update(accepted=False)  # failure resets success count
    adapter.update(accepted=True)  # success 1 again
    adapter.update(accepted=True)  # success 2
    assert adapter.sigma == 0.1  # no expand yet


def test_sigma_init_clamped_to_bounds():
    adapter = StepSizeAdapter(sigma_0=100.0, dim=10, sigma_max=0.5, sigma_min=0.01)
    assert adapter.sigma_init == 0.5
    assert adapter.sigma == 0.5


def test_restart_resets_sigma_and_counters():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    assert adapter.sigma == 0.2
    adapter.restart()
    assert adapter.sigma == adapter.sigma_init == 0.1
    adapter.update(accepted=True)
    adapter.update(accepted=True)
    assert adapter.sigma == 0.1
