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
    adapter = StepSizeAdapter(sigma_0=1e-5, dim=100, sigma_min=1e-5)
    for _ in range(15):
        adapter.update(accepted=False)
    assert adapter.sigma == 1e-5


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


def test_failure_resets_success_count():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(accepted=True)  # success 1
    adapter.update(accepted=True)  # success 2
    adapter.update(accepted=False)  # failure resets success count
    adapter.update(accepted=True)  # success 1 again
    adapter.update(accepted=True)  # success 2
    assert adapter.sigma == 0.1  # no expand yet
