from optimizer.step_size_adapter import StepSizeAdapter


def test_initial_sigma():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10)
    assert adapter.sigma == 0.1


def test_first_update_is_improvement():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10)
    assert adapter.update(1.0) is True


def test_no_improvement():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10)
    adapter.update(10.0)
    assert adapter.update(5.0) is False


def test_expand_after_success_tolerance():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(1.0)
    adapter.update(2.0)
    adapter.update(3.0)  # 3rd consecutive success -> expand
    assert adapter.sigma > 0.1


def test_expand_doubles_sigma():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(1.0)
    adapter.update(2.0)
    adapter.update(3.0)
    assert adapter.sigma == 0.2


def test_expand_capped_at_sigma_max():
    adapter = StepSizeAdapter(sigma_0=0.15, dim=10, sigma_max=0.2, success_tolerance=3)
    adapter.update(1.0)
    adapter.update(2.0)
    adapter.update(3.0)
    assert adapter.sigma == 0.2  # capped, not 0.3


def test_shrink_after_failure_tolerance():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=2)
    # failure_tolerance = ceil(2) = 2
    adapter.update(10.0)  # first, accepted
    adapter.update(5.0)  # fail 1
    adapter.update(3.0)  # fail 2 -> shrink
    assert adapter.sigma == 0.05


def test_shrink_floored_at_sigma_min():
    adapter = StepSizeAdapter(sigma_0=1e-5, dim=1, sigma_min=1e-5)
    adapter.update(10.0)
    adapter.update(5.0)  # fail 1 -> shrink (dim=1, tol=1)
    assert adapter.sigma == 1e-5


def test_failure_tolerance_is_ceil_dim():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=5)
    adapter.update(10.0)
    # 4 failures should not shrink (tolerance = 5)
    for _ in range(4):
        adapter.update(0.0)
    assert adapter.sigma == 0.1
    # 5th failure triggers shrink
    adapter.update(0.0)
    assert adapter.sigma == 0.05


def test_success_resets_failure_count():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=3)
    adapter.update(10.0)
    adapter.update(5.0)  # fail 1
    adapter.update(3.0)  # fail 2
    adapter.update(11.0)  # success resets failure count
    adapter.update(1.0)  # fail 1 (not 3)
    adapter.update(0.0)  # fail 2
    assert adapter.sigma == 0.1  # no shrink yet


def test_failure_resets_success_count():
    adapter = StepSizeAdapter(sigma_0=0.1, dim=10, success_tolerance=3)
    adapter.update(1.0)  # success 1
    adapter.update(2.0)  # success 2
    adapter.update(0.0)  # failure resets success count
    adapter.update(3.0)  # success 1 again
    adapter.update(4.0)  # success 2
    assert adapter.sigma == 0.1  # no expand yet
