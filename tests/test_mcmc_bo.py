import numpy as np
import torch


def test_turbo_state_init():
    from acq.mcmc_bo import TurboState

    state = TurboState(dim=5, batch_size=2)
    assert state.dim == 5
    assert state.batch_size == 2
    assert state.length == 0.8
    assert np.isfinite(state.failure_tolerance)


def test_turbo_state_update_success():
    from acq.mcmc_bo import TurboState

    state = TurboState(dim=5, batch_size=2)
    state.best_value = 1.0
    # Provide better Y_next to trigger success
    state.update_state(np.array([1.5, 1.6]))
    assert state.success_counter == 1
    assert state.failure_counter == 0


def test_turbo_state_update_failure():
    from acq.mcmc_bo import TurboState

    state = TurboState(dim=5, batch_size=2)
    state.best_value = 2.0
    # Provide worse Y_next to trigger failure
    state.update_state(np.array([1.0, 1.1]))
    assert state.failure_counter == 1
    assert state.success_counter == 0


def test_cdf():
    from acq.mcmc_bo import cdf

    result = cdf(torch.tensor(0.0))
    assert torch.isclose(result, torch.tensor(0.5), atol=1e-6)


def test_get_point_in_tr():
    from acq.mcmc_bo import get_point_in_tr

    x = torch.tensor([0.5, 0.5])
    assert get_point_in_tr(x, 0, 1)

    x_outside = torch.tensor([1.5, 0.5])
    assert not get_point_in_tr(x_outside, 0, 1)
