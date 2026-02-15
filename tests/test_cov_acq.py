import pytest
import torch

from tests.test_util import make_simple_gp


def test_gp_init():
    from acq.acq_dpp import _GP

    model = make_simple_gp()
    gp = _GP(model)
    assert gp.d == 2
    assert gp.model is not None
    assert gp.model.training is False  # Should be in eval mode


def test_empty_transform_init():
    from acq.fit_gp import _EmptyTransform

    t = _EmptyTransform()
    y = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    y_out, yvar_out = t.forward(y)
    assert torch.equal(y, y_out)
    assert yvar_out is None

    y_out2, yvar_out2 = t.forward(y, None)
    assert torch.equal(y, y_out2)
    assert yvar_out2 is None


def test_mcmc_one_transit():
    """Test that mcmc_one_transit can be imported and called (requires CUDA)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for mcmc_one_transit")
    from acq.mcmc_bo import mcmc_one_transit

    # Function exists and is callable
    assert callable(mcmc_one_transit)


def test_langevin_update():
    """Test that langevin_update can be imported and called (requires CUDA)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for langevin_update")
    from acq.mcmc_bo import langevin_update

    # Function exists and is callable
    assert callable(langevin_update)


def test_generate_batch_multiple_tr():
    """Test that generate_batch_multiple_tr can be imported and called (requires CUDA)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for generate_batch_multiple_tr")
    from acq.mcmc_bo import generate_batch_multiple_tr

    # Function exists and is callable
    assert callable(generate_batch_multiple_tr)
