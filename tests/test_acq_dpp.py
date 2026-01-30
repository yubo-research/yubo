from tests.test_util import make_simple_gp as _make_simple_gp


def test_gp_init():
    from acq.acq_dpp import _GP

    model = _make_simple_gp()
    gp = _GP(model)
    assert gp.d == 2
    assert gp.model is not None


def test_acq_dpp_init():
    from acq.acq_dpp import AcqDPP

    model = _make_simple_gp()
    acq = AcqDPP(model, num_X_samples=16)
    assert acq is not None
    assert acq._num_dim == 2
