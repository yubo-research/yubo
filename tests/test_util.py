def gp_parabola(*, num_samples=3, num_dim=2):
    import torch

    from acq.fit_gp import fit_gp_XY

    X_0 = 0.3 * torch.ones(size=(num_dim,))
    X = torch.rand(size=torch.Size([num_samples, num_dim]), dtype=torch.double)

    Y = -((X - X_0) ** 2).sum(dim=1)
    Y = Y + 0.25 * torch.randn(size=Y.shape)
    Y = Y[:, None]

    return fit_gp_XY(X, Y), X_0


def make_simple_gp():
    import torch
    from botorch.models import SingleTaskGP

    X = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    Y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    model = SingleTaskGP(X, Y)
    model.eval()
    return model


def make_acq_dpp_from_simple_gp(*, num_X_samples: int = 16):
    from acq.acq_dpp import AcqDPP

    acq = AcqDPP(make_simple_gp(), num_X_samples=num_X_samples)
    assert acq is not None
    assert acq._num_dim == 2
    return acq


def assert_telemetry_format_fit_select_values():
    from common.telemetry import Telemetry

    t = Telemetry()
    t.set_dt_fit(1.234)
    t.set_dt_select(5.6789)
    assert t.format() == "fit_dt = 1.234 select_dt = 5.679"


def assert_telemetry_format_all_na():
    from common.telemetry import Telemetry

    t = Telemetry()
    assert t.format() == "fit_dt = N/A select_dt = N/A"


def assert_telemetry_reset_clears_dt_fields():
    from common.telemetry import Telemetry

    t = Telemetry()
    t.set_dt_fit(1.0)
    t.set_dt_select(2.0)
    t.reset()
    assert t._dt_fit is None
    assert t._dt_select is None
