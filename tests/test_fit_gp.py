import numpy as np
import torch


def test_mk_x():
    from types import SimpleNamespace

    from acq.fit_gp import mk_x

    policy = SimpleNamespace(get_params=lambda: np.array([0.0, 0.0]))
    x = mk_x(policy)
    assert x.shape == (2,)


def test_mk_yx():
    from types import SimpleNamespace

    from acq.fit_gp import mk_yx

    traj = SimpleNamespace(rreturn=1.0)
    policy = SimpleNamespace(get_params=lambda: np.array([0.0, 0.0]))
    datum = SimpleNamespace(trajectory=traj, policy=policy)
    y, x = mk_yx(datum)
    assert y == 1.0
    assert x.shape == (2,)


def test_extract_X_Y():
    from types import SimpleNamespace

    from acq.fit_gp import extract_X_Y

    traj1 = SimpleNamespace(rreturn=1.0)
    traj2 = SimpleNamespace(rreturn=2.0)
    policy1 = SimpleNamespace(get_params=lambda: np.array([0.0, 0.0]))
    policy2 = SimpleNamespace(get_params=lambda: np.array([0.5, 0.5]))
    data = [
        SimpleNamespace(trajectory=traj1, policy=policy1),
        SimpleNamespace(trajectory=traj2, policy=policy2),
    ]
    Y, X = extract_X_Y(data, torch.float64, "cpu")
    assert Y.shape == (2, 1)
    assert X.shape == (2, 2)


def test_mk_policies():
    from types import SimpleNamespace

    from acq.fit_gp import mk_policies

    policy_0 = SimpleNamespace(
        clone=lambda: SimpleNamespace(set_params=lambda p: None),
    )
    X_cand = torch.tensor([[0.5, 0.5], [0.3, 0.3]], dtype=torch.float64)
    policies = mk_policies(policy_0, X_cand)
    assert len(policies) == 2


def test_fit_gp_XY():
    from acq.fit_gp import fit_gp_XY

    X = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    Y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    gp = fit_gp_XY(X, Y)
    assert gp is not None


def test_empty_transform_init():
    from acq.fit_gp import _EmptyTransform

    t = _EmptyTransform()
    assert t is not None


def test_empty_transform_forward():
    from acq.fit_gp import _EmptyTransform

    t = _EmptyTransform()
    Y = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
    Y_out, Yvar_out = t.forward(Y)
    assert torch.equal(Y, Y_out)
    assert Yvar_out is None


def test_get_closure():
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from acq.fit_gp import get_closure

    X = torch.tensor([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], dtype=torch.float64)
    Y = torch.tensor([[1.0], [2.0], [1.5]], dtype=torch.float64)
    model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    def identity_warp(y):
        return y

    closure = get_closure(mll, identity_warp)
    assert closure is not None
