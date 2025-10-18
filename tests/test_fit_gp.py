import pytest
import torch


@pytest.fixture
def simple_data():
    torch.manual_seed(42)
    num_samples = 10
    num_dim = 3
    X = torch.rand(size=(num_samples, num_dim), dtype=torch.double)
    Y = torch.sin(X.sum(dim=1, keepdim=True)) + 0.1 * torch.randn(num_samples, 1, dtype=torch.double)
    return X, Y


@pytest.fixture
def empty_data():
    X = torch.empty((0, 3), dtype=torch.double)
    Y = torch.empty((0, 1), dtype=torch.double)
    return X, Y


@pytest.fixture
def single_point_data():
    X = torch.tensor([[0.5, 0.3, 0.7]], dtype=torch.double)
    Y = torch.tensor([[1.0]], dtype=torch.double)
    return X, Y


def test_fit_gp_basic(simple_data):
    from acq.fit_gp import fit_gp_XY

    X, Y = simple_data
    gp = fit_gp_XY(X, Y)

    assert gp is not None
    posterior = gp.posterior(X)
    assert posterior.mean.shape == (len(X), 1)


def test_fit_gp_empty_data(empty_data):
    from acq.fit_gp import fit_gp_XY

    X, Y = empty_data
    gp = fit_gp_XY(X, Y)

    assert gp is not None
    test_X = torch.rand(5, 3, dtype=torch.double)
    posterior = gp.posterior(test_X)
    assert posterior.mean.shape == (5, 1)


def test_fit_gp_single_point(single_point_data):
    from acq.fit_gp import fit_gp_XY

    X, Y = single_point_data
    gp = fit_gp_XY(X, Y)

    assert gp is not None
    posterior = gp.posterior(X)
    assert posterior.mean.shape == (1, 1)


@pytest.mark.parametrize(
    "model_spec",
    [
        None,
        "gp",
        "rff128",
        "rff256",
        "dumbo",
        "rdumbo",
        "sparse",
    ],
)
def test_fit_gp_model_types(simple_data, model_spec):
    from acq.fit_gp import fit_gp_XY

    X, Y = simple_data
    gp = fit_gp_XY(X, Y, model_spec=model_spec)

    assert gp is not None
    posterior = gp.posterior(X)
    assert posterior.mean.shape == (len(X), 1)


@pytest.mark.parametrize(
    "model_spec",
    [
        "gp+wi",
        "gp+wos",
        "gp+woy",
    ],
)
def test_fit_gp_warping(simple_data, model_spec):
    from acq.fit_gp import fit_gp_XY

    X, Y = simple_data
    gp = fit_gp_XY(X, Y, model_spec=model_spec)

    assert gp is not None
    posterior = gp.posterior(X)
    assert posterior.mean.shape == (len(X), 1)

    if "+wi" in model_spec:
        assert hasattr(gp, "input_transform")
        assert gp.input_transform is not None
    if "+wos" in model_spec or "+woy" in model_spec:
        assert hasattr(gp, "outcome_warp")
        assert gp.outcome_warp is not None


def test_fit_gp_predictions_reasonable(simple_data):
    from acq.fit_gp import fit_gp_XY

    X, Y = simple_data
    gp = fit_gp_XY(X, Y)

    posterior = gp.posterior(X)
    mean = posterior.mean
    stddev = posterior.stddev

    assert mean.shape == Y.shape
    assert torch.all(torch.isfinite(mean))
    assert torch.all(torch.isfinite(stddev))
    assert torch.all(stddev > 0)


def test_fit_gp_different_dimensions():
    from acq.fit_gp import fit_gp_XY

    torch.manual_seed(123)

    for num_dim in [1, 2, 5, 10]:
        X = torch.rand(size=(15, num_dim), dtype=torch.double)
        Y = torch.rand(size=(15, 1), dtype=torch.double)

        gp = fit_gp_XY(X, Y)
        assert gp is not None

        test_X = torch.rand(3, num_dim, dtype=torch.double)
        posterior = gp.posterior(test_X)
        assert posterior.mean.shape == (3, 1)


def test_fit_gp_eval_mode(simple_data):
    from acq.fit_gp import fit_gp_XY

    X, Y = simple_data
    gp = fit_gp_XY(X, Y)

    assert not gp.training


def test_fit_gp_device_placement(simple_data):
    from acq.fit_gp import fit_gp_XY

    X, Y = simple_data
    gp = fit_gp_XY(X, Y)

    for param in gp.parameters():
        assert param.device == X.device
        assert param.dtype == X.dtype


@pytest.mark.parametrize(
    "model_spec",
    [
        "dumbo",
        "rdumbo",
        "sparse",
    ],
)
def test_fit_gp_dumbo_empty(empty_data, model_spec):
    from acq.fit_gp import fit_gp_XY

    X, Y = empty_data
    gp = fit_gp_XY(X, Y, model_spec=model_spec)

    assert gp is not None
    test_X = torch.rand(5, 3, dtype=torch.double)
    posterior = gp.posterior(test_X)
    assert posterior.mean.shape == (5, 1)
