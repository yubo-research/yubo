import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

from model.enn_fit_t import enn_fit
from model.enn_gp_t import EpistemicNearestNeighborsGP
from model.enn_likelihood_t import subsample_loglik
from model.enn_t import EpistemicNearestNeighborsT


def test_enngp_matches_ennt_posterior_cpu():
    g = torch.Generator(device="cpu").manual_seed(11)
    n = 40
    d = 4
    m = 1
    train_X = torch.rand((n, d), generator=g)
    train_Y = torch.randn((n, m), generator=g)
    train_Yvar = torch.zeros_like(train_Y)
    Xq = torch.rand((13, d), generator=g)
    base = EpistemicNearestNeighborsT(k=3)
    base.add(train_X, train_Y, train_Yvar)
    mvn_base = base.posterior(Xq)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    model.set_k(3)
    model.train()
    mvn_gp = model.forward(Xq)
    torch.testing.assert_close(mvn_gp.mean, mvn_base.mu.squeeze(-1), atol=1e-6, rtol=0)
    torch.testing.assert_close(mvn_gp.variance.sqrt(), mvn_base.se.squeeze(-1), atol=1e-6, rtol=0)


def test_enngp_fractional_k_blend_cpu():
    g = torch.Generator(device="cpu").manual_seed(29)
    n = 30
    d = 3
    m = 1
    train_X = torch.rand((n, d), generator=g)
    train_Y = torch.randn((n, m), generator=g)
    train_Yvar = torch.zeros_like(train_Y)
    Xq = torch.rand((10, d), generator=g)
    base = EpistemicNearestNeighborsT(k=1)
    base.add(train_X, train_Y, train_Yvar)
    k_low = 4
    k_high = 5
    mvn_low = base.posterior(Xq, k=k_low)
    base.posterior(Xq, k=k_high)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    # Note: k is now an integer, so we can't interpolate between k_low and k_high
    # Just use k_low for this test - it should match mvn_low exactly
    model.set_k(k_low)
    model.train()
    mvn_gp = model.forward(Xq)
    torch.testing.assert_close(mvn_gp.mean, mvn_low.mu.squeeze(-1), atol=1e-6, rtol=0)
    torch.testing.assert_close(mvn_gp.variance.sqrt(), mvn_low.se.squeeze(-1), atol=1e-6, rtol=0)


def test_enngp_fit_gpytorch_mll_regression():
    g = torch.Generator(device="cpu").manual_seed(123)
    n = 30
    d = 3
    m = 1
    train_X = torch.rand((n, d), generator=g)
    train_Y = torch.randn((n, m), generator=g)
    train_Yvar = torch.full_like(train_Y, 0.05)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def test_enngp_small_dataset_fit_gpytorch_mll_no_nan_prior():
    g = torch.Generator(device="cpu").manual_seed(0)
    dtype = torch.double
    device = torch.device("cpu")
    num_train = 3
    num_left = num_train // 2
    num_right = num_train - num_left
    x_left = 0.4 * torch.rand(num_left, 1, dtype=dtype, device=device, generator=g)
    x_right = 0.6 + 0.4 * torch.rand(num_right, 1, dtype=dtype, device=device, generator=g)
    x = torch.cat([x_left, x_right], dim=0)
    perm = torch.randperm(num_train, generator=g, device=device)
    x = x[perm]
    noise = 0.0
    y = -((x - 0.3) ** 2) + noise * torch.randn_like(x)
    y_var = torch.full_like(y, noise**2)
    model = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def test_enngp_fit_gpytorch_mll_n1():
    g = torch.Generator(device="cpu").manual_seed(5)
    dtype = torch.double
    device = torch.device("cpu")
    n = 1
    d = 2
    m = 1
    x = torch.rand((n, d), generator=g, dtype=dtype, device=device)
    y = torch.randn((n, m), generator=g, dtype=dtype, device=device)
    y_var = torch.full_like(y, 0.01)
    model = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def test_enngp_n0_forward_prior_like():
    g = torch.Generator(device="cpu").manual_seed(17)
    dtype = torch.double
    device = torch.device("cpu")
    n = 0
    d = 3
    m = 1
    train_X = torch.empty((n, d), dtype=dtype, device=device)
    train_Y = torch.empty((n, m), dtype=dtype, device=device)
    train_Yvar = torch.empty((n, m), dtype=dtype, device=device)
    Xq = torch.rand((5, d), generator=g, dtype=dtype, device=device)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    model.eval()
    with torch.no_grad():
        mvn = model.forward(Xq)
    assert mvn.mean.shape == (Xq.shape[0],)
    assert mvn.variance.shape == (Xq.shape[0],)
    torch.testing.assert_close(mvn.mean, torch.zeros_like(mvn.mean), atol=1e-6, rtol=0)
    torch.testing.assert_close(mvn.variance.sqrt(), torch.ones_like(mvn.mean), atol=1e-6, rtol=0)


def test_enngp_n0_fit_gpytorch_mll_like_notebook():
    dtype = torch.double
    device = torch.device("cpu")
    num_train = 0
    d = 3
    m = 1
    x = torch.empty((num_train, d), dtype=dtype, device=device)
    y = torch.empty((num_train, m), dtype=dtype, device=device)
    y_var = torch.empty((num_train, m), dtype=dtype, device=device)
    model = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def test_enngp_n0_enn_fit():
    dtype = torch.double
    device = torch.device("cpu")
    num_train = 0
    d = 3
    m = 1
    x = torch.empty((num_train, d), dtype=dtype, device=device)
    y = torch.empty((num_train, m), dtype=dtype, device=device)
    y_var = torch.empty((num_train, m), dtype=dtype, device=device)
    model = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    result = enn_fit(model)
    assert "k" in result
    assert "var_scale" in result
    assert result["k"] >= 1.0
    assert result["var_scale"] > 0.0


def test_enngp_fit_gpytorch_mll_n2():
    g = torch.Generator(device="cpu").manual_seed(9)
    dtype = torch.double
    device = torch.device("cpu")
    n = 2
    d = 2
    m = 1
    x = torch.rand((n, d), generator=g, dtype=dtype, device=device)
    y = torch.randn((n, m), generator=g, dtype=dtype, device=device)
    y_var = torch.full_like(y, 0.01)
    model = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)


def test_enn_fit_matches_bruteforce_grid_search():
    g = torch.Generator(device="cpu").manual_seed(11)
    dtype = torch.double
    device = torch.device("cpu")
    n = 20
    d = 2
    m = 1
    train_X = torch.rand((n, d), generator=g, dtype=dtype, device=device)
    train_Y = torch.randn((n, m), generator=g, dtype=dtype, device=device)
    train_Yvar = torch.full_like(train_Y, 0.05)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    k_values = torch.tensor([3, 10, 30, 100], dtype=dtype, device=device)
    var_scale_values = torch.logspace(-2, 4, 10, dtype=dtype, device=device)
    torch.manual_seed(42)
    result = enn_fit(model, k_values=k_values, var_scale_values=var_scale_values)
    tuned = model.tuned_hyperparams()
    assert result["k"] == tuned["k"]
    assert abs(result["var_scale"] - tuned["var_scale"]) < 1e-6
    model_ref = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    train_X_ref = model_ref.train_inputs[0]
    train_Y_ref = model_ref.train_targets
    max_k = min(100, max(1, len(model_ref._enn)))
    k_list = [int(v) for v in k_values.tolist() if 3 <= int(v) <= max_k]
    if not k_list:
        k_list = [1]
    var_scale_list = var_scale_values.tolist()
    if train_Y_ref.ndim == 2 and train_Y_ref.shape[-1] == 1:
        y_ref = train_Y_ref.squeeze(-1)
    else:
        y_ref = train_Y_ref
    if train_Yvar.ndim == 2 and train_Yvar.shape[-1] == 1:
        y_var_ref = train_Yvar.squeeze(-1)
    else:
        y_var_ref = train_Yvar
    P = 10
    num_iterations = 2
    torch.manual_seed(42)
    for _ in range(num_iterations):
        median_var_scale = torch.tensor(var_scale_list, dtype=dtype, device=device).median().item()
        best_k = None
        best_k_mll = None
        with torch.no_grad():
            model_ref.set_var_scale(median_var_scale)
            for k in k_list:
                model_ref.set_k(k)
                value = subsample_loglik(model_ref, train_X_ref, y_ref, y_var_ref, P=P).item()
                if best_k_mll is None or value > best_k_mll:
                    best_k_mll = value
                    best_k = k
        assert best_k is not None
        best_var_scale = None
        best_var_scale_mll = None
        with torch.no_grad():
            model_ref.set_k(best_k)
            for var_scale in var_scale_list:
                model_ref.set_var_scale(var_scale)
                value = subsample_loglik(model_ref, train_X_ref, y_ref, y_var_ref, P=P).item()
                if best_var_scale_mll is None or value > best_var_scale_mll:
                    best_var_scale_mll = value
                    best_var_scale = var_scale
        assert best_var_scale is not None

        with torch.no_grad():
            model_ref.set_var_scale(best_var_scale)
            model_ref.set_k(best_k)
    assert float(best_k) == result["k"]
    assert abs(best_var_scale - result["var_scale"]) < 1e-6


def test_enn_gp_demo_num_train_1000():
    torch.manual_seed(42)
    dtype = torch.double
    device = torch.device("cpu")
    num_train = 1000
    noise = 0.1
    num_left = num_train // 2
    num_right = num_train - num_left
    x_left = 0.4 * torch.rand(num_left, 1, dtype=dtype, device=device)
    x_right = 0.6 + 0.4 * torch.rand(num_right, 1, dtype=dtype, device=device)
    x = torch.cat([x_left, x_right], dim=0)
    perm = torch.randperm(num_train, device=device)
    x = x[perm]
    y = -((x - 0.3) ** 2) + noise * torch.randn_like(x)
    y_var = torch.full_like(y, noise**2)

    model = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    enn_fit(model)
    tuned = model.tuned_hyperparams()
    assert "k" in tuned
    assert "var_scale" in tuned
    assert tuned["k"] >= 1.0
    assert tuned["var_scale"] > 0.0

    xs = torch.linspace(0.0, 1.0, 30, dtype=dtype, device=device).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        mvn = model.forward(xs)
        mu = mvn.mean.squeeze(-1)
        se = mvn.variance.sqrt().squeeze(-1)
    assert mu.shape == (30,)
    assert se.shape == (30,)
    assert torch.all(se > 0.0)

    model2 = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    enn_fit(model2)
    tuned2 = model2.tuned_hyperparams()
    assert "k" in tuned2
    assert "var_scale" in tuned2
    assert tuned2["k"] >= 1.0
    assert tuned2["var_scale"] > 0.0

    model2.eval()
    with torch.no_grad():
        mvn2 = model2.forward(xs)
        mu2 = mvn2.mean.squeeze(-1)
        se2 = mvn2.variance.sqrt().squeeze(-1)
    assert mu2.shape == (30,)
    assert se2.shape == (30,)
    assert torch.all(se2 > 0.0)


def test_subsample_loglik_basic():
    g = torch.Generator(device="cpu").manual_seed(100)
    n = 20
    d = 3
    train_X = torch.rand((n, d), generator=g, dtype=torch.float64)
    train_Y = torch.randn((n, 1), generator=g, dtype=torch.float64)
    train_Yvar = torch.zeros_like(train_Y)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    y = train_Y.squeeze(-1)
    y_var = train_Yvar.squeeze(-1)
    loglik = subsample_loglik(model, train_X, y, y_var, P=5)
    assert isinstance(loglik, torch.Tensor)
    assert loglik.ndim == 0
    assert torch.isfinite(loglik)


def test_subsample_loglik_different_P():
    g = torch.Generator(device="cpu").manual_seed(101)
    n = 30
    d = 2
    train_X = torch.rand((n, d), generator=g, dtype=torch.float64)
    train_Y = torch.randn((n, 1), generator=g, dtype=torch.float64)
    train_Yvar = torch.zeros_like(train_Y)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    y = train_Y.squeeze(-1)
    y_var = train_Yvar.squeeze(-1)
    loglik_P5 = subsample_loglik(model, train_X, y, y_var, P=5)
    loglik_P10 = subsample_loglik(model, train_X, y, y_var, P=10)
    assert isinstance(loglik_P5, torch.Tensor)
    assert isinstance(loglik_P10, torch.Tensor)
    assert torch.isfinite(loglik_P5)
    assert torch.isfinite(loglik_P10)


def test_subsample_loglik_empty_data():
    n = 0
    d = 2
    train_X = torch.empty((n, d), dtype=torch.float64)
    train_Y = torch.empty((n, 1), dtype=torch.float64)
    train_Yvar = torch.empty((n, 1), dtype=torch.float64)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    y = train_Y.squeeze(-1) if n > 0 else torch.empty(0, dtype=torch.float64)
    y_var = train_Yvar.squeeze(-1) if n > 0 else torch.empty(0, dtype=torch.float64)
    loglik = subsample_loglik(model, train_X, y, y_var, P=5)
    assert isinstance(loglik, torch.Tensor)
    assert loglik.item() == 0.0


def test_enn_fit_n_equals_one():
    """Test that enn_fit works when there is only 1 training point."""
    torch.manual_seed(42)
    dtype = torch.float64
    device = torch.device("cpu")

    train_X = torch.rand(1, 2, dtype=dtype, device=device)
    train_Y = torch.randn(1, 1, dtype=dtype, device=device)
    train_Yvar = torch.zeros(1, 1, dtype=dtype, device=device)

    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)

    result = enn_fit(model, P=10)

    assert "k" in result
    assert "var_scale" in result
    assert result["k"] == 1
    assert result["var_scale"] > 0


def test_plot_enn_gp_demo_n_equals_one():
    """Test the exact notebook code from plot_enn_gp_demo when num_train=1."""
    torch.manual_seed(42)
    dtype = torch.double
    device = torch.device("cpu")
    noise = 0.1
    num_train = 1
    P = 10

    num_left = num_train // 2
    num_right = num_train - num_left
    x_left = 0.4 * torch.rand(num_left, 1, dtype=dtype, device=device)
    x_right = 0.6 + 0.4 * torch.rand(num_right, 1, dtype=dtype, device=device)
    x = torch.cat([x_left, x_right], dim=0)
    perm = torch.randperm(num_train, device=device)
    x = x[perm]
    y = torch.sin(8 * torch.pi * x) + noise * torch.randn_like(x)
    y_var = torch.full_like(y, noise**2)
    ss = y.std()
    y = (y - y.mean()) / ss
    y_var = y_var / ss**2

    model = EpistemicNearestNeighborsGP(train_X=x, train_Y=y, train_Yvar=y_var)
    result = enn_fit(model, P=P)

    assert "k" in result
    assert "var_scale" in result
    assert result["k"] == 1
    assert result["var_scale"] > 0

    xs = torch.linspace(0.0, 1.0, 30, dtype=dtype, device=device).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        mvn = model.posterior(xs, observation_noise=True)
        mu = mvn.mean.squeeze(-1)
        se = mvn.variance.sqrt().squeeze(-1)

    assert mu.shape == (30,)
    assert se.shape == (30,)
    assert torch.all(torch.isfinite(mu))
    assert torch.all(torch.isfinite(se))
    assert torch.all(se > 0)


def test_enn_fit_with_subsample_loglik():
    g = torch.Generator(device="cpu").manual_seed(106)
    n = 30
    d = 2
    train_X = torch.rand((n, d), generator=g, dtype=torch.float64)
    train_Y = torch.randn((n, 1), generator=g, dtype=torch.float64)
    train_Yvar = torch.zeros_like(train_Y)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    result = enn_fit(model, P=5)
    assert "k" in result
    assert "var_scale" in result
    assert isinstance(result["k"], float)
    assert isinstance(result["var_scale"], float)
    assert 1.0 <= result["k"] <= 100.0
    assert result["var_scale"] > 0.0


def test_k_constraint_max_100():
    g = torch.Generator(device="cpu").manual_seed(107)
    n = 200
    d = 2
    train_X = torch.rand((n, d), generator=g, dtype=torch.float64)
    train_Y = torch.randn((n, 1), generator=g, dtype=torch.float64)
    train_Yvar = torch.zeros_like(train_Y)
    model = EpistemicNearestNeighborsGP(train_X, train_Y, train_Yvar)
    # set_k should clamp k to max(1, len(model._enn)) - no longer clamped to 100
    max_k = max(1, len(model._enn))
    # Test that k < max_k works (no clamping needed)
    model.set_k(150)
    hyperparams = model.tuned_hyperparams()
    assert hyperparams["k"] == 150.0
    # Test that k > max_k gets clamped to max_k
    model.set_k(300)
    hyperparams2 = model.tuned_hyperparams()
    assert hyperparams2["k"] == max_k
    # Test that k < max_k works
    model.set_k(50)
    hyperparams3 = model.tuned_hyperparams()
    assert hyperparams3["k"] == 50.0
