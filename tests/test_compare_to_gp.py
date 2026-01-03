import pytest


def test_fit_result_smoke():
    from experiments.enn.compare_to_gp import FitResult

    r = FitResult(mean_loglik=0.0, tuned={})
    assert r.mean_loglik == 0.0
    assert r.tuned == {}


def test_torch_options_smoke():
    import numpy as np

    from experiments.enn.compare_to_gp import DimSweepSpec, GPFitOptions, GPVariabilitySpec, TorchOptions

    o = TorchOptions()
    assert o.device == "cpu"

    _ = np.random.default_rng(0)
    ds = DimSweepSpec(dims=[1, 3], seed=0, num_reps=2, num_samples_in=5, num_samples_out=6, sigma_noise=0.1)
    assert ds.dims == [1, 3]
    fit_opts = GPFitOptions(max_attempts=2, pick_best_of_all_attempts=True)
    assert fit_opts.max_attempts == 2
    gpv = GPVariabilitySpec(
        func_name="ackley",
        num_dim=3,
        seed_data=0,
        sigma_noise=0.1,
        num_samples_in=5,
        num_samples_out=6,
        num_fit_reps=2,
        use_loocv=False,
        gp_fit_options=fit_opts,
    )
    assert gpv.num_fit_reps == 2


def test_mean_loglik_pure_function_invalid_args():
    from experiments.enn.compare_to_gp import mean_loglik_pure_function

    with pytest.raises(ValueError):
        mean_loglik_pure_function("", seed=0, num_samples_in=5, num_samples_out=5, model_type="gp", sigma_noise=0.1)
    with pytest.raises(ValueError):
        mean_loglik_pure_function("ackley-2d", seed=0, num_samples_in=0, num_samples_out=5, model_type="gp", sigma_noise=0.1)
    with pytest.raises(ValueError):
        mean_loglik_pure_function("ackley-2d", seed=0, num_samples_in=5, num_samples_out=0, model_type="gp", sigma_noise=0.1)
    with pytest.raises(ValueError):
        mean_loglik_pure_function("ackley-2d", seed=0, num_samples_in=5, num_samples_out=5, model_type="gp", sigma_noise=-1.0)
    with pytest.raises(ValueError):
        mean_loglik_pure_function("ackley-2d", seed=0, num_samples_in=5, num_samples_out=5, model_type="nope", sigma_noise=0.1)


def test_mean_loglik_pure_function_gp_smoke(monkeypatch):
    import experiments.enn.compare_to_gp as c2g

    monkeypatch.setattr(c2g, "_fit_gpytorch_mll", lambda mll: None)

    out = c2g.mean_loglik_pure_function(
        "ackley-2d",
        seed=0,
        num_samples_in=8,
        num_samples_out=7,
        model_type="gp",
        sigma_noise=0.1,
    )
    assert isinstance(out.mean_loglik, float)
    assert out.tuned is not None
    assert out.mean_loglik == out.mean_loglik


def test_mean_loglik_pure_function_enn_smoke(monkeypatch):
    import experiments.enn.compare_to_gp as c2g

    def _fast_enn_fit(model):
        model.set_k(3)
        model.set_var_scale(1.0)
        return {"k": 3.0, "var_scale": 1.0}

    monkeypatch.setattr(c2g, "_enn_fit", _fast_enn_fit)

    out = c2g.mean_loglik_pure_function(
        "ackley-2d",
        seed=0,
        num_samples_in=8,
        num_samples_out=7,
        model_type="enn",
        sigma_noise=0.1,
    )
    assert isinstance(out.mean_loglik, float)
    assert out.tuned["k"] >= 1.0
    assert out.tuned["var_scale"] > 0.0
    assert out.tuned["y_var"] >= 0.0
    assert out.mean_loglik == out.mean_loglik


def test_sweep_ackley_dim_ll_gp_vs_enn_smoke(monkeypatch):
    import experiments.enn.compare_to_gp as c2g

    def _stub_mean_loglik_pure_function(pure_function_name: str, *, model_type: str, seed: int, **kwargs):
        d = int(pure_function_name.split("-")[1][:-1])
        base = float(d)
        ll = (base if model_type == "gp" else -base) + float(seed)
        return c2g.FitResult(mean_loglik=ll, tuned={})

    monkeypatch.setattr(c2g, "mean_loglik_pure_function", _stub_mean_loglik_pure_function)

    df = c2g.sweep_dim_ll_gp_vs_enn(
        "ackley",
        sigma_noise=0.1,
        dims=[1, 3, 10],
        seed=0,
        num_reps=2,
        num_samples_in=100,
        num_samples_out=100,
    )
    assert list(df.columns) == ["num_dim", "ll_gp_mean", "ll_gp_se", "ll_enn_mean", "ll_enn_se"]
    assert df["num_dim"].tolist() == [1, 3, 10]
    assert df["ll_gp_mean"].tolist() == [1.5, 3.5, 10.5]
    assert df["ll_enn_mean"].tolist() == [-0.5, -2.5, -9.5]
    assert df["ll_gp_se"].tolist() == [0.5, 0.5, 0.5]
    assert df["ll_enn_se"].tolist() == [0.5, 0.5, 0.5]

    df2 = c2g.sweep_ackley_dim_ll_gp_vs_enn(
        sigma_noise=0.1,
        dims=[1, 3, 10],
        seed=0,
        num_reps=2,
        num_samples_in=100,
        num_samples_out=100,
    )
    assert df2.equals(df)


def test_check_gp_fit_instability_smoke(monkeypatch):
    import experiments.enn.compare_to_gp as c2g

    monkeypatch.setattr(c2g, "_fit_gpytorch_mll_with_options", lambda mll, opts: None)

    df = c2g.check_gp_fit_instability(
        "ackley",
        num_dim=3,
        seed_data=0,
        sigma_noise=0.1,
        num_samples_in=10,
        num_samples_out=10,
        num_fit_reps=3,
    )
    assert list(df.columns) == ["fit_rep", "ll_gp", "gp_noise", "gp_lengthscale_mean", "gp_outputscale"]
    assert len(df) == 3
    assert df["fit_rep"].tolist() == [0.0, 1.0, 2.0]

    df_loocv = c2g.check_gp_fit_instability(
        "ackley",
        num_dim=3,
        seed_data=0,
        sigma_noise=0.1,
        num_samples_in=10,
        num_samples_out=10,
        num_fit_reps=2,
        use_loocv=True,
    )
    assert list(df_loocv.columns) == ["fit_rep", "ll_gp", "gp_noise", "gp_lengthscale_mean", "gp_outputscale"]
    assert len(df_loocv) == 2
