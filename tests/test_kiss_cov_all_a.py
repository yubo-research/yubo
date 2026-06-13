from __future__ import annotations

import contextlib
from types import SimpleNamespace

_MODAL_UHD_REMOTE_ARGS = {}


def _mock_modal_uhd_app_init(self, name=""):
    pass


def _mock_modal_uhd_remote(*args, **_kwargs):
    _MODAL_UHD_REMOTE_ARGS["args"] = args
    return "result"


def _mock_modal_uhd_attach_remote(fn):
    fn.remote = _mock_modal_uhd_remote
    return fn


def _mock_modal_uhd_app_function(self, **kwargs):
    return _mock_modal_uhd_attach_remote


def _mock_modal_uhd_app_run(self):
    return contextlib.nullcontext()


def test_cov_cdf():
    import torch

    from acq.mcmc_bo import cdf

    result = cdf(torch.tensor(0.0))
    assert 0.49 < float(result) < 0.51


def test_cov_estimate():
    import torch

    from acq.fit_gp import estimate

    def _gp_posterior(self, X):
        return SimpleNamespace(mean=torch.zeros(X.shape[0], 1))

    _GP = type("_GP", (), {"posterior": _gp_posterior})
    y = estimate(_GP(), torch.zeros(3, 2))
    assert y.shape == (3,)


def test_cov_closure_warping():
    import torch
    from torch import nn

    from acq.fit_gp import get_closure

    def _model_init(self):
        nn.Module.__init__(self)
        self.p = nn.Parameter(torch.tensor(1.0))
        self.train_inputs = (torch.tensor([[0.0]]),)
        self.train_targets = torch.tensor([0.0])

    def _model_forward(self, *args):
        return SimpleNamespace(mean=torch.zeros(1), variance=torch.ones(1))

    def _model_transform_inputs(self, X):
        return X

    _Model = type(
        "_Model",
        (nn.Module,),
        {
            "__init__": _model_init,
            "forward": _model_forward,
            "transform_inputs": _model_transform_inputs,
        },
    )

    def _mll_init(self):
        nn.Module.__init__(self)
        self.model = _Model()
        self.q = nn.Parameter(torch.tensor(1.0))

    def _mll_forward(self, model_output, targets, *args, **kwargs):
        return torch.tensor(1.0, requires_grad=True)

    _MLL = type("_MLL", (nn.Module,), {"__init__": _mll_init, "forward": _mll_forward})

    closure = get_closure(_MLL(), lambda y: y)
    assert closure is not None


def test_cov_sal_the_prior_setter():
    from gpytorch.priors import NormalPrior

    from acq.sal_transform import SALTransform

    t = SALTransform(a_prior=NormalPrior(0, 1))
    assert t is not None


def test_cov_y_the_prior_setter():
    from gpytorch.priors import NormalPrior

    from acq.y_transform import YTransform

    t = YTransform(a_prior=NormalPrior(0, 1))
    assert t is not None


def test_cov_optimizers_in(tmp_path):
    from analysis.data_sets import optimizers_in

    exp = tmp_path / "results" / "exp" / "prob"
    opt = exp / "opt_a"
    opt.mkdir(parents=True)
    result = optimizers_in(str(tmp_path / "results"), "exp", "prob")
    assert result == ["opt_a"]


def test_cov_safe_float():
    import numpy as np

    from analysis.data_sets import _ensure_float_traces

    arr = np.array(["bad", "1.0"], dtype=object)
    result = _ensure_float_traces(arr)
    assert np.isnan(result[0])
    assert result[1] == 1.0


def test_cov_report_bad_init_traces(tmp_path):
    import json

    from analysis.data_locator import DataLocator
    from analysis.data_sets import load_multiple_traces

    results = tmp_path / "results"
    prob_dir = results / "prob1"
    opt_dir = prob_dir / "opt1"
    traces_dir = opt_dir / "traces"
    traces_dir.mkdir(parents=True)
    (opt_dir / "config.json").write_text(json.dumps({"opt_name": "opt1", "env_tag": "f:sphere-2d"}))
    (traces_dir / "00000.jsonl").write_text('{"rreturn": 1.0}\n{"rreturn": 2.0}\n')

    dl = DataLocator(results_path=str(results), exp_dir="", opt_names=["opt1"])
    out = load_multiple_traces(dl)
    assert out is not None


def test_cov_plotting_types(tmp_path):
    import numpy as np

    from analysis.data_locator import DataLocator
    from analysis.plotting_trace_types import (
        PlotRLComparisonResult,
        PlotRLFinalComparisonResult,
        RLTracesWithCumDtProp,
    )
    from analysis.plotting_types import (
        PlotResultsCombinedResult,
        PlotResultsResult,
        PlotRLExperimentResult,
        PlotRLExperimentVsTimeResult,
    )

    dl = DataLocator(results_path=str(tmp_path), exp_dir="", opt_names=[])
    t = RLTracesWithCumDtProp(data_locator=dl, traces=np.zeros((2, 3)), cum_dt_prop=None)
    assert t.traces.shape == (2, 3)
    r = PlotRLComparisonResult(fig=None, axs=None, seq=t, batch=None)
    assert r.seq == t
    rf = PlotRLFinalComparisonResult(fig=None, axs=None, seq=t, batch=None)
    assert rf.seq == t
    pe = PlotRLExperimentResult(fig=None, ax=None, data_locator=dl, traces=np.array([1.0]))
    assert pe.traces is not None
    pet = PlotRLExperimentVsTimeResult(fig=None, ax=None, data_locator=dl, traces=np.array([1.0]), t=np.array([0.0]))
    assert pet.t is not None
    pr = PlotResultsResult(curves=(None, None), final=(None, None), seq_data=None, batch_data=None)
    assert pr.curves == (None, None)
    prc = PlotResultsCombinedResult(fig=None, axs=None, seq_data=None, batch_data=None)
    assert prc.fig is None


def test_cov_bat_worker_run_batch_run():
    from experiments.bat_optimal_init_figures import run, run_batch, worker

    assert worker("true") == 0
    run_batch(["true"], b_dry_run=True)
    run(["true"], max_parallel=1, b_dry_run=True)


def test_cov_batches_worker_run_batch_run(tmp_path):
    from experiments.batches_impl import run, run_batch, worker

    assert worker("true") == 0
    run_batch(
        [{"exp_dir": str(tmp_path), "opt_name": "dummy", "env_tag": "x"}],
        b_dry_run=True,
    )
    run(
        [{"exp_dir": str(tmp_path), "opt_name": "dummy", "env_tag": "x"}],
        max_parallel=1,
        b_dry_run=True,
    )


def test_cov_experiment_cli_local_main(monkeypatch, tmp_path):
    from click.testing import CliRunner

    from experiments.experiment import cli, local

    runner = CliRunner()

    toml = tmp_path / "exp.toml"
    toml.write_text(
        '[experiment]\nenv_tag = "f:sphere-2d"\npolicy_tag = "pure-function"\nopt_name = "random"\nnum_arms = 1\nnum_rounds = 1\nnum_reps = 1\nexp_dir = "'
        + str(tmp_path / "out")
        + '"\n'
    )

    monkeypatch.setattr("experiments.experiment_sampler.sampler", lambda cfg, distributor_fn: None)
    res = runner.invoke(cli, ["local", str(toml)])
    assert res.exit_code == 0

    _ = local


def test_cov_fit_mnist():
    from ops.fit_mnist import fit_mnist

    model = fit_mnist(num_epochs=1, batch_size=512, timeout_seconds=5)
    assert model is not None


def test_cov_calc_pstar_scales():
    from figures.pts.fig_pstar_scale import calc_pstar_scales

    assert callable(calc_pstar_scales.get_raw_f())
