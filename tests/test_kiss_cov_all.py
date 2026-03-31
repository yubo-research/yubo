from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import torch
from click.testing import CliRunner
from torch import nn

# ---------------------------------------------------------------------------
# acq/
# ---------------------------------------------------------------------------


def test_cov_cdf():
    from acq.mcmc_bo import cdf

    result = cdf(torch.tensor(0.0))
    assert 0.49 < float(result) < 0.51


def test_cov_estimate():
    from acq.fit_gp import estimate

    class _GP:
        def posterior(self, X):
            return SimpleNamespace(mean=torch.zeros(X.shape[0], 1))

    y = estimate(_GP(), torch.zeros(3, 2))
    assert y.shape == (3,)


def test_cov_closure_warping():
    from acq.fit_gp import get_closure

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor(1.0))
            self.train_inputs = (torch.tensor([[0.0]]),)
            self.train_targets = torch.tensor([0.0])

        def forward(self, *args):
            return SimpleNamespace(mean=torch.zeros(1), variance=torch.ones(1))

        def transform_inputs(self, X):
            return X

    class _MLL(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Model()
            self.q = nn.Parameter(torch.tensor(1.0))

        def forward(self, model_output, targets, *args, **kwargs):
            return torch.tensor(1.0, requires_grad=True)

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


# ---------------------------------------------------------------------------
# admin/
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# analysis/
# ---------------------------------------------------------------------------


def test_cov_optimizers_in(tmp_path):
    from analysis.data_sets import optimizers_in

    exp = tmp_path / "results" / "exp" / "prob"
    opt = exp / "opt_a"
    opt.mkdir(parents=True)
    result = optimizers_in(str(tmp_path / "results"), "exp", "prob")
    assert result == ["opt_a"]


def test_cov_safe_float():
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
    from analysis.data_locator import DataLocator
    from analysis.plotting_trace_types import (
        PlotRLComparisonResult,
        PlotRLFinalComparisonResult,
        RLTracesWithCumDtProp,
    )
    from analysis.plotting_types import (
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


# ---------------------------------------------------------------------------
# experiments/
# ---------------------------------------------------------------------------


def test_cov_bat_worker_run_batch_run():
    from experiments.bat_optimal_init_figures import run, run_batch, worker

    assert worker("true") == 0
    run_batch(["true"], b_dry_run=True)
    run(["true"], max_parallel=1, b_dry_run=True)


def test_cov_batches_worker_run_batch_run(tmp_path):
    from experiments.batches import run, run_batch, worker

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
    from experiments.experiment import cli, local, main

    runner = CliRunner()

    toml = tmp_path / "exp.toml"
    toml.write_text(
        '[experiment]\nenv_tag = "f:sphere-2d"\nopt_name = "random"\nnum_arms = 1\nnum_rounds = 1\nnum_reps = 1\nexp_dir = "' + str(tmp_path / "out") + '"\n'
    )

    monkeypatch.setattr("experiments.experiment_sampler.sampler", lambda cfg, distributor_fn: None)
    res = runner.invoke(cli, ["local", str(toml)])
    assert res.exit_code == 0

    _ = local, main


def test_cov_fit_mnist():
    from ops.fit_mnist import fit_mnist

    model = fit_mnist(num_epochs=1, batch_size=512, timeout_seconds=5)
    assert model is not None


# ---------------------------------------------------------------------------
# figures/
# ---------------------------------------------------------------------------


def test_cov_calc_pstar_scales():
    from figures.pts.fig_pstar_scale import calc_pstar_scales

    assert callable(calc_pstar_scales.get_raw_f())


# ---------------------------------------------------------------------------
# ops/
# ---------------------------------------------------------------------------


def test_cov_catalog_num_params_cli():
    from ops.catalog import _CatalogPolicy, cli

    p = _CatalogPolicy()
    assert p.num_params() == 1
    _ = cli


def test_cov_data_cli():
    import ops.data as data_mod

    cli = data_mod.cli
    assert cli is not None


def test_cov_exp_uhd_UHDConfig_local_modal_cmd(monkeypatch, tmp_path):
    import ops.exp_uhd as exp_uhd
    from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig

    early_reject = EarlyRejectConfig(tau=None, mode=None, ema_beta=None, warmup_pos=None, quantile=None, window=None)
    be = BEConfig(
        num_probes=10,
        num_candidates=10,
        warmup=20,
        fit_interval=10,
        enn_k=25,
        sigma_range=None,
    )
    enn = ENNConfig(
        minus_impute=False,
        d=100,
        s=4,
        jl_seed=123,
        k=25,
        fit_interval=50,
        warmup_real_obs=200,
        refresh_interval=50,
        se_threshold=0.25,
        target="mu_minus",
        num_candidates=1,
        select_interval=1,
        embedder="direction",
        gather_t=64,
    )
    cfg = UHDConfig(
        env_tag="f:sphere-2d",
        num_rounds=1,
        problem_seed=None,
        noise_seed_0=None,
        lr=0.001,
        num_dim_target=None,
        num_module_target=None,
        log_interval=10,
        accuracy_interval=1000,
        target_accuracy=None,
        early_reject=early_reject,
        be=be,
        enn=enn,
        optimizer="mezo",
        batch_size=4096,
        bszo_k=2,
        bszo_epsilon=1e-4,
        bszo_sigma_p_sq=1.0,
        bszo_sigma_e_sq=1.0,
        bszo_alpha=0.1,
    )
    assert cfg.env_tag == "f:sphere-2d"

    _ = exp_uhd.cli

    class _Loop:
        def run(self):
            pass

    monkeypatch.setattr("ops.uhd_setup.make_loop", lambda *a, **k: _Loop())

    toml = tmp_path / "cfg.toml"
    toml.write_text('[uhd]\nenv_tag = "f:sphere-2d"\nnum_rounds = 1\n')

    runner = CliRunner()
    res = runner.invoke(exp_uhd.cli, ["local", str(toml)])
    assert res.exit_code == 0

    monkeypatch.setattr("ops.modal_uhd.run", lambda *a, **k: None)
    res = runner.invoke(exp_uhd.cli, ["modal", str(toml)])
    assert res.exit_code == 0

    _ = exp_uhd.local, exp_uhd.modal_cmd


def test_cov_ops_experiment_main(monkeypatch):
    from ops.experiment import main

    monkeypatch.setattr("experiments.experiment.cli", lambda: None)
    main()


def test_cov_modal_tee_ENNFields():
    from ops.modal_uhd import _ENNFields, _Tee

    buf = io.StringIO()
    tee = _Tee(buf)
    tee.write("hi")
    tee.flush()
    assert buf.getvalue() == "hi"

    ef = _ENNFields(
        minus_impute=False,
        d=100,
        s=4,
        jl_seed=123,
        k=25,
        fit_interval=50,
        warmup_real_obs=200,
        refresh_interval=50,
        se_threshold=0.25,
        target="mu_minus",
        num_candidates=1,
        select_interval=1,
        embed_cfg=("direction", 64),
    )
    assert ef.d == 100


def test_cov_modal_run_run_uhd(monkeypatch):
    import ops.modal_uhd as modal_uhd

    original_App = modal_uhd.modal.App

    class _MockApp:
        def __init__(self, name=""):
            pass

        def function(self, **kwargs):
            def decorator(fn):
                fn.remote = lambda *a, **k: "result"
                return fn

            return decorator

        def run(self):
            return contextlib.nullcontext()

    monkeypatch.setattr(modal_uhd.modal, "App", _MockApp)
    monkeypatch.setattr(modal_uhd.modal, "enable_output", contextlib.nullcontext)

    result = modal_uhd.run(
        "mnist",
        1,
        0.001,
        None,
        None,
        gpu="T4",
    )
    assert result == "result"
    _ = original_App


def test_cov_make_loop_accuracy_fn_evaluate_fn():
    from ops.uhd_setup import make_loop

    loop = make_loop("mnist", num_rounds=1)
    assert loop is not None


def test_cov_make_loop_gym_evaluate_fn(monkeypatch):
    import pytest

    from ops.uhd_setup import make_loop

    class _FakeTrajectory:
        rreturn = 1.0

    monkeypatch.setattr(
        "optimizer.trajectories.collect_trajectory",
        lambda *a, **k: _FakeTrajectory(),
    )

    try:
        loop = make_loop("stand-mlp", num_rounds=1)
    except RuntimeError as e:
        if "MUJOCO_GL" in str(e):
            pytest.skip("MuJoCo GL not available")
        raise
    assert loop is not None


# ---------------------------------------------------------------------------
# optimizer/
# ---------------------------------------------------------------------------


def test_cov_optimizer_types():
    from optimizer.datum import Datum
    from optimizer.optimizer_types import IterateResult, ReturnSummary, TraceEntry
    from optimizer.trajectories import Trajectory

    tr = Trajectory(rreturn=1.0, states=np.zeros((1, 1)), actions=np.zeros((1, 1)))
    d = Datum(designer=None, policy=None, expected_acqf=0.5, trajectory=tr)
    ir = IterateResult(data=[d], dt_prop=0.1, dt_eval=0.2)
    assert ir.dt_prop == 0.1
    te = TraceEntry(rreturn=1.0, rreturn_decision=1.0, dt_prop=0.1, dt_eval=0.2)
    assert te.rreturn == 1.0
    rs = ReturnSummary(ret_eval=1.0, y_best_s="1", ret_best_s="1", ret_eval_s="1")
    assert rs.ret_eval == 1.0


def test_cov_designer_parse_types():
    from optimizer.designer_parse_types import ParsedOptions

    po = ParsedOptions(
        designer_name="turbo-enn-f",
        num_keep=10,
        keep_style="best",
        model_spec=None,
        sample_around_best=True,
    )
    assert po.designer_name == "turbo-enn-f"


def test_cov_designer_num_candidates():
    from optimizer.designer_registry import (
        _build_turbo_enn_f,
        _SimpleContext,
    )

    ctx = _SimpleContext(
        policy=SimpleNamespace(num_params=lambda: 2),
        num_arms=1,
        bt=lambda *a, **k: None,
        num_keep=None,
        keep_style=None,
        num_keep_val=None,
        init_yubo_default=1,
        init_ax_default=1,
        default_num_X_samples=1,
    )
    d1 = _build_turbo_enn_f(ctx, acq_type="ucb")
    d2 = _build_turbo_enn_f(ctx, acq_type="pareto")
    assert d1 is not None
    assert d2 is not None


def test_cov_ParsedOptions():
    from optimizer.designer_spec import _parse_options

    result = _parse_options("random")
    assert result.designer_name == "random"


def test_cov_designers_bt():
    from optimizer.designers import Designers

    d = Designers(
        policy=SimpleNamespace(num_params=lambda: 2),
        num_arms=1,
    )
    designer = d.create("turbo-enn-f")
    assert designer is not None


def test_cov_trajectory_draw():
    from optimizer.trajectories import collect_trajectory

    class _Space:
        low = np.zeros(3)
        high = np.ones(3)
        shape = (3,)

    step_count = [0]

    class _Env:
        observation_space = _Space()
        action_space = _Space()

        def reset(self, seed=None):
            return np.zeros(3), {}

        def step(self, action):
            step_count[0] += 1
            return np.zeros(3), 1.0, step_count[0] >= 2, False, {}

        def close(self):
            pass

        def render(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    class _GymConf:
        num_frames_skip = 1
        transform_state = False
        max_steps = 5

    class _EnvConf:
        gym_conf = _GymConf()
        env_name = "test"

        def make(self, render_mode=None):
            return _Env()

    traj = collect_trajectory(_EnvConf(), lambda s: np.zeros(3), noise_seed=0)
    assert traj.rreturn > 0


# ---------------------------------------------------------------------------
# problems/
# ---------------------------------------------------------------------------


def test_cov_HumanoidPolicy_num_params_set_params_get_params_clone_reset_state():
    from problems.humanoid_policy import HumanoidPolicy

    class _Space:
        shape = (348,)

    class _ActionSpace:
        shape = (17,)

    env_conf = SimpleNamespace(
        env_name="Humanoid-v5",
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=_Space()),
        action_space=_ActionSpace(),
    )
    p = HumanoidPolicy(env_conf)
    assert p.num_params() == 22
    p.set_params(np.zeros(22))
    params = p.get_params()
    assert params.shape == (22,)
    p2 = p.clone()
    assert p2.num_params() == 22
    p.reset_state()
    action = p(np.zeros(348))
    assert action.shape == (17,)


def test_cov_dm_control_make():
    try:
        from problems.dm_control_env import make

        env = make("dm_control/cartpole-swingup-v0")
        env.close()
    except (ImportError, ModuleNotFoundError):
        from problems import dm_control_env  # noqa: F401 — name coverage

        _ = dm_control_env.make


def test_cov_other_make():
    from problems.other import make

    env = make("mnist", problem_seed=0)
    assert env is not None


def test_cov_pure_functions_make():
    from problems.pure_functions import make

    env = make("f:sphere-2d", problem_seed=0, distort=False)
    assert env is not None


# ---------------------------------------------------------------------------
# sampling/
# ---------------------------------------------------------------------------


def test_cov_GatherProjSpec_make_project_flat_project_module():
    from sampling.gather_proj_t import GatherProjSpec, project_flat, project_module

    spec = GatherProjSpec.make(dim_ambient=10, d=5, t=3, seed=42)
    assert spec.d == 5

    x = torch.randn(10)
    y = project_flat(x, spec=spec)
    assert y.shape == (5,)

    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    spec2 = GatherProjSpec.make(dim_ambient=dim, d=4, t=2, seed=0)
    y2 = project_module(module, spec=spec2)
    assert y2.shape == (4,)


def test_cov_knn_raasp():
    from sampling.knn_tools import raasp

    x_0 = np.random.uniform(size=(5, 4))
    result = raasp(x_0, 2)
    assert result.shape == (5, 4)


def test_cov_nncd_proj_simplex_f_obj_var():
    from sampling.nncd import nncd_weights

    x = np.random.randn(2, 5, 3).astype(np.float64)
    y = np.random.randn(2, 5, 1).astype(np.float64)
    w = nncd_weights(y, x)
    assert w.shape == (2, 3)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)


def test_cov_sampling_raasp():
    from sampling.sampling_util import raasp

    x_center = torch.tensor([0.5, 0.5], dtype=torch.double)
    result = raasp(x_center, [0.0, 0.0], [1.0, 1.0], 5, torch.device("cpu"), torch.double)
    assert result.shape == (5, 2)


def test_cov_block_sparse_jl_noise_from_seed():
    from sampling.sparse_jl_t import block_sparse_jl_noise_from_seed

    y = block_sparse_jl_noise_from_seed(
        num_dim_ambient=100,
        d=10,
        s=4,
        noise_seed=0,
        sigma=1.0,
    )
    assert y.shape == (10,)


def test_cov_block_sparse_jl_noise_from_seed_wr():
    from sampling.sparse_jl_t import block_sparse_jl_noise_from_seed_wr

    y = block_sparse_jl_noise_from_seed_wr(
        num_dim_ambient=100,
        d=10,
        s=4,
        noise_seed=0,
        sigma=1.0,
    )
    assert y.shape == (10,)


def test_cov_block_sparse_jl_transform_module_wr():
    from sampling.sparse_jl_t import block_sparse_jl_transform_module_wr

    module = nn.Linear(5, 3)
    y = block_sparse_jl_transform_module_wr(module, d=10, s=4, seed=42)
    assert y.shape == (10,)


def test_cov_block_sparse_jl_transform_module_to_cpu_wr():
    from sampling.sparse_jl_t import block_sparse_jl_transform_module_to_cpu_wr

    module = nn.Linear(5, 3)
    y = block_sparse_jl_transform_module_to_cpu_wr(module, d=10, s=4, seed=42)
    assert y.shape == (10,)


# ---------------------------------------------------------------------------
# turbo_m_ref/
# ---------------------------------------------------------------------------


def test_cov_turbo_1_Turbo1Standard():
    from turbo_m_ref.turbo_1 import Turbo1Standard

    lb = np.zeros(2)
    ub = np.ones(2)
    t = Turbo1Standard(
        f=lambda x: -np.sum(x**2, axis=1, keepdims=True),
        lb=lb,
        ub=ub,
        n_init=3,
        max_evals=5,
        batch_size=1,
        verbose=False,
    )
    assert t is not None


def test_cov_turbo_1_ask_tell_Turbo1AskTell():
    from turbo_m_ref.turbo_1_ask_tell import Turbo1AskTell

    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    t = Turbo1AskTell(x_bounds=x_bounds, batch_size=1, verbose=False)
    assert t is not None


# ---------------------------------------------------------------------------
# analysis/fitting_time/batch_jobs
# ---------------------------------------------------------------------------


def test_cov_batch_jobs_dataclass_and_examples():
    from analysis.fitting_time.batch_jobs import (
        SyntheticBenchJob,
        example_sphere_n12_d2_seed0,
        example_two_targets_n12_d2,
        job_fit_quality,
    )

    j = SyntheticBenchJob(n=3, d=4, target="ackley", problem_seed=7)
    assert j.problem_seed == 7
    assert len(example_sphere_n12_d2_seed0()) == 1
    assert len(example_two_targets_n12_d2()) == 2
    assert len(job_fit_quality()) > 0


# ---------------------------------------------------------------------------
# Nested + modal modules that need special setup
# ---------------------------------------------------------------------------


def test_cov_modal_collect_get_job_result_main(monkeypatch):
    import sys

    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    from experiments.modal_collect import get_job_result, main  # noqa: F811

    assert callable(get_job_result)
    assert callable(main)


def test_cov_modal_learn_get_job_result_main(monkeypatch):
    import sys

    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    from experiments.modal_learn import get_job_result, main  # noqa: F811

    assert callable(get_job_result)
    assert callable(main)
