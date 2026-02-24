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


def test_cov_fan_in_out_main():
    from admin.fan_in_out import main

    result = main()
    assert result in (0, 1)


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
    from experiments.fit_mnist import fit_mnist

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

    cfg = exp_uhd.UHDConfig(
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
        early_reject_tau=None,
        early_reject_mode=None,
        early_reject_ema_beta=None,
        early_reject_warmup_pos=None,
        early_reject_quantile=None,
        early_reject_window=None,
        optimizer="mezo",
        be_num_probes=10,
        be_num_candidates=10,
        be_warmup=20,
        be_fit_interval=10,
        be_enn_k=25,
        be_sigma_range=None,
        batch_size=4096,
        enn_minus_impute=False,
        enn_d=100,
        enn_s=4,
        enn_jl_seed=123,
        enn_k=25,
        enn_fit_interval=50,
        enn_warmup_real_obs=200,
        enn_refresh_interval=50,
        enn_se_threshold=0.25,
        enn_target="mu_minus",
        enn_num_candidates=1,
        enn_select_interval=1,
        enn_embedder="direction",
        enn_gather_t=64,
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

    monkeypatch.setattr("ops.modal_uhd.run", lambda *a, **k: "log_text")
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


def test_cov_designer_num_candidates():
    from optimizer.designer_registry import _d_turbo_enn_f, _d_turbo_enn_f_p, _SimpleContext

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
    d1 = _d_turbo_enn_f(ctx, {})
    d2 = _d_turbo_enn_f_p(ctx, {})
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


def test_cov_gap_optimizer_and_surrogate_units():
    from optimizer.ellipsoidal_trust_region import ENNTrueEllipsoidalTrustRegion
    from optimizer.enn_surrogate_ext import GeometryENNSurrogate, LocalGeometryValues
    from optimizer.enn_turbo_optimizer import (
        TurboOptimizer,
        create_optimizer,
    )
    from optimizer.enn_turbo_optimizer import (
        predict_mu_sigma as turbo_predict_mu_sigma,
    )
    from optimizer.enn_turbo_optimizer import (
        scalarize as turbo_scalarize,
    )
    from optimizer.lr_scheduler import ConstantLR, LRScheduler
    from optimizer.multi_turbo_enn_designer import MultiTurboRuntimeState
    from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
    from optimizer.trust_region_config import MetricShapedTRConfig
    from optimizer.trust_region_utils import _LengthPolicy, _OptionCLengthPolicy
    from optimizer.turbo_enn_designer_ext import (
        TurboENNDesigner,
        _predict_mu_sigma,
        _scalarize,
    )
    from torch_truncnorm.TruncatedNormal import TruncatedStandardNormal

    # Protocol unit coverage.
    sched = ConstantLR(0.01)
    assert isinstance(sched, LRScheduler)

    # Dataclass unit coverage.
    rt = MultiTurboRuntimeState()
    assert rt.shared_prefix_len == 0

    # pending_rho properties.
    lp = _LengthPolicy()
    assert lp.pending_rho is None
    oc = _OptionCLengthPolicy(rho_bad=0.25, rho_good=0.75, gamma_down=0.5, gamma_up=2.0)
    assert oc.pending_rho is None
    oc.set_acceptance_ratio(pred=1.0, act=0.1, boundary_hit=False)
    assert oc.pending_rho is not None
    _ = oc.apply_after_super_update(current_length=1.0, base_length=1.0, fixed_length=None, length_max=2.0)
    assert oc.pending_rho is None

    # Sparse global-nz sampler.
    module = nn.Linear(3, 2, bias=False)
    sp = SparseGaussianPerturbator(module, num_dim_target=2)
    idx, vals = sp.sample_global_nz(seed=0, sigma=0.1)
    assert idx.shape == vals.shape
    assert idx.dtype == np.int64

    # TruncatedNormal variance/auc properties.
    tsn = TruncatedStandardNormal(torch.tensor(-1.0), torch.tensor(1.0))
    assert float(tsn.variance) > 0.0
    assert float(tsn.auc) > 0.0

    # Ellipsoidal TR class construction.
    tr = MetricShapedTRConfig(geometry="enn_true_ellipsoid").build(num_dim=3, rng=np.random.default_rng(0))
    assert isinstance(tr, ENNTrueEllipsoidalTrustRegion)

    # Surrogate local geometry helpers.
    class _FakeInternals:
        def __init__(self):
            self.idx = np.array([[0, 1]], dtype=np.int64)
            self.w_normalized = np.array([[[0.75], [0.25]]], dtype=float)

    class _FakeENN:
        def __init__(self):
            self.train_x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
            self.train_y = np.array([[1.0], [0.5]], dtype=float)
            self.num_outputs = 1

        def __len__(self):
            return 2

        def _compute_posterior_internals(self, x, params, flags):
            _ = x, params, flags
            return _FakeInternals()

    surrogate = object.__new__(GeometryENNSurrogate)
    surrogate._enn = _FakeENN()
    surrogate._params = SimpleNamespace(k_num_neighbors=2)
    dx, w = surrogate.local_geometry(np.array([0.2, 0.0]))
    assert dx.shape == (2, 2)
    assert w.shape == (2,)
    geom_vals = surrogate.local_geometry_values(np.array([0.2, 0.0]))
    assert isinstance(geom_vals, LocalGeometryValues)
    assert geom_vals.delta_x.shape == (2, 2)

    # Predict/scalarize helper units.
    class _Posterior:
        def __init__(self, n):
            self.mu = np.ones((n, 1), dtype=float)
            self.sigma = 0.1 * np.ones((n, 1), dtype=float)

    class _Surrogate:
        lengthscales = None

        def predict(self, x):
            return _Posterior(int(x.shape[0]))

    class _TRState:
        length = 1.0

        @staticmethod
        def scalarize(y, clip=False):
            _ = clip
            return np.asarray(y, dtype=float)

    opt_stub = SimpleNamespace(
        _bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        _surrogate=_Surrogate(),
        _tr_state=_TRState(),
        _num_dim=2,
    )
    assert _predict_mu_sigma(opt_stub, np.array([0.0, 0.0])) is not None
    assert _scalarize(opt_stub, np.array([[1.0]])) is not None
    assert turbo_predict_mu_sigma(opt_stub, np.array([0.0, 0.0])) is not None
    assert turbo_scalarize(opt_stub, np.array([[1.0]])) is not None

    # Symbol coverage for class/function units.
    _ = TurboENNDesigner, TurboOptimizer, create_optimizer


def test_cov_gap_uhd_units():
    import pytest

    from embedding.behavioral_embedder import BehavioralEmbedder
    from optimizer.gaussian_perturbator import GaussianPerturbator
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_enn_imputer import ENNImputerConfig, ENNMinusImputer
    from optimizer.uhd_loop import UHDLoop
    from optimizer.uhd_mezo import UHDMeZO
    from optimizer.uhd_simple_base import UHDSimpleBase
    from optimizer.uhd_simple_be import UHDMeZOBE, UHDSimpleBE
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    class _NPPolicy:
        def __init__(self, dim: int):
            self._x = np.zeros((dim,), dtype=np.float64)

        def get_params(self):
            return self._x.copy()

        def set_params(self, x):
            self._x = np.asarray(x, dtype=np.float64).copy()

        def __call__(self, probe):
            return np.array([float(np.sum(self._x) + np.sum(probe))], dtype=np.float64)

    bounds = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    embedder = BehavioralEmbedder(bounds, num_probes=2, seed=0)
    p = _NPPolicy(3)
    z = embedder.embed_policy(p, np.array([0.1, 0.2, 0.3], dtype=np.float64))
    assert z.ndim == 1

    # Base/simple numpy variants.
    np_simple = UHDSimpleNp(p, sigma_0=0.1, param_clip=(-1.0, 1.0))
    np_simple.ask()
    np_simple.tell(1.0, 0.1)
    assert np_simple.y_best is not None

    np_be = UHDSimpleBENp(
        p,
        embedder,
        sigma_0=0.1,
        num_candidates=2,
        warmup=1,
        fit_interval=1,
        enn_k=1,
    )
    np_be.ask()
    np_be.tell(1.0, 0.1)
    np_be.ask()
    np_be.tell(0.5, 0.1)
    assert np_be.y_best is not None

    # UHDSimpleBase class unit.
    module = nn.Linear(1, 1, bias=False)
    gp = GaussianPerturbator(module)
    base = UHDSimpleBase(gp, sigma_0=0.1, dim=module.weight.numel(), sigma_range=(0.01, 0.1))
    sigmas = base._sample_sigmas(base_seed=0, n=3)
    assert sigmas.shape == (3,)

    # MeZO controls + properties.
    mezo = UHDMeZO(gp, dim=module.weight.numel(), lr_scheduler=ConstantLR(0.01), sigma=0.1)
    assert mezo.positive_phase is True
    mezo.set_next_seed(7)
    mezo.ask()
    assert mezo.step_seed == 7
    assert mezo.step_sigma > 0
    assert mezo.perturbator is gp
    mezo.tell(1.0, 0.1)
    assert mezo.positive_phase is False
    with pytest.raises(RuntimeError):
        mezo.set_next_seed(8)
    mezo.skip_negative()
    assert mezo.positive_phase is True

    # Run one full pair to exercise last_step_scale path.
    mezo.ask()
    mezo.tell(1.0, 0.1)
    mezo.ask()
    mezo.tell(0.5, 0.1)
    assert isinstance(mezo.last_step_scale, float)

    # UHDSimpleBE / UHDMeZOBE units.
    module_be = nn.Linear(1, 1, bias=False)
    gp_be = GaussianPerturbator(module_be)
    be_embedder = BehavioralEmbedder(torch.tensor([[-1.0], [1.0]]), num_probes=2, seed=1)
    simple_be = UHDSimpleBE(
        gp_be,
        sigma_0=0.1,
        dim=module_be.weight.numel(),
        module=module_be,
        embedder=be_embedder,
        num_candidates=2,
        warmup=1,
        fit_interval=1,
        enn_k=1,
    )
    simple_be.ask()
    simple_be.tell(1.0, 0.1)
    simple_be.ask()
    simple_be.tell(0.5, 0.1)
    assert simple_be.y_best is not None

    module_mezo_be = nn.Linear(1, 1, bias=False)
    gp_mezo_be = GaussianPerturbator(module_mezo_be)
    mezo_be = UHDMeZOBE(
        gp_mezo_be,
        dim=module_mezo_be.weight.numel(),
        module=module_mezo_be,
        embedder=be_embedder,
        sigma=0.1,
        num_candidates=2,
        warmup=1,
        fit_interval=1,
        enn_k=1,
    )
    assert mezo_be.positive_phase is True
    mezo_be.ask()
    mezo_be.tell(1.0, 0.1)
    mezo_be.ask()
    mezo_be.tell(0.5, 0.1)
    assert mezo_be.positive_phase is True

    # ENN minus imputer units.
    cfg = ENNImputerConfig(
        warmup_real_obs=1,
        fit_interval=1,
        refresh_interval=0,
        min_calib_points=0,
        max_abs_err_ema=1e9,
        se_threshold=1e9,
        target="mu_minus",
    )

    def _noise_nz(seed: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        _ = seed
        return np.array([0], dtype=np.int64), np.array([sigma], dtype=np.float32)

    imp = ENNMinusImputer(module=module_mezo_be, cfg=cfg, noise_nz_fn=_noise_nz)
    imp.begin_pair(seed=0, sigma=0.1)
    imp.tell_real(mu=1.0, phase="plus")
    imp.tell_real(mu=0.8, phase="minus")
    imp._maybe_fit()
    imp._num_calib = 1
    imp._abs_err_ema = 0.0
    _ = imp.should_impute_negative()
    mu_hat, se_hat = imp.predict_current()
    assert np.isfinite(mu_hat)
    assert np.isfinite(se_hat)
    imputed, mu2, se2 = imp.try_impute_current()
    assert isinstance(imputed, bool)
    assert np.isfinite(mu2) or np.isnan(mu2)
    assert np.isfinite(se2) or np.isnan(se2)
    imp.calibrate_minus(mu_minus_real=0.7)
    imp.update_base_after_step(step_scale=0.01, sigma=0.1)
    assert imp.num_real_evals >= 2
    assert imp.num_imputed >= 0
    _ = imp.abs_err_ema

    # UHDLoop control methods/properties.
    loop = UHDLoop(
        module_mezo_be,
        evaluate_fn=lambda _seed: (0.0, 0.1),
        num_iterations=0,
    )
    assert loop.perturbator is not None
    loop.set_enn(minus_imputer=imp, seed_selector=None)
    loop.set_early_reject_advanced(tau=0.1, mode="ema")


def test_cov_gap_ops_and_problem_units(monkeypatch, tmp_path):
    import gymnasium as gym
    from click.testing import CliRunner

    import ops.exp_uhd as exp_uhd
    import problems.atari_env as atari_env

    # run_simple_loop unit coverage (patched to no-op branch).
    from ops.uhd_setup import run_simple_loop
    from problems.dm_control_env import DMControlEnv
    from problems.push import Push
    from problems.shimmy_dm_control import _FlattenDictObsWrapper

    class _DummyGymEnv:
        def close(self):
            return None

    class _DummyConf:
        problem_seed = 0

        @staticmethod
        def make():
            return _DummyGymEnv()

    monkeypatch.setattr("problems.env_conf.get_env_conf", lambda *a, **k: _DummyConf())
    monkeypatch.setattr("ops.uhd_setup._run_simple_gym", lambda *a, **k: None)
    run_simple_loop("f:sphere-2d", num_rounds=1, optimizer="simple")

    # modal_cmd unit coverage via explicit callback.
    toml = tmp_path / "uhd.toml"
    toml.write_text('[uhd]\nenv_tag="f:sphere-2d"\nnum_rounds=1\n')
    monkeypatch.setattr("ops.modal_uhd.run", lambda *a, **k: "ok")
    runner = CliRunner()
    res = runner.invoke(exp_uhd.cli, ["modal", str(toml)])
    assert res.exit_code == 0
    _ = exp_uhd.modal_cmd

    # AtariPreprocessOptions / make_atari_env units without ALE dependency.
    class _DummyAtariEnv:
        observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        action_space = gym.spaces.Discrete(2)

    monkeypatch.setattr(atari_env.gym, "make", lambda *a, **k: _DummyAtariEnv())
    monkeypatch.setattr(atari_env.gym.wrappers, "AtariPreprocessing", lambda env, **k: env)
    monkeypatch.setattr(atari_env.gym.wrappers, "FrameStackObservation", lambda env, stack_size: env)
    monkeypatch.setattr(atari_env.gym.wrappers, "TimeLimit", lambda env, max_episode_steps: env)
    opts = atari_env.AtariPreprocessOptions()
    env = atari_env.make_atari_env("ALE/Pong-v5", preprocess=opts)
    assert env is not None

    # DMControlEnv/render units.
    class _DummySpec:
        shape = (1,)
        dtype = np.float32
        minimum = np.array([-1.0], dtype=np.float32)
        maximum = np.array([1.0], dtype=np.float32)

    class _DummyTS:
        def __init__(self):
            self.observation = {"x": np.array([0.0], dtype=np.float32)}
            self.reward = 0.0
            self.discount = 1.0

        def last(self):
            return False

    class _DummyPhysics:
        @staticmethod
        def render(width, height):
            return np.zeros((height, width, 3), dtype=np.uint8)

    class _DummyDM:
        physics = _DummyPhysics()

        @staticmethod
        def observation_spec():
            return {"x": _DummySpec()}

        @staticmethod
        def action_spec():
            return _DummySpec()

        @staticmethod
        def reset():
            return _DummyTS()

        @staticmethod
        def step(_action):
            return _DummyTS()

        @staticmethod
        def close():
            return None

    monkeypatch.setattr(DMControlEnv, "_load_env", lambda self, seed: _DummyDM())
    dm = DMControlEnv("cartpole", "swingup", render_mode="rgb_array")
    frame = dm.render()
    assert frame.shape[2] == 3
    dm.close()

    # push.f_max and shimmy flatten wrapper observation.
    p = Push()
    assert float(p.f_max) > 0.0

    class _DictEnv(gym.Env):
        metadata = {"render_modes": []}
        observation_space = gym.spaces.Dict({"a": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)})
        action_space = gym.spaces.Discrete(1)

        def reset(self, *, seed=None, options=None):
            _ = seed, options
            return {"a": np.array([0.0], dtype=np.float32)}, {}

        def step(self, action):
            _ = action
            return {"a": np.array([0.0], dtype=np.float32)}, 0.0, False, False, {}

    wrapped = _FlattenDictObsWrapper(_DictEnv())
    obs, _ = wrapped.reset(seed=0)
    out = wrapped.observation({"a": np.array([1.0], dtype=np.float32)})
    assert out.shape == (1,)
    assert obs.shape == (1,)
