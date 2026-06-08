from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


def test_cov_optimizer_types():
    from optimizer.datum import Datum
    from optimizer.optimizer_types import IterateResult, ReturnSummary, TraceEntry
    from optimizer.trajectory import Trajectory

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

    space = SimpleNamespace(low=np.zeros(3), high=np.ones(3), shape=(3,))
    step_count = [0]

    def _env_reset(self, seed=None):
        return np.zeros(3), {}

    def _env_step(self, action):
        step_count[0] += 1
        return np.zeros(3), 1.0, step_count[0] >= 2, False, {}

    def _env_close(self):
        pass

    def _env_render(self):
        return np.zeros((4, 4, 3), dtype=np.uint8)

    _Env = type(
        "_Env",
        (),
        {
            "observation_space": space,
            "action_space": space,
            "reset": _env_reset,
            "step": _env_step,
            "close": _env_close,
            "render": _env_render,
        },
    )

    gym_conf = SimpleNamespace(num_frames_skip=1, transform_state=False, max_steps=5)

    def _envconf_make(self, render_mode=None):
        return _Env()

    _EnvConf = type(
        "_EnvConf",
        (),
        {"gym_conf": gym_conf, "env_name": "test", "make": _envconf_make},
    )

    traj = collect_trajectory(_EnvConf(), lambda s: np.zeros(3), noise_seed=0)
    assert traj.rreturn > 0


def test_cov_HumanoidPolicy_num_params_set_params_get_params_clone_reset_state():
    from problems.humanoid_policy import HumanoidPolicy

    env_conf = SimpleNamespace(
        env_name="Humanoid-v5",
        problem_seed=0,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(348,))),
        action_space=SimpleNamespace(shape=(17,)),
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
    import pytest

    try:
        from problems.dm_control_env import make

        env = make("dm_control/cartpole-swingup-v0")
        env.close()
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        if isinstance(exc, AttributeError) and "actuator_armature" not in str(exc):
            raise
        pytest.skip(f"dm_control is unavailable or incompatible: {exc}")


def test_cov_other_make():
    from problems.other import make

    env = make("mnist", problem_seed=0)
    assert env is not None


def test_cov_pure_functions_make():
    from problems.pure_functions import make

    env = make("f:sphere-2d", problem_seed=0, distort=False)
    assert env is not None


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


def test_cov_modal_collect_get_job_result_main(monkeypatch):
    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    from experiments.modal_collect import get_job_result, main

    assert callable(get_job_result)
    assert callable(main)


def test_cov_modal_learn_get_job_result_main(monkeypatch):
    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    from experiments.modal_learn import get_job_result, main

    assert callable(get_job_result)
    assert callable(main)
