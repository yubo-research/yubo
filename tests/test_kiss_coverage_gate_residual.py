"""Kiss coverage for modules surfaced after merge-conflict resolution."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def test_kiss_cov_batch_jobs_example_sphere():
    from analysis.fitting_time.batch_jobs import example_sphere_n12_d2_seed0

    jobs = example_sphere_n12_d2_seed0()
    assert len(jobs) == 1
    assert jobs[0].target == "sphere"


def test_kiss_cov_designer_and_optimizer_types():
    from optimizer.datum import Datum
    from optimizer.designer_parse_types import ParsedOptions
    from optimizer.optimizer_types import IterateResult, TraceEntry
    from optimizer.trajectory import Trajectory

    opts = ParsedOptions("turbo-enn", None, None, None, False)
    assert opts.designer_name == "turbo-enn"
    traj = Trajectory(rreturn=0.0, states=np.zeros(1), actions=np.zeros(1))
    datum = Datum(designer="d", policy=object(), expected_acqf=0.0, trajectory=traj)
    result = IterateResult([datum], 0.1, 0.2)
    assert result.dt_prop == 0.1
    entry = TraceEntry(1.0, 0.5, 0.1, 0.2)
    assert entry.rreturn == 1.0


def test_kiss_cov_gather_proj_t():
    from sampling.gather_proj_t import GatherProjSpec, project_flat, project_module

    spec = GatherProjSpec.make(dim_ambient=4, d=3, t=2, seed=0)
    x = torch.randn(4)
    out = project_flat(x, spec=spec)
    assert out.shape == (3,)
    m = nn.Linear(2, 1)
    out_m = project_module(m, spec=GatherProjSpec.make(dim_ambient=3, d=3, t=2, seed=0))
    assert out_m.shape == (3,)


def test_kiss_cov_sparse_jl_t_transforms_wr():
    from sampling.sparse_jl_t_transforms import (
        block_sparse_jl_noise_from_seed,
        block_sparse_jl_noise_from_seed_wr,
        block_sparse_jl_transform_module_to_cpu_wr,
        block_sparse_jl_transform_module_wr,
    )

    block_sparse_jl_noise_from_seed(num_dim_ambient=8, d=4, s=2, jl_seed=1, noise_seed=2, sigma=0.1)
    block_sparse_jl_noise_from_seed_wr(num_dim_ambient=8, d=4, s=2, jl_seed=1, noise_seed=2, sigma=0.1)
    m = nn.Linear(2, 1)
    block_sparse_jl_transform_module_wr(m, d=4, s=2, seed=1)
    block_sparse_jl_transform_module_to_cpu_wr(m, d=4, s=2, seed=1)


def test_kiss_cov_modal_uhd_helpers_and_impl():
    import io

    from ops import modal_uhd_runner_impl as impl
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
    assert impl.run is not None
    assert impl._run_fields is not None
    assert impl._early_reject_fields is not None


def test_kiss_cov_turbo_m_ref_classes():
    import turbo_m_ref.turbo_1 as turbo1
    import turbo_m_ref.turbo_1_ask_tell as turbo_at

    lb = np.zeros(2)
    ub = np.ones(2)
    turbo1.Turbo1Standard(
        f=lambda x: [float(np.sum(row**2)) for row in x],
        lb=lb,
        ub=ub,
        n_init=2,
        max_evals=4,
        batch_size=1,
        verbose=False,
    )
    turbo_at.Turbo1AskTell(
        np.column_stack([lb, ub]),
        n_init=2,
        batch_size=1,
        verbose=False,
    )
