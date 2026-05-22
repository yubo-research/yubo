from functools import partial

from .designer_registry_option_handlers import (
    _build_turbo_enn_f,
    _d_morbo_enn_fit,
    _d_pss_sweep_kmcmc,
    _d_pss_sweep_num_mcmc,
    _d_rff,
    _d_sts_ar,
    _d_sts_sweep,
    _d_ts_sweep,
    _d_turbo_enn_fit,
    _d_turbo_enn_fit_ucb,
    _d_turbo_enn_p,
    _d_turbo_enn_sweep,
)
from .designer_types import DesignerDef, DesignerOptionSpec

_DESIGNER_DEFS: list[DesignerDef] = [
    DesignerDef(
        name="ts_sweep",
        builder=_d_ts_sweep,
        option_specs=(
            DesignerOptionSpec(
                name="num_candidates",
                required=True,
                value_type="int",
                description="Number of TS candidates (lanczos sampler).",
                example_suffix="num_candidates=10000",
            ),
        ),
    ),
    DesignerDef(
        name="rff",
        builder=_d_rff,
        option_specs=(
            DesignerOptionSpec(
                name="num_candidates",
                required=True,
                value_type="int",
                description="Number of TS candidates (RFF sampler).",
                example_suffix="num_candidates=10000",
            ),
        ),
    ),
    DesignerDef(
        name="pss_sweep_kmcmc",
        builder=_d_pss_sweep_kmcmc,
        option_specs=(
            DesignerOptionSpec(
                name="k_mcmc",
                required=True,
                value_type="int",
                description="Number of MCMC chains per refinement (PSS).",
                example_suffix="k_mcmc=8",
            ),
        ),
    ),
    DesignerDef(
        name="pss_sweep_num_mcmc",
        builder=_d_pss_sweep_num_mcmc,
        option_specs=(
            DesignerOptionSpec(
                name="num_mcmc",
                required=True,
                value_type="int",
                description="Total number of MCMC samples per refinement (PSS).",
                example_suffix="num_mcmc=16",
            ),
        ),
    ),
    DesignerDef(
        name="sts_sweep",
        builder=_d_sts_sweep,
        option_specs=(
            DesignerOptionSpec(
                name="num_refinements",
                required=True,
                value_type="int",
                description="Number of STS refinements.",
                example_suffix="num_refinements=30",
            ),
        ),
    ),
    DesignerDef(
        name="turbo-enn-sweep",
        builder=_d_turbo_enn_sweep,
        option_specs=(
            DesignerOptionSpec(
                name="k",
                required=True,
                value_type="int",
                description="Ensemble size for TuRBO-ENN sweep.",
                example_suffix="k=10",
            ),
            DesignerOptionSpec(
                name="idx",
                required=False,
                value_type="str",
                description="ENN index driver: flat (default) or hnsw.",
                example_suffix="idx=hnsw",
                allowed_values=("flat", "hnsw", "exact"),
            ),
        ),
    ),
    DesignerDef(
        name="turbo-enn-fit",
        builder=_d_turbo_enn_fit,
        option_specs=(
            DesignerOptionSpec(
                name="acq_type",
                required=True,
                value_type="str",
                description="Acquisition type for fit-time candidate generation.",
                example_suffix="acq_type=ucb",
                allowed_values=("pareto", "thompson", "ucb"),
            ),
            DesignerOptionSpec(
                name="idx",
                required=False,
                value_type="str",
                description="ENN index driver: flat (default) or hnsw.",
                example_suffix="idx=hnsw",
                allowed_values=("flat", "hnsw", "exact"),
            ),
        ),
    ),
    DesignerDef(
        name="morbo-enn-fit",
        builder=_d_morbo_enn_fit,
        option_specs=(
            DesignerOptionSpec(
                name="acq_type",
                required=True,
                value_type="str",
                description="Acquisition type for fit-time candidate generation (MORBO TR).",
                example_suffix="acq_type=ucb",
                allowed_values=("pareto", "thompson", "ucb"),
            ),
            DesignerOptionSpec(
                name="idx",
                required=False,
                value_type="str",
                description="ENN index driver: flat (default) or hnsw.",
                example_suffix="idx=hnsw",
                allowed_values=("flat", "hnsw", "exact"),
            ),
        ),
    ),
    DesignerDef(
        name="sts-ar",
        builder=_d_sts_ar,
        option_specs=(
            DesignerOptionSpec(
                name="num_acc_rej",
                required=True,
                value_type="int",
                description="Number of accept/reject steps.",
                example_suffix="num_acc_rej=10",
            ),
        ),
    ),
    DesignerDef(
        name="turbo-enn-p",
        builder=_d_turbo_enn_p,
        option_specs=(
            DesignerOptionSpec(
                name="idx",
                required=False,
                value_type="str",
                description="ENN index driver: flat (default) or hnsw.",
                example_suffix="idx=hnsw",
                allowed_values=("flat", "hnsw", "exact"),
            ),
        ),
    ),
    DesignerDef(
        name="turbo-enn-fit-ucb",
        builder=_d_turbo_enn_fit_ucb,
        option_specs=(
            DesignerOptionSpec(
                name="nfs",
                required=False,
                value_type="int",
                description="Number of fit samples for hyperparameter selection (default 100).",
                example_suffix="nfs=50",
            ),
            DesignerOptionSpec(
                name="k",
                required=False,
                value_type="int",
                description="ENN ensemble size k (default 10).",
                example_suffix="k=20",
            ),
            DesignerOptionSpec(
                name="idx",
                required=False,
                value_type="str",
                description="ENN index driver: flat (default) or hnsw.",
                example_suffix="idx=hnsw",
                allowed_values=("flat", "hnsw", "exact"),
            ),
        ),
    ),
    DesignerDef(
        name="turbo-enn-f",
        builder=partial(_build_turbo_enn_f, acq_type="ucb"),
        option_specs=(),
    ),
    DesignerDef(
        name="turbo-enn-f-p",
        builder=partial(_build_turbo_enn_f, acq_type="pareto"),
        option_specs=(),
    ),
]

_DESIGNER_OPTION_SPECS: dict[str, list[DesignerOptionSpec]] = {d.name: list(d.option_specs) for d in _DESIGNER_DEFS if d.option_specs}
