from functools import partial

from .designer_registry_extra_builders import _build_eggroll
from .designer_registry_mars import MARS_DESIGNER_DEFS
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
    _d_turbo_enn_varentropy_ucb,
)
from .designer_types import DesignerDef, DesignerOptionSpec

_IDX_OPTION_SPEC = DesignerOptionSpec(
    name="idx",
    required=False,
    value_type="str",
    description="ENN index driver: flat (default), hnsw, or exact.",
    example_suffix="idx=hnsw",
    allowed_values=("flat", "hnsw", "exact"),
)

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
            _IDX_OPTION_SPEC,
        ),
    ),
    DesignerDef(
        name="turbo-enn-p",
        builder=_d_turbo_enn_p,
        option_specs=(_IDX_OPTION_SPEC,),
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
            _IDX_OPTION_SPEC,
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
            _IDX_OPTION_SPEC,
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
                name="num_init",
                required=False,
                value_type="int",
                description="Initial evaluations before TuRBO proposals.",
                example_suffix="num_init=1",
            ),
            _IDX_OPTION_SPEC,
        ),
    ),
    DesignerDef(
        name="turbo-enn-varentropy-ucb",
        builder=_d_turbo_enn_varentropy_ucb,
        option_specs=(
            DesignerOptionSpec(
                name="k",
                required=False,
                value_type="int",
                description="Number of ENN neighbors in the local evidence distribution.",
                example_suffix="k=10",
            ),
            DesignerOptionSpec(
                name="varentropy_scale",
                required=False,
                value_type="float",
                description="Multiplier for normalized neighbor-weight varentropy in the ENN uncertainty scale.",
                example_suffix="varentropy_scale=0.5",
            ),
            DesignerOptionSpec(
                name="num_init",
                required=False,
                value_type="int",
                description="Initial Latin-hypercube evaluations before TuRBO proposals.",
                example_suffix="num_init=16",
            ),
            DesignerOptionSpec(
                name="num_candidates",
                required=False,
                value_type="int",
                description="Number of TuRBO candidates scored by UCB.",
                example_suffix="num_candidates=2048",
            ),
            DesignerOptionSpec(
                name="candidate_rv",
                required=False,
                value_type="str",
                description="TuRBO candidate generator: sobol, uniform, or gpu_uniform.",
                example_suffix="candidate_rv=sobol",
                allowed_values=("sobol", "uniform", "gpu_uniform"),
            ),
            _IDX_OPTION_SPEC,
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
    DesignerDef(
        name="eggroll",
        builder=_build_eggroll,
        option_specs=(
            DesignerOptionSpec(
                name="noiser",
                required=False,
                value_type="str",
                description="HyperscaleES noiser name.",
                example_suffix="noiser=eggroll",
            ),
            DesignerOptionSpec(
                name="sigma",
                required=False,
                value_type="float",
                description="Initial noiser sigma.",
                example_suffix="sigma=0.05",
            ),
            DesignerOptionSpec(
                name="sigma_decay",
                required=False,
                value_type="float",
                description="Per-generation multiplicative sigma decay.",
                example_suffix="sigma_decay=0.999",
            ),
            DesignerOptionSpec(
                name="lr",
                required=False,
                value_type="float",
                description="Initial optimizer learning rate.",
                example_suffix="lr=0.02",
            ),
            DesignerOptionSpec(
                name="lr_decay",
                required=False,
                value_type="float",
                description="Per-update multiplicative learning-rate decay.",
                example_suffix="lr_decay=0.9995",
            ),
            DesignerOptionSpec(
                name="rank",
                required=False,
                value_type="int",
                description="Low-rank EggRoll perturbation rank.",
                example_suffix="rank=8",
            ),
            DesignerOptionSpec(
                name="rank_transform",
                required=False,
                value_type="bool",
                description="Rank-transform population scores before HyperscaleES fitness normalization.",
                example_suffix="rank_transform=false",
            ),
            DesignerOptionSpec(
                name="deterministic_policy",
                required=False,
                value_type="bool",
                description="Use distribution modes/means for policy actions instead of sampling.",
                example_suffix="deterministic_policy=false",
            ),
            DesignerOptionSpec(
                name="steps",
                required=False,
                value_type="int",
                description="Rollout horizon per sampled policy.",
                example_suffix="steps=200",
            ),
            DesignerOptionSpec(
                name="num_envs",
                required=False,
                value_type="int",
                description="Held-out center-policy evaluation episodes per generation.",
                example_suffix="num_envs=8",
            ),
            DesignerOptionSpec(
                name="batch_size",
                required=False,
                value_type="int",
                description="External scorer candidate batch size.",
                example_suffix="batch_size=4",
            ),
            DesignerOptionSpec(
                name="jax_sim",
                required=False,
                value_type="bool",
                description="Use JAX EggRoll rollouts for Isaac Lab env tags (host callbacks).",
                example_suffix="jax_sim=true",
            ),
            DesignerOptionSpec(
                name="suppress_noiser_stdout",
                required=False,
                value_type="bool",
                description="Suppress noisy upstream HyperscaleES tracing prints.",
                example_suffix="suppress_noiser_stdout=true",
            ),
        ),
    ),
] + MARS_DESIGNER_DEFS

_DESIGNER_OPTION_SPECS: dict[str, list[DesignerOptionSpec]] = {d.name: list(d.option_specs) for d in _DESIGNER_DEFS if d.option_specs}
