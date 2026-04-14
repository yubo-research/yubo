import importlib
from functools import partial

from .designer_errors import NoSuchDesignerError
from .designer_types import DesignerOptionSpec


class _SimpleContext:
    def __init__(
        self,
        policy,
        num_arms,
        bt,
        *,
        num_keep,
        keep_style,
        num_keep_val,
        init_yubo_default,
        init_ax_default,
        default_num_X_samples,
        env_conf=None,
    ):
        self.policy = policy
        self.num_arms = num_arms
        self.bt = bt
        self.num_keep = num_keep
        self.keep_style = keep_style
        self.num_keep_val = num_keep_val
        self.init_yubo_default = init_yubo_default
        self.init_ax_default = init_ax_default
        self.default_num_X_samples = default_num_X_samples
        self.env_conf = env_conf


def _load_symbol(module: str, name: str):
    mod = importlib.import_module(module)
    return getattr(mod, name)


def _no_opts(name: str, build, ctx: _SimpleContext, opts: dict):
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer '{name}' does not support options (got: {keys}).")
    return build(ctx)


def _require_int(opts: dict, key: str, *, example: str) -> int:
    if key not in opts:
        raise NoSuchDesignerError(f"Designer option '{key}' is required. Example: '{example}'.")
    v = opts[key]
    if not isinstance(v, int):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int.")
    return v


def _optional_int(opts: dict, key: str, default: int, *, example: str) -> int:
    if key not in opts:
        return int(default)
    v = opts[key]
    if not isinstance(v, int):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int. Example: '{example}'.")
    return int(v)


def _optional_number(opts: dict, key: str, default: float | None, *, example: str) -> float | None:
    if key not in opts:
        return None if default is None else float(default)
    v = opts[key]
    if not isinstance(v, (int, float)):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a number. Example: '{example}'.")
    return float(v)


def _optional_bool(opts: dict, key: str, default: bool, *, example: str) -> bool:
    if key not in opts:
        return bool(default)
    v = opts[key]
    if not isinstance(v, bool):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a bool. Example: '{example}'.")
    return bool(v)


def _optional_str_in(opts: dict, key: str, default: str, allowed: set[str], *, example: str) -> str:
    if key not in opts:
        return default
    v = opts[key]
    if not isinstance(v, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
    if v not in allowed:
        raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
    return v


def _require_str_in(opts: dict, key: str, allowed: set[str], *, example: str) -> str:
    if key not in opts:
        raise NoSuchDesignerError(f"Designer option '{key}' is required. Example: '{example}'.")
    v = opts[key]
    if not isinstance(v, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
    if v not in allowed:
        raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
    return v


def _reject_unknown_opts(name: str, opts: dict, allowed: set[str]) -> None:
    unknown = sorted(set(opts) - set(allowed))
    if unknown:
        raise NoSuchDesignerError(f"Designer '{name}' does not support options (got: {', '.join(unknown)}).")


def _parse_tr_core(
    opts: dict,
    *,
    name: str,
    example_prefix: str,
) -> tuple[str, str | None, int | None, str]:
    geometry = _optional_str_in(
        opts,
        "geometry",
        "box",
        {
            "enn_iso",
            "enn_metr",
            "grad_metr",
            "enn_ellip",
            "grad_ellip",
        },
        example=f"{example_prefix}/geometry=enn_metr",
    )
    covmat = None
    if "covmat" in opts:
        covmat = _optional_str_in(
            opts,
            "covmat",
            "dense",
            {"dense", "low_rank"},
            example=f"{example_prefix}/covmat=dense",
        )
    rank = None
    if "rank" in opts:
        rank = _optional_int(opts, "rank", 1, example=f"{example_prefix}/rank=10")
    update_option = _optional_str_in(
        opts,
        "update_option",
        "option_a",
        {"option_a", "option_b", "option_c"},
        example=f"{example_prefix}/update_option=option_c",
    )
    return geometry, covmat, rank, update_option


def _parse_candidate_rv(opts: dict, *, name: str, default: str) -> str:
    return _optional_str_in(
        opts,
        "candidate_rv",
        default,
        {"uniform", "sobol", "gpu_uniform"},
        example=f"{name}/candidate_rv={default}",
    )


def _parse_optional_num_candidates(opts: dict, *, name: str) -> int | None:
    if "num_candidates" not in opts:
        return None
    return _optional_int(opts, "num_candidates", 0, example=f"{name}/num_candidates=64")


def _parse_tr_shape_options(opts: dict, *, name: str) -> dict:
    return {
        "p_raasp": _optional_number(opts, "p_raasp", 0.2, example=f"{name}/p_raasp=0.2"),
        "radial_mode": _optional_str_in(
            opts,
            "radial_mode",
            "ball_uniform",
            {"ball_uniform", "boundary"},
            example=f"{name}/radial_mode=ball_uniform",
        ),
        "shape_period": _optional_int(opts, "shape_period", 5, example=f"{name}/shape_period=5"),
        "shape_ema": _optional_number(opts, "shape_ema", 0.2, example=f"{name}/shape_ema=0.2"),
        "shape_jitter": _optional_number(opts, "shape_jitter", 1e-6, example=f"{name}/shape_jitter=1e-6"),
        "shape_kappa_max": _optional_number(opts, "shape_kappa_max", 1e4, example=f"{name}/shape_kappa_max=1e4"),
        "rho_bad": _optional_number(opts, "rho_bad", 0.25, example=f"{name}/rho_bad=0.25"),
        "rho_good": _optional_number(opts, "rho_good", 0.75, example=f"{name}/rho_good=0.75"),
        "gamma_down": _optional_number(opts, "gamma_down", 0.5, example=f"{name}/gamma_down=0.5"),
        "gamma_up": _optional_number(opts, "gamma_up", 2.0, example=f"{name}/gamma_up=2.0"),
        "boundary_tol": _optional_number(opts, "boundary_tol", 0.1, example=f"{name}/boundary_tol=0.1"),
    }


def _parse_module_tr_options(opts: dict, *, name: str) -> dict:
    return {
        "enabled": bool(opts.get("module_tr", False)),
        "block_prob": _optional_number(opts, "module_tr_block_prob", 0.5, example=f"{name}/module_tr_block_prob=0.5"),
        "min_num_params": _optional_int(
            opts,
            "module_tr_min_num_params",
            10000,
            example=f"{name}/module_tr_min_num_params=10000",
        ),
    }


def _parse_accel_options(
    opts: dict,
    *,
    name: str,
    default_use_accel: bool,
) -> dict:
    return {
        "use_accel": _optional_bool(opts, "use_accel", default_use_accel, example=f"{name}/use_accel=true"),
        "accel": _optional_str_in(
            opts,
            "accel",
            "auto",
            {"auto", "triton", "jax", "mlx"},
            example=f"{name}/accel=triton",
        ),
    }


def _tr_geometry_option_specs(*, prefix: str, geometry_desc: str) -> list[DesignerOptionSpec]:
    return [
        DesignerOptionSpec(
            name="geometry",
            required=False,
            value_type="str",
            description=geometry_desc,
            example=f"{prefix}/geometry=enn_metr",
            allowed_values=(
                "enn_iso",
                "enn_metr",
                "grad_metr",
                "enn_ellip",
                "grad_ellip",
            ),
        ),
        DesignerOptionSpec(
            name="covmat",
            required=False,
            value_type="str",
            description="Optional metric/ellipsoidal covmat override.",
            example=f"{prefix}/covmat=dense",
            allowed_values=("dense", "low_rank"),
        ),
        DesignerOptionSpec(
            name="rank",
            required=False,
            value_type="int",
            description="Optional low-rank geometry rank.",
            example=f"{prefix}/covmat=low_rank/rank=10",
        ),
        DesignerOptionSpec(
            name="update_option",
            required=False,
            value_type="str",
            description="Optional ellipsoidal length-update policy (default: option_a).",
            example=f"{prefix}/update_option=option_c",
            allowed_values=("option_a", "option_b", "option_c"),
        ),
    ]


def _tr_accel_option_specs(*, prefix: str) -> list[DesignerOptionSpec]:
    return [
        DesignerOptionSpec(
            name="use_accel",
            required=False,
            value_type="bool",
            description="Whether to enable trust-region accel for non-box geometry (default: true for non-box, false for box).",
            example=f"{prefix}/use_accel=true",
        ),
        DesignerOptionSpec(
            name="accel",
            required=False,
            value_type="str",
            description="Optional trust-region accel backend override.",
            example=f"{prefix}/accel=triton",
            allowed_values=("auto", "triton", "jax", "mlx"),
        ),
    ]


def _ellipsoid_option_specs(*, prefix: str) -> list[DesignerOptionSpec]:
    return [
        DesignerOptionSpec(
            name="p_raasp",
            required=False,
            value_type="float",
            description="Optional ellipsoidal RAASP sparsity probability.",
            example=f"{prefix}/p_raasp=0.2",
        ),
        DesignerOptionSpec(
            name="radial_mode",
            required=False,
            value_type="str",
            description="Optional ellipsoidal radial sampling mode.",
            example=f"{prefix}/radial_mode=ball_uniform",
            allowed_values=("ball_uniform", "boundary"),
        ),
        DesignerOptionSpec(
            name="shape_period",
            required=False,
            value_type="int",
            description="Optional ellipsoidal geometry refresh period.",
            example=f"{prefix}/shape_period=5",
        ),
        DesignerOptionSpec(
            name="shape_ema",
            required=False,
            value_type="float",
            description="Optional ellipsoidal geometry EMA factor.",
            example=f"{prefix}/shape_ema=0.2",
        ),
        DesignerOptionSpec(
            name="shape_jitter",
            required=False,
            value_type="float",
            description="Optional SPD jitter used in ellipsoidal geometry updates.",
            example=f"{prefix}/shape_jitter=1e-6",
        ),
        DesignerOptionSpec(
            name="shape_kappa_max",
            required=False,
            value_type="float",
            description="Optional maximum condition-number cap for ellipsoidal geometry.",
            example=f"{prefix}/shape_kappa_max=1e4",
        ),
        DesignerOptionSpec(
            name="rho_bad",
            required=False,
            value_type="float",
            description="Optional option_c shrink threshold.",
            example=f"{prefix}/rho_bad=0.25",
        ),
        DesignerOptionSpec(
            name="rho_good",
            required=False,
            value_type="float",
            description="Optional option_c grow threshold.",
            example=f"{prefix}/rho_good=0.75",
        ),
        DesignerOptionSpec(
            name="gamma_down",
            required=False,
            value_type="float",
            description="Optional option_c shrink multiplier.",
            example=f"{prefix}/gamma_down=0.5",
        ),
        DesignerOptionSpec(
            name="gamma_up",
            required=False,
            value_type="float",
            description="Optional option_c grow multiplier.",
            example=f"{prefix}/gamma_up=2.0",
        ),
        DesignerOptionSpec(
            name="boundary_tol",
            required=False,
            value_type="float",
            description="Optional ellipsoidal boundary-hit tolerance.",
            example=f"{prefix}/boundary_tol=0.1",
        ),
    ]


def _build_turbo_enn_trust_region_spec(params: dict):
    TurboENNTrustRegionSpec = _load_symbol("optimizer.turbo_enn_designer_ext", "TurboENNTrustRegionSpec")
    data = dict(params)
    data["tr_type"] = "turbo" if data.get("tr_type") is None else data["tr_type"]
    data["tr_geometry"] = "box" if data.get("tr_geometry") is None else data["tr_geometry"]
    return TurboENNTrustRegionSpec(**data)


def _build_module_tr_spec(params: dict):
    ModuleTRSpec = _load_symbol("optimizer.turbo_enn_designer_ext", "ModuleTRSpec")
    return ModuleTRSpec(**params)


def _build_turbo_enn_ext_config(params: dict):
    TurboENNExtConfig = _load_symbol("optimizer.turbo_enn_designer_ext", "TurboENNExtConfig")
    return TurboENNExtConfig(**params)


def _turbo_ref(ctx: _SimpleContext, *, ard: bool, surrogate_type: str = "original"):
    TuRBORefDesigner = _load_symbol("optimizer.turbo_ref_designer", "TuRBORefDesigner")
    return TuRBORefDesigner(
        ctx.policy,
        num_init=ctx.init_yubo_default,
        ard=ard,
        surrogate_type=surrogate_type,
    )


def _turbo_enn(ctx: _SimpleContext, **kw):
    TurboENNDesigner = _load_symbol("optimizer.turbo_enn_designer", "TurboENNDesigner")
    return TurboENNDesigner(ctx.policy, **kw)


def _turbo_enn_ext(ctx: _SimpleContext, **kw):
    TurboENNDesigner = _load_symbol("optimizer.turbo_enn_designer_ext", "TurboENNDesigner")
    return TurboENNDesigner(ctx.policy, **kw)


def _mtv(ctx: _SimpleContext, *, acq_kwargs: dict):
    AcqMTV = _load_symbol("acq.acq_mtv", "AcqMTV")
    return ctx.bt(AcqMTV, init_sobol=0, acq_kwargs=acq_kwargs)


def _build_policy_ctor(ctx: _SimpleContext, module: str, name: str, **kwargs):
    Ctor = _load_symbol(module, name)
    return Ctor(ctx.policy, **kwargs)


def _build_maximin(ctx: _SimpleContext, *, toroidal: bool):
    AcqMinDist = _load_symbol("acq.acq_min_dist", "AcqMinDist")
    return ctx.bt(lambda m: AcqMinDist(m, toroidal=toroidal))


def _build_bt_acq(
    ctx: _SimpleContext,
    module: str,
    name: str,
    *,
    acq_kwargs=None,
    init_sobol=1,
    opt_sequential=False,
    start_at_max=False,
):
    Acq = _load_symbol(module, name)
    return ctx.bt(
        Acq,
        acq_kwargs=acq_kwargs,
        init_sobol=init_sobol,
        opt_sequential=opt_sequential,
        start_at_max=start_at_max,
    )


def _build_turbo_ref(ctx: _SimpleContext, kind: str):
    if kind == "turbo-1":
        return _turbo_ref(ctx, ard=True)
    if kind == "turbo-1-iso":
        return _turbo_ref(ctx, ard=False)
    return _turbo_ref(ctx, ard=True, surrogate_type="none")


def _build_turbo_enn(ctx: _SimpleContext, kind: str):
    if kind == "turbo-enn":
        return _turbo_enn(ctx, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val)
    if kind == "turbo-enn-p":
        return _turbo_enn(
            ctx,
            turbo_mode="turbo-enn",
            k=10,
            num_keep=ctx.num_keep_val,
            num_fit_samples=None,
            num_fit_candidates=None,
            acq_type="pareto",
        )
    if kind == "turbo-enn-fit-ucb":
        return _turbo_enn(
            ctx,
            turbo_mode="turbo-enn",
            k=10,
            num_keep=ctx.num_keep_val,
            num_fit_samples=100,
            num_fit_candidates=100,
            acq_type="ucb",
        )
    if kind == "turbo-zero":
        return _turbo_enn(ctx, turbo_mode="turbo-zero", num_fit_samples=None, num_fit_candidates=None)
    if kind == "turbo-one":
        return _turbo_enn(
            ctx,
            turbo_mode="turbo-one",
            num_init=ctx.init_yubo_default,
            num_fit_samples=None,
            num_fit_candidates=None,
        )
    if kind == "lhd_only":
        return _turbo_enn(ctx, turbo_mode="lhd-only", num_fit_samples=None, num_fit_candidates=None)
    if kind == "morbo-zero":
        return _turbo_enn(
            ctx,
            turbo_mode="turbo-zero",
            tr_type="morbo",
            num_fit_samples=None,
            num_fit_candidates=None,
        )
    if kind == "morbo-one":
        return _turbo_enn(
            ctx,
            turbo_mode="turbo-one",
            num_init=ctx.init_yubo_default,
            tr_type="morbo",
            num_fit_samples=None,
            num_fit_candidates=None,
        )
    return _turbo_enn(ctx, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, tr_type="morbo")


def _build_turbo_enn_py(ctx: _SimpleContext, kind: str):
    """Build TurboENNDesigner with Python backend (force Python, no Rust)."""
    if kind == "turbo_py-enn-p":
        return _turbo_enn(
            ctx,
            turbo_mode="turbo-enn",
            k=10,
            num_keep=ctx.num_keep_val,
            num_fit_samples=None,
            num_fit_candidates=None,
            acq_type="pareto",
            use_python=True,
        )
    if kind == "turbo_py-enn-fit-ucb":
        return _turbo_enn(
            ctx,
            turbo_mode="turbo-enn",
            k=10,
            num_keep=ctx.num_keep_val,
            num_fit_samples=100,
            num_fit_candidates=100,
            acq_type="ucb",
            use_python=True,
        )
    raise NoSuchDesignerError(f"Unknown turbo_py kind: {kind}")


def _build_mts(ctx: _SimpleContext, kind: str):
    MTSDesigner = _load_symbol("optimizer.mts_designer", "MTSDesigner")
    if kind == "mts":
        return MTSDesigner(
            ctx.policy,
            keep_style=ctx.keep_style,
            num_keep=ctx.num_keep,
            init_style="find",
        )
    if kind == "mts-stagger":
        return MTSDesigner(
            ctx.policy,
            keep_style=ctx.keep_style,
            num_keep=ctx.num_keep,
            init_style="find",
            use_stagger=True,
        )
    if kind == "mts-ts":
        return MTSDesigner(
            ctx.policy,
            keep_style=ctx.keep_style,
            num_keep=ctx.num_keep,
            init_style="ts",
        )
    return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="meas")


def _build_mtv_family(ctx: _SimpleContext, kind: str):
    base_kwargs_by_kind = {
        "mtv": {"sample_type": "pss"},
        "pss": {"ts_only": True, "sample_type": "pss"},
        "mtv-sts": {"sample_type": "sts", "num_refinements": 30},
        "mtv-mts": {"sample_type": "mts", "num_refinements": 30},
        "mtv-sts2": {"sample_type": "sts2", "num_refinements": 30},
        "mtv-sts-t": {
            "sample_type": "sts",
            "num_refinements": 30,
            "x_max_type": "ts_meas",
        },
        "sts": {"ts_only": True, "sample_type": "sts", "num_refinements": 30},
        "sts-ch": {
            "ts_only": True,
            "ts_chain": True,
            "sample_type": "sts",
            "num_refinements": 30,
        },
        "sts-ns": {
            "ts_only": True,
            "sample_type": "sts",
            "num_refinements": 30,
            "no_stagger": True,
        },
        "sts-ui": {
            "ts_only": True,
            "sample_type": "sts",
            "num_refinements": 30,
            "no_stagger": False,
            "x_max_type": "rand",
        },
        "sts-t": {
            "ts_only": True,
            "sample_type": "sts",
            "num_refinements": 30,
            "x_max_type": "ts_meas",
        },
        "sts-m": {
            "ts_only": True,
            "sample_type": "sts",
            "num_refinements": 30,
            "x_max_type": "meas",
        },
        "sts2": {"ts_only": True, "sample_type": "sts2", "num_refinements": 30},
    }
    base = base_kwargs_by_kind.get(kind)
    if base is None:
        raise NoSuchDesignerError(kind)
    return _mtv(ctx, acq_kwargs={"num_X_samples": ctx.default_num_X_samples} | base)


def _build_pathwise(ctx: _SimpleContext, kind: str):
    PathwiseThompsonSampling = _load_symbol("botorch.acquisition.thompson_sampling", "PathwiseThompsonSampling")
    if kind == "path":
        return ctx.bt(PathwiseThompsonSampling, init_sobol=ctx.init_yubo_default)
    if kind == "path-b":
        return ctx.bt(
            PathwiseThompsonSampling,
            init_sobol=ctx.init_yubo_default,
            num_restarts=20,
            raw_samples=100,
        )
    return ctx.bt(PathwiseThompsonSampling, init_sobol=ctx.init_yubo_default, start_at_max=True)


def _build_sobol_mc(ctx: _SimpleContext, kind: str):
    if kind == "sobol_ucb":
        return _build_bt_acq(
            ctx,
            "botorch.acquisition.monte_carlo",
            "qUpperConfidenceBound",
            init_sobol=ctx.init_ax_default,
            acq_kwargs={"beta": 1},
        )
    if kind == "sobol_ei":
        return _build_bt_acq(
            ctx,
            "botorch.acquisition.monte_carlo",
            "qNoisyExpectedImprovement",
            init_sobol=ctx.init_ax_default,
            acq_kwargs={"X_baseline": None},
        )
    return _build_bt_acq(
        ctx,
        "botorch.acquisition.max_value_entropy_search",
        "qLowerBoundMaxValueEntropy",
        init_sobol=ctx.init_ax_default,
        acq_kwargs={"candidate_set": None},
    )


_SIMPLE_BUILDERS = {
    "cma": partial(_build_policy_ctor, module="optimizer.cma_designer", name="CMAESDesigner"),
    "optuna": partial(_build_policy_ctor, module="optimizer.optuna_designer", name="OptunaDesigner"),
    "ax": partial(_build_policy_ctor, module="optimizer.ax_designer", name="AxDesigner"),
    "random": partial(_build_policy_ctor, module="optimizer.random_designer", name="RandomDesigner"),
    "sobol": partial(_build_policy_ctor, module="optimizer.sobol_designer", name="SobolDesigner"),
    "lhd": partial(_build_policy_ctor, module="optimizer.lhd_designer", name="LHDDesigner"),
    "center": partial(_build_policy_ctor, module="optimizer.center_designer", name="CenterDesigner"),
    "vecchia": lambda ctx: _load_symbol("optimizer.vecchia_designer", "VecchiaDesigner")(ctx.policy, num_candidates_per_arm=ctx.default_num_X_samples),
    "mcmcbo": lambda ctx: _load_symbol("optimizer.mcmc_bo_designer", "MCMCBODesigner")(ctx.policy, num_init=ctx.init_yubo_default),
    "maximin": partial(_build_maximin, toroidal=False),
    "maximin-toroidal": partial(_build_maximin, toroidal=True),
    "variance": partial(_build_bt_acq, module="acq.acq_var", name="AcqVar"),
    "btsobol": partial(_build_bt_acq, module="acq.acq_sobol", name="AcqSobol"),
    "ts": partial(
        _build_bt_acq,
        module="acq.acq_ts",
        name="AcqTS",
        acq_kwargs={"sampler": "cholesky", "num_candidates": 1000},
    ),
    "ts-10000": partial(
        _build_bt_acq,
        module="acq.acq_ts",
        name="AcqTS",
        acq_kwargs={"sampler": "lanczos", "num_candidates": 10000},
    ),
    "sr": partial(_build_bt_acq, module="botorch.acquisition.monte_carlo", name="qSimpleRegret"),
    "ucb": partial(
        _build_bt_acq,
        module="botorch.acquisition.monte_carlo",
        name="qUpperConfidenceBound",
        acq_kwargs={"beta": 1},
    ),
    "ei": partial(
        _build_bt_acq,
        module="botorch.acquisition.monte_carlo",
        name="qNoisyExpectedImprovement",
        acq_kwargs={"X_baseline": None},
    ),
    "lei": partial(
        _build_bt_acq,
        module="botorch.acquisition.logei",
        name="qLogNoisyExpectedImprovement",
        acq_kwargs={"X_baseline": None},
    ),
    "lei-m": partial(
        _build_bt_acq,
        module="botorch.acquisition.logei",
        name="qLogNoisyExpectedImprovement",
        acq_kwargs={"X_baseline": None},
        start_at_max=True,
    ),
    "gibbon": partial(
        _build_bt_acq,
        module="botorch.acquisition.max_value_entropy_search",
        name="qLowerBoundMaxValueEntropy",
        acq_kwargs={"candidate_set": None},
        opt_sequential=True,
    ),
    "dpp": lambda ctx: _build_bt_acq(
        ctx,
        module="acq.acq_dpp",
        name="AcqDPP",
        init_sobol=1,
        acq_kwargs={"num_X_samples": ctx.default_num_X_samples},
    ),
    "turbo-1": partial(_build_turbo_ref, kind="turbo-1"),
    "turbo-1-iso": partial(_build_turbo_ref, kind="turbo-1-iso"),
    "turbo-0": partial(_build_turbo_ref, kind="turbo-0"),
    "turbo-enn": partial(_build_turbo_enn, kind="turbo-enn"),
    "turbo-enn-p": partial(_build_turbo_enn, kind="turbo-enn-p"),
    "turbo-enn-fit-ucb": partial(_build_turbo_enn, kind="turbo-enn-fit-ucb"),
    "turbo_py-enn-p": partial(_build_turbo_enn_py, kind="turbo_py-enn-p"),
    "turbo_py-enn-fit-ucb": partial(_build_turbo_enn_py, kind="turbo_py-enn-fit-ucb"),
    "turbo-zero": partial(_build_turbo_enn, kind="turbo-zero"),
    "turbo-one": partial(_build_turbo_enn, kind="turbo-one"),
    "lhd_only": partial(_build_turbo_enn, kind="lhd_only"),
    "morbo-zero": partial(_build_turbo_enn, kind="morbo-zero"),
    "morbo-one": partial(_build_turbo_enn, kind="morbo-one"),
    "morbo-enn": partial(_build_turbo_enn, kind="morbo-enn"),
    "mts": partial(_build_mts, kind="mts"),
    "mts-stagger": partial(_build_mts, kind="mts-stagger"),
    "mts-ts": partial(_build_mts, kind="mts-ts"),
    "mts-meas": partial(_build_mts, kind="mts-meas"),
    "mtv": partial(_build_mtv_family, kind="mtv"),
    "pss": partial(_build_mtv_family, kind="pss"),
    "mtv-sts": partial(_build_mtv_family, kind="mtv-sts"),
    "mtv-mts": partial(_build_mtv_family, kind="mtv-mts"),
    "mtv-sts2": partial(_build_mtv_family, kind="mtv-sts2"),
    "mtv-sts-t": partial(_build_mtv_family, kind="mtv-sts-t"),
    "sts": partial(_build_mtv_family, kind="sts"),
    "sts-ch": partial(_build_mtv_family, kind="sts-ch"),
    "sts-ns": partial(_build_mtv_family, kind="sts-ns"),
    "sts-ui": partial(_build_mtv_family, kind="sts-ui"),
    "sts-t": partial(_build_mtv_family, kind="sts-t"),
    "sts-m": partial(_build_mtv_family, kind="sts-m"),
    "sts2": partial(_build_mtv_family, kind="sts2"),
    "path": partial(_build_pathwise, kind="path"),
    "path-b": partial(_build_pathwise, kind="path-b"),
    "path-m": partial(_build_pathwise, kind="path-m"),
    "sobol_ucb": partial(_build_sobol_mc, kind="sobol_ucb"),
    "sobol_ei": partial(_build_sobol_mc, kind="sobol_ei"),
    "sobol_gibbon": partial(_build_sobol_mc, kind="sobol_gibbon"),
}


_SIMPLE_DISPATCH = {k: v for k, v in _SIMPLE_BUILDERS.items()}


def _d_ts_sweep(ctx: _SimpleContext, opts: dict):
    num_candidates = _require_int(opts, "num_candidates", example="ts_sweep/num_candidates=10000")
    return _build_bt_acq(
        ctx,
        "acq.acq_ts",
        "AcqTS",
        acq_kwargs={"sampler": "lanczos", "num_candidates": num_candidates},
    )


def _d_rff(ctx: _SimpleContext, opts: dict):
    num_candidates = _require_int(opts, "num_candidates", example="rff/num_candidates=10000")
    return _build_bt_acq(
        ctx,
        "acq.acq_ts",
        "AcqTS",
        acq_kwargs={"sampler": "rff", "num_candidates": num_candidates},
    )


def _d_pss_sweep_kmcmc(ctx: _SimpleContext, opts: dict):
    k_mcmc = _require_int(opts, "k_mcmc", example="pss_sweep_kmcmc/k_mcmc=8")
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "num_X_samples": ctx.default_num_X_samples,
            "sample_type": "pss",
            "k_mcmc": k_mcmc,
        },
    )


def _d_pss_sweep_num_mcmc(ctx: _SimpleContext, opts: dict):
    num_mcmc = _require_int(opts, "num_mcmc", example="pss_sweep_num_mcmc/num_mcmc=16")
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "num_X_samples": ctx.default_num_X_samples,
            "sample_type": "pss",
            "k_mcmc": None,
            "num_mcmc": num_mcmc,
        },
    )


def _d_sts_sweep(ctx: _SimpleContext, opts: dict):
    num_refinements = _require_int(opts, "num_refinements", example="sts_sweep/num_refinements=30")
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": num_refinements,
        },
    )


def _d_turbo_enn_sweep(ctx: _SimpleContext, opts: dict):
    k = _require_int(opts, "k", example="turbo-enn-sweep/k=10")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=k,
        num_keep=None,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
    )


def _build_turbo_enn_p_with_opts(
    ctx: _SimpleContext,
    opts: dict,
    *,
    name: str,
    use_python: bool = False,
):
    _reject_unknown_opts(
        name,
        opts,
        {"k", "candidate_rv", "geometry", "covmat", "rank", "update_option", "use_accel", "accel"},
    )
    k = _optional_int(opts, "k", 10, example=f"{name}/k=16")
    candidate_rv = _parse_candidate_rv(opts, name=name, default="uniform")
    geometry, covmat, rank, update_option = _parse_tr_core(
        opts,
        name=name,
        example_prefix=name,
    )
    shape = _parse_tr_shape_options(opts, name=name)
    accel = _parse_accel_options(opts, name=name, default_use_accel=(geometry != "box"))
    trust_region = _build_turbo_enn_trust_region_spec(
        {
            "tr_type": None,
            "tr_geometry": geometry,
            "covmat": covmat,
            "metric_rank": rank,
            "fixed_length": None,
            "update_option": update_option,
            **shape,
            "use_accel": accel["use_accel"],
            "accel": None if accel["accel"] == "auto" else accel["accel"],
        }
    )
    module_tr = _build_module_tr_spec({"enabled": False, "block_prob": 0.5, "min_num_params": 10000})
    config = _build_turbo_enn_ext_config(
        {
            "turbo_mode": "turbo-enn",
            "num_init": None,
            "k": k,
            "num_keep": ctx.num_keep_val,
            "num_fit_samples": None,
            "num_fit_candidates": None,
            "acq_type": "pareto",
            "trust_region": trust_region,
            "use_y_var": False,
            "num_candidates": None,
            "candidate_rv": candidate_rv,
            "num_metrics": None,
            "use_python": use_python,
            "module_tr": module_tr,
            "rng": None,
        }
    )
    return _turbo_enn_ext(ctx, config=config)


def _d_turbo_enn_p(ctx: _SimpleContext, opts: dict):
    return _build_turbo_enn_p_with_opts(ctx, opts, name="turbo-enn-p")


def _d_turbo_py_enn_p(ctx: _SimpleContext, opts: dict):
    return _build_turbo_enn_p_with_opts(ctx, opts, name="turbo_py-enn-p", use_python=True)


def _d_turbo_enn_fit(ctx: _SimpleContext, opts: dict):
    _reject_unknown_opts(
        "turbo-enn-fit",
        opts,
        {
            "acq_type",
            "k",
            "candidate_rv",
            "geometry",
            "covmat",
            "rank",
            "update_option",
            "p_raasp",
            "radial_mode",
            "shape_period",
            "shape_ema",
            "shape_jitter",
            "shape_kappa_max",
            "rho_bad",
            "rho_good",
            "gamma_down",
            "gamma_up",
            "boundary_tol",
            "num_candidates",
            "num_fit_samples",
            "num_fit_candidates",
            "fixed_length",
            "use_accel",
            "accel",
            "module_tr",
            "module_tr_block_prob",
            "module_tr_min_num_params",
        },
    )
    acq_type = _require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="turbo-enn-fit/acq_type=ucb",
    )
    k = _optional_int(opts, "k", 10, example="turbo-enn-fit/k=16")
    candidate_rv = _parse_candidate_rv(opts, name="turbo-enn-fit", default="sobol")
    geometry, covmat, rank, update_option = _parse_tr_core(
        opts,
        name="turbo-enn-fit",
        example_prefix="turbo-enn-fit",
    )
    shape = _parse_tr_shape_options(opts, name="turbo-enn-fit")
    accel = _parse_accel_options(opts, name="turbo-enn-fit", default_use_accel=(geometry != "box"))
    num_candidates = _parse_optional_num_candidates(opts, name="turbo-enn-fit")
    num_fit_samples = _optional_int(
        opts,
        "num_fit_samples",
        100,
        example="turbo-enn-fit/num_fit_samples=10",
    )
    num_fit_candidates = _optional_int(
        opts,
        "num_fit_candidates",
        100,
        example="turbo-enn-fit/num_fit_candidates=100",
    )
    fixed_length = _optional_number(
        opts,
        "fixed_length",
        None,
        example="turbo-enn-fit/fixed_length=1.6",
    )
    module = _parse_module_tr_options(opts, name="turbo-enn-fit")
    trust_region = _build_turbo_enn_trust_region_spec(
        {
            "tr_type": None,
            "tr_geometry": geometry,
            "covmat": covmat,
            "metric_rank": rank,
            "fixed_length": fixed_length,
            "update_option": update_option,
            **shape,
            "use_accel": accel["use_accel"],
            "accel": None if accel["accel"] == "auto" else accel["accel"],
        }
    )
    module = _build_module_tr_spec(module)
    config = _build_turbo_enn_ext_config(
        {
            "turbo_mode": "turbo-enn",
            "num_init": None,
            "k": k,
            "num_keep": ctx.num_keep_val,
            "num_fit_samples": num_fit_samples,
            "num_fit_candidates": num_fit_candidates,
            "acq_type": acq_type,
            "trust_region": trust_region,
            "use_y_var": False,
            "num_candidates": num_candidates,
            "candidate_rv": candidate_rv,
            "num_metrics": None,
            "use_python": False,
            "module_tr": module,
            "rng": None,
        }
    )
    return _turbo_enn_ext(ctx, config=config)


def _d_turbo_enn_f(ctx: _SimpleContext, opts: dict):
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer 'turbo-enn-f' does not support options (got: {keys}).")

    def _num_candidates(num_dim, num_arms):
        return 100 * num_arms

    TurboENNDesigner = _load_symbol("optimizer.turbo_enn_designer", "TurboENNDesigner")
    return TurboENNDesigner(
        ctx.policy,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100,
        acq_type="ucb",
        num_candidates=_num_candidates,
        candidate_rv="uniform",
    )


def _d_turbo_enn_f_p(ctx: _SimpleContext, opts: dict):
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer 'turbo-enn-f-p' does not support options (got: {keys}).")

    def _num_candidates(num_dim, num_arms):
        return 100 * num_arms

    TurboENNDesigner = _load_symbol("optimizer.turbo_enn_designer", "TurboENNDesigner")
    return TurboENNDesigner(
        ctx.policy,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100,
        acq_type="pareto",
        num_candidates=_num_candidates,
        candidate_rv="uniform",
    )


def _d_morbo_enn_fit(ctx: _SimpleContext, opts: dict):
    _reject_unknown_opts("morbo-enn-fit", opts, {"acq_type", "k", "candidate_rv"})
    acq_type = _require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="morbo-enn-fit/acq_type=ucb",
    )
    k = _optional_int(opts, "k", 10, example="morbo-enn-fit/k=16")
    candidate_rv = _optional_str_in(
        opts,
        "candidate_rv",
        "uniform",
        {"uniform", "sobol", "gpu_uniform"},
        example="morbo-enn-fit/candidate_rv=uniform",
    )
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=k,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100 * ctx.num_arms,
        acq_type=acq_type,
        tr_type="morbo",
        candidate_rv=candidate_rv,
    )


def _turbo_enn_multi(ctx: _SimpleContext, **kw):
    MultiTurboHarnessConfig = _load_symbol("optimizer.multi_turbo_enn_designer", "MultiTurboHarnessConfig")
    MultiTurboENNConfig = _load_symbol("optimizer.multi_turbo_enn_designer", "MultiTurboENNConfig")
    MultiTurboENNDesigner = _load_symbol("optimizer.multi_turbo_enn_designer", "MultiTurboENNDesigner")
    data = dict(kw)
    if "turbo_mode" not in data:
        raise NoSuchDesignerError("Designer option 'turbo_mode' is required.")
    harness = MultiTurboHarnessConfig(
        num_regions=data.pop("num_regions", 2),
        strategy=data.pop("strategy", "independent"),
        arm_mode=data.pop("arm_mode", "split"),
        pool_multiplier=data.pop("pool_multiplier", 2),
    )
    trust_region = _build_turbo_enn_trust_region_spec(
        {
            "tr_type": data.pop("tr_type", None),
            "tr_geometry": data.pop("tr_geometry", None),
            "covmat": data.pop("covmat", None),
            "metric_rank": data.pop("metric_rank", None),
            "fixed_length": data.pop("tr_length_fixed", None),
            "update_option": data.pop("update_option", "option_a"),
            "p_raasp": data.pop("p_raasp", 0.2),
            "radial_mode": data.pop("radial_mode", "ball_uniform"),
            "shape_period": data.pop("shape_period", 5),
            "shape_ema": data.pop("shape_ema", 0.2),
            "shape_jitter": data.pop("shape_jitter", 1e-6),
            "shape_kappa_max": data.pop("shape_kappa_max", 1e4),
            "rho_bad": data.pop("rho_bad", 0.25),
            "rho_good": data.pop("rho_good", 0.75),
            "gamma_down": data.pop("gamma_down", 0.5),
            "gamma_up": data.pop("gamma_up", 2.0),
            "boundary_tol": data.pop("boundary_tol", 0.1),
            "use_accel": data.pop("use_accel", False),
            "accel": data.pop("accel", None),
        }
    )
    module_tr = _build_module_tr_spec(
        {
            "enabled": data.pop("module_tr", False),
            "block_prob": data.pop("module_tr_block_prob", 0.5),
            "min_num_params": data.pop("module_tr_min_num_params", 10000),
        }
    )
    region = _build_turbo_enn_ext_config(
        {
            "turbo_mode": data.pop("turbo_mode"),
            "num_init": data.pop("num_init", None),
            "k": data.pop("k", None),
            "num_keep": data.pop("num_keep", None),
            "num_fit_samples": data.pop("num_fit_samples", None),
            "num_fit_candidates": data.pop("num_fit_candidates", None),
            "acq_type": data.pop("acq_type", "pareto"),
            "trust_region": trust_region,
            "use_y_var": data.pop("use_y_var", False),
            "num_candidates": data.pop("num_candidates", None),
            "candidate_rv": data.pop("candidate_rv", None),
            "num_metrics": data.pop("num_metrics", None),
            "use_python": data.pop("use_python", False),
            "module_tr": module_tr,
            "rng": data.pop("rng", None),
        }
    )
    if data:
        keys = ", ".join(sorted(data.keys()))
        raise NoSuchDesignerError(f"Unknown designer option(s): {keys}")
    config = MultiTurboENNConfig(harness=harness, region=region)
    return MultiTurboENNDesigner(ctx.policy, config=config)


def _d_turbo_enn_multi_ext(ctx: _SimpleContext, opts: dict):
    acq_type = _require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="turbo-enn-multi/acq_type=ucb",
    )
    num_regions = _require_int(opts, "num_regions", example="turbo-enn-multi/num_regions=4")
    strategy = _optional_str_in(
        opts,
        "strategy",
        "independent",
        {"independent", "shared_data"},
        example="turbo-enn-multi/strategy=independent",
    )
    arm_mode = _optional_str_in(
        opts,
        "arm_mode",
        "split",
        {"split", "per_region", "allocated"},
        example="turbo-enn-multi/arm_mode=allocated",
    )
    pool_multiplier = _optional_int(opts, "pool_multiplier", 2, example="turbo-enn-multi/pool_multiplier=2")
    candidate_rv = _parse_candidate_rv(opts, name="turbo-enn-multi", default="sobol")
    num_candidates = _parse_optional_num_candidates(opts, name="turbo-enn-multi")
    num_fit_samples = _optional_int(opts, "num_fit_samples", 100, example="turbo-enn-multi/num_fit_samples=50")
    num_fit_candidates = _optional_int(
        opts,
        "num_fit_candidates",
        100,
        example="turbo-enn-multi/num_fit_candidates=50",
    )
    geometry, covmat, rank, update_option = _parse_tr_core(
        opts,
        name="turbo-enn-multi",
        example_prefix="turbo-enn-multi",
    )
    tr_length_fixed = _optional_number(opts, "tr_length_fixed", None, example="turbo-enn-multi/tr_length_fixed=1.6")
    shape = _parse_tr_shape_options(opts, name="turbo-enn-multi")
    accel = _parse_accel_options(opts, name="turbo-enn-multi", default_use_accel=(geometry != "box"))
    return _turbo_enn_multi(
        ctx,
        turbo_mode="turbo-enn",
        num_regions=num_regions,
        strategy=strategy,
        arm_mode=arm_mode,
        pool_multiplier=pool_multiplier,
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=num_fit_samples,
        num_fit_candidates=num_fit_candidates,
        acq_type=acq_type,
        tr_type="turbo",
        tr_geometry=geometry,
        covmat=covmat,
        metric_rank=rank,
        tr_length_fixed=tr_length_fixed,
        update_option=update_option,
        p_raasp=shape["p_raasp"],
        radial_mode=shape["radial_mode"],
        shape_period=shape["shape_period"],
        shape_ema=shape["shape_ema"],
        shape_jitter=shape["shape_jitter"],
        shape_kappa_max=shape["shape_kappa_max"],
        rho_bad=shape["rho_bad"],
        rho_good=shape["rho_good"],
        gamma_down=shape["gamma_down"],
        gamma_up=shape["gamma_up"],
        boundary_tol=shape["boundary_tol"],
        use_accel=accel["use_accel"],
        accel=None if accel["accel"] == "auto" else accel["accel"],
        candidate_rv=candidate_rv,
        num_candidates=num_candidates,
    )


def _d_sts_ar(ctx: _SimpleContext, opts: dict):
    num_acc_rej = _require_int(opts, "num_acc_rej", example="sts-ar/num_acc_rej=10")
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 0,
            "num_acc_rej": num_acc_rej,
        },
    )


def _ppo(ctx: _SimpleContext, opts: dict):
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer 'ppo' does not support options (got: {keys}).")
    if ctx.env_conf is None:
        raise NoSuchDesignerError("Designer 'ppo' requires env_conf to be set.")
    PPODesigner = _load_symbol("optimizer.ppo_designer", "PPODesigner")
    return PPODesigner(ctx.policy, ctx.env_conf)


_DESIGNER_OPTION_SPECS: dict[str, list[DesignerOptionSpec]] = {
    "ts_sweep": [
        DesignerOptionSpec(
            name="num_candidates",
            required=True,
            value_type="int",
            description="Number of TS candidates (lanczos sampler).",
            example="ts_sweep/num_candidates=10000",
        )
    ],
    "rff": [
        DesignerOptionSpec(
            name="num_candidates",
            required=True,
            value_type="int",
            description="Number of TS candidates (RFF sampler).",
            example="rff/num_candidates=10000",
        )
    ],
    "pss_sweep_kmcmc": [
        DesignerOptionSpec(
            name="k_mcmc",
            required=True,
            value_type="int",
            description="Number of MCMC chains per refinement (PSS).",
            example="pss_sweep_kmcmc/k_mcmc=8",
        )
    ],
    "pss_sweep_num_mcmc": [
        DesignerOptionSpec(
            name="num_mcmc",
            required=True,
            value_type="int",
            description="Total number of MCMC samples per refinement (PSS).",
            example="pss_sweep_num_mcmc/num_mcmc=16",
        )
    ],
    "sts_sweep": [
        DesignerOptionSpec(
            name="num_refinements",
            required=True,
            value_type="int",
            description="Number of STS refinements.",
            example="sts_sweep/num_refinements=30",
        )
    ],
    "turbo-enn-sweep": [
        DesignerOptionSpec(
            name="k",
            required=True,
            value_type="int",
            description="Ensemble size for TuRBO-ENN sweep.",
            example="turbo-enn-sweep/k=10",
        )
    ],
    "turbo-enn-p": [
        DesignerOptionSpec(
            name="k",
            required=False,
            value_type="int",
            description="Optional ENN neighborhood size override (default: 10).",
            example="turbo-enn-p/k=16",
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Optional candidate sampler override (default: uniform).",
            example="turbo-enn-p/candidate_rv=uniform",
            allowed_values=("uniform", "sobol", "gpu_uniform"),
        ),
    ]
    + _tr_geometry_option_specs(
        prefix="turbo-enn-p",
        geometry_desc="Optional trust-region geometry override for no-fit Pareto ENN (default: box when omitted).",
    )
    + _tr_accel_option_specs(prefix="turbo-enn-p"),
    "turbo_py-enn-p": [
        DesignerOptionSpec(
            name="k",
            required=False,
            value_type="int",
            description="Optional ENN neighborhood size override (default: 10).",
            example="turbo_py-enn-p/k=16",
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Optional candidate sampler override (default: uniform).",
            example="turbo_py-enn-p/candidate_rv=uniform",
            allowed_values=("uniform", "sobol", "gpu_uniform"),
        ),
    ]
    + _tr_geometry_option_specs(
        prefix="turbo_py-enn-p",
        geometry_desc="Optional trust-region geometry override for Python no-fit Pareto ENN (default: box when omitted).",
    )
    + _tr_accel_option_specs(prefix="turbo_py-enn-p"),
    "turbo-enn-fit": [
        DesignerOptionSpec(
            name="acq_type",
            required=True,
            value_type="str",
            description="Acquisition type for fit-time candidate generation.",
            example="turbo-enn-fit/acq_type=ucb",
            allowed_values=("pareto", "thompson", "ucb"),
        ),
        DesignerOptionSpec(
            name="k",
            required=False,
            value_type="int",
            description="Optional ENN neighborhood size override (default: 10).",
            example="turbo-enn-fit/acq_type=ucb/k=16",
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Optional candidate sampler override (default: sobol).",
            example="turbo-enn-fit/acq_type=ucb/candidate_rv=sobol",
            allowed_values=("uniform", "sobol", "gpu_uniform"),
        ),
        DesignerOptionSpec(
            name="num_candidates",
            required=False,
            value_type="int",
            description="Optional candidate count override for fit-time generation.",
            example="turbo-enn-fit/acq_type=ucb/num_candidates=64",
        ),
        DesignerOptionSpec(
            name="num_fit_samples",
            required=False,
            value_type="int",
            description="Optional ENN fit-sample count override (default: 100).",
            example="turbo-enn-fit/acq_type=ucb/num_fit_samples=10",
        ),
        DesignerOptionSpec(
            name="num_fit_candidates",
            required=False,
            value_type="int",
            description="Optional ENN fit-candidate count override (default: 100).",
            example="turbo-enn-fit/acq_type=ucb/num_fit_candidates=100",
        ),
    ]
    + _tr_geometry_option_specs(
        prefix="turbo-enn-fit/acq_type=ucb",
        geometry_desc="Optional trust-region geometry override (default: box when omitted).",
    )
    + _ellipsoid_option_specs(prefix="turbo-enn-fit/acq_type=ucb")
    + [
        DesignerOptionSpec(
            name="fixed_length",
            required=False,
            value_type="float",
            description="Optional fixed trust-region length override.",
            example="turbo-enn-fit/acq_type=ucb/fixed_length=1.6",
        ),
    ]
    + _tr_accel_option_specs(prefix="turbo-enn-fit/acq_type=ucb"),
    "turbo-enn-multi": [
        DesignerOptionSpec(
            name="acq_type",
            required=True,
            value_type="str",
            description="Acquisition type for multi-region selection.",
            example="turbo-enn-multi/acq_type=ucb",
            allowed_values=("pareto", "thompson", "ucb"),
        ),
        DesignerOptionSpec(
            name="num_regions",
            required=True,
            value_type="int",
            description="Number of trust regions.",
            example="turbo-enn-multi/num_regions=4",
        ),
        DesignerOptionSpec(
            name="strategy",
            required=False,
            value_type="str",
            description="Data sharing strategy across regions.",
            example="turbo-enn-multi/strategy=independent",
            allowed_values=("independent", "shared_data"),
        ),
        DesignerOptionSpec(
            name="arm_mode",
            required=False,
            value_type="str",
            description="Arm allocation mode across regions.",
            example="turbo-enn-multi/arm_mode=allocated",
            allowed_values=("split", "per_region", "allocated"),
        ),
        DesignerOptionSpec(
            name="pool_multiplier",
            required=False,
            value_type="int",
            description="Proposal pool scaling when arm_mode=allocated.",
            example="turbo-enn-multi/pool_multiplier=2",
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Candidate random variable family.",
            example="turbo-enn-multi/candidate_rv=uniform",
            allowed_values=("sobol", "uniform", "gpu_uniform"),
        ),
        DesignerOptionSpec(
            name="num_candidates",
            required=False,
            value_type="int",
            description="Number of trust-region candidates.",
            example="turbo-enn-multi/num_candidates=64",
        ),
        DesignerOptionSpec(
            name="num_fit_samples",
            required=False,
            value_type="int",
            description="ENN fit samples for surrogate update.",
            example="turbo-enn-multi/num_fit_samples=50",
        ),
        DesignerOptionSpec(
            name="num_fit_candidates",
            required=False,
            value_type="int",
            description="ENN fit candidates per surrogate step.",
            example="turbo-enn-multi/num_fit_candidates=50",
        ),
    ]
    + [
        DesignerOptionSpec(
            name="tr_length_fixed",
            required=False,
            value_type="float",
            description="Fixed trust-region length.",
            example="turbo-enn-multi/tr_length_fixed=1.6",
        ),
    ]
    + _tr_geometry_option_specs(
        prefix="turbo-enn-multi",
        geometry_desc="Trust-region geometry (default: box when omitted).",
    )
    + _ellipsoid_option_specs(prefix="turbo-enn-multi")
    + _tr_accel_option_specs(prefix="turbo-enn-multi"),
    "morbo-enn-fit": [
        DesignerOptionSpec(
            name="acq_type",
            required=True,
            value_type="str",
            description="Acquisition type for fit-time candidate generation (MORBO TR).",
            example="morbo-enn-fit/acq_type=ucb",
            allowed_values=("pareto", "thompson", "ucb"),
        ),
        DesignerOptionSpec(
            name="k",
            required=False,
            value_type="int",
            description="Optional ENN neighborhood size override (default: 10).",
            example="morbo-enn-fit/acq_type=ucb/k=16",
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Optional candidate sampler override (default: uniform).",
            example="morbo-enn-fit/acq_type=ucb/candidate_rv=uniform",
            allowed_values=("uniform", "sobol", "gpu_uniform"),
        ),
    ],
    "sts-ar": [
        DesignerOptionSpec(
            name="num_acc_rej",
            required=True,
            value_type="int",
            description="Number of accept/reject steps.",
            example="sts-ar/num_acc_rej=10",
        )
    ],
}


_DESIGNER_DISPATCH = {name: partial(_no_opts, name, builder) for name, builder in _SIMPLE_BUILDERS.items()} | {
    "ts_sweep": _d_ts_sweep,
    "rff": _d_rff,
    "pss_sweep_kmcmc": _d_pss_sweep_kmcmc,
    "pss_sweep_num_mcmc": _d_pss_sweep_num_mcmc,
    "sts_sweep": _d_sts_sweep,
    "turbo-enn-sweep": _d_turbo_enn_sweep,
    "turbo-enn-p": _d_turbo_enn_p,
    "turbo_py-enn-p": _d_turbo_py_enn_p,
    "turbo-enn-fit": _d_turbo_enn_fit,
    "turbo-enn-multi": _d_turbo_enn_multi_ext,
    "turbo-enn-f": _d_turbo_enn_f,
    "turbo-enn-f-p": _d_turbo_enn_f_p,
    "morbo-enn-fit": _d_morbo_enn_fit,
    "sts-ar": _d_sts_ar,
    "ppo": _ppo,
}
