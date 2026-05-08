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


def _require_str_in(opts: dict, key: str, allowed: set[str], *, example: str) -> str:
    if key not in opts:
        raise NoSuchDesignerError(f"Designer option '{key}' is required. Example: '{example}'.")
    v = opts[key]
    if not isinstance(v, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
    if v not in allowed:
        raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
    return v


def _pop_optional_int(opts: dict, key: str, default: int | None, *, example: str) -> int | None:
    if key not in opts:
        return default
    v = opts.pop(key)
    if not isinstance(v, int) or isinstance(v, bool):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int. Example: '{example}'.")
    return v


def _pop_optional_float(opts: dict, key: str, default: float, *, example: str) -> float:
    if key not in opts:
        return default
    v = opts.pop(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a float. Example: '{example}'.")
    return float(v)


def _pop_optional_str_in(opts: dict, key: str, default: str | None, allowed: set[str], *, example: str) -> str | None:
    if key not in opts:
        return default
    v = opts.pop(key)
    if not isinstance(v, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string. Example: '{example}'.")
    if v not in allowed:
        raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
    return v


def _turbo_ref(ctx: _SimpleContext, *, ard: bool, surrogate_type: str = "original"):
    TuRBORefDesigner = _load_symbol("optimizer.turbo_ref_designer", "TuRBORefDesigner")
    return TuRBORefDesigner(
        ctx.policy,
        num_init=ctx.init_yubo_default,
        ard=ard,
        surrogate_type=surrogate_type,
    )


_EGGROLL_TURBO_OPTION_KEYS = {
    "steps_per_episode",
    "eval_episodes",
    "deterministic_policy",
    "param_scale",
    "seed_offset",
}


def _is_eggroll_jax_context(ctx: _SimpleContext) -> bool:
    if ctx.env_conf is None:
        return False
    env_name = str(getattr(ctx.env_conf, "env_name", ""))
    try:
        from problems.eggroll_env_adapters import supports_eggroll_env_adapter
    except ImportError:
        return False
    return supports_eggroll_env_adapter(env_name) and hasattr(ctx.policy, "model_cls") and hasattr(ctx.policy, "params")


def _split_eggroll_turbo_options(name: str, ctx: _SimpleContext, opts: dict | None) -> tuple[dict, dict]:
    opts = dict(opts or {})
    eggroll_opts = {key: opts.pop(key) for key in list(opts) if key in _EGGROLL_TURBO_OPTION_KEYS}
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer '{name}' does not support options (got: {keys}).")
    if eggroll_opts and not _is_eggroll_jax_context(ctx):
        keys = ", ".join(sorted(eggroll_opts))
        raise NoSuchDesignerError(f"Designer '{name}' options are EggRoll-only for JAX envs/policies (got: {keys}).")
    return opts, eggroll_opts


def _turbo_enn(ctx: _SimpleContext, *, opts: dict | None = None, designer_name: str = "turbo-enn", **kw):
    _unused, eggroll_opts = _split_eggroll_turbo_options(designer_name, ctx, opts)
    if _is_eggroll_jax_context(ctx):
        EggRollJAXVectorDesigner = _load_symbol("optimizer.eggroll_vector_designer", "EggRollJAXVectorDesigner")
        return EggRollJAXVectorDesigner(ctx.policy, ctx.env_conf, **kw, **eggroll_opts)
    TurboENNDesigner = _load_symbol("optimizer.turbo_enn_designer", "TurboENNDesigner")
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


def _build_turbo_enn(ctx: _SimpleContext, kind: str, opts: dict | None = None):
    if kind == "turbo-enn":
        return _turbo_enn(ctx, opts=opts, designer_name=kind, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val)
    if kind == "turbo-enn-p":
        return _turbo_enn(
            ctx,
            opts=opts,
            designer_name=kind,
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
            opts=opts,
            designer_name=kind,
            turbo_mode="turbo-enn",
            k=10,
            num_keep=ctx.num_keep_val,
            num_fit_samples=100,
            num_fit_candidates=100,
            acq_type="ucb",
        )
    if kind == "turbo-zero":
        return _turbo_enn(ctx, opts=opts, designer_name=kind, turbo_mode="turbo-zero", num_fit_samples=None, num_fit_candidates=None)
    if kind == "turbo-one":
        return _turbo_enn(
            ctx,
            opts=opts,
            designer_name=kind,
            turbo_mode="turbo-one",
            num_init=ctx.init_yubo_default,
            num_fit_samples=None,
            num_fit_candidates=None,
        )
    if kind == "lhd_only":
        return _turbo_enn(ctx, opts=opts, designer_name=kind, turbo_mode="lhd-only", num_fit_samples=None, num_fit_candidates=None)
    if kind == "morbo-zero":
        return _turbo_enn(
            ctx,
            opts=opts,
            designer_name=kind,
            turbo_mode="turbo-zero",
            tr_type="morbo",
            num_fit_samples=None,
            num_fit_candidates=None,
        )
    if kind == "morbo-one":
        return _turbo_enn(
            ctx,
            opts=opts,
            designer_name=kind,
            turbo_mode="turbo-one",
            num_init=ctx.init_yubo_default,
            tr_type="morbo",
            num_fit_samples=None,
            num_fit_candidates=None,
        )
    return _turbo_enn(ctx, opts=opts, designer_name=kind, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, tr_type="morbo")


def _build_turbo_enn_py(ctx: _SimpleContext, kind: str, opts: dict | None = None):
    """Build TurboENNDesigner with Python backend (force Python, no Rust)."""
    if kind == "turbo_py-enn-p":
        return _turbo_enn(
            ctx,
            opts=opts,
            designer_name=kind,
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
            opts=opts,
            designer_name=kind,
            turbo_mode="turbo-enn",
            k=10,
            num_keep=ctx.num_keep_val,
            num_fit_samples=100,
            num_fit_candidates=100,
            acq_type="ucb",
            use_python=True,
        )
    raise NoSuchDesignerError(f"Unknown turbo_py kind: {kind}")


def _build_sparse_enn(ctx: _SimpleContext, opts: dict | None = None):
    opts = dict(opts or {})
    clock_scale = _pop_optional_float(opts, "clock_scale", 3.0, example="sparse-enn/clock_scale=3.0")
    min_failures = _pop_optional_float(opts, "min_failures", 4.0, example="sparse-enn/min_failures=4")
    num_pert = _pop_optional_int(opts, "num_pert", 20, example="sparse-enn/num_pert=20")
    k = _pop_optional_int(opts, "k", 10, example="sparse-enn/k=10")
    num_init = _pop_optional_int(opts, "num_init", None, example="sparse-enn/num_init=20")
    num_candidates = _pop_optional_int(opts, "num_candidates", None, example="sparse-enn/num_candidates=1000")
    num_fit_samples = _pop_optional_int(opts, "num_fit_samples", None, example="sparse-enn/num_fit_samples=100")
    num_fit_candidates = _pop_optional_int(opts, "num_fit_candidates", None, example="sparse-enn/num_fit_candidates=100")
    acq_type = _pop_optional_str_in(
        opts,
        "acq_type",
        "pareto",
        {"pareto", "thompson", "ucb"},
        example="sparse-enn/acq_type=ucb",
    )
    candidate_rv = _pop_optional_str_in(
        opts,
        "candidate_rv",
        None,
        {"sobol", "uniform", "gpu_uniform"},
        example="sparse-enn/candidate_rv=uniform",
    )
    if acq_type != "pareto":
        if num_fit_samples is None:
            num_fit_samples = 100
        if num_fit_candidates is None:
            num_fit_candidates = 100
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer 'sparse-enn' does not support options (got: {keys}).")
    SparseENNDesigner = _load_symbol("optimizer.sparse_enn_designer", "SparseENNDesigner")
    return SparseENNDesigner(
        ctx.policy,
        clock_scale=clock_scale,
        num_pert=num_pert,
        min_failures=min_failures,
        num_init=num_init,
        k=k,
        num_keep=ctx.num_keep_val,
        num_fit_samples=num_fit_samples,
        num_fit_candidates=num_fit_candidates,
        acq_type=acq_type,
        num_candidates=num_candidates,
        candidate_rv=candidate_rv,
    )


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
    opts = dict(opts)
    k = _require_int(opts, "k", example="turbo-enn-sweep/k=10")
    opts.pop("k", None)
    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="turbo-enn-sweep",
        turbo_mode="turbo-enn",
        k=k,
        num_keep=None,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
    )


def _d_turbo_enn_fit(ctx: _SimpleContext, opts: dict):
    opts = dict(opts)
    acq_type = _require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="turbo-enn-fit/acq_type=ucb",
    )
    opts.pop("acq_type", None)
    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="turbo-enn-fit",
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100,
        acq_type=acq_type,
        tr_type=None,
    )


def _d_turbo_enn_f(ctx: _SimpleContext, opts: dict):
    def _num_candidates(num_dim, num_arms):
        return 100 * num_arms

    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="turbo-enn-f",
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
    def _num_candidates(num_dim, num_arms):
        return 100 * num_arms

    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="turbo-enn-f-p",
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
    opts = dict(opts)
    acq_type = _require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="morbo-enn-fit/acq_type=ucb",
    )
    opts.pop("acq_type", None)
    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="morbo-enn-fit",
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100 * ctx.num_arms,
        acq_type=acq_type,
        tr_type="morbo",
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


def _eggroll(ctx: _SimpleContext, opts: dict):
    if ctx.env_conf is None:
        raise NoSuchDesignerError("Designer 'eggroll' requires env_conf to be set.")
    EggRollDesigner = _load_symbol("optimizer.eggroll_designer", "EggRollDesigner")
    return EggRollDesigner(ctx.policy, ctx.env_conf, **opts)


def _d_turbo_enn_simple(ctx: _SimpleContext, opts: dict, *, kind: str):
    return _build_turbo_enn(ctx, kind, opts=opts)


def _d_turbo_enn_py_simple(ctx: _SimpleContext, opts: dict, *, kind: str):
    return _build_turbo_enn_py(ctx, kind, opts=opts)


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
    "turbo-enn-fit": [
        DesignerOptionSpec(
            name="acq_type",
            required=True,
            value_type="str",
            description="Acquisition type for fit-time candidate generation.",
            example="turbo-enn-fit/acq_type=ucb",
            allowed_values=("pareto", "thompson", "ucb"),
        )
    ],
    "morbo-enn-fit": [
        DesignerOptionSpec(
            name="acq_type",
            required=True,
            value_type="str",
            description="Acquisition type for fit-time candidate generation (MORBO TR).",
            example="morbo-enn-fit/acq_type=ucb",
            allowed_values=("pareto", "thompson", "ucb"),
        )
    ],
    "sparse-enn": [
        DesignerOptionSpec(
            name="clock_scale",
            required=False,
            value_type="float",
            description="Multiplier on the expected sparse proposal support for the failure clock.",
            example="sparse-enn/clock_scale=3.0",
        ),
        DesignerOptionSpec(
            name="min_failures",
            required=False,
            value_type="float",
            description="Lower bound on the batch-level failure tolerance.",
            example="sparse-enn/min_failures=4",
        ),
        DesignerOptionSpec(
            name="num_pert",
            required=False,
            value_type="int",
            description="Target number of perturbed coordinates in sparse proposals.",
            example="sparse-enn/num_pert=20",
        ),
        DesignerOptionSpec(
            name="k",
            required=False,
            value_type="int",
            description="ENN neighborhood or ensemble-size parameter.",
            example="sparse-enn/k=10",
        ),
        DesignerOptionSpec(
            name="num_init",
            required=False,
            value_type="int",
            description="Number of initialization points.",
            example="sparse-enn/num_init=20",
        ),
        DesignerOptionSpec(
            name="num_candidates",
            required=False,
            value_type="int",
            description="Number of trust-region candidates.",
            example="sparse-enn/num_candidates=1000",
        ),
        DesignerOptionSpec(
            name="num_fit_samples",
            required=False,
            value_type="int",
            description="Number of ENN fit samples.",
            example="sparse-enn/num_fit_samples=100",
        ),
        DesignerOptionSpec(
            name="num_fit_candidates",
            required=False,
            value_type="int",
            description="Number of ENN fit candidates.",
            example="sparse-enn/num_fit_candidates=100",
        ),
        DesignerOptionSpec(
            name="acq_type",
            required=False,
            value_type="str",
            description="Acquisition type.",
            example="sparse-enn/acq_type=ucb",
            allowed_values=("pareto", "thompson", "ucb"),
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Random source for candidate values.",
            example="sparse-enn/candidate_rv=uniform",
            allowed_values=("sobol", "uniform", "gpu_uniform"),
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
    "eggroll": [
        DesignerOptionSpec(
            name="noiser",
            required=False,
            value_type="str",
            description="HyperscaleES noiser name.",
            example="eggroll/noiser=eggroll",
        ),
        DesignerOptionSpec(
            name="sigma",
            required=False,
            value_type="float",
            description="Initial noiser sigma.",
            example="eggroll/sigma=0.05",
        ),
        DesignerOptionSpec(
            name="sigma_decay",
            required=False,
            value_type="float",
            description="Per-generation multiplicative sigma decay.",
            example="eggroll/sigma_decay=0.999",
        ),
        DesignerOptionSpec(
            name="lr",
            required=False,
            value_type="float",
            description="Initial optimizer learning rate.",
            example="eggroll/lr=0.02",
        ),
        DesignerOptionSpec(
            name="lr_decay",
            required=False,
            value_type="float",
            description="Per-update multiplicative learning-rate decay.",
            example="eggroll/lr_decay=0.9995",
        ),
        DesignerOptionSpec(
            name="rank",
            required=False,
            value_type="int",
            description="Low-rank EggRoll perturbation rank.",
            example="eggroll/rank=8",
        ),
        DesignerOptionSpec(
            name="rank_transform",
            required=False,
            value_type="bool",
            description="Rank-transform population scores before HyperscaleES fitness normalization.",
            example="eggroll/rank_transform=false",
        ),
        DesignerOptionSpec(
            name="deterministic_policy",
            required=False,
            value_type="bool",
            description="Use distribution modes/means for policy actions instead of sampling.",
            example="eggroll/deterministic_policy=false",
        ),
        DesignerOptionSpec(
            name="steps",
            required=False,
            value_type="int",
            description="Rollout horizon per sampled policy.",
            example="eggroll/steps=200",
        ),
        DesignerOptionSpec(
            name="eval_episodes",
            required=False,
            value_type="int",
            description="Held-out center-policy evaluation episodes per generation.",
            example="eggroll/eval_episodes=8",
        ),
        DesignerOptionSpec(
            name="suppress_noiser_stdout",
            required=False,
            value_type="bool",
            description="Suppress noisy upstream HyperscaleES tracing prints.",
            example="eggroll/suppress_noiser_stdout=true",
        ),
    ],
}


_DESIGNER_DISPATCH = {name: partial(_no_opts, name, builder) for name, builder in _SIMPLE_BUILDERS.items()} | {
    "turbo-enn": partial(_d_turbo_enn_simple, kind="turbo-enn"),
    "turbo-enn-p": partial(_d_turbo_enn_simple, kind="turbo-enn-p"),
    "turbo-enn-fit-ucb": partial(_d_turbo_enn_simple, kind="turbo-enn-fit-ucb"),
    "turbo_py-enn-p": partial(_d_turbo_enn_py_simple, kind="turbo_py-enn-p"),
    "turbo_py-enn-fit-ucb": partial(_d_turbo_enn_py_simple, kind="turbo_py-enn-fit-ucb"),
    "turbo-zero": partial(_d_turbo_enn_simple, kind="turbo-zero"),
    "turbo-one": partial(_d_turbo_enn_simple, kind="turbo-one"),
    "lhd_only": partial(_d_turbo_enn_simple, kind="lhd_only"),
    "morbo-zero": partial(_d_turbo_enn_simple, kind="morbo-zero"),
    "morbo-one": partial(_d_turbo_enn_simple, kind="morbo-one"),
    "morbo-enn": partial(_d_turbo_enn_simple, kind="morbo-enn"),
    "sparse-enn": _build_sparse_enn,
    "ts_sweep": _d_ts_sweep,
    "rff": _d_rff,
    "pss_sweep_kmcmc": _d_pss_sweep_kmcmc,
    "pss_sweep_num_mcmc": _d_pss_sweep_num_mcmc,
    "sts_sweep": _d_sts_sweep,
    "turbo-enn-sweep": _d_turbo_enn_sweep,
    "turbo-enn-fit": _d_turbo_enn_fit,
    "turbo-enn-f": _d_turbo_enn_f,
    "turbo-enn-f-p": _d_turbo_enn_f_p,
    "morbo-enn-fit": _d_morbo_enn_fit,
    "sts-ar": _d_sts_ar,
    "ppo": _ppo,
    "eggroll": _eggroll,
}
