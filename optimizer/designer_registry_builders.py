import importlib

from .designer_errors import NoSuchDesignerError
from .designer_registry_context import _SimpleContext


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


def _optional_int(opts: dict, key: str, *, default: int, example: str) -> int:
    if key not in opts:
        return default
    v = opts[key]
    if not isinstance(v, int):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int. Example: '{example}'.")
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


def _require_str_in(opts: dict, key: str, allowed: set[str], *, example: str) -> str:
    if key not in opts:
        raise NoSuchDesignerError(f"Designer option '{key}' is required. Example: '{example}'.")
    v = opts[key]
    if not isinstance(v, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
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
    "num_envs",
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


def _build_ppo(ctx: _SimpleContext):
    """On-policy PPO as a Designer (see optimizer/ppo_designer.py). Requires env_conf + actor-critic policy."""
    PPODesigner = _load_symbol("optimizer.ppo_designer", "PPODesigner")
    if ctx.env_conf is None:
        raise NoSuchDesignerError("Designer 'ppo' requires env_conf (use a Gym-style env_tag so the Optimizer has an environment runtime).")
    return PPODesigner(ctx.policy, ctx.env_conf)


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
    if kind == "turbo-zero":
        return _turbo_enn(ctx, opts=opts, designer_name=kind, turbo_mode="turbo-zero", num_fit_samples=None, num_fit_candidates=None)
    turbo_one_acq = {
        "turbo-one": "thompson",
        "turbo-one-nds": "pareto",
        "turbo-one-ucb": "ucb",
    }
    if kind in turbo_one_acq:
        return _turbo_enn(
            ctx,
            opts=opts,
            designer_name=kind,
            turbo_mode="turbo-one",
            num_init=ctx.init_yubo_default,
            num_fit_samples=None,
            num_fit_candidates=None,
            acq_type=turbo_one_acq[kind],
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
    if kind == "morbo-enn":
        return _turbo_enn(
            ctx,
            opts=opts,
            designer_name=kind,
            turbo_mode="turbo-enn",
            k=10,
            num_keep=ctx.num_keep_val,
            tr_type="morbo",
        )
    assert False, ("Should not be reached", kind)


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


def _build_eggroll(ctx: _SimpleContext, opts: dict):
    if ctx.env_conf is None:
        raise NoSuchDesignerError("Designer 'eggroll' requires env_conf to be set.")
    EggRollDesigner = _load_symbol("optimizer.eggroll_designer", "EggRollDesigner")
    return EggRollDesigner(ctx.policy, ctx.env_conf, **opts)


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
