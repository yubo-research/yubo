from .designer_errors import NoSuchDesignerError
from .designer_registry_builders import (
    _build_bt_acq,
    _index_driver_from_opts,
    _load_symbol,
    _mtv,
    _optional_int,
    _reject_unknown_opts,
    _require_int,
    _require_str_in,
)
from .designer_registry_builders import (
    _turbo_enn as _base_turbo_enn,
)
from .designer_registry_context import _SimpleContext

_TURBO_OPTION_KEYS = {
    "acq_type",
    "candidate_rv",
    "idx",
    "k",
    "num_candidates",
    "num_fit_candidates",
    "num_fit_samples",
    "num_init",
    "num_keep",
    "num_metrics",
    "tr_type",
    "use_python",
    "use_y_var",
}
_EGGROLL_TURBO_OPTION_KEYS = {
    "deterministic_policy",
    "num_envs",
    "param_scale",
    "seed_offset",
    "steps_per_episode",
}


def _turbo_enn(
    ctx: _SimpleContext,
    *,
    opts: dict | None = None,
    designer_name: str = "turbo-enn",
    **kw,
):
    opts = dict(opts or {})
    allowed = _TURBO_OPTION_KEYS | _EGGROLL_TURBO_OPTION_KEYS
    unknown = set(opts) - allowed
    if unknown:
        raise NoSuchDesignerError(f"Designer '{designer_name}' does not support option(s): {', '.join(sorted(unknown))}.")
    conflicts = set(opts) & set(kw)
    if conflicts:
        raise NoSuchDesignerError(f"Designer '{designer_name}' got duplicate option(s): {', '.join(sorted(conflicts))}.")
    kw.update(opts)
    index_driver = kw.pop("idx", None)
    if index_driver is not None:
        kw["index_driver"] = _index_driver_from_opts({"idx": index_driver}, example=f"{designer_name}/idx=hnsw")
    eggroll_opts = {k: kw.pop(k) for k in list(kw) if k in _EGGROLL_TURBO_OPTION_KEYS}
    if eggroll_opts:
        if ctx.env_conf is None:
            raise NoSuchDesignerError(f"Designer '{designer_name}' EggRoll options require env_conf.")
        EggRollJAXVectorDesigner = _load_symbol("optimizer.eggroll_vector_designer", "EggRollJAXVectorDesigner")
        return EggRollJAXVectorDesigner(ctx.policy, ctx.env_conf, **eggroll_opts, **kw)
    return _base_turbo_enn(ctx, **kw)


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
    _reject_unknown_opts(
        "turbo-enn-sweep",
        opts,
        {"k", "idx"} | _TURBO_OPTION_KEYS | _EGGROLL_TURBO_OPTION_KEYS,
    )
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


def _d_turbo_enn_p(ctx: _SimpleContext, opts: dict):
    opts = dict(opts)
    _reject_unknown_opts(
        "turbo-enn-p",
        opts,
        {"idx"} | _EGGROLL_TURBO_OPTION_KEYS,
    )
    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="turbo-enn-p",
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
    )


def _d_turbo_enn_fit(ctx: _SimpleContext, opts: dict):
    opts = dict(opts)
    _reject_unknown_opts(
        "turbo-enn-fit",
        opts,
        {"acq_type", "idx"} | _TURBO_OPTION_KEYS | _EGGROLL_TURBO_OPTION_KEYS,
    )
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


def _build_turbo_enn_f(ctx: _SimpleContext, *, acq_type: str, opts: dict | None = None):
    """Factory for turbo-enn-f variants. acq_type: 'ucb' or 'pareto'."""

    def _num_candidates(num_dim, num_arms):
        return 100 * num_arms

    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="turbo-enn-f" if acq_type == "ucb" else "turbo-enn-f-p",
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100,
        acq_type=acq_type,
        num_candidates=_num_candidates,
        candidate_rv="uniform",
    )


def _d_morbo_enn_fit(ctx: _SimpleContext, opts: dict):
    opts = dict(opts)
    _reject_unknown_opts(
        "morbo-enn-fit",
        opts,
        {"acq_type", "idx"} | _TURBO_OPTION_KEYS | _EGGROLL_TURBO_OPTION_KEYS,
    )
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


def _d_turbo_enn_fit_ucb(ctx: _SimpleContext, opts: dict):
    opts = dict(opts)
    eggroll_allowed = {
        "steps_per_episode",
        "num_envs",
        "deterministic_policy",
        "param_scale",
        "seed_offset",
    }
    allowed = {"nfs", "k", "idx", "num_init"} | eggroll_allowed
    unknown = set(opts) - allowed
    if unknown:
        u = ", ".join(sorted(unknown))
        raise NoSuchDesignerError(f"Designer 'turbo-enn-fit-ucb' does not support option(s): {u}. Use nfs and/or k. Example: 'turbo-enn-fit-ucb/nfs=100/k=10'.")
    nfs = _optional_int(opts, "nfs", default=100, example="turbo-enn-fit-ucb/nfs=50")
    k = _optional_int(opts, "k", default=10, example="turbo-enn-fit-ucb/k=20")
    opts.pop("nfs", None)
    opts.pop("k", None)
    return _turbo_enn(
        ctx,
        opts=opts,
        designer_name="turbo-enn-fit-ucb",
        turbo_mode="turbo-enn",
        k=k,
        num_keep=ctx.num_keep_val,
        num_fit_samples=nfs,
        num_fit_candidates=100,
        acq_type="ucb",
    )


def _d_turbo_enn_varentropy_ucb(ctx: _SimpleContext, opts: dict):
    ENNVarentropySurrogateConfig = _load_symbol("optimizer.enn_varentropy_config", "ENNVarentropySurrogateConfig")
    TurboENNVarentropyDesigner = _load_symbol("optimizer.turbo_enn_varentropy_designer", "TurboENNVarentropyDesigner")
    TurboENNVarentropyDesignerConfig = _load_symbol(
        "optimizer.turbo_enn_varentropy_config",
        "TurboENNVarentropyDesignerConfig",
    )

    opts = dict(opts)
    allowed = {
        "candidate_rv",
        "idx",
        "include_noise_in_sigma",
        "k",
        "normalize_varentropy",
        "num_candidates",
        "num_init",
        "scale_x",
        "varentropy_scale",
        "variance_eps",
    }
    _reject_unknown_opts("turbo-enn-varentropy-ucb", opts, allowed)
    index_driver = _index_driver_from_opts(opts, example="turbo-enn-varentropy-ucb/idx=flat") if "idx" in opts else "flat"
    config = ENNVarentropySurrogateConfig(
        k=_optional_int(opts, "k", default=10, example="turbo-enn-varentropy-ucb/k=10"),
        index_driver=index_driver or "flat",
        scale_x=_optional_bool(opts, "scale_x", default=False),
        varentropy_scale=_optional_float(opts, "varentropy_scale", default=0.5),
        variance_eps=_optional_float(opts, "variance_eps", default=1e-12),
        normalize_varentropy=_optional_bool(opts, "normalize_varentropy", default=True),
        include_noise_in_sigma=_optional_bool(opts, "include_noise_in_sigma", default=False),
    )
    return TurboENNVarentropyDesigner(
        ctx.policy,
        config=TurboENNVarentropyDesignerConfig(
            enn=config,
            acq_type="ucb",
            num_init=_optional_int_or_none(opts, "num_init"),
            num_candidates=_optional_int_or_none(opts, "num_candidates"),
            candidate_rv=_optional_str_or_none(opts, "candidate_rv", {"sobol", "uniform", "gpu_uniform"}),
        ),
    )


def _optional_int_or_none(opts: dict, key: str) -> int | None:
    if key not in opts:
        return None
    value = opts[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int.")
    return int(value)


def _optional_float(opts: dict, key: str, *, default: float) -> float:
    if key not in opts:
        return default
    value = opts[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NoSuchDesignerError(f"Designer option '{key}' must be numeric.")
    return float(value)


def _optional_bool(opts: dict, key: str, *, default: bool) -> bool:
    if key not in opts:
        return default
    value = opts[key]
    if not isinstance(value, bool):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a bool.")
    return bool(value)


def _optional_str_or_none(opts: dict, key: str, allowed: set[str]) -> str | None:
    if key not in opts:
        return None
    value = opts[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
    if value not in allowed:
        raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
    return value
