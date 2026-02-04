import importlib
from functools import partial

from .designer_errors import NoSuchDesignerError
from .designer_opts import OptParser as _OptParser
from .designer_spec import DesignerOptionSpec


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


def _load_symbol(module: str, name: str):
    mod = importlib.import_module(module)
    return getattr(mod, name)


def _no_opts(name: str, build, ctx: _SimpleContext, opts: dict):
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer '{name}' does not support options (got: {keys}).")
    return build(ctx)


def _turbo_ref(ctx: _SimpleContext, *, ard: bool, surrogate_type: str = "gp"):
    TuRBORefDesigner = _load_symbol("optimizer.turbo_ref_designer", "TuRBORefDesigner")
    return TuRBORefDesigner(ctx.policy, num_init=ctx.init_yubo_default, ard=ard, surrogate_type=surrogate_type)


def _turbo_enn(ctx: _SimpleContext, **kw):
    TurboENNDesigner = _load_symbol("optimizer.turbo_enn_designer", "TurboENNDesigner")
    return TurboENNDesigner(ctx.policy, **kw)


def _turbo_enn_multi(ctx: _SimpleContext, **kw):
    MultiTurboENNDesigner = _load_symbol("optimizer.multi_turbo_enn_designer", "MultiTurboENNDesigner")
    MultiTurboENNConfig = _load_symbol("optimizer.multi_turbo_enn_designer", "MultiTurboENNConfig")
    config = MultiTurboENNConfig(**kw)
    return MultiTurboENNDesigner(ctx.policy, config=config)


def _mtv(ctx: _SimpleContext, *, acq_kwargs: dict):
    AcqMTV = _load_symbol("acq.acq_mtv", "AcqMTV")
    return ctx.bt(AcqMTV, init_sobol=0, acq_kwargs=acq_kwargs)


def _build_policy_ctor(ctx: _SimpleContext, module: str, name: str, **kwargs):
    Ctor = _load_symbol(module, name)
    return Ctor(ctx.policy, **kwargs)


def _build_maximin(ctx: _SimpleContext, *, toroidal: bool):
    AcqMinDist = _load_symbol("acq.acq_min_dist", "AcqMinDist")
    return ctx.bt(lambda m: AcqMinDist(m, toroidal=toroidal))


def _build_bt_acq(ctx: _SimpleContext, module: str, name: str, *, acq_kwargs=None, init_sobol=1, opt_sequential=False, start_at_max=False):
    Acq = _load_symbol(module, name)
    return ctx.bt(Acq, acq_kwargs=acq_kwargs, init_sobol=init_sobol, opt_sequential=opt_sequential, start_at_max=start_at_max)


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
        return _turbo_enn(ctx, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, num_fit_samples=None, num_fit_candidates=None, acq_type="pareto")
    if kind == "turbo-zero":
        return _turbo_enn(ctx, turbo_mode="turbo-zero", num_fit_samples=None, num_fit_candidates=None)
    if kind == "turbo-one":
        return _turbo_enn(ctx, turbo_mode="turbo-one", num_init=ctx.init_yubo_default, num_fit_samples=None, num_fit_candidates=None)
    if kind == "lhd_only":
        return _turbo_enn(ctx, turbo_mode="lhd-only", num_fit_samples=None, num_fit_candidates=None)
    if kind == "morbo-zero":
        return _turbo_enn(ctx, turbo_mode="turbo-zero", tr_type="morbo", num_fit_samples=None, num_fit_candidates=None)
    if kind == "morbo-one":
        return _turbo_enn(ctx, turbo_mode="turbo-one", num_init=ctx.init_yubo_default, tr_type="morbo", num_fit_samples=None, num_fit_candidates=None)
    return _turbo_enn(ctx, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, tr_type="morbo")


def _build_mts(ctx: _SimpleContext, kind: str):
    MTSDesigner = _load_symbol("optimizer.mts_designer", "MTSDesigner")
    if kind == "mts":
        return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="find")
    if kind == "mts-stagger":
        return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="find", use_stagger=True)
    if kind == "mts-ts":
        return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="ts")
    return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="meas")


def _build_mtv_family(ctx: _SimpleContext, kind: str):
    base_kwargs_by_kind = {
        "mtv": {"sample_type": "pss"},
        "pss": {"ts_only": True, "sample_type": "pss"},
        "mtv-sts": {"sample_type": "sts", "num_refinements": 30},
        "mtv-mts": {"sample_type": "mts", "num_refinements": 30},
        "mtv-sts2": {"sample_type": "sts2", "num_refinements": 30},
        "mtv-sts-t": {"sample_type": "sts", "num_refinements": 30, "x_max_type": "ts_meas"},
        "sts": {"ts_only": True, "sample_type": "sts", "num_refinements": 30},
        "sts-ch": {"ts_only": True, "ts_chain": True, "sample_type": "sts", "num_refinements": 30},
        "sts-ns": {"ts_only": True, "sample_type": "sts", "num_refinements": 30, "no_stagger": True},
        "sts-ui": {"ts_only": True, "sample_type": "sts", "num_refinements": 30, "no_stagger": False, "x_max_type": "rand"},
        "sts-t": {"ts_only": True, "sample_type": "sts", "num_refinements": 30, "x_max_type": "ts_meas"},
        "sts-m": {"ts_only": True, "sample_type": "sts", "num_refinements": 30, "x_max_type": "meas"},
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
        return ctx.bt(PathwiseThompsonSampling, init_sobol=ctx.init_yubo_default, num_restarts=20, raw_samples=100)
    return ctx.bt(PathwiseThompsonSampling, init_sobol=ctx.init_yubo_default, start_at_max=True)


def _build_sobol_mc(ctx: _SimpleContext, kind: str):
    if kind == "sobol_ucb":
        return _build_bt_acq(ctx, "botorch.acquisition.monte_carlo", "qUpperConfidenceBound", init_sobol=ctx.init_ax_default, acq_kwargs={"beta": 1})
    if kind == "sobol_ei":
        return _build_bt_acq(
            ctx, "botorch.acquisition.monte_carlo", "qNoisyExpectedImprovement", init_sobol=ctx.init_ax_default, acq_kwargs={"X_baseline": None}
        )
    return _build_bt_acq(
        ctx, "botorch.acquisition.max_value_entropy_search", "qLowerBoundMaxValueEntropy", init_sobol=ctx.init_ax_default, acq_kwargs={"candidate_set": None}
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
    "ts": partial(_build_bt_acq, module="acq.acq_ts", name="AcqTS", acq_kwargs={"sampler": "cholesky", "num_candidates": 1000}),
    "ts-10000": partial(_build_bt_acq, module="acq.acq_ts", name="AcqTS", acq_kwargs={"sampler": "lanczos", "num_candidates": 10000}),
    "sr": partial(_build_bt_acq, module="botorch.acquisition.monte_carlo", name="qSimpleRegret"),
    "ucb": partial(_build_bt_acq, module="botorch.acquisition.monte_carlo", name="qUpperConfidenceBound", acq_kwargs={"beta": 1}),
    "ei": partial(_build_bt_acq, module="botorch.acquisition.monte_carlo", name="qNoisyExpectedImprovement", acq_kwargs={"X_baseline": None}),
    "lei": partial(_build_bt_acq, module="botorch.acquisition.logei", name="qLogNoisyExpectedImprovement", acq_kwargs={"X_baseline": None}),
    "lei-m": partial(
        _build_bt_acq, module="botorch.acquisition.logei", name="qLogNoisyExpectedImprovement", acq_kwargs={"X_baseline": None}, start_at_max=True
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
    num_candidates = _OptParser.require_int(opts, "num_candidates", example="ts_sweep/num_candidates=10000")
    return _build_bt_acq(ctx, "acq.acq_ts", "AcqTS", acq_kwargs={"sampler": "lanczos", "num_candidates": num_candidates})


def _d_rff(ctx: _SimpleContext, opts: dict):
    num_candidates = _OptParser.require_int(opts, "num_candidates", example="rff/num_candidates=10000")
    return _build_bt_acq(ctx, "acq.acq_ts", "AcqTS", acq_kwargs={"sampler": "rff", "num_candidates": num_candidates})


def _d_pss_sweep_kmcmc(ctx: _SimpleContext, opts: dict):
    k_mcmc = _OptParser.require_int(opts, "k_mcmc", example="pss_sweep_kmcmc/k_mcmc=8")
    return _mtv(ctx, acq_kwargs={"ts_only": True, "num_X_samples": ctx.default_num_X_samples, "sample_type": "pss", "k_mcmc": k_mcmc})


def _d_pss_sweep_num_mcmc(ctx: _SimpleContext, opts: dict):
    num_mcmc = _OptParser.require_int(opts, "num_mcmc", example="pss_sweep_num_mcmc/num_mcmc=16")
    return _mtv(ctx, acq_kwargs={"ts_only": True, "num_X_samples": ctx.default_num_X_samples, "sample_type": "pss", "k_mcmc": None, "num_mcmc": num_mcmc})


def _d_sts_sweep(ctx: _SimpleContext, opts: dict):
    num_refinements = _OptParser.require_int(opts, "num_refinements", example="sts_sweep/num_refinements=30")
    return _mtv(ctx, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": ctx.default_num_X_samples, "num_refinements": num_refinements})


def _d_turbo_enn_sweep(ctx: _SimpleContext, opts: dict):
    k = _OptParser.require_int(opts, "k", example="turbo-enn-sweep/k=10")
    geometry = _OptParser.optional_str_in(
        opts,
        "geometry",
        {"box", "enn_ellipsoid", "enn_grad_ellipsoid"},
    )
    sampler = _OptParser.optional_str_in(opts, "sampler", {"full", "low_rank"})
    rank = _OptParser.optional_int(opts, "rank")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=k,
        num_keep=None,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
        tr_geometry=geometry,
        ellipsoid_sampler=sampler,
        ellipsoid_rank=rank,
    )


def _d_turbo_enn_fit(ctx: _SimpleContext, opts: dict):
    acq_type = _OptParser.require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="turbo-enn-fit/acq_type=ucb",
    )
    k = _OptParser.optional_int(opts, "k")
    candidate_rv = _OptParser.optional_str_in(opts, "candidate_rv", {"sobol", "uniform", "gpu_uniform"})
    num_candidates = _OptParser.optional_int(opts, "num_candidates")
    num_fit_samples = _OptParser.optional_int(opts, "num_fit_samples")
    num_fit_candidates = _OptParser.optional_int(opts, "num_fit_candidates")
    geometry = _OptParser.optional_str_in(
        opts,
        "geometry",
        {"box", "enn_ellipsoid", "enn_grad_ellipsoid"},
    )
    sampler = _OptParser.optional_str_in(opts, "sampler", {"full", "low_rank"})
    rank = _OptParser.optional_int(opts, "rank")
    tr_length_fixed = _OptParser.optional_float(opts, "tr_length_fixed")
    if k is not None and int(k) <= 0:
        raise NoSuchDesignerError("Designer option 'k' must be > 0.")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10 if k is None else int(k),
        num_keep=ctx.num_keep_val,
        num_fit_samples=100 if num_fit_samples is None else num_fit_samples,
        num_fit_candidates=100 if num_fit_candidates is None else num_fit_candidates,
        acq_type=acq_type,
        tr_type=None,
        tr_geometry=geometry,
        ellipsoid_sampler=sampler,
        ellipsoid_rank=rank,
        tr_length_fixed=tr_length_fixed,
        num_candidates=num_candidates,
        candidate_rv=candidate_rv,
    )


def _d_turbo_enn_multi(ctx: _SimpleContext, opts: dict):
    acq_type = _OptParser.require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="turbo-enn-multi/acq_type=thompson",
    )
    candidate_rv = _OptParser.optional_str_in(opts, "candidate_rv", {"sobol", "uniform", "gpu_uniform"})
    num_candidates = _OptParser.optional_int(opts, "num_candidates")
    num_fit_samples = _OptParser.optional_int(opts, "num_fit_samples")
    num_fit_candidates = _OptParser.optional_int(opts, "num_fit_candidates")
    num_regions = _OptParser.require_int(opts, "num_regions", example="turbo-enn-multi/num_regions=4")
    strategy = _OptParser.optional_str_in(opts, "strategy", {"independent", "shared_data"})
    arm_mode = _OptParser.optional_str_in(opts, "arm_mode", {"split", "per_region", "allocated"})
    pool_multiplier = _OptParser.optional_int(opts, "pool_multiplier")
    geometry = _OptParser.optional_str_in(
        opts,
        "geometry",
        {"box", "enn_ellipsoid", "enn_grad_ellipsoid"},
    )
    sampler = _OptParser.optional_str_in(opts, "sampler", {"full", "low_rank"})
    rank = _OptParser.optional_int(opts, "rank")
    return _turbo_enn_multi(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100 if num_fit_samples is None else num_fit_samples,
        num_fit_candidates=100 if num_fit_candidates is None else num_fit_candidates,
        acq_type=acq_type,
        tr_type=None,
        tr_geometry=geometry,
        ellipsoid_sampler=sampler,
        ellipsoid_rank=rank,
        num_regions=num_regions,
        strategy=strategy if strategy is not None else "independent",
        arm_mode=arm_mode if arm_mode is not None else "split",
        pool_multiplier=pool_multiplier if pool_multiplier is not None else 2,
        num_candidates=num_candidates,
        candidate_rv=candidate_rv,
    )


def _d_turbo_enn_f(ctx: _SimpleContext, opts: dict):
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer 'turbo-enn-f' does not support options (got: {keys}).")

    def num_candidates(num_dim, num_arms):
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
        num_candidates=num_candidates,
        candidate_rv="uniform",
    )


def _d_morbo_enn_fit(ctx: _SimpleContext, opts: dict):
    acq_type = _OptParser.require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="morbo-enn-fit/acq_type=ucb",
    )
    num_metrics = _OptParser.optional_int(opts, "num_metrics")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100 * ctx.num_arms,
        acq_type=acq_type,
        tr_type="morbo",
        num_metrics=num_metrics,
    )


def _d_morbo_enn_multi(ctx: _SimpleContext, opts: dict):
    acq_type = _OptParser.require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="morbo-enn-multi/acq_type=ucb",
    )
    num_regions = _OptParser.require_int(opts, "num_regions", example="morbo-enn-multi/num_regions=4")
    strategy = _OptParser.optional_str_in(opts, "strategy", {"independent", "shared_data"})
    arm_mode = _OptParser.optional_str_in(opts, "arm_mode", {"split", "per_region", "allocated"})
    pool_multiplier = _OptParser.optional_int(opts, "pool_multiplier")
    candidate_rv = _OptParser.optional_str_in(opts, "candidate_rv", {"sobol", "uniform", "gpu_uniform"})
    num_candidates = _OptParser.optional_int(opts, "num_candidates")
    num_fit_samples = _OptParser.optional_int(opts, "num_fit_samples")
    num_fit_candidates = _OptParser.optional_int(opts, "num_fit_candidates")
    num_metrics = _OptParser.optional_int(opts, "num_metrics")
    return _turbo_enn_multi(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100 if num_fit_samples is None else num_fit_samples,
        num_fit_candidates=100 if num_fit_candidates is None else num_fit_candidates,
        acq_type=acq_type,
        tr_type="morbo",
        num_metrics=num_metrics,
        num_regions=num_regions,
        strategy=strategy if strategy is not None else "independent",
        arm_mode=arm_mode if arm_mode is not None else "split",
        pool_multiplier=pool_multiplier if pool_multiplier is not None else 2,
        num_candidates=num_candidates,
        candidate_rv=candidate_rv,
    )


def _d_sts_ar(ctx: _SimpleContext, opts: dict):
    num_acc_rej = _OptParser.require_int(opts, "num_acc_rej", example="sts-ar/num_acc_rej=10")
    return _mtv(
        ctx, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": ctx.default_num_X_samples, "num_refinements": 0, "num_acc_rej": num_acc_rej}
    )


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
            name="num_candidates", required=True, value_type="int", description="Number of TS candidates (RFF sampler).", example="rff/num_candidates=10000"
        )
    ],
    "pss_sweep_kmcmc": [
        DesignerOptionSpec(
            name="k_mcmc", required=True, value_type="int", description="Number of MCMC chains per refinement (PSS).", example="pss_sweep_kmcmc/k_mcmc=8"
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
            name="num_refinements", required=True, value_type="int", description="Number of STS refinements.", example="sts_sweep/num_refinements=30"
        )
    ],
    "turbo-enn-sweep": [
        DesignerOptionSpec(name="k", required=True, value_type="int", description="Ensemble size for TuRBO-ENN sweep.", example="turbo-enn-sweep/k=10"),
        DesignerOptionSpec(
            name="geometry",
            required=False,
            value_type="str",
            description="Trust-region geometry for TuRBO-ENN.",
            example="turbo-enn-sweep/geometry=enn_ellipsoid",
            allowed_values=("box", "enn_ellipsoid", "enn_grad_ellipsoid"),
        ),
        DesignerOptionSpec(
            name="sampler",
            required=False,
            value_type="str",
            description="Ellipsoid sampler for geometry (full or low_rank).",
            example="turbo-enn-sweep/sampler=low_rank",
            allowed_values=("full", "low_rank"),
        ),
        DesignerOptionSpec(
            name="rank",
            required=False,
            value_type="int",
            description="Rank cap for low-rank ellipsoid sampling.",
            example="turbo-enn-sweep/rank=5",
        ),
    ],
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
            description="ENN neighbor count (k) for TuRBO-ENN.",
            example="turbo-enn-fit/k=80",
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Candidate random variable for TuRBO candidate generation.",
            example="turbo-enn-fit/candidate_rv=uniform",
            allowed_values=("sobol", "uniform", "gpu_uniform"),
        ),
        DesignerOptionSpec(
            name="num_candidates",
            required=False,
            value_type="int",
            description="Number of candidates to generate per selection step.",
            example="turbo-enn-fit/num_candidates=64",
        ),
        DesignerOptionSpec(
            name="num_fit_samples",
            required=False,
            value_type="int",
            description="Number of surrogate-fit samples (ENN fit budget).",
            example="turbo-enn-fit/num_fit_samples=50",
        ),
        DesignerOptionSpec(
            name="num_fit_candidates",
            required=False,
            value_type="int",
            description="Number of candidates used during surrogate fit (ENN fit budget).",
            example="turbo-enn-fit/num_fit_candidates=50",
        ),
        DesignerOptionSpec(
            name="geometry",
            required=False,
            value_type="str",
            description="Trust-region geometry for TuRBO-ENN.",
            example="turbo-enn-fit/geometry=enn_ellipsoid",
            allowed_values=("box", "enn_ellipsoid", "enn_grad_ellipsoid"),
        ),
        DesignerOptionSpec(
            name="sampler",
            required=False,
            value_type="str",
            description="Ellipsoid sampler for geometry (full or low_rank).",
            example="turbo-enn-fit/sampler=low_rank",
            allowed_values=("full", "low_rank"),
        ),
        DesignerOptionSpec(
            name="rank",
            required=False,
            value_type="int",
            description="Rank cap for low-rank ellipsoid sampling.",
            example="turbo-enn-fit/rank=5",
        ),
        DesignerOptionSpec(
            name="tr_length_fixed",
            required=False,
            value_type="float",
            description="Fixed trust-region length for ellipsoid runs.",
            example="turbo-enn-fit/tr_length_fixed=1.6",
        ),
    ],
    "turbo-enn-multi": [
        DesignerOptionSpec(
            name="acq_type",
            required=True,
            value_type="str",
            description="Acquisition type for multi-region TuRBO-ENN.",
            example="turbo-enn-multi/acq_type=thompson",
            allowed_values=("pareto", "thompson", "ucb"),
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Candidate random variable for TuRBO candidate generation.",
            example="turbo-enn-multi/candidate_rv=uniform",
            allowed_values=("sobol", "uniform", "gpu_uniform"),
        ),
        DesignerOptionSpec(
            name="num_candidates",
            required=False,
            value_type="int",
            description="Number of candidates to generate per selection step (per region).",
            example="turbo-enn-multi/num_candidates=64",
        ),
        DesignerOptionSpec(
            name="num_fit_samples",
            required=False,
            value_type="int",
            description="Number of surrogate-fit samples per region (ENN fit budget).",
            example="turbo-enn-multi/num_fit_samples=50",
        ),
        DesignerOptionSpec(
            name="num_fit_candidates",
            required=False,
            value_type="int",
            description="Number of candidates used during surrogate fit per region (ENN fit budget).",
            example="turbo-enn-multi/num_fit_candidates=50",
        ),
        DesignerOptionSpec(
            name="num_regions",
            required=True,
            value_type="int",
            description="Number of trust regions (multi-start regions).",
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
            description="Multiplier for per-region proposal pool in allocated mode.",
            example="turbo-enn-multi/pool_multiplier=2",
        ),
        DesignerOptionSpec(
            name="geometry",
            required=False,
            value_type="str",
            description="Trust-region geometry for TuRBO-ENN.",
            example="turbo-enn-multi/geometry=enn_ellipsoid",
            allowed_values=("box", "enn_ellipsoid", "enn_grad_ellipsoid"),
        ),
        DesignerOptionSpec(
            name="sampler",
            required=False,
            value_type="str",
            description="Ellipsoid sampler for geometry (full or low_rank).",
            example="turbo-enn-multi/sampler=low_rank",
            allowed_values=("full", "low_rank"),
        ),
        DesignerOptionSpec(
            name="rank",
            required=False,
            value_type="int",
            description="Rank cap for low-rank ellipsoid sampling.",
            example="turbo-enn-multi/rank=5",
        ),
    ],
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
            name="num_metrics",
            required=False,
            value_type="int",
            description="Number of metrics for MORBO scalarization.",
            example="morbo-enn-fit/num_metrics=2",
        ),
    ],
    "morbo-enn-multi": [
        DesignerOptionSpec(
            name="acq_type",
            required=True,
            value_type="str",
            description="Acquisition type for multi-region MORBO-ENN.",
            example="morbo-enn-multi/acq_type=ucb",
            allowed_values=("pareto", "thompson", "ucb"),
        ),
        DesignerOptionSpec(
            name="num_regions",
            required=True,
            value_type="int",
            description="Number of trust regions (multi-start regions).",
            example="morbo-enn-multi/num_regions=4",
        ),
        DesignerOptionSpec(
            name="strategy",
            required=False,
            value_type="str",
            description="Data sharing strategy across regions.",
            example="morbo-enn-multi/strategy=independent",
            allowed_values=("independent", "shared_data"),
        ),
        DesignerOptionSpec(
            name="arm_mode",
            required=False,
            value_type="str",
            description="Arm allocation mode across regions.",
            example="morbo-enn-multi/arm_mode=allocated",
            allowed_values=("split", "per_region", "allocated"),
        ),
        DesignerOptionSpec(
            name="pool_multiplier",
            required=False,
            value_type="int",
            description="Multiplier for per-region proposal pool in allocated mode.",
            example="morbo-enn-multi/pool_multiplier=2",
        ),
        DesignerOptionSpec(
            name="candidate_rv",
            required=False,
            value_type="str",
            description="Candidate random variable for TuRBO candidate generation.",
            example="morbo-enn-multi/candidate_rv=uniform",
            allowed_values=("sobol", "uniform", "gpu_uniform"),
        ),
        DesignerOptionSpec(
            name="num_candidates",
            required=False,
            value_type="int",
            description="Number of candidates to generate per selection step (per region).",
            example="morbo-enn-multi/num_candidates=64",
        ),
        DesignerOptionSpec(
            name="num_fit_samples",
            required=False,
            value_type="int",
            description="Number of surrogate-fit samples per region (ENN fit budget).",
            example="morbo-enn-multi/num_fit_samples=50",
        ),
        DesignerOptionSpec(
            name="num_fit_candidates",
            required=False,
            value_type="int",
            description="Number of candidates used during surrogate fit per region (ENN fit budget).",
            example="morbo-enn-multi/num_fit_candidates=50",
        ),
        DesignerOptionSpec(
            name="num_metrics",
            required=False,
            value_type="int",
            description="Number of metrics for MORBO scalarization.",
            example="morbo-enn-multi/num_metrics=2",
        ),
    ],
    "sts-ar": [
        DesignerOptionSpec(name="num_acc_rej", required=True, value_type="int", description="Number of accept/reject steps.", example="sts-ar/num_acc_rej=10")
    ],
}


_DESIGNER_DISPATCH = {name: partial(_no_opts, name, builder) for name, builder in _SIMPLE_BUILDERS.items()} | {
    "ts_sweep": _d_ts_sweep,
    "rff": _d_rff,
    "pss_sweep_kmcmc": _d_pss_sweep_kmcmc,
    "pss_sweep_num_mcmc": _d_pss_sweep_num_mcmc,
    "sts_sweep": _d_sts_sweep,
    "turbo-enn-sweep": _d_turbo_enn_sweep,
    "turbo-enn-fit": _d_turbo_enn_fit,
    "turbo-enn-multi": _d_turbo_enn_multi,
    "turbo-enn-f": _d_turbo_enn_f,
    "morbo-enn-fit": _d_morbo_enn_fit,
    "morbo-enn-multi": _d_morbo_enn_multi,
    "sts-ar": _d_sts_ar,
}
