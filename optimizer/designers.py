# Lazy imports to reduce dependency depth - imported when needed in methods
from dataclasses import dataclass
from typing import NamedTuple


class NoSuchDesignerError(Exception):
    pass


_GENERAL_OPT_KEYS = {"num_keep", "keep_style", "model_spec", "sample_around_best"}


@dataclass(frozen=True, slots=True)
class DesignerOptionSpec:
    name: str
    required: bool
    value_type: str
    description: str
    example: str
    allowed_values: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class DesignerCatalogEntry:
    base_name: str
    options: list[DesignerOptionSpec]
    dispatch: object


class DesignerSpec(NamedTuple):
    base: str
    general: dict
    specific: dict


def _parse_opt_value(raw: str):
    s = raw.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_slash_opts(name: str) -> tuple[str, dict]:
    # New format: base/opt=value/opt2=value2
    parts = [p for p in name.split("/") if p != ""]
    if not parts:
        raise NoSuchDesignerError("Empty designer name")
    base = parts[0]
    opts: dict[str, object] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise NoSuchDesignerError(f"Invalid designer option '{part}'. Expected 'key=value' in '{name}'.")
        k, v = part.split("=", 1)
        k = k.strip()
        if not k:
            raise NoSuchDesignerError(f"Invalid designer option '{part}'. Empty key in '{name}'.")
        if k in opts:
            raise NoSuchDesignerError(f"Duplicate option '{k}' in '{name}'.")
        opts[k] = _parse_opt_value(v)
    return base, opts


def _parse_designer_spec(designer_name: str) -> DesignerSpec:
    parsed = _parse_options(designer_name)
    base_with_slash = parsed.designer_name
    general = {
        "num_keep": parsed.num_keep,
        "keep_style": parsed.keep_style,
        "model_spec": parsed.model_spec,
        "sample_around_best": parsed.sample_around_best,
    }

    base, slash_opts = _parse_slash_opts(base_with_slash)
    all_opts = dict(slash_opts)

    for k in _GENERAL_OPT_KEYS:
        if k in all_opts:
            general[k] = all_opts.pop(k)

    return DesignerSpec(base=base, general=general, specific=all_opts)


def _parse_options(designer_name):
    class ParsedOptions(NamedTuple):
        designer_name: str
        num_keep: int | None
        keep_style: str | None
        model_spec: str | None
        sample_around_best: bool

    if ":" in designer_name:
        designer_name, options_str = designer_name.split(":")
        options = options_str.split("-")
    else:
        options = []

    num_keep = None
    keep_style = None
    model_spec = None
    sample_around_best = False

    keep_style_map = {
        "s": "some",
        "b": "best",
        "r": "random",
        "t": "trailing",
        "p": "lap",
    }

    for option in options:
        if option[0] == "K":
            keep_style = keep_style_map.get(option[1])
            assert keep_style is not None, option
            num_keep = int(option[2:])
            print(f"OPTION: num_keep = {num_keep} keep_style = {keep_style}")
        elif option[0] == "M":
            model_spec = option[1:]
            print(f"OPTION model_spec = {option}")
        elif option[0] == "O":
            if option[1:] == "sab":
                sample_around_best = True
        else:
            assert False, ("Unknown option", option)

    return ParsedOptions(
        designer_name=designer_name,
        num_keep=num_keep,
        keep_style=keep_style,
        model_spec=model_spec,
        sample_around_best=sample_around_best,
    )


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


def _turbo_ref(ctx: _SimpleContext, *, ard: bool, surrogate_type: str = "gp"):
    from .turbo_ref_designer import TuRBORefDesigner

    return TuRBORefDesigner(ctx.policy, num_init=ctx.init_yubo_default, ard=ard, surrogate_type=surrogate_type)


def _turbo_enn(
    ctx: _SimpleContext,
    *,
    turbo_mode: str,
    k: int | None = None,
    num_init: int | None = None,
    num_keep: int | None = None,
    num_fit_samples: int | None = 100,
    num_fit_candidates: int | None = 100,
    acq_type: str | None = None,
    tr_type: str | None = None,
):
    from .turbo_enn_designer import TurboENNDesigner

    kw = {"turbo_mode": turbo_mode}
    if k is not None:
        kw["k"] = k
    if num_init is not None:
        kw["num_init"] = num_init
    if num_keep is not None:
        kw["num_keep"] = num_keep
    if num_fit_samples is not None:
        kw["num_fit_samples"] = num_fit_samples
    if num_fit_candidates is not None:
        kw["num_fit_candidates"] = num_fit_candidates
    if acq_type is not None:
        kw["acq_type"] = acq_type
    if tr_type is not None:
        kw["tr_type"] = tr_type
    return TurboENNDesigner(ctx.policy, **kw)


def _mtv(ctx: _SimpleContext, *, acq_kwargs: dict):
    from acq.acq_mtv import AcqMTV

    return ctx.bt(AcqMTV, init_sobol=0, acq_kwargs=acq_kwargs)


def _h_cma(ctx: _SimpleContext):
    from .cma_designer import CMAESDesigner

    return CMAESDesigner(ctx.policy)


def _h_optuna(ctx: _SimpleContext):
    from .optuna_designer import OptunaDesigner

    return OptunaDesigner(ctx.policy)


def _h_ax(ctx: _SimpleContext):
    from .ax_designer import AxDesigner

    return AxDesigner(ctx.policy)


def _h_random(ctx: _SimpleContext):
    from .random_designer import RandomDesigner

    return RandomDesigner(ctx.policy)


def _h_sobol(ctx: _SimpleContext):
    from .sobol_designer import SobolDesigner

    return SobolDesigner(ctx.policy)


def _h_lhd(ctx: _SimpleContext):
    from .lhd_designer import LHDDesigner

    return LHDDesigner(ctx.policy)


def _h_center(ctx: _SimpleContext):
    from .center_designer import CenterDesigner

    return CenterDesigner(ctx.policy)


def _h_maximin(ctx: _SimpleContext):
    from acq.acq_min_dist import AcqMinDist

    return ctx.bt(lambda m: AcqMinDist(m, toroidal=False))


def _h_maximin_toroidal(ctx: _SimpleContext):
    from acq.acq_min_dist import AcqMinDist

    return ctx.bt(lambda m: AcqMinDist(m, toroidal=True))


def _h_variance(ctx: _SimpleContext):
    from acq.acq_var import AcqVar

    return ctx.bt(AcqVar)


def _h_btsobol(ctx: _SimpleContext):
    from acq.acq_sobol import AcqSobol

    return ctx.bt(AcqSobol)


def _h_sr(ctx: _SimpleContext):
    from botorch.acquisition.monte_carlo import qSimpleRegret

    return ctx.bt(qSimpleRegret)


def _h_ts(ctx: _SimpleContext):
    from acq.acq_ts import AcqTS

    return ctx.bt(AcqTS, acq_kwargs={"sampler": "cholesky", "num_candidates": 1000})


def _h_ts_10000(ctx: _SimpleContext):
    from acq.acq_ts import AcqTS

    return ctx.bt(AcqTS, acq_kwargs={"sampler": "lanczos", "num_candidates": 10000})


def _h_ucb(ctx: _SimpleContext):
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound

    return ctx.bt(qUpperConfidenceBound, acq_kwargs={"beta": 1})


def _h_ei(ctx: _SimpleContext):
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

    return ctx.bt(qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None})


def _h_lei(ctx: _SimpleContext):
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement

    return ctx.bt(qLogNoisyExpectedImprovement, acq_kwargs={"X_baseline": None})


def _h_lei_m(ctx: _SimpleContext):
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement

    return ctx.bt(qLogNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}, start_at_max=True)


def _h_gibbon(ctx: _SimpleContext):
    from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

    return ctx.bt(
        qLowerBoundMaxValueEntropy,
        opt_sequential=True,
        acq_kwargs={"candidate_set": None},
    )


def _h_turbo_1(ctx: _SimpleContext):
    return _turbo_ref(ctx, ard=True)


def _h_turbo_1_iso(ctx: _SimpleContext):
    return _turbo_ref(ctx, ard=False)


def _h_turbo_0(ctx: _SimpleContext):
    return _turbo_ref(ctx, ard=True, surrogate_type="none")


def _h_turbo_enn(ctx: _SimpleContext):
    return _turbo_enn(ctx, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val)


def _h_turbo_enn_p(ctx: _SimpleContext):
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
    )


def _h_turbo_zero(ctx: _SimpleContext):
    return _turbo_enn(ctx, turbo_mode="turbo-zero", num_fit_samples=None, num_fit_candidates=None)


def _h_turbo_one(ctx: _SimpleContext):
    return _turbo_enn(ctx, turbo_mode="turbo-one", num_init=ctx.init_yubo_default, num_fit_samples=None, num_fit_candidates=None)


def _h_lhd_only(ctx: _SimpleContext):
    return _turbo_enn(ctx, turbo_mode="lhd-only", num_fit_samples=None, num_fit_candidates=None)


def _h_morbo_zero(ctx: _SimpleContext):
    return _turbo_enn(ctx, turbo_mode="turbo-zero", tr_type="morbo", num_fit_samples=None, num_fit_candidates=None)


def _h_morbo_one(ctx: _SimpleContext):
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-one",
        num_init=ctx.init_yubo_default,
        tr_type="morbo",
        num_fit_samples=None,
        num_fit_candidates=None,
    )


def _h_morbo_enn(ctx: _SimpleContext):
    return _turbo_enn(ctx, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, tr_type="morbo")


def _h_dpp(ctx: _SimpleContext):
    from acq.acq_dpp import AcqDPP

    return ctx.bt(AcqDPP, init_sobol=1, acq_kwargs={"num_X_samples": ctx.default_num_X_samples})


def _h_vecchia(ctx: _SimpleContext):
    from .vecchia_designer import VecchiaDesigner

    return VecchiaDesigner(ctx.policy, num_candidates_per_arm=ctx.default_num_X_samples)


def _h_mtv(ctx: _SimpleContext):
    return _mtv(ctx, acq_kwargs={"num_X_samples": ctx.default_num_X_samples, "sample_type": "pss"})


def _h_pss(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "num_X_samples": ctx.default_num_X_samples,
            "sample_type": "pss",
        },
    )


def _h_mtv_sts(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "num_X_samples": ctx.default_num_X_samples,
            "sample_type": "sts",
            "num_refinements": 30,
        },
    )


def _h_mtv_mts(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "num_X_samples": ctx.default_num_X_samples,
            "sample_type": "mts",
            "num_refinements": 30,
        },
    )


def _h_mtv_sts2(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "num_X_samples": ctx.default_num_X_samples,
            "sample_type": "sts2",
            "num_refinements": 30,
        },
    )


def _h_mtv_sts_t(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "num_X_samples": ctx.default_num_X_samples,
            "sample_type": "sts",
            "num_refinements": 30,
            "x_max_type": "ts_meas",
        },
    )


def _h_sts(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 30,
        },
    )


def _h_sts_ch(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "ts_chain": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 30,
        },
    )


def _h_sts_ns(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 30,
            "no_stagger": True,
        },
    )


def _h_sts_ui(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 30,
            "no_stagger": False,
            "x_max_type": "rand",
        },
    )


def _h_sts_t(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 30,
            "x_max_type": "ts_meas",
        },
    )


def _h_sts_m(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 30,
            "x_max_type": "meas",
        },
    )


def _h_sts2(ctx: _SimpleContext):
    return _mtv(
        ctx,
        acq_kwargs={
            "ts_only": True,
            "sample_type": "sts2",
            "num_X_samples": ctx.default_num_X_samples,
            "num_refinements": 30,
        },
    )


def _h_path(ctx: _SimpleContext):
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

    return ctx.bt(PathwiseThompsonSampling, init_sobol=ctx.init_yubo_default)


def _h_path_b(ctx: _SimpleContext):
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

    return ctx.bt(
        PathwiseThompsonSampling,
        init_sobol=ctx.init_yubo_default,
        num_restarts=20,
        raw_samples=100,
    )


def _h_path_m(ctx: _SimpleContext):
    from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

    return ctx.bt(PathwiseThompsonSampling, init_sobol=ctx.init_yubo_default, start_at_max=True)


def _h_mcmcbo(ctx: _SimpleContext):
    from .mcmc_bo_designer import MCMCBODesigner

    return MCMCBODesigner(ctx.policy, num_init=ctx.init_yubo_default)


def _h_mts(ctx: _SimpleContext):
    from .mts_designer import MTSDesigner

    return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="find")


def _h_mts_stagger(ctx: _SimpleContext):
    from .mts_designer import MTSDesigner

    return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="find", use_stagger=True)


def _h_mts_ts(ctx: _SimpleContext):
    from .mts_designer import MTSDesigner

    return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="ts")


def _h_mts_meas(ctx: _SimpleContext):
    from .mts_designer import MTSDesigner

    return MTSDesigner(ctx.policy, keep_style=ctx.keep_style, num_keep=ctx.num_keep, init_style="meas")


def _h_sobol_ucb(ctx: _SimpleContext):
    from botorch.acquisition.monte_carlo import qUpperConfidenceBound

    return ctx.bt(qUpperConfidenceBound, init_sobol=ctx.init_ax_default, acq_kwargs={"beta": 1})


def _h_sobol_ei(ctx: _SimpleContext):
    from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

    return ctx.bt(qNoisyExpectedImprovement, init_sobol=ctx.init_ax_default, acq_kwargs={"X_baseline": None})


def _h_sobol_gibbon(ctx: _SimpleContext):
    from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy

    return ctx.bt(qLowerBoundMaxValueEntropy, init_sobol=ctx.init_ax_default, acq_kwargs={"candidate_set": None})


_SIMPLE_DISPATCH = {
    "cma": _h_cma,
    "optuna": _h_optuna,
    "ax": _h_ax,
    "maximin": _h_maximin,
    "maximin-toroidal": _h_maximin_toroidal,
    "variance": _h_variance,
    "random": _h_random,
    "sobol": _h_sobol,
    "lhd": _h_lhd,
    "btsobol": _h_btsobol,
    "center": _h_center,
    "sr": _h_sr,
    "ts": _h_ts,
    "ts-10000": _h_ts_10000,
    "ucb": _h_ucb,
    "ei": _h_ei,
    "lei": _h_lei,
    "lei-m": _h_lei_m,
    "gibbon": _h_gibbon,
    "turbo-1": _h_turbo_1,
    "turbo-1-iso": _h_turbo_1_iso,
    "turbo-0": _h_turbo_0,
    "turbo-enn": _h_turbo_enn,
    "turbo-enn-p": _h_turbo_enn_p,
    "turbo-zero": _h_turbo_zero,
    "turbo-one": _h_turbo_one,
    "lhd_only": _h_lhd_only,
    "morbo-zero": _h_morbo_zero,
    "morbo-one": _h_morbo_one,
    "morbo-enn": _h_morbo_enn,
    "dpp": _h_dpp,
    "vecchia": _h_vecchia,
    "mtv": _h_mtv,
    "pss": _h_pss,
    "mtv-sts": _h_mtv_sts,
    "mtv-mts": _h_mtv_mts,
    "mtv-sts2": _h_mtv_sts2,
    "mtv-sts-t": _h_mtv_sts_t,
    "sts": _h_sts,
    "sts-ch": _h_sts_ch,
    "sts-ns": _h_sts_ns,
    "sts-ui": _h_sts_ui,
    "sts-t": _h_sts_t,
    "sts-m": _h_sts_m,
    "sts2": _h_sts2,
    "path": _h_path,
    "path-b": _h_path_b,
    "path-m": _h_path_m,
    "mcmcbo": _h_mcmcbo,
    "mts": _h_mts,
    "mts-stagger": _h_mts_stagger,
    "mts-ts": _h_mts_ts,
    "mts-meas": _h_mts_meas,
    "sobol_ucb": _h_sobol_ucb,
    "sobol_ei": _h_sobol_ei,
    "sobol_gibbon": _h_sobol_gibbon,
}


def _wrap_no_opts(name: str, fn):
    def _wrapped(ctx: _SimpleContext, opts: dict):
        if opts:
            keys = ", ".join(sorted(opts))
            raise NoSuchDesignerError(f"Designer '{name}' does not support options (got: {keys}).")
        return fn(ctx)

    return _wrapped


def _require_opt(opts: dict, key: str, *, example: str):
    if key not in opts:
        raise NoSuchDesignerError(f"Designer option '{key}' is required. Example: '{example}'.")
    return opts[key]


def _d_ts_sweep(ctx: _SimpleContext, opts: dict):
    from acq.acq_ts import AcqTS

    num_candidates = _require_opt(opts, "num_candidates", example="ts_sweep/num_candidates=10000")
    if not isinstance(num_candidates, int):
        raise NoSuchDesignerError("ts_sweep option 'num_candidates' must be an int.")
    return ctx.bt(AcqTS, acq_kwargs={"sampler": "lanczos", "num_candidates": num_candidates})


def _d_rff(ctx: _SimpleContext, opts: dict):
    from acq.acq_ts import AcqTS

    num_candidates = _require_opt(opts, "num_candidates", example="rff/num_candidates=10000")
    if not isinstance(num_candidates, int):
        raise NoSuchDesignerError("rff option 'num_candidates' must be an int.")
    return ctx.bt(AcqTS, acq_kwargs={"sampler": "rff", "num_candidates": num_candidates})


def _d_pss_sweep_kmcmc(ctx: _SimpleContext, opts: dict):
    k_mcmc = _require_opt(opts, "k_mcmc", example="pss_sweep_kmcmc/k_mcmc=8")
    if not isinstance(k_mcmc, int):
        raise NoSuchDesignerError("pss_sweep_kmcmc option 'k_mcmc' must be an int.")
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
    num_mcmc = _require_opt(opts, "num_mcmc", example="pss_sweep_num_mcmc/num_mcmc=16")
    if not isinstance(num_mcmc, int):
        raise NoSuchDesignerError("pss_sweep_num_mcmc option 'num_mcmc' must be an int.")
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
    num_refinements = _require_opt(opts, "num_refinements", example="sts_sweep/num_refinements=30")
    if not isinstance(num_refinements, int):
        raise NoSuchDesignerError("sts_sweep option 'num_refinements' must be an int.")
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
    k = _require_opt(opts, "k", example="turbo-enn-sweep/k=10")
    if not isinstance(k, int):
        raise NoSuchDesignerError("turbo-enn-sweep option 'k' must be an int.")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=k,
        num_keep=None,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
    )


def _d_turbo_enn_fit(ctx: _SimpleContext, opts: dict):
    acq_type = _require_opt(opts, "acq_type", example="turbo-enn-fit/acq_type=ucb")
    if not isinstance(acq_type, str):
        raise NoSuchDesignerError("turbo-enn-fit option 'acq_type' must be a string.")
    if acq_type not in {"pareto", "thompson", "ucb"}:
        raise NoSuchDesignerError("turbo-enn-fit option 'acq_type' must be one of: pareto, thompson, ucb.")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100,
        acq_type=acq_type,
        tr_type=None,
    )


def _d_turbo_enn_f(ctx: _SimpleContext, opts: dict):
    if opts:
        keys = ", ".join(sorted(opts))
        raise NoSuchDesignerError(f"Designer 'turbo-enn-f' does not support options (got: {keys}).")

    def num_candidates(num_dim, num_arms):
        return 100 * num_arms

    from .turbo_enn_designer import TurboENNDesigner

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
    acq_type = _require_opt(opts, "acq_type", example="morbo-enn-fit/acq_type=ucb")
    if not isinstance(acq_type, str):
        raise NoSuchDesignerError("morbo-enn-fit option 'acq_type' must be a string.")
    if acq_type not in {"pareto", "thompson", "ucb"}:
        raise NoSuchDesignerError("morbo-enn-fit option 'acq_type' must be one of: pareto, thompson, ucb.")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100 * ctx.num_arms,
        acq_type=acq_type,
        tr_type="morbo",
    )


def _d_sts_ar(ctx: _SimpleContext, opts: dict):
    num_acc_rej = _require_opt(opts, "num_acc_rej", example="sts-ar/num_acc_rej=10")
    if not isinstance(num_acc_rej, int):
        raise NoSuchDesignerError("sts-ar option 'num_acc_rej' must be an int.")
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


_DESIGNER_DISPATCH = {name: _wrap_no_opts(name, fn) for name, fn in _SIMPLE_DISPATCH.items()} | {
    "ts_sweep": _d_ts_sweep,
    "rff": _d_rff,
    "pss_sweep_kmcmc": _d_pss_sweep_kmcmc,
    "pss_sweep_num_mcmc": _d_pss_sweep_num_mcmc,
    "sts_sweep": _d_sts_sweep,
    "turbo-enn-sweep": _d_turbo_enn_sweep,
    "turbo-enn-fit": _d_turbo_enn_fit,
    "turbo-enn-f": _d_turbo_enn_f,
    "morbo-enn-fit": _d_morbo_enn_fit,
    "sts-ar": _d_sts_ar,
}


class Designers:
    def __init__(self, policy, num_arms):
        self._policy = policy
        self._num_arms = num_arms

    def is_valid(self, designer_name):
        # Must be cheap: used for validation / CLI help.
        try:
            spec = _parse_designer_spec(designer_name)
        except Exception:
            return False
        return spec.base in _DESIGNER_DISPATCH

    def _bt_designer(
        self,
        acq_factory,
        acq_kwargs=None,
        init_sobol=1,
        opt_sequential=False,
        num_restarts=10,
        raw_samples=10,
        start_at_max=False,
        num_keep=None,
        keep_style=None,
        model_spec=None,
        sample_around_best=False,
    ):
        from .bt_designer import BTDesigner

        return BTDesigner(
            self._policy,
            acq_factory,
            acq_kwargs=acq_kwargs,
            num_keep=num_keep,
            keep_style=keep_style,
            model_spec=model_spec,
            init_sobol=init_sobol,
            opt_sequential=opt_sequential,
            optimizer_options={
                "batch_limit": 10,
                "maxiter": 1000,
                "sample_around_best": sample_around_best,
            },
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            start_at_max=start_at_max,
        )

    def create(self, designer_name):
        spec = _parse_designer_spec(designer_name)
        handler = _DESIGNER_DISPATCH.get(spec.base)
        if handler is None:
            raise NoSuchDesignerError(spec.base)

        num_keep = spec.general["num_keep"]
        keep_style = spec.general["keep_style"]
        model_spec = spec.general["model_spec"]
        sample_around_best = spec.general["sample_around_best"]

        init_ax_default = max(5, 2 * self._policy.num_params())
        init_yubo_default = self._num_arms
        default_num_X_samples = max(64, 10 * self._num_arms)

        def bt(*args, **kw):
            return self._bt_designer(
                *args,
                num_keep=num_keep,
                keep_style=keep_style,
                model_spec=model_spec,
                sample_around_best=sample_around_best,
                **kw,
            )

        num_keep_val = num_keep if keep_style == "trailing" else None
        ctx = _SimpleContext(
            self._policy,
            self._num_arms,
            bt,
            num_keep=num_keep,
            keep_style=keep_style,
            num_keep_val=num_keep_val,
            init_yubo_default=init_yubo_default,
            init_ax_default=init_ax_default,
            default_num_X_samples=default_num_X_samples,
        )
        return handler(ctx, spec.specific)

    def catalog(self) -> list[DesignerCatalogEntry]:
        """
        Return a catalog of supported designer base names.

        Notes:
        - This lists *designer-specific* options only (not general BT options like num_keep).
        - Dispatch functions are the underlying builders (e.g. `_h_sobol`, `_d_ts_sweep`).
        """
        entries: list[DesignerCatalogEntry] = []

        # "No opts" designers expose their underlying _h_* dispatchers.
        for base_name, fn in _SIMPLE_DISPATCH.items():
            entries.append(
                DesignerCatalogEntry(
                    base_name=base_name,
                    options=_DESIGNER_OPTION_SPECS.get(base_name, []),
                    dispatch=fn,
                )
            )

        # Option-enabled designers (not in _SIMPLE_DISPATCH) expose their _d_* dispatchers.
        opt_dispatch = {
            "ts_sweep": _d_ts_sweep,
            "rff": _d_rff,
            "pss_sweep_kmcmc": _d_pss_sweep_kmcmc,
            "pss_sweep_num_mcmc": _d_pss_sweep_num_mcmc,
            "sts_sweep": _d_sts_sweep,
            "turbo-enn-sweep": _d_turbo_enn_sweep,
            "turbo-enn-fit": _d_turbo_enn_fit,
            "turbo-enn-f": _d_turbo_enn_f,
            "morbo-enn-fit": _d_morbo_enn_fit,
            "sts-ar": _d_sts_ar,
        }
        for base_name, fn in opt_dispatch.items():
            entries.append(
                DesignerCatalogEntry(
                    base_name=base_name,
                    options=_DESIGNER_OPTION_SPECS.get(base_name, []),
                    dispatch=fn,
                )
            )

        entries.sort(key=lambda e: e.base_name)
        return entries
