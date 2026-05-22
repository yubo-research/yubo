from .designer_registry_builders import (
    _build_bt_acq,
    _index_driver_from_opts,
    _load_symbol,
    _mtv,
    _optional_int,
    _reject_unknown_opts,
    _require_int,
    _require_str_in,
    _turbo_enn,
)
from .designer_registry_context import _SimpleContext


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
    _reject_unknown_opts("turbo-enn-sweep", opts, {"k", "idx"})
    k = _require_int(opts, "k", example="turbo-enn-sweep/k=10")
    index_driver = _index_driver_from_opts(opts, example="turbo-enn-sweep/k=10/idx=hnsw")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=k,
        num_keep=None,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
        index_driver=index_driver,
    )


def _d_turbo_enn_p(ctx: _SimpleContext, opts: dict):
    _reject_unknown_opts("turbo-enn-p", opts, {"idx"})
    index_driver = _index_driver_from_opts(opts, example="turbo-enn-p/idx=hnsw")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=None,
        num_fit_candidates=None,
        acq_type="pareto",
        index_driver=index_driver,
    )


def _d_turbo_enn_fit(ctx: _SimpleContext, opts: dict):
    _reject_unknown_opts("turbo-enn-fit", opts, {"acq_type", "idx"})
    acq_type = _require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="turbo-enn-fit/acq_type=ucb",
    )
    index_driver = _index_driver_from_opts(opts, example="turbo-enn-fit/acq_type=ucb/idx=hnsw")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100,
        acq_type=acq_type,
        tr_type=None,
        index_driver=index_driver,
    )


def _build_turbo_enn_f(ctx: _SimpleContext, *, acq_type: str):
    """Factory for turbo-enn-f variants. acq_type: 'ucb' or 'pareto'."""

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
        acq_type=acq_type,
        num_candidates=_num_candidates,
        candidate_rv="uniform",
    )


def _d_morbo_enn_fit(ctx: _SimpleContext, opts: dict):
    _reject_unknown_opts("morbo-enn-fit", opts, {"acq_type", "idx"})
    acq_type = _require_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        example="morbo-enn-fit/acq_type=ucb",
    )
    index_driver = _index_driver_from_opts(opts, example="morbo-enn-fit/acq_type=ucb/idx=hnsw")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=10,
        num_keep=ctx.num_keep_val,
        num_fit_samples=100,
        num_fit_candidates=100 * ctx.num_arms,
        acq_type=acq_type,
        tr_type="morbo",
        index_driver=index_driver,
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
    _reject_unknown_opts("turbo-enn-fit-ucb", opts, {"nfs", "k", "idx"})
    nfs = _optional_int(opts, "nfs", default=100, example="turbo-enn-fit-ucb/nfs=50")
    k = _optional_int(opts, "k", default=10, example="turbo-enn-fit-ucb/k=20")
    index_driver = _index_driver_from_opts(opts, example="turbo-enn-fit-ucb/idx=hnsw/nfs=100/k=10")
    return _turbo_enn(
        ctx,
        turbo_mode="turbo-enn",
        k=k,
        num_keep=ctx.num_keep_val,
        num_fit_samples=nfs,
        num_fit_candidates=100,
        acq_type="ucb",
        index_driver=index_driver,
    )
