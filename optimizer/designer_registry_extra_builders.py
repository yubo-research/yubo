from __future__ import annotations

from typing import Any

from .designer_errors import NoSuchDesignerError
from .designer_registry_builders import _load_symbol
from .designer_registry_context import _SimpleContext


def _optional_int_or_none(opts: dict[str, Any], key: str, *, default: int | None, example: str) -> int | None:
    if key not in opts:
        return default
    value = opts[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int or none. Example: '{example}'.")
    return value


def _optional_float(opts: dict[str, Any], key: str, *, default: float, example: str) -> float:
    if key not in opts:
        return default
    value = opts[key]
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a float. Example: '{example}'.")
    return float(value)


def _optional_str_in(
    opts: dict[str, Any],
    key: str,
    allowed: set[str],
    *,
    default: str | None,
    example: str,
) -> str | None:
    if key not in opts:
        return default
    value = opts[key]
    if value is None and default is None:
        return None
    if not isinstance(value, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string. Example: '{example}'.")
    if value not in allowed:
        raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
    return value


def _reject_unknown_opts(name: str, opts: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(opts) - allowed
    if unknown:
        raise NoSuchDesignerError(f"Designer '{name}' does not support option(s): {', '.join(sorted(unknown))}.")


def _build_sparse_enn(ctx: _SimpleContext, opts: dict[str, Any]):
    opts = dict(opts)
    allowed = {
        "acq_type",
        "candidate_rv",
        "clock_scale",
        "k",
        "min_failures",
        "num_candidates",
        "num_fit_candidates",
        "num_fit_samples",
        "num_init",
        "num_pert",
    }
    _reject_unknown_opts("sparse-enn", opts, allowed)
    acq_type = _optional_str_in(
        opts,
        "acq_type",
        {"pareto", "thompson", "ucb"},
        default="pareto",
        example="sparse-enn/acq_type=ucb",
    )
    num_fit_samples_default = 100 if acq_type == "ucb" else None
    num_fit_candidates_default = 100 if acq_type == "ucb" else None
    SparseENNDesigner = _load_symbol("optimizer.sparse_enn_designer", "SparseENNDesigner")
    return SparseENNDesigner(
        ctx.policy,
        clock_scale=_optional_float(opts, "clock_scale", default=3.0, example="sparse-enn/clock_scale=3.0"),
        num_pert=_optional_int_or_none(opts, "num_pert", default=20, example="sparse-enn/num_pert=20"),
        min_failures=_optional_float(opts, "min_failures", default=4.0, example="sparse-enn/min_failures=4"),
        num_init=_optional_int_or_none(opts, "num_init", default=None, example="sparse-enn/num_init=20"),
        k=_optional_int_or_none(opts, "k", default=10, example="sparse-enn/k=10"),
        num_keep=ctx.num_keep_val,
        num_fit_samples=_optional_int_or_none(
            opts,
            "num_fit_samples",
            default=num_fit_samples_default,
            example="sparse-enn/num_fit_samples=100",
        ),
        num_fit_candidates=_optional_int_or_none(
            opts,
            "num_fit_candidates",
            default=num_fit_candidates_default,
            example="sparse-enn/num_fit_candidates=100",
        ),
        acq_type=acq_type,
        num_candidates=_optional_int_or_none(opts, "num_candidates", default=None, example="sparse-enn/num_candidates=1000"),
        candidate_rv=_optional_str_in(
            opts,
            "candidate_rv",
            {"sobol", "uniform", "gpu_uniform"},
            default=None,
            example="sparse-enn/candidate_rv=uniform",
        ),
    )


def _build_eggroll(ctx: _SimpleContext, opts: dict[str, Any]):
    EggRollDesigner = _load_symbol("optimizer.eggroll_designer", "EggRollDesigner")
    return EggRollDesigner(ctx.policy, ctx.env_conf, **dict(opts))
