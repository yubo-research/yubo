from __future__ import annotations

import importlib
from dataclasses import replace
from functools import partial
from typing import Any

from .designer_errors import NoSuchDesignerError
from .designer_registry_context import _SimpleContext
from .designer_types import DesignerDef, DesignerOptionSpec
from .mars_config import BayesianMarsSurrogateConfig, MarsSurrogateConfig
from .turbo_mars_config import TurboBayesianMARSDesignerConfig, TurboMARSDesignerConfig, stable_bmars_config


def _d_turbo_mars(ctx: _SimpleContext, opts: dict[str, Any], *, acq_type: str):
    TurboMARSDesigner = _load_symbol("optimizer.turbo_mars_designer", "TurboMARSDesigner")
    opts = dict(opts)
    _reject_unknown("turbo-mars", opts, _COMMON_KEYS | _MARS_KEYS)
    mars = _mars_config(opts)
    return TurboMARSDesigner(ctx.policy, config=TurboMARSDesignerConfig(mars=mars, acq_type=acq_type, **_turbo_kwargs(opts)))


def _d_turbo_bmars(ctx: _SimpleContext, opts: dict[str, Any], *, acq_type: str):
    TurboBayesianMARSDesigner = _load_symbol("optimizer.turbo_mars_designer", "TurboBayesianMARSDesigner")
    opts = dict(opts)
    _reject_unknown("turbo-bmars", opts, _COMMON_KEYS | _MARS_KEYS | _BMARS_KEYS)
    bmars = _bmars_config(opts)
    return TurboBayesianMARSDesigner(
        ctx.policy,
        config=TurboBayesianMARSDesignerConfig(bmars=bmars, acq_type=acq_type, **_turbo_kwargs(opts)),
    )


def _mars_config(opts: dict[str, Any]) -> MarsSurrogateConfig:
    return MarsSurrogateConfig(
        max_terms=_optional_int(opts, "max_terms", 64),
        interaction_order=_optional_int(opts, "interaction_order", 2),
        num_bootstrap=_optional_int(opts, "num_bootstrap", 8),
        active_rank=_optional_int(opts, "active_rank", 8),
        trailing_obs=_optional_int_or_none(opts, "trailing_obs", 256),
        feature_screen=_optional_int(opts, "feature_screen", 512),
        knots_per_feature=_optional_int(opts, "knots_per_feature", 3),
        ridge=_optional_float(opts, "ridge", 1e-6),
        active_samples=_optional_int(opts, "active_samples", 256),
    )


def _bmars_config(opts: dict[str, Any]) -> BayesianMarsSurrogateConfig:
    cfg = stable_bmars_config()
    basis = _bmars_basis_config(opts, cfg.basis)
    return replace(
        cfg,
        basis=basis,
        prior_precision=_optional_float(opts, "prior_precision", cfg.prior_precision),
        min_noise_variance=_optional_float(opts, "min_noise_variance", cfg.min_noise_variance),
        include_noise_in_sigma=_optional_bool(opts, "include_noise_in_sigma", cfg.include_noise_in_sigma),
        basis_sampler=_optional_str(opts, "basis_sampler", cfg.basis_sampler, {"deterministic", "mcmc"}),
        mcmc_steps=_optional_int(opts, "mcmc_steps", cfg.mcmc_steps),
        mcmc_burn_in=_optional_int(opts, "mcmc_burn_in", cfg.mcmc_burn_in),
        mcmc_thin=_optional_int(opts, "mcmc_thin", cfg.mcmc_thin),
        mcmc_num_models=_optional_int(opts, "mcmc_num_models", cfg.mcmc_num_models),
        mcmc_pool_size=_optional_int_or_none(opts, "mcmc_pool_size", cfg.mcmc_pool_size),
        mcmc_term_prior=_optional_float_or_none(opts, "mcmc_term_prior", cfg.mcmc_term_prior),
    )


def _bmars_basis_config(opts: dict[str, Any], basis: MarsSurrogateConfig) -> MarsSurrogateConfig:
    return replace(
        basis,
        max_terms=_optional_int(opts, "max_terms", basis.max_terms),
        interaction_order=_optional_int(opts, "interaction_order", basis.interaction_order),
        num_bootstrap=_optional_int(opts, "num_bootstrap", basis.num_bootstrap),
        active_rank=_optional_int(opts, "active_rank", basis.active_rank),
        trailing_obs=_optional_int_or_none(opts, "trailing_obs", basis.trailing_obs),
        feature_screen=_optional_int(opts, "feature_screen", basis.feature_screen),
        knots_per_feature=_optional_int(opts, "knots_per_feature", basis.knots_per_feature),
        ridge=_optional_float(opts, "ridge", basis.ridge),
        active_samples=_optional_int(opts, "active_samples", basis.active_samples),
    )


def _turbo_kwargs(opts: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_init": _optional_int_or_none(opts, "num_init", None),
        "num_keep": _optional_int_or_none(opts, "num_keep", None),
        "num_candidates": _optional_int_or_none(opts, "num_candidates", None),
        "candidate_rv": _optional_str(opts, "candidate_rv", None, {"sobol", "uniform", "gpu_uniform"}),
    }


def _reject_unknown(name: str, opts: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(opts) - allowed
    if unknown:
        raise NoSuchDesignerError(f"Designer '{name}' does not support option(s): {', '.join(sorted(unknown))}.")


def _load_symbol(module: str, name: str):
    return getattr(importlib.import_module(module), name)


def _optional_int(opts: dict[str, Any], key: str, default: int) -> int:
    value = opts.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int.")
    return int(value)


def _optional_int_or_none(opts: dict[str, Any], key: str, default: int | None) -> int | None:
    value = opts.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise NoSuchDesignerError(f"Designer option '{key}' must be an int or none.")
    return int(value)


def _optional_float(opts: dict[str, Any], key: str, default: float) -> float:
    value = opts.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NoSuchDesignerError(f"Designer option '{key}' must be numeric.")
    return float(value)


def _optional_float_or_none(opts: dict[str, Any], key: str, default: float | None) -> float | None:
    value = opts.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise NoSuchDesignerError(f"Designer option '{key}' must be numeric or none.")
    return float(value)


def _optional_bool(opts: dict[str, Any], key: str, default: bool) -> bool:
    value = opts.get(key, default)
    if not isinstance(value, bool):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a bool.")
    return bool(value)


def _optional_str(opts: dict[str, Any], key: str, default: str | None, allowed: set[str]) -> str | None:
    value = opts.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise NoSuchDesignerError(f"Designer option '{key}' must be a string.")
    if value not in allowed:
        raise NoSuchDesignerError(f"Designer option '{key}' must be one of: {', '.join(sorted(allowed))}.")
    return value


def _spec(name: str, value_type: str, example: str) -> DesignerOptionSpec:
    return DesignerOptionSpec(name=name, required=False, value_type=value_type, description=f"MARS option {name}.", example_suffix=example)


_COMMON_KEYS = {"candidate_rv", "num_candidates", "num_init", "num_keep"}
_MARS_KEYS = {
    "active_rank",
    "active_samples",
    "feature_screen",
    "interaction_order",
    "knots_per_feature",
    "max_terms",
    "num_bootstrap",
    "ridge",
    "trailing_obs",
}
_BMARS_KEYS = {
    "basis_sampler",
    "include_noise_in_sigma",
    "mcmc_burn_in",
    "mcmc_num_models",
    "mcmc_pool_size",
    "mcmc_steps",
    "mcmc_term_prior",
    "mcmc_thin",
    "min_noise_variance",
    "prior_precision",
}

_MARS_SPECS = tuple(
    _spec(name, value_type, example)
    for name, value_type, example in (
        ("num_init", "int", "num_init=8"),
        ("num_candidates", "int", "num_candidates=128"),
        ("candidate_rv", "str", "candidate_rv=sobol"),
        ("max_terms", "int", "max_terms=12"),
        ("interaction_order", "int", "interaction_order=1"),
        ("num_bootstrap", "int", "num_bootstrap=3"),
        ("trailing_obs", "int", "trailing_obs=32"),
        ("feature_screen", "int", "feature_screen=16"),
        ("active_samples", "int", "active_samples=32"),
    )
)
_BMARS_SPECS = _MARS_SPECS + tuple(
    _spec(name, value_type, example)
    for name, value_type, example in (
        ("basis_sampler", "str", "basis_sampler=mcmc"),
        ("include_noise_in_sigma", "bool", "include_noise_in_sigma=true"),
        ("mcmc_steps", "int", "mcmc_steps=32"),
        ("mcmc_burn_in", "int", "mcmc_burn_in=16"),
        ("mcmc_num_models", "int", "mcmc_num_models=8"),
    )
)

MARS_DESIGNER_DEFS = [DesignerDef(f"turbo-mars-{acq}", partial(_d_turbo_mars, acq_type=acq), _MARS_SPECS) for acq in ("ucb", "pareto", "thompson")] + [
    DesignerDef(f"turbo-bmars-{acq}", partial(_d_turbo_bmars, acq_type=acq), _BMARS_SPECS) for acq in ("ucb", "pareto", "thompson")
]
