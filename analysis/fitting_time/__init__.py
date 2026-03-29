"""Surrogate models for timing / analysis (DNGO, SMAC-style RF, etc.)."""

from __future__ import annotations

__all__ = [
    "DNGOConfig",
    "DNGOSurrogate",
    "SMACRFConfig",
    "SMACRFSurrogate",
    "SyntheticSineSurrogateBenchmark",
    "benchmark_synthetic_sine_surrogates",
    "fit_dngo",
    "fit_enn",
    "fit_exact_gp",
    "fit_smac_rf",
    "fit_svgp_default",
    "fit_svgp_linear",
    "normalized_rmse",
    "predictive_gaussian_log_likelihood",
]


def __getattr__(name: str):
    if name == "DNGOConfig":
        from analysis.fitting_time.dngo import DNGOConfig

        return DNGOConfig
    if name == "DNGOSurrogate":
        from analysis.fitting_time.dngo import DNGOSurrogate

        return DNGOSurrogate
    if name == "SMACRFConfig":
        from analysis.fitting_time.smac_rf import SMACRFConfig

        return SMACRFConfig
    if name == "SMACRFSurrogate":
        from analysis.fitting_time.smac_rf import SMACRFSurrogate

        return SMACRFSurrogate
    if name in (
        "fit_dngo",
        "fit_enn",
        "fit_exact_gp",
        "fit_smac_rf",
        "fit_svgp_default",
        "fit_svgp_linear",
    ):
        from analysis.fitting_time import fitting_time as _ft

        return getattr(_ft, name)
    if name in (
        "SyntheticSineSurrogateBenchmark",
        "benchmark_synthetic_sine_surrogates",
        "normalized_rmse",
        "predictive_gaussian_log_likelihood",
    ):
        from analysis.fitting_time import evaluate as _ev

        return getattr(_ev, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
