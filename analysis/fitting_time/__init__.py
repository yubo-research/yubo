"""Surrogate models for timing / analysis (DNGO, SMAC-style RF, etc.)."""

from __future__ import annotations

__all__ = [
    "BMResult",
    "DNGOConfig",
    "DNGOSurrogate",
    "MuSe",
    "SMACRFConfig",
    "SMACRFSurrogate",
    "SyntheticBenchJob",
    "SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME",
    "SURROGATE_BENCHMARK_KEYS",
    "SURROGATE_BENCHMARK_ROWS",
    "SyntheticSineSurrogateBenchmark",
    "benchmark_synthetic_sine_surrogates",
    "draw_benchmark_synthetic_xy",
    "env_action_coords_to_surrogate_unit_x",
    "normalize_benchmark_function_name",
    "fit_dngo",
    "fit_enn",
    "fit_exact_gp",
    "fit_smac_rf",
    "fit_svgp_default",
    "fit_svgp_linear",
    "fit_vecchia",
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
    if name == "SyntheticBenchJob":
        from analysis.fitting_time.batch_jobs import SyntheticBenchJob

        return SyntheticBenchJob
    if name in (
        "fit_dngo",
        "fit_enn",
        "fit_exact_gp",
        "fit_smac_rf",
        "fit_svgp_default",
        "fit_svgp_linear",
        "fit_vecchia",
    ):
        from analysis.fitting_time import fitting_time as _ft

        return getattr(_ft, name)
    if name in (
        "BMResult",
        "MuSe",
        "SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME",
        "SURROGATE_BENCHMARK_KEYS",
        "SURROGATE_BENCHMARK_ROWS",
        "SyntheticSineSurrogateBenchmark",
        "benchmark_synthetic_sine_surrogates",
        "draw_benchmark_synthetic_xy",
        "env_action_coords_to_surrogate_unit_x",
        "normalize_benchmark_function_name",
        "normalized_rmse",
        "predictive_gaussian_log_likelihood",
    ):
        from analysis.fitting_time import evaluate as _ev

        return getattr(_ev, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
