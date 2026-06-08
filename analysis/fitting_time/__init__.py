"""Surrogate models for timing / analysis (DNGO, SMAC-style RF, etc.)."""

from __future__ import annotations

import importlib

__all__ = [
    "BMResult",
    "DNGOConfig",
    "DNGOSurrogate",
    "MuSe",
    "SMACRFConfig",
    "SMACRFSurrogate",
    "SyntheticBenchJob",
    "SYNTHETIC_BENCHMARK_N_TEST",
    "SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME",
    "SURROGATE_BENCHMARK_KEYS",
    "SURROGATE_BENCHMARK_ROWS",
    "SyntheticSineSurrogateBenchmark",
    "EnnFitIndTimingResult",
    "EnnFullOptTimingResult",
    "EnnQueryTimingResult",
    "EnnIncrementalIndexDriver",
    "EnnIncrementalTimingResult",
    "benchmark_enn_fit_ind_timing",
    "benchmark_enn_full_optimization_proposal_timing",
    "benchmark_enn_incremental_add_timing",
    "benchmark_enn_query_timing",
    "benchmark_enn_fit_timing",
    "benchmark_synthetic_sine_surrogates",
    "draw_benchmark_synthetic_xy",
    "EnnFitTimingResult",
    "enn_fit_quality_ns",
    "enn_incremental_checkpoint_ns",
    "env_action_coords_to_surrogate_unit_x",
    "normalize_benchmark_function_name",
    "fit_dngo",
    "fit_enn",
    "fit_enn_hnsw",
    "fit_exact_gp",
    "fit_smac_rf",
    "fit_svgp_default",
    "fit_svgp_linear",
    "fit_vecchia",
    "normalized_rmse",
    "predictive_gaussian_log_likelihood",
]


_LAZY_MODULE_BY_NAME = {
    "DNGOConfig": "analysis.fitting_time.dngo",
    "DNGOSurrogate": "analysis.fitting_time.dngo",
    "SMACRFConfig": "analysis.fitting_time.smac_rf",
    "SMACRFSurrogate": "analysis.fitting_time.smac_rf",
    "SyntheticBenchJob": "analysis.fitting_time.batch_jobs",
    "EnnFitTimingResult": "analysis.fitting_time.fitting_time_enn_fit",
    "benchmark_enn_fit_timing": "analysis.fitting_time.fitting_time_enn_fit",
    "enn_fit_quality_ns": "analysis.fitting_time.fitting_time_enn_fit",
    "EnnFitIndTimingResult": "analysis.fitting_time.fitting_time_enn_fit_ind",
    "benchmark_enn_fit_ind_timing": "analysis.fitting_time.fitting_time_enn_fit_ind",
    "EnnFullOptTimingResult": "analysis.fitting_time.fitting_time_enn_full_opt",
    "benchmark_enn_full_optimization_proposal_timing": "analysis.fitting_time.fitting_time_enn_full_opt",
    "EnnQueryTimingResult": "analysis.fitting_time.fitting_time_enn_query",
    "benchmark_enn_query_timing": "analysis.fitting_time.fitting_time_enn_query",
    "EnnIncrementalIndexDriver": "analysis.fitting_time.fitting_time_enn_incremental",
    "EnnIncrementalTimingResult": "analysis.fitting_time.fitting_time_enn_incremental",
    "benchmark_enn_incremental_add_timing": "analysis.fitting_time.fitting_time_enn_incremental",
    "enn_incremental_checkpoint_ns": "analysis.fitting_time.fitting_time_enn_incremental",
    "fit_dngo": "analysis.fitting_time.fitting_time",
    "fit_enn": "analysis.fitting_time.fitting_time",
    "fit_enn_hnsw": "analysis.fitting_time.fitting_time",
    "fit_exact_gp": "analysis.fitting_time.fitting_time",
    "fit_smac_rf": "analysis.fitting_time.fitting_time",
    "fit_svgp_default": "analysis.fitting_time.fitting_time",
    "fit_svgp_linear": "analysis.fitting_time.fitting_time",
    "fit_vecchia": "analysis.fitting_time.fitting_time",
    "BMResult": "analysis.fitting_time.evaluate",
    "MuSe": "analysis.fitting_time.evaluate",
    "SYNTHETIC_BENCHMARK_N_TEST": "analysis.fitting_time.evaluate",
    "SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME": "analysis.fitting_time.evaluate",
    "SURROGATE_BENCHMARK_KEYS": "analysis.fitting_time.evaluate",
    "SURROGATE_BENCHMARK_ROWS": "analysis.fitting_time.evaluate",
    "SyntheticSineSurrogateBenchmark": "analysis.fitting_time.evaluate",
    "benchmark_synthetic_sine_surrogates": "analysis.fitting_time.evaluate",
    "draw_benchmark_synthetic_xy": "analysis.fitting_time.evaluate",
    "env_action_coords_to_surrogate_unit_x": "analysis.fitting_time.evaluate",
    "normalize_benchmark_function_name": "analysis.fitting_time.evaluate",
    "normalized_rmse": "analysis.fitting_time.evaluate",
    "predictive_gaussian_log_likelihood": "analysis.fitting_time.evaluate",
}


def __getattr__(name: str):
    module_name = _LAZY_MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)
