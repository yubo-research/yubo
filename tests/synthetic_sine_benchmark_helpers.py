from __future__ import annotations

import math

from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_KEYS,
    BMResult,
    MuSe,
    SyntheticSineSurrogateBenchmark,
)


def _float_equal(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def assert_surrogate_benchmark_equal(
    left: SyntheticSineSurrogateBenchmark,
    right: SyntheticSineSurrogateBenchmark,
) -> None:
    assert left.results.keys() == right.results.keys()
    for key in left.results:
        for field in ("fit_seconds", "normalized_rmse", "log_likelihood"):
            a = getattr(left.results[key], field)
            b = getattr(right.results[key], field)
            assert _float_equal(a.mu, b.mu), (key, field, "mu", a.mu, b.mu)
            assert _float_equal(a.se, b.se), (key, field, "se", a.se, b.se)


def bench_result(
    fit_s: float,
    nrmse: float,
    ll: float,
    *,
    se_f: float = 0.0,
    se_n: float = 0.0,
    se_l: float = 0.0,
) -> BMResult:
    return BMResult(MuSe(fit_s, se_f), MuSe(nrmse, se_n), MuSe(ll, se_l))


def make_surrogate_benchmark(**kwargs: BMResult) -> SyntheticSineSurrogateBenchmark:
    missing = bench_result(
        float("nan"),
        float("nan"),
        float("nan"),
        se_f=float("nan"),
        se_n=float("nan"),
        se_l=float("nan"),
    )
    return SyntheticSineSurrogateBenchmark(results={k: kwargs.get(k, missing) for k in SURROGATE_BENCHMARK_KEYS})
