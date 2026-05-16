from __future__ import annotations

from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_KEYS,
    BMResult,
    MuSe,
    SyntheticSineSurrogateBenchmark,
)


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
