"""Benchmark result container."""

from __future__ import annotations

from dataclasses import dataclass

from .evaluate_metrics import SURROGATE_BENCHMARK_KEYS, BMResult
from .evaluate_table import print_synthetic_benchmark_table


@dataclass(frozen=True)
class SyntheticSineSurrogateBenchmark:
    """Per-surrogate :class:`BMResult` (mean and SEM over replicates).

    Data is chosen by the required ``function_name`` passed to
    :func:`benchmark_synthetic_sine_surrogates`: use
    :data:`SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME` for ``FittingTime.ipynb``-style
    ``x ~ U(-1,1)^{N×D}`` with ``y = mean(sin(2π x_u), dim=1) + 0.1 ε`` where
    ``x_u = (x+1)/2`` (same distribution for ``x_u`` as the legacy ``U(0,1)`` draw); any
    other name uses ``f:{name}-{D}d`` from :mod:`problems.pure_functions` on
    ``U(-1,1)^{N×D}`` (same noise scale). Surrogates receive ``(x+1)/2`` in ``[0,1]`` via
    :func:`env_action_coords_to_surrogate_unit_x`.

    Keys are :data:`SURROGATE_BENCHMARK_KEYS`.
    """

    results: dict[str, BMResult]

    def __post_init__(self) -> None:
        got = frozenset(self.results)
        want = frozenset(SURROGATE_BENCHMARK_KEYS)
        if got != want:
            raise ValueError(f"results keys must match SURROGATE_BENCHMARK_KEYS exactly; missing {want - got}, extra {got - want}")

    def print_table(self) -> None:
        print_synthetic_benchmark_table(self.results)
