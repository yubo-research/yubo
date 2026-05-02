"""Metrics and end-to-end surrogate benchmarks (explicit synthetic targets)."""

from __future__ import annotations

from .evaluate_class import SyntheticSineSurrogateBenchmark
from .evaluate_draw import (
    SYNTHETIC_BENCHMARK_N_TEST,
    draw_benchmark_synthetic_xy,
    synthetic_benchmark_data_seed,
)
from .evaluate_metrics import (
    SURROGATE_BENCHMARK_KEYS,
    SURROGATE_BENCHMARK_ROWS,
    SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME,
    BMResult,
    MuSe,
    _mean_and_sem,
    env_action_coords_to_surrogate_unit_x,
    normalize_benchmark_function_name,
    normalized_rmse,
    predictive_gaussian_log_likelihood,
)
from .evaluate_triples import (
    _surrogate_metric_triples_from_tensors,
    aggregate_surrogate_replicates,
    benchmark_single_surrogate,
)

__all__ = [
    "BMResult",
    "MuSe",
    "SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME",
    "SURROGATE_BENCHMARK_ROWS",
    "SURROGATE_BENCHMARK_KEYS",
    "SyntheticSineSurrogateBenchmark",
    "_mean_and_sem",
    "benchmark_single_surrogate_with_data",
    "benchmark_synthetic_sine_surrogates",
    "SYNTHETIC_BENCHMARK_N_TEST",
    "draw_benchmark_synthetic_xy",
    "env_action_coords_to_surrogate_unit_x",
    "normalize_benchmark_function_name",
    "normalized_rmse",
    "predictive_gaussian_log_likelihood",
    "synthetic_benchmark_data_seed",
]


def benchmark_synthetic_sine_surrogates(
    *,
    N: int,
    D: int,
    function_name: str,
    problem_seed: int = 0,
    num_reps: int = 1,
    b_fast_only: bool = False,
) -> SyntheticSineSurrogateBenchmark:
    """Run ENN, SMAC RF, DNGO, exact GP, two SVGP variants, and RF Vecchia on synthetic data in ``D`` dims.

    If ``b_fast_only`` is true, only ENN and SMAC RF are fit; DNGO, exact GP, both SVGPs, and Vecchia
    are skipped and their metrics are set to ``nan`` (same as an unavailable surrogate row).

    ``function_name`` is **required** (non-empty after strip). Use
    :data:`SYNTHETIC_BENCHMARK_SINE_FUNCTION_NAME` (``\"sine\"``) for the FittingTime
    target: ``x ~ U(-1,1)^{N×D}``, ``y = mean(sin(2π (x+1)/2)) + 0.1 ε`` (equivalently
    ``x_u ~ U(0,1)`` with ``y = mean(sin(2π x_u)) + noise``).

    Any other name builds ``f"f:{function_name}-{D}d"`` via
    :mod:`problems.pure_functions`, draws ``x ~ U(-1,1)^{N×D}``, and sets ``y`` to the
    environment reward plus ``0.1 ε``. The test draw has a fixed size
    :data:`SYNTHETIC_BENCHMARK_N_TEST` (not ``N``).
    Fitted surrogates use ``(x+1)/2`` in ``[0,1]`` (ENN and SMAC only
    when ``b_fast_only``); metrics always use the original ``y`` / ``y_test`` from the env draw.

    **Replicates:** for ``num_reps`` > 1, the full benchmark is run ``num_reps`` times with
    a deterministic data seed derived from ``(problem_seed, function_name, rep_index)``.
    Each replicate gets a fresh synthetic draw, and different functions with the same
    base ``problem_seed`` also get different draws. Returned :class:`BMResult` fields are
    the sample mean and standard error of the mean (finite values only; NaNs omitted from
    mean/SEM, or all-NaN → NaN mean and SEM).

    Reproducible RNG within one replicate: :func:`synthetic_benchmark_data_seed` determines
    the train draw seed, and ``+1`` is used for the test draw inside
    :func:`draw_benchmark_synthetic_xy`. The same draw is then shared across all surrogates
    evaluated for that replicate.

    If the SMAC RF path is unavailable or raises (e.g. missing ``smac``, bad install,
    runtime fit failure), SMAC RF fields are set to ``nan``.

    **LogLik** (nats): ``sum_i log N(y_test_i | y_hat_i, v_i)`` with **predictive**
    variance aligned to noisy ``y_test`` (``0.1^2`` observation noise in the draw).
    ENN uses epistemic ``se_i^2`` plus ``0.1^2``. SMAC RF uses forest variance plus
    ``0.1^2``. DNGO uses BLR predictive variance (includes learned ``1/\\beta``).
    Exact GP and both SVGP variants use ``posterior(..., observation_noise=True)``.
    Vecchia uses ``pyvecch`` posterior variance on the original ``y`` scale.
    If ``pyvecch`` is missing or fit/predict fails, Vecchia fields are ``nan`` (same pattern as
    optional SMAC RF). For :func:`~analysis.fitting_time.fitting_time.fit_vecchia` only, set
    ``YUBO_ALLOW_PYVECCH_ON_DARWIN=0`` on macOS to return NaNs if import still crashes in your
    environment.
    """
    if num_reps < 1:
        raise ValueError("num_reps must be >= 1")

    from analysis.fitting_time.fitting_time import (
        fit_dngo,
        fit_enn,
        fit_exact_gp,
        fit_smac_rf,
        fit_svgp_default,
        fit_svgp_linear,
        fit_vecchia,
    )

    rows: list[dict[str, tuple[float, float, float]]] = []
    for rep in range(num_reps):
        seed = synthetic_benchmark_data_seed(
            function_name=function_name,
            problem_seed=problem_seed,
            rep_index=rep,
        )
        x, y, x_test, y_test = draw_benchmark_synthetic_xy(N=N, D=D, function_name=function_name, problem_seed=seed)
        rows.append(
            _surrogate_metric_triples_from_tensors(
                x,
                y,
                x_test,
                y_test,
                fit_enn=fit_enn,
                fit_smac_rf=fit_smac_rf,
                fit_dngo=fit_dngo,
                fit_exact_gp=fit_exact_gp,
                fit_svgp_default=fit_svgp_default,
                fit_svgp_linear=fit_svgp_linear,
                fit_vecchia=fit_vecchia,
                b_fast_only=b_fast_only,
            )
        )
    return aggregate_surrogate_replicates(rows)


def benchmark_single_surrogate_with_data(
    *,
    N: int,
    D: int,
    function_name: str,
    surrogate_key: str,
    data_seed: int,
) -> tuple[float, float, float]:
    """Benchmark a single surrogate on synthetic data and return (fit_seconds, normalized_rmse, log_likelihood).

    Unlike :func:`benchmark_synthetic_sine_surrogates`, this runs only ONE surrogate. The caller
    is responsible for iterating over surrogates and replicates.

    Args:
        N: Number of training points.
        D: Dimensionality.
        function_name: Target function (e.g. "sphere", "rosenbrock").
        surrogate_key: One of :data:`SURROGATE_BENCHMARK_KEYS` (e.g. "enn", "vecchia").
        data_seed: Seed for the data draw (use :func:`synthetic_benchmark_data_seed` to compute).

    Returns:
        Tuple of (fit_seconds, normalized_rmse, log_likelihood).
    """
    x, y, x_test, y_test = draw_benchmark_synthetic_xy(N=N, D=D, function_name=function_name, problem_seed=data_seed)
    return benchmark_single_surrogate(x, y, x_test, y_test, surrogate_key)
