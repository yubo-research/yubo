"""Python-defined job lists for :func:`experiments.modal_synthetic_sine_benchmark.batch`.

Add a parameterless function that returns a list of :class:`SyntheticBenchJob`, then run::

    modal run experiments/modal_synthetic_sine_benchmark.py::batch \\
      --jobs-fn your_function_name --output-dir results/_xxx

``--jobs-fn`` must be a bare identifier (function name in this module).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticBenchJob:
    """One synthetic surrogate benchmark run (same fields as the single-job CLI)."""

    n: int
    d: int
    target: str
    problem_seed: int = 0


def example_sphere_n12_d2_seed0() -> list[SyntheticBenchJob]:
    """Same as ``--target sphere --n 12 --d 2 --problem-seed 0``."""
    return [SyntheticBenchJob(n=12, d=2, target="sphere", problem_seed=0)]


def example_two_targets_n12_d2() -> list[SyntheticBenchJob]:
    """Small demo: sphere and sine at ``N=12``, ``D=2``."""
    return [
        SyntheticBenchJob(n=12, d=2, target="sphere", problem_seed=0),
        SyntheticBenchJob(n=12, d=2, target="sine", problem_seed=0),
    ]


def job_fit_quality() -> list[SyntheticBenchJob]:
    jobs = []
    for n in [3000, 10000]:  # [30, 100, 300, 1000]:
        for d in [3, 10, 30, 100]:
            for fn in ["sphere", "ackley", "rosenbrock", "booth"]:
                jobs.append(SyntheticBenchJob(n=n, d=d, target=fn, problem_seed=17))
    return jobs
