from __future__ import annotations

import importlib
import re
import types

from analysis.fitting_time.evaluate import normalize_benchmark_function_name


def load_synthetic_sine_benchmark_jobs(
    jobs_fn: str,
    *,
    _batch_jobs_module: types.ModuleType | None = None,
) -> list[tuple[int, int, str, int]]:
    """Call ``jobs_fn`` on :mod:`analysis.fitting_time.batch_jobs`; return normalized rows.

    ``jobs_fn`` must be a single Python identifier naming a **parameterless** callable that
    returns a non-empty ``list`` of :class:`~analysis.fitting_time.batch_jobs.SyntheticBenchJob`.
    The ``_batch_jobs_module`` argument is for tests only (inject a fake module).
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", jobs_fn):
        raise ValueError("jobs_fn must be a single Python identifier (e.g. example_sphere_n12_d2_seed0)")
    mod = _batch_jobs_module
    if mod is None:
        mod = importlib.import_module("analysis.fitting_time.batch_jobs")
    job_callable = getattr(mod, jobs_fn, None)
    if job_callable is None or not callable(job_callable):
        raise ValueError(f"analysis.fitting_time.batch_jobs.{jobs_fn} is missing or not callable")
    job_cls = getattr(mod, "SyntheticBenchJob", None)
    if job_cls is None:
        raise ValueError("batch_jobs module must define SyntheticBenchJob")

    raw = job_callable()
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{jobs_fn}() must return a non-empty list")
    out: list[tuple[int, int, str, int]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, job_cls):
            raise TypeError(f"{jobs_fn}()[{i}] must be SyntheticBenchJob, got {type(item).__name__}")
        if item.n < 1 or item.d < 1:
            raise ValueError(f"{jobs_fn}()[{i}]: n and d must be positive, got n={item.n}, d={item.d}")
        fn = normalize_benchmark_function_name(item.target)
        out.append((item.n, item.d, fn, int(item.problem_seed)))
    return out
