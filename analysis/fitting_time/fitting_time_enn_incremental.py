"""Incremental ENN ``add()`` plus index sync timing on synthetic benchmark draws."""

from __future__ import annotations

import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence

import numpy as np

from .evaluate_metrics import (
    normalize_benchmark_function_name,
    predictive_gaussian_log_likelihood,
)
from .fitting_time import _ENN_POSTERIOR_CHUNK, _SYNTHETIC_OBS_VAR
from .fitting_time_enn_incremental_draw import (
    _train_xy_unit_cube_segment,
    draw_benchmark_test_xy_unit_cube,
)

ENN_INCREMENTAL_CHECKPOINT_NS: tuple[int, ...] = (
    1,
    3,
    10,
    30,
    100,
    300,
    1000,
    3000,
    10000,
    30000,
    100000,
    300000,
    1_000_000,
)


class EnnIncrementalIndexDriver(Enum):
    FLAT = "flat"
    HNSW = "hnsw"
    HNSW_DISK = "hnsw_disk"

    def to_enn_index_driver(self):
        from enn.turbo.config.enn_index_driver import ENNIndexDriver

        if self is EnnIncrementalIndexDriver.HNSW:
            return ENNIndexDriver.HNSW
        if self is EnnIncrementalIndexDriver.HNSW_DISK:
            return ENNIndexDriver.HNSW_DISK
        return ENNIndexDriver.FLAT


@contextmanager
def enn_disk_work_dir(index_driver: EnnIncrementalIndexDriver) -> Iterator[str | None]:
    """Yield a temp directory for disk-backed ENN; ``None`` for in-memory drivers."""
    if index_driver is not EnnIncrementalIndexDriver.HNSW_DISK:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="yubo_enn_hnsw_disk_") as work_dir:
        yield work_dir


def epistemic_nn_driver_kwargs(
    index_driver: EnnIncrementalIndexDriver,
    *,
    work_dir: str | None,
) -> dict:
    kwargs: dict = {"index_driver": index_driver.to_enn_index_driver()}
    if index_driver is EnnIncrementalIndexDriver.HNSW_DISK:
        if work_dir is None:
            raise ValueError("work_dir is required when index_driver is hnsw_disk")
        kwargs["work_dir"] = work_dir
        kwargs["enn_storage"] = "disk"
    return kwargs


@dataclass(frozen=True)
class EnnIncrementalTimingResult:
    n: tuple[int, ...]
    add_seconds: tuple[float, ...]
    log_likelihood: tuple[float, ...]
    target: str
    d: int
    problem_seed: int
    index_driver: EnnIncrementalIndexDriver


def enn_incremental_checkpoint_ns() -> tuple[int, ...]:
    return ENN_INCREMENTAL_CHECKPOINT_NS


def _checkpoint_enn_params(n_obs: int):
    from enn.enn.enn_params import ENNParams

    from .fitting_time import enn_fit_k_and_num_fit_samples

    k_eff, _ = enn_fit_k_and_num_fit_samples(n_obs)
    return ENNParams(
        k_num_neighbors=k_eff,
        epistemic_variance_scale=1.0,
        aleatoric_variance_scale=0.0,
    )


def _enn_posterior_mu_se(enn_model, x_test: np.ndarray, enn_params) -> tuple[np.ndarray, np.ndarray]:
    from enn.enn.enn_params import PosteriorFlags

    x_t = np.asarray(x_test, dtype=np.float64)
    n_test = int(x_t.shape[0])
    chunk = max(1, int(_ENN_POSTERIOR_CHUNK))
    flags = PosteriorFlags(observation_noise=False)
    if n_test <= chunk:
        post = enn_model.posterior(x_t, params=enn_params, flags=flags)
        y_hat = np.asarray(post.mu, dtype=np.float64).reshape(-1, 1)
        se = np.asarray(post.se, dtype=np.float64).reshape(-1, 1)
        return y_hat, se
    mu_parts: list[np.ndarray] = []
    se_parts: list[np.ndarray] = []
    for start in range(0, n_test, chunk):
        sl = slice(start, min(start + chunk, n_test))
        post_b = enn_model.posterior(x_t[sl], params=enn_params, flags=flags)
        mu_parts.append(np.asarray(post_b.mu, dtype=np.float64).reshape(-1, 1))
        se_parts.append(np.asarray(post_b.se, dtype=np.float64).reshape(-1, 1))
    return np.concatenate(mu_parts, axis=0), np.concatenate(se_parts, axis=0)


def enn_test_log_likelihood(
    enn_model,
    *,
    D: int,
    function_name: str,
    problem_seed: int,
    n_obs: int,
) -> float:
    target = normalize_benchmark_function_name(function_name)
    x_test, y_test = draw_benchmark_test_xy_unit_cube(D=int(D), function_name=target, problem_seed=int(problem_seed))
    params = _checkpoint_enn_params(int(n_obs))
    y_hat, se = _enn_posterior_mu_se(enn_model, x_test, params)
    pred_var = se**2 + _SYNTHETIC_OBS_VAR
    return predictive_gaussian_log_likelihood(y_test, y_hat, pred_var)


def benchmark_enn_incremental_add_timing(
    *,
    D: int,
    function_name: str,
    problem_seed: int,
    index_driver: EnnIncrementalIndexDriver = EnnIncrementalIndexDriver.FLAT,
    checkpoints: Sequence[int] | None = None,
) -> EnnIncrementalTimingResult:
    from enn.enn.enn_class import EpistemicNearestNeighbors

    target = normalize_benchmark_function_name(function_name)
    d = int(D)
    seed = int(problem_seed)
    ckpts = tuple(checkpoints) if checkpoints is not None else enn_incremental_checkpoint_ns()
    if len(ckpts) == 0:
        raise ValueError("checkpoints must be non-empty")
    prev_n = 0
    for n_chk in ckpts:
        if int(n_chk) <= prev_n:
            raise ValueError(f"checkpoints must be strictly increasing, got {ckpts}")

    yvar_row = np.array([[float(_SYNTHETIC_OBS_VAR)]], dtype=np.float64)

    ns: list[int] = []
    add_seconds: list[float] = []
    log_likelihood: list[float] = []

    with enn_disk_work_dir(index_driver) as work_dir:
        enn_model = EpistemicNearestNeighbors(
            np.zeros((0, d), dtype=np.float64),
            np.zeros((0, 1), dtype=np.float64),
            **epistemic_nn_driver_kwargs(index_driver, work_dir=work_dir),
        )

        for n_chk in ckpts:
            n_target = int(n_chk)
            x_seg, y_seg = _train_xy_unit_cube_segment(
                D=d,
                function_name=target,
                problem_seed=seed,
                n_train=n_target,
                start_row=prev_n,
            )
            t_0 = time.perf_counter()
            for i in range(x_seg.shape[0]):
                enn_model.add(x_seg[i : i + 1], y_seg[i : i + 1], yvar_row)
            enn_model.ensure_index_sync()
            add_seconds.append(time.perf_counter() - t_0)
            prev_n = n_target
            log_likelihood.append(
                enn_test_log_likelihood(
                    enn_model,
                    D=d,
                    function_name=target,
                    problem_seed=seed,
                    n_obs=n_target,
                )
            )
            ns.append(n_target)

    return EnnIncrementalTimingResult(
        n=tuple(ns),
        add_seconds=tuple(add_seconds),
        log_likelihood=tuple(log_likelihood),
        target=target,
        d=d,
        problem_seed=seed,
        index_driver=index_driver,
    )
