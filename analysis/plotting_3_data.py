from pathlib import Path
from typing import NamedTuple

import numpy as np

import analysis.data_sets as ds
from analysis.data_locator import DataLocator
from analysis.plot_util import (
    collect_config_rows,
    normalize_results_and_exp_dir,
    uniq_int,
)


class _LoadedTraces(NamedTuple):
    dl_r: DataLocator
    tr_r: np.ndarray
    tr_dt_prop: np.ndarray
    tr_dt_total: np.ndarray


def _infer_params_from_configs(
    results_path: str,
    exp_dir: str,
    *,
    problem_seq: str,
    problem_batch: str,
    opt_names: list[str],
) -> dict[str, int]:
    """Infer (num_rounds_seq, num_rounds_batch, num_arms_batch, num_reps) from config.json."""
    results_path, exp_dir = normalize_results_and_exp_dir(results_path, exp_dir)
    root = Path(results_path) / exp_dir

    rows = collect_config_rows(root, opt_names, include_opt_name=False)

    seq = [r for r in rows if r["env_tag"] == problem_seq]
    batch = [r for r in rows if r["env_tag"] == problem_batch]

    out: dict[str, int] = {}
    nr_seq = uniq_int([r["num_rounds"] for r in seq])
    nr_batch = uniq_int([r["num_rounds"] for r in batch])
    na_batch = uniq_int([r["num_arms"] for r in batch])
    reps_seq = uniq_int([r["num_reps"] for r in seq])
    reps_batch = uniq_int([r["num_reps"] for r in batch])

    if nr_seq is not None:
        out["num_rounds_seq"] = nr_seq
    if nr_batch is not None:
        out["num_rounds_batch"] = nr_batch
    if na_batch is not None:
        out["num_arms_batch"] = na_batch

    reps = reps_seq if reps_seq is not None else reps_batch
    if reps is not None:
        out["num_reps"] = reps

    return out


def _load_traces(
    results_path: str,
    exp_dir: str,
    *,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
    key: str,
) -> tuple[DataLocator, np.ndarray]:
    results_path, exp_dir = normalize_results_and_exp_dir(results_path, exp_dir)
    dl = DataLocator(
        results_path,
        exp_dir,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        opt_names=opt_names,
        problems={problem},
        key=key,
    )
    traces = ds.load_multiple_traces(dl)
    return dl, traces


def _best_so_far(ys: np.ndarray) -> np.ndarray:
    # numpy.ma does not provide maximum.accumulate; handle masks explicitly.
    if np.ma.isMaskedArray(ys):
        data = ys.filled(-np.inf)
        acc = np.maximum.accumulate(data, axis=-1)
        acc[~np.isfinite(acc)] = np.nan
        return np.ma.array(acc, mask=np.ma.getmaskarray(ys))
    return np.maximum.accumulate(ys, axis=-1)


def _sum_traces(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise sum preserving masks when present."""
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        return np.ma.array(a) + np.ma.array(b)
    return a + b


def _mean_final_cum_dt_total_by_opt(traces_dt_total: np.ndarray, optimizers: list[str]) -> dict[str, float]:
    z = traces_dt_total.squeeze(0)  # [n_opt, n_rep, n_round]
    out: dict[str, float] = {}
    for i_opt, opt_name in enumerate(optimizers):
        dt = z[i_opt, ...]
        cum = np.ma.cumsum(dt, axis=-1) if np.ma.isMaskedArray(dt) else np.cumsum(dt, axis=-1)
        out[opt_name] = float(np.ma.mean(cum[:, -1]))
    return out


def _load_r_and_dt(
    results_path: str,
    exp_dir: str,
    *,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
):
    dl_r, tr_r = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="rreturn",
    )
    _, tr_dt_prop = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_prop",
    )
    _, tr_dt_eval = _load_traces(
        results_path,
        exp_dir,
        opt_names=opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_eval",
    )
    tr_dt_total = _sum_traces(tr_dt_prop, tr_dt_eval)
    return _LoadedTraces(dl_r=dl_r, tr_r=tr_r, tr_dt_prop=tr_dt_prop, tr_dt_total=tr_dt_total)
