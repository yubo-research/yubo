"""Trace loading and statistics functions for plotting_2 module."""

import json
from pathlib import Path

import numpy as np

import analysis.data_sets as ds
from analysis.data_io import data_is_done
from analysis.data_locator import DataLocator
from analysis.plot_util import normalize_results_and_exp_dir
from analysis.plotting_2_util import scan_experiment_configs


def load_rl_traces(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int | None,
    problem: str,
    key: str = "rreturn",
):
    results_path, exp_dir = normalize_results_and_exp_dir(results_path, exp_dir)
    root = Path(results_path) / exp_dir

    data_locator = DataLocator(
        results_path,
        exp_dir,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        opt_names=opt_names,
        problems={problem},
        problems_exact=True,
        key=key,
    )

    problems_found = data_locator.problems()
    if not problems_found:
        env_tags, opt_names_found = scan_experiment_configs(root)
        env_tags_sorted = sorted(env_tags)
        opt_names_sorted = sorted(opt_names_found)

        root_exists = root.exists()
        root_hint = ""
        if root_exists:
            try:
                entries = sorted([p.name for p in root.iterdir()])[:5]
                root_hint = f" root_entries[:5]={entries!r}"
            except Exception:
                root_hint = ""

        problem_present = any(problem in env for env in env_tags_sorted)
        if not problem_present:
            raise ValueError(
                "No matching problems found. "
                f"results_path={results_path!r}, exp_dir={exp_dir!r}, root={str(root)!r}, root_exists={root_exists}, "
                f"requested problem={problem!r}.{root_hint} "
                f"available_env_tags={env_tags_sorted!r} "
                "This usually means you're pointing at the wrong experiment directory for that problem."
            )

        raise ValueError(
            "No matching runs after applying filters. "
            f"results_path={results_path!r}, exp_dir={exp_dir!r}, root={str(root)!r}, root_exists={root_exists}, "
            f"requested problem={problem!r}, num_arms={num_arms}, num_rounds={num_rounds}, num_reps={num_reps}, "
            f"requested opt_names={opt_names!r}. "
            f"available_env_tags={env_tags_sorted!r} available_opt_names={opt_names_sorted!r}"
        )
    traces = ds.load_multiple_traces(data_locator)
    return data_locator, traces


def count_done_reps(trace_dir: str) -> int:
    p = Path(trace_dir)
    traces_subdir = p / "traces"
    if traces_subdir.exists():
        p = traces_subdir
    jsonl_files = sorted(p.glob("*.jsonl"))
    if jsonl_files:
        return int(sum(1 for fn in jsonl_files if data_is_done(str(fn))))
    n = 0
    for fn in sorted(p.iterdir()):
        if not fn.is_file():
            continue
        if fn.name.endswith(".done") or fn.name.endswith(".jsonl"):
            continue
        if data_is_done(str(fn)):
            n += 1
    return int(n)


def print_dataset_summary(
    results_path: str,
    exp_dir: str,
    *,
    problem: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int | None,
):
    data_locator = DataLocator(
        results_path,
        exp_dir,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        opt_names=opt_names,
        problems={problem},
        problems_exact=True,
        key="rreturn",
    )
    for opt in data_locator.optimizers():
        paths = data_locator(problem, opt)
        if not paths:
            continue
        trace_dir = paths[0]
        cfg_reps = None
        cfg_path = Path(trace_dir) / "config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg_reps = json.load(f).get("num_reps")
            except Exception:
                cfg_reps = None
        reps_done = count_done_reps(trace_dir)
        root = Path(trace_dir).name
        print(
            f"PLOT: env={problem} opt={opt} arms={num_arms} rounds={num_rounds} reps_done={reps_done} reps_cfg={cfg_reps} dir={root}"
        )


def mean_final_by_optimizer(
    data_locator: DataLocator, traces: np.ndarray
) -> dict[str, float]:
    """Return {opt_name: mean(final_value_over_reps)} for a single-problem trace tensor."""
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)
    if z.ndim != 3 or z.shape[2] == 0:
        raise ValueError(
            f"Empty traces: shape={getattr(z, 'shape', None)} for key={getattr(data_locator, 'key', None)}"
        )
    out: dict[str, float] = {}
    for i_opt, opt_name in enumerate(optimizers):
        y_final = z[i_opt, :, -1]
        try:
            out[opt_name] = float(np.ma.mean(y_final))
        except Exception:
            out[opt_name] = float(np.mean(np.asarray(y_final, dtype=float)))
    return out


def median_final_by_optimizer(
    data_locator: DataLocator, traces: np.ndarray
) -> dict[str, float]:
    """Return {opt_name: median(final_value_over_reps)} for a single-problem trace tensor."""
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)
    if z.ndim != 3 or z.shape[2] == 0:
        raise ValueError(
            f"Empty traces: shape={getattr(z, 'shape', None)} for key={getattr(data_locator, 'key', None)}"
        )
    out: dict[str, float] = {}
    for i_opt, opt_name in enumerate(optimizers):
        y_final = z[i_opt, :, -1]
        try:
            out[opt_name] = float(np.ma.median(y_final))
        except Exception:
            out[opt_name] = float(np.median(np.asarray(y_final, dtype=float)))
    return out


def normalized_ranks_0_1(scores_1d: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores_1d, dtype=float)
    bad = ~np.isfinite(scores)
    if np.any(bad):
        scores = scores.copy()
        scores[bad] = -np.inf

    n = int(scores.shape[0])
    if n <= 1:
        return np.ones_like(scores, dtype=float)

    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    for i_rank, i_opt in enumerate(order, start=1):
        ranks[i_opt] = float(i_rank)

    return 1.0 - (ranks - 1.0) / float(n - 1)


def mean_normalized_rank_score_by_optimizer(
    data_locator: DataLocator,
    traces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)
    n_opt = int(z.shape[0])
    n_rep = int(z.shape[1])
    n_round = int(z.shape[2])

    scores_by_rep = np.full((n_opt, n_rep), np.nan, dtype=float)
    for i_rep in range(n_rep):
        norm_ranks = np.full((n_opt, n_round), np.nan, dtype=float)
        for t in range(n_round):
            norm_ranks[:, t] = normalized_ranks_0_1(z[:, i_rep, t])
        scores_by_rep[:, i_rep] = np.nanmean(norm_ranks, axis=1)

    means = np.nanmean(scores_by_rep, axis=1)
    stes = np.nanstd(scores_by_rep, axis=1) / np.sqrt(float(n_rep))
    assert means.shape == (len(optimizers),)
    assert stes.shape == (len(optimizers),)
    return means, stes


def cum_dt_prop_from_dt_prop_traces(dt_prop_traces: np.ndarray) -> np.ndarray:
    """Convert dt_prop traces into cumulative dt_prop traces."""
    z = dt_prop_traces.squeeze(0)
    if np.ma.isMaskedArray(z):
        z_cum = np.ma.cumsum(z, axis=-1)
    else:
        z_cum = np.cumsum(z, axis=-1)
    return np.expand_dims(z_cum, axis=0)


def print_cum_dt_prop(
    cum_dt_prop_final_by_opt: dict[str, float] | None,
    opt_order: list[str] | None,
    *,
    header: str,
) -> None:
    if not cum_dt_prop_final_by_opt:
        return
    order = opt_order or sorted(cum_dt_prop_final_by_opt.keys())
    items = [
        f"\\texttt{{{o}}} {cum_dt_prop_final_by_opt[o]:.1f}s"
        for o in order
        if o in cum_dt_prop_final_by_opt
    ]
    if not items:
        return
    print(header)
    print("  " + ", ".join(items))


def load_cum_dt_prop(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    *,
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
) -> tuple[DataLocator, np.ndarray]:
    """Load dt_prop and compute cum_dt_prop = cumsum(dt_prop) over the run."""
    data_locator_dt, traces_dt = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_prop",
    )
    traces_cum = cum_dt_prop_from_dt_prop_traces(traces_dt)
    return data_locator_dt, traces_cum


def best_so_far(traces: np.ndarray) -> np.ndarray:
    z = traces
    if np.ma.isMaskedArray(z):
        z2 = z.filled(np.nan)
        z2 = np.maximum.accumulate(z2, axis=-1)
        return np.ma.masked_invalid(z2)
    z2 = np.asarray(z, dtype=float)
    z2 = np.maximum.accumulate(z2, axis=-1)
    return z2


def cum_time_from_dt(dt_prop: np.ndarray, dt_eval: np.ndarray) -> np.ndarray:
    z = dt_prop + dt_eval
    if np.ma.isMaskedArray(z):
        return np.ma.cumsum(z, axis=-1)
    return np.cumsum(np.asarray(z, dtype=float), axis=-1)


def interp_1d(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 2:
        return np.full_like(xq, np.nan, dtype=float)
    xo = x[ok]
    yo = y[ok]
    order = np.argsort(xo, kind="mergesort")
    xo = xo[order]
    yo = yo[order]
    xo, uniq = np.unique(xo, return_index=True)
    yo = yo[uniq]
    if xo.shape[0] < 2:
        return np.full_like(xq, np.nan, dtype=float)
    xq = np.asarray(xq, dtype=float)
    xq_clip = np.clip(xq, xo[0], xo[-1])
    return np.interp(xq_clip, xo, yo)
