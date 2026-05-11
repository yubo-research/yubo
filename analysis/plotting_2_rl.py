"""Single-experiment RL plots (learning curves vs rounds and vs wall time)."""

import matplotlib.pyplot as plt
import numpy as np

import analysis.plotting as ap
from analysis.plotting_2_curves import plot_learning_curves
from analysis.plotting_2_trace import (
    best_so_far,
    cum_time_from_dt,
    interp_1d,
    load_cum_dt_prop,
    load_rl_traces,
    mean_final_by_optimizer,
)
from analysis.plotting_2_util import infer_experiment_from_configs
from analysis.plotting_types import (
    PlotRLExperimentResult as _PlotRLExperimentResult,
)
from analysis.plotting_types import (
    PlotRLExperimentVsTimeResult as _PlotRLExperimentVsTimeResult,
)


def plot_rl_experiment(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
    title: str = None,
    figsize: tuple = (10, 6),
):
    data_locator, traces = load_rl_traces(results_path, exp_dir, opt_names, num_arms, num_rounds, num_reps, problem)
    cum_dt_prop_final_by_opt = None
    try:
        data_locator_dt, traces_cum = load_cum_dt_prop(
            results_path,
            exp_dir,
            opt_names,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
        )
        cum_dt_prop_final_by_opt = mean_final_by_optimizer(data_locator_dt, traces_cum)
    except ValueError:
        cum_dt_prop_final_by_opt = None

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_learning_curves(
        ax,
        data_locator,
        traces,
        num_arms=num_arms,
        title=title,
        cum_dt_prop_final_by_opt=cum_dt_prop_final_by_opt,
        opt_names_all=opt_names,
    )
    plt.tight_layout()
    return _PlotRLExperimentResult(fig=fig, ax=ax, data_locator=data_locator, traces=traces)


def _plot_rl_vs_time_one_rep(ax, *, ti, yi, color, marker, label):
    x, yy = np.asarray(ti[0], dtype=float), np.asarray(yi[0], dtype=float)
    ok = np.isfinite(x) & np.isfinite(yy)
    ax.plot(
        x[ok],
        yy[ok],
        color=color,
        label=label,
        marker=marker,
        markevery=max(1, int(np.sum(ok) / 10)),
    )


def _plot_rl_vs_time_multi_rep(ax, *, ti, yi, n_rep: int, color, marker, label, n_grid: int):
    t_ends = [
        float(np.nanmax(np.asarray(ti[r], dtype=float)[np.isfinite(np.asarray(ti[r], dtype=float))]))
        for r in range(n_rep)
        if np.any(np.isfinite(np.asarray(ti[r], dtype=float)))
    ]
    if not t_ends:
        return
    t_max = float(np.nanmin(t_ends))
    if not np.isfinite(t_max) or t_max <= 0:
        return
    xq = np.linspace(0.0, t_max, int(n_grid))

    yq = np.full((n_rep, xq.shape[0]), np.nan, dtype=float)
    for r in range(n_rep):
        yq[r, :] = interp_1d(np.asarray(ti[r], dtype=float), np.asarray(yi[r], dtype=float), xq)

    mu = np.nanmean(yq, axis=0)
    se = np.nanstd(yq, axis=0) / np.sqrt(float(n_rep))
    ax.plot(
        xq,
        mu,
        color=color,
        label=label,
        marker=marker,
        markevery=max(1, int(xq.shape[0] / 10)),
    )
    ax.fill_between(xq, mu - se, mu + se, color=color, alpha=0.25)


def plot_rl_experiment_vs_time(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    num_arms: int,
    num_rounds: int,
    num_reps: int,
    problem: str,
    title: str | None = None,
    figsize: tuple = (10, 6),
    n_grid: int = 200,
):
    data_locator_ret, traces_ret = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="rreturn",
    )
    _, traces_dt_prop = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_prop",
    )
    _, traces_dt_eval = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
        key="dt_eval",
    )

    y = best_so_far(traces_ret.squeeze(0))
    t = cum_time_from_dt(traces_dt_prop.squeeze(0), traces_dt_eval.squeeze(0))

    optimizers = data_locator_ret.optimizers()
    n_opt, n_rep = int(y.shape[0]), int(y.shape[1])
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for i_opt in range(n_opt):
        color, marker, label = ap.colors[i_opt], ap.markers[i_opt], optimizers[i_opt]
        ti, yi = t[i_opt, ...], y[i_opt, ...]

        if n_rep == 1:
            _plot_rl_vs_time_one_rep(ax, ti=ti, yi=yi, color=color, marker=marker, label=label)
            continue

        _plot_rl_vs_time_multi_rep(
            ax,
            ti=ti,
            yi=yi,
            n_rep=n_rep,
            color=color,
            marker=marker,
            label=label,
            n_grid=n_grid,
        )

    ax.set_xlabel("Cumulative time (s)")
    ax.set_ylabel("Return (best so far)")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return _PlotRLExperimentVsTimeResult(fig=fig, ax=ax, data_locator=data_locator_ret, traces=traces_ret, t=t)


def plot_rl_experiment_vs_time_auto(
    results_path: str,
    exp_dir: str,
    *,
    problem: str | None = None,
    opt_names: list[str] | None = None,
    num_arms: int | None = None,
    num_rounds: int | None = None,
    num_reps: int | None = None,
    title: str | None = None,
    figsize: tuple = (10, 6),
    n_grid: int = 200,
):
    info = infer_experiment_from_configs(results_path, exp_dir)
    results_path, exp_dir = info["results_path"], info["exp_dir"]

    if opt_names is None:
        opt_names = info["opt_names"]
    if not opt_names:
        raise ValueError(f"No opt_names found for results_path={results_path!r}, exp_dir={exp_dir!r}")

    if problem is None:
        env_tags = info["env_tags"]
        if len(env_tags) != 1:
            raise ValueError(f"Multiple env_tags found {env_tags!r}; pass problem= explicitly")
        problem = env_tags[0]

    num_arms = num_arms or info["num_arms"]
    num_rounds = num_rounds or info["num_rounds"]
    num_reps = num_reps or info["num_reps"]

    missing = [
        k
        for k, v in [
            ("num_arms", num_arms),
            ("num_rounds", num_rounds),
            ("num_reps", num_reps),
        ]
        if v is None
    ]
    if missing:
        raise ValueError(f"Couldn't infer {missing} from config.json; pass explicitly")

    return plot_rl_experiment_vs_time(
        results_path,
        exp_dir,
        opt_names,
        num_arms=int(num_arms),
        num_rounds=int(num_rounds),
        num_reps=int(num_reps),
        problem=problem,
        title=title,
        figsize=figsize,
        n_grid=n_grid,
    )
