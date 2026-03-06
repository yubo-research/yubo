"""Main plotting module for RL experiment visualization."""

import matplotlib.pyplot as plt
import numpy as np

import analysis.plotting as ap
from analysis.data_locator import DataLocator
from analysis.plotting_2_trace import (
    best_so_far,
    cum_dt_prop_from_dt_prop_traces,
    cum_time_from_dt,
    interp_1d,
    load_cum_dt_prop,
    load_rl_traces,
    mean_final_by_optimizer,
    mean_normalized_rank_score_by_optimizer,
    median_final_by_optimizer,
    print_cum_dt_prop,
    print_dataset_summary,
)
from analysis.plotting_2_util import (
    consolidate_bottom_legend,
    get_denoise_value,
    infer_experiment_from_configs,
    infer_params_from_configs,
    noise_label,
    speedup_x_label,
)

from .plotting_types import (
    PlotResultsResult as _PlotResultsResult,
)
from .plotting_types import (
    PlotRLComparisonResult as _PlotRLComparisonResult,
)
from .plotting_types import (
    PlotRLExperimentResult as _PlotRLExperimentResult,
)
from .plotting_types import (
    PlotRLExperimentVsTimeResult as _PlotRLExperimentVsTimeResult,
)
from .plotting_types import (
    PlotRLFinalComparisonResult as _PlotRLFinalComparisonResult,
)
from .plotting_types import (
    RLTracesWithCumDtProp as _RLTracesWithCumDtProp,
)

# Re-export for backward compatibility
_noise_label = noise_label
_speedup_x_label = speedup_x_label
_consolidate_bottom_legend = consolidate_bottom_legend
_get_denoise_value = get_denoise_value
_scan_experiment_configs = None  # Not re-exported
_infer_params_from_configs = infer_params_from_configs
_count_done_reps = None  # Not re-exported
_print_dataset_summary = print_dataset_summary
_mean_final_by_optimizer = mean_final_by_optimizer
_median_final_by_optimizer = median_final_by_optimizer


def _load_rl_with_cum_dt_prop(results_path, exp_dir, opt_names, *, num_arms, num_rounds, num_reps, problem):
    data_locator, traces = load_rl_traces(
        results_path,
        exp_dir,
        opt_names,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
        problem=problem,
    )
    cum_dt_prop = None
    try:
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
        cum_dt_prop = median_final_by_optimizer(data_locator_dt, cum_dt_prop_from_dt_prop_traces(traces_dt))
    except ValueError:
        cum_dt_prop = None
    return _RLTracesWithCumDtProp(data_locator=data_locator, traces=traces, cum_dt_prop=cum_dt_prop)


def _try_load_rl_with_cum_dt_prop(results_path, exp_dir, opt_names, *, num_arms, num_rounds, num_reps, problem):
    try:
        return _load_rl_with_cum_dt_prop(
            results_path,
            exp_dir,
            opt_names,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
        )
    except ValueError:
        return None


def _print_cum_dt_props(
    cum_dt_prop_seq,
    cum_dt_prop_batch,
    *,
    opt_names_seq,
    opt_names_batch,
    opt_names_all,
    problem_seq,
    problem_batch,
):
    print_cum_dt_prop(
        cum_dt_prop_seq,
        opt_names_all or opt_names_seq,
        header=f"cumulative proposal times ({problem_seq})",
    )
    print_cum_dt_prop(
        cum_dt_prop_batch,
        opt_names_all or opt_names_batch,
        header=f"cumulative proposal times ({problem_batch})",
    )


_normalized_ranks_0_1 = None  # Not re-exported
_mean_normalized_rank_score_by_optimizer = mean_normalized_rank_score_by_optimizer
_cum_dt_prop_from_dt_prop_traces = cum_dt_prop_from_dt_prop_traces
_print_cum_dt_prop = print_cum_dt_prop
_load_cum_dt_prop = load_cum_dt_prop
_best_so_far = best_so_far
_cum_time_from_dt = cum_time_from_dt
_interp_1d = interp_1d


def plot_learning_curves(
    ax,
    data_locator: DataLocator,
    traces: np.ndarray,
    num_arms: int = 1,
    title: str = None,
    xlabel: str = "N",
    ylabel: str = "$y_{best}$",
    markersize: int = 5,
    cum_dt_prop_final_by_opt: dict[str, float] | None = None,
    x_start: int = 1,
    opt_names_all: list[str] | None = None,
    show_title: bool = True,
):
    optimizers = data_locator.optimizers()
    z = traces.squeeze(0)

    for i_opt, opt_name in enumerate(optimizers):
        y = z[i_opt, ...]
        x = num_arms * (int(x_start) + np.arange(y.shape[1]))
        style_idx = opt_names_all.index(opt_name) if opt_names_all and opt_name in opt_names_all else i_opt
        if opt_name.startswith("turbo-enn"):
            color = "#333333"
            marker = "s"
        else:
            color = ap.colors[style_idx]
            marker = ap.markers[style_idx]
        label = opt_name
        if cum_dt_prop_final_by_opt is not None and opt_name in cum_dt_prop_final_by_opt:
            label = f"{opt_name} ({cum_dt_prop_final_by_opt[opt_name]:.1f}s)"
        ap.filled_err(
            x=x,
            ys=y,
            ax=ax,
            se=True,
            alpha=0.25,
            color=color,
            color_line=color,
            label=label,
            marker=marker,
            max_markers=10,
            markersize=markersize,
        )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if show_title and title:
        ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)


def plot_final_performance(
    ax,
    data_locator: DataLocator,
    traces: np.ndarray,
    title: str = None,
    ylabel: str = "Mean normalized rank",
    opt_names_all: list[str] | None = None,
    show_title: bool = True,
):
    optimizers = data_locator.optimizers()
    means, stes = mean_normalized_rank_score_by_optimizer(data_locator, traces)

    colors = []
    for opt_name in optimizers:
        if opt_name.startswith("turbo-enn"):
            colors.append("#333333")
        elif opt_names_all and opt_name in opt_names_all:
            colors.append(ap.colors[opt_names_all.index(opt_name)])
        else:
            colors.append(ap.colors[optimizers.index(opt_name)])

    x_pos = np.arange(len(optimizers))
    ax.bar(x_pos, means, yerr=2 * stes, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(optimizers, rotation=45, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    if show_title and title:
        ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-0.05, 1.05)


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

    ax.set_xlabel("Cumulative time (s)", fontsize=12)
    ax.set_ylabel("Return (best so far)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
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


def plot_rl_comparison(
    results_path: str,
    exp_dir: str,
    opt_names_seq: list[str],
    opt_names_batch: list[str],
    problem_seq: str,
    problem_batch: str,
    num_reps: int | None = None,
    num_rounds_seq: int = 100,
    num_rounds_batch: int = 30,
    num_arms_seq: int = 1,
    num_arms_batch: int = 50,
    suptitle: str = None,
    figsize: tuple = (12, 5),
    cum_dt_prop: bool = False,
    opt_names_all: list[str] | None = None,
    show_titles: bool = True,
    print_titles: bool = False,
):
    seq = _load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
    )
    batch = _try_load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_batch,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
        problem=problem_batch,
    )
    data_locator_seq, traces_seq, cum_dt_prop_seq = (
        seq.data_locator,
        seq.traces,
        seq.cum_dt_prop,
    )
    data_locator_batch = None if batch is None else batch.data_locator
    traces_batch = None if batch is None else batch.traces
    cum_dt_prop_batch = None if batch is None else batch.cum_dt_prop

    _print_cum_dt_props(
        cum_dt_prop_seq,
        cum_dt_prop_batch,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        opt_names_all=opt_names_all,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
    )

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    denoise_seq = get_denoise_value(data_locator_seq, problem_seq)
    speedup_seq = speedup_x_label(cum_dt_prop_seq, problem_seq)
    line1_seq = f"{noise_label(problem_seq)}, {speedup_seq}" if speedup_seq else noise_label(problem_seq)
    parts_seq = [f"num_arms = {num_arms_seq}"]
    if denoise_seq is not None:
        parts_seq.append(f"{'num_denoise_obs' if problem_seq.endswith(':fn') else 'num_denoise_passive'} = {denoise_seq}")
    title_seq = f"{line1_seq}\n{', '.join(parts_seq)}"
    if print_titles:
        print(title_seq)
    plot_learning_curves(
        axs[0],
        data_locator_seq,
        traces_seq,
        num_arms=num_arms_seq,
        title=title_seq,
        cum_dt_prop_final_by_opt=cum_dt_prop_seq,
        opt_names_all=opt_names_all or opt_names_seq,
        show_title=show_titles,
    )

    if data_locator_batch is not None and traces_batch is not None:
        denoise_batch = get_denoise_value(data_locator_batch, problem_batch)
        speedup_batch = speedup_x_label(cum_dt_prop_batch, problem_batch)
        line1_batch = f"{noise_label(problem_batch)}, {speedup_batch}" if speedup_batch else noise_label(problem_batch)
        parts_batch = [f"num_arms = {num_arms_batch}"]
        if denoise_batch is not None:
            parts_batch.append(f"{'num_denoise_obs' if problem_batch.endswith(':fn') else 'num_denoise_passive'} = {denoise_batch}")
        title_batch = f"{line1_batch}\n{', '.join(parts_batch)}"
        if print_titles:
            print(title_batch)
        plot_learning_curves(
            axs[1],
            data_locator_batch,
            traces_batch,
            num_arms=num_arms_batch,
            title=title_batch,
            cum_dt_prop_final_by_opt=cum_dt_prop_batch,
            opt_names_all=opt_names_all or opt_names_batch,
            show_title=show_titles,
        )
    else:
        axs[1].axis("off")

    axs[0].set_xlabel("N", fontsize=16)
    if axs[1].axison:
        axs[1].set_xlabel("N", fontsize=16)
        axs[1].set_ylabel("")
        axs[1].tick_params(axis="y", labelleft=False, labelsize=16)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    consolidate_bottom_legend(fig, axs, fontsize=16, ncol=6)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    return _PlotRLComparisonResult(fig=fig, axs=axs, seq=seq, batch=batch)


def plot_rl_final_comparison(
    results_path: str,
    exp_dir: str,
    opt_names_seq: list[str],
    opt_names_batch: list[str],
    problem_seq: str,
    problem_batch: str,
    num_reps: int | None = None,
    num_rounds_seq: int = 100,
    num_rounds_batch: int = 30,
    num_arms_seq: int = 1,
    num_arms_batch: int = 50,
    suptitle: str = None,
    figsize: tuple = (14, 5),
    opt_names_all: list[str] | None = None,
    show_titles: bool = True,
    print_titles: bool = False,
):
    seq = _load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
        problem=problem_seq,
    )
    batch = _try_load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_batch,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
        problem=problem_batch,
    )
    data_locator_seq, traces_seq, cum_dt_prop_seq = (
        seq.data_locator,
        seq.traces,
        seq.cum_dt_prop,
    )
    data_locator_batch = None if batch is None else batch.data_locator
    traces_batch = None if batch is None else batch.traces
    cum_dt_prop_batch = None if batch is None else batch.cum_dt_prop

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    denoise_seq = get_denoise_value(data_locator_seq, problem_seq)
    speedup_seq = speedup_x_label(cum_dt_prop_seq, problem_seq)
    line1_seq = f"{noise_label(problem_seq)}, {speedup_seq}" if speedup_seq else noise_label(problem_seq)
    parts_seq = [f"num_arms = {num_arms_seq}"]
    if denoise_seq is not None:
        parts_seq.append(f"{'num_denoise' if problem_seq.endswith(':fn') else 'num_denoise_passive'} = {denoise_seq}")
    title_seq = f"{line1_seq}\n{', '.join(parts_seq)}"
    if print_titles:
        print(title_seq)
    plot_final_performance(
        axs[0],
        data_locator_seq,
        traces_seq,
        title=title_seq,
        opt_names_all=opt_names_all or opt_names_seq,
        show_title=show_titles,
    )

    if data_locator_batch is not None and traces_batch is not None:
        denoise_batch = get_denoise_value(data_locator_batch, problem_batch)
        speedup_batch = speedup_x_label(cum_dt_prop_batch, problem_batch)
        line1_batch = f"{noise_label(problem_batch)}, {speedup_batch}" if speedup_batch else noise_label(problem_batch)
        parts_batch = [f"num_arms = {num_arms_batch}"]
        if denoise_batch is not None:
            parts_batch.append(f"{'num_denoise' if problem_batch.endswith(':fn') else 'num_denoise_passive'} = {denoise_batch}")
        title_batch = f"{line1_batch}\n{', '.join(parts_batch)}"
        if print_titles:
            print(title_batch)
        plot_final_performance(
            axs[1],
            data_locator_batch,
            traces_batch,
            title=title_batch,
            opt_names_all=opt_names_all or opt_names_batch,
            show_title=show_titles,
        )
    else:
        axs[1].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    plt.tight_layout()

    return _PlotRLFinalComparisonResult(fig=fig, axs=axs, seq=seq, batch=batch)


_long_names = {
    "tlunar": "LunarLander-v3",
    "push": "Push-v3",
    "hop": "Hopper-v5",
    "bw-heur": "BipedalWalker-v3",
    "dna": "LASSO-DNA",
}


def plot_results(
    results_path: str,
    exp_dir: str,
    opt_names: list[str],
    problem: str,
    num_reps: int | None = None,
    num_rounds_seq: int | None = None,
    num_rounds_batch: int | None = None,
    num_arms_seq: int | None = None,
    num_arms_batch: int | None = None,
    exclude_seq: list[str] | None = None,
    exclude_batch: list[str] | None = None,
):
    problem_name = _long_names.get(problem, problem)
    problem_seq, problem_batch = problem, f"{problem}:fn"
    opt_names_seq = [o for o in opt_names if o not in (exclude_seq or [])]
    opt_names_batch = [o for o in opt_names if o not in (exclude_batch or [])]

    inferred = infer_params_from_configs(
        results_path,
        exp_dir,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        opt_names=opt_names,
    )
    num_reps = num_reps or inferred.get("num_reps")
    num_rounds_seq = num_rounds_seq or inferred.get("num_rounds_seq", 100)
    num_rounds_batch = num_rounds_batch or inferred.get("num_rounds_batch", 30)
    num_arms_seq = num_arms_seq or inferred.get("num_arms_seq", 1)
    num_arms_batch = num_arms_batch or inferred.get("num_arms_batch", 50)

    print_dataset_summary(
        results_path,
        exp_dir,
        problem=problem_seq,
        opt_names=opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps,
    )
    print_dataset_summary(
        results_path,
        exp_dir,
        problem=problem_batch,
        opt_names=opt_names_batch,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps,
    )

    print(problem_name)
    fig_curves, axs_curves, seq_data, batch_data = plot_rl_comparison(
        results_path,
        exp_dir,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        num_reps=num_reps,
        num_rounds_seq=num_rounds_seq,
        num_rounds_batch=num_rounds_batch,
        num_arms_seq=num_arms_seq,
        num_arms_batch=num_arms_batch,
        suptitle=None,
        cum_dt_prop=problem in {"tlunar", "push"},
        opt_names_all=opt_names,
        show_titles=False,
        print_titles=True,
    )

    print(f"{problem_name} Final Performance Comparison (±2 SE)")
    fig_final, axs_final, _, _ = plot_rl_final_comparison(
        results_path,
        exp_dir,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        num_reps=num_reps,
        num_rounds_seq=num_rounds_seq,
        num_rounds_batch=num_rounds_batch,
        num_arms_seq=num_arms_seq,
        num_arms_batch=num_arms_batch,
        suptitle=None,
        opt_names_all=opt_names,
        show_titles=False,
        print_titles=True,
    )

    return _PlotResultsResult(
        curves=(fig_curves, axs_curves),
        final=(fig_final, axs_final),
        seq_data=seq_data,
        batch_data=batch_data,
    )


def compute_pareto_data(
    results_path: str,
    exp_dir: dict[str, str],
    opt_names: list[str],
    baseline_opt: str = "turbo-one",
    exclude_opts: list[str] | None = None,
    mode: str = "both",
):
    if exclude_opts is None:
        exclude_opts = ["random"]
    opt_names_filtered = [o for o in opt_names if o not in exclude_opts]

    all_returns, all_times = _collect_returns_times(results_path, exp_dir, opt_names_filtered, mode)
    return _normalize_returns_times(all_returns, all_times, opt_names_filtered, baseline_opt)


def _collect_returns_times(results_path, exp_dir, opt_names_filtered, mode):
    all_returns, all_times = {}, {}
    for problem, exp in exp_dir.items():
        problem_seq, problem_batch = problem, f"{problem}:fn"
        inferred = infer_params_from_configs(
            results_path,
            exp,
            problem_seq=problem_seq,
            problem_batch=problem_batch,
            opt_names=opt_names_filtered,
        )
        num_reps = inferred.get("num_reps")
        num_rounds_seq, num_rounds_batch = (
            inferred.get("num_rounds_seq", 100),
            inferred.get("num_rounds_batch", 30),
        )
        num_arms_seq, num_arms_batch = (
            inferred.get("num_arms_seq", 1),
            inferred.get("num_arms_batch", 50),
        )

        configs_to_load = []
        if mode in ("seq", "both"):
            configs_to_load.append((f"{problem}_seq", problem_seq, num_arms_seq, num_rounds_seq))
        if mode in ("batch", "both"):
            configs_to_load.append((f"{problem}_batch", problem_batch, num_arms_batch, num_rounds_batch))

        for label, prob, num_arms, num_rounds in configs_to_load:
            _try_update_returns_times(
                results_path,
                exp,
                opt_names_filtered,
                num_arms=num_arms,
                num_rounds=num_rounds,
                num_reps=num_reps,
                problem=prob,
                label=label,
                all_returns=all_returns,
                all_times=all_times,
            )
    return all_returns, all_times


def _try_update_returns_times(
    results_path,
    exp,
    opt_names_filtered,
    *,
    num_arms,
    num_rounds,
    num_reps,
    problem,
    label,
    all_returns,
    all_times,
):
    try:
        data_locator, traces = load_rl_traces(
            results_path,
            exp,
            opt_names_filtered,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
        )
        all_returns[label] = mean_final_by_optimizer(data_locator, traces)
    except ValueError:
        pass
    try:
        data_locator_dt, traces_dt = load_rl_traces(
            results_path,
            exp,
            opt_names_filtered,
            num_arms=num_arms,
            num_rounds=num_rounds,
            num_reps=num_reps,
            problem=problem,
            key="dt_prop",
        )
        all_times[label] = mean_final_by_optimizer(data_locator_dt, cum_dt_prop_from_dt_prop_traces(traces_dt))
    except ValueError:
        pass


def _normalize_returns_times(all_returns, all_times, opt_names_filtered, baseline_opt):
    problems = sorted(set(all_returns.keys()) & set(all_times.keys()))
    opts_in_data = set()
    for p in problems:
        opts_in_data.update(all_returns[p].keys())
        opts_in_data.update(all_times[p].keys())
    opts_in_data = [o for o in opt_names_filtered if o in opts_in_data]

    r_matrix = np.full((len(problems), len(opts_in_data)), np.nan)
    t_matrix = np.full((len(problems), len(opts_in_data)), np.nan)

    for i_p, p in enumerate(problems):
        for i_o, o in enumerate(opts_in_data):
            if o in all_returns.get(p, {}):
                r_matrix[i_p, i_o] = all_returns[p][o]
            if o in all_times.get(p, {}):
                t_matrix[i_p, i_o] = all_times[p][o]

    if baseline_opt not in opts_in_data:
        raise ValueError(f"Baseline optimizer {baseline_opt!r} not found in data. Available: {opts_in_data}")
    i_baseline = opts_in_data.index(baseline_opt)

    r_baseline = r_matrix[:, i_baseline : i_baseline + 1]
    r_centered = r_matrix - r_baseline
    rms = np.sqrt(np.nanmean(r_centered**2, axis=1, keepdims=True))
    rms = np.where(rms == 0, 1.0, rms)
    r_normalized = r_centered / rms

    t_baseline = t_matrix[:, i_baseline : i_baseline + 1]
    speedup = t_baseline / t_matrix

    return {
        "problems": problems,
        "opt_names": opts_in_data,
        "r_normalized": r_normalized,
        "speedup": speedup,
        "r_raw": r_matrix,
        "t_raw": t_matrix,
        "baseline_opt": baseline_opt,
    }
