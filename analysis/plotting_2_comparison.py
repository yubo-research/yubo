"""Side-by-side RL comparison plots (sequential vs batch)."""

import matplotlib.pyplot as plt

from analysis.plotting_2_curves import plot_final_performance, plot_learning_curves
from analysis.plotting_2_helpers import (
    _load_rl_with_cum_dt_prop,
    _print_cum_dt_props,
    _try_load_rl_with_cum_dt_prop,
)
from analysis.plotting_2_util import (
    consolidate_bottom_legend,
    get_denoise_value,
    noise_label,
    speedup_x_label,
)
from analysis.plotting_types import (
    PlotRLComparisonResult as _PlotRLComparisonResult,
)
from analysis.plotting_types import (
    PlotRLFinalComparisonResult as _PlotRLFinalComparisonResult,
)


def plot_rl_comparison(
    results_path: str,
    exp_dir: str,
    opt_names_seq: list[str],
    opt_names_batch: list[str],
    problem_seq: str,
    problem_batch: str,
    num_reps: int | None = None,
    num_reps_seq: int | None = None,
    num_reps_batch: int | None = None,
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
    _ = cum_dt_prop
    num_reps_seq = num_reps_seq if num_reps_seq is not None else num_reps
    num_reps_batch = num_reps_batch if num_reps_batch is not None else num_reps
    seq = _load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps_seq,
        problem=problem_seq,
    )
    batch = _try_load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_batch,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps_batch,
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

    axs[0].set_xlabel("N")
    if axs[1].axison:
        axs[1].set_xlabel("N")
        axs[1].set_ylabel("")
        axs[1].tick_params(axis="y", labelleft=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    consolidate_bottom_legend(fig, axs, ncol=3)
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
    num_reps_seq: int | None = None,
    num_reps_batch: int | None = None,
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
    num_reps_seq = num_reps_seq if num_reps_seq is not None else num_reps
    num_reps_batch = num_reps_batch if num_reps_batch is not None else num_reps
    seq = _load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_seq,
        num_arms=num_arms_seq,
        num_rounds=num_rounds_seq,
        num_reps=num_reps_seq,
        problem=problem_seq,
    )
    batch = _try_load_rl_with_cum_dt_prop(
        results_path,
        exp_dir,
        opt_names_batch,
        num_arms=num_arms_batch,
        num_rounds=num_rounds_batch,
        num_reps=num_reps_batch,
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
        fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()

    return _PlotRLFinalComparisonResult(fig=fig, axs=axs, seq=seq, batch=batch)
