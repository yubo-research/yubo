"""Single-figure RL comparison plots that combine curves and bars."""

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
from analysis.plotting_types import PlotRLComparisonResult as _PlotRLComparisonResult


def _comparison_title(
    data_locator,
    problem: str,
    num_arms: int,
    cum_dt_prop: dict[str, float] | None,
    *,
    denoise_key_for_frozen: str,
) -> str:
    denoise = get_denoise_value(data_locator, problem)
    speedup = speedup_x_label(cum_dt_prop, problem)
    line1 = f"{noise_label(problem)}, {speedup}" if speedup else noise_label(problem)
    parts = [f"num_arms = {num_arms}"]
    if denoise is not None:
        key = denoise_key_for_frozen if problem.endswith(":fn") else "num_denoise_passive"
        parts.append(f"{key} = {denoise}")
    return f"{line1}\n{', '.join(parts)}"


def plot_rl_combined_comparison_from_data(
    seq,
    batch,
    *,
    problem_seq: str,
    problem_batch: str,
    num_arms_seq: int = 1,
    num_arms_batch: int = 50,
    suptitle: str = None,
    figsize: tuple = (12, 8),
    opt_names_seq: list[str] | None = None,
    opt_names_batch: list[str] | None = None,
    opt_names_all: list[str] | None = None,
    renames: dict[str, str] | None = None,
    show_titles: bool = True,
    print_titles: bool = False,
):
    data_locator_seq, traces_seq, cum_dt_prop_seq = (
        seq.data_locator,
        seq.traces,
        seq.cum_dt_prop,
    )
    data_locator_batch = None if batch is None else batch.data_locator
    traces_batch = None if batch is None else batch.traces
    cum_dt_prop_batch = None if batch is None else batch.cum_dt_prop

    fig, axs = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex=False,
        sharey="row",
        gridspec_kw={"height_ratios": [2.2, 1.35]},
    )

    title_seq = _comparison_title(
        data_locator_seq,
        problem_seq,
        num_arms_seq,
        cum_dt_prop_seq,
        denoise_key_for_frozen="num_denoise_obs",
    )
    if print_titles:
        print(title_seq)
    plot_learning_curves(
        axs[0, 0],
        data_locator_seq,
        traces_seq,
        num_arms=num_arms_seq,
        title=title_seq,
        cum_dt_prop_final_by_opt=cum_dt_prop_seq,
        opt_names_all=opt_names_all or opt_names_seq,
        renames=renames,
        show_title=show_titles,
    )
    plot_final_performance(
        axs[1, 0],
        data_locator_seq,
        traces_seq,
        title=None,
        opt_names_all=opt_names_all or opt_names_seq,
        renames=renames,
        show_title=False,
    )

    if data_locator_batch is not None and traces_batch is not None:
        title_batch = _comparison_title(
            data_locator_batch,
            problem_batch,
            num_arms_batch,
            cum_dt_prop_batch,
            denoise_key_for_frozen="num_denoise_obs",
        )
        if print_titles:
            print(title_batch)
        plot_learning_curves(
            axs[0, 1],
            data_locator_batch,
            traces_batch,
            num_arms=num_arms_batch,
            title=title_batch,
            cum_dt_prop_final_by_opt=cum_dt_prop_batch,
            opt_names_all=opt_names_all or opt_names_batch,
            renames=renames,
            show_title=show_titles,
        )
        plot_final_performance(
            axs[1, 1],
            data_locator_batch,
            traces_batch,
            title=None,
            opt_names_all=opt_names_all or opt_names_batch,
            renames=renames,
            show_title=False,
        )
    else:
        axs[0, 1].axis("off")
        axs[1, 1].axis("off")

    axs[0, 0].set_xlabel("N")
    axs[1, 0].set_xlabel("")
    axs[1, 0].set_ylabel("Mean normalized rank")
    if axs[0, 1].axison:
        axs[0, 1].set_xlabel("N")
        axs[0, 1].set_ylabel("")
        axs[0, 1].tick_params(axis="y", labelleft=False)
        axs[1, 1].set_xlabel("")
        axs[1, 1].set_ylabel("")
        axs[1, 1].tick_params(axis="y", labelleft=False)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    consolidate_bottom_legend(fig, [axs[0, 0], axs[0, 1]], ncol=3, renames=renames)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    return _PlotRLComparisonResult(fig=fig, axs=axs, seq=seq, batch=batch)


def plot_rl_combined_comparison(
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
    figsize: tuple = (12, 8),
    cum_dt_prop: bool = False,
    opt_names_all: list[str] | None = None,
    renames: dict[str, str] | None = None,
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
    _print_cum_dt_props(
        seq.cum_dt_prop,
        None if batch is None else batch.cum_dt_prop,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        opt_names_all=opt_names_all,
        renames=renames,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
    )
    return plot_rl_combined_comparison_from_data(
        seq,
        batch,
        problem_seq=problem_seq,
        problem_batch=problem_batch,
        num_arms_seq=num_arms_seq,
        num_arms_batch=num_arms_batch,
        suptitle=suptitle,
        figsize=figsize,
        opt_names_seq=opt_names_seq,
        opt_names_batch=opt_names_batch,
        opt_names_all=opt_names_all,
        renames=renames,
        show_titles=show_titles,
        print_titles=print_titles,
    )
