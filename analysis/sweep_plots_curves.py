"""Single-column sweep plots: parameter errorbar figure and curve+bars figure."""

import matplotlib.pyplot as plt
import numpy as np

from analysis.sweep_plots_data import (
    _collect_plot_curves_data,
    _iter_matching_run_dirs,
    _load_traces_or_skip,
)
from analysis.sweep_plots_panels import _draw_plot_curves_panels
from analysis.sweep_plots_style import _default_results_dir


def plot_param_sweep(
    exp_dir,
    param_key="k",
    regex_pattern=r"k=(\d+)",
    xlabel=r"$K$",
    title="TuRBO-ENN Sweep",
    param_name_for_print=None,
    trace_key="rreturn",
    results_dir=None,
    use_mean_over_rounds=False,
):
    """
    Generalized function to load and plot parameter sweep (K or P or other).
    Works for any exp_dir with subdirs containing config.json with opt_name.

    If use_mean_over_rounds=True, plots the mean of y_best over all rounds
    instead of the final y_best.
    """
    if results_dir is None:
        results_dir = _default_results_dir()
    exp_path = results_dir / exp_dir

    if param_name_for_print is None:
        param_name_for_print = param_key.upper()

    param_values = []
    final_y_bests = []

    if not exp_path.exists():
        print(f"Directory not found: {exp_path}")
        print(f"Run prep for {exp_dir} first.")
        return

    for subdir, val in _iter_matching_run_dirs(exp_path, regex_pattern, None):
        traces = _load_traces_or_skip(subdir, val, trace_key, param_name_for_print)
        if traces is None:
            continue

        if use_mean_over_rounds:
            y_best_curves = np.maximum.accumulate(traces, axis=1)
            y_best_per_rep = np.nanmean(y_best_curves, axis=1)
        else:
            y_best_per_rep = np.nanmax(traces, axis=1)

        param_values.append(val)
        final_y_bests.append(y_best_per_rep)
        n = len(y_best_per_rep)
        mean_y = np.nanmean(y_best_per_rep)
        sem_y = np.nanstd(y_best_per_rep) / np.sqrt(n) if n > 0 else 0
        metric_name = "mean y_best" if use_mean_over_rounds else "final y_best"
        print(f"{param_name_for_print}={val}: {n} reps, {metric_name} = {mean_y:.2f} ± {sem_y:.2f}")

    if not param_values:
        print("No valid sweep data found.")
        return

    param_values = np.array(param_values)
    sort_idx = np.argsort(param_values)
    param_values = param_values[sort_idx]
    final_y_bests = [final_y_bests[i] for i in sort_idx]

    means = np.array([np.nanmean(y) for y in final_y_bests])
    ses = np.array([np.nanstd(y) / np.sqrt(len(y)) if len(y) > 0 else 0 for y in final_y_bests])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(param_values, means, yerr=ses, fmt="o-", capsize=5, markersize=8, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    if use_mean_over_rounds:
        ax.set_ylabel(r"Mean $y_{\mathrm{best}}$ (over rounds)")
    else:
        ax.set_ylabel(r"Final $y_{\mathrm{best}}$")
    ax.set_title(title)
    ax.set_xticks(param_values)
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_curves(
    exp_dir,
    param_key="k",
    regex_pattern=r"k=(\d+)",
    xlabel="Round",
    ylabel=r"$y_{\mathrm{best}}$",
    title="TuRBO-ENN Curves",
    param_name_for_print=None,
    trace_key="rreturn",
    results_dir=None,
    env_tag=None,
):
    """
    Plot y_best vs. round curves for all parameter values on the same plot.
    Each curve shows the mean y_best at each round with error bars (SEM).
    A second panel is a bar chart of mean y_best over rounds (mean ± SEM across
    reps) for
    each series (same labels as the legend: param_name=value). If any bar mean
    is negative, all bar heights are shifted by subtracting min(bar_means) so
    the lowest bar is at zero, and a dashed horizontal line marks that baseline.
    If env_tag is set, only runs whose config.json env_tag matches are included.
    """
    if results_dir is None:
        results_dir = _default_results_dir()
    exp_path = results_dir / exp_dir

    if param_name_for_print is None:
        param_name_for_print = param_key.upper()

    if not exp_path.exists():
        print(f"Directory not found: {exp_path}")
        print(f"Run prep for {exp_dir} first.")
        return

    loaded = _collect_plot_curves_data(exp_path, regex_pattern, env_tag, trace_key, param_name_for_print)
    if loaded is None:
        print("No valid sweep data found.")
        return
    param_values, all_curves = loaded

    fig, (ax_curve, ax_bar) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    _draw_plot_curves_panels(
        ax_curve,
        ax_bar,
        param_values,
        all_curves,
        param_name_for_print=param_name_for_print,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        show_legend=True,
        show_curve_ylabel=True,
        show_bar_ylabel=True,
        show_title=True,
        panel_label=None,
        curve_ylabel_fontsize=None,
    )

    plt.show()
