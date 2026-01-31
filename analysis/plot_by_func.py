import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.data_locator import DataLocator
from analysis.data_sets import load_multiple_traces, range_summarize
from analysis.plotting import subplots
from experiments.func_names import (
    funcs_bowl,
    funcs_multimodal,
    funcs_other,
    funcs_plate,
    funcs_ridges,
    funcs_valley,
)


def _load_traces_for_func(results_path, exp_dir, opt_names, func_name, num_dims):
    all_traces = []
    dims_to_load = [None] if num_dims is None else num_dims
    for dim in dims_to_load:
        kwargs = {
            "results_path": results_path,
            "exp_dir": exp_dir,
            "opt_names": opt_names,
            "problems": {func_name},
            "key": "return",
            "grep_for": "TRACE",
        }
        if dim is not None:
            kwargs["num_dim"] = dim
        data_locator = DataLocator(**kwargs)
        traces = load_multiple_traces(data_locator)
        if traces.size > 0:
            all_traces.append(traces)
    return all_traces


def _get_display_name(name, renames):
    return renames.get(name, name) if renames else name


def _plot_func_subplot(ax, all_traces, opt_names, renames, func_name):
    if len(all_traces) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return False
    combined_traces = all_traces[0] if len(all_traces) == 1 else np.concatenate(all_traces, axis=2)
    mu, se = range_summarize(combined_traces)
    sort_indices = np.argsort(-mu)
    sorted_mu, sorted_se = mu[sort_indices], se[sort_indices]
    sorted_opt_names = [opt_names[i] for i in sort_indices]
    x_pos = np.arange(len(opt_names))
    for i_opt, opt_name in enumerate(sorted_opt_names):
        ax.errorbar(
            x_pos[i_opt],
            sorted_mu[i_opt],
            2 * sorted_se[i_opt],
            fmt="o",
            color="black",
            capsize=4,
            capthick=1,
            label=_get_display_name(opt_name, renames),
            markersize=4,
            ecolor="black",
        )
    ax.set_title(func_name, fontsize=10)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [_get_display_name(n, renames) for n in sorted_opt_names],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.1, 1.1])
    return True


def _func_groups():
    return {
        "Multimodal": funcs_multimodal,
        "Bowl": funcs_bowl,
        "Plate": funcs_plate,
        "Valley": funcs_valley,
        "Ridges": funcs_ridges,
        "Other": funcs_other,
    }


def _flatten_func_groups(func_groups):
    all_funcs = []
    group_labels = []
    for group_name, funcs in func_groups.items():
        all_funcs.extend(funcs)
        group_labels.extend([group_name] * len(funcs))
    return all_funcs, group_labels


def _grid_layout(num_funcs, max_cols=4, max_rows=8):
    from typing import NamedTuple

    class _GridLayout(NamedTuple):
        rows: int
        cols: int
        fig_height: float

    cols = min(max_cols, num_funcs)
    rows = (num_funcs + cols - 1) // cols
    if rows > max_rows:
        rows = max_rows
        cols = min(6, (num_funcs + rows - 1) // rows)
    fig_height = min(rows * 2.5, 20)
    return _GridLayout(rows=rows, cols=cols, fig_height=fig_height)


def _hide_extra_axes(axs, num_funcs):
    if len(axs) > num_funcs:
        for ax in axs[num_funcs:]:
            ax.set_visible(False)


def _safe_plot_func(ax, results_path, exp_dir, opt_names, func_name, num_dims, renames):
    try:
        all_traces = _load_traces_for_func(results_path, exp_dir, opt_names, func_name, num_dims)
        _plot_func_subplot(ax, all_traces, opt_names, renames, func_name)
    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Error:\n{str(e)[:50]}...",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )


def _add_group_labels(fig, axs, group_labels, cols):
    current_group = None
    group_start_idx = 0
    for i_func in range(len(group_labels)):
        group_name = group_labels[i_func]
        if current_group != group_name:
            if current_group is not None and group_start_idx < i_func:
                _add_group_label(fig, axs, group_start_idx, i_func - 1, cols, current_group)
            current_group = group_name
            group_start_idx = i_func
    if current_group is not None and group_start_idx < len(group_labels):
        _add_group_label(fig, axs, group_start_idx, len(group_labels) - 1, cols, current_group)


def plot_by_func(
    results_path,
    exp_dir,
    opt_names,
    renames=None,
    save_path=None,
    figsize=12,
    num_dims=None,
):
    """
    Create plots grouped by function type, with one subplot per function.
    Each subplot shows error bars for each optimizer, averaged over all dimensions, rounds, and replications.
    """

    func_groups = _func_groups()
    all_funcs, group_labels = _flatten_func_groups(func_groups)
    num_funcs = len(all_funcs)
    rows, cols, fig_height = _grid_layout(num_funcs)
    fig, axs = subplots(rows, cols, figsize=fig_height)
    _hide_extra_axes(axs, num_funcs)

    for i_func, func_name in enumerate(all_funcs):
        _safe_plot_func(axs[i_func], results_path, exp_dir, opt_names, func_name, num_dims, renames)

    _add_group_labels(fig, axs, group_labels, cols)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.tight_layout()
    plt.show()

    if num_funcs > rows * cols:
        print(f"Note: Showing first {rows * cols} of {num_funcs} functions. Consider using plot_by_func_grouped() for better organization.")

    return fig, axs


def _add_group_label(fig, axs, start_idx, end_idx, cols, group_name):
    """Add a group label above a group of subplots."""
    if start_idx > end_idx:
        return

    # Find the top row of this group
    start_row = start_idx // cols

    # Find the first subplot in this row
    first_in_row_idx = start_row * cols
    if first_in_row_idx >= len(axs):
        return

    # Get the position of the first subplot in the row
    first_ax = axs[first_in_row_idx]
    pos = first_ax.get_position()

    # Calculate label position - center horizontally across all columns
    label_x = 0.5  # Center of the figure
    label_y = pos.y1 + 0.03  # Just above the top row

    # Add the label
    fig.text(
        label_x,
        label_y,
        group_name,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        transform=fig.transFigure,
    )


def plot_by_func_grouped(
    results_path,
    exp_dir,
    opt_names,
    renames=None,
    save_dir=None,
    num_dims=None,
    cols=4,
    figheight_per_row=2.5,
    suptitle=True,
    dpi=300,
):
    """
    Create separate plots for each function group (Multimodal, Bowl, etc.).
    Each group gets its own figure with appropriate subplot layout.
    """

    func_groups = _func_groups()

    figures = []

    for group_name, funcs in func_groups.items():
        if not funcs:
            continue

        num_funcs = len(funcs)
        cols_this = min(cols, num_funcs)
        rows = (num_funcs + cols_this - 1) // cols_this

        fig_height = rows * figheight_per_row
        fig, axs = subplots(rows, cols_this, figsize=fig_height)
        _hide_extra_axes(axs, num_funcs)

        for i_func, func_name in enumerate(funcs):
            _safe_plot_func(axs[i_func], results_path, exp_dir, opt_names, func_name, num_dims, renames)

        if suptitle:
            fig.suptitle(f"{group_name}", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            plt.tight_layout()
        plt.show()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fn = f"{group_name.lower().replace(' ', '_')}.pdf"
            fig.savefig(os.path.join(save_dir, fn), format="pdf", bbox_inches="tight", dpi=dpi)

        figures.append(fig)

    return figures


def plot_by_func_publication(results_path, exp_dir, opt_names, renames=None, save_path=None, num_dims=None):
    """
    Publication-ready version with larger fonts and cleaner styling.
    """
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
        }
    )

    fig, axs = plot_by_func(
        results_path,
        exp_dir,
        opt_names,
        renames,
        save_path,
        figsize=16,
        num_dims=num_dims,
    )

    return fig, axs
