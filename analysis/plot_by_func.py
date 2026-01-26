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

    func_groups = {
        "Multimodal": funcs_multimodal,
        "Bowl": funcs_bowl,
        "Plate": funcs_plate,
        "Valley": funcs_valley,
        "Ridges": funcs_ridges,
        "Other": funcs_other,
    }

    all_funcs = []
    group_labels = []
    for group_name, funcs in func_groups.items():
        all_funcs.extend(funcs)
        group_labels.extend([group_name] * len(funcs))

    num_funcs = len(all_funcs)
    cols = min(4, num_funcs)
    rows = (num_funcs + cols - 1) // cols

    # Limit figure size for readability
    max_rows = 8
    if rows > max_rows:
        rows = max_rows
        cols = min(6, (num_funcs + rows - 1) // rows)

    fig_height = min(rows * 2.5, 20)
    fig, axs = subplots(rows, cols, figsize=fig_height)

    if len(axs) > num_funcs:
        for ax in axs[num_funcs:]:
            ax.set_visible(False)

    # Track current group for drawing boxes
    current_group = None
    group_start_idx = 0

    for i_func, func_name in enumerate(all_funcs):
        ax = axs[i_func]

        try:
            all_traces = []

            if num_dims is None:
                data_locator = DataLocator(
                    results_path=results_path,
                    exp_dir=exp_dir,
                    opt_names=opt_names,
                    problems={func_name},
                    key="return",
                    grep_for="TRACE",
                )
                traces = load_multiple_traces(data_locator)
                if traces.size > 0:
                    all_traces.append(traces)
            else:
                for dim in num_dims:
                    data_locator = DataLocator(
                        results_path=results_path,
                        exp_dir=exp_dir,
                        opt_names=opt_names,
                        problems={func_name},
                        key="return",
                        grep_for="TRACE",
                        num_dim=dim,
                    )
                    traces = load_multiple_traces(data_locator)
                    if traces.size > 0:
                        all_traces.append(traces)

            if len(all_traces) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Concatenate traces from all dimensions
            if len(all_traces) == 1:
                combined_traces = all_traces[0]
            else:
                # Stack along the replication dimension
                combined_traces = np.concatenate(all_traces, axis=2)

            mu, se = range_summarize(combined_traces)

            # Sort optimizers by performance (highest to lowest)
            sort_indices = np.argsort(-mu)
            sorted_mu = mu[sort_indices]
            sorted_se = se[sort_indices]
            sorted_opt_names = [opt_names[i] for i in sort_indices]

            x_pos = np.arange(len(opt_names))

            for i_opt, opt_name in enumerate(sorted_opt_names):
                display_name = renames.get(opt_name, opt_name) if renames else opt_name

                ax.errorbar(
                    x_pos[i_opt],
                    sorted_mu[i_opt],
                    2 * sorted_se[i_opt],
                    fmt="o",
                    color="black",
                    capsize=4,
                    capthick=1,
                    label=display_name,
                    markersize=4,
                    ecolor="black",
                )

            ax.set_title(func_name, fontsize=10)
            ax.set_ylabel("Score", fontsize=9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [
                    renames.get(name, name) if renames else name
                    for name in sorted_opt_names
                ],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            ax.grid(True, alpha=0.3)
            ax.set_ylim([-0.1, 1.1])

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

    # Add group labels
    current_group = None
    group_start_idx = 0

    for i_func, func_name in enumerate(all_funcs):
        group_name = group_labels[i_func]

        if current_group != group_name:
            # Add label for previous group if it exists
            if current_group is not None and group_start_idx < i_func:
                _add_group_label(
                    fig, axs, group_start_idx, i_func - 1, cols, current_group
                )

            # Start new group
            current_group = group_name
            group_start_idx = i_func

    # Add final group label
    if current_group is not None and group_start_idx < num_funcs:
        _add_group_label(fig, axs, group_start_idx, num_funcs - 1, cols, current_group)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.tight_layout()
    plt.show()

    if num_funcs > rows * cols:
        print(
            f"Note: Showing first {rows * cols} of {num_funcs} functions. Consider using plot_by_func_grouped() for better organization."
        )

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

    func_groups = {
        "Multimodal": funcs_multimodal,
        "Bowl": funcs_bowl,
        "Plate": funcs_plate,
        "Valley": funcs_valley,
        "Ridges": funcs_ridges,
        "Other": funcs_other,
    }

    figures = []

    for group_name, funcs in func_groups.items():
        if not funcs:
            continue

        num_funcs = len(funcs)
        cols = min(cols, num_funcs)
        rows = (num_funcs + cols - 1) // cols

        fig_height = rows * figheight_per_row
        fig, axs = subplots(rows, cols, figsize=fig_height)

        if len(axs) > num_funcs:
            for ax in axs[num_funcs:]:
                ax.set_visible(False)

        for i_func, func_name in enumerate(funcs):
            ax = axs[i_func]

            try:
                all_traces = []

                if num_dims is None:
                    data_locator = DataLocator(
                        results_path=results_path,
                        exp_dir=exp_dir,
                        opt_names=opt_names,
                        problems={func_name},
                        key="return",
                        grep_for="TRACE",
                    )
                    traces = load_multiple_traces(data_locator)
                    if traces.size > 0:
                        all_traces.append(traces)
                else:
                    for dim in num_dims:
                        data_locator = DataLocator(
                            results_path=results_path,
                            exp_dir=exp_dir,
                            opt_names=opt_names,
                            problems={func_name},
                            key="return",
                            grep_for="TRACE",
                            num_dim=dim,
                        )
                        traces = load_multiple_traces(data_locator)
                        if traces.size > 0:
                            all_traces.append(traces)

                if len(all_traces) == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                if len(all_traces) == 1:
                    combined_traces = all_traces[0]
                else:
                    combined_traces = np.concatenate(all_traces, axis=2)

                mu, se = range_summarize(combined_traces)

                # Sort optimizers by performance (highest to lowest)
                sort_indices = np.argsort(-mu)
                sorted_mu = mu[sort_indices]
                sorted_se = se[sort_indices]
                sorted_opt_names = [opt_names[i] for i in sort_indices]

                x_pos = np.arange(len(opt_names))

                for i_opt, opt_name in enumerate(sorted_opt_names):
                    display_name = (
                        renames.get(opt_name, opt_name) if renames else opt_name
                    )

                    ax.errorbar(
                        x_pos[i_opt],
                        sorted_mu[i_opt],
                        2 * sorted_se[i_opt],
                        fmt="o",
                        color="black",
                        capsize=4,
                        capthick=1,
                        label=display_name,
                        markersize=4,
                        ecolor="black",
                    )

                ax.set_title(func_name, fontsize=10)
                ax.set_ylabel("Score", fontsize=9)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    [
                        renames.get(name, name) if renames else name
                        for name in sorted_opt_names
                    ],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.1, 1.1])

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

        if suptitle:
            fig.suptitle(f"{group_name}", fontsize=14, fontweight="bold")
            plt.tight_layout(rect=(0, 0, 1, 0.95))
        else:
            plt.tight_layout()
        plt.show()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fn = f"{group_name.lower().replace(' ', '_')}.pdf"
            fig.savefig(
                os.path.join(save_dir, fn), format="pdf", bbox_inches="tight", dpi=dpi
            )

        figures.append(fig)

    return figures


def plot_by_func_publication(
    results_path, exp_dir, opt_names, renames=None, save_path=None, num_dims=None
):
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
