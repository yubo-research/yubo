import matplotlib.pyplot as plt

from analysis.plot_by_func_core import (
    add_group_labels,
    flatten_func_groups,
    func_groups,
    grid_layout,
    hide_extra_axes,
    plot_by_func_grouped,
    safe_plot_func,
)
from analysis.plotting import subplots


__all__ = ["plot_by_func", "plot_by_func_grouped", "plot_by_func_publication"]


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

    func_groups_map = func_groups()
    all_funcs, group_labels = flatten_func_groups(func_groups_map)
    num_funcs = len(all_funcs)
    rows, cols, fig_height = grid_layout(num_funcs)
    fig, axs = subplots(rows, cols, figsize=fig_height)
    hide_extra_axes(axs, num_funcs)

    for i_func, func_name in enumerate(all_funcs):
        safe_plot_func(axs[i_func], results_path, exp_dir, opt_names, func_name, num_dims, renames)

    add_group_labels(fig, axs, group_labels, cols)

    if save_path:
        fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")

    plt.tight_layout()
    plt.show()

    if num_funcs > rows * cols:
        print(f"Note: Showing first {rows * cols} of {num_funcs} functions. Consider using plot_by_func_grouped() for better organization.")

    return fig, axs


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
