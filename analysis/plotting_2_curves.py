"""Learning-curve and final-performance plots for RL optimizers."""

import numpy as np

import analysis.plotting as ap
from analysis.data_locator import DataLocator
from analysis.plotting_2_trace import mean_normalized_rank_score_by_optimizer


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
            linewidth = 3.0
            markersize_use = max(markersize + 2, 7)
        else:
            color = ap.colors[style_idx]
            marker = ap.markers[style_idx]
            linewidth = 2.0
            markersize_use = markersize
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
            markersize=markersize_use,
        )
        if ax.lines:
            ax.lines[-1].set_linewidth(linewidth)
            ax.lines[-1].set_markeredgewidth(max(1.5, linewidth - 1.0))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, fontsize=ax.yaxis.label.get_size() * 1.15)
    if show_title and title:
        ax.set_title(title)
    ax.tick_params(axis="both")
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
    ax.set_ylabel(ylabel)
    if show_title and title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(-0.05, 1.05)
