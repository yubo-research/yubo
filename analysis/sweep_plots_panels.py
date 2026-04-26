"""Curve + bar panel drawing for sweep plots."""

import numpy as np

import analysis.plotting as ap
from analysis.sweep_plots_metrics import (
    _bar_heights_shift_if_negative,
    _mean_ybest_mean_sem_per_curve,
)
from analysis.sweep_plots_style import (
    _add_panel_caption_label,
    _scaled_fontsize,
    _style_publication_axes,
)


def _show_no_data_panel(
    ax_curve,
    ax_bar,
    *,
    title: str,
    show_title: bool,
    panel_label: str | None,
) -> None:
    ax_curve.set_axis_off()
    ax_curve.text(
        0.5,
        0.5,
        "No data",
        transform=ax_curve.transAxes,
        ha="center",
        va="center",
    )
    ax_bar.set_axis_off()
    if show_title and title:
        ax_curve.set_title(title)
    if panel_label:
        _add_panel_caption_label(ax_curve, panel_label)


def _plot_curve_series(
    ax_curve,
    param_values: list,
    all_curves: list,
    *,
    param_name_for_print: str,
) -> list[str]:
    colors: list[str] = []
    for i, (val, curves) in enumerate(zip(param_values, all_curves, strict=True)):
        n_reps, n_rounds = curves.shape
        rounds = np.arange(1, n_rounds + 1)
        means = np.nanmean(curves, axis=0)
        ses = np.nanstd(curves, axis=0) / np.sqrt(n_reps)
        label = f"{param_name_for_print}={val}"
        color = ap.colors[i % len(ap.colors)]
        marker = ap.markers[i % len(ap.markers)]
        ax_curve.plot(
            rounds,
            means,
            label=label,
            linewidth=2.2,
            color=color,
            marker=marker,
            markersize=6,
            markevery=max(1, n_rounds // 10),
        )
        ax_curve.fill_between(
            rounds,
            means - ses,
            means + ses,
            alpha=0.16,
            color=color,
        )
        colors.append(color)
    return colors


def _style_curve_panel(
    ax_curve,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    show_title: bool,
    show_curve_ylabel: bool,
    panel_label: str | None,
    curve_ylabel_fontsize: float | None,
) -> None:
    ax_curve.set_xlabel(xlabel)
    if show_curve_ylabel:
        y_fs = {"fontsize": curve_ylabel_fontsize} if curve_ylabel_fontsize else {}
        ax_curve.set_ylabel(ylabel, **y_fs)
    if show_title and title:
        ax_curve.set_title(title)
    if panel_label:
        _add_panel_caption_label(ax_curve, panel_label)
    ax_curve.grid(True, alpha=0.3)
    if hasattr(ax_curve, "margins"):
        ax_curve.margins(x=0.02)
    if hasattr(ax_curve, "tick_params"):
        ax_curve.tick_params(
            axis="both",
            labelsize=_scaled_fontsize(0.84, minimum=10.0),
        )


def _draw_final_ybest_bar_panel(
    ax_bar,
    param_values: list,
    all_curves: list,
    *,
    param_name_for_print: str,
    colors: list[str] | None = None,
    show_ylabel: bool = False,
    ylabel: str = "Relative\nmean($y_{best}$)",
) -> None:
    series_labels = [f"{param_name_for_print}={val}" for val in param_values]
    bar_means, bar_sems = _mean_ybest_mean_sem_per_curve(all_curves)
    bar_heights, bar_shift = _bar_heights_shift_if_negative(bar_means)
    x_pos = np.arange(len(param_values))
    bar_colors = colors if colors is not None else ["steelblue"] * len(param_values)
    ax_bar.bar(
        x_pos,
        bar_heights,
        yerr=bar_sems,
        capsize=4,
        color=bar_colors,
        ecolor="black",
        alpha=0.85,
    )
    if bar_shift != 0:
        ax_bar.axhline(
            0.0,
            linestyle="--",
            color="gray",
            linewidth=1.0,
            alpha=0.85,
            zorder=0,
        )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(series_labels)
    if show_ylabel:
        ax_bar.set_ylabel(ylabel)
    ax_bar.grid(True, axis="y", alpha=0.3)
    if hasattr(ax_bar, "tick_params"):
        ax_bar.tick_params(
            axis="x",
            labelsize=_scaled_fontsize(0.66, minimum=8.5),
            pad=0.5,
        )
        ax_bar.tick_params(
            axis="y",
            labelsize=_scaled_fontsize(0.74, minimum=9.0),
        )


def _draw_plot_curves_panels(
    ax_curve,
    ax_bar,
    param_values: list,
    all_curves: list,
    *,
    param_name_for_print: str,
    xlabel: str,
    ylabel: str,
    title: str,
    show_legend: bool,
    show_curve_ylabel: bool,
    show_bar_ylabel: bool = False,
    bar_ylabel: str = "Relative\nmean($y_{best}$)",
    show_title: bool = True,
    panel_label: str | None = None,
    curve_ylabel_fontsize: float | None = None,
) -> None:
    _ = show_legend
    if not param_values:
        _show_no_data_panel(
            ax_curve,
            ax_bar,
            title=title,
            show_title=show_title,
            panel_label=panel_label,
        )
        return

    _style_publication_axes(ax_curve)
    _style_publication_axes(ax_bar)

    colors = _plot_curve_series(
        ax_curve,
        param_values,
        all_curves,
        param_name_for_print=param_name_for_print,
    )
    _style_curve_panel(
        ax_curve,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        show_title=show_title,
        show_curve_ylabel=show_curve_ylabel,
        panel_label=panel_label,
        curve_ylabel_fontsize=curve_ylabel_fontsize,
    )
    _draw_final_ybest_bar_panel(
        ax_bar,
        param_values,
        all_curves,
        param_name_for_print=param_name_for_print,
        colors=colors,
        show_ylabel=show_bar_ylabel,
        ylabel=bar_ylabel,
    )
