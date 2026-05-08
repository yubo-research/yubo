"""2x2 curves+bars figure with a shared bottom legend row."""

from pathlib import Path

import matplotlib.pyplot as plt

from analysis.sweep_plots_data import _collect_plot_curves_data
from analysis.sweep_plots_four_paths import (
    _normalize_panel_sources,
    _resolve_panel_exp_paths,
)
from analysis.sweep_plots_legends import _render_bottom_legend_row
from analysis.sweep_plots_panels import _draw_plot_curves_panels
from analysis.sweep_plots_style import _default_results_dir, _scaled_fontsize


def _plot_curves_four_sources_impl(
    panel_sources: tuple[
        tuple[str, str | None] | tuple[str, str | None, str, str],
        ...,
    ],
    *,
    param_key: str,
    regex_pattern: str,
    xlabel: str,
    ylabel: str,
    param_name_for_print: str | None,
    trace_key: str,
    results_dir,
    panel_labels: tuple[str, ...] | None,
    save_path: str | Path | None,
):
    # Local import: IPython %aimport/autoreload runs this body in a scope where
    # module-level imports are not visible as bare names.
    from matplotlib import gridspec

    results_dir = _default_results_dir() if results_dir is None else results_dir
    param_name_for_print = param_key.upper() if param_name_for_print is None else param_name_for_print

    letters = ("a", "b", "c", "d") if panel_labels is None else panel_labels
    if len(panel_sources) != len(letters):
        raise ValueError("panel_sources and panel_labels must have the same length")

    normalized_sources = _normalize_panel_sources(
        panel_sources,
        regex_pattern=regex_pattern,
        param_name_for_print=param_name_for_print,
    )
    exp_paths = _resolve_panel_exp_paths(results_dir, normalized_sources)
    if exp_paths is None:
        return

    fig = plt.figure(figsize=(10.5, 7.7), layout="constrained")
    layout_engine = getattr(fig, "get_layout_engine", lambda: None)()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(w_pad=0.01, h_pad=0.01, wspace=0.01, hspace=0.01)
    gs = gridspec.GridSpec(
        5,
        2,
        figure=fig,
        height_ratios=[2.55, 0.68, 2.55, 0.68, 0.44],
        wspace=0.1,
        hspace=0.08,
    )

    legend_handles = None
    legend_labels = None
    column_legends: dict[int, tuple[list, list[str], str]] = {}
    for j, ((_, env_tag, panel_regex_pattern, panel_param_name), exp_path) in enumerate(zip(normalized_sources, exp_paths, strict=True)):
        row_base = (j // 2) * 2
        col = j % 2
        ax_curve = fig.add_subplot(gs[row_base, col])
        ax_bar = fig.add_subplot(gs[row_base + 1, col])
        loaded = _collect_plot_curves_data(
            exp_path,
            panel_regex_pattern,
            env_tag,
            trace_key,
            panel_param_name,
        )
        param_values, all_curves = ([], []) if loaded is None else loaded

        _draw_plot_curves_panels(
            ax_curve,
            ax_bar,
            param_values,
            all_curves,
            param_name_for_print=panel_param_name,
            xlabel=xlabel,
            ylabel=ylabel,
            title="",
            show_legend=(j == 0),
            show_curve_ylabel=(col == 0),
            show_bar_ylabel=(col == 0),
            show_title=False,
            panel_label=letters[j],
            curve_ylabel_fontsize=(_scaled_fontsize(1.02, minimum=15.0) if col == 0 else None),
        )
        if hasattr(ax_curve, "get_legend_handles_labels"):
            handles, labels = ax_curve.get_legend_handles_labels()
            if labels and col not in column_legends:
                column_legends[col] = (handles, labels, panel_param_name)
            if j == 0 and labels:
                legend_handles, legend_labels = handles, labels

    _render_bottom_legend_row(
        fig,
        gs,
        column_legends,
        legend_handles,
        legend_labels,
    )
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
