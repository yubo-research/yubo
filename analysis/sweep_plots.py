"""Plot helpers for TuRBO-ENN (and similar) parameter sweeps under a results exp_dir."""

import json
import re
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np

from analysis.data_sets import load_traces

DEFAULT_SYNTH_10D_ENV_TAGS = (
    "f:sphere-10d",
    "f:ackley-10d",
    "f:booth-10d",
    "f:rosenbrock-10d",
)

# Order for 2x2 grid: top-left (a), top-right (b), bottom-left (c), bottom-right (d)
FOUR_ENV_PANEL_LABELS = ("a", "b", "c", "d")


def _panel_label_text(letter: str) -> str:
    return f"({letter})"


def _scaled_fontsize(scale: float, *, minimum: float = 8.0) -> float:
    base = float(plt.rcParams.get("font.size", 12.0))
    return max(minimum, base * scale)


def _style_publication_axes(ax) -> None:
    spines = getattr(ax, "spines", None)
    if spines is not None:
        for spine_name in ("top", "right"):
            spine = spines.get(spine_name)
            if spine is not None:
                spine.set_visible(False)
    if hasattr(ax, "tick_params"):
        ax.tick_params(direction="out")
    if hasattr(ax, "set_axisbelow"):
        ax.set_axisbelow(True)


def _add_panel_caption_label(
    ax,
    letter: str,
    *,
    fontsize: float | None = None,
) -> None:
    label_fontsize = fontsize if fontsize is not None else _scaled_fontsize(0.72, minimum=11.0)
    ax.text(
        0.01,
        1.01,
        _panel_label_text(letter),
        transform=ax.transAxes,
        fontsize=label_fontsize,
        fontweight="semibold",
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.2},
    )


def _default_results_dir() -> Path:
    return Path.home() / "Projects/yubo/results"


def _iter_matching_run_dirs(
    exp_path: Path,
    regex_pattern: str,
    env_tag: str | None,
) -> Iterator[tuple[Path, int]]:
    rx = re.compile(regex_pattern)
    for subdir in sorted(exp_path.iterdir()):
        if not subdir.is_dir():
            continue
        config_file = subdir / "config.json"
        if not config_file.exists():
            continue
        with open(config_file) as f:
            config = json.load(f)
        if env_tag is not None and str(config.get("env_tag", "")) != str(env_tag):
            continue
        opt_name = config.get("opt_name", "")
        match = rx.search(opt_name)
        if not match:
            continue
        yield subdir, int(match.group(1))


def _load_traces_or_skip(
    subdir: Path,
    val: int,
    trace_key: str,
    param_name_for_print: str,
):
    try:
        traces = load_traces(str(subdir), key=trace_key)
    except Exception as e:
        print(f"Skipping {param_name_for_print}={val}: {e}")
        return None
    if traces is None or len(traces) == 0:
        print(f"No traces for {param_name_for_print}={val}")
        return None
    return traces


def _collect_plot_curves_data(
    exp_path: Path,
    regex_pattern: str,
    env_tag: str | None,
    trace_key: str,
    param_name_for_print: str,
) -> tuple[list[int], list] | None:
    param_values: list[int] = []
    all_curves: list = []
    for subdir, val in _iter_matching_run_dirs(exp_path, regex_pattern, env_tag):
        traces = _load_traces_or_skip(subdir, val, trace_key, param_name_for_print)
        if traces is None:
            continue
        y_best_curves = np.maximum.accumulate(traces, axis=1)
        param_values.append(val)
        all_curves.append(y_best_curves)
    if not param_values:
        return None
    sort_idx = np.argsort(param_values)
    param_values = [param_values[i] for i in sort_idx]
    all_curves = [all_curves[i] for i in sort_idx]
    return param_values, all_curves


def _show_no_data_panel(
    ax_curve,
    ax_bar,
    *,
    title: str,
    show_title: bool,
    panel_label: str | None,
) -> None:
    import analysis.sweep_plots as _sp

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
        _sp._add_panel_caption_label(ax_curve, panel_label)


def _plot_curve_series(
    ax_curve,
    param_values: list,
    all_curves: list,
    *,
    param_name_for_print: str,
) -> None:
    for val, curves in zip(param_values, all_curves, strict=True):
        n_reps, n_rounds = curves.shape
        rounds = np.arange(1, n_rounds + 1)
        means = np.nanmean(curves, axis=0)
        ses = np.nanstd(curves, axis=0) / np.sqrt(n_reps)
        label = f"{param_name_for_print}={val}"
        ax_curve.plot(rounds, means, label=label, linewidth=2.2)
        ax_curve.fill_between(rounds, means - ses, means + ses, alpha=0.16)


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
    import analysis.sweep_plots as _sp

    ax_curve.set_xlabel(xlabel)
    if show_curve_ylabel:
        y_fs = {"fontsize": curve_ylabel_fontsize} if curve_ylabel_fontsize else {}
        ax_curve.set_ylabel(ylabel, **y_fs)
    if show_title and title:
        ax_curve.set_title(title)
    if panel_label:
        _sp._add_panel_caption_label(ax_curve, panel_label)
    ax_curve.grid(True, alpha=0.3)
    if hasattr(ax_curve, "margins"):
        ax_curve.margins(x=0.02)
    if hasattr(ax_curve, "tick_params"):
        ax_curve.tick_params(
            axis="both",
            labelsize=_sp._scaled_fontsize(0.84, minimum=10.0),
        )


def _draw_final_ybest_bar_panel(
    ax_bar,
    param_values: list,
    all_curves: list,
    *,
    param_name_for_print: str,
) -> None:
    import analysis.sweep_plots as _sp

    series_labels = [f"{param_name_for_print}={val}" for val in param_values]
    bar_means, bar_sems = _sp._final_ybest_mean_sem_per_curve(all_curves)
    bar_heights, bar_shift = _sp._bar_heights_shift_if_negative(bar_means)
    x_pos = np.arange(len(param_values))
    ax_bar.bar(
        x_pos,
        bar_heights,
        yerr=bar_sems,
        capsize=4,
        color="steelblue",
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
    ax_bar.grid(True, axis="y", alpha=0.3)
    if hasattr(ax_bar, "tick_params"):
        ax_bar.tick_params(
            axis="x",
            labelsize=_sp._scaled_fontsize(0.8, minimum=10.0),
            pad=0,
        )
        ax_bar.tick_params(
            axis="y",
            labelsize=_sp._scaled_fontsize(0.8, minimum=10.0),
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
    show_title: bool = True,
    panel_label: str | None = None,
    curve_ylabel_fontsize: float | None = None,
) -> None:
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

    _plot_curve_series(
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
    )


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


def _final_ybest_mean_sem_per_curve(
    all_curves: list,
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    sems: list[float] = []
    for curves in all_curves:
        n_reps, _ = curves.shape
        final_per_rep = curves[:, -1]
        means.append(float(np.nanmean(final_per_rep)))
        sems.append(float(np.nanstd(final_per_rep) / np.sqrt(n_reps)) if n_reps > 0 else 0.0)
    return means, sems


def _bar_heights_shift_if_negative(bar_means: list[float]) -> tuple[list[float], float]:
    """If any mean is negative, subtract min(bar_means) so the smallest bar is at zero."""
    shift = 0.0
    if bar_means:
        lo = min(bar_means)
        if lo < 0:
            shift = lo
    heights = [m - shift for m in bar_means]
    return heights, shift


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
    A second panel is a bar chart of final y_best (mean ± SEM across reps) for
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
        show_title=True,
        panel_label=None,
        curve_ylabel_fontsize=None,
    )

    plt.show()


def plot_curves_four_envs(
    exp_dir,
    param_key="k",
    regex_pattern=r"k=(\d+)",
    xlabel="Round",
    ylabel=r"$y_{\mathrm{best}}$",
    param_name_for_print=None,
    trace_key="rreturn",
    results_dir=None,
    env_tags: tuple[str, ...] | None = None,
    panel_labels: tuple[str, ...] | None = None,
):
    """
    One 2x2 figure: each cell repeats the plot_curves() layout (curves over
    rounds, then final-y_best bar chart) for a synthetic 10d benchmark —
    sphere, ackley, booth, rosenbrock by default (top-left, top-right,
    bottom-left, bottom-right).

    No subplot titles. Panel labels (a)–(d) by default for figure captions.
    Legend on the top-left panel only; y-axis label on the left column only,
    with a larger font for ``y_{\\mathrm{best}}``.
    """
    if results_dir is None:
        results_dir = _default_results_dir()

    if param_name_for_print is None:
        param_name_for_print = param_key.upper()

    tags = (
        env_tags
        if env_tags is not None
        else (
            "f:sphere-10d",
            "f:ackley-10d",
            "f:booth-10d",
            "f:rosenbrock-10d",
        )
    )
    panel_sources = tuple((exp_dir, env_tag) for env_tag in tags)
    plot_curves_four_sources(
        panel_sources=panel_sources,
        param_key=param_key,
        regex_pattern=regex_pattern,
        xlabel=xlabel,
        ylabel=ylabel,
        param_name_for_print=param_name_for_print,
        trace_key=trace_key,
        results_dir=results_dir,
        panel_labels=panel_labels,
    )


def plot_curves_four_sources(
    panel_sources: tuple[
        tuple[str, str | None] | tuple[str, str | None, str, str],
        ...,
    ],
    param_key="k",
    regex_pattern=r"k=(\d+)",
    xlabel="Round",
    ylabel=r"$y_{\mathrm{best}}$",
    param_name_for_print=None,
    trace_key="rreturn",
    results_dir=None,
    panel_labels: tuple[str, ...] | None = None,
):
    """
    Plot a 2x2 curves+bars figure using four arbitrary (exp_dir, env_tag) sources.

    This is useful when the four panels do not live under one shared results
    directory, but should still share the same publication styling.
    """
    # Local import: IPython %aimport/autoreload runs this body in a scope where
    # module-level imports are not visible as bare names.
    from matplotlib import gridspec

    if results_dir is None:
        results_dir = _default_results_dir()

    if param_name_for_print is None:
        param_name_for_print = param_key.upper()

    letters = panel_labels if panel_labels is not None else ("a", "b", "c", "d")
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
        height_ratios=[2.55, 0.68, 2.55, 0.68, 0.24],
        wspace=0.1,
        hspace=0.06,
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
        if loaded is None:
            param_values, all_curves = [], []
        else:
            param_values, all_curves = loaded

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
    plt.show()


def _normalize_single_panel_source(
    source: tuple[str, str | None] | tuple[str, str | None, str, str],
    *,
    regex_pattern: str,
    param_name_for_print: str,
) -> tuple[str, str | None, str, str]:
    if len(source) == 2:
        exp_dir, env_tag = source
        return exp_dir, env_tag, regex_pattern, param_name_for_print
    if len(source) == 4:
        exp_dir, env_tag, panel_regex_pattern, panel_param_name = source
        return exp_dir, env_tag, panel_regex_pattern, panel_param_name
    raise ValueError("Each panel source must be (exp_dir, env_tag) or (exp_dir, env_tag, regex_pattern, param_name_for_print)")


def _normalize_panel_sources(
    panel_sources: tuple[
        tuple[str, str | None] | tuple[str, str | None, str, str],
        ...,
    ],
    *,
    regex_pattern: str,
    param_name_for_print: str,
) -> list[tuple[str, str | None, str, str]]:
    return [
        _normalize_single_panel_source(
            source,
            regex_pattern=regex_pattern,
            param_name_for_print=param_name_for_print,
        )
        for source in panel_sources
    ]


def _resolve_panel_exp_paths(
    results_dir: Path,
    normalized_sources: list[tuple[str, str | None, str, str]],
) -> list[Path] | None:
    exp_paths: list[Path] = []
    for exp_dir, _, _, _ in normalized_sources:
        exp_path = results_dir / exp_dir
        if not exp_path.exists():
            print(f"Directory not found: {exp_path}")
            print(f"Run prep for {exp_dir} first.")
            return None
        exp_paths.append(exp_path)
    return exp_paths


def _shared_bottom_legend_kwargs(num_labels: int) -> dict[str, float | bool | int | str]:
    return {
        "loc": "center",
        "ncols": num_labels,
        "frameon": False,
        "fontsize": _scaled_fontsize(0.8, minimum=11.0),
        "handlelength": 1.8,
        "columnspacing": 1.2,
        "handletextpad": 0.5,
        "borderaxespad": 0.0,
    }


def _split_bottom_legend_kwargs(
    *,
    anchor_x: float,
    num_labels: int,
    title: str,
) -> dict[str, float | bool | int | str | tuple[float, float]]:
    fontsize = _scaled_fontsize(0.74, minimum=10.0)
    return {
        "loc": "center",
        "bbox_to_anchor": (anchor_x, 0.5),
        "ncols": num_labels,
        "title": title,
        "frameon": False,
        "fontsize": fontsize,
        "title_fontsize": fontsize,
        "handlelength": 1.8,
        "columnspacing": 1.0,
        "handletextpad": 0.45,
        "borderaxespad": 0.0,
    }


def _render_matching_column_legends(
    ax_legend,
    left_handles: list,
    left_labels: list[str],
) -> None:
    ax_legend.legend(
        left_handles,
        left_labels,
        **_shared_bottom_legend_kwargs(len(left_labels)),
    )


def _render_split_column_legends(
    ax_legend,
    left_handles: list,
    left_labels: list[str],
    left_param: str,
    right_handles: list,
    right_labels: list[str],
    right_param: str,
) -> None:
    left_legend = ax_legend.legend(
        left_handles,
        left_labels,
        **_split_bottom_legend_kwargs(
            anchor_x=0.25,
            num_labels=len(left_labels),
            title=left_param,
        ),
    )
    ax_legend.add_artist(left_legend)
    ax_legend.legend(
        right_handles,
        right_labels,
        **_split_bottom_legend_kwargs(
            anchor_x=0.75,
            num_labels=len(right_labels),
            title=right_param,
        ),
    )


def _render_bottom_legend_row(
    fig,
    gs,
    column_legends: dict[int, tuple[list, list[str], str]],
    legend_handles,
    legend_labels,
) -> None:
    if len(column_legends) == 2:
        ax_legend = fig.add_subplot(gs[4, :])
        ax_legend.set_axis_off()
        left_handles, left_labels, left_param = column_legends[0]
        right_handles, right_labels, right_param = column_legends[1]
        if left_labels == right_labels:
            _render_matching_column_legends(ax_legend, left_handles, left_labels)
            return
        _render_split_column_legends(
            ax_legend,
            left_handles,
            left_labels,
            left_param,
            right_handles,
            right_labels,
            right_param,
        )
        return
    if legend_handles and legend_labels:
        ax_legend = fig.add_subplot(gs[4, :])
        ax_legend.set_axis_off()
        ax_legend.legend(
            legend_handles,
            legend_labels,
            **_shared_bottom_legend_kwargs(len(legend_labels)),
        )
