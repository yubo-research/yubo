"""Public API for 2x2 multi-environment / multi-source sweep figures."""

from pathlib import Path

from analysis.sweep_plots_four_impl import _plot_curves_four_sources_impl
from analysis.sweep_plots_style import _default_results_dir


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
    save_path: str | Path | None = None,
):
    """
    One 2x2 figure: each cell repeats the plot_curves() layout (curves over
    rounds, then mean-y_best-over-rounds bar chart) for a synthetic 10d benchmark —
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
        save_path=save_path,
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
    save_path: str | Path | None = None,
):
    """
    Plot a 2x2 curves+bars figure using four arbitrary (exp_dir, env_tag) sources.

    This is useful when the four panels do not live under one shared results
    directory, but should still share the same publication styling.
    """
    _plot_curves_four_sources_impl(
        panel_sources,
        param_key=param_key,
        regex_pattern=regex_pattern,
        xlabel=xlabel,
        ylabel=ylabel,
        param_name_for_print=param_name_for_print,
        trace_key=trace_key,
        results_dir=results_dir,
        panel_labels=panel_labels,
        save_path=save_path,
    )
