"""Plot helpers for TuRBO-ENN (and similar) parameter sweeps under a results exp_dir."""

import matplotlib.pyplot as plt
import numpy as np

from analysis.data_sets import load_traces

from . import sweep_plots_curves as _curves
from . import sweep_plots_data as _data
from . import sweep_plots_four_impl as _four_impl
from . import sweep_plots_four_paths as _four_paths
from . import sweep_plots_four_public as _four_public
from . import sweep_plots_legends as _legends
from . import sweep_plots_metrics as _metrics
from . import sweep_plots_panels as _panels
from . import sweep_plots_style as _style

plot_curves = _curves.plot_curves
plot_param_sweep = _curves.plot_param_sweep
_collect_plot_curves_data = _data._collect_plot_curves_data
_iter_matching_run_dirs = _data._iter_matching_run_dirs
_load_traces_or_skip = _data._load_traces_or_skip
plot_curves_four_envs = _four_public.plot_curves_four_envs
plot_curves_four_sources = _four_public.plot_curves_four_sources
_plot_curves_four_sources_impl = _four_impl._plot_curves_four_sources_impl
_normalize_panel_sources = _four_paths._normalize_panel_sources
_normalize_single_panel_source = _four_paths._normalize_single_panel_source
_resolve_panel_exp_paths = _four_paths._resolve_panel_exp_paths
_render_bottom_legend_row = _legends._render_bottom_legend_row
_render_matching_column_legends = _legends._render_matching_column_legends
_render_split_column_legends = _legends._render_split_column_legends
_shared_bottom_legend_kwargs = _legends._shared_bottom_legend_kwargs
_split_bottom_legend_kwargs = _legends._split_bottom_legend_kwargs
_bar_heights_shift_if_negative = _metrics._bar_heights_shift_if_negative
_mean_ybest_mean_sem_per_curve = _metrics._mean_ybest_mean_sem_per_curve
_draw_plot_curves_panels = _panels._draw_plot_curves_panels
DEFAULT_SYNTH_10D_ENV_TAGS = _style.DEFAULT_SYNTH_10D_ENV_TAGS
FOUR_ENV_PANEL_LABELS = _style.FOUR_ENV_PANEL_LABELS
_add_panel_caption_label = _style._add_panel_caption_label
_default_results_dir = _style._default_results_dir
_panel_label_text = _style._panel_label_text
_scaled_fontsize = _style._scaled_fontsize
_style_publication_axes = _style._style_publication_axes

__all__ = (
    "DEFAULT_SYNTH_10D_ENV_TAGS",
    "FOUR_ENV_PANEL_LABELS",
    "_add_panel_caption_label",
    "_bar_heights_shift_if_negative",
    "_collect_plot_curves_data",
    "_default_results_dir",
    "_draw_plot_curves_panels",
    "_iter_matching_run_dirs",
    "_load_traces_or_skip",
    "_mean_ybest_mean_sem_per_curve",
    "_normalize_panel_sources",
    "_normalize_single_panel_source",
    "_panel_label_text",
    "_plot_curves_four_sources_impl",
    "_render_bottom_legend_row",
    "_render_matching_column_legends",
    "_render_split_column_legends",
    "_resolve_panel_exp_paths",
    "_scaled_fontsize",
    "_shared_bottom_legend_kwargs",
    "_split_bottom_legend_kwargs",
    "_style_publication_axes",
    "load_traces",
    "np",
    "plot_curves",
    "plot_curves_four_envs",
    "plot_curves_four_sources",
    "plot_param_sweep",
    "plt",
)
