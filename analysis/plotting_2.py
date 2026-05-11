"""Main plotting module for RL experiment visualization."""

from analysis import plotting_2_trace as _trace
from analysis import plotting_2_util as _util

from . import plotting_2_combined as _combined
from . import plotting_2_comparison as _comparison
from . import plotting_2_curves as _curves
from . import plotting_2_helpers as _helpers
from . import plotting_2_pareto as _pareto
from . import plotting_2_results as _results
from . import plotting_2_rl as _rl


best_so_far = _trace.best_so_far
cum_dt_prop_from_dt_prop_traces = _trace.cum_dt_prop_from_dt_prop_traces
cum_time_from_dt = _trace.cum_time_from_dt
interp_1d = _trace.interp_1d
load_cum_dt_prop = _trace.load_cum_dt_prop
load_rl_traces = _trace.load_rl_traces
mean_final_by_optimizer = _trace.mean_final_by_optimizer
mean_normalized_rank_score_by_optimizer = _trace.mean_normalized_rank_score_by_optimizer
median_final_by_optimizer = _trace.median_final_by_optimizer
print_cum_dt_prop = _trace.print_cum_dt_prop
print_dataset_summary = _trace.print_dataset_summary

consolidate_bottom_legend = _util.consolidate_bottom_legend
get_denoise_value = _util.get_denoise_value
infer_experiment_from_configs = _util.infer_experiment_from_configs
infer_params_from_configs = _util.infer_params_from_configs
noise_label = _util.noise_label
speedup_x_label = _util.speedup_x_label

plot_rl_comparison = _comparison.plot_rl_comparison
plot_rl_final_comparison = _comparison.plot_rl_final_comparison
plot_rl_combined_comparison = _combined.plot_rl_combined_comparison
plot_final_performance = _curves.plot_final_performance
plot_learning_curves = _curves.plot_learning_curves
_load_rl_with_cum_dt_prop = _helpers._load_rl_with_cum_dt_prop
_print_cum_dt_props = _helpers._print_cum_dt_props
_try_load_rl_with_cum_dt_prop = _helpers._try_load_rl_with_cum_dt_prop
compute_pareto_data = _pareto.compute_pareto_data
plot_results = _results.plot_results
plot_results_combined = _results.plot_results_combined
plot_rl_experiment = _rl.plot_rl_experiment
plot_rl_experiment_vs_time = _rl.plot_rl_experiment_vs_time
plot_rl_experiment_vs_time_auto = _rl.plot_rl_experiment_vs_time_auto


# Re-export for backward compatibility
_noise_label = noise_label
_speedup_x_label = speedup_x_label
_consolidate_bottom_legend = consolidate_bottom_legend
_get_denoise_value = get_denoise_value
_scan_experiment_configs = None  # Not re-exported
_infer_params_from_configs = infer_params_from_configs
_count_done_reps = None  # Not re-exported
_print_dataset_summary = print_dataset_summary
_mean_final_by_optimizer = mean_final_by_optimizer
_median_final_by_optimizer = median_final_by_optimizer

_normalized_ranks_0_1 = None  # Not re-exported
_mean_normalized_rank_score_by_optimizer = mean_normalized_rank_score_by_optimizer
_cum_dt_prop_from_dt_prop_traces = cum_dt_prop_from_dt_prop_traces
_print_cum_dt_prop = print_cum_dt_prop
_load_cum_dt_prop = load_cum_dt_prop
_best_so_far = best_so_far
_cum_time_from_dt = cum_time_from_dt
_interp_1d = interp_1d
