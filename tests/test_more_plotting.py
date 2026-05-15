import matplotlib

matplotlib.use("Agg")


def test_plot_sorted_agg_exists():
    from analysis.plotting import plot_sorted_agg

    assert callable(plot_sorted_agg)


def test_plot_compare_problem_exists():
    from analysis.plotting import plot_compare_problem

    assert callable(plot_compare_problem)


def test_load_rl_traces_exists():
    from analysis.plotting_2 import load_rl_traces

    assert callable(load_rl_traces)


def test_plot_learning_curves_exists():
    from analysis.plotting_2 import plot_learning_curves

    assert callable(plot_learning_curves)


def test_plot_final_performance_exists():
    from analysis.plotting_2 import plot_final_performance

    assert callable(plot_final_performance)


def test_plot_rl_experiment_exists():
    from analysis.plotting_2 import plot_rl_experiment

    assert callable(plot_rl_experiment)


def test_plot_rl_experiment_vs_time_exists():
    from analysis.plotting_2 import plot_rl_experiment_vs_time

    assert callable(plot_rl_experiment_vs_time)


def test_plot_rl_experiment_vs_time_auto_exists():
    from analysis.plotting_2 import plot_rl_experiment_vs_time_auto

    assert callable(plot_rl_experiment_vs_time_auto)


def test_plot_rl_comparison_exists():
    from analysis.plotting_2 import plot_rl_comparison

    assert callable(plot_rl_comparison)


def test_plot_rl_final_comparison_exists():
    from analysis.plotting_2 import plot_rl_final_comparison

    assert callable(plot_rl_final_comparison)


def test_plot_results_exists():
    from analysis.plotting_2 import plot_results

    assert callable(plot_results)


def test_compute_pareto_data_exists():
    from analysis.plotting_2 import compute_pareto_data

    assert callable(compute_pareto_data)


def test_plot_results_grid_exists():
    from analysis.plotting_3 import plot_results_grid

    assert callable(plot_results_grid)
