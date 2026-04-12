import json
import os
import tempfile
from unittest.mock import MagicMock

import matplotlib
import numpy as np

matplotlib.use("Agg")


def test_plot_results_allows_distinct_seq_and_batch_reps(monkeypatch):
    from analysis import plotting_2 as ap2

    summary_calls = []
    comparison_calls = []
    final_calls = []

    monkeypatch.setattr(
        "analysis.plotting_2_util.infer_params_from_configs",
        lambda *args, **kwargs: {
            "num_reps": 30,
            "num_rounds_seq": 100,
            "num_rounds_batch": 30,
            "num_arms_seq": 1,
            "num_arms_batch": 50,
        },
    )
    monkeypatch.setattr(
        "analysis.plotting_2_trace.print_dataset_summary",
        lambda *args, **kwargs: summary_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "analysis.plotting_2_comparison.plot_rl_comparison",
        lambda *args, **kwargs: comparison_calls.append(kwargs) or ("fig_c", "axs_c", "seq_data", "batch_data"),
    )
    monkeypatch.setattr(
        "analysis.plotting_2_comparison.plot_rl_final_comparison",
        lambda *args, **kwargs: final_calls.append(kwargs) or ("fig_f", "axs_f", "seq_data_f", "batch_data_f"),
    )

    result = ap2.plot_results(
        "results",
        "exp_dir",
        ["opt1"],
        problem="bw-heur",
        num_reps_seq=30,
        num_reps_batch=10,
    )

    assert summary_calls[0]["num_reps"] == 30
    assert summary_calls[1]["num_reps"] == 10
    assert comparison_calls[0]["num_reps_seq"] == 30
    assert comparison_calls[0]["num_reps_batch"] == 10
    assert final_calls[0]["num_reps_seq"] == 30
    assert final_calls[0]["num_reps_batch"] == 10
    assert result.seq_data == "seq_data"
    assert result.batch_data == "batch_data"


def test_plot_learning_curves_bolds_turbo_enn(monkeypatch):
    import matplotlib.pyplot as plt

    from analysis import plotting_2 as ap2

    locator = MagicMock()
    locator.optimizers.return_value = ["turbo-enn-fit-ucb", "random"]
    traces = np.ones((1, 2, 3, 4))

    fig, ax = plt.subplots()
    ap2.plot_learning_curves(
        ax,
        locator,
        traces,
        opt_names_all=["turbo-enn-fit-ucb", "random"],
    )
    turbo_line, random_line = ax.lines[-2:]

    assert turbo_line.get_label() == "turbo-enn-fit-ucb"
    assert turbo_line.get_linewidth() > random_line.get_linewidth()
    assert turbo_line.get_markersize() > random_line.get_markersize()
    assert turbo_line.get_marker() == "s"
    plt.close(fig)


def test_infer_experiment_from_configs():
    from analysis.plotting_2 import infer_experiment_from_configs

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create exp_dir structure
        exp_dir = "test_exp"
        os.makedirs(os.path.join(tmpdir, exp_dir, "run1"))

        # Create a config.json file
        config = {
            "env_tag": "ackley",
            "opt_name": "random",
            "num_arms": 5,
            "num_rounds": 10,
            "num_reps": 3,
        }
        with open(os.path.join(tmpdir, exp_dir, "run1", "config.json"), "w") as f:
            json.dump(config, f)

        result = infer_experiment_from_configs(tmpdir, exp_dir)
        assert "env_tags" in result
        assert "opt_names" in result
        assert "ackley" in result["env_tags"]
        assert "random" in result["opt_names"]


def test_infer_experiment_from_configs_no_configs():
    import pytest

    from analysis.plotting_2 import infer_experiment_from_configs

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = "empty_exp"
        os.makedirs(os.path.join(tmpdir, exp_dir))

        with pytest.raises(ValueError):
            infer_experiment_from_configs(tmpdir, exp_dir)


def test_infer_experiment_from_configs_not_found():
    import pytest

    from analysis.plotting_2 import infer_experiment_from_configs

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            infer_experiment_from_configs(tmpdir, "nonexistent_exp")
