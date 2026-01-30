"""Tests for analysis/plotting_2_util.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


class TestNoiseLabel:
    def test_frozen_noise(self):
        from analysis.plotting_2_util import noise_label

        assert noise_label("problem:fn") == "Frozen noise"

    def test_natural_noise(self):
        from analysis.plotting_2_util import noise_label

        assert noise_label("problem") == "Natural noise"


class TestSpeedupXLabel:
    def test_returns_none_when_no_data(self):
        from analysis.plotting_2_util import speedup_x_label

        assert speedup_x_label(None, "problem") is None
        assert speedup_x_label({}, "problem") is None

    def test_returns_none_when_no_baseline(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"other-opt": 10.0}, "problem")
        assert result is None

    def test_returns_none_when_no_compare_opt(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"turbo-one": 10.0}, "problem")
        assert result is None

    def test_computes_speedup_for_frozen_noise(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"turbo-one": 100.0, "turbo-enn-p": 10.0}, "problem:fn")
        assert result == "10x speedup"

    def test_computes_speedup_for_natural_noise(self):
        from analysis.plotting_2_util import speedup_x_label

        result = speedup_x_label({"turbo-one": 50.0, "turbo-enn-fit-ucb": 10.0}, "problem")
        assert result == "5x speedup"


class TestConsolidateBottomLegend:
    def test_handles_empty_axes(self):
        import matplotlib.pyplot as plt

        from analysis.plotting_2_util import consolidate_bottom_legend

        fig, axs = plt.subplots(1, 2)
        consolidate_bottom_legend(fig, axs)
        plt.close(fig)

    def test_consolidates_legends(self):
        import matplotlib.pyplot as plt

        from analysis.plotting_2_util import consolidate_bottom_legend

        fig, axs = plt.subplots(1, 2)
        axs[0].plot([1, 2], [1, 2], label="opt1")
        axs[1].plot([1, 2], [2, 3], label="opt2")
        consolidate_bottom_legend(fig, axs)
        plt.close(fig)


class TestGetDenoiseValue:
    def test_returns_none_when_no_data(self):
        from analysis.plotting_2_util import get_denoise_value

        mock_locator = MagicMock()
        mock_locator._load.return_value = []
        result = get_denoise_value(mock_locator, "problem")
        assert result is None

    def test_returns_num_denoise_for_frozen_noise(self):
        from analysis.plotting_2_util import get_denoise_value

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"num_denoise": 5}
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
            mock_locator = MagicMock()
            mock_locator._load.return_value = [("problem", tmpdir)]
            result = get_denoise_value(mock_locator, "problem:fn")
            assert result == 5

    def test_returns_num_denoise_passive_for_natural_noise(self):
        from analysis.plotting_2_util import get_denoise_value

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"num_denoise_passive": 10}
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config, f)
            mock_locator = MagicMock()
            mock_locator._load.return_value = [("problem", tmpdir)]
            result = get_denoise_value(mock_locator, "problem")
            assert result == 10


class TestScanExperimentConfigs:
    def test_returns_empty_for_nonexistent_dir(self):
        from analysis.plotting_2_util import scan_experiment_configs

        env_tags, opt_names = scan_experiment_configs(Path("/nonexistent/path"))
        assert env_tags == set()
        assert opt_names == set()

    def test_scans_configs(self):
        from analysis.plotting_2_util import scan_experiment_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            subdir = root / "exp1"
            subdir.mkdir()
            config = {"env_tag": "test-env", "opt_name": "test-opt"}
            with open(subdir / "config.json", "w") as f:
                json.dump(config, f)
            env_tags, opt_names = scan_experiment_configs(root)
            assert "test-env" in env_tags
            assert "test-opt" in opt_names


class TestInferParamsFromConfigs:
    def test_infers_params(self):
        from analysis.plotting_2_util import infer_params_from_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "results" / "exp"
            root.mkdir(parents=True)
            for i, (env, arms, rounds, reps) in enumerate(
                [
                    ("prob_seq", 1, 100, 10),
                    ("prob_batch", 50, 30, 10),
                ]
            ):
                subdir = root / f"run{i}"
                subdir.mkdir()
                config = {
                    "env_tag": env,
                    "opt_name": "opt1",
                    "num_arms": arms,
                    "num_rounds": rounds,
                    "num_reps": reps,
                }
                with open(subdir / "config.json", "w") as f:
                    json.dump(config, f)
            result = infer_params_from_configs(
                str(Path(tmpdir) / "results"),
                "exp",
                problem_seq="prob_seq",
                problem_batch="prob_batch",
                opt_names=["opt1"],
            )
            assert result.get("num_arms_seq") == 1
            assert result.get("num_arms_batch") == 50
