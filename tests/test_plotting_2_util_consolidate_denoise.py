"""Tests for analysis/plotting_2_util (consolidate_bottom_legend, get_denoise_value)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock


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
