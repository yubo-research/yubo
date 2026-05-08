"""Tests for analysis/plotting_2_trace.py (part 2)."""

from unittest.mock import MagicMock

import numpy as np


class TestMeanFinalByOptimizer:
    def test_computes_mean(self):
        from analysis.plotting_2_trace import mean_final_by_optimizer

        mock_locator = MagicMock()
        mock_locator.optimizers.return_value = ["opt1", "opt2"]
        traces = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
        result = mean_final_by_optimizer(mock_locator, traces)
        assert "opt1" in result
        assert "opt2" in result
        assert result["opt1"] == 4.5
        assert result["opt2"] == 10.5


class TestMedianFinalByOptimizer:
    def test_computes_median(self):
        from analysis.plotting_2_trace import median_final_by_optimizer

        mock_locator = MagicMock()
        mock_locator.optimizers.return_value = ["opt1"]
        traces = np.array([[[[1, 2, 10], [4, 5, 20], [7, 8, 30]]]])
        result = median_final_by_optimizer(mock_locator, traces)
        assert result["opt1"] == 20.0


class TestNormalizedRanks01:
    def test_single_element(self):
        from analysis.plotting_2_trace import normalized_ranks_0_1

        result = normalized_ranks_0_1(np.array([5.0]))
        assert result[0] == 1.0

    def test_ranking(self):
        from analysis.plotting_2_trace import normalized_ranks_0_1

        scores = np.array([10.0, 5.0, 15.0])
        result = normalized_ranks_0_1(scores)
        assert result[2] == 1.0
        assert result[1] == 0.0

    def test_handles_nan(self):
        from analysis.plotting_2_trace import normalized_ranks_0_1

        scores = np.array([10.0, np.nan, 5.0])
        result = normalized_ranks_0_1(scores)
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])
