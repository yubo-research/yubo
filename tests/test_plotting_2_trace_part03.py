"""Tests for analysis/plotting_2_trace.py (part 3)."""

from unittest.mock import MagicMock

import numpy as np
import numpy.ma as ma


class TestMeanNormalizedRankScoreByOptimizer:
    def test_computes_scores(self):
        from analysis.plotting_2_trace import mean_normalized_rank_score_by_optimizer

        mock_locator = MagicMock()
        mock_locator.optimizers.return_value = ["opt1", "opt2"]
        traces = np.array([[[[10, 20], [15, 25]], [[5, 10], [8, 12]]]])
        means, stes = mean_normalized_rank_score_by_optimizer(mock_locator, traces)
        assert len(means) == 2
        assert len(stes) == 2
        assert means[0] > means[1]


class TestCumDtPropFromDtPropTraces:
    def test_cumsum(self):
        from analysis.plotting_2_trace import cum_dt_prop_from_dt_prop_traces

        dt_prop = np.array([[[[1, 2, 3], [4, 5, 6]]]])
        result = cum_dt_prop_from_dt_prop_traces(dt_prop)
        expected = np.array([[[[1, 3, 6], [4, 9, 15]]]])
        np.testing.assert_array_equal(result, expected)

    def test_handles_masked_array(self):
        from analysis.plotting_2_trace import cum_dt_prop_from_dt_prop_traces

        dt_prop = ma.array([[[[1, 2, 3]]]], mask=[[[[False, False, False]]]])
        result = cum_dt_prop_from_dt_prop_traces(dt_prop)
        assert ma.isMaskedArray(result)


class TestPrintCumDtProp:
    def test_prints_nothing_when_empty(self, capsys):
        from analysis.plotting_2_trace import print_cum_dt_prop

        print_cum_dt_prop(None, None, header="test")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_prints_formatted_output(self, capsys):
        from analysis.plotting_2_trace import print_cum_dt_prop

        data = {"opt1": 10.5, "opt2": 20.3}
        print_cum_dt_prop(data, ["opt1", "opt2"], header="Test header")
        captured = capsys.readouterr()
        assert "Test header" in captured.out
        assert "opt1" in captured.out
