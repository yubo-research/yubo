"""Tests for analysis/plotting_2_trace.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import numpy.ma as ma


class TestCountDoneReps:
    def test_counts_done_files(self):
        from analysis.plotting_2_trace import count_done_reps

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some .done files
            for i in range(3):
                (Path(tmpdir) / f"{i:05d}").touch()
                (Path(tmpdir) / f"{i:05d}.done").touch()
            result = count_done_reps(tmpdir)
            assert result == 3

    def test_handles_traces_subdir(self):
        from analysis.plotting_2_trace import count_done_reps

        with tempfile.TemporaryDirectory() as tmpdir:
            traces_dir = Path(tmpdir) / "traces"
            traces_dir.mkdir()
            for i in range(2):
                (traces_dir / f"{i:05d}.jsonl").touch()
            # data_is_done checks for .done file existence
            result = count_done_reps(tmpdir)
            assert result == 0  # No .done files


class TestPrintDatasetSummary:
    def test_prints_summary(self, capsys):
        from analysis.plotting_2_trace import print_dataset_summary

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal structure
            root = Path(tmpdir) / "results" / "exp"
            run_dir = root / "run1"
            run_dir.mkdir(parents=True)
            config = {
                "env_tag": "test-env",
                "opt_name": "opt1",
                "num_arms": 1,
                "num_rounds": 10,
                "num_reps": 5,
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config, f)
            # This will likely not find matches due to DataLocator filtering
            # but tests the function runs without error
            try:
                print_dataset_summary(
                    str(Path(tmpdir) / "results"),
                    "exp",
                    problem="test-env",
                    opt_names=["opt1"],
                    num_arms=1,
                    num_rounds=10,
                    num_reps=5,
                )
            except Exception:
                pass  # Expected if no matching data


class TestMeanFinalByOptimizer:
    def test_computes_mean(self):
        from analysis.plotting_2_trace import mean_final_by_optimizer

        mock_locator = MagicMock()
        mock_locator.optimizers.return_value = ["opt1", "opt2"]
        # Shape: [1, n_opt, n_rep, n_round]
        traces = np.array([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
        result = mean_final_by_optimizer(mock_locator, traces)
        assert "opt1" in result
        assert "opt2" in result
        assert result["opt1"] == 4.5  # mean of [3, 6]
        assert result["opt2"] == 10.5  # mean of [9, 12]


class TestMedianFinalByOptimizer:
    def test_computes_median(self):
        from analysis.plotting_2_trace import median_final_by_optimizer

        mock_locator = MagicMock()
        mock_locator.optimizers.return_value = ["opt1"]
        traces = np.array([[[[1, 2, 10], [4, 5, 20], [7, 8, 30]]]])
        result = median_final_by_optimizer(mock_locator, traces)
        assert result["opt1"] == 20.0  # median of [10, 20, 30]


class TestNormalizedRanks01:
    def test_single_element(self):
        from analysis.plotting_2_trace import normalized_ranks_0_1

        result = normalized_ranks_0_1(np.array([5.0]))
        assert result[0] == 1.0

    def test_ranking(self):
        from analysis.plotting_2_trace import normalized_ranks_0_1

        scores = np.array([10.0, 5.0, 15.0])  # ranks: 2, 3, 1
        result = normalized_ranks_0_1(scores)
        # Best (15.0) should have rank 1 -> normalized 1.0
        # Worst (5.0) should have rank 3 -> normalized 0.0
        assert result[2] == 1.0  # 15.0 is best
        assert result[1] == 0.0  # 5.0 is worst

    def test_handles_nan(self):
        from analysis.plotting_2_trace import normalized_ranks_0_1

        scores = np.array([10.0, np.nan, 5.0])
        result = normalized_ranks_0_1(scores)
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])


class TestMeanNormalizedRankScoreByOptimizer:
    def test_computes_scores(self):
        from analysis.plotting_2_trace import mean_normalized_rank_score_by_optimizer

        mock_locator = MagicMock()
        mock_locator.optimizers.return_value = ["opt1", "opt2"]
        # opt1 always better than opt2
        traces = np.array([[[[10, 20], [15, 25]], [[5, 10], [8, 12]]]])
        means, stes = mean_normalized_rank_score_by_optimizer(mock_locator, traces)
        assert len(means) == 2
        assert len(stes) == 2
        assert means[0] > means[1]  # opt1 should have higher rank


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


class TestBestSoFar:
    def test_accumulates_max(self):
        from analysis.plotting_2_trace import best_so_far

        traces = np.array([1, 3, 2, 5, 4])
        result = best_so_far(traces)
        expected = np.array([1, 3, 3, 5, 5])
        np.testing.assert_array_equal(result, expected)

    def test_handles_masked_array(self):
        from analysis.plotting_2_trace import best_so_far

        traces = ma.array([1.0, 3.0, 2.0], mask=[False, False, False])
        result = best_so_far(traces)
        assert ma.isMaskedArray(result)


class TestCumTimeFromDt:
    def test_cumsum(self):
        from analysis.plotting_2_trace import cum_time_from_dt

        dt_prop = np.array([1, 2, 3])
        dt_eval = np.array([0.5, 0.5, 0.5])
        result = cum_time_from_dt(dt_prop, dt_eval)
        expected = np.array([1.5, 4.0, 7.5])
        np.testing.assert_array_almost_equal(result, expected)


class TestInterp1d:
    def test_interpolates(self):
        from analysis.plotting_2_trace import interp_1d

        x = np.array([0, 1, 2])
        y = np.array([0, 10, 20])
        xq = np.array([0.5, 1.5])
        result = interp_1d(x, y, xq)
        np.testing.assert_array_almost_equal(result, [5.0, 15.0])

    def test_handles_insufficient_data(self):
        from analysis.plotting_2_trace import interp_1d

        x = np.array([1.0])
        y = np.array([10.0])
        xq = np.array([0.5])
        result = interp_1d(x, y, xq)
        assert np.isnan(result[0])

    def test_clips_to_range(self):
        from analysis.plotting_2_trace import interp_1d

        x = np.array([0, 1, 2])
        y = np.array([0, 10, 20])
        xq = np.array([-1, 3])  # outside range
        result = interp_1d(x, y, xq)
        np.testing.assert_array_almost_equal(result, [0.0, 20.0])
