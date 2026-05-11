"""Tests for analysis/plotting_2_trace.py (part 4)."""

import numpy as np
import numpy.ma as ma


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
        xq = np.array([-1, 3])
        result = interp_1d(x, y, xq)
        np.testing.assert_array_almost_equal(result, [0.0, 20.0])
