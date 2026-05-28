from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("enn")

from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from analysis.fitting_time.fitting_time_enn_query import (
    ENN_QUERY_NUM_POINTS,
    benchmark_enn_query_timing,
)


def test_benchmark_enn_query_timing_importable_from_package():
    from analysis.fitting_time import benchmark_enn_query_timing as exported

    assert exported is benchmark_enn_query_timing


def test_benchmark_enn_query_timing_small_checkpoints():
    result = benchmark_enn_query_timing(
        D=2,
        function_name="sphere",
        problem_seed=0,
        checkpoints=(1, 3),
        index_driver=EnnIncrementalIndexDriver.FLAT,
        num_query_points=5,
    )

    assert result.n == (1, 3)
    assert result.num_query_points == 5
    assert len(result.query_seconds) == 2
    assert len(result.query_seconds_per_point) == 2
    assert all(t >= 0.0 for t in result.query_seconds)
    np.testing.assert_allclose(
        result.query_seconds_per_point,
        np.asarray(result.query_seconds) / 5.0,
        rtol=1e-12,
        atol=0.0,
    )
    assert result.target == "sphere"
    assert result.d == 2
    assert result.problem_seed == 0
    assert result.index_driver is EnnIncrementalIndexDriver.FLAT


def test_query_default_num_points_is_100():
    assert ENN_QUERY_NUM_POINTS == 100
