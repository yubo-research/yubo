import numpy as np
import pytest

pytest.importorskip("enn")

from analysis.fitting_time import (
    benchmark_enn_incremental_add_timing,
    draw_benchmark_synthetic_xy,
    enn_incremental_checkpoint_ns,
    env_action_coords_to_surrogate_unit_x,
)
from analysis.fitting_time.fitting_time_enn_incremental import (
    ENN_INCREMENTAL_CHECKPOINT_NS,
    EnnIncrementalIndexDriver,
)
from analysis.fitting_time.fitting_time_enn_incremental_draw import (
    _train_xy_unit_cube_segment,
    draw_benchmark_test_xy_unit_cube,
)


def test_benchmark_enn_incremental_importable_from_package():
    assert callable(benchmark_enn_incremental_add_timing)


def test_enn_incremental_checkpoint_ns_matches_constant():
    assert enn_incremental_checkpoint_ns() == ENN_INCREMENTAL_CHECKPOINT_NS


def test_train_segment_matches_draw_benchmark_synthetic_xy():
    d, n, seed = 2, 30, 17
    x, y, _, _ = draw_benchmark_synthetic_xy(N=n, D=d, function_name="sphere", problem_seed=seed)
    x_ref = env_action_coords_to_surrogate_unit_x(x).detach().cpu().numpy()
    y_ref = y.detach().cpu().numpy()
    x_seg, y_seg = _train_xy_unit_cube_segment(D=d, function_name="sphere", problem_seed=seed, n_train=n, start_row=0)
    np.testing.assert_allclose(x_seg, x_ref, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(y_seg, y_ref, rtol=0.0, atol=0.0)
    x_tail, y_tail = _train_xy_unit_cube_segment(D=d, function_name="sphere", problem_seed=seed, n_train=n, start_row=10)
    np.testing.assert_allclose(x_tail, x_ref[10:], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(y_tail, y_ref[10:], rtol=0.0, atol=0.0)


def test_stream_test_draw_matches_draw_benchmark_synthetic_xy():
    d, seed = 3, 5
    _, _, x_test, y_test = draw_benchmark_synthetic_xy(N=12, D=d, function_name="sine", problem_seed=seed)
    x_ref = env_action_coords_to_surrogate_unit_x(x_test).detach().cpu().numpy()
    y_ref = y_test.detach().cpu().numpy()
    x_u, y_u = draw_benchmark_test_xy_unit_cube(D=d, function_name="sine", problem_seed=seed)
    np.testing.assert_allclose(x_u, x_ref, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(y_u, y_ref, rtol=0.0, atol=0.0)


def test_benchmark_enn_incremental_add_timing_small_checkpoints():
    result = benchmark_enn_incremental_add_timing(
        D=2,
        function_name="sphere",
        problem_seed=0,
        checkpoints=(1, 3, 10),
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    assert result.n == (1, 3, 10)
    assert len(result.add_seconds) == 3
    assert len(result.log_likelihood) == 3
    assert all(t >= 0.0 for t in result.add_seconds)
    assert all(np.isfinite(ll) for ll in result.log_likelihood)
    assert result.target == "sphere"
    assert result.d == 2
    assert result.problem_seed == 0
    assert result.index_driver is EnnIncrementalIndexDriver.FLAT


def test_benchmark_enn_incremental_hnsw_driver():
    result = benchmark_enn_incremental_add_timing(
        D=2,
        function_name="sphere",
        problem_seed=1,
        checkpoints=(1, 3),
        index_driver=EnnIncrementalIndexDriver.HNSW,
    )
    assert result.index_driver is EnnIncrementalIndexDriver.HNSW
    assert result.n == (1, 3)
