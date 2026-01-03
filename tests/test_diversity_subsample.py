import numpy as np
import pytest

from sampling.diversity_subsample import diversity_subsample


def test_basic_functionality():
    np.random.seed(42)
    N, d, M = 100, 2, 10
    x = np.random.randn(N, d)
    indices = diversity_subsample(x, M)

    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int64
    assert len(indices) == M
    assert len(np.unique(indices)) == M
    assert indices.min() >= 0
    assert indices.max() < N


def test_deterministic_with_seed():
    N, d, M = 50, 3, 5
    x = np.random.randn(N, d)

    indices1 = diversity_subsample(x, M, seed=123)
    indices2 = diversity_subsample(x, M, seed=123)

    np.testing.assert_array_equal(indices1, indices2)


def test_spread_out_property():
    np.random.seed(42)
    N, d, M = 200, 2, 20

    x = np.zeros((N, d))
    x[: N // 2, 0] = np.linspace(0, 1, N // 2)
    x[: N // 2, 1] = 0.0
    x[N // 2 :, 0] = np.linspace(0, 1, N // 2)
    x[N // 2 :, 1] = 1.0

    indices = diversity_subsample(x, M, seed=42)
    selected_x = x[indices]

    x1_selected = selected_x[:, 1]

    assert len(np.unique(x1_selected)) >= 2


def test_edge_case_M_equals_one():
    N, d = 50, 3
    x = np.random.randn(N, d)
    indices = diversity_subsample(x, M=1, seed=42)

    assert len(indices) == 1
    assert indices[0] >= 0 and indices[0] < N


def test_edge_case_M_close_to_N():
    N, d = 50, 3
    M = N - 1
    x = np.random.randn(N, d)
    indices = diversity_subsample(x, M, seed=42)

    assert len(indices) == M
    assert len(np.unique(indices)) == M


def test_raises_on_invalid_M():
    N, d = 50, 3
    x = np.random.randn(N, d)

    with pytest.raises(AssertionError):
        diversity_subsample(x, M=N)

    with pytest.raises(AssertionError):
        diversity_subsample(x, M=0)


def test_raises_on_invalid_shape():
    x_1d = np.random.randn(50)

    with pytest.raises(AssertionError):
        diversity_subsample(x_1d, M=10)
