import torch

from model.enn_likelihood_t import subsample_indices_with_extremes


def test_subsample_indices_with_extremes_basic():
    """Test that y_min and y_max indices are always included."""
    torch.manual_seed(42)
    n = 20
    P = 5
    y = torch.randn(n, dtype=torch.float64)
    device = torch.device("cpu")

    y_min_idx = y.argmin().item()
    y_max_idx = y.argmax().item()

    # Run multiple times to verify consistency
    for _ in range(10):
        indices = subsample_indices_with_extremes(n, P, y, device)
        assert len(indices) == P
        assert y_min_idx in indices.tolist()
        assert y_max_idx in indices.tolist()
        assert indices.min() >= 0
        assert indices.max() < n


def test_subsample_indices_with_extremes_all_points():
    """Test when P equals n (all points selected)."""
    n = 10
    P = 10
    y = torch.randn(n, dtype=torch.float64)
    device = torch.device("cpu")

    indices = subsample_indices_with_extremes(n, P, y, device)
    assert len(indices) == n
    assert torch.equal(indices, torch.arange(n))


def test_subsample_indices_with_extremes_small_P():
    """Test when P is very small (just min and max)."""
    n = 20
    P = 2
    y = torch.randn(n, dtype=torch.float64)
    device = torch.device("cpu")

    y_min_idx = y.argmin().item()
    y_max_idx = y.argmax().item()

    indices = subsample_indices_with_extremes(n, P, y, device)
    assert len(indices) == P
    assert y_min_idx in indices.tolist()
    assert y_max_idx in indices.tolist()


def test_subsample_indices_with_extremes_min_equals_max():
    """Test when y_min and y_max are the same point."""
    n = 10
    P = 5
    y = torch.ones(n, dtype=torch.float64)  # All values are the same
    device = torch.device("cpu")

    indices = subsample_indices_with_extremes(n, P, y, device)
    assert len(indices) == P
    # Should still work even though min == max
    assert len(torch.unique(indices)) == P  # All indices should be unique


def test_subsample_indices_with_extremes_P_greater_than_n():
    """Test when P > n (should clamp to n)."""
    n = 10
    P = 20
    y = torch.randn(n, dtype=torch.float64)
    device = torch.device("cpu")

    indices = subsample_indices_with_extremes(n, P, y, device)
    assert len(indices) == n
    assert torch.equal(indices, torch.arange(n))
