def test_farthest_neighbor_import():
    """Test that farthest_neighbor can be imported and is callable."""
    from sampling.knn_tools import farthest_neighbor

    assert callable(farthest_neighbor)
