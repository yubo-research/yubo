def test_load_traces_import():
    """Test that load_traces can be imported and is callable."""
    from analysis.data_sets import load_traces

    assert callable(load_traces)


def test_load_traces_nonexistent_dir():
    """Test that load_traces handles nonexistent directory."""
    from analysis.data_sets import load_traces

    # Use a path that definitely doesn't exist
    nonexistent_path = "/tmp/nonexistent_trace_dir_12345"
    # The function may raise FileNotFoundError for nonexistent paths
    try:
        result = load_traces(nonexistent_path, key="return")
        # If it doesn't raise, result should be None or empty
        assert result is None or len(result) == 0
    except FileNotFoundError:
        # Expected behavior for nonexistent directory
        pass


def test_load_multiple_traces_import():
    """Test that load_multiple_traces can be imported and is callable."""
    from analysis.data_sets import load_multiple_traces

    assert callable(load_multiple_traces)


def test_load_cum_dt_prop_import():
    """Test that load_cum_dt_prop can be imported and is callable."""
    from analysis.plotting_2_trace import load_cum_dt_prop

    assert callable(load_cum_dt_prop)
