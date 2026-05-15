import matplotlib

matplotlib.use("Agg")


def test_plot_by_func_exists():
    from analysis.plot_by_func import plot_by_func

    # Just verify the function exists and is callable
    assert callable(plot_by_func)


def test_plot_by_func_grouped_exists():
    from analysis.plot_by_func import plot_by_func_grouped

    assert callable(plot_by_func_grouped)


def test_plot_by_func_publication_exists():
    from analysis.plot_by_func import plot_by_func_publication

    assert callable(plot_by_func_publication)
