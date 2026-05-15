import matplotlib

matplotlib.use("Agg")


def test_plot_compare_class_exists():
    from analysis.plot_compare import PlotCompare

    assert PlotCompare is not None


def test_plot_compare_function_exists():
    from analysis.plot_compare import plot_compare

    assert callable(plot_compare)


def test_pc_normal_exists():
    from analysis.plot_compare import pc_normal

    assert callable(pc_normal)
