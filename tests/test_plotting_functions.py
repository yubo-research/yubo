import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def test_mk_trans():
    from analysis.plotting import mk_trans

    fig, ax = plt.subplots()
    trans = mk_trans(fig)
    assert trans is not None
    plt.close(fig)


def test_hash_marks():
    from analysis.plotting import hash_marks

    fig, ax = plt.subplots()
    x = np.array([0.0, 0.0])
    y = np.array([1.0, 1.0])
    hash_marks(ax, x, y, k=0.5, color="blue")
    plt.close(fig)


def test_label_subplots():
    from analysis.plotting import label_subplots

    fig, axs = plt.subplots(1, 2)
    label_subplots(axs)
    plt.close(fig)


def test_slabel2():
    from analysis.plotting import slabel2

    fig, ax = plt.subplots()
    slabel2(ax, 0.5, 0.5, "a")
    plt.close(fig)


def test_slabel():
    from analysis.plotting import mk_trans, slabel

    fig, ax = plt.subplots()
    trans = mk_trans(fig)
    slabel(trans, ax, "a")
    plt.close(fig)


def test_subplots():
    from analysis.plotting import subplots

    fig, axs = subplots(2, 2, figsize=4)
    assert fig is not None
    plt.close(fig)


def test_tight():
    from analysis.plotting import tight

    fig, axs = plt.subplots(1, 2)
    tight(axs, sub_aspect=1)
    plt.close(fig)


def test_tight_landscape():
    from analysis.plotting import tight_landscape

    fig, axs = plt.subplots(1, 2)
    tight_landscape(axs)
    plt.close(fig)


def test_filled_err():
    from analysis.plotting import filled_err

    fig, ax = plt.subplots()
    # filled_err expects ys = array of multiple observations (rows are samples)
    ys = np.random.rand(10, 5)  # 10 samples, 5 time steps
    filled_err(ys, ax=ax)
    plt.close(fig)


def test_error_area():
    from analysis.plotting import error_area

    fig, ax = plt.subplots()
    x = np.array([1, 2, 3])
    y = np.array([1.0, 2.0, 1.5])
    err = np.array([0.1, 0.1, 0.1])
    error_area(x, y, err, ax=ax)
    plt.close(fig)


def test_hline():
    from analysis.plotting import hline

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # Need something plotted first for axis()
    hline(0.5, ax=ax)
    plt.close(fig)


def test_vline():
    from analysis.plotting import vline

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # Need something plotted first for axis()
    vline(0.5, ax=ax)
    plt.close(fig)


def test_zc():
    from analysis.plotting import zc

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = zc(x)
    assert result.shape == x.shape
    assert np.isclose(result.mean(), 0.0, atol=1e-10)
    assert np.isclose(result.std(), 1.0, atol=1e-10)


def test_plot_sorted():
    from analysis.plotting import plot_sorted

    fig, ax = plt.subplots()
    optimizers = ["opt1", "opt2", "opt3"]
    mu = np.array([1.0, 2.0, 1.5])
    se = np.array([0.1, 0.1, 0.1])
    plot_sorted(ax, optimizers, mu, se)
    plt.close(fig)
