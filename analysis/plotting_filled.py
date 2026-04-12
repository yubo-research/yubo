import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


def filled_err(
    ys,
    x=None,
    color="#AAAAAA",
    alpha=0.5,
    marker=None,
    linestyle="--",
    color_line="#AAAAAA",
    se=False,
    two=False,
    ax=None,
    label=None,
    alpha_top=1,
    markersize=10,
    fillstyle="none",
    max_markers=None,
    use_median=False,
):
    if ax is None:
        ax = plt
    if use_median:
        mu = np.median(ys, axis=0)
        sg = ss.median_abs_deviation(ys, axis=0)
    else:
        mu = ys.mean(axis=0)
        sg = ys.std(axis=0)
    if x is None:
        x = np.arange(len(mu))
    if se:
        sg = sg / np.sqrt(ys.shape[0])
    if two:
        sg *= 2
    x_full = np.asarray(x)
    mu_full = np.asarray(mu)
    sg_full = np.asarray(sg)
    if x_full.size == 0 or mu_full.size == 0:
        return
    ax.fill_between(
        x_full,
        mu_full - sg_full,
        mu_full + sg_full,
        color=color,
        alpha=alpha,
        linewidth=1,
        label="_",
    )

    if max_markers is not None:
        n_skip = max(1, len(x_full) // max_markers)
        idx = np.arange(0, len(x_full), n_skip)
        if idx.size == 0:
            return
        if idx[-1] != len(x_full) - 1:
            idx = np.append(idx, len(x_full) - 1)
        x_plot = np.asarray(x_full)[idx]
        mu_plot = np.asarray(mu_full)[idx]
    else:
        x_plot = x_full
        mu_plot = mu_full
    ax.plot(
        x_plot,
        mu_plot,
        color=color_line,
        marker=marker,
        linestyle=linestyle,
        label=label,
        alpha=alpha_top,
        markersize=markersize,
        fillstyle=fillstyle,
    )
