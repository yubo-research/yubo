import matplotlib.pyplot as plt
import numpy as np

import analysis.data_sets as ads


def subplots(n, m, figsize):
    fig, axs = plt.subplots(n, m, figsize=(figsize, figsize))
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]
    return fig, axs


def tight(axs):
    for a in axs:
        a.set_box_aspect(1)
    plt.tight_layout()
    # plt.show()


def filled_err(ys, x=None, color="#AAAAAA", alpha=0.5, fmt="--", se=False, ax=None):
    if ax is None:
        ax = plt
    mu = ys.mean(axis=0)
    sg = ys.std(axis=0)
    if x is None:
        x = np.arange(len(mu))
    if se:
        sg = sg / np.sqrt(ys.shape[0])
    ax.fill_between(x, mu - sg, mu + sg, color=color, alpha=alpha, linewidth=1)
    ax.plot(x, mu, fmt, color=color)


def error_area(x, y, yerr, color="#AAAAAA", alpha=0.5, fmt="--", marker=",", ax=plt):
    mu = y
    sg = yerr
    ax.fill_between(x, mu - sg, mu + sg, color=color, alpha=alpha, linewidth=1)
    ax.plot(x, mu, fmt, color=color)


def hline(y0, color="black", ax=None):
    if ax is None:
        ax = plt
    c = ax.axis()
    ax.autoscale(False)
    ax.plot([c[0], c[1]], [y0, y0], "--", linewidth=1, color=color)


def vline(x0, color="black", ax=None):
    if ax is None:
        ax = plt
    c = plt.axis()
    ax.autoscale(False)
    ax.plot([x0, x0], [c[2], c[3]], "--", linewidth=1, color=color)


def zc(x):
    return (x - x.mean()) / x.std()


def plot_sorted(ax, optimizers, mu, se, renames=None):
    i_sort = np.argsort(-mu)
    n = np.arange(len(mu))
    ax.errorbar(n, mu[i_sort], se[i_sort], fmt="ko", capsize=10)
    if renames is None:
        renames = optimizers
    ax.set_xticks(n, [optimizers[i] for i in i_sort], rotation=90)
    ax.set_ylim([0, 1])


def plot_sorted_agg(ax, data_locator, exp_tag, optimizers=None, renames=None):
    problems = sorted(data_locator.problems_in(exp_tag))
    if optimizers is None:
        optimizers = set()
        for problem in problems:
            optimizers.update(data_locator.optimizers_in(exp_tag, problem))
    # optimizers = sorted(optimizers)

    traces = ads.load_multiple_traces(data_locator, exp_tag, problems, optimizers)
    mu, se = ads.range_summarize(traces)
    plot_sorted(ax, optimizers, mu, se, renames=renames)
