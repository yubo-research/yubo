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


def error_area(x, y, yerr, color="#AAAAAA", alpha=0.5, fmt="--", marker="", ax=plt):
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
    ax.errorbar(n, mu[i_sort], se[i_sort], fmt="k,", capsize=6)
    names = list(optimizers)
    if renames is not None:
        for old, new in renames.items():
            if old in names:
                i = names.index(old)
                names[i] = new
    ax.set_xticks(n, [names[i] for i in i_sort], rotation=90)
    ax.set_ylim([0, 1])


def plot_sorted_agg(ax, data_locator, exp_tag, optimizers=None, problems=None, renames=None, i_agg=-1, old_way=True):
    if problems is None:
        problems = sorted(data_locator.problems_in(exp_tag))

    if optimizers is None:
        optimizers = set()
        for problem in problems:
            optimizers.update(data_locator.optimizers_in(exp_tag, problem))
    # optimizers = sorted(optimizers)

    traces = ads.load_multiple_traces(data_locator, exp_tag, problems, optimizers, old_way=old_way)
    if i_agg != -1:
        traces = traces[..., : i_agg + 1]

    mu, se = ads.range_summarize(traces)
    plot_sorted(ax, optimizers, mu, se, renames=renames)


def plot_compare_problem(ax, data_locator, exp_name, problem_name, optimizers, b_normalize, title, renames=None, old_way=True, b_legend=True):
    handles = []
    legend = []
    i_marker = 0
    markers = ["o", "v", "s", "+", "*", "X", "^"]
    traces = ads.load_multiple_traces(data_locator, exp_name, [problem_name], optimizers, old_way=old_way)

    if b_normalize:
        yy = traces.squeeze(0).transpose((1, 0, 2))
        ys = yy.shape
        yy = yy.reshape((ys[0], ys[1] * ys[2]))
        y_min = np.expand_dims(yy.min(axis=-1), (0, 1, 3))
        y_max = np.expand_dims(yy.max(axis=-1), (0, 1, 3))
        z = (traces - y_min) / (y_max - y_min)
    else:
        z = traces

    z = z.squeeze(0)

    for i_opt, optimizer in enumerate(optimizers):
        y = z[i_opt, ...]
        x = 1 + np.arange(y.shape[1])
        filled_err(x=x, ys=y, ax=ax, se=True, alpha=0.25)
        (h,) = ax.plot(
            x,
            y.mean(axis=0),
            linestyle="none",
            marker=markers[i_marker],
            markersize=10,
            color="#444444",
            fillstyle="none",
        )
        if len(x) < 10:
            ax.set_xticks(ticks=x)
        handles.append(h)
        if renames is not None and optimizer in renames:
            name = renames[optimizer]
        else:
            name = optimizer
        legend.append(name)
        i_marker += 1

    if b_legend:
        ax.legend(handles, legend)
    ax.set_xlabel("round")
    return handles, legend
