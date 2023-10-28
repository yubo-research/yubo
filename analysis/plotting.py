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


def plot_agg(data_locator, exp_tag, problem_names, optimizer_names, i_only):
    normalized_summaries = ads.load_as_normalized_summaries(exp_tag, problem_names, optimizer_names, data_locator, i_only)
    agg = ads.aggregate_normalized_summaries(normalized_summaries)
    colors = ["blue", "green", "red", "black", "cyan", "magenta"]
    markers = [".", "o", "v", "^", "s"]
    i_color = 0
    i_marker = 0
    for optimizer_name in optimizer_names:
        n = len(agg[optimizer_name][0])
        mu, sg = agg[optimizer_name]
        error_area(
            np.arange(n),
            mu,
            sg,
            color=colors[i_color],
            marker=markers[i_marker],
        )
        i_color = (i_color + 1) % len(colors)
        i_marker = (i_marker + 1) % len(markers)


def plot_agg_final(ax, data_locator, exp_tag, problems, optimizers, sort=False, ranks=False, i_agg=-1, renames=None):
    if ranks:
        agg = ads.agg_rank_summaries(exp_tag, problems, optimizers, data_locator)
    else:
        normalized_summaries = ads.load_as_normalized_summaries(exp_tag, problems, optimizers, data_locator, i_only=i_agg)
        agg = ads.aggregate_normalized_summaries(normalized_summaries)

    if renames is None:
        renames = list(optimizers)
    if sort:
        data = []
        for rename, optimizer_name in zip(renames, optimizers):
            if optimizer_name not in agg:
                continue
            mu, sg = agg[optimizer_name]

            if not ranks:
                mu = mu[i_agg]
            data.append((-mu, rename, optimizer_name))
        data = sorted(data)
        optimizers = [d[2] for d in data]
        renames = [d[1] for d in data]

    # colors = ["blue", "green", "red", "black", "cyan", "magenta"]
    # markers = [".", "o", "v", "^", "s"]
    # i_color = 0
    # i_marker = 0
    agg_final = {}
    for optimizer_name in optimizers:
        if optimizer_name not in agg:
            continue
        mu, sg = agg[optimizer_name]
        if not ranks:
            mu = mu[i_agg]
            sg = sg[i_agg]
        agg_final[optimizer_name] = (mu, sg)

    n = np.arange(len(optimizers))
    o = np.array([agg_final[n] for n in optimizers])
    ax.errorbar(n, o[:, 0], o[:, 1], fmt="ko", capsize=10)
    # ap.error_area(n, o[:,0], o[:,1])
    # plt.plot(n, o[:,0], 'ko--');
    if ax == plt:
        xticks = plt.xticks
    else:
        xticks = ax.set_xticks
    xticks(n, renames, rotation=90)


def plot_agg_all(ax, data_locator, exp_tag, optimizers=None, sort=False, i_agg=-1, renames=None):
    problems, optimizers_actual = ads.all_in(exp_tag)
    if optimizers is None:
        optimizers = optimizers_actual
    plot_agg_final(ax, data_locator, exp_tag, problems, optimizers, sort=sort, i_agg=i_agg, renames=renames)
    return optimizers
