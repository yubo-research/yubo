import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

import analysis.data_sets as ads

linestyles = ["-", ":", "--", "-."] * 10
markers = ["o", "x", "v", ".", "s"] * 10


def mk_trans(fig, x=10 / 72, y=5 / 72):
    trans = mtransforms.ScaledTranslation(x, y, fig.dpi_scale_trans)
    return trans


def hash_marks(ax, x, y, k, color, alpha=1):
    # Thanks, ChatGPT!
    x = np.asarray(x)
    y = np.asarray(y)

    # Calculate the direction vector
    direction = y - x

    # Normalize the direction vector
    length = np.sqrt(direction[0] ** 2 + direction[1] ** 2)
    direction_normalized = direction / length

    # Calculate a point k of the way from x to y
    mid_point = x + k * direction

    # Calculate the perpendicular vector
    perpendicular = np.array([-direction_normalized[1], direction_normalized[0]])

    # Length of the perpendicular line
    perpendicular_length = 0.05

    # Calculate the endpoints of the perpendicular line
    perpendicular_start = mid_point - (perpendicular_length / 2) * perpendicular
    perpendicular_end = mid_point + (perpendicular_length / 2) * perpendicular

    # Plot the perpendicular line
    ax.plot(
        [perpendicular_start[0], perpendicular_end[0]],
        [perpendicular_start[1], perpendicular_end[1]],
        color=color,
        alpha=alpha,
    )


def slabel(trans, ax, a):
    return ax.text(
        0.8,
        0.97,
        f"({a})",
        transform=ax.transAxes + trans,
        # fontsize=14,
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="1", edgecolor="none", pad=3.0),
    )


def subplots(n, m, figsize):
    fig_width = m / n * figsize
    fig, axs = plt.subplots(n, m, figsize=(fig_width, figsize))
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]
    return fig, axs


def tight(axs, sub_aspect=1):
    for a in axs:
        a.set_box_aspect(sub_aspect)
    plt.tight_layout()
    # plt.show()


def filled_err(ys, x=None, color="#AAAAAA", alpha=0.5, marker=None, linestyle="--", color_line="#AAAAAA", se=False, two=False, ax=None):
    if ax is None:
        ax = plt
    mu = ys.mean(axis=0)
    sg = ys.std(axis=0)
    if x is None:
        x = np.arange(len(mu))
    if se:
        sg = sg / np.sqrt(ys.shape[0])
    if two:
        sg *= 2
    ax.fill_between(x, mu - sg, mu + sg, color=color, alpha=alpha, linewidth=1, label="_nolegend_")
    ax.plot(x, mu, color=color_line, marker=marker, linestyle=linestyle)


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


def plot_sorted(ax, optimizers, mu, se, renames=None, b_sort=True):
    if b_sort:
        i_sort = np.argsort(-mu)
    else:
        i_sort = np.arange(len(mu))
    n = np.arange(len(mu))
    ax.errorbar(n, mu[i_sort], 2 * se[i_sort], fmt="k,", capsize=6)
    names = list(optimizers)
    if renames is not None:
        for old, new in renames.items():
            if old in names:
                i = names.index(old)
                names[i] = new
    ax.set_xticks(n, [names[i] for i in i_sort], rotation=60, ha="right", va="top")
    ax.set_ylim([0, 1])


def plot_sorted_agg(ax, data_locator, renames=None, i_agg=-1, b_sort=True):
    traces = ads.load_multiple_traces(data_locator)

    if i_agg == "mean":
        mu, se = ads.rank_summarize(traces)
    else:
        if i_agg != -1:
            traces = traces[..., : i_agg + 1]
        mu, se = ads.range_summarize(traces)

    plot_sorted(ax, data_locator.optimizers(), mu, se, renames=renames, b_sort=b_sort)


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
