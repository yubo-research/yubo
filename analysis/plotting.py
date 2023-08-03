import matplotlib.pyplot as plt
import numpy as np


def tight(axs):
    for a in axs:
        a.set_box_aspect(1)
    plt.tight_layout()
    plt.show()


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


def error_area(x, y, yerr, color="#AAAAAA", alpha=0.5, fmt="--", marker=","):
    mu = y
    sg = yerr
    plt.fill_between(x, mu - sg, mu + sg, color=color, alpha=alpha, linewidth=1)
    plt.plot(x, mu, fmt, color=color)


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
