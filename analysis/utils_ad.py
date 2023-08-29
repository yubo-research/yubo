import matplotlib.pyplot as plt
import numpy as np


# def filled_err(ys, x=None, color="#AAAAAA", alpha=0.5, fmt="--", se=False, ax=None):
#     if ax is None:
#         ax = plt
#     mu = ys.mean(axis=0)
#     sg = ys.std(axis=0)
#     if x is None:
#         x = np.arange(len(mu))
#     if se:
#         sg = sg / np.sqrt(ys.shape[0])
#     ax.fill_between(x, mu - sg, mu + sg, color=color, alpha=alpha, linewidth=1)
#     ax.plot(x, mu, fmt, color=color)
def filled_err(ys, x=None, color="#AAAAAA", alpha=0.5, fmt="--", se=False, label=""):
    mu = ys.mean(axis=0)
    sg = ys.std(axis=0)
    if x is None:
        x = np.arange(len(mu))
    if se:
        sg = sg / np.sqrt(ys.shape[0])
    plt.errorbar(x, mu, yerr=2 * sg, fmt=".-", color=color, elinewidth=1, label=label)


def filled_arm_err(ys, x=None, color="#AAAAAA", alpha=0.5, fmt="", se=False, label="", arm_number=1):
    mu = ys.mean(axis=0)
    sg = ys.std(axis=0)
    if x is None:
        x = np.arange(len(mu))
    if se:
        sg = sg / np.sqrt(ys.shape[0])
    yerr = 2 * sg * arm_number
    xarm = x * arm_number
    mu = np.interp(x, xarm, mu)
    yerr = np.interp(x, xarm, yerr)
    plt.errorbar(x, mu, yerr=2 * sg, fmt=fmt, color=color, elinewidth=1, label=label)


def filled_init_err(ys, x=None,se=False,  arm_number=1):
    mu = ys.mean(axis=0)
    sg = ys.std(axis=0)
    
    if x is None:
        x = arm_number
    if se:
        sg = sg / np.sqrt(ys.shape[0])
    yerr = 2*sg
    return x, mu, yerr


def error_area(x, y, yerr, color="#AAAAAA", alpha=0.5, fmt="--"):
    mu = y
    sg = yerr
    plt.fill_between(x, mu - sg, mu + sg, color=color, alpha=alpha, linewidth=1)
    plt.plot(x, mu, fmt, color=color)


def _extractKV(line):
    x = line.strip().split()
    d = {}
    for i in range(0, len(x) - 1):
        if x[i] != "=":
            continue
        k = x[i - 1]
        v = x[i + 1]
        d[k] = v
    return d


def load(fn, keys):
    skeys = set(keys)
    data = []
    with open(fn) as f:
        for line in f.readlines():
            d = _extractKV(line)
            if skeys.issubset(set(d.keys())):
                data.append([float(d[k]) for k in keys])
    return np.array(data).squeeze()


def loadKV(fn, keys, grep_for=None):
    if isinstance(keys, str):
        keys = keys.split(",")
    skeys = set(keys)
    data = {k: [] for k in skeys}
    with open(fn) as f:
        for line in f.readlines():
            if grep_for is not None and grep_for not in line:
                continue
            d = _extractKV(line)
            for k in skeys:
                if k in d:
                    data[k].append(float(d[k]))
    out = {}
    for k, v in data.items():
        if len(v) > 0:
            out[k] = np.array(v).squeeze()
            if len(out[k].shape) == 0:
                out[k] = np.array([out[k]])
    return out


def hline(y0, ax=None):
    if ax is None:
        ax = plt
    c = ax.axis()
    ax.autoscale(False)
    ax.plot([c[0], c[1]], [y0, y0], "k--", linewidth=1)


def vline(x0, color="black"):
    c = plt.axis()
    plt.autoscale(False)
    plt.plot([x0, x0], [c[2], c[3]], "--", linewidth=1, color=color)


def zc(x):
    return (x - x.mean()) / x.std()
