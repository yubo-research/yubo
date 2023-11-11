import matplotlib.pyplot as plt
import numpy as np

import analysis.data_sets as ads

import re 

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
    print("x",n, "y", mu[i_sort], "err", se[i_sort])
    names = list(optimizers)
    if renames is not None:
        for old, new in renames.items():
            i = names.index(old)
            names[i] = new
    ax.set_xticks(n, [names[i] for i in i_sort], rotation=90)
    # ax.set_xticks(n, [names[i].split('_')[0] for i in i_sort], rotation=90)

    # print("optimizers", [names[i] for i in i_sort])
    # print("optimizers",[names[i].split('_')[0] for i in i_sort])
    ax.set_ylim([0, 1])


def plot_sorted_agg(ax, data_locator, exp_tag, optimizers=None, renames=None, i_agg=-1):
    problems = sorted(data_locator.problems_in(exp_tag))
    if optimizers is None:
        optimizers = set()
        for problem in problems:
            optimizers.update(data_locator.optimizers_in(exp_tag, problem))
    # optimizers = sorted(optimizers)

    traces = ads.load_multiple_traces(data_locator, exp_tag, problems, optimizers)
    if i_agg != -1:
        traces = traces[..., : i_agg + 1]

    mu, se = ads.range_summarize(traces)
    plot_sorted(ax, optimizers, mu, se, renames=renames)


def plot_sorted_test(ax, optimizers, mu, se, renames=None):
    i_sort = np.argsort(-mu)
    # n = [0,1,2]*6
    names = list(optimizers)
    legend=[]
    legend.extend([names[i].split('_')[0] for i in i_sort])
    n = [int(re.search(r'\d+', names[i]).group()) for i in i_sort]
    # n =  [str(x) for x in my_list]
    colbar= {"mtv":"black", "sobol":"dimgray", "random":"darkgray"}
    labels = [names[i].split('_')[0] for i in i_sort]
    X = n
    Y = mu[i_sort]
    ERR = se[i_sort]
    for j in range(len(X)):
        i = len(X)-j-1
        label = labels[i]
        ax.errorbar(str(X[i]), Y[i], ERR[i], fmt="ko", capsize=10, label=label, color =colbar[label])
    # ax.errorbar(n, mu[i_sort], se[i_sort], fmt="ko", capsize=10, label=labels, color =[colbar[i] for i in labels])
    # print("x",n, "y", mu[i_sort], "err", se[i_sort])
    # print("labels",[names[i].split('_')[0] for i in i_sort])
    if renames is not None:
        for old, new in renames.items():
            i = names.index(old)
            names[i] = new
    # ax.set_xticks(n, [names[i] for i in i_sort], rotation=90)
    # ax.set_xticks(n, [names[i].split('_')[0] for i in i_sort], rotation=90)
    # ax.set_xticks(n, [str(re.search(r'\d+', names[i]).group()) for i in i_sort], rotation=90)

    # # print("optimizers", [names[i] for i in i_sort])
    # # print("optimizers",[names[i].split('_')[0] for i in i_sort])
    # ax.set_ylim([0, 1])
    if ax == plt:
        xticks = plt.xticks
    else:
        xticks = ax.set_xticks


def plot_sorted_agg_test(ax, data_locator, exp_tag, optimizers=None, renames=None, i_agg=-1):
    problems = sorted(data_locator.problems_in(exp_tag))
    if optimizers is None:
        optimizers = set()
        for problem in problems:
            optimizers.update(data_locator.optimizers_in(exp_tag, problem))
    # optimizers = sorted(optimizers)

    traces = ads.load_multiple_traces(data_locator, exp_tag, problems, optimizers)
    if i_agg != -1:
        traces = traces[..., : i_agg + 1]

    mu, se = ads.range_summarize(traces)
    plot_sorted_test(ax, optimizers, mu, se, renames=renames)

def plot_sorted_dim(ax, optimizers, mu, se, num_arms, num_dim, indicator,renames=None):
    i_sort = np.argsort(-mu)
    # n = np.arange(len(mu))
    
    # n = [str(num_arms)]*len(mu)
    # ax.errorbar(n, mu[i_sort], se[i_sort], fmt="ko", capsize=10)
    # print("optimizer", optimizers,"x",n, "y", mu[i_sort], "err", se[i_sort])
    names = list(optimizers)
    if renames is not None:
        for old, new in renames.items():
            i = names.index(old)
            names[i] = new
    Y = mu[i_sort]
    SG =se[i_sort]
    colors = ["black","dimgray","darkgray","silver","lightgray"]
    fmts = ["X-", "+-","o-","^-","8-","s-","*-","D-","v-","P-","1-",",-"]
    linestyles=["None","--",":",".-"]
    legend = []
    for j in range(len(optimizers)):
        y = Y[j]
        sg = SG[j]
        if indicator =="Arm":
            x = str(num_dim)
        if indicator =="Dim":
            x = str(num_arms)
        
        legend.extend([names[i] for i in i_sort][j])
        ax.errorbar(x,y, sg, fmt=fmts[j], capsize=10,color = colors[j], linestyle = linestyles[j],label=[names[i] for i in i_sort][j])
    # ax.set_xticks(n, [names[i] for i in i_sort], rotation=90)
    # ax.set_xticks(n, n, rotation=90)
    # ax.set_ylim([0, 1])
    if ax == plt:
        xticks = plt.xticks
    else:
        xticks = ax.set_xticks


def plot_sorted_agg_dim(ax, data_locator, exp_tag,  num_arms, num_dim, indicator, optimizers=None, renames=None, i_agg=-1):
    problems = sorted(data_locator.problems_in(exp_tag))
    if optimizers is None:
        optimizers = set()
        for problem in problems:
            optimizers.update(data_locator.optimizers_in(exp_tag, problem))
    # optimizers = sorted(optimizers)

    traces = ads.load_multiple_traces(data_locator, exp_tag, problems, optimizers)
    if i_agg != -1:
        traces = traces[..., : i_agg + 1]

    mu, se = ads.range_summarize(traces)
    plot_sorted_dim(ax, optimizers, mu, se,  num_arms, num_dim, indicator,renames=None)