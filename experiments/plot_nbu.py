# %matplotlib inline

import copy
import glob
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.debugger import set_trace
from IPython.display import clear_output


def add(path):
    if path not in sys.path:
        sys.path.append(path)


add("/home/juju/projects/bbo")
import analysis.utils_ad as nbu

# %load_ext autoreload
# %autoreload 1
# %aimport analysis.utils


def plot_result(fn, color="#AAAAAA", label=""):
    o = nbu.loadKV(fn, ["i_sample", "return"], grep_for="TRACE:")
    traces = []
    n_x = None
    for i_sample in np.unique(o["i_sample"]):
        i = np.where(o["i_sample"] == i_sample)[0]
        x = list(o["return"][i])
        if n_x is not None:
            while len(x) < n_x:
                x.append(x[-1])
        n_x = len(x)
        traces.append(x)
    traces = np.array(traces)
    # print("shape", traces.shape)
    # print("Y", traces)
    ys = traces
    mu = ys.mean(axis=0)
    sg = ys.std(axis=0)
    x = np.arange(len(mu))

    # print("shape mu", mu.shape)
    # print("mu", mu)
    # print("shape x", x.shape)
    # print("x", x)
    # print ("Iter rounds:",len(traces), fn)
    nbu.filled_err(traces, fmt="-", color=color, se=True, label=label)


def plot_results(exp_tag, problem_name, num_iterations, ttypes=None):
    import os

    from natsort import natsorted

    ddir = f"/home/juju/projects/bbo/results/{exp_tag}/{problem_name}/{num_iterations}iter"
    colors = ["blue", "green", "red", "black", "cyan", "magenta", "brown", "palegreen", "orange", "darkred", "darkviolet", "hotpink"]
    legend = []
    i_color = 0
    if not ttypes:
        ttypes = glob.glob(f"{ddir}/*")
        ttypes = natsorted(ttypes)
    for ttype in ttypes:
        ttype = ttype.split("/")[-1]
        if ttype in ["dumb", "rs"]:
            continue
        try:
            plot_result(f"{ddir}/{ttype}", color=colors[i_color], label=ttype)
            # legend.extend( [ttype, '_'] )
            legend.extend([ttype])
            i_color += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error in {ttype} {e}")
        if i_color == len(colors):
            i_color = 0

    # env_name = problem_name.split("_")[0]
    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        plt.legend(legend, loc="lower right", bbox_to_anchor=(1.3, 0))
    plt.xlabel("iter")
    plt.ylabel("return")
    plt.title(f"{problem_name} with {num_iterations} iterations")
    # if it doesn’t exist we create one
    myfolder = f"figures/{exp_tag}/{problem_name}/{num_iterations}iter"
    if not os.path.exists(myfolder):
        os.makedirs(myfolder)
    plt.savefig(myfolder + f"/{problem_name}" + ".png")


def plot_arm_result(fn, arm_number, color="#AAAAAA", label="", fmt=""):
    o = nbu.loadKV(fn, ["i_sample", "return"], grep_for="TRACE:")
    traces = []
    n_x = None
    for i_sample in np.unique(o["i_sample"]):
        i = np.where(o["i_sample"] == i_sample)[0]
        x = list(o["return"][i])
        if n_x is not None:
            while len(x) < n_x:
                x.append(x[-1])
        n_x = len(x)
        traces.append(x)
    traces = np.array(traces)
    # print ("Iter rounds:",len(traces), fn)
    nbu.filled_arm_err(traces, fmt=fmt, color=color, se=True, label=label, arm_number=arm_number)


def plot_arm_results(exp_tag, problem_name, num_iterations, ttypes=None):
    import os
    import re

    from natsort import natsorted

    ddir = f"/home/juju/projects/bbo/results/{exp_tag}/{problem_name}/{num_iterations}iter"
    # colors = ["blue", "green", "red", "black", "cyan", "magenta", "brown", "palegreen", "orange", "darkred", "darkviolet", "hotpink"]
    colors = ["black","dimgray","gray","darkgray","silver","lightgray"]
    fmts = ["X-", "+-","o-","^-","8-","s-","*-","D-","v-","P-","1-",",-"]
    legend = []
    i_color = 0
    if not ttypes:
        ttypes = glob.glob(f"{ddir}/*")
        ttypes = natsorted(ttypes)
    for ttype in ttypes:
        ttype = ttype.split("/")[-1]
        if ttype in ["dumb", "rs"]:
            continue
        try:

            pattern = r"_(\d+)arm"
            match = re.search(pattern, ttype)
            if match:
                arm_number = int(match.group(1))
            else:
                arm_number = 1
            plot_arm_result(f"{ddir}/{ttype}", arm_number=arm_number, color=colors[i_color], label=ttype, fmt=fmts[i_color])
            # legend.extend( [ttype, '_'] )
            legend.extend([ttype])
            i_color += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error in {ttype} {e}")
        if i_color == len(colors):
            i_color = 0

    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        plt.legend(legend, loc="lower right", bbox_to_anchor=(1.3, 0))
    plt.xlabel("rounds")
    plt.ylabel("return")
    plt.title(f"Compare value under different arm numbers \n{problem_name} with {num_iterations} iterations")
    # if it doesn’t exist we create one
    myfolder = f"figures/{exp_tag}/{problem_name}/{num_iterations}iter"
    if not os.path.exists(myfolder):
        os.makedirs(myfolder)
    plt.savefig(myfolder + f"/{problem_name}_arm_comp" + ".png")


def plot_init_result(fn,arm_number):
    o = nbu.loadKV(fn, ["i_sample", "return"], grep_for="TRACE:")
    traces = []
    n_x = None
    for i_sample in np.unique(o["i_sample"]):
        i = np.where(o["i_sample"] == i_sample)[0]
        x = list(o["return"][i])
        if n_x is not None:
            while len(x) < n_x:
                x.append(x[-1])
        n_x = len(x)
        traces.append(x)
    traces = np.array(traces)
    x, y, yerr = nbu.filled_init_err(traces, se=True, arm_number=arm_number)
    # print ("Iter rounds:",len(traces), fn)
    # nbu.filled_init_err(traces, x=xvalue,fmt=fmt, color=color, se=True, label=label)
    return x, y, yerr


def plot_init_results(exp_tag, problem_name, num_iterations, ttypes=None):
    import os
    import re

    from natsort import natsorted

    ddir = f"/home/juju/projects/bbo/results/{exp_tag}/{problem_name}/{num_iterations}iter"
    # colors = ["blue", "green", "red", "black", "cyan", "magenta", "brown", "palegreen", "orange", "darkred", "darkviolet", "hotpink"]
    colors = ["black","dimgray","gray","darkgray","silver","lightgray"]
    fmts = ["X-", "+-","o-","^-","8-","s-","*-","D-","v-","P-","1-",",-"]
    legend = []
    i_color = 0
    y_sobol = []
    x_sobol = []
    er_sobol = []
    y_center = []
    x_center = []
    er_center = []
    if not ttypes:
        ttypes = glob.glob(f"{ddir}/*")
        ttypes = natsorted(ttypes)
    for ttype in ttypes:
        ttype = ttype.split("/")[-1]
        if ttype in ["dumb", "rs"]:
            continue
        try:

            pattern = r"_(\d+)arm"
            match = re.search(pattern, ttype)
            if match:
                arm_number = int(match.group(1))
            else:
                arm_number = 1
            pattern_n = re.compile(r"^(.*?)_\d+arm$")
            match_n = re.search(pattern_n, ttype)
            if match_n:
                acqname = str(match_n.group(1))
            # traces = plot_init_result(f"{ddir}/{ttype}")
            x, y, yerr= plot_init_result(f"{ddir}/{ttype}", arm_number=arm_number)
            if acqname == "sobol":
                y_sobol.append(y)
                x_sobol.append(x)
                er_sobol.append(yerr)
            if acqname == "sobol_c":
                y_center.append(y)
                x_center.append(x)
                er_center.append(yerr)

            # legend.extend([acqname])
            # i_color += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error in {ttype} {e}")
        # if i_color == len(colors):
        #     i_color = 0
    # n = len(x_sobol)
    # n_c = len(x_center)
    y_sobol = np.array(y_sobol)[:,0]
    x_sobol = np.array(x_sobol)
    er_sobol = np.array(er_sobol)[:,0]
    y_center = np.array(y_center)[:,0]
    x_center = np.array(x_center)
    er_center = np.array(er_center)[:,0]
    plt.errorbar(x_sobol, y_sobol, er_sobol, fmt=fmts[0], color=colors[0],elinewidth=1, label="sobol")
    # nbu.filled_init_err(y_sobol, x=x_sobol, fmt="o-", color=colors[0], se=True, label="sobol")
    legend.extend(["sobol"])
    
    plt.errorbar( x_center, y_center,er_center, fmt=fmts[1], color=colors[1], elinewidth=1, label="sobol_center")
    legend.extend(["sobol_center"])
    warnings.simplefilter("ignore")
    with warnings.catch_warnings():
        plt.legend(legend, loc="lower right", bbox_to_anchor=(1.3, 0))
    plt.xlabel("arms")
    plt.ylabel("return")
    plt.title(f"Compare value under different arm numbers \n{problem_name} with {num_iterations} iterations")
    # if it doesn’t exist we create one
    myfolder = f"figures/{exp_tag}/{problem_name}/{num_iterations}iter"
    if not os.path.exists(myfolder):
        os.makedirs(myfolder)
    plt.savefig(myfolder + f"/{problem_name}_arm_comp" + ".png")
