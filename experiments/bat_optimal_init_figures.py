#!/usr/bin/env python

import multiprocessing
import os
import random

from experiments.batch_util import run_in_batches


def _worker(cmd):
    import os

    return os.system(cmd)


worker = _worker


def _run_batch(cmds, b_dry_run):
    processes = []

    for cmd in cmds:
        print("RUN:", cmd)
        if not b_dry_run:
            process = multiprocessing.Process(target=_worker, args=(cmd,))
            processes.append(process)
            process.start()

    if not b_dry_run:
        for process in processes:
            process.join()
    print("DONE_BATCH")


run_batch = _run_batch


def prep_cmd(
    exp_dir,
    problem,
    opt,
    num_arms,
    num_replications,
    num_rounds,
    noise=None,
    num_denoise=None,
    num_obs=None,
):
    # TODO: noise subdir?
    assert noise is None, "NYI"

    exp_dir = f"_test_results/{exp_dir}"
    if num_denoise is None:
        num_denoise = ""
    else:
        num_denoise = f"num_denoise={num_denoise}"

    if num_obs is None:
        num_obs = ""
    else:
        num_obs = f"num_obs={num_obs}"

    logs_dir = f"{exp_dir}/logs"
    os.makedirs(logs_dir, exist_ok=True)
    if noise is None:
        noise = ""
    else:
        noise = f"noise={noise}"

    # python experiments/experiment.py num_rounds=30 num_arms=5 env_tag=tlunar opt_name=gibbon num_reps=1 exp_dir=y_test num_denoise=100
    return f"python experiments/experiment.py env_tag={problem} opt_name={opt} num_arms={num_arms} num_reps={num_replications} num_rounds={num_rounds} {num_obs} {num_denoise} {noise} exp_dir={exp_dir} > {logs_dir}/{opt} 2>&1"


def prep_cmds(exp_dir, funcs, dims, num_arms, num_replications, opts, noises):
    num_rounds = 3
    cmds = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                for noise in noises:
                    problem = f"f:{func}-{dim}d"
                    cmds.append(
                        prep_cmd(
                            exp_dir,
                            problem,
                            opt,
                            num_arms,
                            num_replications,
                            num_rounds,
                            noise,
                            num_denoise=None,
                        )
                    )
    return cmds


def _run(cmds, max_parallel, b_dry_run=False):
    run_in_batches(cmds, max_parallel, run_batch, b_dry_run=b_dry_run, num_threads=32)


run = _run


if __name__ == "__main__":
    import sys

    ia = 1
    if sys.argv[ia] == "--dry-run":
        b_dry_run = True
        ia += 1
    else:
        b_dry_run = False
    figure_name = sys.argv[ia]

    funcs_1d = [
        "ackley",
        "dixonprice",
        "griewank",
        "levy",
        "rastrigin",
        "sphere",
        "stybtang",
    ]
    funcs_nd = funcs_1d + ["michalewicz", "rosenbrock"]

    opts_compare = ["sobol", "random", "ei", "ucb", "dpp", "sr", "gibbon", "mtv"]
    opts_ensemble = [
        "mtv_then_ei",
        "mtv_then_sr",
        "mtv_then_gibbon",
        "mtv_then_dpp",
        "mtv_then_ucb",
    ]
    opts_ablate = ["mtv_no-ic", "mtv_no-opt", "mtv_no-pstar"]

    noises = [None]

    if figure_name in ["compare", "rl"]:
        opts = opts_compare
    elif figure_name == "ablate":
        opts = opts_ablate
    elif figure_name == "ensemble":
        opts = opts_ensemble
    else:
        assert False, "Pick a figure_name to reproduce"

    if figure_name != "rl":
        cmds_1d = prep_cmds(
            exp_dir="exp_2024_1d",
            funcs=funcs_1d,
            dims=[1],
            num_arms=3,
            num_replications=100,
            opts=opts,
            noises=noises,
        )
        cmds_3d = prep_cmds(
            exp_dir="exp_2024_3d",
            funcs=funcs_nd,
            dims=[3],
            num_arms=5,
            num_replications=30,
            opts=opts,
            noises=noises,
        )
        cmds_10d = prep_cmds(
            exp_dir="exp_2024_10d",
            funcs=funcs_nd,
            dims=[10],
            num_arms=10,
            num_replications=30,
            opts=opts,
            noises=noises,
        )
        cmds_30d = prep_cmds(
            exp_dir="exp_2024_30d",
            funcs=funcs_nd,
            dims=[30],
            num_arms=10,
            num_replications=30,
            opts=opts,
            noises=noises,
        )
        cmds = cmds_1d + cmds_3d + cmds_10d + cmds_30d
    else:
        cmds = []
        for opt in opts:
            for num_arms, num_obs, problem in [(5, 30, "mcc"), (30, 30, "hop")]:
                cmds.append(
                    prep_cmd(
                        exp_dir=f"experiment_rl_obs_q{num_arms}_o{num_obs}",
                        problem=problem,
                        opt=opt,
                        num_arms=num_arms,
                        num_replications=100,
                        num_rounds=3,
                        noise=None,
                        num_obs=num_obs,
                        num_denoise=None,
                    )
                )

    random.shuffle(cmds)
    run(cmds, max_parallel=10, b_dry_run=b_dry_run)
