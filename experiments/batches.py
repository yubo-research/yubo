#!/usr/bin/env python

import multiprocessing
import os
import random
import time

from experiments.experiment_modal import app, dist_modal
from experiments.experiment_sampler import mk_replicates, prep_d_args


def worker(cmd):
    return os.system(cmd)


def run_batch(d_argss, b_dry_run):
    processes = []

    for d_args in d_argss:
        logs_dir = f"{d_args['exp_dir']}/logs"
        os.makedirs(logs_dir, exist_ok=True)

        cmd = ["python experiments/experiment.py"]
        for k, v in d_args.items():
            cmd.append(f"{k}={v}")
        cmd.append(f"> {logs_dir}/{d_args['opt_name']} 2>&1")
        cmd = " ".join(cmd)
        print("RUN:", cmd)
        # env_tag={problem} opt_name={opt} num_arms={num_arms} num_reps={num_replications} num_rounds={num_rounds} {num_obs} {num_denoise} {noise} exp_dir={exp_dir} > {logs_dir}/{opt} 2>&1"
        if not b_dry_run:
            process = multiprocessing.Process(target=worker, args=(cmd,))
            processes.append(process)
            process.start()

    if not b_dry_run:
        for process in processes:
            process.join()
    print("DONE_BATCH")


def run(cmds, max_parallel, b_dry_run=False):
    for k in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
        os.environ[k] = "16"

    while len(cmds) > 0:
        todo = cmds[:max_parallel]
        cmds = cmds[max_parallel:]
        run_batch(todo, b_dry_run)


def prep_d_argss():
    results_dir = "results"
    exp_dir = "exp_pss_A"

    funcs_nd = ["ackley", "dixonprice", "griewank", "levy", "michalewicz", "rastrigin", "rosenbrock", "sphere", "stybtang"]
    funcs_1d = ["ackley", "dixonprice", "griewank", "levy", "rastrigin", "sphere", "stybtang"]

    # opts_compare = ["sobol", "random", "ei", "ucb", "dpp", "sr", "gibbon", "mtv"]
    # opts_then = ["mtv_then_ei", "mtv_then_sr", "mtv_then_gibbon", "mtv_then_dpp", "mtv_then_ucb"]
    # opts_ablations = ["mtv_no-ic", "mtv_no-opt", "mtv_no-pstar"]

    # opts = opts_then + opts_ablations
    # opts = ["mtv", "ei", "ucb", "gibbon", "dpp"]
    # opts = ["mtv", "ei","ucb", "gibbon", "dpp", "sobol"]
    # opts = ["mtv"]
    # opts = opts_compare
    opts = ["mtv-pss", "mtv-pss-ts", "mtv", "mtv-ts", "ts", "ei", "sobol", "random"]
    opts = [f"mtv-pss-{i}" for i in [1, 3, 10, 30, 100]]

    noises = [None]  # 0, 0.1, 0.3]

    cmds_1d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=3, num_replications=100, opts=opts, noises=noises)

    cmds_3d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[3], num_arms=5, num_replications=30, opts=opts, noises=noises)

    cmds_10d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[10], num_arms=10, num_replications=30, opts=opts, noises=noises)

    cmds_30d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[30], num_arms=10, num_replications=30, opts=opts, noises=noises)

    if False:
        cmds_rl = []
        for opt in opts:
            for num_arms, num_obs, problem in [(5, 30, "mcc"), (30, 30, "hop")]:
                cmds_rl.append(
                    prep_d_args(
                        results_dir,
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

    cmds = cmds_1d + cmds_3d + cmds_10d + cmds_30d
    # cmds = cmds_1d + cmds_10d
    # cmds = cmds_rl
    random.shuffle(cmds)
    return cmds


@app.local_entrypoint()
def main_modal():
    b_dry_run = False

    d_argss = prep_d_argss()
    t_0 = time.time()
    batch_of_d_args = []
    for d_args in d_argss:
        batch_of_d_args.extend(mk_replicates(d_args))
    print(f"START: num_tasks = {len(batch_of_d_args)}")
    if b_dry_run:
        for d_args in batch_of_d_args:
            print("D:", d_args)
    else:
        dist_modal(batch_of_d_args)
    t_f = time.time()
    print(f"TIME_ALL: {t_f-t_0:.2f}")
    print("DONE_ALL")


if __name__ == "__main__":
    d_argss = prep_d_argss()
    t_0 = time.time()
    run(d_argss, max_parallel=3, b_dry_run=True)
    t_f = time.time()
    print(f"TIME_ALL: {t_f-t_0:.2f}")
    print("DONE_ALL")
