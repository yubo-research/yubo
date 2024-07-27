#!/usr/bin/env python

import multiprocessing
import os
import time

from analysis.data_io import data_is_done
from experiments.dist_modal import DistModal
from experiments.experiment_sampler import mk_replicates, prep_d_args
from experiments.modal_interactive import app


def worker(cmd):
    return os.system(cmd)


def run_batch(d_argss, b_dry_run):
    processes = []

    for d_args in d_argss:
        logs_dir = f"{d_args['exp_dir']}/logs"
        os.makedirs(logs_dir, exist_ok=True)

        cmd = ["python experiments/experiment.py"]
        for k, v in d_args.items():
            kk = k.replace("_", "-")
            cmd.append(f"--{kk}={v}")
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
    exp_dir = "exp_pss_ts_hd"

    noises = [None]  # 0, 0.1, 0.3]
    funcs_nd = ["ackley", "dixonprice", "griewank", "levy", "michalewicz", "rastrigin", "rosenbrock", "sphere", "stybtang"]
    funcs_1d = ["ackley", "dixonprice", "griewank", "levy", "rastrigin", "sphere", "stybtang"]

    # opts_compare = ["sobol", "random", "ei", "ucb", "dpp", "sr", "gibbon", "mtv"]
    # opts_then = ["mtv_then_ei", "mtv_then_sr", "mtv_then_gibbon", "mtv_then_dpp", "mtv_then_ucb"]
    # opts_ablations = ["mtv_no-ic", "mtv_no-opt", "mtv_no-pstar"]
    opts_ts = ["mtv-pss-ts", "ts", "dpp", "turbo-1", "turbo-5", "sobol", "random"]
    opts_ts = ["turbo-1", "ts", "dpp"]

    #
    # opts_compare = ["mtv-pss", "mtv-pss-ts", "mtv", "sobol", "random", "ei", "ucb", "dpp", "sr", "gibbon"]
    opts = opts_ts

    # TuRBO repro
    # funcs_nd = ["ackley"]
    # cmds = prep_d_args(
    #     results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[200], num_arms=100, num_replications=10, opts=opts, noises=noises, num_rounds=100, func_category="g"
    # )

    # MTV repro
    if False:
        cmds_1d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=3, num_replications=100, opts=opts, noises=noises, num_rounds=3)
        cmds_3d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[3], num_arms=5, num_replications=30, opts=opts, noises=noises, num_rounds=3)
        cmds_10d = prep_d_args(
            results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[10], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3
        )
        cmds_30d = prep_d_args(
            results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[30], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3
        )
        cmds = cmds_1d + cmds_3d + cmds_10d + cmds_30d

    # Thompson-Sampling in HD
    if True:
        min_rounds = 30
        cmds = []
        # cmds.extend(
        #     prep_d_args(
        #         results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=1, num_replications=100, opts=opts, noises=noises, num_rounds=min_rounds
        #     )
        # )
        for num_dim in [1000]:  # [3, 10, 30, 100, 300, 1000]:
            cmds.extend(
                prep_d_args(
                    results_dir,
                    exp_dir=exp_dir,
                    funcs=funcs_nd,
                    dims=[num_dim],
                    num_arms=1,
                    num_replications=30,
                    opts=opts,
                    noises=noises,
                    num_rounds=max(min_rounds, num_dim),
                )
            )

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

    # random.shuffle(cmds)
    return cmds


@app.local_entrypoint()
def main_modal(job_fn: str, dry_run: bool = False):
    assert not os.path.exists(job_fn), f"{job_fn} exists"

    d_argss = prep_d_argss()
    t_0 = time.time()
    batch_of_d_args = []
    for d_args in d_argss:
        batch_of_d_args.extend(mk_replicates(d_args))
    print(f"START: num_tasks = {len(batch_of_d_args)}")
    if dry_run:
        for d_args in batch_of_d_args:
            if not data_is_done(d_args["trace_fn"]):
                print("D:", d_args)
    else:
        dist_modal = DistModal(job_fn)
        dist_modal(batch_of_d_args)
    t_f = time.time()
    print(f"TIME_ALL: {t_f-t_0:.2f}")
    if dry_run:
        dr = "_DRY_RUN"
    else:
        dr = ""
    print(f"DONE_ALL{dr}")


if __name__ == "__main__":
    import sys

    dry_run = False
    if len(sys) > 1:
        assert sys.argv[1] == "--dry-run"
        dry_run = True

    d_argss = prep_d_argss()
    t_0 = time.time()
    run(d_argss, max_parallel=3, b_dry_run=dry_run)
    t_f = time.time()
    print(f"TIME_ALL: {t_f-t_0:.2f}")
    print("DONE_ALL")
