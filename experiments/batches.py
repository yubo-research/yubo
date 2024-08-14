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


def prep_mtv_repro(results_dir):
    from experiments.func_names import funcs_1d, funcs_nd

    exp_dir = "exp_pss_repro_mtv_3"

    opts = ["ts", "pss"]  # "mtv-pts", "pts", "mtv", "sobol", "random", "ei", "ucb", "dpp", "sr", "gibbon", "lei"]
    noises = [None]

    cmds_1d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=3, num_replications=100, opts=opts, noises=noises, num_rounds=3)
    cmds_3d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[3], num_arms=5, num_replications=30, opts=opts, noises=noises, num_rounds=3)
    cmds_10d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[10], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3)
    cmds_30d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[30], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3)
    return cmds_1d + cmds_3d + cmds_10d + cmds_30d


def prep_ts_hd(results_dir):
    from experiments.func_names import funcs_1d, funcs_nd

    # Thompson-Sampling in HD

    exp_dir = "_test"  # TEST "exp_pss_ts_hd"

    opts = ["pts", "ts", "dpp", "turbo-1", "turbo-5", "sobol", "random"]
    noises = [None]

    min_rounds = 30
    cmds = []
    # cmds.extend(
    #     prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=1, num_replications=100, opts=opts, noises=noises, num_rounds=min_rounds)
    # )
    for num_dim in [1000]:  # TEST[3, 10, 30, 100, 300, 1000]:
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

    return cmds


def prep_turbo_ackley_repro(results_dir):
    # exp_dir = "exp_pss_repro_ackley"
    # funcs_nd = ["ackley"]

    # noises = [None]

    # return prep_d_args(
    #     results_dir,
    #     exp_dir=exp_dir,
    #     funcs=funcs_nd,
    #     dims=[200],
    #     num_arms=100,
    #     num_replications=10,
    #     opts=opts,
    #     noises=noises,
    #     num_rounds=100,
    #     func_category="g",
    # )

    # Ran manually with:
    # ./experiments/experiment.py --exp-dir=result-repro --env-tag=g:ackley-200d --num-arms=100 --num-rounds=100 --num-reps=10 --opt-name=turbo
    # And again with --opt_name=   cma, pts, random, sobol
    # TuRBO took about four days.
    # PTS took about eight days.
    pass


def prep_d_argss():
    results_dir = "results"

    # assert False, "Select prep function"

    return prep_ts_hd(results_dir)


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
    if len(sys.argv) > 1:
        assert sys.argv[1] == "--dry-run"
        dry_run = True

    d_argss = prep_d_argss()
    t_0 = time.time()
    run(d_argss, max_parallel=5, b_dry_run=dry_run)
    t_f = time.time()
    print(f"TIME_ALL: {t_f-t_0:.2f}")
    print("DONE_ALL")
