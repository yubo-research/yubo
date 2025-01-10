#!/usr/bin/env python

import multiprocessing
import os
import time

import experiments.batch_preps as batch_preps


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
        # env_tag={problem} opt_name={opt} num_arms={num_arms} num_reps={num_replications} num_rounds={num_rounds} {num_denoise} {noise} exp_dir={exp_dir} > {logs_dir}/{opt} 2>&1"
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


def prep_d_argss(batch_tag):
    results_dir = "results"

    batch_tags = batch_preps.__dict__.keys()

    assert batch_tag in batch_tags, f"Unknown batch_tag: {batch_tag}"
    return getattr(batch_preps, batch_tag)(results_dir)


if __name__ == "__main__":
    import sys

    dry_run = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "--dry-run":
            dry_run = True
            sys.argv = sys.argv[1:]

    batch_tag = sys.argv[1]

    d_argss = prep_d_argss(batch_tag)
    t_0 = time.time()
    run(d_argss, max_parallel=5, b_dry_run=dry_run)
    t_f = time.time()
    print(f"TIME_ALL: {t_f-t_0:.2f}")
    print("DONE_ALL")
