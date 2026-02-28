#!/usr/bin/env python

import multiprocessing
import os
import time


def _worker(cmd):
    return os.system(cmd)


worker = _worker


def _run_batch(d_argss, b_dry_run):
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
            process = multiprocessing.Process(target=_worker, args=(cmd,))
            processes.append(process)
            process.start()

    if not b_dry_run:
        for process in processes:
            process.join()
    print("DONE_BATCH")


run_batch = _run_batch


def _run(cmds, max_parallel, b_dry_run=False):
    from experiments.batch_util import run_in_batches

    run_in_batches(cmds, max_parallel, run_batch, b_dry_run=b_dry_run, num_threads=16)


run = _run


def prep_d_argss(batch_tag):
    import experiments.batch_preps as batch_preps

    results_dir = "results"
    preps = {k: v for k, v in batch_preps.__dict__.items() if k.startswith("prep_") and callable(v)}

    fn = preps.get(batch_tag)
    if fn is None and not batch_tag.startswith("prep_"):
        fn = preps.get(f"prep_{batch_tag}")

    assert fn is not None, f"Unknown batch_tag: {batch_tag} (known: {sorted(preps.keys())})"
    return fn(results_dir)


if __name__ == "__main__":
    import sys

    dry_run = False
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        dry_run = True
        sys.argv.pop(1)

    batch_tag = sys.argv[1]

    d_argss = prep_d_argss(batch_tag)
    t_0 = time.time()
    run(d_argss, max_parallel=5, b_dry_run=dry_run)
    t_f = time.time()
    print(f"TIME_ALL: {t_f - t_0:.2f}")
    print("DONE_ALL")
