#!/usr/bin/env python

import os
import multiprocessing
import random

def worker(cmd):
    import os
    return os.system(cmd)
    
def run_batch(cmds, b_dry_run):
    processes = []

    for cmd in cmds:
        print ("RUN:", cmd)
        if not b_dry_run:
            process = multiprocessing.Process(target=worker, args=(cmd,))
            processes.append(process)
            process.start()

    if not b_dry_run:
        for process in processes:
            process.join()    
    print ("DONE_BATCH")

def prep_cmd(ddir, problem, opt, num_arms, num_replications, num_rounds, noise, num_denoise):
    if noise is not None:
        out_dir = f"results/{ddir}-{noise:.3f}/{problem}"
    else:
        out_dir = f"results/{ddir}/{problem}"
    if num_denoise is None:
        num_denoise = ""
    os.makedirs(out_dir, exist_ok=True)
    # experiments/exp_2.py env_tag ttype num_arms num_replications num_rounds noise
    if noise is None:
        noise = ""
    return f"python experiments/experiment.py {problem} {opt} {num_arms} {num_replications} {num_rounds} {num_denoise} {noise} > {out_dir}/{opt} 2>&1"

    
def prep_cmds(ddir, funcs, dims, num_arms, num_replications, opts, noises):
    num_rounds = 3
    cmds = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                for noise in noises:
                    problem = f"f:{func}-{dim}d"
                    cmds.append(prep_cmd(ddir, problem, opt, num_arms, num_replications, num_rounds, noise, num_denoise=None))
    return cmds

def run(cmds, max_parallel, b_dry_run=False):
    import os
    
    for k in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS']:
        os.environ[k] = "32"

    while len(cmds) > 0:
        todo = cmds[:max_parallel]
        cmds = cmds[max_parallel:]
        run_batch(todo, b_dry_run)
        
if __name__=="__main__":
    funcs_nd = ['ackley', 'dixonprice', 'griewank', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'sphere', 'stybtang']
    funcs_1d = ['ackley', 'dixonprice', 'griewank', 'levy', 'rastrigin', 'sphere', 'stybtang']

    opts_compare = ["sobol", "random", "ei", "ucb", "dpp", "sr", "gibbon", "mtv"]
    opts_then = ["mtv_then_ts", "mtv_then_ei", "mtv_then_sr", "mtv_then_gibbon", "mtv_then_dpp", "mtv_then_ucb"]
    opts_ablations = ["mtv_no-ic", "mtv_no-opt", "mtv_no-pstar"]

    # opts = opts_compare + opts_then + opts_ablations
    # opts = opts_then + opts_ablations
    opts = ["mtv", "ei", "ucb", "gibbon"]
    
    noises = [None] # 0, 0.1, 0.3]
    
    cmds_1d = prep_cmds(
        ddir="exp_2_mtv_1d_c",
        funcs=funcs_1d,
        dims=[1],
        num_arms=3,
        num_replications=100,
        opts=opts,
        noises=noises
    )

    cmds_3d = prep_cmds(
        ddir="exp_2_mtv_3d_c",
        funcs=funcs_nd,
        dims=[3],
        num_arms=5,
        num_replications=30,
        opts=opts,
        noises=noises
    )

    cmds_10d = prep_cmds(
        ddir="exp_2_mtv_10d_c",
        funcs=funcs_nd,
        dims=[10],
        num_arms=10,
        num_replications=30,
        opts=opts,
        noises=noises
    )

    cmds_30d = prep_cmds(
        ddir="exp_2_mtv_30d_c",
        funcs=funcs_nd,
        dims=[30],
        num_arms=10,
        num_replications=30,
        opts=opts,
        noises=noises
    )




    if True:
        cmds_rl = []
        for opt in opts:
            for problem, num_arms in [("f:tlunar", 5)]:
                cmds_rl.append(
                    prep_cmd(
                        ddir="experiment_rl",
                        problem=problem,
                        opt=opt,
                        num_arms=num_arms,
                        num_replications=30,
                        num_rounds=100,
                        noise=None,
                        num_denoise=100,
                    )
                )

    # cmds = cmds_1d + cmds_3d + cmds_10d + cmds_30d
    # cmds = cmds_1d + cmds_10d
    cmds = cmds_rl
    # random.shuffle(cmds)
    run(cmds, max_parallel=2, b_dry_run=False)
