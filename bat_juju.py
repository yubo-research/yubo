#!/usr/bin/env python

import os
import multiprocessing

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

def prep_cmds(ddir, funcs, dims, num_arms, num_samples, opts):
    cmds = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                problem = f"f:{func}-{dim}d"
                out_dir = f"results/{ddir}/{problem}"
                os.makedirs(out_dir, exist_ok=True)
                cmd = f"python experiments/exp_2.py {problem} {opt} {num_arms} 100 {num_samples} > {out_dir}/{opt} 2>&1"
                cmds.append(cmd)
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
    funcs_10d = ['ackley', 'dixonprice', 'griewank', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'sphere', 'stybtang']
    funcs_1d = ['ackley', 'dixonprice', 'griewank', 'levy', 'rastrigin', 'sphere', 'stybtang']

    # opts=["sobol", "sobol_c", "ei", "ucb", "ei_c", "mcmc_ts", "mtav_ei", "mtav_ts", "mtav_ucb", "ucb_c"],

    opts = ["sobol", "random","mtv"]
    
    # cmds = prep_cmds(
    #     ddir="exp_2_figure/plot_1d_1arm",
    #     funcs=funcs_1d,
    #     dims=[1],
    #     num_arms=1,
    #     num_samples=1,
    #     opts=opts,
    # ) + prep_cmds(
    #     ddir="exp_2_figure/plot_2d_1arm",
    #     funcs=funcs_10d,
    #     dims=[2],
    #     num_arms=1,
    #     num_samples=1,
    #     opts=opts,
    # )+  
    cmds = prep_cmds(
        ddir="exp_2_figure/plot_3d_1arm",
        funcs=funcs_10d,
        dims=[3],
        num_arms=1,
        num_samples=1,
        opts=opts,
    )+ prep_cmds(
        ddir="exp_2_figure/plot_10d_1arm",
        funcs=funcs_10d,
        dims=[10],
        num_arms=1,
        num_samples=1,
        opts=opts,
    )+ prep_cmds(
        ddir="exp_2_figure/plot_30d_1arm",
        funcs=funcs_10d,
        dims=[30],
        num_arms=1,
        num_samples=1,
        opts=opts,
    )+ prep_cmds(
        ddir="exp_2_figure/plot_100d_1arm",
        funcs=funcs_10d,
        dims=[100],
        num_arms=1,
        num_samples=1,
        opts=opts,
    )

    run(cmds, max_parallel=10, b_dry_run=False)
