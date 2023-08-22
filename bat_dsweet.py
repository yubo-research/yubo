#!/usr/bin/env python

import multiprocessing

def worker(cmd):
    import os
    return os.system(cmd)
    
def run_batch(cmds):
    processes = []

    for cmd in cmds:
        process = multiprocessing.Process(target=worker, args=(cmd,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()    
    print ("DONE_BATCH")

def run(ddir, funcs, dims, opts, max_parallel):
    import os
    
    for k in ['MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS']:
        os.environ[k] = "32"

    cmds = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                problem = f"f:{func}-{dim}d"
                out_dir = f"results/{ddir}/{problem}"
                os.makedirs(out_dir, exist_ok=True)
                cmd = f"python experiments/exp_2.py {problem} {opt} > {out_dir}/{opt} 2>&1"
                cmds.append(cmd)
                
                if len(cmds) == max_parallel:
                    run_batch(cmds)
                    cmds = []

    if len(cmds) > 0:
        run_batch(cmds)

if __name__=="__main__":
    funcs = ['ackley', 'dixonprice', 'griewank', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'sphere', 'stybtang']
    
    run(
        ddir="exp_2_mtavs",
        funcs=funcs,
        dims=[10],
        # opts=[f"mtav_msts_beta={beta}" for beta in [0, 1, 2, 3]],
        # opts=["mtav_g", "ei", "ucb"],
        # opts=["ei_c", "mcmc_ts", "mtav_ei", "mtav_ts", "mtav_ucb", "ucb_c"],
        # opts=["mtav_beta=1"],
        opts=["sobol", "sobol_center"],
        max_parallel=7,
    )
