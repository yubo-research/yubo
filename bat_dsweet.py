#!/usr/bin/env python

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
                cmd = f"(python experiments/exp_2.py {problem} {opt} &> {out_dir}/{opt})"
                cmds.append(cmd)
                
                if len(cmds) == max_parallel:
                    cmds.append("wait")
                    cmd = " & ".join(cmds)
                    print (cmd)
                    os.system(cmd)
                    cmds = []


if __name__=="__main__":
    funcs = ['ackley', 'dixonprice', 'griewank', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'sphere', 'stybtang']

    run(
        ddir="exp_2_bic",
        funcs=funcs,
        dims=[10],
        opts=["mtav_msts", "mtav_msts_bic"],
        max_parallel=5
    )
