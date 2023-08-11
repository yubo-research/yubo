#!/usr/bin/env python
# coding: utf-8

# In[ ]:
    # ========= ===============================================================
    # Character Meaning
    # --------- ---------------------------------------------------------------
    # 'r'       open for reading (default)
    # 'w'       open for writing, truncating the file first
    # 'x'       create a new file and open it for writing
    # 'a'       open for writing, appending to the end of the file if it exists
    # 'b'       binary mode
    # 't'       text mode (default)
    # '+'       open a disk file for updating (reading and writing)
    # 'U'       universal newline mode (deprecated)
    # ========= ===============================================================

import os

import numpy as np

from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy

def sample(env_conf, ttype, tag, num_iterations, num_arms,foldername):
    policy = default_policy(env_conf)
    opt = Optimizer(env_conf, policy, num_arms=num_arms)
    name = env_conf.env_name[2:].replace("-", "_")
    for i_iter, te in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations)):
        dir_path = f"results/{foldername}/{name}/{num_iterations}iter/"
        filename = f"{ttype}_{num_arms}arm"
        path = os.path.join(dir_path, filename)
        # print(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "a") as f:
            f.write(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}\n")
#             print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}", file=f)
        

if __name__ == "__main__":
    import sys
    import time

    from problems.env_conf import get_env_conf

    env_tag = sys.argv[1]
    ttype = sys.argv[2]
    # "random", "sobol", "minimax", "minimax-toroidal", "variance", "iopt_ei", "ioptv_ei",
    # "idopt", "ei", "iei", "ucb", "iucb", "ax"
    q_num = int(sys.argv[3])
    
    num_iterations = int(sys.argv[4])
    # print(type(q_num))
    # FOLDER NAME IF YOU NEED TO CHANGE 
    foldername="exp_test"
    for i_sample in range(30):
        t0 = time.time()
        seed = 17 + i_sample
        env_conf = get_env_conf(env_tag, seed)
        name = env_conf.env_name[2:].replace("-", "_")
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_iterations, num_arms=q_num,foldername=foldername)
        with open(f"results/{foldername}/{name}/{num_iterations}iter/{ttype}_{q_num}arm", "a") as f:
            f.write(f"TIME_SAMPLE: {time.time() - t0:.2f}\n")
    f.close()
            # print(f"TIME_SAMPLE: {time.time() - t0:.2f}", file=f)
            
            