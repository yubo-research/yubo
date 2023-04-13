#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np

from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy

def sample(env_conf, ttype, tag, num_iterations, num_arms):
    policy = default_policy(env_conf)
    opt = Optimizer(env_conf, policy, num_arms=num_arms)
    name = env_conf.env_name[2:].replace("-", "_")
    for i_iter, rreturn in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations)):
        dir_path = f"results/exp_1_o/{name}/"
        filename = f"{ttype}"
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, filename), "a") as f:
            # print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {rreturn:.3f}", file=f)
            print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {rreturn:.3f}", file=f)
        
# def sample(env_conf, ttype, tag, num_iterations):
#     policy = default_policy(env_conf)
#     opt = Optimizer(env_conf, policy)
#     name = env_conf.env_name[2:].replace("-", "_")
#     for i_iter, rreturn in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations)):
#         dir_path = f"results/exp_1_m/{name}/"
#         filename = f"{ttype}"
#         os.makedirs(dir_path, exist_ok=True)
#         with open(os.path.join(dir_path, filename), "a") as f:
#             print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {rreturn:.3f}", file=f)


if __name__ == "__main__":
    import sys
    import time

    from problems.env_conf import get_env_conf

    env_tag = sys.argv[1]
    ttype = sys.argv[2]
    # "random", "sobol", "minimax", "minimax-toroidal", "variance", "iopt_ei", "ioptv_ei",
    # "idopt", "ei", "iei", "ucb", "iucb", "ax"

    num_iterations = 100
    for i_sample in range(30):
        t0 = time.time()
        seed = 17 + i_sample
        env_conf = get_env_conf(env_tag, seed)
        name = env_conf.env_name[2:].replace("-", "_")
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_iterations, num_arms=1)
        with open(f"results/exp_1_o/{name}/{ttype}", "a") as f:
            print(f"TIME_SAMPLE: {time.time() - t0:.2f}", file=f)
            
            
    
#     env_tag = sys.argv[1]
#     ttype = sys.argv[2]

#     num_iterations = 100
#     for i_sample in range(30):
#         t0 = time.time()
#         seed = 17 + i_sample
#         env_conf = get_env_conf(env_tag, seed)
#         sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_iterations, num_arms=1)
#         print(f"TIME_SAMPLE: {time.time() - t0:.2f}")

