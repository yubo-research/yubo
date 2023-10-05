#!/usr/bin/env python

import os
import multiprocessing

normalized_summaries = load_as_normalized_summaries(exp_tag, problem_names, optimizer_names, data_locator)
summarized_agg_by_opt= aggregate_normalized_summaries(normalized_summaries)
        
if __name__=="__main__":
    funcs_10d = ['ackley', 'dixonprice', 'griewank', 'levy', 'michalewicz', 'rastrigin', 'rosenbrock', 'sphere', 'stybtang']
    funcs_1d = ['ackley', 'dixonprice', 'griewank', 'levy', 'rastrigin', 'sphere', 'stybtang']

    # opts=["sobol", "sobol_c", "ei", "ucb", "ei_c", "mcmc_ts", "mtav_ei", "mtav_ts", "mtav_ucb", "ucb_c"],

    # opts = ["mtv_then_ei", "mtv_then_ucb"]
    opts = ["mtv"]
    nums_arms = [1,3,10,30,100]
    cmds = prep_cmds(
        ddir="exp_2_mtv_1d",
        funcs=funcs_1d,
        dims=[1],
        nums_arms=nums_arms,
        num_samples=3,
        opts=opts,
    ) + prep_cmds(
        ddir="exp_2_mtv_10d",
        funcs=funcs_10d,
        dims=[10],
        nums_arms=nums_arms,
        num_samples=3,
        opts=opts,
    )

    run(cmds, max_parallel=10, b_dry_run=False)