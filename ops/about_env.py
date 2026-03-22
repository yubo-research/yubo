#!/usr/bin/env python

if __name__ == "__main__":
    import sys

    import numpy as np

    import common.all_bounds as all_bounds
    from optimizer.trajectories import collect_trajectory
    from problems.problem import build_problem

    if len(sys.argv) < 3:
        print("usage: about_env.py <env_tag> <policy_tag>", file=sys.stderr)
        sys.exit(1)

    env_tag = sys.argv[1]
    policy_tag = sys.argv[2]

    problem = build_problem(env_tag, policy_tag, problem_seed=17)
    env_runtime = problem.env
    policy = problem.build_policy()

    noise_levels = []
    abs_returns = []
    for _ in range(10):
        x = all_bounds.p_low + all_bounds.p_width * np.random.uniform(size=(policy.num_params(),))
        r = []
        for _ in range(10):
            policy.set_params(x)
            r.append(collect_trajectory(env_runtime, policy).rreturn)

        r = np.array(r)
        noise_levels.append(r.std())
        abs_returns.append(np.abs(r).mean())

    noise_level = np.mean(noise_levels)
    abs_return = np.mean(abs_returns)
    print(f"{env_tag} noise_level = {noise_level:.4f} abs_r = {abs_return:.4f} num_params = {policy.num_params()}")
