if __name__ == "__main__":
    import sys

    from figures.fig_util import expository_problem, mean_contours, show
    from optimizer.optimizer import Optimizer
    from problems.env_conf import default_policy, get_env_conf

    out_dir = sys.argv[1]

    env_tag, seed = expository_problem()
    num_arms = 4

    env_conf = get_env_conf(env_tag, seed)
    policy = default_policy(env_conf)
    default_num_X_samples = max(64, 10 * num_arms)

    opt = Optimizer(env_conf, policy, num_arms)

    for i_iter in range(3):
        opt.collect_trace(ttype="mtav_msts", num_iterations=1)
        acqf = opt._designers["mtav_msts"].fig_last_acqf.acq_function
        x_arms = opt._designers["mtav_msts"].fig_last_arms

    mean_contours(out_dir, acqf.model)
    # pmax_contours(acqf.model)
    # var_contours(acqf.model)

    with open(f"{out_dir}/x_max", "w") as f:
        f.write(show(acqf.X_max) + "\n")

    with open(f"{out_dir}/x_samples", "w") as f:
        for x in acqf.X_samples:
            f.write(show(x) + "\n")

    with open(f"{out_dir}/x_arms", "w") as f:
        for x in x_arms:
            f.write(show(x) + "\n")
