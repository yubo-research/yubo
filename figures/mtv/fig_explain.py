if __name__ == "__main__":
    import os
    import shutil
    import sys

    from figures.fig_util import expository_problem, mean_gp_contours, show
    from optimizer.arm_best_obs import ArmBestObs
    from optimizer.optimizer import Optimizer

    out_dir = sys.argv[1]
    out_dir += "/fig_explain"
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    env_conf, policy, designer_name = expository_problem()

    num_arms = 4
    default_num_X_samples = max(64, 10 * num_arms)

    opt = Optimizer(env_conf, policy, num_arms=num_arms, num_denoise=None, num_obs=1, arm_selector=ArmBestObs())

    for i_iter in range(3):
        opt.collect_trace(designer_name=designer_name, num_iterations=1)
        acqf = opt._designers[designer_name].fig_last_acqf.acq_function
        x_arms = opt._designers[designer_name].fig_last_arms

    mean_gp_contours(out_dir, acqf.model)
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
