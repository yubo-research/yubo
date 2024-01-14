if __name__ == "__main__":
    import os
    import shutil
    import sys

    from figures.fig_util import expository_problem, mean_func_contours, show
    from optimizer.arm_best_obs import ArmBestObs
    from optimizer.optimizer import Optimizer

    out_dir = sys.argv[1]
    out_dir += "/fig_sequence"
    shutil.rmtree(out_dir, ignore_errors=True)

    env_conf, policy, ttype = expository_problem()

    num_arms = 4
    default_num_X_samples = max(64, 10 * num_arms)

    opt = Optimizer(env_conf, policy, num_arms=num_arms, num_denoise=None, num_obs=1, arm_selector=ArmBestObs())

    for i_iter in range(3):
        opt.collect_trace(ttype=ttype, num_iterations=1)
        acqf = opt._designers[ttype].fig_last_acqf.acq_function
        x_arms = opt._designers[ttype].fig_last_arms

        if i_iter in [0, 1, 2]:
            od = f"{out_dir}/{i_iter}"
            os.makedirs(od, exist_ok=True)
            mean_func_contours(od, env_conf)
            # pmax_contours(acqf.model)
            # TODO: contours of underlying problem function (not the GP)
            # var_contours(od, acqf.model)

            with open(f"{od}/x_arms", "w") as f:
                for x in x_arms:
                    f.write(show(x) + "\n")
