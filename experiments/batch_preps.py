from experiments.experiment_sampler import prep_args_1, prep_d_args
from experiments.func_names import funcs_1d, funcs_all, funcs_nd


def prep_mtv_repro(results_dir):
    exp_dir = "exp_batches_mtv"

    opts = ["turbo-1", "lei", "sts", "mtv-sts", "optuna", "mtv", "sobol", "random", "ei", "ucb", "dpp", "sr", "gibbon", "mcmcbo", "ts-10000"]
    noises = [None]

    # cmds_1d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=3, num_replications=100, opts=opts, noises=noises, num_rounds=3)
    # cmds_3d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[3], num_arms=5, num_replications=30, opts=opts, noises=noises, num_rounds=3)
    # cmds_10d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[10], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3)
    # cmds_30d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[30], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3)
    cmds_100d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[100], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3)
    cmds_300d = prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_nd, dims=[300], num_arms=10, num_replications=30, opts=opts, noises=noises, num_rounds=3)

    # return cmds_1d + cmds_3d + cmds_10d + cmds_30d  # + cmds_100d + cmds_300d
    return cmds_100d + cmds_300d


def prep_ts_hd(results_dir):
    exp_dir = "exp_pss_ts_hd"

    # opts = ["sts2", "lei", "turbo-1", "sts", "sts-t", "optuna", "ei", "ucb", "gibbon", "sr", "ts", "turbo-1", "sobol", "random"]
    opts = ["sts-t:rdumbo"]
    noises = [None]

    min_rounds = 30
    cmds = []
    cmds.extend(
        prep_d_args(results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=1, num_replications=100, opts=opts, noises=noises, num_rounds=min_rounds)
    )

    for num_dim in [3, 10, 30, 100, 300, 1000]:
        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=funcs_nd,
                dims=[num_dim],
                num_arms=1,
                num_replications=30,
                opts=opts,
                noises=noises,
                num_rounds=max(min_rounds, num_dim),
            )
        )

    return cmds


def prep_turbo_ackley_repro(results_dir):
    # exp_dir = "exp_pss_repro_ackley"
    # funcs_nd = ["ackley"]

    # noises = [None]

    # return prep_d_args(
    #     results_dir,
    #     exp_dir=exp_dir,
    #     funcs=funcs_nd,
    #     dims=[200],
    #     num_arms=100,
    #     num_replications=10,
    #     opts=opts,
    #     noises=noises,
    #     num_rounds=100,
    #     func_category="g",
    # )

    # Ran manually with:
    # ./experiments/experiment.py --exp-dir=result-repro --env-tag=g:ackley-200d --num-arms=100 --num-rounds=100 --num-reps=10 --opt-name=turbo
    # And again with --opt_name=   cma, pts, random, sobol
    # TuRBO took about four days.
    # PTS took about eight days.
    pass


def prep_sweep_q(results_dir):
    exp_dir = "exp_sweep_q"

    opts = ["pts"]
    noises = [None]

    num_func_evals = 1000
    cmds = []

    for num_arms in [1, 3, 10, 30, 100]:
        num_rounds = int(num_func_evals / num_arms + 0.5)
        cmds.extend(
            prep_d_args(
                results_dir, exp_dir=exp_dir, funcs=funcs_1d, dims=[1], num_arms=num_arms, num_replications=100, opts=opts, noises=noises, num_rounds=num_rounds
            )
        )
        for num_dim in [3, 10, 30, 100]:
            cmds.extend(
                prep_d_args(
                    results_dir,
                    exp_dir=exp_dir,
                    funcs=funcs_nd,
                    dims=[num_dim],
                    num_arms=num_arms,
                    num_replications=30,
                    opts=opts,
                    noises=noises,
                    num_rounds=num_rounds,
                )
            )

    return cmds


def prep_ts_sweep(results_dir):
    exp_dir = "exp_ts_sweep"

    opts = ["pts", "ei", "ucb"]
    opts += [f"ts_sweep-{n}" for n in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]]

    return prep_d_args(
        results_dir,
        exp_dir=exp_dir,
        funcs=["ackley"],
        dims=[30],
        num_arms=1,
        num_replications=30,
        opts=opts,
        noises=[None],
        num_rounds=10,
    )


def prep_cum_time_dim(results_dir):
    exp_dir = "exp_cum_time_dim"

    opts = ["sts", "pss"]

    return prep_d_args(
        results_dir,
        exp_dir=exp_dir,
        funcs=["sphere"],
        dims=[1, 3, 10, 30, 100, 300],
        num_arms=1,
        num_replications=3,
        opts=opts,
        noises=[None],
        num_rounds=10,
    )


def prep_cum_time_obs(results_dir):
    exp_dir = "exp_cum_time_obs"

    opts = ["sts", "pss"]

    cmds = []

    for num_rounds in [3, 10, 30, 100, 300]:
        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=funcs_nd,
                dims=[10],
                num_arms=10,
                num_replications=3,
                opts=opts,
                noises=[None],
                num_rounds=num_rounds,
            )
        )
    return cmds


def prep_pss_sweep(results_dir):
    exp_dir = "exp_pss_sweep"

    opts = ["random"]
    # opts += [f"pss_sweep_kmcmc-{n}" for n in [1, 3, 10, 30, 100]]
    opts += [f"pss_sweep_num_mcmc-{n}" for n in [10, 30, 100, 300, 1000]]

    return prep_d_args(
        results_dir,
        exp_dir=exp_dir,
        funcs=funcs_nd,
        dims=[1, 3, 10, 30, 100, 300],
        num_arms=1,
        num_replications=10,
        opts=opts,
        noises=[None],
        num_rounds=10,
        func_category="f",
    )


def prep_sts_sweep(results_dir):
    exp_dir = "exp_sts_sweep"

    opts = ["random"]
    opts += [f"sts_sweep-{n:04d}" for n in [10, 30, 100, 300, 1000]]

    return prep_d_args(
        results_dir,
        exp_dir=exp_dir,
        funcs=funcs_nd,
        dims=[1, 3, 10, 30, 100, 300],
        num_arms=1,
        num_replications=10,
        opts=opts,
        noises=[None],
        num_rounds=10,
        func_category="f",
    )


def prep_sequential_35(results_dir):
    exp_dir = "exp_sequential_35"

    # opts = ["mcmcbo", "turbo-1", "lei", "optuna", "sts", "sobol"]
    # opts = ["sr", "random", "ucb", "gibbon", "pss", "ts-10000", "path"]
    # opts = ["tsroots"]
    # opts = ["sts", "sts-ns", "sts-ui", "sts-m", "sts-t"]
    opts = ["sts2"]
    # opts = [f"sts-ar-{i:04d}" for i in [1, 3, 10, 30, 100]]
    noises = [None]

    min_rounds = 30
    cmds = []

    for num_dim in [1, 3, 10, 30, 100]:
        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=funcs_all,
                dims=[num_dim],
                num_arms=1,
                num_replications=10,
                opts=opts,
                noises=noises,
                num_rounds=max(min_rounds, num_dim),
            )
        )

    return cmds


def prep_mtv_36(results_dir):
    exp_dir = "exp_mtv_36"

    # opts = ["turbo-1", "lei", "sts", "mtv-sts", "optuna", "mtv", "sobol", "random", "ucb", "dpp", "sr", "gibbon", "mcmcbo", "ts-10000"]
    opts = ["cma"]
    noises = [None]

    cmds = []

    for num_dim in [300]:  # [3, 10, 30, 100]:  # 1, 300
        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=funcs_all,
                dims=[num_dim],
                num_arms=max(3, min(10, num_dim)),
                num_replications=10,
                opts=opts,
                noises=noises,
                num_rounds=3,
            )
        )

    return cmds


def prep_seq(results_dir):
    exp_dir = "exp_enn_2_seq"

    opts = [
        # "random",
        # # "lei",
        # # "ucb",
        # "turbo-0",
        # "turbo-1",
        # # "optuna",
        # "turbo-enn-10",
        # "enn-3",
        # "enn-sw-3",
        "enn-ss-3",
    ]

    noises = [None]

    min_rounds = 30
    cmds = []

    # dims = [1, 3, 10, 30, 100]
    dims = [1, 3, 10, 30, 100, 300, 1000]
    for num_dim in dims:
        if num_dim == 1000 and opts == "path:Osab":
            continue

        if num_dim <= 100:
            num_replications = 3
        else:
            num_replications = 3
        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=funcs_all,
                dims=[num_dim],
                num_arms=1,
                num_replications=num_replications,
                opts=opts,
                noises=noises,
                num_rounds=max(min_rounds, num_dim),
            )
        )

    return cmds


def prep_rl_three(results_dir, name):
    exp_dir = f"exp_enn_{name}"

    opts = [
        "turbo-enn-10",
        "turbo-f",
        "turbo-0",
        "random",
        "enn-p-10",
    ]

    cmds = []
    for opt in opts:
        for num_arms, num_rounds, num_denoise in [
            (1, 100, 1),
            # (10, 100, 10),
            # (50, 30, 50),
        ]:
            # prep_args_1(results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, noise=None, num_denoise=None):
            if num_arms == 1 and opt == "cma":
                continue
            cmds.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem=f"{name}:fn",
                    opt=opt,
                    num_arms=num_arms,
                    num_replications=100,
                    num_rounds=num_rounds,
                    noise=None,
                    num_denoise=num_denoise,
                )
            )

    return cmds


def prep_tlunar(results_dir):
    return prep_rl_three(results_dir, "tlunar")


def prep_swim(results_dir):
    return prep_rl_three(results_dir, "swim")


def prep_hop(results_dir):
    return prep_rl_three(results_dir, "hop")


def prep_rl_one(results_dir, name):
    exp_dir = f"exp_enn_{name}"

    opts = ["turbo-f", "turbo-enn-10", "random"]

    cmds = []
    for opt in opts:
        cmds.append(
            prep_args_1(
                results_dir,
                exp_dir=exp_dir,
                problem=f"{name}:fn",
                opt=opt,
                num_arms=100,
                num_replications=30,
                num_rounds=1000,
                noise=None,
                num_denoise=1,
            )
        )

    return cmds


def prep_ant(results_dir):
    return prep_rl_one(results_dir, "ant")


def prep_human(results_dir):
    return prep_rl_one(results_dir, "human")
