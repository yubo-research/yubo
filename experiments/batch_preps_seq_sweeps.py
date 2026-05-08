from experiments.batch_preps_core import func_brief_2, prep_args_1, prep_d_args


def prep_seq(results_dir):
    exp_dir = "exp_enn_seq_jan"

    opts = [
        "sobol",
        # "random",
        # "optuna",
        # "ucb",
        # "ucb:Msparse",
        # "vecchia",
        # "turbo-one",
        # "turbo-zero",
        # "turbo-enn-fit-ucb",
        # "turbo-enn-f",
        # "turbo-one-f",
        # "turbo-zero-f",
        # "turbo-enn-p",
    ]

    noises = [None]

    min_rounds = 30
    cmds = []

    dims = [1]  # , 3, 10, 30, 100, 300, 1000]

    for num_dim in dims:
        if num_dim <= 100:
            num_replications = 30
        else:
            num_replications = 10

        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=func_brief_2,
                dims=[num_dim],
                num_arms=1,
                num_replications=num_replications,
                opts=opts,
                noises=noises,
                num_rounds=max(min_rounds, num_dim),
            )
        )

    return cmds


def prep_sweep_k(results_dir):
    exp_dir = "exp_enn_rebuttal_sweep_k"

    opts = [
        "turbo-enn-p/k=3",
        "turbo-enn-p/k=10",
        "turbo-enn-p/k=30",
        "turbo-enn-p/k=100",
    ]

    cmds = []

    dims = [10]

    for num_dim in dims:
        if num_dim <= 100:
            num_replications = 30
        else:
            num_replications = 10

        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=["ackley", "booth", "rosenbrock", "sphere"],
                dims=[num_dim],
                num_arms=10,
                num_replications=num_replications,
                opts=opts,
                noises=[None],
                num_rounds=100,
            )
        )

    return cmds


def prep_sweep_p(results_dir):
    exp_dir = "exp_enn_rebuttal_sweep_p"

    opts = [
        "turbo-enn-fit-ucb/nfs=10",
        "turbo-enn-fit-ucb/nfs=30",
        "turbo-enn-fit-ucb/nfs=100",
        "turbo-enn-fit-ucb/nfs=300",
        "turbo-enn-fit-ucb/nfs=1000",
    ]

    cmds = []

    dims = [10]

    for num_dim in dims:
        if num_dim <= 100:
            num_replications = 30
        else:
            num_replications = 10

        cmds.extend(
            prep_d_args(
                results_dir,
                exp_dir=exp_dir,
                funcs=["ackley", "booth", "rosenbrock", "sphere"],
                dims=[num_dim],
                num_arms=10,
                num_replications=num_replications,
                opts=opts,
                noises=[None],
                num_rounds=100,
            )
        )

    return cmds


def prep_push(results_dir):
    exp_dir = "exp_ennbo_rebuttal_push"

    opts = [
        # "random",
        # "optuna",
        # "turbo-zero",
        # "turbo-one",
        # "turbo-enn-fit-ucb",
        # "turbo-enn-p",
        # "cma",
        # "smac",
        # "dngo",
        # "vecchia",
        "turbo-one-ucb",
        "turbo-one-nds",
    ]

    cmds = []
    for opt in opts:
        for num_arms, num_rounds, num_denoise, num_denoise_passive, fn in [
            (1, 10000, 1, 30, False),
            (50, 300, 50, None, True),
        ]:
            # prep_args_1(results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, noise=None, num_denoise=None):
            if num_arms == 1 and opt == "cma":
                continue
            cmds.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem="push:fn" if fn else "push",
                    opt=opt,
                    num_arms=num_arms,
                    num_replications=30,
                    num_rounds=num_rounds,
                    noise=None,
                    num_denoise=num_denoise,
                    num_denoise_passive=num_denoise_passive,
                )
            )

    return cmds
