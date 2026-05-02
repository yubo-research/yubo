from experiments.batch_preps_core import prep_args_1


def _rebuttal_prep_cmds(results_dir, exp_dir, opts, loops, problem_base):
    cmds = []
    for opt in opts:
        for num_arms, num_rounds, num_reps, num_denoise, num_denoise_passive, fn in loops:
            if num_arms == 1 and opt == "cma":
                continue
            cmds.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem=f"{problem_base}:fn" if fn else problem_base,
                    opt=opt,
                    num_arms=num_arms,
                    num_replications=num_reps,
                    num_rounds=num_rounds,
                    noise=None,
                    num_denoise=num_denoise,
                    num_denoise_passive=num_denoise_passive,
                )
            )
    return cmds


def prep_tlunar(results_dir):
    # exp_dir = "exp_compare_tlunar_rust"
    exp_dir = "exp_ennbo_rebuttal_tlunar"

    opts = [
        # "random",
        # "optuna",
        # "cma",
        # "turbo-zero",
        # "turbo-one",
        # "turbo-1",
        # "turbo-enn-fit-ucb",
        # "turbo-enn-p",
        # "ucb",
        # "lei",
        # "smac",
        # "dngo",
        "vecchia",
        # "ucb:Msparse",
        # "turbo-one-ucb",
        # "turbo-one-nds",
    ]

    cmds = []
    for opt in opts:
        for num_arms, num_rounds, num_denoise, num_denoise_passive, fn in [
            # (1, 10000, 1, 30, False),
            (50, 30, 50, None, True),
        ]:
            # prep_args_1(results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, noise=None, num_denoise=None):
            if num_arms == 1 and opt == "cma":
                continue
            cmds.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem="tlunar:fn" if fn else "tlunar",
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


def prep_hop(results_dir):
    return _rebuttal_prep_cmds(
        results_dir,
        "exp_ennbo_rebuttal_hop",
        [
            "turbo-one-ucb",
            "turbo-one-nds",
        ],
        [
            (1, 10000, 30, None, 30, False),
            (50, 1000, 30, 10, None, True),
        ],
        "hop",
    )


def prep_bw(results_dir):
    exp_dir = "exp_ennbo_rebuttal_bw"

    opts = [
        # "random",
        # "optuna",
        # "cma",
        # "turbo-zero",
        # "turbo-one",
        # "turbo-enn-fit-ucb",
        # "turbo-enn-p",
        # "ucb",
        # "lei",
        # "smac",
        # "dngo",
        # "vecchia",
        # "ucb:Msparse",
        "turbo-one-ucb",
        "turbo-one-nds",
    ]

    cmds = []
    for opt in opts:
        for num_arms, num_rounds, num_reps, num_denoise, num_denoise_passive, fn in [
            (1, 10000, 30, None, 30, False),
            # (50, 100, 30, 30, None, True),
        ]:
            # prep_args_1(results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, noise=None, num_denoise=None):
            if num_arms == 1 and opt == "cma":
                continue
            cmds.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem="bw-heur:fn" if fn else "bw-heur",
                    opt=opt,
                    num_arms=num_arms,
                    num_replications=num_reps,
                    num_rounds=num_rounds,
                    noise=None,
                    num_denoise=num_denoise,
                    num_denoise_passive=num_denoise_passive,
                )
            )

    return cmds


def prep_leukemia(results_dir):
    return _rebuttal_prep_cmds(
        results_dir,
        "exp_ennbo_leukemia",
        [
            "turbo-one-ucb",
            "turbo-one-nds",
        ],
        [
            (10, 1000, 30, None, 10, False),
            (10, 1000, 30, 10, None, True),
        ],
        "leukemia",
    )


def prep_dna(results_dir):
    return _rebuttal_prep_cmds(
        results_dir,
        "exp_ennbo_dna",
        [
            "turbo-one-ucb",
            "turbo-one-nds",
        ],
        [
            (1, 1000, 30, None, 10, False),
            (1, 1000, 30, 10, None, True),
        ],
        "dna",
    )
