from experiments.batch_preps_core import prep_args_1


def _eggroll_opt(sigma: float) -> str:
    return f"eggroll/noiser=eggroll/sigma={sigma}/rank=1/lr=0.01/optax=adamw/batch_size=10/num_envs=1/steps=1000"


EGGROLL_OPT = _eggroll_opt(0.05)
EGGROLL_OPT_S01 = _eggroll_opt(0.1)
EGGROLL_OPT_S02 = _eggroll_opt(0.2)
EGGROLL_OPT_S03 = _eggroll_opt(0.3)


def _prep_cheetah_eggroll_at_sigma(results_dir, *, exp_dir: str, opt: str):
    configs = []
    for policy_tag in ("linear", "mlp-32-16"):
        configs.append(
            prep_args_1(
                results_dir,
                exp_dir=exp_dir,
                problem="cheetah:fn",
                opt=opt,
                num_arms=10,
                num_replications=10,
                num_rounds=1000,
                policy_tag=policy_tag,
                num_denoise=1,
                num_denoise_passive=None,
            )
        )
    return configs


def prep_cheetah_eggroll_s01(results_dir):
    return _prep_cheetah_eggroll_at_sigma(
        results_dir,
        exp_dir="exp_cheetah_eggroll_s01",
        opt=EGGROLL_OPT_S01,
    )


def prep_cheetah_eggroll_s02(results_dir):
    return _prep_cheetah_eggroll_at_sigma(
        results_dir,
        exp_dir="exp_cheetah_eggroll_s02",
        opt=EGGROLL_OPT_S02,
    )


def prep_cheetah_eggroll_s03(results_dir):
    return _prep_cheetah_eggroll_at_sigma(
        results_dir,
        exp_dir="exp_cheetah_eggroll_s03",
        opt=EGGROLL_OPT_S03,
    )


def prep_cheetah_policy_optimizer(results_dir):
    exp_dir = "exp_cheetah_policy_optimizer"
    policies = ["linear", "mlp-32-16"]
    opts = ["random", "turbo-enn-fit-ucb", EGGROLL_OPT]
    configs = []
    for policy_tag in policies:
        for opt in opts:
            configs.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem="cheetah:fn",
                    opt=opt,
                    num_arms=10,
                    num_replications=10,
                    num_rounds=1000,
                    policy_tag=policy_tag,
                    num_denoise=1,
                    num_denoise_passive=None,
                )
            )
    return configs


def _prep_tlunar_sweep_rows(results_dir, exp_dir, opts, loop_rows):
    cmds = []
    for opt in opts:
        for num_arms, num_rounds, num_denoise, num_denoise_passive, fn in loop_rows:
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


def _prep_bw_sweep_rows(results_dir, exp_dir, opts, loop_rows):
    cmds = []
    for opt in opts:
        for (
            num_arms,
            num_rounds,
            num_reps,
            num_denoise,
            num_denoise_passive,
            fn,
        ) in loop_rows:
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


def prep_sweep_k_tlunar(results_dir):
    return _prep_tlunar_sweep_rows(
        results_dir,
        "exp_ennbo_rebuttal_sweep_k_tlunar",
        [
            "turbo-enn-fit-ucb/k=3",
            "turbo-enn-fit-ucb/k=10",
            "turbo-enn-fit-ucb/k=30",
            "turbo-enn-fit-ucb/k=100",
            "turbo-enn-fit-ucb/k=300",
        ],
        [(50, 30, 50, None, True)],
    )


def prep_sweep_p_tlunar(results_dir):
    return _prep_tlunar_sweep_rows(
        results_dir,
        "exp_ennbo_rebuttal_sweep_p_tlunar",
        [
            "turbo-enn-fit-ucb/nfs=10",
            "turbo-enn-fit-ucb/nfs=30",
            "turbo-enn-fit-ucb/nfs=100",
            "turbo-enn-fit-ucb/nfs=300",
            "turbo-enn-fit-ucb/nfs=1000",
        ],
        [(1, 10000, 1, 30, False)],
    )


def prep_sweep_k_bw(results_dir):
    return _prep_bw_sweep_rows(
        results_dir,
        "exp_ennbo_rebuttal_sweep_k_bw",
        [
            "turbo-enn-fit-ucb/k=3",
            "turbo-enn-fit-ucb/k=10",
            "turbo-enn-fit-ucb/k=30",
            "turbo-enn-fit-ucb/k=100",
            "turbo-enn-fit-ucb/k=300",
        ],
        [(50, 100, 30, 30, None, True)],
    )


def prep_sweep_p_bw(results_dir):
    return _prep_bw_sweep_rows(
        results_dir,
        "exp_ennbo_rebuttal_sweep_p_bw",
        [
            "turbo-enn-fit-ucb/nfs=10",
            "turbo-enn-fit-ucb/nfs=30",
            "turbo-enn-fit-ucb/nfs=100",
            "turbo-enn-fit-ucb/nfs=300",
            "turbo-enn-fit-ucb/nfs=1000",
        ],
        [(1, 10000, 30, None, 30, False)],
    )
