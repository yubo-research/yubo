REP_INDEX_BASE = 18


def problem_seed_from_rep_index(i_rep: int) -> int:
    return REP_INDEX_BASE + int(i_rep)


def noise_seed_0_from_problem_seed(problem_seed: int) -> int:
    return 10 * int(problem_seed)


def global_seed_for_run(problem_seed: int) -> int:
    return int(problem_seed) + 27
