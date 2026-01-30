from .pure_functions import PureFunctionEnv


def make(env_name, problem_seed):
    if env_name == "mopta08":
        from problems.mopta08 import Mopta08

        return PureFunctionEnv(Mopta08(), Mopta08.num_dim, problem_seed=problem_seed, distort=False)
    elif env_name == "push":
        from problems.push import Push

        return PureFunctionEnv(Push(), Push.num_dim, problem_seed=problem_seed, distort=False)
    elif env_name == "leukemia":
        from problems.leukemia_env import LeukemiaEnv

        return LeukemiaEnv(seed=problem_seed)
    elif env_name == "dna":
        from problems.dna_env import DnaEnv

        return DnaEnv(seed=problem_seed)
    elif env_name == "rcv1":
        from problems.rcv1_env import Rcv1Env

        return Rcv1Env(seed=problem_seed)
    assert False, ("Unknown env_name", env_name)
