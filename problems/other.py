from .pure_functions import PureFunctionEnv


def make(env_name, problem_seed):
    if env_name == "mopta08":
        from problems.mopta08 import Mopta08

        return PureFunctionEnv(Mopta08(), Mopta08.num_dim, problem_seed=problem_seed, distort=False)
    assert False, ("Unknown env_name", env_name)
