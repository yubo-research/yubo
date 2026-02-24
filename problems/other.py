from .pure_functions import PureFunctionEnv


def _make(env_name, problem_seed):
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
    elif env_name == "mnist":
        from problems.mnist_env import MnistEnv

        return MnistEnv(batch_size=4096)
    elif env_name == "mnist_fulltrain":
        from problems.mnist_env import MnistEnv

        # Batch size here is used only for test accuracy / any batch-based code paths.
        # The UHD fast path can override evaluation to score the full 60k train set.
        return MnistEnv(batch_size=4096)
    elif env_name == "mnist_fulltrain_acc":
        from problems.mnist_env import MnistEnv

        # Same MNIST module/data; uhd_setup switches the objective to full-train accuracy.
        return MnistEnv(batch_size=4096)
    assert False, ("Unknown env_name", env_name)


make = _make
