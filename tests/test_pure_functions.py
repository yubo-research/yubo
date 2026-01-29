import numpy as np


def test_pure_function_env_init():
    from problems.benchmark_functions_1 import Sphere
    from problems.pure_functions import PureFunctionEnv

    env = PureFunctionEnv(Sphere(), num_dim=5, problem_seed=0, distort=False)
    assert env.action_space is not None
    assert env.observation_space is not None


def test_pure_function_env_step():
    from problems.benchmark_functions_1 import Sphere
    from problems.pure_functions import PureFunctionEnv

    env = PureFunctionEnv(Sphere(), num_dim=5, problem_seed=0, distort=False)
    action = np.zeros(5)
    result = env.step(action)
    assert len(result) == 4
    assert result[2] is True


def test_pure_function_env_reset():
    from problems.benchmark_functions_1 import Sphere
    from problems.pure_functions import PureFunctionEnv

    env = PureFunctionEnv(Sphere(), num_dim=5, problem_seed=0, distort=False)
    result = env.reset(seed=42)
    assert result == (0, None)


def test_pure_function_env_close():
    from problems.benchmark_functions_1 import Sphere
    from problems.pure_functions import PureFunctionEnv

    env = PureFunctionEnv(Sphere(), num_dim=5, problem_seed=0, distort=False)
    env.close()


def test_pure_function_env_distort():
    from problems.benchmark_functions_1 import Sphere
    from problems.pure_functions import PureFunctionEnv

    env = PureFunctionEnv(Sphere(), num_dim=5, problem_seed=42, distort=True)
    action = np.zeros(5)
    result = env.step(action)
    assert np.isfinite(result[1])
