import numpy as np
from gymnasium.spaces import Box

import common.all_bounds as all_bounds
from problems.benchmark_functions import all_benchmarks


def make(name, problem_seed):
    _, name = name.split(":")
    name, num_dim = name.split("-")
    assert num_dim[-1] == "d"
    num_dim = int(num_dim[:-1])

    all_bf = all_benchmarks()
    if name in all_bf:
        return PureFunctionEnv(all_bf[name](), num_dim, problem_seed)
    assert False, name


# all domains are [-1,1]**num_dim
class PureFunctionEnv:
    def __init__(self, function, num_dim, problem_seed, distort=True):
        self._function = function

        # state is either 0 or 1
        # state==0 means that the policy has not been called.
        # state==1 means that it has.
        self.observation_space = Box(low=0.0, high=1.0, dtype=np.float32)
        # action is the input to the function (before some transformations)
        self.action_space = Box(low=-np.ones(num_dim, dtype=np.float32), high=np.ones(num_dim, dtype=np.float32))

        if distort:
            rng = np.random.default_rng(problem_seed)

            # Distort the parameter space, moving the center
            #  to a randomly-chosen corner of the bounding box.
            self._x_0 = all_bounds.x_low + all_bounds.x_width * rng.uniform(size=(num_dim,))
        else:
            self._x_0 = np.zeros(shape=(num_dim,))

        assert all_bounds.x_low == -1
        assert all_bounds.x_high == 1

    def step(self, action):
        # mimic: state, reward, done = env.step(action)[:3]
        # Here "action" is just the x values to supply to the pure function.
        #  The only meaningful return value is the output of the pure function.
        # We go through this just to preserve the step(action) interface for
        #  future use.
        #

        assert np.all(action >= -1) and np.all(action <= 1), ("action", action)
        x = action - self._x_0

        # consider action == -1
        # (action - x0) / (1 + x0) == (-1 - x0) / (1 + x0) == -1
        x[x < 0] /= 1 + self._x_0[x < 0]

        # consider action == 1
        # (action - x0) / (1 - x0) == (1 - x0) / (1 - x0) == 1
        x[x > 0] /= 1 - self._x_0[x > 0]

        assert np.all(x >= -1) and np.all(x <= 1), ("x", x)
        return 1, -self._function(x), True, None

    def reset(self, seed):
        return 0, None

    def close(self):
        pass
