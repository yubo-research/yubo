import numpy as np
from gymnasium.spaces import Box

import common.all_bounds as all_bounds
from problems.benchmark_functions import all_benchmarks


def make(name, seed):
    # TODO: noise
    _, name = name.split(":")
    name, num_dim = name.split("-")
    assert num_dim[-1] == "d"
    num_dim = int(num_dim[:-1])

    all_bf = all_benchmarks()
    if name in all_bf:
        return PureFunctionEnv(all_bf[name](), num_dim, seed)
    assert False, name


# all domains are [-1,1]**num_dim
class PureFunctionEnv:
    ALPHA = 1.0

    def __init__(self, function, num_dim, seed):
        self._function = function

        # state is either 0 or 1
        # state==0 means that the policy has not been called.
        # state==1 means that it has.
        self.observation_space = Box(low=0.0, high=1.0, dtype=np.float32)
        # action is the input to the function (before some transformations)
        self.action_space = Box(low=-np.ones(num_dim), high=np.ones(num_dim), dtype=np.float32)

        rng = np.random.default_rng(seed)

        # Distort the parameter space, moving the center
        #  to a randomly-chosen corner of the bounding box.
        # alpha = scale of the corner hypercube
        alpha = PureFunctionEnv.ALPHA
        self._x_0 = all_bounds.x_low + (1 - alpha) * all_bounds.x_width + alpha * all_bounds.x_width * rng.uniform(size=(num_dim,))
        i = rng.choice(np.arange(len(self._x_0)), size=(num_dim // 2,))
        self._x_0[i] = (all_bounds.x_low + alpha * all_bounds.x_width * rng.uniform(size=(num_dim,)))[i]

        assert all_bounds.x_low == -1
        assert all_bounds.x_high == 1

    def step(self, action):
        # state, reward, done = env.step(action)[:3]
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
