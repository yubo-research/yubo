import numpy as np
from gymnasium.spaces import Box


def make(name, seed):
    # TODO: noise
    _, name = name.split(":")
    name, num_dim = name.split("-")
    assert num_dim[-1] == "d"
    num_dim = int(num_dim[:-1])
    if name == "sphere":
        return PureFunctionEnv(Sphere(seed, num_dim), num_dim)
    assert False, name


# all maxes are 0
# all domains are [-1,1]**D
class Sphere:
    def __init__(self, seed, num_dim):
        rng = np.random.default_rng(seed)
        self._x_0 = rng.uniform(size=(num_dim,))

    def __call__(self, x):
        return -(((x - self._x_0) ** 2)).mean()


class PureFunctionEnv:
    def __init__(self, function, num_dim):
        self._function = function

        self.observation_space = Box(low=0.0, high=1.0, dtype=np.float32)

        self.action_space = Box(low=-np.ones(num_dim), high=np.ones(num_dim), dtype=np.float32)

    def step(self, action):
        # state, reward, done = env.step(action)[:3]
        return 1, self._function(action), True, None

    def reset(self, seed):
        return 0, None

    def close(self):
        pass
