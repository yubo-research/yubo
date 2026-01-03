import numpy as np

import common.all_bounds as all_bounds
from problems.benchmark_functions import all_benchmarks
from problems.turbo_ackley import TurboAckley


def _all_pure_functions():
    all_bf = all_benchmarks()
    all_bf["tackley"] = TurboAckley
    from problems.double_ackley import DoubleAckley

    all_bf["doubleackley"] = DoubleAckley
    return all_bf


def make(name, problem_seed, distort):
    _, name = name.split(":")
    name, num_dim = name.split("-")
    assert num_dim[-1] == "d"
    num_dim = int(num_dim[:-1])

    all_bf = _all_pure_functions()
    if name in all_bf:
        return PureFunctionEnv(all_bf[name](), num_dim, problem_seed, distort=distort)
    assert False, name


# all domains are [-1,1]**num_dim
class PureFunctionEnv:
    def __init__(self, function, num_dim, problem_seed, *, distort):
        # print("PROBLEM_SEED:", problem_seed)
        self._function = function

        # state is either 0 or 1
        # state==0 means that the policy has not been called.
        # state==1 means that it has.
        self.observation_space = all_bounds.get_box_1d01()
        # action is the input to the function (before some transformations)
        self.action_space = all_bounds.get_box_bounds_x(num_dim)

        if distort:
            rng = np.random.default_rng(problem_seed)

            # Distort the parameter space, moving the center
            #  to a randomly-chosen corner of the bounding box.
            self._x_0 = all_bounds.x_low + all_bounds.x_width * rng.uniform(size=(num_dim,))
        else:
            self._x_0 = all_bounds.x_low + (all_bounds.x_width / 2) * np.ones(shape=(num_dim,))

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

    def reward_batch(self, actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2:
            raise ValueError(actions.shape)
        if actions.shape[1] != self.action_space.shape[0]:
            raise ValueError((actions.shape, self.action_space.shape))
        if not (np.all(actions >= -1) and np.all(actions <= 1)):
            raise ValueError("actions out of bounds")

        x0 = np.asarray(self._x_0, dtype=float).reshape(-1)
        x_raw = actions - x0
        neg = x_raw < 0
        pos = x_raw > 0
        denom_neg = 1.0 + x0
        denom_pos = 1.0 - x0
        x = x_raw
        if np.any(neg):
            x = np.where(neg, x / denom_neg, x)
        if np.any(pos):
            x = np.where(pos, x / denom_pos, x)
        if not (np.all(x >= -1) and np.all(x <= 1)):
            raise ValueError("transformed x out of bounds")

        try:
            y = self._function(x)
            y = np.asarray(y, dtype=float)
            if y.shape == (actions.shape[0],):
                return -y
        except Exception:
            pass

        out = np.empty((actions.shape[0],), dtype=float)
        for i in range(actions.shape[0]):
            out[i] = float(-self._function(x[i]))
        return out

    def reset(self, seed):
        if hasattr(self._function, "reset"):
            self._function.reset(seed)
        return 0, None

    def close(self):
        pass
