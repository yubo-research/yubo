import numpy as np

import common.all_bounds as all_bounds


class PureFunctionPolicy:
    def __init__(self, env_conf):
        self._env_conf = env_conf
        if env_conf.action_space is not None:
            all_bounds = env_conf.action_space
            self._params = np.random.uniform(all_bounds.low, all_bounds.high, size=(len(all_bounds.low),))
        else:
            # Default to uniform distribution in the range [0, 1]
            self._params = np.random.uniform(0, 1, size=(env_conf.dim if hasattr(env_conf, 'dim') else 200,))


    def num_params(self):
        # Ensure _params is always initialized
        if self._params is None:
            raise ValueError("Policy parameters are not initialized.")
        return len(self._params)

    def set_params(self, x):
        self._params = x.copy()

    def get_params(self):
        return self._params

    def clone(self):
        cloned_policy = PureFunctionPolicy(self._env_conf)
        cloned_policy._params = np.copy(self._params) if self._params is not None else None
        return cloned_policy

    def __call__(self, state):
        print(f"DEBUG: state = {state}")
        if not np.all((state >= 0) & (state <= 1)):
            print(f"DEBUG: State is out of range: {state}")
        return self._params

