import numpy as np

import common.all_bounds as all_bounds


class TurboLunarPolicy:
    def __init__(self, env_conf):
        self._env_conf = env_conf
        self._w_orig = self._w = np.zeros(12)
        self.problem_seed = self._env_conf.problem_seed

    def num_params(self):
        return 12

    def set_params(self, x):
        # w in [0,2]
        self._w_orig = x
        self._w = 2 * (x - all_bounds.x_low) / all_bounds.x_width

    def get_params(self):
        return self._w_orig

    def clone(self):
        tlp = TurboLunarPolicy(self._env_conf)
        tlp._w = self._w
        return tlp

    def __call__(self, state):
        assert self._w.min() >= 0, self._w
        assert self._w.max() <= 2, self._w
        angle_targ = state[0] * self._w[0] + state[2] * self._w[1]
        if angle_targ > self._w[2]:
            angle_targ = self._w[2]
        if angle_targ < -self._w[2]:
            angle_targ = -self._w[2]
        hover_targ = self._w[3] * np.abs(state[0])

        angle_todo = (angle_targ - state[4]) * self._w[4] - (state[5]) * self._w[5]
        hover_todo = (hover_targ - state[1]) * self._w[6] - (state[3]) * self._w[7]

        if state[6] or state[7]:
            angle_todo = self._w[8]
            hover_todo = -(state[3]) * self._w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > self._w[10]:
            a = 2
        elif angle_todo < -self._w[11]:
            a = 3
        elif angle_todo > +self._w[11]:
            a = 1
        return a

    # s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
