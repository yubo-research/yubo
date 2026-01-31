import copy

import numpy as np

from problems.np_policy_util import set_params_pm1


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _softmin(x, k):
    x = np.asarray(x, dtype=np.float64)
    return -_logsumexp(-k * x) / k


class BipedalWalkerFeatPolicy:
    def __init__(self, env_conf):
        assert env_conf.env_name == "BipedalWalker-v3", env_conf.env_name
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._num_state = int(env_conf.gym_conf.state_space.shape[0])
        self._num_action = int(env_conf.action_space.shape[0])
        assert self._num_state == 24, self._num_state
        assert self._num_action == 4, self._num_action

        self._num_feat = 16
        self._num_w = self._num_action * self._num_feat
        self._num_p = 1 + self._num_w + self._num_action

        self._x_orig = np.zeros(self._num_p, dtype=np.float64)
        self._W = np.zeros((self._num_action, self._num_feat), dtype=np.float64)
        self._b = np.zeros(self._num_action, dtype=np.float64)
        self._phase = 0.0
        self._phase_inc = 0.02
        self._set_derived(self._x_orig)

    def num_params(self):
        return self._num_p

    def set_params(self, x):
        set_params_pm1(self, x)

    def get_params(self):
        return self._x_orig.copy()

    def clone(self):
        p = copy.deepcopy(self)
        p._phase = 0.0
        return p

    def _set_derived(self, x):
        i = 0
        self._phase_inc = 0.005 + (x[i] + 1.0) * 0.5 * (0.08 - 0.005)
        i += 1
        self._W = (2.0 * x[i : i + self._num_w]).reshape(self._W.shape)
        i += self._num_w
        self._b = 2.0 * x[i : i + self._num_action]

    def __call__(self, state):
        s = np.asarray(state, dtype=np.float64)
        assert s.shape == (24,), s.shape

        hull_angle = float(s[0]) / np.pi
        hull_ang_vel = float(s[1]) / 5.0
        vel_x = float(s[2]) / 5.0
        vel_y = float(s[3]) / 5.0

        hip_1 = float(s[4]) / np.pi
        hip_1_spd = float(s[5]) / 5.0
        knee_1 = float(s[6]) / np.pi
        knee_1_spd = float(s[7]) / 5.0
        c1 = float(s[8])

        hip_2 = float(s[9]) / np.pi
        hip_2_spd = float(s[10]) / 5.0
        knee_2 = float(s[11]) / np.pi
        knee_2_spd = float(s[12]) / 5.0
        c2 = float(s[13])

        lidar = np.asarray(s[14:24], dtype=np.float64)
        _ = _softmin(lidar[:5], 10.0)
        _ = _softmin(lidar[5:], 10.0)

        self._phase += self._phase_inc
        if self._phase >= 2.0 * np.pi:
            self._phase -= 2.0 * np.pi

        feat = np.array(
            [
                hull_angle,
                hull_ang_vel,
                vel_x,
                vel_y,
                hip_1,
                knee_1,
                hip_2,
                knee_2,
                hip_1_spd,
                knee_1_spd,
                hip_2_spd,
                knee_2_spd,
                c1,
                c2,
                np.sin(self._phase),
                np.cos(self._phase),
            ],
            dtype=np.float64,
        )
        z = self._W @ feat + self._b
        return np.tanh(z)
