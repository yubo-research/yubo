import numpy as np


def _map_pm1(x, lo, hi):
    return lo + (x + 1.0) * 0.5 * (hi - lo)


class BipedalWalkerPolicy:
    def __init__(self, env_conf):
        assert env_conf.env_name == "BipedalWalker-v3", env_conf.env_name
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._num_state = int(env_conf.gym_conf.state_space.shape[0])
        self._num_action = int(env_conf.action_space.shape[0])
        assert self._num_state == 24, self._num_state
        assert self._num_action == 4, self._num_action

        self._num_p = 16
        self._x_orig = np.zeros((self._num_p,), dtype=np.float64)
        self._x_center = np.array(
            [
                -0.2,
                0.11111111111111116,
                -0.33333333333333337,
                -0.19999999999999996,
                -0.19999999999999996,
                0.11111111111111116,
                -0.125,
                -0.375,
                -0.23076923076923084,
                -0.5,
                -0.09999999999999998,
                -0.1428571428571429,
                0.0,
                -0.33333333333333337,
                -0.11111111111111116,
                -1.0,
            ],
            dtype=np.float64,
        )
        self._x_scale = 0.35

        self._STATE_STAY = 1
        self._STATE_PUT_DOWN = 2
        self._STATE_PUSH_OFF = 3

        self.reset_state()
        self._set_derived(self._x_orig)

    def num_params(self):
        return self._num_p

    def set_params(self, x):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self._num_p,), x.shape
        assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
        self._x_orig = x.copy()
        self._set_derived(self._x_orig)

    def get_params(self):
        return self._x_orig.copy()

    def clone(self):
        lp = BipedalWalkerPolicy(self._env_conf)
        lp._num_p = self._num_p
        lp._x_orig = self._x_orig.copy()
        lp._x_center = self._x_center.copy()
        lp._x_scale = float(self._x_scale)
        lp._set_derived(lp._x_orig)
        return lp

    def _set_derived(self, x):
        x = np.clip(self._x_center + self._x_scale * np.asarray(x, dtype=np.float64), -1.0, 1.0)

        self._speed = _map_pm1(x[0], 0.15, 0.50)
        self._hip_swing = _map_pm1(x[1], 0.6, 1.5)
        self._knee_swing = _map_pm1(x[2], -1.0, 0.2)
        self._hip_putdown = _map_pm1(x[3], -0.1, 0.4)
        self._knee_support = _map_pm1(x[4], -0.1, 0.4)
        self._knee_push = _map_pm1(x[5], 0.5, 1.4)

        self._hip_p = _map_pm1(x[6], 0.2, 1.8)
        self._hip_d = _map_pm1(x[7], 0.0, 0.8)
        self._knee_p = _map_pm1(x[8], 1.5, 8.0)
        self._knee_d = _map_pm1(x[9], 0.0, 1.0)

        self._torso_p = _map_pm1(x[10], 0.0, 2.0)
        self._torso_d = _map_pm1(x[11], 0.0, 3.5)
        self._vert_d = _map_pm1(x[12], 0.0, 30.0)

        self._act_scale = _map_pm1(x[13], 0.25, 1.0)
        self._swap_timeout = int(round(_map_pm1(x[14], 5.0, 50.0)))
        self._hazard_k = _map_pm1(x[15], 0.0, 1.0)

    def reset_state(self):
        self._state = self._STATE_STAY
        self._moving_leg = 0
        self._supporting_knee_angle = 0.0
        self._t_in_state = 0

    def _compute_targets(self, s, moving, supporting, moving_s, supporting_s, hazard, moving_contact):
        hip_targ = [None, None]
        knee_targ = [None, None]
        state_handlers = {
            self._STATE_STAY: lambda: self._handle_stay(s, moving, supporting, supporting_s, hazard, hip_targ, knee_targ),
            self._STATE_PUT_DOWN: lambda: self._handle_put_down(s, moving, supporting, moving_s, moving_contact, hip_targ, knee_targ),
            self._STATE_PUSH_OFF: lambda: self._handle_push_off(s, moving, supporting, supporting_s, knee_targ),
        }
        handler = state_handlers.get(self._state)
        if handler:
            handler()
        return hip_targ, knee_targ

    def _handle_stay(self, s, moving, supporting, supporting_s, hazard, hip_targ, knee_targ):
        hip_targ[moving] = self._hip_swing
        knee_targ[moving] = self._knee_swing - self._hazard_k * hazard
        self._supporting_knee_angle += 0.03 + (0.03 if s[2] > self._speed else 0)
        self._supporting_knee_angle = min(self._supporting_knee_angle, self._knee_support)
        knee_targ[supporting] = self._supporting_knee_angle
        if float(s[supporting_s]) < 0.10:
            self._state, self._t_in_state = self._STATE_PUT_DOWN, 0

    def _handle_put_down(self, s, moving, supporting, moving_s, moving_contact, hip_targ, knee_targ):
        hip_targ[moving] = self._hip_putdown
        knee_targ[moving] = self._knee_support
        knee_targ[supporting] = self._supporting_knee_angle
        if moving_contact:
            self._state, self._t_in_state = self._STATE_PUSH_OFF, 0
            self._supporting_knee_angle = min(float(s[moving_s + 2]), self._knee_support)

    def _handle_push_off(self, s, moving, supporting, supporting_s, knee_targ):
        knee_targ[moving] = self._supporting_knee_angle
        knee_targ[supporting] = self._knee_push
        if float(s[supporting_s + 2]) > 0.88 or s[2] > 1.2 * self._speed:
            self._state, self._t_in_state = self._STATE_STAY, 0
            self._moving_leg = 1 - self._moving_leg
            self._supporting_knee_angle = self._knee_support

    def _compute_control(self, s, hip_targ, knee_targ):
        hip_todo = np.zeros(2, dtype=np.float64)
        knee_todo = np.zeros(2, dtype=np.float64)
        joint_params = [
            (hip_targ[0], 4, 5, hip_todo, 0),
            (hip_targ[1], 9, 10, hip_todo, 1),
            (knee_targ[0], 6, 7, knee_todo, 0),
            (knee_targ[1], 11, 12, knee_todo, 1),
        ]
        for targ, pos_idx, vel_idx, arr, idx in joint_params:
            if targ is not None:
                p_gain = self._hip_p if arr is hip_todo else self._knee_p
                d_gain = self._hip_d if arr is hip_todo else self._knee_d
                arr[idx] = p_gain * (float(targ) - float(s[pos_idx])) - d_gain * float(s[vel_idx])
        hip_todo -= self._torso_p * (0.0 - float(s[0])) - self._torso_d * float(s[1])
        knee_todo -= self._vert_d * float(s[3])
        return hip_todo, knee_todo

    def __call__(self, state):
        s = np.asarray(state, dtype=np.float64)
        assert s.shape == (24,), s.shape
        forward_min = float(np.min(np.asarray(s[20:], dtype=np.float64)))
        hazard = float(np.clip((0.75 - forward_min) / 0.75, 0.0, 1.0))
        moving = int(self._moving_leg)
        supporting = 1 - moving
        moving_s, supporting_s = 4 + 5 * moving, 4 + 5 * supporting
        moving_contact = bool(s[moving_s + 4] > 0.5)
        supporting_contact = bool(s[supporting_s + 4] > 0.5)

        hip_targ, knee_targ = self._compute_targets(s, moving, supporting, moving_s, supporting_s, hazard, moving_contact)

        if self._t_in_state >= self._swap_timeout and not (moving_contact or supporting_contact):
            self._state, self._t_in_state = self._STATE_STAY, 0
            self._moving_leg = 1 - self._moving_leg
            self._supporting_knee_angle = self._knee_support

        hip_todo, knee_todo = self._compute_control(s, hip_targ, knee_targ)
        a = np.clip(
            self._act_scale * np.array([hip_todo[0], knee_todo[0], hip_todo[1], knee_todo[1]], dtype=np.float64),
            -1.0,
            1.0,
        )
        self._t_in_state += 1
        return a
