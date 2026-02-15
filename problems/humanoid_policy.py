import math

import numpy as np

from problems.np_policy_util import set_params_pm1


def _map_pm1(x, lo, hi):
    return lo + (x + 1.0) * 0.5 * (hi - lo)


class HumanoidPolicy:
    """Heuristic walking controller for Humanoid-v5.

    Uses a CPG for leg oscillation, PD control for joint tracking,
    torso stabilization, and root angular velocity damping.

    22 parameters:
    - 16 original CPG+PD params
    - 6 reactive feedback (yaw damping, z-reflex, velocity corrections, arm PD)
    """

    # --- observation indices ---
    _ABD_Z, _ABD_Y, _ABD_X = 5, 6, 7
    _RHX, _RHZ, _RHY, _RK = 8, 9, 10, 11
    _LHX, _LHZ, _LHY, _LK = 12, 13, 14, 15
    _RS1, _RS2, _RE = 16, 17, 18
    _LS1, _LS2, _LE = 19, 20, 21
    _VX, _VY, _VZ = 22, 23, 24
    _WX, _WY, _WZ = 25, 26, 27
    _ABD_ZV, _ABD_YV, _ABD_XV = 28, 29, 30
    _RHXV, _RHZV, _RHYV, _RKV = 31, 32, 33, 34
    _LHXV, _LHZV, _LHYV, _LKV = 35, 36, 37, 38
    _RS1V, _RS2V, _REV = 39, 40, 41
    _LS1V, _LS2V, _LEV = 42, 43, 44

    def __init__(self, env_conf):
        assert env_conf.env_name == "Humanoid-v5", env_conf.env_name
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._num_state = int(env_conf.gym_conf.state_space.shape[0])
        self._num_action = int(env_conf.action_space.shape[0])
        assert self._num_state == 348, self._num_state
        assert self._num_action == 17, self._num_action

        self._num_p = 22
        self._x_orig = np.zeros((self._num_p,), dtype=np.float64)
        # CMA-ES bootstrapped center (22-param, Stage 2 of 4-stage run)
        self._x_center = np.array(
            [
                -0.7189,  # 0: cpg_freq
                -1.0000,  # 1: hip_y_amp
                -0.4084,  # 2: hip_y_offset
                -0.8137,  # 3: knee_amp
                0.3720,  # 4: knee_offset
                0.7719,  # 5: knee_phase
                -0.3334,  # 6: hip_x_amp
                -0.5932,  # 7: hip_kp
                -1.0000,  # 8: hip_kd
                -0.6972,  # 9: knee_kp
                -0.7967,  # 10: knee_kd
                0.9843,  # 11: torso_kp
                -1.0000,  # 12: torso_kd
                -0.0254,  # 13: torso_lean
                -0.7133,  # 14: gyro_kd
                0.2651,  # 15: act_scale
                -1.0000,  # 16: yaw_kd
                -0.9654,  # 17: z_reflex_gain
                -0.2690,  # 18: vx_lean_gain
                -0.5907,  # 19: vy_hip_gain
                -0.7790,  # 20: arm_kp
                -0.7567,  # 21: arm_kd
            ],
            dtype=np.float64,
        )
        self._x_scale = 0.01

        self.reset_state()
        self._set_derived(self._x_orig)

    def num_params(self):
        return self._num_p

    def set_params(self, x):
        set_params_pm1(self, x)

    def get_params(self):
        return self._x_orig.copy()

    def clone(self):
        p = HumanoidPolicy(self._env_conf)
        p._num_p = self._num_p
        p._x_orig = self._x_orig.copy()
        p._x_center = self._x_center.copy()
        p._x_scale = float(self._x_scale)
        p._set_derived(p._x_orig)
        return p

    def _set_derived(self, x):
        x = np.clip(
            self._x_center + self._x_scale * np.asarray(x, dtype=np.float64),
            -1.0,
            1.0,
        )

        # --- original 16 params ---
        self._cpg_freq = _map_pm1(x[0], 0.02, 0.12)
        self._hip_y_amp = _map_pm1(x[1], 0.0, 0.5)
        self._hip_y_offset = _map_pm1(x[2], -0.3, 0.3)
        self._knee_amp = _map_pm1(x[3], 0.0, 0.5)
        self._knee_offset = _map_pm1(x[4], -0.5, 0.1)
        self._knee_phase = _map_pm1(x[5], -math.pi, math.pi)
        self._hip_x_amp = _map_pm1(x[6], 0.0, 0.3)
        self._hip_kp = _map_pm1(x[7], 0.5, 5.0)
        self._hip_kd = _map_pm1(x[8], 0.0, 1.0)
        self._knee_kp = _map_pm1(x[9], 0.5, 5.0)
        self._knee_kd = _map_pm1(x[10], 0.0, 1.0)
        self._torso_kp = _map_pm1(x[11], 0.5, 8.0)
        self._torso_kd = _map_pm1(x[12], 0.0, 2.0)
        self._torso_lean = _map_pm1(x[13], -0.2, 0.2)
        self._gyro_kd = _map_pm1(x[14], 0.0, 2.0)
        self._act_scale = _map_pm1(x[15], 0.3, 1.0)

        # --- reactive feedback (params 16-21) ---
        self._yaw_kd = _map_pm1(x[16], 0.0, 2.0)
        self._z_reflex_gain = _map_pm1(x[17], 0.0, 5.0)
        self._vx_lean_gain = _map_pm1(x[18], 0.0, 0.5)
        self._vy_hip_gain = _map_pm1(x[19], 0.0, 1.0)
        self._arm_kp = _map_pm1(x[20], 0.1, 1.5)
        self._arm_kd = _map_pm1(x[21], 0.01, 0.3)

    def reset_state(self):
        self._phi = 0.0

    def __call__(self, state):
        s = np.asarray(state, dtype=np.float64)
        assert s.shape == (348,), s.shape
        a = np.zeros(17, dtype=np.float64)

        # --- gyro damping (pitch, roll, yaw) ---
        gp = -self._gyro_kd * s[self._WY]
        gr = -self._gyro_kd * s[self._WX]
        gy = -self._yaw_kd * s[self._WZ]

        # --- z-height reflex: extend knees when z drops ---
        z_err = max(0.0, 1.25 - s[0])
        z_ext = -self._z_reflex_gain * z_err

        # --- velocity-based lean correction ---
        lean_corr = -self._vx_lean_gain * s[self._VX]

        # --- lateral hip correction ---
        lat_corr = -self._vy_hip_gain * s[self._VY]

        # --- CPG ---
        self._phi = (self._phi + self._cpg_freq) % (2.0 * math.pi)
        rp = self._phi
        lp = self._phi + math.pi

        # --- leg targets ---
        r_hy = self._hip_y_offset + self._hip_y_amp * math.sin(rp)
        l_hy = self._hip_y_offset + self._hip_y_amp * math.sin(lp)
        r_k = self._knee_offset + self._knee_amp * math.sin(rp + self._knee_phase)
        l_k = self._knee_offset + self._knee_amp * math.sin(lp + self._knee_phase)
        r_hx = self._hip_x_amp * math.sin(rp)
        l_hx = self._hip_x_amp * math.sin(lp)

        hkp, hkd = self._hip_kp, self._hip_kd
        kkp, kkd = self._knee_kp, self._knee_kd

        # --- legs: hip y (sagittal) ---
        a[5] = hkp * (r_hy - s[self._RHY]) - hkd * s[self._RHYV]
        a[9] = hkp * (l_hy - s[self._LHY]) - hkd * s[self._LHYV]
        # --- legs: knee (with z-reflex) ---
        a[6] = kkp * (r_k + z_ext - s[self._RK]) - kkd * s[self._RKV]
        a[10] = kkp * (l_k + z_ext - s[self._LK]) - kkd * s[self._LKV]
        # --- legs: hip x (lateral, with velocity correction) ---
        a[3] = hkp * (r_hx - s[self._RHX]) - hkd * s[self._RHXV] + lat_corr
        a[7] = hkp * (l_hx - s[self._LHX]) - hkd * s[self._LHXV] + lat_corr
        # --- legs: hip z (transverse, with yaw damping) ---
        a[4] = -hkp * s[self._RHZ] - hkd * s[self._RHZV] + gy
        a[8] = -hkp * s[self._LHZ] - hkd * s[self._LHZV] + gy

        # --- torso (abdomen PD + gyro + lean correction) ---
        tkp, tkd = self._torso_kp, self._torso_kd
        a[0] = tkp * (self._torso_lean + lean_corr - s[self._ABD_Y]) - tkd * s[self._ABD_YV] + gp
        a[1] = -tkp * s[self._ABD_Z] - tkd * s[self._ABD_ZV] + gy
        a[2] = -tkp * s[self._ABD_X] - tkd * s[self._ABD_XV] + gr

        # --- arms (tunable PD) ---
        for ai, pi, vi in [
            (11, self._RS1, self._RS1V),
            (12, self._RS2, self._RS2V),
            (13, self._RE, self._REV),
            (14, self._LS1, self._LS1V),
            (15, self._LS2, self._LS2V),
            (16, self._LE, self._LEV),
        ]:
            a[ai] = -self._arm_kp * s[pi] - self._arm_kd * s[vi]

        a *= self._act_scale
        return np.clip(a, -1.0, 1.0)
