import numpy as np


def _init_indices(policy, *, joint_angle_idx, joint_vel_idx, contact_idx, hazard_idx):
    if joint_angle_idx is None:
        policy._joint_angle_idx = np.arange(policy._action_dim, dtype=np.int64)
    else:
        policy._joint_angle_idx = np.asarray(joint_angle_idx, dtype=np.int64)
    if joint_vel_idx is None:
        policy._joint_vel_idx = np.arange(policy._action_dim, dtype=np.int64)
    else:
        policy._joint_vel_idx = np.asarray(joint_vel_idx, dtype=np.int64)

    assert policy._joint_angle_idx.ndim == 1
    assert policy._joint_vel_idx.ndim == 1
    assert policy._joint_angle_idx.shape == (policy._action_dim,), policy._joint_angle_idx.shape
    assert policy._joint_vel_idx.shape == (policy._action_dim,), policy._joint_vel_idx.shape

    if contact_idx is None:
        policy._contact_idx = None
    else:
        policy._contact_idx = np.asarray(contact_idx, dtype=np.int64)
        assert policy._contact_idx.shape == (2,), policy._contact_idx.shape

    if hazard_idx is None:
        policy._hazard_idx = None
    else:
        policy._hazard_idx = np.asarray(hazard_idx, dtype=np.int64)
        assert policy._hazard_idx.ndim == 1
        assert policy._hazard_idx.size >= 1


def _compute_num_params(policy):
    k = policy._num_fsm_states
    d_obs = policy._obs_dim
    d_mem = policy._memory_dim
    d_act = policy._action_dim
    d_h = policy._delta_hidden_dim
    d_delta = policy._delta_feat_dim
    d_targ = policy._target_feat_dim
    return int(
        (d_h * d_delta)
        + d_h
        + (k * d_h)
        + k
        + 1
        + (max(0, d_mem - 1) * d_obs)
        + max(0, d_mem - 1)
        + (k * d_act)
        + (k * d_act * d_targ)
        + (k * d_act)
        + (k * d_act)
        + 3
    )


def _set_delta_params(policy, x, i, *, d_h, d_delta, k):
    policy._delta_w1 = x[i : i + d_h * d_delta].reshape(d_h, d_delta)
    i += d_h * d_delta
    policy._delta_b1 = x[i : i + d_h].reshape(d_h)
    i += d_h
    policy._delta_w2 = x[i : i + k * d_h].reshape(k, d_h)
    i += k * d_h
    policy._delta_b2 = x[i : i + k].reshape(k)
    i += k
    return i


def _set_timer_param(policy, x, i):
    policy._timer_gamma_logit = float(x[i])
    return i + 1


def _set_memory_params(policy, x, i, *, d_mem, d_obs):
    if d_mem > 1:
        r = d_mem - 1
        policy._memory_w = x[i : i + r * d_obs].reshape(r, d_obs)
        i += r * d_obs
        policy._memory_b = x[i : i + r].reshape(r)
        i += r
        return i
    policy._memory_w = None
    policy._memory_b = None
    return i


def _set_target_params(policy, x, i, *, k, d_act, d_targ):
    policy._target_base = x[i : i + k * d_act].reshape(k, d_act)
    i += k * d_act
    policy._target_coeff = x[i : i + k * d_act * d_targ].reshape(k, d_act, d_targ)
    i += k * d_act * d_targ
    return i


def _set_gain_params(policy, x, i, *, k, d_act):
    policy._kp_logit = x[i : i + k * d_act].reshape(k, d_act)
    i += k * d_act
    policy._kd_logit = x[i : i + k * d_act].reshape(k, d_act)
    i += k * d_act
    return i


def _set_smoothing_params(policy, x, i):
    policy._memory_alpha_logit = float(x[i])
    i += 1
    policy._action_alpha_logit = float(x[i])
    i += 1
    policy._action_scale_logit = float(x[i])
    i += 1
    return i


def _finalize_derived(policy):
    policy._timer_gamma = 0.5 + 0.49 * (1.0 / (1.0 + np.exp(-policy._timer_gamma_logit)))
    policy._memory_alpha = 1.0 / (1.0 + np.exp(-policy._memory_alpha_logit))
    policy._action_alpha = 1.0 / (1.0 + np.exp(-policy._action_alpha_logit))
    policy._action_scale = 0.25 + 0.75 * (1.0 / (1.0 + np.exp(-policy._action_scale_logit)))
    policy._kp = 0.1 + 6.0 * (1.0 / (1.0 + np.exp(-policy._kp_logit)))
    policy._kd = 0.0 + 2.0 * (1.0 / (1.0 + np.exp(-policy._kd_logit)))


class ReactorPolicy:
    def __init__(
        self,
        env_conf,
        *,
        num_modes=3,
        memory_dim=6,
        delta_hidden=8,
        joint_angle_idx=None,
        joint_vel_idx=None,
        contact_idx=None,
        hazard_idx=None,
        vx_idx=None,
        return_metrics=False,
    ):
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._obs_dim = int(env_conf.gym_conf.state_space.shape[0])
        self._action_dim = int(env_conf.action_space.shape[0])
        self._num_fsm_states = int(num_modes)
        self._memory_dim = int(memory_dim)
        self._delta_hidden_dim = int(delta_hidden)
        assert self._num_fsm_states >= 1
        assert self._memory_dim >= 1
        assert self._delta_hidden_dim >= 1

        self._feat_dim = self._obs_dim + self._memory_dim
        _init_indices(
            self,
            joint_angle_idx=joint_angle_idx,
            joint_vel_idx=joint_vel_idx,
            contact_idx=contact_idx,
            hazard_idx=hazard_idx,
        )

        self._vx_idx = None if vx_idx is None else int(vx_idx)
        self._return_metrics = bool(return_metrics)
        self._delta_feat_dim = 6
        self._target_feat_dim = 4
        self._num_params = _compute_num_params(self)

        self._x = np.zeros((self._num_params,), dtype=np.float64)
        self.reset_state()
        self._set_derived(self._x)

    def num_params(self):
        return self._num_params

    def get_params(self):
        return self._x.copy()

    def set_params(self, x):
        x = np.asarray(x, dtype=np.float64)
        assert x.shape == (self._num_params,), x.shape
        assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
        self._x = x.copy()
        self._set_derived(self._x)

    def clone(self):
        p = ReactorPolicy(
            self._env_conf,
            num_modes=self._num_fsm_states,
            memory_dim=self._memory_dim,
            delta_hidden=self._delta_hidden_dim,
            joint_angle_idx=self._joint_angle_idx,
            joint_vel_idx=self._joint_vel_idx,
            contact_idx=self._contact_idx,
            hazard_idx=self._hazard_idx,
            vx_idx=self._vx_idx,
            return_metrics=self._return_metrics,
        )
        p._x = self._x.copy()
        p._set_derived(p._x)
        return p

    def wants_vector_return(self) -> bool:
        return bool(self._return_metrics)

    def reset_state(self):
        self._fsm_state = 0
        self._m_state = np.zeros((self._memory_dim,), dtype=np.float64)
        self._prev_action = np.zeros((self._action_dim,), dtype=np.float64)
        self._metrics_steps = 0
        self._metrics_switches = 0
        self._metrics_state_counts = np.zeros((self._num_fsm_states,), dtype=np.float64)
        self._metrics_sat = 0.0
        self._metrics_abs_action = 0.0
        self._metrics_abs_daction = 0.0
        self._metrics_track = 0.0
        self._metrics_dtarget = 0.0
        self._metrics_mem_norm = 0.0
        self._metrics_prev_target = None

    def metrics(self):
        steps = int(self._metrics_steps)
        if steps <= 0:
            return np.zeros((8,), dtype=np.float64)

        switch_rate = float(self._metrics_switches) / float(steps)

        p = self._metrics_state_counts / max(1.0, float(steps))
        p = np.maximum(p, 1e-12)
        ent = -float(np.sum(p * np.log(p)))
        ent_norm = ent / float(np.log(max(2, self._num_fsm_states)))
        state_collapse = 1.0 - ent_norm

        sat_frac = float(self._metrics_sat) / float(steps)
        abs_action = float(self._metrics_abs_action) / float(steps)
        abs_daction = float(self._metrics_abs_daction) / float(steps)
        track = float(self._metrics_track) / float(steps)
        dtarget = float(self._metrics_dtarget) / float(steps)
        mem_norm = float(self._metrics_mem_norm) / float(steps)

        return -np.asarray(
            [
                switch_rate,
                state_collapse,
                sat_frac,
                abs_action,
                abs_daction,
                track,
                dtarget,
                mem_norm,
            ],
            dtype=np.float64,
        )

    def _hazard(self, o):
        if self._hazard_idx is None:
            return 0.0
        vals = np.asarray(o[self._hazard_idx], dtype=np.float64)
        return float(np.min(vals))

    def _vx(self, o):
        if self._vx_idx is None:
            return 0.0
        return float(o[self._vx_idx])

    def _contacts(self, o):
        if self._contact_idx is None:
            return 0.0, 0.0
        c0 = float(o[int(self._contact_idx[0])])
        c1 = float(o[int(self._contact_idx[1])])
        return float(c0 > 0.5), float(c1 > 0.5)

    def _delta_features(self, o, t_in_state):
        vx = self._vx(o)
        c0, c1 = self._contacts(o)
        hz = self._hazard(o)
        return np.array([1.0, vx, c0, c1, hz, float(t_in_state)], dtype=np.float64)

    def _target_features(self, o, t_in_state):
        vx = self._vx(o)
        hz = self._hazard(o)
        return np.array([1.0, vx, hz, float(t_in_state)], dtype=np.float64)

    def _set_derived(self, x):
        k = self._num_fsm_states
        d_obs = self._obs_dim
        d_mem = self._memory_dim
        d_act = self._action_dim
        d_h = self._delta_hidden_dim
        d_delta = self._delta_feat_dim
        d_targ = self._target_feat_dim
        i = 0
        i = _set_delta_params(self, x, i, d_h=d_h, d_delta=d_delta, k=k)
        i = _set_timer_param(self, x, i)
        i = _set_memory_params(self, x, i, d_mem=d_mem, d_obs=d_obs)
        i = _set_target_params(self, x, i, k=k, d_act=d_act, d_targ=d_targ)
        i = _set_gain_params(self, x, i, k=k, d_act=d_act)
        i = _set_smoothing_params(self, x, i)
        assert i == self._num_params
        _finalize_derived(self)

    def __call__(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        assert o.shape == (self._obs_dim,), o.shape

        t_in_state = self._m_state[0] if self._memory_dim >= 1 else 0.0
        delta_feat = self._delta_features(o, t_in_state)

        delta_h = np.tanh(self._delta_w1 @ delta_feat + self._delta_b1)
        delta_logits = self._delta_w2 @ delta_h + self._delta_b2
        prev_state = int(self._fsm_state)
        next_state = int(np.argmax(delta_logits))
        self._fsm_state = next_state
        if next_state != prev_state:
            self._metrics_switches += 1
        self._metrics_state_counts[next_state] += 1.0

        if self._memory_dim >= 1:
            if next_state != prev_state:
                self._m_state[0] = 0.0
            self._m_state[0] = self._timer_gamma * self._m_state[0] + 1.0

        if self._memory_dim > 1:
            assert self._memory_w is not None
            memory_incr = self._memory_w @ o + self._memory_b
            self._m_state[1:] = (1.0 - self._memory_alpha) * self._m_state[1:] + self._memory_alpha * memory_incr

        t_in_state = self._m_state[0] if self._memory_dim >= 1 else 0.0
        targ_feat = self._target_features(o, t_in_state)
        target_angles = self._target_base[self._fsm_state] + self._target_coeff[self._fsm_state] @ targ_feat

        joint_angles = o[self._joint_angle_idx]
        joint_vels = o[self._joint_vel_idx]
        torque_cmd = self._kp[self._fsm_state] * (target_angles - joint_angles) - self._kd[self._fsm_state] * joint_vels
        torque_cmd = np.tanh(self._action_scale * torque_cmd)

        action = (1.0 - self._action_alpha) * self._prev_action + self._action_alpha * torque_cmd
        action = np.clip(action, -1.0, 1.0)

        self._metrics_steps += 1
        self._metrics_sat += float(np.mean(np.abs(action) > 0.95))
        self._metrics_abs_action += float(np.mean(np.abs(action)))
        self._metrics_abs_daction += float(np.mean(np.abs(action - self._prev_action)) * 0.5)
        self._metrics_track += float(np.mean(np.tanh(np.abs(target_angles - joint_angles))))
        if self._metrics_prev_target is not None:
            self._metrics_dtarget += float(np.mean(np.tanh(np.abs(target_angles - self._metrics_prev_target))))
        self._metrics_prev_target = target_angles.copy()
        self._metrics_mem_norm += float(np.tanh(np.linalg.norm(self._m_state) / max(1.0, float(self._memory_dim))))

        self._prev_action = action
        return action


class ReactorPolicyFactory:
    def __init__(
        self,
        *,
        num_modes=3,
        memory_dim=6,
        delta_hidden=8,
        joint_angle_idx=None,
        joint_vel_idx=None,
        contact_idx=None,
        hazard_idx=None,
        vx_idx=None,
        return_metrics=False,
    ):
        self._num_modes = int(num_modes)
        self._memory_dim = int(memory_dim)
        self._delta_hidden = int(delta_hidden)
        self._joint_angle_idx = joint_angle_idx
        self._joint_vel_idx = joint_vel_idx
        self._contact_idx = contact_idx
        self._hazard_idx = hazard_idx
        self._vx_idx = vx_idx
        self._return_metrics = bool(return_metrics)

    def __call__(self, env_conf):
        return ReactorPolicy(
            env_conf,
            num_modes=self._num_modes,
            memory_dim=self._memory_dim,
            delta_hidden=self._delta_hidden,
            joint_angle_idx=self._joint_angle_idx,
            joint_vel_idx=self._joint_vel_idx,
            contact_idx=self._contact_idx,
            hazard_idx=self._hazard_idx,
            vx_idx=self._vx_idx,
            return_metrics=self._return_metrics,
        )
