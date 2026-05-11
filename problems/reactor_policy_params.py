import numpy as np


def init_indices(policy, *, joint_angle_idx, joint_vel_idx, contact_idx, hazard_idx):
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


def compute_num_params(policy):
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


def set_delta_params(policy, x, i, *, d_h, d_delta, k):
    policy._delta_w1 = x[i : i + d_h * d_delta].reshape(d_h, d_delta)
    i += d_h * d_delta
    policy._delta_b1 = x[i : i + d_h].reshape(d_h)
    i += d_h
    policy._delta_w2 = x[i : i + k * d_h].reshape(k, d_h)
    i += k * d_h
    policy._delta_b2 = x[i : i + k].reshape(k)
    i += k
    return i


def set_timer_param(policy, x, i):
    policy._timer_gamma_logit = float(x[i])
    return i + 1


def set_memory_params(policy, x, i, *, d_mem, d_obs):
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


def set_target_params(policy, x, i, *, k, d_act, d_targ):
    policy._target_base = x[i : i + k * d_act].reshape(k, d_act)
    i += k * d_act
    policy._target_coeff = x[i : i + k * d_act * d_targ].reshape(k, d_act, d_targ)
    i += k * d_act * d_targ
    return i


def set_gain_params(policy, x, i, *, k, d_act):
    policy._kp_logit = x[i : i + k * d_act].reshape(k, d_act)
    i += k * d_act
    policy._kd_logit = x[i : i + k * d_act].reshape(k, d_act)
    i += k * d_act
    return i


def set_smoothing_params(policy, x, i):
    policy._memory_alpha_logit = float(x[i])
    i += 1
    policy._action_alpha_logit = float(x[i])
    i += 1
    policy._action_scale_logit = float(x[i])
    i += 1
    return i


def finalize_derived(policy):
    policy._timer_gamma = 0.5 + 0.49 * (1.0 / (1.0 + np.exp(-policy._timer_gamma_logit)))
    policy._memory_alpha = 1.0 / (1.0 + np.exp(-policy._memory_alpha_logit))
    policy._action_alpha = 1.0 / (1.0 + np.exp(-policy._action_alpha_logit))
    policy._action_scale = 0.25 + 0.75 * (1.0 / (1.0 + np.exp(-policy._action_scale_logit)))
    policy._kp = 0.1 + 6.0 * (1.0 / (1.0 + np.exp(-policy._kp_logit)))
    policy._kd = 0.0 + 2.0 * (1.0 / (1.0 + np.exp(-policy._kd_logit)))


def set_derived(policy, x):
    k = policy._num_fsm_states
    d_obs = policy._obs_dim
    d_mem = policy._memory_dim
    d_act = policy._action_dim
    d_h = policy._delta_hidden_dim
    d_delta = policy._delta_feat_dim
    d_targ = policy._target_feat_dim
    i = 0
    i = set_delta_params(policy, x, i, d_h=d_h, d_delta=d_delta, k=k)
    i = set_timer_param(policy, x, i)
    i = set_memory_params(policy, x, i, d_mem=d_mem, d_obs=d_obs)
    i = set_target_params(policy, x, i, k=k, d_act=d_act, d_targ=d_targ)
    i = set_gain_params(policy, x, i, k=k, d_act=d_act)
    i = set_smoothing_params(policy, x, i)
    assert i == policy._num_params
    finalize_derived(policy)
