from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def run_reactor_policy_params():
    from problems import reactor_policy_params as rpp

    p = SimpleNamespace()
    p._action_dim = 2
    p._num_fsm_states = 1
    p._obs_dim = 2
    p._memory_dim = 1
    p._delta_hidden_dim = 1
    p._delta_feat_dim = 1
    p._target_feat_dim = 1
    rpp.init_indices(p, joint_angle_idx=None, joint_vel_idx=None, contact_idx=None, hazard_idx=None)
    rpp.init_indices(
        p,
        joint_angle_idx=np.array([0, 1], dtype=np.int64),
        joint_vel_idx=np.array([0, 1], dtype=np.int64),
        contact_idx=np.array([0, 1], dtype=np.int64),
        hazard_idx=np.array([0], dtype=np.int64),
    )
    q = SimpleNamespace(
        _num_fsm_states=1,
        _obs_dim=1,
        _memory_dim=1,
        _action_dim=1,
        _delta_hidden_dim=1,
        _delta_feat_dim=1,
        _target_feat_dim=1,
    )
    rpp.init_indices(
        q,
        joint_angle_idx=None,
        joint_vel_idx=None,
        contact_idx=np.array([0, 1], dtype=np.int64),
        hazard_idx=np.array([0], dtype=np.int64),
    )
    n = rpp.compute_num_params(q)
    q._num_params = n
    rpp.set_derived(q, np.zeros(n, dtype=np.float64))
    assert rpp is not None
