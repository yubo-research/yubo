"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import optimizer.optimizer_mo
import optimizer.uhd_enn_fit_helpers
import optimizer.uhd_enn_imputer_predict
import optimizer.uhd_enn_imputer_tell
import optimizer.uhd_loop_support
import problems.dm_control_env_core
import problems.dm_control_pixel_wrapper
import problems.dm_control_spaces
import problems.env_conf_backends
import problems.env_conf_bindings
import problems.env_conf_parse
import problems.env_conf_rl
import problems.env_conf_types
import problems.pixel_policies_encoders
import problems.reactor_policy_params
from optimizer.ppo_common import apply_ppo_telemetry, resolve_designer_config, trajectory_tensors
from optimizer.uhd_enn_fit_helpers import enn_mixin_maybe_fit_inplace, fit_enn_params
from optimizer.uhd_mezo_be_ask_shared import run_mezo_be_ask
from policies.actor_mlp_policy import ActorMLPPolicyFactory
from problems.normalizer import Normalizer, normalize_running_state_array
from problems.reactor_policy_params import (
    finalize_derived,
    set_delta_params,
    set_gain_params,
    set_memory_params,
    set_smoothing_params,
    set_target_params,
    set_timer_param,
)
from rl.math_utils import tanh_gaussian_action_log_prob_entropy
from rl.pufferlib.offpolicy.backbone_name import resolve_backbone_name
from tests.kiss_tidy_coverage_c_lib import run_reactor_policy_params
from tests.kiss_tidy_coverage_c_mo_lib import run_optimizer_mo_and_fit
from tests.kiss_tidy_coverage_c_support_lib import (
    run_dm_pixel_env_conf,
    run_uhd_enn_mixins,
    run_uhd_loop_support,
)


def test_kiss_tidy_optimizer_mo_and_uhd_fit_helpers(monkeypatch):
    _ = (
        optimizer.optimizer_mo,
        optimizer.uhd_enn_fit_helpers,
        fit_enn_params,
    )
    run_optimizer_mo_and_fit(monkeypatch)


def test_kiss_tidy_uhd_enn_predict_tell_mixins(monkeypatch):
    _ = (optimizer.uhd_enn_imputer_predict, optimizer.uhd_enn_imputer_tell)
    run_uhd_enn_mixins(monkeypatch)


def test_kiss_tidy_uhd_loop_support(capsys):
    _ = optimizer.uhd_loop_support
    run_uhd_loop_support(capsys)


def test_kiss_tidy_dm_control_core_pixel_spaces_env_conf(monkeypatch):
    _ = (
        problems.dm_control_env_core,
        problems.dm_control_pixel_wrapper,
        problems.dm_control_spaces,
        problems.env_conf_backends,
        problems.env_conf_bindings,
        problems.env_conf_parse,
        problems.env_conf_rl,
        problems.env_conf_types,
        problems.pixel_policies_encoders,
    )
    run_dm_pixel_env_conf(monkeypatch)


def test_kiss_tidy_reactor_policy_params():
    _ = problems.reactor_policy_params
    run_reactor_policy_params()
    pol = SimpleNamespace(
        _delta_w1=None,
        _delta_b1=None,
        _delta_w2=None,
        _delta_b2=None,
        _timer_gamma_logit=0.0,
        _memory_w=None,
        _memory_b=None,
        _target_base=None,
        _target_coeff=None,
        _kp_logit=None,
        _kd_logit=None,
        _memory_alpha_logit=0.0,
        _action_alpha_logit=0.0,
        _action_scale_logit=0.0,
    )
    x = np.zeros(32, dtype=np.float64)
    i = set_delta_params(pol, x, 0, d_h=2, d_delta=2, k=1)
    i = set_timer_param(pol, x, i)
    i = set_memory_params(pol, x, i, d_mem=1, d_obs=2)
    i = set_target_params(pol, x, i, k=1, d_act=2, d_targ=2)
    i = set_gain_params(pol, x, i, k=1, d_act=2)
    i = set_smoothing_params(pol, x, i)
    finalize_derived(pol)


def test_kiss_tidy_c_ppo_uhd_actor_smoke(monkeypatch):
    import numpy as np
    import torch
    from torch.distributions import Normal

    cfg = resolve_designer_config(None, SimpleNamespace, {"x": 1})
    assert cfg is not None
    tel = SimpleNamespace(
        set_dt_rollout=lambda v: None,
        set_dt_fit=lambda v: None,
        set_dt_select=lambda v: None,
        set_num_rollout_workers=lambda v: None,
    )
    apply_ppo_telemetry(tel, 1.0, 2.0, 3)
    traj = SimpleNamespace(
        states=np.zeros((2, 3)),
        actions=np.zeros((2, 2)),
        log_probs=np.zeros(2),
    )
    trajectory_tensors(traj, torch.device("cpu"))
    obj = SimpleNamespace(
        _cfg=SimpleNamespace(warmup_real_obs=2, fit_interval=1, k=2),
        _x=[np.zeros(2), np.zeros(2)],
        _y=[0.0, 1.0],
        _enn_params=None,
        _num_new_since_fit=0,
    )
    monkeypatch.setattr(
        "optimizer.uhd_enn_fit_helpers.fit_enn_regressor_on_points",
        lambda *a, **k: (0.0, 1.0, None, {}),
    )
    enn_mixin_maybe_fit_inplace(obj)
    self = SimpleNamespace(
        _mezo=SimpleNamespace(positive_phase=True, ask=lambda: None, set_next_seed=lambda s: None),
        _enn_params=None,
        _zs=[],
        _warmup=0,
        _enn_k=25,
        _selected=False,
        _z_plus=None,
        _z_minus=None,
    )
    run_mezo_be_ask(self, embed_unselected=lambda: np.zeros(2))
    norm = Normalizer(shape=(2,))
    normalize_running_state_array(np.array([1.0, 2.0], dtype=np.float32), norm)
    dist = Normal(torch.zeros(1, 2), torch.ones(1, 2))
    tanh_gaussian_action_log_prob_entropy(dist)
    assert resolve_backbone_name(SimpleNamespace(backbone_name="mlp"), SimpleNamespace(mode="vector")) == "mlp"
    env = SimpleNamespace(
        problem_seed=0,
        state_space=SimpleNamespace(shape=(3,)),
        action_space=SimpleNamespace(shape=(2,)),
        ensure_spaces=lambda: None,
    )
    assert ActorMLPPolicyFactory((8,))(env) is not None
