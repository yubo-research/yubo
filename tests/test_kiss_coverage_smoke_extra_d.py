from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from optimizer import opt_trajectories as opt_trajectories_mod
from optimizer.opt_trajectories import (
    collect_denoised_trajectory,
    collect_trajectory_with_noise,
    evaluate_for_best,
    mean_return_over_runs,
)
from rl.core.sac_update import (
    SACUpdateBatch,
    SACUpdateHyperParams,
    SACUpdateModules,
    SACUpdateOptimizers,
    sac_update_step,
)


def test_kiss_cov_sac_update_and_opt_trajectories(monkeypatch):
    def _policy_sample(self, obs, deterministic=False):
        _ = deterministic
        act = torch.tanh(obs[..., :1])
        lp = torch.zeros(obs.shape[0], dtype=obs.dtype)
        return (act, lp)

    _Policy = type("_Policy", (), {"sample": _policy_sample})

    q1 = nn.Linear(2, 1)
    q2 = nn.Linear(2, 1)
    q1_t = nn.Linear(2, 1)
    q2_t = nn.Linear(2, 1)

    def _qwrap_init(self, base):
        nn.Module.__init__(self)
        self.base = base

    def _qwrap_forward(self, obs, act):
        return self.base(torch.cat([obs, act], dim=-1)).squeeze(-1)

    _QWrap = type("_QWrap", (nn.Module,), {"__init__": _qwrap_init, "forward": _qwrap_forward})

    actor = _Policy()
    modules = SACUpdateModules(
        actor=actor,
        q1=_QWrap(q1),
        q2=_QWrap(q2),
        q1_target=_QWrap(q1_t),
        q2_target=_QWrap(q2_t),
        log_alpha=nn.Parameter(torch.tensor(0.0)),
    )
    opts = SACUpdateOptimizers(
        actor=torch.optim.AdamW([modules.log_alpha], lr=1e-3),
        critic=torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=1e-3),
        alpha=torch.optim.AdamW([modules.log_alpha], lr=1e-3),
    )
    batch = SACUpdateBatch(
        obs=torch.zeros((4, 1)),
        act=torch.zeros((4, 1)),
        rew=torch.zeros(4),
        nxt=torch.zeros((4, 1)),
        done=torch.zeros(4),
    )
    hyper = SACUpdateHyperParams(gamma=0.99, tau=0.01, target_entropy=-1.0)
    a, c, al = sac_update_step(modules=modules, optimizers=opts, batch=batch, hyper=hyper)
    assert np.isfinite(a)
    assert np.isfinite(c)
    assert np.isfinite(al)

    monkeypatch.setattr(
        opt_trajectories_mod,
        "collect_trajectory",
        lambda env_conf, policy, noise_seed=0: opt_trajectories_mod.Trajectory(float(noise_seed), None, None, None),
    )
    conf = SimpleNamespace(noise_seed_0=10, frozen_noise=False)
    traj, seed = collect_trajectory_with_noise(conf, object(), i_noise=1, denoise_seed=2)
    assert traj.rreturn == 13.0
    assert seed == 13
    mean, se, all_same, num_steps_total = mean_return_over_runs(conf, object(), num_denoise=2, i_noise=1)
    assert np.isfinite(mean)
    assert np.isfinite(se)
    assert all_same is False
    assert num_steps_total >= 0
    den, _ = collect_denoised_trajectory(conf, object(), num_denoise=2, i_noise=1)
    assert den.rreturn is not None
    best = evaluate_for_best(conf, object(), 2)
    assert np.isfinite(best)
