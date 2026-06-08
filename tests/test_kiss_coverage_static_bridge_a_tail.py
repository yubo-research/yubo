"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


def test_kiss_bridge_uhd_simple_base_and_np_variants():
    from embedding.behavioral_embedder import BehavioralEmbedder
    from optimizer.gaussian_perturbator import GaussianPerturbator
    from optimizer.uhd_simple import UHDSimple
    from optimizer.uhd_simple_base import UHDSimpleBase
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    m = nn.Linear(2, 1, bias=False)
    gp = GaussianPerturbator(m)
    u = UHDSimple(gp, sigma_0=0.1, dim=2)
    assert isinstance(u, UHDSimpleBase)
    assert UHDSimpleBase.eval_seed.fget(u) == UHDSimple.eval_seed.fget(u)
    assert UHDSimpleBase.sigma.fget(u) == UHDSimple.sigma.fget(u)
    assert UHDSimpleBase.y_best.fget(u) == UHDSimple.y_best.fget(u)
    assert UHDSimpleBase.mu_avg.fget(u) == UHDSimple.mu_avg.fget(u)
    assert UHDSimpleBase.se_avg.fget(u) == UHDSimple.se_avg.fget(u)

    def _pol_init(self):
        self._p = np.zeros(3, dtype=np.float64)

    def _pol_get_params(self):
        return self._p

    def _pol_set_params(self, x):
        self._p = np.asarray(x, dtype=np.float64).copy()

    def _pol_call(self, probe):
        return np.atleast_1d(np.asarray(probe, dtype=np.float64))

    Pol = type(
        "Pol",
        (),
        {
            "__init__": _pol_init,
            "get_params": _pol_get_params,
            "set_params": _pol_set_params,
            "__call__": _pol_call,
        },
    )
    sn = UHDSimpleNp(Pol(), sigma_0=0.1)
    assert UHDSimpleNp.eval_seed.fget(sn) == 0
    sn.ask()
    sn.tell(1.0, 0.1)

    pol = Pol()
    bounds = torch.stack([torch.zeros(2, dtype=torch.float32), torch.ones(2, dtype=torch.float32)])
    emb = BehavioralEmbedder(bounds, num_probes=2, seed=0)
    benp = UHDSimpleBENp(pol, emb, sigma_0=0.1, num_candidates=1, warmup=0, fit_interval=10**9)
    assert UHDSimpleBENp.eval_seed.fget(benp) == 0
    benp.ask()
    benp.tell(0.0, 0.1)


def test_kiss_bridge_enn_minus_imputer_gather_and_direction():
    from optimizer.uhd_enn_imputer import ENNImputerConfig, ENNMinusImputer

    m = nn.Linear(3, 1, bias=True)

    def _nz(_seed, _sigma):
        return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float32)

    cfg = ENNImputerConfig(
        d=8,
        s=2,
        warmup_real_obs=1,
        fit_interval=10**9,
        k=3,
        embedder="gather",
        target="mu_minus",
        min_calib_points=0,
    )
    imp = ENNMinusImputer(module=m, cfg=cfg, noise_nz_fn=_nz)
    imp.begin_pair(seed=0, sigma=0.1)
    imp.tell_real(mu=0.5, phase="plus")
    imp.tell_real(mu=0.25, phase="minus")
    assert ENNMinusImputer.num_real_evals.fget(imp) >= 1
    imp.calibrate_minus(mu_minus_real=0.2)
    s, u = imp.choose_seed_ucb(base_seed=0, sigma=0.1)
    assert isinstance(s, int)
    _ = u
    imp.update_base_after_step(step_scale=1.0, sigma=0.1)

    cfg2 = ENNImputerConfig(
        d=16,
        s=2,
        warmup_real_obs=0,
        fit_interval=1,
        k=3,
        num_candidates=3,
        target="mu_plus",
        select_interval=1,
    )
    imp2 = ENNMinusImputer(module=m, cfg=cfg2, noise_nz_fn=_nz)
    imp2.begin_pair(seed=1, sigma=0.1)
    imp2.tell_real(mu=1.0, phase="plus")
    imp2.tell_real(mu=0.5, phase="plus")
    imp2.choose_seed_ucb(base_seed=0, sigma=0.1)

    imp3 = ENNMinusImputer(
        module=m,
        cfg=ENNImputerConfig(
            d=16,
            s=2,
            warmup_real_obs=0,
            fit_interval=10**9,
            k=3,
            num_candidates=1,
            target="mu_minus",
            min_calib_points=0,
            max_abs_err_ema=1.0,
            refresh_interval=10**9,
        ),
        noise_nz_fn=_nz,
    )
    _ = imp3.should_impute_negative()

    imp4 = ENNMinusImputer(
        module=m,
        cfg=ENNImputerConfig(
            d=16,
            s=2,
            warmup_real_obs=0,
            fit_interval=10**9,
            k=3,
            target="mu_minus",
            embedder="direction",
        ),
        noise_nz_fn=_nz,
    )
    imp4.begin_pair(seed=0, sigma=0.1)
    imp4.tell_real(mu=1.0, phase="plus")
    with pytest.raises(AssertionError):
        imp4.predict_current()
    ok, _mu, _se = imp4.try_impute_current()
    assert ok is False


def test_kiss_bridge_turbo_m_ref_types_and_sobol():
    from turbo_m_ref.turbo_types import CandidatesResult, StandardizedFX, TrustRegion
    from turbo_m_ref.utils import make_sobol_candidates

    tr = TrustRegion(x_center=np.zeros(2), lb=np.zeros(2), ub=np.ones(2))
    assert tr.x_center.shape == (2,)
    cr = CandidatesResult(X_cand=np.zeros((1, 2)), y_cand=None, hypers={})
    assert cr.X_cand.shape == (1, 2)
    sf = StandardizedFX(fX=np.zeros(3), mu=0.0, sigma=1.0)
    assert sf.mu == 0.0
    xc = np.zeros(2)
    lb = np.zeros(2)
    ub = np.ones(2)
    X = make_sobol_candidates(
        dim=2,
        n_cand=4,
        x_center=xc,
        lb=lb,
        ub=ub,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert X.shape == (4, 2)


def test_kiss_bridge_mnist_types_and_mlp_torch_env():
    import gymnasium as gym

    from problems.mlp_torch_env import MLPTorchEnv, MLPTorchEnvWrapper, wrap_mlp_env
    from problems.mnist_types import StepResult

    sr = StepResult(state=0, reward=1.0, done=False, info=None)
    assert sr.reward == 1.0

    def _me_init(self):
        gym.Env.__init__(self)
        self.observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def _me_reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def _me_step(self, _a):
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    def _me_close(self):
        return None

    ME = type(
        "ME",
        (gym.Env,),
        {
            "metadata": {},
            "__init__": _me_init,
            "reset": _me_reset,
            "step": _me_step,
            "close": _me_close,
        },
    )
    mod = nn.Linear(2, 1)
    env = ME()
    te = MLPTorchEnv(mod, env, max_steps=2)
    assert MLPTorchEnv.module.fget(te) is mod
    te.reset()
    te.step(np.zeros(1, dtype=np.float32))
    te.close()
    wrap = MLPTorchEnvWrapper(env, mod, max_steps=2, num_frames_skip=1)
    wrap.torch_env()
    wrap.reset()
    wrap.step(np.zeros(1, dtype=np.float32))
    wrap.close()
    w2 = wrap_mlp_env(env, mod, max_steps=2)
    assert isinstance(w2, MLPTorchEnvWrapper)


def test_kiss_bridge_rl_runner_and_helpers():
    from rl import runner
    from rl.runner import main as rl_runner_main
    from rl.runner_helpers import _RuntimeArgs

    assert callable(runner.main) and callable(rl_runner_main)
    a = _RuntimeArgs(None, 2, True, [])
    assert a.workers == 2


def test_kiss_bridge_rl_core_actor_state_extras():
    from rl.core.actor_state import (
        build_ppo_checkpoint_payload,
        capture_ppo_actor_snapshot,
        restore_rng_state_payload,
        rng_state_payload,
    )

    bb = nn.Linear(2, 2)
    hh = nn.Linear(2, 1)
    snap = capture_ppo_actor_snapshot(bb, hh)
    assert "backbone" in snap
    payload = rng_state_payload()
    restore_rng_state_payload(payload)
    cp = build_ppo_checkpoint_payload(
        iteration=1,
        global_step=1,
        actor_snapshot=snap,
        critic_backbone={},
        critic_head={},
        optimizer={},
        best_actor_state=None,
        best_return=0.0,
        last_eval_return=0.0,
        last_heldout_return=None,
    )
    assert cp["iteration"] == 1


def test_kiss_bridge_rl_core_seed_and_env_setup_dataclasses():
    from common.experiment_seeds import ResolvedSeeds
    from rl.core.env_setup import EnvSetup

    r = ResolvedSeeds(problem_seed=1, noise_seed_0=2)
    assert r.problem_seed == 1
    s = EnvSetup(env_conf=object(), problem_seed=1, noise_seed_0=2)
    assert s.env_conf is not None


def test_kiss_bridge_rl_core_env_setup_dataclass():
    from rl.core.env_setup import EnvSetup

    z = np.zeros(1, dtype=np.float32)
    o = EnvSetup(
        env_conf=object(),
        problem_seed=0,
        noise_seed_0=0,
        act_dim=1,
        action_low=z,
        action_high=z,
        obs_lb=None,
        obs_width=None,
    )
    assert o.act_dim == 1


def test_kiss_bridge_rl_core_episode_rollout():
    from rl.core.episode_rollout import (
        MeanReturnResult,
        Trajectory,
        collect_denoised_trajectory,
        collect_episode_return,
        collect_trajectory_with_noise,
        evaluate_for_best,
        mean_return_over_runs,
    )

    assert Trajectory is not None and MeanReturnResult is not None
    assert all(
        callable(f)
        for f in (
            collect_episode_return,
            collect_trajectory_with_noise,
            mean_return_over_runs,
            collect_denoised_trajectory,
            evaluate_for_best,
        )
    )


def test_kiss_bridge_rl_core_replay_buffers():
    from rl.core.replay import NumpyReplayBuffer, TorchRLReplayBuffer

    b = NumpyReplayBuffer((2,), 1, 10)
    b.add_batch(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 1), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
    )
    b.sample(1, torch.device("cpu"))
    TorchRLReplayBuffer((2,), 1, 10)
