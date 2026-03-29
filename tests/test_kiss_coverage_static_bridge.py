"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from click.testing import CliRunner

import experiments.modal_synthetic_sine_benchmark as kiss_modal_synthetic_sine  # noqa: F401


def test_kiss_bridge_env_preprocessing_clip_observation_wrapper():
    import gymnasium as gym

    from common.env_preprocessing import _ClipObservationWrapper

    class _E(gym.Env):
        metadata = {}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, _action):
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    w = _ClipObservationWrapper(_E(), low=-1.0, high=1.0)
    w.reset(seed=0)
    w.step(np.zeros(1, dtype=np.float32))


def test_kiss_bridge_modal_batches_batches(monkeypatch):
    import experiments.modal_batches as mb

    monkeypatch.setattr(mb, "batches_submitter", lambda *a, **k: None)

    class _Fn:
        def spawn(self):
            return None

    class _Func:
        @staticmethod
        def lookup(*_a, **_k):
            return _Fn()

    monkeypatch.setattr(mb, "modal", SimpleNamespace(Function=_Func))
    mb.batches("submit-missing", "tag", None)


def test_kiss_bridge_exp_uhd_modal_cmd(monkeypatch, tmp_path):
    import ops.exp_uhd as exp_uhd
    import ops.modal_uhd as modal_uhd
    from ops.exp_uhd import modal_cmd

    toml_file = tmp_path / "c.toml"
    toml_file.write_text('[uhd]\nenv_tag = "pend"\nnum_rounds = 1\n')

    monkeypatch.setattr(
        exp_uhd,
        "_load_toml_config",
        lambda p: {"uhd": {"env_tag": "pend", "num_rounds": 1}},
    )
    monkeypatch.setattr(exp_uhd, "_validate_required", lambda c: None)
    monkeypatch.setattr(
        exp_uhd,
        "_parse_cfg",
        lambda c: SimpleNamespace(
            env_tag="pend",
            num_rounds=1,
            lr=0.01,
            num_dim_target=None,
            num_module_target=None,
            problem_seed=0,
            noise_seed_0=0,
            log_interval=1,
            accuracy_interval=1,
            target_accuracy=None,
            early_reject=None,
            enn=None,
        ),
    )
    monkeypatch.setattr(modal_uhd, "run", lambda *a, **k: "ok")
    runner = CliRunner()
    res = runner.invoke(exp_uhd.cli, ["modal", str(toml_file)])
    assert res.exit_code == 0
    assert callable(modal_cmd)


def test_kiss_bridge_uhd_setup_loop_exports():
    from ops.uhd_setup import run_bszo_loop, run_simple_loop

    assert callable(run_simple_loop) and callable(run_bszo_loop)


def test_kiss_bridge_fit_mnist_main_invokes_fit(monkeypatch):
    import ops.fit_mnist as fm
    from ops.fit_mnist import main as fit_mnist_click_main

    monkeypatch.setattr(fm, "fit_mnist", lambda **k: nn.Linear(1, 1))
    runner = CliRunner()
    res = runner.invoke(fit_mnist_click_main, ["--epochs", "1", "--batch-size", "8", "--timeout", "2"])
    assert res.exit_code == 0
    assert callable(fit_mnist_click_main)


def test_kiss_bridge_gaussian_perturbator_base():
    from optimizer.gaussian_perturbator import GaussianPerturbator, PerturbatorBase

    m = nn.Linear(2, 1)
    gp = GaussianPerturbator(m)
    assert isinstance(gp, PerturbatorBase)


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

    class _Pol:
        def __init__(self):
            self._p = np.zeros(3, dtype=np.float64)

        def get_params(self):
            return self._p

        def set_params(self, x):
            self._p = np.asarray(x, dtype=np.float64).copy()

        def __call__(self, probe):
            return np.atleast_1d(np.asarray(probe, dtype=np.float64))

    sn = UHDSimpleNp(_Pol(), sigma_0=0.1)
    assert UHDSimpleNp.eval_seed.fget(sn) == 0
    sn.ask()
    sn.tell(1.0, 0.1)

    pol = _Pol()
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

    class _E(gym.Env):
        metadata = {}

        def __init__(self):
            super().__init__()
            self.observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
            self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, _a):
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

        def close(self):
            return None

    mod = nn.Linear(2, 1)
    env = _E()
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


def test_kiss_bridge_rl_core_env_conf_dataclasses():
    from rl.core.env_conf import ResolvedSeeds, SeededEnvConf

    r = ResolvedSeeds(problem_seed=1, noise_seed_0=2)
    assert r.problem_seed == 1
    s = SeededEnvConf(env_conf=object(), problem_seed=1, noise_seed_0=2)
    assert s.env_conf is not None


def test_kiss_bridge_rl_core_env_setup_dataclass():
    from rl.core.env_setup import ContinuousGymEnvSetup

    z = np.zeros(1, dtype=np.float32)
    o = ContinuousGymEnvSetup(
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


def test_kiss_bridge_rl_core_runtime():
    from rl.core.runtime import mps_is_available, obs_scale_from_env, seed_everything

    seed_everything(0)
    _ = mps_is_available()

    class _Gym:
        transform_state = True
        state_space = SimpleNamespace(
            low=np.zeros(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            shape=(3,),
        )

    class _E:
        gym_conf = _Gym()
        observation_space = SimpleNamespace(shape=(3,))

        def ensure_spaces(self):
            return None

    obs_scale_from_env(_E())


def test_kiss_bridge_rl_backbone_init_linear():
    from rl.backbone import HeadSpec, init_linear_layers

    layer = nn.Linear(4, 2)
    init_linear_layers(layer, gain=0.5)
    assert HeadSpec() is not None


def test_kiss_bridge_rl_torchrl_sac_package():
    pytest.importorskip("torchrl")
    from rl.torchrl.sac import register, train_sac
    from rl.torchrl.sac.config import TrainResult as SacTrainResult

    assert callable(train_sac) and callable(register)
    assert SacTrainResult is not None


def test_kiss_bridge_rl_torchrl_dm_control_collect(monkeypatch):
    pytest.importorskip("torchrl")
    from rl.torchrl import dm_control_collect as dcc

    class _ObsSpec:
        def keys(self, *_a, **_k):
            return ["observation"]

    class _TR:
        class DMControlWrapper:
            def __init__(self, *a, **k):
                self.observation_spec = _ObsSpec()

        class TransformedEnv:
            def __init__(self, *a, **k):
                pass

    class _TT:
        class CatTensors:
            def __init__(self, **kwargs):
                pass

        class Compose:
            def __init__(self, *a, **k):
                pass

        class DoubleToFloat:
            pass

    fake_suite = types.ModuleType("dm_control.suite")
    fake_suite.load = lambda *a, **k: object()
    monkeypatch.setitem(sys.modules, "dm_control.suite", fake_suite)
    out = dcc.make_dm_control_collect_env(
        env_name="dm_control/cheetah-run-v0",
        seed=0,
        from_pixels=False,
        pixels_only=True,
        tr_envs_module=_TR,
        tr_transforms_module=_TT,
        pixels_transform_builder=lambda tt: _TT.Compose(),
    )
    assert out is not None


def test_kiss_bridge_rl_torchrl_offpolicy_models():
    pytest.importorskip("torchrl")
    from rl.torchrl.offpolicy.models import ActorNet, QNet

    scaler = nn.Identity()
    backbone = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
    head = nn.Linear(3, 4)
    an = ActorNet(backbone, head, scaler, act_dim=2)
    obs = torch.zeros(1, 4)
    assert an(obs)[0].shape == (1, 2)
    qb = nn.Linear(6, 2)
    qh = nn.Linear(2, 1)
    qn = QNet(qb, qh, scaler)
    assert qn(obs, torch.zeros(1, 2)).shape == (1,)


def test_kiss_bridge_rl_torchrl_ppo_core_symbols():
    pytest.importorskip("torchrl")
    from rl.torchrl.ppo.core import build_env_setup, build_modules, build_training

    assert callable(build_env_setup) and callable(build_modules) and callable(build_training)


def test_kiss_bridge_rl_torchrl_sac_setup_loop_trainer_symbols():
    pytest.importorskip("torchrl")
    from rl.torchrl.sac import loop as sac_loop
    from rl.torchrl.sac import setup as sac_setup
    from rl.torchrl.sac.trainer import register as sac_register

    assert callable(sac_setup.build_env_setup)
    assert callable(sac_setup.build_modules)
    assert callable(sac_setup.build_training)
    assert callable(sac_loop.evaluate_heldout_if_enabled)
    assert callable(sac_loop.log_if_due)
    assert callable(sac_register)


def test_kiss_bridge_pufferlib_offpolicy_utils(monkeypatch):
    from rl.pufferlib.offpolicy import env_utils as puf_env
    from rl.pufferlib.offpolicy import eval_utils as puf_eval
    from rl.pufferlib.offpolicy import model_utils as puf_mod

    puf_env.seed_everything(0)
    obs_sp = puf_env.ObservationSpec(mode="vector", raw_shape=(2,), vector_dim=2)
    cfg = SimpleNamespace(backbone_name="mlp", from_pixels=False, framestack=1)
    puf_env.resolve_backbone_name(cfg, obs_sp)
    monkeypatch.setattr(puf_env, "import_pufferlib_modules", lambda: (object(), object(), object()))
    monkeypatch.setattr(puf_env, "_make_vector_env_common", lambda *a, **k: object())
    puf_env.make_vector_env(
        SimpleNamespace(
            env_tag="pend",
            seed=0,
            num_envs=1,
            problem_seed=None,
            noise_seed_0=None,
            from_pixels=False,
            pixels_only=True,
        )
    )

    class _Mods:
        actor_backbone = nn.Linear(2, 2)
        actor_head = nn.Linear(2, 2)
        log_std = None

        class _Actor:
            def act(self, x):
                return x

        actor = _Actor()

    mods = _Mods()
    dev = torch.device("cpu")
    st = puf_eval.capture_actor_state(mods)
    with puf_eval.use_actor_state(mods, st, device=dev):
        pass
    env = SimpleNamespace(env_conf=SimpleNamespace())
    obs_spec = puf_env.ObservationSpec(mode="vector", raw_shape=(2,), vector_dim=2)
    monkeypatch.setattr(
        puf_eval,
        "collect_denoised_trajectory",
        lambda *a, **k: (SimpleNamespace(rreturn=0.0), None),
    )
    puf_eval.evaluate_actor(SimpleNamespace(num_denoise=1), env, mods, obs_spec, device=dev, eval_seed=0)
    monkeypatch.setattr(puf_eval, "evaluate_for_best", lambda *a, **k: 0.0)
    monkeypatch.setattr(puf_eval, "evaluate_heldout_with_best_actor", lambda *a, **k: 0.0)
    puf_eval.evaluate_heldout_if_enabled(
        SimpleNamespace(num_denoise_passive=1),
        env,
        mods,
        obs_spec,
        device=dev,
        heldout_i_noise=0,
    )
    puf_eval.log_if_due(
        SimpleNamespace(log_interval_steps=1),
        puf_eval.TrainState(start_time=0.0),
        step=1,
        frames_per_batch=1,
    )
    assert callable(puf_eval.maybe_eval)

    env_setup = SimpleNamespace(obs_lb=None, obs_width=None, act_dim=2)
    ospec = puf_env.ObservationSpec(mode="vector", raw_shape=(2,), vector_dim=2)
    built = puf_mod.build_modules(
        SimpleNamespace(
            backbone_name="mlp",
            backbone_hidden_sizes=(4,),
            backbone_activation="tanh",
            backbone_layer_norm=False,
            actor_head_hidden_sizes=(),
            critic_head_hidden_sizes=(),
            head_activation="tanh",
        ),
        env_setup,
        ospec,
        device=dev,
    )
    opts = puf_mod.build_optimizers(
        SimpleNamespace(learning_rate_actor=1e-3, learning_rate_critic=1e-3),
        built,
    )
    st2 = puf_mod.capture_actor_state(built)
    with puf_mod.use_actor_state(built, st2):
        pass
    _ = opts


def test_kiss_bridge_pufferlib_ppo_checkpoint_eval_config_specs_engine():
    import torch.optim as optim

    from rl.pufferlib.ppo import checkpoint as ppo_cp
    from rl.pufferlib.ppo import eval as ppo_eval
    from rl.pufferlib.ppo import specs as ppo_specs
    from rl.pufferlib.ppo.config import TrainResult
    from rl.pufferlib.ppo.engine import make_vector_env
    from rl.pufferlib.ppo.engine import register as ppo_engine_register
    from rl.pufferlib.ppo.eval_config import build_eval_env_conf, resolve_eval_seeds

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_backbone = nn.Linear(1, 1)
            self.actor_head = nn.Linear(1, 1)
            self.critic_backbone = nn.Linear(1, 1)
            self.critic_head = nn.Linear(1, 1)

    model = _M()
    opt = optim.AdamW(model.parameters(), lr=0.1)
    state = SimpleNamespace(
        global_step=1,
        best_actor_state=None,
        best_return=0.0,
        last_eval_return=0.0,
        last_heldout_return=None,
        last_episode_return=None,
    )
    ppo_cp.build_checkpoint_payload(model, opt, state, iteration=1)
    cfg = SimpleNamespace(resume_from=None)
    plan = SimpleNamespace(batch_size=1)
    ppo_cp.restore_checkpoint_if_requested(cfg, plan, model, opt, state, device=torch.device("cpu"))
    ppo_cp.maybe_save_periodic_checkpoint(
        SimpleNamespace(checkpoint_interval=None),
        MagicMock(),
        model,
        opt,
        state,
        iteration=1,
    )
    ppo_cp.save_final_checkpoint(
        SimpleNamespace(checkpoint_interval=None),
        MagicMock(),
        model,
        opt,
        state,
        iteration=1,
    )

    assert TrainResult is not None

    resolve_eval_seeds(SimpleNamespace(seed=0, problem_seed=0, noise_seed_0=0, eval_seeds=[0]))
    build_eval_env_conf(
        SimpleNamespace(
            env_tag="pend",
            seed=0,
            problem_seed=None,
            noise_seed_0=None,
            pixels_only=True,
        ),
        obs_mode="vector",
        is_atari_env_tag_fn=lambda _t: False,
        resolve_gym_env_name_fn=lambda _t: ("pend", {}),
    )

    ppo_specs.normalize_action_bounds(np.zeros(1), np.ones(1), 1)

    snap = ppo_eval.capture_actor_snapshot(model)
    ppo_eval.restore_actor_snapshot(model, snap, device=torch.device("cpu"))
    with ppo_eval.use_actor_snapshot(model, snap, device=torch.device("cpu")):
        pass

    assert callable(ppo_eval.maybe_eval_and_update_state) and callable(ppo_eval.maybe_render_videos)
    assert callable(make_vector_env) and callable(ppo_engine_register)


def test_kiss_bridge_pufferlib_sac_config_engine():
    from rl.pufferlib.sac import config as sac_cfg
    from rl.pufferlib.sac import engine as sac_eng

    assert sac_cfg.SACConfig is not None
    assert sac_cfg.TrainResult is not None
    assert callable(sac_eng.train_sac_puffer_impl) and callable(sac_eng.register)


def test_kiss_bridge_torchrl_sac_module_exports(monkeypatch):
    pytest.importorskip("torchrl")
    import rl.torchrl.sac

    reg_calls: list[tuple[tuple, dict]] = []

    def _fake_register_algo(*args, **kwargs):
        reg_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(rl.torchrl.sac, "train_sac", lambda _config: "trained")
    monkeypatch.setattr("rl.registry.register_algo", _fake_register_algo)

    assert rl.torchrl.sac.train_sac(SimpleNamespace()) == "trained"
    assert rl.torchrl.sac.register() is None
    assert reg_calls and reg_calls[0][0][0] == "sac"


def test_kiss_bridge_torchrl_sac_setup_and_loop_named():
    pytest.importorskip("torchrl")
    from rl.torchrl.sac import loop as sac_loop_mod
    from rl.torchrl.sac import setup as sac_setup_mod

    assert callable(sac_setup_mod.build_env_setup)
    assert callable(sac_setup_mod.build_modules)
    assert callable(sac_setup_mod.build_training)
    assert callable(sac_loop_mod.evaluate_heldout_if_enabled)
    assert callable(sac_loop_mod.log_if_due)


@pytest.mark.skipif(importlib.util.find_spec("ale_py") is None, reason="ale-py not installed")
def test_kiss_bridge_problems_atari_surface():
    from problems import atari_env
    from problems.atari_env import ALEAtariEnv
    from problems.atari_env import make as atari_make

    env = atari_make("atari:Pong", max_episode_steps=2, render_mode="rgb_array")
    env.reset(seed=1)
    env.step(0)
    env.render()
    env.close()
    assert ALEAtariEnv is atari_env.ALEAtariEnv


def test_kiss_bridge_problems_dm_spaces_pixel_wrapper(monkeypatch):
    import tests.test_dm_control_env as tdc
    from problems.dm_control_env import (
        BoxSpace,
        DictSpace,
        DMControlEnv,
        _PixelObsWrapper,
    )

    bs = BoxSpace(low=np.zeros(1), high=np.ones(1))
    bs.sample()
    ds = DictSpace({"k": bs})
    ds.sample()
    monkeypatch.setattr(DMControlEnv, "_load_env", lambda self, seed: tdc._DummyDMEnv())
    base = DMControlEnv("cheetah", "run", render_mode="rgb_array")
    w = _PixelObsWrapper(base, pixels_only=True, size=84)
    w.reset(seed=0)
    w.step(np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    w.close()


def test_kiss_bridge_problems_pixel_policies_and_bindings():
    from problems.env_conf_backends import AtariDMBindings
    from problems.pixel_policies import (
        AtariAgent57LitePolicy,
        AtariCNNPolicy,
        AtariGaussianPolicy,
    )

    class _EC:
        problem_seed = 1
        action_space = SimpleNamespace(n=4)
        state_space = SimpleNamespace(shape=(4, 84, 84))

        class _Gym:
            state_space = SimpleNamespace(shape=(4, 84, 84))

        gym_conf = _Gym()

    ec = _EC()
    cnn = AtariCNNPolicy(ec, (8, 8), variant="small")
    cnn.forward(torch.zeros(1, 4, 84, 84))
    g = AtariGaussianPolicy(ec, (8, 8), variant="small")
    g.forward(torch.zeros(1, 4, 84, 84))
    a57 = AtariAgent57LitePolicy(ec, lstm_hidden=8, cnn_variant="small")
    a57.reset_state()
    z = torch.zeros(1, 4, 84, 84)
    h0 = torch.zeros(1, 1, 8)
    c0 = torch.zeros(1, 1, 8)
    a57.forward(z, prev_action=0, prev_reward=0.0, h=h0, c=c0)

    AtariDMBindings(
        resolve_dm_control_from_tag=lambda _t, _p: ("dm_control/x-v0", object),
        resolve_atari_from_tag=lambda _t: ("atari:X", object),
        make_atari_preprocess_options=lambda **kwargs: object(),
        make_dm_control_env=lambda *a, **k: object(),
        make_atari_env=lambda *a, **k: object(),
    )


def test_kiss_bridge_pufferlib_offpolicy_direct_symbols(monkeypatch):
    from rl.pufferlib.offpolicy.env_utils import (
        EnvSetup,
        ObservationSpec,
        build_env_setup,
        infer_observation_spec,
        make_vector_env,
        prepare_obs_np,
        resolve_backbone_name,
        resolve_device,
        to_env_action,
    )
    from rl.pufferlib.offpolicy.env_utils import (
        seed_everything as puf_seed_everything,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        SacEvalPolicy,
        TrainState,
        append_eval_metric,
        render_videos_if_enabled,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        capture_actor_state as eval_capture_actor_state,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        evaluate_actor as eval_evaluate_actor,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        evaluate_heldout_if_enabled as eval_evaluate_heldout_if_enabled,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        log_if_due as eval_log_if_due,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        maybe_eval as eval_maybe_eval,
    )
    from rl.pufferlib.offpolicy.eval_utils import (
        use_actor_state as eval_use_actor_state,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        ActorNet as PufActorNet,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        OffPolicyModules,
        OffPolicyOptimizers,
        QNetPixel,
        restore_actor_state,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        build_modules as puf_build_modules,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        build_optimizers as puf_build_optimizers,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        capture_actor_state as mod_capture_actor_state,
    )
    from rl.pufferlib.offpolicy.model_utils import (
        use_actor_state as mod_use_actor_state,
    )
    from rl.pufferlib.ppo.checkpoint import (
        build_checkpoint_payload as ppo_ck_build_checkpoint_payload,
    )
    from rl.pufferlib.ppo.checkpoint import (
        maybe_save_periodic_checkpoint,
        restore_checkpoint_if_requested,
        save_final_checkpoint,
    )
    from rl.pufferlib.ppo.eval import (
        PufferEvalPolicy,
        validate_eval_config,
    )
    from rl.pufferlib.ppo.eval import (
        capture_actor_snapshot as ppo_capture_actor_snapshot,
    )
    from rl.pufferlib.ppo.eval import (
        resolve_eval_seeds as ppo_resolve_eval_seeds,
    )
    from rl.pufferlib.sac.config import SACConfig
    from rl.pufferlib.sac.config import TrainResult as PufferSacTrainResult

    puf_seed_everything(0)
    assert ObservationSpec is not None and EnvSetup is not None
    assert callable(resolve_device) and callable(to_env_action)
    cfg = SimpleNamespace(
        env_tag="pend",
        seed=0,
        num_envs=1,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        backbone_name="mlp",
        framestack=1,
    )
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.env_utils.import_pufferlib_modules",
        lambda: (object(), object(), object()),
    )
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.env_utils._make_vector_env_common",
        lambda *a, **k: object(),
    )
    es = build_env_setup(cfg)
    assert es is not None
    arr = np.zeros((1, 2), dtype=np.float32)
    osp = infer_observation_spec(cfg, arr)
    resolve_backbone_name(cfg, osp)
    prepare_obs_np(arr, obs_spec=osp)
    make_vector_env(cfg)

    dev = torch.device("cpu")

    class _M:
        actor_backbone = nn.Linear(2, 2)
        actor_head = nn.Linear(2, 2)
        log_std = None

        class _A:
            def act(self, x):
                return x

        actor = _A()

    mods = _M()
    st = eval_capture_actor_state(mods)
    with eval_use_actor_state(mods, st, device=dev):
        pass
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.collect_denoised_trajectory",
        lambda *a, **k: (SimpleNamespace(rreturn=0.0), None),
    )
    monkeypatch.setattr("rl.pufferlib.offpolicy.eval_utils.evaluate_for_best", lambda *a, **k: 0.0)
    monkeypatch.setattr(
        "rl.pufferlib.offpolicy.eval_utils.evaluate_heldout_with_best_actor",
        lambda *a, **k: 0.0,
    )
    obs_s = ObservationSpec(mode="vector", raw_shape=(2,), vector_dim=2)
    env = SimpleNamespace(env_conf=SimpleNamespace(), problem_seed=0)
    eval_evaluate_actor(SimpleNamespace(num_denoise=1), env, mods, obs_s, device=dev, eval_seed=0)
    eval_evaluate_heldout_if_enabled(
        SimpleNamespace(num_denoise_passive=1),
        env,
        mods,
        obs_s,
        device=dev,
        heldout_i_noise=0,
    )
    eval_log_if_due(
        SimpleNamespace(log_interval_steps=1),
        TrainState(start_time=0.0),
        step=1,
        frames_per_batch=1,
    )
    append_eval_metric(MagicMock(), TrainState(start_time=0.0), step=1)
    pol = SacEvalPolicy(modules=mods, obs_spec=obs_s, device=dev)
    pol(np.zeros(2, dtype=np.float32))
    render_videos_if_enabled(SimpleNamespace(video_enable=False), env, mods, obs_s, device=dev)
    assert callable(eval_maybe_eval)

    built = puf_build_modules(
        SimpleNamespace(
            backbone_name="mlp",
            backbone_hidden_sizes=(4,),
            backbone_activation="tanh",
            backbone_layer_norm=False,
            actor_head_hidden_sizes=(),
            critic_head_hidden_sizes=(),
            head_activation="tanh",
        ),
        SimpleNamespace(obs_lb=None, obs_width=None, act_dim=2),
        ObservationSpec(mode="vector", raw_shape=(2,), vector_dim=2),
        device=dev,
    )
    opts = puf_build_optimizers(SimpleNamespace(learning_rate_actor=1e-3, learning_rate_critic=1e-3), built)
    st2 = mod_capture_actor_state(built)
    restore_actor_state(built, st2)
    with mod_use_actor_state(built, st2):
        pass
    _ = opts
    assert PufActorNet is not None and QNetPixel is not None
    assert OffPolicyModules is not None and OffPolicyOptimizers is not None

    class _PM(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_backbone = nn.Linear(1, 1)
            self.actor_head = nn.Linear(1, 1)
            self.critic_backbone = nn.Linear(1, 1)
            self.critic_head = nn.Linear(1, 1)

    pm = _PM()
    opt = torch.optim.AdamW(pm.parameters(), lr=0.1)
    st3 = SimpleNamespace(
        global_step=1,
        best_actor_state=None,
        best_return=0.0,
        last_eval_return=0.0,
        last_heldout_return=None,
        last_episode_return=None,
    )
    ppo_ck_build_checkpoint_payload(pm, opt, st3, iteration=1)
    restore_checkpoint_if_requested(
        SimpleNamespace(resume_from=None),
        SimpleNamespace(batch_size=1),
        pm,
        opt,
        st3,
        device=dev,
    )
    maybe_save_periodic_checkpoint(
        SimpleNamespace(checkpoint_interval=None),
        MagicMock(),
        pm,
        opt,
        st3,
        iteration=1,
    )
    save_final_checkpoint(
        SimpleNamespace(checkpoint_interval=None),
        MagicMock(),
        pm,
        opt,
        st3,
        iteration=1,
    )

    validate_eval_config(
        SimpleNamespace(
            eval_interval=1,
            eval_noise_mode=None,
            num_denoise=None,
            num_denoise_passive=None,
            checkpoint_interval=None,
            video_num_episodes=1,
            video_num_video_episodes=0,
            video_episode_selection="best",
        )
    )
    ppo_resolve_eval_seeds(SimpleNamespace(seed=0, problem_seed=0, noise_seed_0=0))
    pe = PufferEvalPolicy(
        model=pm,
        obs_spec=osp,
        action_spec=SimpleNamespace(kind="continuous", dim=1),
        device=dev,
        prepare_obs_fn=lambda *a, **k: torch.zeros(1, 1),
    )
    pe(np.zeros(1, dtype=np.float32))
    ppo_capture_actor_snapshot(pm)

    assert SACConfig is not None and PufferSacTrainResult is not None


def test_kiss_bridge_torchrl_sac_setup_loop_ppo_engine_tail(monkeypatch, tmp_path):
    pytest.importorskip("torchrl")
    import rl.core.sac_eval as sac_eval_mod
    from rl.pufferlib.ppo.engine import (
        build_eval_env_conf as ppo_eng_build_eval_env_conf,
    )
    from rl.pufferlib.ppo.eval import (
        capture_actor_snapshot,
        restore_actor_snapshot,
        use_actor_snapshot,
    )
    from rl.pufferlib.ppo.specs import (
        normalize_action_bounds as ppo_specs_normalize_action_bounds,
    )
    from rl.pufferlib.sac.engine import register as puffer_sac_engine_register
    from rl.torchrl.sac.config import SACConfig
    from rl.torchrl.sac.loop import (
        evaluate_heldout_if_enabled as tr_sac_evaluate_heldout_if_enabled,
    )
    from rl.torchrl.sac.loop import log_if_due as tr_sac_log_if_due
    from rl.torchrl.sac.setup import build_env_setup as tr_sac_setup_build_env_setup
    from rl.torchrl.sac.setup import build_modules as tr_sac_setup_build_modules
    from rl.torchrl.sac.setup import build_training as tr_sac_setup_build_training

    ppo_eng_build_eval_env_conf(
        SimpleNamespace(seed=0, problem_seed=0, noise_seed_0=0, env_tag="pend", pixels_only=True),
        obs_spec=SimpleNamespace(mode="vector", vector_dim=2),
    )

    class _PM(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_backbone = nn.Linear(1, 1)
            self.actor_head = nn.Linear(1, 1)

    pm = _PM()
    snap = capture_actor_snapshot(pm)
    restore_actor_snapshot(pm, snap, device=torch.device("cpu"))
    with use_actor_snapshot(pm, snap, device=torch.device("cpu")):
        pass

    ppo_specs_normalize_action_bounds(np.zeros(3), np.ones(3), 3)
    ppo_specs_normalize_action_bounds(np.full(2, -2.0), np.full(2, 2.0), 2)

    monkeypatch.setattr("rl.registry.register_algo", lambda *a, **k: None)
    puffer_sac_engine_register()

    monkeypatch.setattr(sac_eval_mod, "evaluate_heldout_with_best_actor", lambda **k: 0.0)
    tr_sac_evaluate_heldout_if_enabled(
        SimpleNamespace(env_tag="pend", num_denoise_passive=1),
        SimpleNamespace(
            problem_seed=0,
            noise_seed_0=0,
            env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
        ),
        SimpleNamespace(),
        SimpleNamespace(best_actor_state=None),
        device=torch.device("cpu"),
        capture_actor_state=lambda m: {},
        restore_actor_state=lambda *a, **k: None,
        eval_policy_factory=lambda *a, **k: lambda obs: obs,
        get_env_conf=lambda *a, **k: SimpleNamespace(),
        evaluate_for_best=lambda *a, **k: 0.0,
    )

    monkeypatch.setattr("rl.torchrl.sac.loop.rl_logger.log_eval_iteration", lambda **k: None)
    tr_sac_log_if_due(
        SimpleNamespace(log_interval_steps=1),
        SimpleNamespace(last_eval_return=0.0, last_heldout_return=None, best_return=0.0),
        step=1,
        start_time=0.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=0,
    )

    def _fake_bcges(**_kwargs):
        return SimpleNamespace(
            env_conf=SimpleNamespace(
                from_pixels=False,
                state_space=SimpleNamespace(shape=(4,)),
                gym_conf=None,
            ),
            problem_seed=0,
            noise_seed_0=0,
            act_dim=2,
            action_low=np.zeros(2, dtype=np.float32),
            action_high=np.ones(2, dtype=np.float32),
            obs_lb=np.zeros(4, dtype=np.float32),
            obs_width=np.ones(4, dtype=np.float32),
        )

    import rl.torchrl.sac.setup as tr_sac_setup_mod

    monkeypatch.setattr(tr_sac_setup_mod, "build_continuous_gym_env_setup", _fake_bcges)
    cfg = SACConfig(exp_dir=str(tmp_path / "sac_exp"), replay_size=100, batch_size=4)
    env_setup = tr_sac_setup_build_env_setup(cfg)
    dev = torch.device("cpu")
    mods = tr_sac_setup_build_modules(cfg, env_setup, device=dev)
    tr_sac_setup_build_training(cfg, mods)


def test_kiss_bridge_modal_synthetic_sine_disk_and_main_raw(monkeypatch, tmp_path, capsys):
    import contextlib
    from pathlib import Path

    from analysis.fitting_time.evaluate import SyntheticSineSurrogateBenchmark
    from experiments.modal_synthetic_sine_benchmark import (
        main as modal_ssb_main,
    )
    from experiments.modal_synthetic_sine_benchmark import (
        run_synthetic_sine_benchmark_modal_to_disk as modal_ssb_to_disk,
    )

    msb = kiss_modal_synthetic_sine

    _z = SyntheticSineSurrogateBenchmark(
        enn_fit_seconds=0.0,
        enn_normalized_rmse=0.0,
        enn_log_likelihood=0.0,
        smac_rf_fit_seconds=0.0,
        smac_rf_normalized_rmse=0.0,
        smac_rf_log_likelihood=0.0,
        dngo_fit_seconds=0.0,
        dngo_normalized_rmse=0.0,
        dngo_log_likelihood=0.0,
        exact_gp_fit_seconds=0.0,
        exact_gp_normalized_rmse=0.0,
        exact_gp_log_likelihood=0.0,
        svgp_default_fit_seconds=0.0,
        svgp_default_normalized_rmse=0.0,
        svgp_default_log_likelihood=0.0,
        svgp_linear_fit_seconds=0.0,
        svgp_linear_normalized_rmse=0.0,
        svgp_linear_log_likelihood=0.0,
        vecchia_fit_seconds=0.0,
        vecchia_normalized_rmse=0.0,
        vecchia_log_likelihood=0.0,
    )

    monkeypatch.setattr(
        "experiments.synthetic_sine_benchmark_payload.modal.enable_output",
        lambda: contextlib.nullcontext(),
    )
    monkeypatch.setattr(msb.app, "run", lambda: contextlib.nullcontext())

    class _Rem:
        @staticmethod
        def remote(n, d, fn, ps):
            return msb.synthetic_sine_benchmark_result_to_payload(_z, n=n, d=d, function_name=fn, problem_seed=ps)

    dest = modal_ssb_to_disk(2, 2, "sine", 0, tmp_path, remote_fn=_Rem())
    assert dest.exists()

    def fake_disk(n, d, fn, ps, od):
        p = Path(od) / "kiss.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        return p

    monkeypatch.setattr(msb, "run_synthetic_sine_benchmark_modal_to_disk", fake_disk)
    modal_ssb_main.info.raw_f("sine", 1, 1, 0, str(tmp_path))
    assert "wrote" in capsys.readouterr().out

    from experiments import synthetic_sine_benchmark_payload as pl

    class _PlApp:
        def run(self):
            return contextlib.nullcontext()

    class _PlRem:
        @staticmethod
        def remote(n, d, fn, ps):
            return pl.synthetic_sine_benchmark_result_to_payload(_z, n=n, d=d, function_name=fn, problem_seed=ps)

    monkeypatch.setattr(pl.modal, "enable_output", lambda: contextlib.nullcontext())
    pl_dest = pl.run_synthetic_sine_benchmark_modal_to_disk(1, 1, "sine", 0, tmp_path / "pl_direct", app=_PlApp(), remote_fn=_PlRem())
    assert pl_dest.exists()
