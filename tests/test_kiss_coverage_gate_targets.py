from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from kiss_gate_runner_cfg import runner_dummy_config_cls
from kiss_gate_stubs_dm import _DM
from kiss_gate_stubs_enn_core import _ENN
from kiss_gate_stubs_enn_imputer import _ENNImputer
from kiss_gate_stubs_gym_preproc import _Env as _GymClipEnv
from kiss_gate_stubs_mlp_env import _Env as _MlpEnv
from kiss_gate_stubs_mnist import _TinyMNIST
from kiss_gate_stubs_modal_lookup import _Lookup
from kiss_gate_stubs_uhd_np import _Embed, _Pert, _Policy


def test_kiss_cov_types_and_small_helpers():
    from problems.mnist_types import StepResult
    from turbo_m_ref.turbo_types import CandidatesResult, StandardizedFX, TrustRegion

    step = StepResult(state=None, reward=1.0, done=False, info={})
    assert step.reward == 1.0

    tr = TrustRegion(x_center=np.zeros(2), lb=np.zeros(2), ub=np.ones(2))
    cand = CandidatesResult(X_cand=np.zeros((2, 2)), y_cand=None, hypers={})
    sfx = StandardizedFX(fX=np.zeros(2), mu=0.0, sigma=1.0)
    assert tr.lb.shape == (2,)
    assert cand.X_cand.shape == (2, 2)
    assert sfx.sigma == 1.0


def test_kiss_cov_runner_main(monkeypatch, tmp_path):
    import rl.runner as runner

    cfg_path = tmp_path / "rl.toml"
    cfg_path.write_text('[rl]\nalgo="dummy"\n[rl.dummy]\nseed=7\nexp_dir="tmp"\n')

    import common.config_toml as ct
    import rl.runner_helpers as rh

    monkeypatch.setattr(rh, "split_config_and_args", lambda argv: (str(cfg_path), argv[1:]))
    monkeypatch.setattr(
        rh,
        "parse_runtime_args",
        lambda rest: SimpleNamespace(workers=1, workers_cli_set=False, cleaned=rest),
    )
    monkeypatch.setattr(ct, "parse_set_args", lambda cleaned: {})
    monkeypatch.setattr(
        ct,
        "load_toml",
        lambda _path: {"rl": {"algo": "dummy", "dummy": {"seed": 7, "exp_dir": "tmp"}}},
    )
    monkeypatch.setattr(ct, "apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr(runner, "_extract_run_cfg", lambda cfg: ([], 1))

    _Cfg = runner_dummy_config_cls()
    called = {"train": 0}

    def _train_fn(_cfg):
        called["train"] += 1
        return {"ok": True}

    monkeypatch.setattr(
        "rl.registry.get_algo",
        lambda _algo_name: SimpleNamespace(config_cls=_Cfg, train_fn=_train_fn),
    )
    monkeypatch.setattr("rl.builtins.register_all", lambda: None)
    runner.main(["config.toml"])
    assert called["train"] == 1


def test_kiss_cov_dm_control_collect_and_mlp_torch_env(monkeypatch):
    import rl.torchrl.collect_utils as cu
    from problems.mlp_torch_env import MLPTorchEnv, MLPTorchEnvWrapper, wrap_mlp_env

    monkeypatch.setattr("problems.dm_control_env._configure_headless_render_backend", lambda _mode: None)
    monkeypatch.setattr("problems.dm_control_env._parse_env_name", lambda _name: ("cartpole", "swingup"))
    monkeypatch.setattr("dm_control.suite.load", lambda *_args, **_kwargs: _DM())

    monkeypatch.setattr(
        cu,
        "_gym_wrapper_without_isaaclab_probe",
        lambda base: SimpleNamespace(base=base),
    )
    monkeypatch.setattr(
        cu.tr_envs,
        "TransformedEnv",
        lambda wrapped, *a, **k: SimpleNamespace(wrapped=wrapped, base=wrapped.base),
    )

    fake_env_conf = SimpleNamespace(
        make_gym_env=lambda seed=0: SimpleNamespace(reset=lambda seed=None: (None, {}), is_discrete=False),
        problem_seed=1,
    )
    out = cu.make_collect_env(fake_env_conf, env_index=0)
    assert out is not None

    env = _MlpEnv()
    module = torch.nn.Linear(1, 1)
    tenv = MLPTorchEnv(module=module, env=env, max_steps=1)
    tenv.reset(seed=0)
    _ = tenv.step(np.zeros(1))
    tenv.close()

    wrapper = MLPTorchEnvWrapper(env=env, policy_module=module, max_steps=2, num_frames_skip=2)
    wrapper.reset(seed=1)
    _ = wrapper.step(np.zeros(1))
    wrapper.close()
    assert wrap_mlp_env(env, module).torch_env().module is module


def test_kiss_cov_env_preprocessing_and_episode_rollout():
    from common.env_preprocessing import (
        _ClipObservationWrapper,
        apply_gym_preprocessing,
    )
    from rl.core.episode_rollout import _unpack_step_result, collect_episode_return

    env = _GymClipEnv()
    wrapped = _ClipObservationWrapper(env, low=-1.0, high=1.0)
    obs0, _ = wrapped.reset(seed=0)
    obs1, *_ = wrapped.step(np.array([0.0]))
    assert float(obs0[0]) == 1.0
    assert float(obs1[0]) == 1.0

    prep = apply_gym_preprocessing(env, preprocess_spec=SimpleNamespace(enabled=False))
    assert prep is env
    assert _unpack_step_result((np.zeros(1), 1.0, False, {}))[2] is False

    env_conf = SimpleNamespace(
        make=lambda: _GymClipEnv(),
        gym_conf=SimpleNamespace(max_steps=3),
    )
    ret = collect_episode_return(env_conf, lambda _obs: np.array([0.0]), noise_seed=0)
    assert ret > 0.0


def test_kiss_cov_uhd_np(monkeypatch):
    from optimizer.uhd_simple_base import UHDSimpleBase
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    base = UHDSimpleBase(_Pert(), sigma_0=0.1, dim=3)
    assert base.eval_seed == 0
    assert base.sigma > 0
    assert base.y_best is None
    assert base.mu_avg == 0.0
    assert base.se_avg == 0.0

    p = _Policy()
    simple = UHDSimpleNp(p, sigma_0=0.1, param_clip=(-1.0, 1.0))
    simple.ask()
    simple.tell(1.0, 0.1)
    assert simple.eval_seed == 1
    assert simple.y_best == 1.0
    assert simple.mu_avg == 1.0
    assert simple.se_avg == 0.1

    monkeypatch.setattr(
        "optimizer.uhd_simple_be_np.EpistemicNearestNeighbors",
        lambda *args, **kwargs: _ENN(),
    )
    monkeypatch.setattr("optimizer.uhd_simple_be_np.enn_fit", lambda *args, **kwargs: object())
    be = UHDSimpleBENp(p, _Embed(), sigma_0=0.1, warmup=1, fit_interval=1, num_candidates=2)
    be.ask()
    be.tell(1.0, 0.1)
    be.ask()
    be.tell(0.5, 0.2)
    assert be.eval_seed >= 0
    assert be.sigma > 0
    assert be.y_best is not None


def test_kiss_cov_enn_imputer_and_cli_callbacks(monkeypatch, tmp_path):
    import ops.exp_uhd as exp_uhd
    from experiments import modal_batches
    from optimizer.uhd_enn_imputer import ENNImputerConfig, ENNMinusImputer

    monkeypatch.setattr(
        "optimizer.uhd_enn_imputer.EpistemicNearestNeighbors",
        lambda *args, **kwargs: _ENNImputer(),
    )
    monkeypatch.setattr("optimizer.uhd_enn_imputer.enn_fit", lambda *args, **kwargs: object())

    module = torch.nn.Linear(2, 1, bias=False)
    cfg = ENNImputerConfig(
        warmup_real_obs=1,
        fit_interval=1,
        min_calib_points=0,
        max_abs_err_ema=1.0,
        se_threshold=1.0,
        refresh_interval=1000,
    )
    imputer = ENNMinusImputer(
        module=module,
        cfg=cfg,
        noise_nz_fn=lambda seed, sigma: (
            np.array([0, 1], dtype=np.int64),
            np.array([sigma, -sigma], dtype=np.float32),
        ),
    )
    imputer.begin_pair(seed=1, sigma=0.1)
    imputer.tell_real(mu=1.0, phase="plus")
    imputer.tell_real(mu=0.5, phase="minus")
    imputer.calibrate_minus(mu_minus_real=0.4)
    _ = imputer.choose_seed_ucb(base_seed=10, sigma=0.1)
    _ = imputer.should_impute_negative()
    _ = imputer.predict_current()
    _ = imputer.try_impute_current()
    imputer.update_base_after_step(step_scale=0.01, sigma=0.1)
    assert imputer.num_real_evals >= 2
    assert imputer.num_imputed >= 0

    monkeypatch.setattr("ops.modal_uhd.run", lambda *args, **kwargs: "ok")
    toml = tmp_path / "cfg.toml"
    toml.write_text('[uhd]\nenv_tag="f:sphere-2d"\nnum_rounds=1\n')
    exp_uhd.modal_cmd(str(toml), (), None, "A100")

    monkeypatch.setattr(
        modal_batches.modal,
        "Function",
        SimpleNamespace(lookup=lambda *_args, **_kwargs: _Lookup()),
    )
    monkeypatch.setattr(modal_batches, "batches_submitter", lambda *args, **kwargs: None)
    monkeypatch.setattr(modal_batches, "status", lambda: None)
    monkeypatch.setattr(modal_batches, "collect", lambda: None)
    monkeypatch.setattr(modal_batches, "clean_up", lambda: None)
    modal_batches.batches("work", batch_tag=None, num=1)
    modal_batches.batches("submit-missing", batch_tag="x", num=None)


def test_kiss_cov_fit_mnist_main_entry(monkeypatch):
    import torchvision.datasets as tv_datasets

    monkeypatch.setattr(tv_datasets, "MNIST", lambda *args, **kwargs: _TinyMNIST())
    monkeypatch.setattr(sys, "argv", ["ops.fit_mnist"])
    with pytest.raises(SystemExit) as ex:
        runpy.run_module("ops.fit_mnist", run_name="__main__")
    assert ex.value.code == 0
