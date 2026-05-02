from __future__ import annotations

import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


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


def test_kiss_cov_runner_main_and_eval_config(monkeypatch, tmp_path):
    import rl.pufferlib.ppo.eval_config as eval_config
    import rl.runner as runner

    monkeypatch.setattr(eval_config, "resolve_run_seeds", lambda **kwargs: SimpleNamespace(problem_seed=3, noise_seed_0=4))
    assert eval_config.resolve_eval_seeds(SimpleNamespace(seed=1, problem_seed=None, noise_seed_0=None)) == (3, 4)

    monkeypatch.setattr(
        eval_config,
        "build_seeded_env_conf_from_run",
        lambda **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(gym_conf=None),
            problem_seed=kwargs["seed"] + 1,
            noise_seed_0=kwargs["seed"] + 2,
        ),
    )
    monkeypatch.setattr(eval_config, "resolve_run_seeds", lambda **kwargs: SimpleNamespace(problem_seed=5, noise_seed_0=6))
    fake_env_conf = eval_config.build_eval_env_conf(
        SimpleNamespace(env_tag="x", seed=1, problem_seed=None, noise_seed_0=None, pixels_only=False),
        obs_mode="vector",
        is_atari_env_tag_fn=lambda _tag: False,
        resolve_gym_env_name_fn=lambda _tag: ("CartPole-v1", {}),
    )
    assert fake_env_conf.env_name == "CartPole-v1"

    cfg_path = tmp_path / "rl.toml"
    cfg_path.write_text('[rl]\nalgo="dummy"\n[rl.dummy]\nseed=7\nexp_dir="tmp"\n')

    monkeypatch.setattr(runner, "split_config_and_args", lambda argv: (str(cfg_path), argv[1:]))
    monkeypatch.setattr(runner, "parse_runtime_args", lambda rest: SimpleNamespace(workers=1, workers_cli_set=False, cleaned=rest))
    monkeypatch.setattr(runner, "parse_set_args", lambda cleaned: {})
    monkeypatch.setattr(runner, "load_toml", lambda _path: {"rl": {"algo": "dummy", "dummy": {"seed": 7, "exp_dir": "tmp"}}})
    monkeypatch.setattr(runner, "apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr(runner, "_extract_run_cfg", lambda cfg: ([], 1))
    monkeypatch.setattr(runner, "resolve_algo_name", lambda algo, backend=None: algo)

    class _Cfg:
        seed = 7
        exp_dir = "tmp"
        problem_seed = None
        noise_seed_0 = None

        @classmethod
        def from_dict(cls, _d):
            return cls()

    called = {"train": 0}

    def _train_fn(_cfg):
        called["train"] += 1
        return {"ok": True}

    monkeypatch.setattr(runner, "get_algo", lambda _algo_name, backend=None: SimpleNamespace(config_cls=_Cfg, train_fn=_train_fn))
    monkeypatch.setattr("rl.builtins.register_all", lambda: None)
    runner.main(["config.toml"])
    assert called["train"] == 1


def test_kiss_cov_dm_control_collect_and_mlp_torch_env(monkeypatch):
    from problems.mlp_torch_env import MLPTorchEnv, MLPTorchEnvWrapper, wrap_mlp_env
    from rl.torchrl import dm_control_collect

    class _DM:
        pass

    monkeypatch.setattr("problems.dm_control_env._configure_headless_render_backend", lambda _mode: None)
    monkeypatch.setattr("problems.dm_control_env._parse_env_name", lambda _name: ("cartpole", "swingup"))
    monkeypatch.setattr("dm_control.suite.load", lambda *_args, **_kwargs: _DM())

    class _TrEnv:
        class DMControlWrapper:
            def __init__(self, dm_env, from_pixels, pixels_only):
                self.dm_env = dm_env
                self.from_pixels = from_pixels
                self.pixels_only = pixels_only
                self.observation_spec = SimpleNamespace(keys=lambda *args: ["state"])

        class TransformedEnv:
            def __init__(self, base, transforms):
                self.base = base
                self.transforms = transforms

    class _TrTf:
        class CatTensors:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class DoubleToFloat:
            pass

        class Compose:
            def __init__(self, *parts):
                self.parts = parts

    out = dm_control_collect.make_dm_control_collect_env(
        env_name="dm_control/cartpole-swingup-v0",
        seed=1,
        from_pixels=False,
        pixels_only=False,
        tr_envs_module=_TrEnv,
        tr_transforms_module=_TrTf,
        pixels_transform_builder=lambda _m: "pix",
    )
    assert isinstance(out.base, _TrEnv.DMControlWrapper)

    class _Space:
        shape = (1,)

    class _Env:
        observation_space = _Space()
        action_space = _Space()

        def __init__(self):
            self.n = 0

        def reset(self, seed=None):
            return np.zeros(1), {"seed": seed}

        def step(self, action):
            self.n += 1
            return np.zeros(1), 1.0, False, self.n > 1, {}

        def close(self):
            return None

    env = _Env()
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
    import gymnasium as gym

    from common.env_preprocessing import _ClipObservationWrapper, apply_gym_preprocessing
    from rl.core.episode_rollout import _unpack_step_result, collect_episode_return

    class _Space:
        low = np.array([-1.0])
        high = np.array([1.0])

    class _Env(gym.Env):
        action_space = _Space()
        observation_space = _Space()

        def __init__(self):
            self.k = 0

        def reset(self, seed=None, options=None):
            return np.array([2.0]), {}

        def step(self, _a):
            self.k += 1
            return np.array([3.0]), 1.0, self.k >= 2, False, {}

        def close(self):
            return None

    env = _Env()
    wrapped = _ClipObservationWrapper(env, low=-1.0, high=1.0)
    obs0, _ = wrapped.reset(seed=0)
    obs1, *_ = wrapped.step(np.array([0.0]))
    assert float(obs0[0]) == 1.0
    assert float(obs1[0]) == 1.0

    prep = apply_gym_preprocessing(env, preprocess_spec=SimpleNamespace(enabled=False))
    assert prep is env
    assert _unpack_step_result((np.zeros(1), 1.0, False, {}))[2] is False

    env_conf = SimpleNamespace(
        make=lambda: _Env(),
        gym_conf=SimpleNamespace(max_steps=3),
    )
    ret = collect_episode_return(env_conf, lambda _obs: np.array([0.0]), noise_seed=0)
    assert ret > 0.0


def test_kiss_cov_checkpoint_and_uhd_np(monkeypatch, tmp_path):
    from optimizer.uhd_simple_base import UHDSimpleBase
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp
    from rl.pufferlib.ppo import checkpoint as ppo_ckpt

    payload = ppo_ckpt.build_checkpoint_payload(
        model=SimpleNamespace(
            actor_backbone=torch.nn.Linear(1, 1),
            actor_head=torch.nn.Linear(1, 1),
            critic_backbone=torch.nn.Linear(1, 1),
            critic_head=torch.nn.Linear(1, 1),
            log_std=None,
        ),
        optimizer=torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=1e-3),
        state=SimpleNamespace(
            global_step=1,
            best_actor_state=None,
            best_return=1.0,
            last_eval_return=1.0,
            last_heldout_return=None,
            last_episode_return=1.0,
        ),
        iteration=1,
    )
    assert payload["iteration"] == 1

    model = SimpleNamespace(
        actor_backbone=torch.nn.Linear(1, 1),
        actor_head=torch.nn.Linear(1, 1),
        critic_backbone=torch.nn.Linear(1, 1),
        critic_head=torch.nn.Linear(1, 1),
        log_std=None,
    )
    optimizer = torch.optim.AdamW(model.actor_backbone.parameters(), lr=1e-3)
    expected_loaded = {
        "actor_backbone": model.actor_backbone.state_dict(),
        "actor_head": model.actor_head.state_dict(),
        "critic_backbone": model.critic_backbone.state_dict(),
        "critic_head": model.critic_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": 2,
        "global_step": 3,
    }
    monkeypatch.setattr(
        ppo_ckpt,
        "load_checkpoint",
        lambda _path, device: expected_loaded,
    )
    monkeypatch.setattr(ppo_ckpt, "restore_backbone_head_snapshot", lambda *args, **kwargs: None)
    state = SimpleNamespace(
        start_iteration=0, global_step=0, best_actor_state=None, best_return=0.0, last_eval_return=0.0, last_heldout_return=None, last_episode_return=0.0
    )
    ppo_ckpt.restore_checkpoint_if_requested(
        SimpleNamespace(resume_from=str(tmp_path / "x.ckpt")), SimpleNamespace(batch_size=1), model, optimizer, state, device=torch.device("cpu")
    )
    called = {"save": 0}
    mgr = SimpleNamespace(save_both=lambda payload, iteration: called.__setitem__("save", called["save"] + 1))
    ppo_ckpt.maybe_save_periodic_checkpoint(SimpleNamespace(checkpoint_interval=1), mgr, model, optimizer, state, iteration=1)
    ppo_ckpt.save_final_checkpoint(SimpleNamespace(checkpoint_interval=1), mgr, model, optimizer, state, iteration=2)
    assert called["save"] == 2

    class _Pert:
        def accept(self):
            return None

        def unperturb(self):
            return None

    base = UHDSimpleBase(_Pert(), sigma_0=0.1, dim=3)
    assert base.eval_seed == 0
    assert base.sigma > 0
    assert base.y_best is None
    assert base.mu_avg == 0.0
    assert base.se_avg == 0.0

    class _Policy:
        def __init__(self):
            self._x = np.zeros(3)

        def get_params(self):
            return self._x

        def set_params(self, x):
            self._x = np.asarray(x)

    p = _Policy()
    simple = UHDSimpleNp(p, sigma_0=0.1, param_clip=(-1.0, 1.0))
    simple.ask()
    simple.tell(1.0, 0.1)
    assert simple.eval_seed == 1
    assert simple.y_best == 1.0
    assert simple.mu_avg == 1.0
    assert simple.se_avg == 0.1

    class _Embed:
        def embed_policy(self, _policy, x):
            return np.asarray(x, dtype=np.float64)

    class _Posterior:
        def __init__(self, n):
            self.mu = np.zeros((n, 1))
            self.se = np.ones((n, 1))

    class _ENN:
        def posterior(self, x, params=None, flags=None):
            _ = params, flags
            return _Posterior(len(x))

    monkeypatch.setattr("optimizer.uhd_simple_be_np.EpistemicNearestNeighbors", lambda *args, **kwargs: _ENN())
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

    class _Posterior:
        def __init__(self, n):
            self.mu = np.zeros((n, 1))
            self.se = np.zeros((n, 1))

    class _ENN:
        def posterior(self, x, params=None, flags=None):
            _ = params, flags
            return _Posterior(len(x))

    monkeypatch.setattr("optimizer.uhd_enn_imputer.EpistemicNearestNeighbors", lambda *args, **kwargs: _ENN())
    monkeypatch.setattr("optimizer.uhd_enn_imputer.enn_fit", lambda *args, **kwargs: object())

    module = torch.nn.Linear(2, 1, bias=False)
    cfg = ENNImputerConfig(warmup_real_obs=1, fit_interval=1, min_calib_points=0, max_abs_err_ema=1.0, se_threshold=1.0, refresh_interval=1000)
    imputer = ENNMinusImputer(
        module=module,
        cfg=cfg,
        noise_nz_fn=lambda seed, sigma: (np.array([0, 1], dtype=np.int64), np.array([sigma, -sigma], dtype=np.float32)),
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

    class _Lookup:
        def spawn(self):
            return None

    monkeypatch.setattr(modal_batches.modal, "Function", SimpleNamespace(lookup=lambda *_args, **_kwargs: _Lookup()))
    monkeypatch.setattr(modal_batches, "batches_submitter", lambda *args, **kwargs: None)
    monkeypatch.setattr(modal_batches, "status", lambda: None)
    monkeypatch.setattr(modal_batches, "collect", lambda: None)
    monkeypatch.setattr(modal_batches, "clean_up", lambda: None)
    modal_batches.batches("work", batch_tag=None, num=1)
    modal_batches.batches("submit-missing", batch_tag="x", num=None)


def test_kiss_cov_fit_mnist_main_entry(monkeypatch):
    import torchvision.datasets as tv_datasets
    from torch.utils.data import Dataset

    class _TinyMNIST(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return torch.zeros((1, 28, 28), dtype=torch.float32), torch.tensor(idx % 10, dtype=torch.long)

    monkeypatch.setattr(tv_datasets, "MNIST", lambda *args, **kwargs: _TinyMNIST())
    monkeypatch.setattr(sys, "argv", ["ops.fit_mnist"])
    with pytest.raises(SystemExit) as ex:
        runpy.run_module("ops.fit_mnist", run_name="__main__")
    assert ex.value.code == 0
