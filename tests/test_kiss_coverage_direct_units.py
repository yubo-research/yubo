from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch


def test_kiss_cov_direct_exp_uhd_modal_and_runner_main(monkeypatch, tmp_path):
    from ops.exp_uhd import modal_cmd
    from rl.runner import main

    monkeypatch.setattr("ops.modal_uhd.run", lambda *args, **kwargs: "ok")
    toml = tmp_path / "cfg.toml"
    toml.write_text('[uhd]\nenv_tag="f:sphere-2d"\nnum_rounds=1\n')
    modal_cmd(str(toml), (), None, "A100")

    cfg_path = tmp_path / "rl.toml"
    cfg_path.write_text('[rl]\nalgo="dummy"\n[rl.dummy]\nseed=7\nexp_dir="tmp"\n')
    monkeypatch.setattr("rl.runner.split_config_and_args", lambda argv: (str(cfg_path), argv[1:]))
    monkeypatch.setattr("rl.runner.parse_runtime_args", lambda rest: SimpleNamespace(workers=1, workers_cli_set=False, cleaned=rest))
    monkeypatch.setattr("rl.runner.parse_set_args", lambda cleaned: {})
    monkeypatch.setattr("rl.runner.load_toml", lambda _path: {"rl": {"algo": "dummy", "dummy": {"seed": 7, "exp_dir": "tmp"}}})
    monkeypatch.setattr("rl.runner.apply_overrides", lambda cfg, overrides: None)
    monkeypatch.setattr("rl.runner._extract_run_cfg", lambda cfg: ([], 1))
    monkeypatch.setattr("rl.runner.resolve_algo_name", lambda algo, backend=None: algo)

    class _Cfg:
        seed = 7
        exp_dir = "tmp"
        problem_seed = None
        noise_seed_0 = None

        @classmethod
        def from_dict(cls, _d):
            return cls()

    monkeypatch.setattr("rl.runner.get_algo", lambda _algo_name, backend=None: SimpleNamespace(config_cls=_Cfg, train_fn=lambda _cfg: {"ok": True}))
    monkeypatch.setattr("rl.builtins.register_all", lambda: None)
    main(["--config", "config.toml"])


def test_kiss_cov_direct_rl_core_units(monkeypatch):
    from optimizer.gaussian_perturbator import PerturbatorBase
    from rl.core.actor_state import build_ppo_checkpoint_payload, capture_ppo_actor_snapshot, restore_rng_state_payload, rng_state_payload
    from rl.core.env_conf import ResolvedSeeds, SeededEnvConf, build_seeded_env_conf, build_seeded_env_conf_from_run
    from rl.core.env_setup import ContinuousGymEnvSetup, build_continuous_gym_env_setup
    from rl.core.replay import NumpyReplayBuffer
    from rl.core.runtime import mps_is_available, obs_scale_from_env, seed_everything
    from rl.runner_helpers import _RuntimeArgs

    class _P(PerturbatorBase):
        def _apply(self, *, seed: int, sigma: float, chunk_size: int = 2**16) -> None:
            _ = (seed, sigma, chunk_size)

    module = torch.nn.Linear(2, 1)
    p = _P(module)
    p.perturb(0, 0.1)
    p.unperturb()

    b = torch.nn.Linear(2, 2)
    h = torch.nn.Linear(2, 1)
    snap = capture_ppo_actor_snapshot(b, h, log_std=None)
    payload = build_ppo_checkpoint_payload(
        iteration=1,
        global_step=2,
        actor_snapshot=snap,
        critic_backbone={},
        critic_head={},
        optimizer={},
        best_actor_state=None,
        best_return=1.0,
        last_eval_return=1.0,
        last_heldout_return=None,
    )
    assert payload["iteration"] == 1
    restore_rng_state_payload(rng_state_payload())

    seeded = build_seeded_env_conf(
        env_tag="x",
        problem_seed=3,
        noise_seed_0=4,
        from_pixels=False,
        pixels_only=True,
        get_env_conf_fn=lambda *args, **kwargs: SimpleNamespace(ok=True),
    )
    assert isinstance(seeded, SeededEnvConf)
    assert isinstance(ResolvedSeeds(problem_seed=1, noise_seed_0=2), ResolvedSeeds)
    seeded2 = build_seeded_env_conf_from_run(
        env_tag="x",
        seed=1,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        get_env_conf_fn=lambda *args, **kwargs: SimpleNamespace(ok=True),
    )
    assert isinstance(seeded2, SeededEnvConf)

    env_setup = build_continuous_gym_env_setup(
        env_tag="x",
        seed=1,
        problem_seed=None,
        noise_seed_0=None,
        from_pixels=False,
        pixels_only=True,
        get_env_conf_fn=lambda *args, **kwargs: SimpleNamespace(
            ensure_spaces=lambda: None,
            action_space=SimpleNamespace(shape=(2,), low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])),
        ),
        obs_scale_from_env_fn=lambda env_conf: (None, None),
    )
    assert isinstance(env_setup, ContinuousGymEnvSetup)

    rb = NumpyReplayBuffer(obs_shape=(3,), act_dim=2, capacity=8)
    rb.add_batch(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.zeros((4, 3), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )
    _ = rb.sample(2, "cpu")

    _ = mps_is_available()
    seed_everything(1, cuda_is_available_fn=lambda: False)
    _ = obs_scale_from_env(
        SimpleNamespace(
            gym_conf=SimpleNamespace(transform_state=True, state_space=SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,))),
            ensure_spaces=lambda: None,
        )
    )
    assert isinstance(_RuntimeArgs(None, 1, False, []), _RuntimeArgs)


def test_kiss_cov_direct_eval_config_and_uhd_setup(monkeypatch):
    from ops.uhd_setup import run_bszo_loop, run_simple_loop
    from rl.pufferlib.ppo.eval_config import build_eval_env_conf, resolve_eval_seeds

    monkeypatch.setattr("rl.pufferlib.ppo.eval_config.resolve_run_seeds", lambda **kwargs: SimpleNamespace(problem_seed=3, noise_seed_0=4))
    assert resolve_eval_seeds(SimpleNamespace(seed=1, problem_seed=None, noise_seed_0=None)) == (3, 4)
    monkeypatch.setattr(
        "rl.pufferlib.ppo.eval_config.build_seeded_env_conf_from_run",
        lambda **kwargs: SimpleNamespace(env_conf=SimpleNamespace(gym_conf=None), problem_seed=5, noise_seed_0=6),
    )
    cfg = SimpleNamespace(env_tag="x", seed=1, problem_seed=None, noise_seed_0=None, pixels_only=False)
    got = build_eval_env_conf(cfg, obs_mode="vector", is_atari_env_tag_fn=lambda tag: False, resolve_gym_env_name_fn=lambda tag: ("CartPole-v1", {}))
    assert got.env_name == "CartPole-v1"

    fake_env_conf = SimpleNamespace(
        problem_seed=1,
        noise_seed_0=2,
        make=lambda: SimpleNamespace(torch_env=lambda: None),
        make_torch_env=lambda: SimpleNamespace(torch_env=lambda: SimpleNamespace(module=torch.nn.Linear(2, 1))),
    )
    monkeypatch.setattr("common.seed_all.seed_all", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("problems.env_conf.get_env_conf", lambda *args, **kwargs: fake_env_conf)
    called = {"simple": 0, "bszo": 0}
    monkeypatch.setattr("ops.uhd_setup._run_simple_torch", lambda *args, **kwargs: called.__setitem__("simple", called["simple"] + 1))
    monkeypatch.setattr("ops.uhd_setup._run_simple_gym", lambda *args, **kwargs: called.__setitem__("simple", called["simple"] + 1))
    run_simple_loop("x", 1)
    monkeypatch.setattr("ops.uhd_setup._run_bszo_iterations", lambda *args, **kwargs: called.__setitem__("bszo", called["bszo"] + 1))
    monkeypatch.setattr(
        "optimizer.uhd_bszo.UHDBSZO", lambda *args, **kwargs: SimpleNamespace(k=1, ask=lambda: None, tell=lambda mu, se: None, eval_seed=0, y_best=None)
    )
    monkeypatch.setattr("optimizer.lr_scheduler.ConstantLR", lambda lr: lr)
    monkeypatch.setattr("ops.uhd_setup._get_device", lambda: torch.device("cpu"))
    run_bszo_loop("x", 1)
    assert called["simple"] >= 1
    assert called["bszo"] >= 1


def test_kiss_cov_direct_fit_main_episode_rollout_backbone_and_turbo(monkeypatch):
    from ops.fit_mnist import main as fit_main
    from rl.backbone import init_linear_layers
    from rl.core.episode_rollout import (
        MeanReturnResult,
        Trajectory,
        collect_denoised_trajectory,
        collect_trajectory_with_noise,
        evaluate_for_best,
        mean_return_over_runs,
    )
    from turbo_m_ref.utils import make_sobol_candidates

    called = {"fit": 0}
    monkeypatch.setattr("ops.fit_mnist.fit_mnist", lambda **kwargs: called.__setitem__("fit", called["fit"] + 1))
    fit_main(1, 8, 1e-3, 1)
    assert called["fit"] == 1

    lin = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    init_linear_layers(lin, gain=0.5)

    class _Conf:
        noise_seed_0 = 10
        frozen_noise = False
        gym_conf = SimpleNamespace(max_steps=2)

        @staticmethod
        def make():
            class _Env:
                action_space = SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]))

                def __init__(self):
                    self.n = 0

                def reset(self, seed=None):
                    return np.zeros(1), {}

                def step(self, action):
                    self.n += 1
                    return np.zeros(1), 1.0, self.n > 1, False, {}

                def close(self):
                    return None

            return _Env()

    def policy(_obs):
        return np.array([0.0])

    tr, ns = collect_trajectory_with_noise(_Conf(), policy, i_noise=1, denoise_seed=2)
    assert isinstance(tr, Trajectory)
    assert isinstance(ns, int)
    mr = mean_return_over_runs(_Conf(), policy, 2, i_noise=1)
    assert isinstance(mr, MeanReturnResult)
    den, _ = collect_denoised_trajectory(_Conf(), policy, num_denoise=2, i_noise=1)
    assert isinstance(den, Trajectory)
    assert np.isfinite(evaluate_for_best(_Conf(), policy, 2, i_noise=999))

    x = make_sobol_candidates(
        dim=4,
        n_cand=8,
        x_center=np.zeros(4),
        lb=np.zeros(4),
        ub=np.ones(4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert x.shape == (8, 4)


def test_kiss_cov_direct_torchrl_replay_buffer():
    from rl.core.replay import TorchRLReplayBuffer

    rb = TorchRLReplayBuffer(obs_shape=(3,), act_dim=2, capacity=16)
    rb.add_batch(
        obs=np.zeros((8, 3), dtype=np.float32),
        act=np.zeros((8, 2), dtype=np.float32),
        rew=np.zeros(8, dtype=np.float32),
        nxt=np.zeros((8, 3), dtype=np.float32),
        done=np.zeros(8, dtype=np.float32),
    )
    obs, act, rew, nxt, done = rb.sample(4, "cpu")
    assert obs.shape[0] == 4
    state = rb.state_dict()
    rb.load_state_dict(state)
    assert done.shape[0] == 4
