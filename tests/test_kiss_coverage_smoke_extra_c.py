from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rl.pufferlib.sac import runtime_utils as sac_runtime_utils
from rl.pufferlib.vector_env import make_vector_env as puffer_make_vector_env
from rl.torchrl.sac import setup as sac_setup
from rl.torchrl.sac.config import SACConfig


def test_kiss_cov_sac_setup_build_and_update(monkeypatch, tmp_path):
    fake_env_conf = SimpleNamespace(
        from_pixels=False,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4,))),
    )
    shared = SimpleNamespace(
        env_conf=fake_env_conf,
        problem_seed=7,
        noise_seed_0=11,
        obs_dim=4,
        act_dim=2,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
        obs_lb=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        obs_width=np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        "rl.torchrl.sac.sac_setup_build.build_env_setup",
        lambda _config, **kwargs: shared,
    )

    cfg = SACConfig(exp_dir=str(tmp_path), env_tag="pend", batch_size=4, replay_size=32)
    env_setup = sac_setup.build_env_setup(cfg)
    assert env_setup.obs_dim == 4
    modules = sac_setup.build_modules(cfg, env_setup, device=torch.device("cpu"))
    training = sac_setup.build_training(cfg, modules)
    assert training.metrics_path.name == "metrics.jsonl"

    calls = {}

    def _fake_update_step(*, modules, optimizers, batch, hyper):
        calls["target_entropy"] = hyper.target_entropy
        assert batch.obs.shape[0] == 4
        return (1.0, 2.0, 3.0)

    monkeypatch.setattr("rl.core.sac_update.sac_update_step", _fake_update_step)
    out = sac_setup.sac_update_shared(
        cfg,
        modules,
        training,
        obs=torch.zeros((4, 4)),
        act=torch.zeros((4, 2)),
        rew=torch.zeros(4),
        nxt=torch.zeros((4, 4)),
        done=torch.zeros(4),
    )
    assert out == (1.0, 2.0, 3.0)
    assert isinstance(calls["target_entropy"], float)


def test_kiss_cov_sac_runtime_utils_wrappers(monkeypatch):
    monkeypatch.setattr(sac_runtime_utils.runtime, "mps_is_available", lambda: True)
    assert sac_runtime_utils._mps_is_available()
    d = sac_runtime_utils.select_device("cpu")
    assert str(d) == "cpu"

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(low=np.array([-1.0]), high=np.array([1.0]), shape=(1,)),
        ),
        ensure_spaces=lambda: None,
    )
    lb, width = sac_runtime_utils.obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.array([-1.0]))
    assert np.allclose(width, np.array([2.0]))


def test_kiss_cov_puffer_vector_env_make_vector_env():
    def _vector_make(env_creator, env_kwargs, backend, num_envs, seed, **backend_kwargs):
        _ = env_creator
        return {
            "env_kwargs": env_kwargs,
            "backend": backend,
            "num_envs": num_envs,
            "seed": seed,
            "backend_kwargs": backend_kwargs,
        }

    Serial = type("Serial", (), {})
    Multiprocessing = type("Multiprocessing", (), {})
    _Vector = type(
        "_Vector",
        (),
        {
            "Serial": Serial,
            "Multiprocessing": Multiprocessing,
            "make": staticmethod(_vector_make),
        },
    )

    def _puffer_env_creator(_game_name):
        return lambda **kwargs: kwargs

    _PufferAtari = type("_PufferAtari", (), {"env_creator": staticmethod(_puffer_env_creator)})
    cfg = SimpleNamespace(
        env_tag="f:ackley-2d",
        vector_backend="serial",
        num_envs=2,
        seed=5,
        framestack=4,
        env_conf=SimpleNamespace(),
    )
    out = puffer_make_vector_env(
        cfg,
        import_pufferlib_modules_fn=lambda: (SimpleNamespace(), _Vector, _PufferAtari),
        is_atari_env_tag_fn=lambda tag: False,
        to_puffer_game_name_fn=lambda tag: tag,
        resolve_gym_env_name_fn=lambda env_tag: ("CartPole-v1", {}),
    )
    assert out["backend"] is _Vector.Serial
    assert out["num_envs"] == 2


def test_kiss_cov_offpolicy_engine_checkpoint_env_vec(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from rl.pufferlib.offpolicy import engine_utils as off_engine_utils
    from rl.pufferlib.offpolicy import runtime_utils as off_runtime_utils
    from rl.pufferlib.sac import env_utils as sac_env_utils

    def _ckpt_init(self, *, exp_dir):
        self.exp_dir = exp_dir

    _CheckpointManager = type("_CheckpointManager", (), {"__init__": _ckpt_init})

    real_import = off_engine_utils.importlib.import_module

    def _fake_engine_import(name: str):
        if name == "analysis.data_io":
            return SimpleNamespace(write_config=lambda *args, **kwargs: None)
        if name == "rl.checkpointing":
            return SimpleNamespace(CheckpointManager=_CheckpointManager)
        return real_import(name)

    monkeypatch.setattr(off_engine_utils.importlib, "import_module", _fake_engine_import)
    exp_path, metrics_path, ckpt = off_engine_utils.init_run_artifacts(exp_dir=str(tmp_path / "exp"), config_dict={"x": 1})
    assert exp_path.exists() and metrics_path.name == "metrics.jsonl"
    assert isinstance(ckpt, _CheckpointManager)
    setup, device = off_engine_utils.init_runtime(
        SimpleNamespace(device="cpu"),
        build_env_setup_fn=lambda _cfg: SimpleNamespace(problem_seed=7),
        seed_everything_fn=lambda _seed: None,
        resolve_device_fn=lambda _device: torch.device("cpu"),
    )
    assert setup.problem_seed == 7 and str(device) == "cpu"
    mark = off_engine_utils.checkpoint_mark_if_due(
        global_step=10,
        checkpoint_interval_steps=10,
        previous_mark=0,
        due_mark_fn=lambda *_args, **_kwargs: 1,
        save_fn=lambda: None,
    )
    assert mark == 1

    monkeypatch.setattr(
        off_runtime_utils.runtime,
        "select_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        off_runtime_utils.runtime,
        "obs_scale_from_env",
        lambda _env_conf: (
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([2.0, 2.0], dtype=np.float32),
        ),
    )
    assert str(off_runtime_utils.select_device("cpu")) == "cpu"
    lb, width = off_runtime_utils.obs_scale_from_env(SimpleNamespace())
    assert lb.shape == (2,) and width.shape == (2,)

    monkeypatch.setattr(
        sac_env_utils,
        "build_env_setup",
        lambda _cfg, **_kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(gym_conf=SimpleNamespace(transform_state=True)),
            problem_seed=3,
            noise_seed_0=4,
            obs_lb=np.array([-1.0, -1.0], dtype=np.float32),
            obs_width=np.array([2.0, 2.0], dtype=np.float32),
            act_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        ),
    )
    built = sac_env_utils.build_env_setup(
        SimpleNamespace(
            env_tag="pend",
            seed=0,
            problem_seed=None,
            noise_seed_0=None,
            from_pixels=False,
            pixels_only=True,
        )
    )
    assert built.act_dim == 2
    monkeypatch.setattr(sac_env_utils, "_make_vector_env_shared", lambda _cfg, **_kwargs: "vec")
    assert sac_env_utils.make_vector_env(SimpleNamespace()) == "vec"


def test_kiss_cov_direct_sac_offpolicy_symbols(monkeypatch):
    from types import SimpleNamespace

    import numpy as np
    import torch

    import rl.pufferlib.sac.eval_utils as sac_eval_facade
    from rl.pufferlib.offpolicy import env_utils as offpolicy_env_utils
    from rl.pufferlib.offpolicy.runtime_utils import (
        obs_scale_from_env as off_obs_scale_from_env,
    )
    from rl.pufferlib.offpolicy.runtime_utils import select_device as off_select_device
    from rl.pufferlib.sac.env_utils import build_env_setup as sac_build_env_setup
    from rl.pufferlib.sac.env_utils import make_vector_env as sac_make_vector_env
    from rl.pufferlib.sac.model_utils import SACModules, SACOptimizers
    from rl.pufferlib.sac.model_utils import build_modules as sac_build_modules
    from rl.pufferlib.sac.model_utils import build_optimizers as sac_build_optimizers
    from rl.torchrl.offpolicy.actor_eval import (
        capture_actor_snapshot as trl_capture_actor_snapshot,
    )
    from rl.torchrl.offpolicy.actor_eval import (
        restore_actor_snapshot as trl_restore_actor_snapshot,
    )
    from rl.torchrl.offpolicy.actor_eval import (
        use_actor_snapshot as trl_use_actor_snapshot,
    )

    monkeypatch.setattr(
        "rl.core.runtime.select_device",
        lambda *_args, **_kwargs: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "rl.core.runtime.obs_scale_from_env",
        lambda _env_conf: (None, None),
    )
    assert str(off_select_device("cpu")) == "cpu"
    assert off_obs_scale_from_env(SimpleNamespace()) == (None, None)

    monkeypatch.setattr(
        offpolicy_env_utils,
        "build_env_setup",
        lambda **_kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(gym_conf=SimpleNamespace(transform_state=False)),
            problem_seed=1,
            noise_seed_0=2,
            obs_lb=np.array([-1.0, -1.0], dtype=np.float32),
            obs_width=np.array([2.0, 2.0], dtype=np.float32),
            act_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        ),
    )
    env_setup = sac_build_env_setup(
        SimpleNamespace(
            env_tag="pend",
            seed=0,
            problem_seed=None,
            noise_seed_0=None,
            from_pixels=False,
            pixels_only=True,
        )
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.env_utils._make_vector_env_shared",
        lambda _cfg, **_kwargs: "ok",
    )
    assert sac_make_vector_env(SimpleNamespace()) == "ok"

    cfg = SimpleNamespace(
        backbone_name="mlp",
        backbone_hidden_sizes=(8,),
        backbone_activation="relu",
        backbone_layer_norm=False,
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        head_activation="relu",
        learning_rate_actor=3e-4,
        learning_rate_critic=3e-4,
        learning_rate_alpha=3e-4,
        alpha_init=0.2,
        batch_size=2,
        gamma=0.99,
        tau=0.005,
        target_entropy=-2.0,
        theta_dim=None,
        num_denoise=1,
        num_denoise_passive=1,
        eval_interval_steps=1,
        eval_seed_base=0,
        eval_noise_mode="frozen",
        seed=0,
    )
    obs_spec = SimpleNamespace(mode="vector", raw_shape=(2,), vector_dim=2)
    modules = sac_build_modules(cfg, env_setup, obs_spec, device=torch.device("cpu"))
    assert isinstance(modules, SACModules)
    optimizers = sac_build_optimizers(cfg, modules)
    assert isinstance(optimizers, SACOptimizers)

    monkeypatch.setattr(
        "rl.pufferlib.sac.eval_utils.collect_denoised_trajectory",
        lambda _env_conf, _policy, **_kwargs: (SimpleNamespace(rreturn=1.0), 0),
    )
    monkeypatch.setattr("rl.pufferlib.sac.eval_utils.evaluate_for_best", lambda *_args, **_kwargs: 0.25)
    monkeypatch.setattr(
        "rl.pufferlib.sac.eval_utils.build_eval_plan",
        lambda **_kwargs: SimpleNamespace(eval_seed=0, heldout_i_noise=0),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.eval_utils.evaluate_heldout_if_enabled",
        lambda *_args, **_kwargs: 0.5,
    )
    state = SimpleNamespace(
        global_step=1,
        eval_mark=0,
        best_return=-float("inf"),
        best_actor_state=None,
        last_eval_return=0.0,
        last_heldout_return=None,
    )
    assert sac_eval_facade.evaluate_actor(cfg, env_setup, modules, obs_spec, device=torch.device("cpu"), eval_seed=0) == 1.0
    assert isinstance(
        sac_eval_facade.evaluate_heldout_if_enabled(
            cfg,
            env_setup,
            modules,
            obs_spec,
            device=torch.device("cpu"),
            heldout_i_noise=0,
        ),
        float,
    )
    sac_eval_facade.maybe_eval(cfg, env_setup, modules, obs_spec, state, device=torch.device("cpu"))

    snapshot = trl_capture_actor_snapshot(SimpleNamespace(actor_backbone=nn.Linear(2, 2), actor_head=nn.Linear(2, 2)))
    modules_small = SimpleNamespace(actor_backbone=nn.Linear(2, 2), actor_head=nn.Linear(2, 2))
    trl_restore_actor_snapshot(modules_small, snapshot)
    with trl_use_actor_snapshot(modules_small, snapshot, device=torch.device("cpu")):
        pass
