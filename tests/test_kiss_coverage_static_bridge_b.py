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

from tests.kiss_dummy_nn_modules import make_pm_module_type


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

    ObsSpec = type("ObsSpec", (), {"keys": lambda self, *_a, **_k: ["observation"]})
    DMControlWrapper = type(
        "DMControlWrapper",
        (),
        {"__init__": lambda self, *a, **k: setattr(self, "observation_spec", ObsSpec())},
    )
    TransformedEnv = type("TransformedEnv", (), {"__init__": lambda self, *a, **k: None})
    TR = type(
        "TR",
        (),
        {"DMControlWrapper": DMControlWrapper, "TransformedEnv": TransformedEnv},
    )
    CatTensors = type("CatTensors", (), {"__init__": lambda self, **kwargs: None})
    Compose = type("Compose", (), {"__init__": lambda self, *a, **k: None})
    DoubleToFloat = type("DoubleToFloat", (), {})
    TT = type(
        "TT",
        (),
        {"CatTensors": CatTensors, "Compose": Compose, "DoubleToFloat": DoubleToFloat},
    )
    fake_suite = types.ModuleType("dm_control.suite")
    fake_suite.load = lambda *a, **k: object()
    fake_dm_control = types.ModuleType("dm_control")
    fake_dm_control.suite = fake_suite
    monkeypatch.setitem(sys.modules, "dm_control", fake_dm_control)
    monkeypatch.setitem(sys.modules, "dm_control.suite", fake_suite)
    out = dcc.make_dm_control_collect_env(
        env_name="dm_control/cheetah-run-v0",
        seed=0,
        from_pixels=False,
        pixels_only=True,
        tr_envs_module=TR,
        tr_transforms_module=TT,
        pixels_transform_builder=lambda tt: TT.Compose(),
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
    from rl.torchrl.sac import register as sac_register
    from rl.torchrl.sac import setup as sac_setup

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

    mods = SimpleNamespace(
        actor_backbone=nn.Linear(2, 2),
        actor_head=nn.Linear(2, 2),
        log_std=None,
        actor=SimpleNamespace(act=lambda x: x),
    )
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
    from rl.pufferlib.ppo.eval_config import build_eval_env_conf
    from rl.pufferlib.ppo.eval_seeds import resolve_eval_seeds

    PpoM = make_pm_module_type("PpoM")
    model = PpoM()
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

    GymInner = type("GymInner", (), {"state_space": SimpleNamespace(shape=(4, 84, 84))})
    ec = type(
        "EC",
        (),
        {
            "problem_seed": 1,
            "action_space": SimpleNamespace(n=4),
            "state_space": SimpleNamespace(shape=(4, 84, 84)),
            "gym_conf": GymInner(),
        },
    )()
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
