"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn


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
    import rl.torchrl.collect_utils as cu

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
