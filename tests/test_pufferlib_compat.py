import sys

import gymnasium as gym

from rl.pufferlib_compat import _install_gym_alias


def test_install_gym_alias_registers_gym_module():
    sys.modules.pop("gym", None)
    out = _install_gym_alias()
    assert out is gym
    assert sys.modules.get("gym") is gym


def test_install_gym_alias_adds_framestack_alias_when_missing(monkeypatch):
    class _Wrappers:
        FrameStackObservation = object()

    class _GymLike:
        wrappers = _Wrappers()

    monkeypatch.setitem(sys.modules, "gymnasium", _GymLike)
    monkeypatch.delitem(sys.modules, "gym", raising=False)
    out = _install_gym_alias()
    assert out is _GymLike
    assert hasattr(_GymLike.wrappers, "FrameStack")
