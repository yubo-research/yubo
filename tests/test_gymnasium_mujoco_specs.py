from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pytest

from tests.test_mjx_env_adapter import _fake_halfcheetah_env


def test_halfcheetah_gymnasium_oracle_direct_step_and_close(monkeypatch) -> None:
    from problems.gymnasium_mujoco_specs import _HalfCheetahGymnasiumOracle

    closed = []
    env = _fake_halfcheetah_env()
    env.close = lambda: closed.append(True)
    mujoco_mod = ModuleType("mujoco")
    mujoco_mod.mj_forward = lambda _model, _data: None
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_mod)

    oracle = _HalfCheetahGymnasiumOracle(env)
    obs = oracle.obs(np.asarray([1.0, 2.0]), np.asarray([3.0]))
    step = oracle.step(
        np.asarray([1.0, 2.0]),
        np.asarray([3.0]),
        np.asarray([1.5, 2.5]),
        np.asarray([3.5]),
        np.asarray([1.0, -2.0], dtype=np.float32),
    )
    oracle.close()

    np.testing.assert_array_equal(obs, np.asarray([2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(step[0], np.asarray([2.5, 3.5], dtype=np.float32))
    assert step[1] == pytest.approx(9.5)
    assert step[2] == np.asarray(False, dtype=bool)
    assert step[3] == np.asarray(False, dtype=bool)
    assert step[4][0] == pytest.approx(1.5)
    assert step[4][1] == pytest.approx(10.0)
    assert closed == [True]
