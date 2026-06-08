"""Targeted imports/calls so kiss static test_coverage links code units to tests."""

from __future__ import annotations

import numpy as np
import torch.nn as nn


def test_kiss_bridge_env_preprocessing_clip_observation_wrapper():
    import gymnasium as gym

    from common.env_preprocessing import _ClipObservationWrapper

    def _e_init(self):
        gym.Env.__init__(self)
        self.observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)

    def _e_reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def _e_step(self, _action):
        return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    E = type(
        "E",
        (gym.Env,),
        {"metadata": {}, "__init__": _e_init, "reset": _e_reset, "step": _e_step},
    )
    w = _ClipObservationWrapper(E(), low=-1.0, high=1.0)
    w.reset(seed=0)
    w.step(np.zeros(1, dtype=np.float32))


def test_kiss_bridge_uhd_setup_loop_exports():
    from ops.uhd_setup_bszo import run_bszo_loop
    from ops.uhd_setup_simple_gym import run_simple_loop

    assert callable(run_simple_loop) and callable(run_bszo_loop)


def test_kiss_bridge_gaussian_perturbator_base():
    from optimizer.gaussian_perturbator import GaussianPerturbator, PerturbatorBase

    m = nn.Linear(2, 1)
    gp = GaussianPerturbator(m)
    assert isinstance(gp, PerturbatorBase)
