from __future__ import annotations

import numpy as np
from torchrl_video_helper_stubs import _BoxSpaceStub


class _NonGymRenderEnv:
    def __init__(self):
        self.action_space = _BoxSpaceStub(low=[-1.0, -1.0], high=[1.0, 1.0])
        self.observation_space = _BoxSpaceStub(low=[-1.0, -1.0], high=[1.0, 1.0])
        self._step_count = 0

    def reset(self, *, seed=None):
        _ = seed
        self._step_count = 0
        return np.asarray([0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        _ = action
        self._step_count += 1
        terminated = self._step_count >= 1
        return np.asarray([0.0, 0.0], dtype=np.float32), 1.0, terminated, False, {}

    def render(self):
        pixel = np.uint8(min(255, 40 * (self._step_count + 1)))
        return np.full((8, 8, 3), pixel, dtype=np.uint8)

    def close(self):
        return


class _NonGymEnvConfStub:
    def __init__(self, *, max_steps=2):
        self.gym_conf = None
        self.max_steps = int(max_steps)

    def make(self, **_kwargs):
        return _NonGymRenderEnv()
