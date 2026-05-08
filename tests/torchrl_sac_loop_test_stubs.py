from __future__ import annotations

import numpy as np


class _ActionSpaceStub:
    def __init__(self, sample_value):
        self._sample_value = np.asarray(sample_value, dtype=np.float32)

    def sample(self):
        return self._sample_value.copy()


class _TrainEnvStub:
    def __init__(self, *, sample_value, step_result, reset_state):
        self.action_space = _ActionSpaceStub(sample_value)
        self._step_result = step_result
        self._reset_state = np.asarray(reset_state, dtype=np.float32)

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return self._step_result

    def reset(self):
        return self._reset_state.copy(), {}


class _ReplayStub:
    def __init__(self):
        self.items = []

    def add(self, value):
        self.items.append(value)
