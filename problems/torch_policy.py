import numpy as np
import torch
from torch import nn

from .normalizer import Normalizer


class TorchPolicy:
    def __init__(self, module: nn.Module, env_conf):
        self.module = module
        self.problem_seed = env_conf.problem_seed

        if env_conf.gym_conf is not None and env_conf.gym_conf.transform_state:
            num_state = env_conf.gym_conf.state_space.shape[0]
            self._normalizer = Normalizer(shape=(num_state,))
        else:
            self._normalizer = None

        self._clamp = env_conf.gym_conf is not None

    def __call__(self, state):
        if self._normalizer is not None:
            state = np.asarray(state, dtype=np.float32)
            self._normalizer.update(state)
            mean, std = self._normalizer.mean_and_std()
            std = np.where(std == 0, 1.0, std)
            state = (state - mean) / std

        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            action_t = self.module(state_t)
        if self._clamp:
            action_t = torch.clamp(action_t, -1, 1)
        return action_t.detach().cpu().numpy()
