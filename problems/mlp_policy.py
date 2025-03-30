import copy

import numpy as np
import torch
import torch.nn as nn

from .normalizer import Normalizer


class MLPPolicyFactory:
    def __init__(self, hidden_sizes):
        self._hidden_sizes = hidden_sizes

    def __call__(self, env_conf):
        return MLPPolicy(env_conf, self._hidden_sizes)


class MLPPolicy(nn.Module):
    def __init__(self, env_conf, hidden_sizes):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf

        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = env_conf.action_space.shape[0]

        layers = []
        dims = [num_state] + list(hidden_sizes) + [num_action]

        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(dims[-2], dims[-1]))
        layers.append(nn.Tanh())  # to bound action space

        self.model = nn.Sequential(*layers)

        self._normalizer = Normalizer(shape=(num_state,))

    def _normalize(self, state):
        self._normalizer.update(state)
        loc, scale = self._normalizer.mean_and_std()

        i = np.where(scale == 0)[0]
        state = state - loc
        scale[i] = 1
        state[i] = 0.0
        state = state / scale
        return state

    def forward(self, x):
        return self.model(x)

    def __call__(self, state):
        """
        Takes a state (NumPy array or torch.Tensor) and returns an action (NumPy array).
        Applies online normalization to the input.
        """

        state = self._normalize(state)
        state = torch.from_numpy(state).float()

        with torch.inference_mode():
            action = self.forward(state)
        return action.numpy()

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_params(self):
        with torch.inference_mode():
            params = np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])
        return params

    def set_params(self, flat_params):
        with torch.inference_mode():
            idx = 0
            for p in self.parameters():
                shape = p.shape
                size = p.numel()
                p.copy_(torch.from_numpy(flat_params[idx : idx + size].reshape(shape)).float())
                idx += size

    def clone(self):
        return copy.deepcopy(self)
