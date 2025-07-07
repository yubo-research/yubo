import copy

import numpy as np
import torch
import torch.nn as nn

from .normalizer import Normalizer


class MLPPolicyFactory:
    def __init__(self, hidden_sizes):
        if not isinstance(hidden_sizes, (list, tuple)) or not all(isinstance(x, int) and x > 0 for x in hidden_sizes):
            raise ValueError("hidden_sizes must be a sequence of positive integers")
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
            linear = nn.Linear(dims[i], dims[i + 1], bias=True)
            layers.append(linear)
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            layers.append(nn.ReLU())

        linear = nn.Linear(dims[-2], dims[-1], bias=True)
        layers.append(linear)
        nn.init.uniform_(linear.weight, a=-3e-3, b=3e-3)
        nn.init.zeros_(linear.bias)
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)
        self._normalizer = Normalizer(shape=(num_state,))

    def _normalize(self, state):
        state = state.copy()
        self._normalizer.update(state)
        loc, scale = self._normalizer.mean_and_std()

        zero_scale_mask = scale == 0
        state = state - loc
        scale[zero_scale_mask] = 1
        state[zero_scale_mask] = 0.0
        state = state / scale
        return state

    def forward(self, x):
        return self.model(x)

    def __call__(self, state):
        state = self._normalize(state)
        state_tensor = torch.from_numpy(state).float().to(self.device)

        with torch.inference_mode():
            action = self.forward(state_tensor)
        return action.cpu().numpy()

    @property
    def device(self):
        return next(self.parameters()).device

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_params(self):
        with torch.inference_mode():
            params = []
            for p in self.parameters():
                params.append(p.data.cpu().numpy().flatten())
            return np.concatenate(params)

    def set_params(self, flat_params):
        with torch.inference_mode():
            idx = 0
            for p in self.parameters():
                shape = p.shape
                size = p.numel()
                param_data = torch.from_numpy(flat_params[idx : idx + size].reshape(shape)).float()
                p.copy_(param_data.to(p.device))
                idx += size

    def clone(self):
        return copy.deepcopy(self)
