import torch
from torch import nn

from policies.policy_mixin import PolicyParamsMixin


class _DummySpace:
    shape = (1,)


class _DummyEnv:
    observation_space = _DummySpace()
    action_space = _DummySpace()

    def step(self, _action):
        return None, 1.0, False, False, {}

    def close(self):
        return None


class _NoisePolicy(nn.Module, PolicyParamsMixin):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(2, 1, bias=False)
        with torch.inference_mode():
            fp = torch.cat([p.detach().reshape(-1) for p in self.parameters()]).cpu().numpy()
        self._flat_params_init = fp
        self._const_scale = 1.0
