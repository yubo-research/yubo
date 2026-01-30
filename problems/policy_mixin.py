import copy

import numpy as np
import torch


class PolicyParamsMixin:
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_params(self):
        with torch.inference_mode():
            flat_params = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])
        return (flat_params - self._flat_params_init) / self._const_scale

    def set_params(self, flat_params):
        assert flat_params.min() >= -1 and flat_params.max() <= 1, (
            flat_params.min(),
            flat_params.max(),
        )
        assert flat_params.shape == self._flat_params_init.shape, (
            flat_params.shape,
            self._flat_params_init.shape,
        )
        flat_params = self._flat_params_init + flat_params * self._const_scale
        with torch.inference_mode():
            idx = 0
            for p in self.parameters():
                shape = p.shape
                size = p.numel()
                p.copy_(torch.from_numpy(flat_params[idx : idx + size].reshape(shape)).float())
                idx += size

    def clone(self):
        p = copy.deepcopy(self)
        if hasattr(p, "reset_state"):
            p.reset_state()
        return p
