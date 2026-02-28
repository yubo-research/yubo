import numpy as np
import torch
import torch.nn as nn


class BehavioralEmbedder:
    def __init__(self, bounds: torch.Tensor, num_probes: int, seed: int = 0):
        lb, ub = bounds[0], bounds[1]
        gen = torch.Generator()
        gen.manual_seed(seed)
        uniform = torch.rand((num_probes, *lb.shape), generator=gen)
        self.probes = lb + uniform * (ub - lb)

    def embed(self, module: nn.Module) -> torch.Tensor:
        device = next(module.parameters()).device
        with torch.inference_mode():
            outputs = module(self.probes.to(device))
        return outputs.reshape(-1)

    def embed_policy(self, policy, x) -> np.ndarray:
        policy.set_params(x)
        if hasattr(policy, "reset_state"):
            policy.reset_state()
        probes_np = self.probes.numpy()
        outputs = [np.atleast_1d(policy(probe)) for probe in probes_np]
        return np.concatenate(outputs).reshape(-1)
