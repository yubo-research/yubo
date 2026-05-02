from __future__ import annotations

import torch
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder
from optimizer.gaussian_perturbator import GaussianPerturbator


def linear_module_gaussian_embedder():
    module = nn.Linear(3, 2, bias=True)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bounds = torch.zeros(2, 3)
    bounds[1] = 1.0
    embedder = BehavioralEmbedder(bounds, num_probes=4, seed=0)
    return module, dim, gp, embedder


def run_many_random_ask_tell(uhd, module, *, n: int) -> None:
    for _ in range(n):
        uhd.ask()
        mu = float(torch.randn(1).item())
        uhd.tell(mu, 0.0)
    assert torch.isfinite(module.weight.data).all()
