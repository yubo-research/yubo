from typing import Any, List

import torch
import torch.nn as nn

from uhd.opt_turbo import UHDBOConfig


def mk_random_perturbations_config(eps: float, num_trail: int = 32) -> UHDBOConfig:
    assert isinstance(eps, (int, float)) and eps >= 0
    assert isinstance(num_trail, int) and num_trail > 0

    class _Metric:
        def measure(self, controller: nn.Module) -> torch.Tensor:
            return controller().detach()

    class _Embedder:
        def embed(self, params: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

    class _Perturber:
        def __init__(self, eps: float) -> None:
            self.eps = float(eps)
            self._backup = None

        def perturb(self, target: Any, ys) -> None:
            assert self._backup is None
            lb = target.lb
            ub = target.ub
            width = ub - lb
            num_dim = int(target.numel())
            device = target.device
            dtype = target.dtype
            orig = target.clone_flat()
            noise = self.eps * width * torch.randn(num_dim, device=device, dtype=dtype)
            target.add_(noise)
            target.clamp_(float(lb), float(ub))
            self._backup = {
                "orig": orig,
            }

        def unperturb(self, target: Any) -> None:
            assert self._backup is not None
            orig_vals = self._backup["orig"]
            indices = torch.arange(orig_vals.numel(), device=target.device, dtype=torch.long)
            target.scatter_(indices, orig_vals)
            target.clamp_(target.lb, target.ub)
            self._backup = None

        def incorporate(self, target: Any) -> None:
            self._backup = None

    class _Selector:
        def select(self, embeddings: List[torch.Tensor]) -> int:
            return 0

    return UHDBOConfig(
        num_candidates=1,
        perturber=_Perturber(eps),
        embedder=_Embedder(),
        selector=_Selector(),
        metric=_Metric(),
        num_trail=num_trail,
    )
