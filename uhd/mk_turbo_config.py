from typing import Any, List

import numpy as np
import torch
import torch.nn as nn

from sampling.sampling_util import gumbel
from uhd.opt_turbo import UHDBOConfig
from uhd.trust_region import TrustRegionAdjustor


def _compute_signal_length_from_y(ys: List[float]) -> float:
    if len(ys) <= 1:
        signal = 0.0
    else:
        y_np = np.asarray(ys, dtype=float)
        denom = 2.0 * gumbel(len(y_np))
        denom = denom if denom > 0 else 1.0
        signal = ((y_np.max() - np.median(y_np)) / (1e-6 + y_np.std()) / denom) ** 2
    scale = 1.0 / (1e-6 + signal)
    s_min = 0.1
    s_max = 1.0
    length = float(np.minimum(s_max, np.maximum(s_min, scale)))
    return length


def mk_turbo_config(
    use_tr: bool,
    num_raasp: float,
    num_trail: int = 32,
    num_candidates: int = 1,
    alpha: float = 1.0,
) -> UHDBOConfig:
    assert isinstance(use_tr, bool)
    assert isinstance(num_raasp, (int, float)) and num_raasp >= -1.0
    assert isinstance(num_trail, int) and num_trail > 0
    assert isinstance(num_candidates, int) and num_candidates > 0
    assert isinstance(alpha, (int, float))
    alpha = float(alpha)

    class _Metric:
        def measure(self, controller: nn.Module) -> float:
            return float(controller().detach().item())

    class _Embedder:
        def embed(self, params: torch.Tensor) -> torch.Tensor:
            return torch.tensor(0.0)

    class _Perturber:
        def __init__(self, use_tr: bool, num_raasp: float, alpha: float = 1.0) -> None:
            self.use_tr = use_tr
            self.num_raasp_raw = float(num_raasp)
            assert isinstance(alpha, (int, float))
            self.alpha = float(alpha)
            self._backup = None
            self._tr = None
            self._batch_size = 1

        def perturb(self, target: Any, ys: List[float]) -> Any:
            assert self._backup is None
            if hasattr(target, "num_dims"):
                num_dim = int(target.num_dims)
            else:
                num_dim = int(target.numel())
            if self.num_raasp_raw < 0.0:
                k = int(-self.num_raasp_raw * num_dim)
            else:
                k = int(self.num_raasp_raw)
            k = min(max(1, k), num_dim)
            lb = target.lb
            ub = target.ub
            width = ub - lb
            device = target.device
            dtype = target.dtype
            idx = torch.randperm(num_dim, device=device)[:k]

            orig_vals = target.gather(idx).clone()

            if self.use_tr:
                if self._tr is None:
                    self._tr = TrustRegionAdjustor(
                        dim=num_dim, batch_size=self._batch_size
                    )
                half_tr = 1e-3  # 0.5 * float(self._tr.length)
                print("HALF_TR:", half_tr)
                u01 = torch.rand(k, device=device, dtype=dtype)
                step = (2.0 * u01 - 1.0) * half_tr * width
                new_vals = orig_vals + step
            else:
                new_vals = lb + width * torch.rand(k, device=device, dtype=dtype)

            target.scatter_(idx, new_vals)
            target.clamp_(lb, ub)
            self._backup = {
                "indices": idx.clone(),
                "orig": orig_vals,
            }

            return target

        def unperturb(self, target: Any) -> None:
            assert self._backup is not None
            indices = self._backup["indices"]
            orig_vals = self._backup["orig"]
            with torch.no_grad():
                target.scatter_(indices, orig_vals)
                target.clamp_(target.lb, target.ub)
            self._backup = None

        def tr_update(self, y: float) -> None:
            if self.use_tr and self._tr is not None:
                self._tr.update([float(y)])

        def incorporate(self, target: Any) -> None:
            assert self._backup is not None
            indices = self._backup["indices"]
            orig_vals = self._backup["orig"]
            if self.alpha == 1.0:
                self._backup = None
                return
            current = target.gather(indices)
            new_vals = orig_vals + self.alpha * (current - orig_vals)
            with torch.no_grad():
                target.scatter_(indices, new_vals)
                target.clamp_(target.lb, target.ub)
            self._backup = None

    class _Selector:
        def select(self, embeddings: List[torch.Tensor]) -> int:
            return 0

    return UHDBOConfig(
        num_candidates=num_candidates,
        perturber=_Perturber(use_tr=use_tr, num_raasp=num_raasp, alpha=alpha),
        embedder=_Embedder(),
        selector=_Selector(),
        metric=_Metric(),
        num_trail=num_trail,
    )
