import torch
from torch import nn


def apply_weight_decay(module, lr: float, weight_decay: float) -> None:
    if weight_decay <= 0.0:
        return
    decay = 1.0 - lr * weight_decay
    for p in module.parameters():
        p.data.mul_(decay)


class PerturbatorBase:
    def __init__(self, module: nn.Module):
        self._module = module
        self._perturbed = False
        self._seed: int | None = None

    def _device(self) -> torch.device:
        return next(self._module.parameters()).device

    def _rng(self, seed: int) -> torch.Generator:
        g = torch.Generator(device=str(self._device()))
        g.manual_seed(int(seed))
        return g

    def _apply(self, *, seed: int, sigma: float, chunk_size: int = 2**16) -> None:
        raise NotImplementedError

    def perturb(self, seed: int, sigma: float) -> None:
        assert not self._perturbed, "Already perturbed"
        self._seed = seed
        self._sigma = sigma
        self._perturbed = True
        self._apply(seed=seed, sigma=sigma)

    def accept(self) -> None:
        assert self._perturbed, "Not perturbed"
        self._perturbed = False
        self._seed = None

    def unperturb(self) -> None:
        assert self._perturbed, "Not perturbed"
        assert self._seed is not None
        self._apply(seed=self._seed, sigma=-self._sigma)
        self.accept()


class GaussianPerturbator(PerturbatorBase):
    def _apply(self, *, seed: int, sigma: float, chunk_size: int = 2**16) -> None:
        g = self._rng(seed)
        device = self._device()
        for p in self._module.parameters():
            assert p.device == device, "GaussianPerturbator requires all params on one device"
            flat = p.data.view(-1)
            n = flat.numel()
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                noise = torch.randn(
                    (end - start,),
                    device=device,
                    dtype=flat.dtype,
                    generator=g,
                )
                noise.mul_(sigma)
                flat[start:end].add_(noise)
