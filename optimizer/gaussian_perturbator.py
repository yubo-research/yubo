import torch
from torch import nn


def apply_noise_inplace(module: nn.Module, noise: list[torch.Tensor], *, sign: int) -> None:
    for param, n in zip(module.parameters(), noise, strict=True):
        if sign > 0:
            param.data.add_(n)
        else:
            param.data.sub_(n)


def mark_perturbed(state_obj, *, seed: int, sigma: float) -> None:
    state_obj._seed = seed
    state_obj._sigma = sigma
    state_obj._perturbed = True


class GaussianPerturbator:
    def __init__(self, module: nn.Module):
        self._module = module
        self._perturbed = False
        self._seed: int | None = None

    def _generate_noise(self, sigma: float) -> list[torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self._seed)
        noise = []
        for param in self._module.parameters():
            n = torch.randn(param.shape, generator=rng) * sigma
            noise.append(n.to(param.device))
        return noise

    def perturb(self, seed: int, sigma: float) -> None:
        assert not self._perturbed, "Already perturbed"
        mark_perturbed(self, seed=seed, sigma=sigma)
        apply_noise_inplace(self._module, self._generate_noise(sigma), sign=1)

    def accept(self) -> None:
        assert self._perturbed, "Not perturbed"
        self._perturbed = False
        self._seed = None

    def unperturb(self) -> None:
        assert self._perturbed, "Not perturbed"
        apply_noise_inplace(self._module, self._generate_noise(self._sigma), sign=-1)
        self.accept()
