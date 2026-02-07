import torch
from torch import nn


class GaussianPerturbator:
    def __init__(self, module: nn.Module):
        self._module = module
        self._perturbed = False
        self._seed: int | None = None
        self._sigma: float | None = None

    def _generate_noise(self) -> list[torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self._seed)
        noise = []
        for param in self._module.parameters():
            n = torch.randn(param.shape, generator=rng) * self._sigma
            noise.append(n.to(param.device))
        return noise

    def perturb(self, seed: int, sigma: float) -> None:
        assert not self._perturbed, "Already perturbed"
        self._seed = seed
        self._sigma = sigma
        self._perturbed = True
        for param, n in zip(self._module.parameters(), self._generate_noise(), strict=True):
            param.data.add_(n)

    def accept(self) -> None:
        assert self._perturbed, "Not perturbed"
        self._perturbed = False
        self._seed = None
        self._sigma = None

    def unperturb(self) -> None:
        assert self._perturbed, "Not perturbed"
        for param, n in zip(self._module.parameters(), self._generate_noise(), strict=True):
            param.data.sub_(n)
        self.accept()
