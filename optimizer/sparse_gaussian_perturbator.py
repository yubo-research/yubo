import torch
from torch import nn


class SparseGaussianPerturbator:
    """RAASP-style sparse Gaussian perturbation.

    Like GaussianPerturbator but only perturbs a random subset of
    dimensions per step.  The expected number of perturbed dimensions
    is ``num_dim_target``; each dimension is included independently
    with probability ``min(num_dim_target / dim, 1)``.

    The mask and noise are both generated deterministically from the
    seed, so ``unperturb()`` can replay the exact same perturbation.
    """

    def __init__(self, module: nn.Module, *, num_dim_target: float):
        self._module = module
        dim = sum(p.numel() for p in module.parameters())
        if 0 < num_dim_target < 1:
            self._prob = num_dim_target
        else:
            self._prob = min(num_dim_target / dim, 1.0)
        self._perturbed = False
        self._seed: int | None = None

    def _generate_noise(self, sigma: float) -> list[torch.Tensor]:
        rng = torch.Generator()
        rng.manual_seed(self._seed)
        noise = []
        any_nonzero = False
        for param in self._module.parameters():
            mask = torch.rand(param.shape, generator=rng) < self._prob
            n = torch.randn(param.shape, generator=rng) * sigma
            n = n * mask
            any_nonzero = any_nonzero or mask.any().item()
            noise.append(n.to(param.device))

        # Guarantee at least one dimension is perturbed.
        if not any_nonzero:
            rng2 = torch.Generator()
            rng2.manual_seed(self._seed + 2**31)
            flat = torch.cat([n.reshape(-1) for n in noise])
            idx = torch.randint(len(flat), (1,), generator=rng2).item()
            val = torch.randn(1, generator=rng2).item() * sigma
            offset = 0
            for i, param in enumerate(self._module.parameters()):
                numel = param.numel()
                if offset <= idx < offset + numel:
                    local_idx = idx - offset
                    noise[i].reshape(-1)[local_idx] = val
                    break
                offset += numel

        return noise

    def perturb(self, seed: int, sigma: float) -> None:
        assert not self._perturbed, "Already perturbed"
        self._seed = seed
        self._sigma = sigma
        self._perturbed = True
        for param, n in zip(self._module.parameters(), self._generate_noise(sigma), strict=True):
            param.data.add_(n)

    def accept(self) -> None:
        assert self._perturbed, "Not perturbed"
        self._perturbed = False
        self._seed = None

    def unperturb(self) -> None:
        assert self._perturbed, "Not perturbed"
        for param, n in zip(
            self._module.parameters(),
            self._generate_noise(self._sigma),
            strict=True,
        ):
            param.data.sub_(n)
        self.accept()
