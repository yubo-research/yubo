import torch
from torch import nn


class SubmodulePerturbator:
    """RAASP-style perturbation at the submodule level.

    Instead of selecting individual dimensions, this perturbator
    randomly selects entire leaf submodules (those that own parameters)
    to perturb.  All parameters within a selected submodule are
    perturbed with iid Gaussian noise; unselected submodules are left
    untouched.

    ``num_module_target`` controls how many submodules are perturbed on
    average.  If 0 < num_module_target < 1, it is treated as a fraction
    of the total leaf-module count.
    """

    def __init__(self, module: nn.Module, *, num_module_target: float):
        self._module = module
        self._leaf_modules = [m for m in module.modules() if list(m.parameters(recurse=False))]
        n = len(self._leaf_modules)
        if 0 < num_module_target < 1:
            self._prob = num_module_target
        else:
            self._prob = min(num_module_target / n, 1.0)
        self._perturbed = False
        self._seed: int | None = None

    def _generate_noise(self, sigma: float) -> list[list[torch.Tensor]]:
        """Return noise grouped by leaf module.

        Each element is a list of tensors (one per parameter in that
        leaf module).  Unselected modules get zero tensors.
        """
        rng = torch.Generator()
        rng.manual_seed(self._seed)

        # Draw one Bernoulli per leaf module.
        n = len(self._leaf_modules)
        mask = torch.rand(n, generator=rng) < self._prob

        # Guarantee at least one module is selected.
        if not mask.any():
            idx = torch.randint(n, (1,), generator=rng).item()
            mask[idx] = True

        noise_groups: list[list[torch.Tensor]] = []
        for i, leaf in enumerate(self._leaf_modules):
            group = []
            for p in leaf.parameters(recurse=False):
                n_tensor = torch.randn(p.shape, generator=rng) * sigma
                if not mask[i]:
                    n_tensor = torch.zeros_like(n_tensor)
                group.append(n_tensor.to(p.device))
            noise_groups.append(group)
        return noise_groups

    def perturb(self, seed: int, sigma: float) -> None:
        assert not self._perturbed, "Already perturbed"
        self._seed = seed
        self._sigma = sigma
        self._perturbed = True
        for leaf, group in zip(self._leaf_modules, self._generate_noise(sigma), strict=True):
            for p, n in zip(leaf.parameters(recurse=False), group, strict=True):
                p.data.add_(n)

    def accept(self) -> None:
        assert self._perturbed, "Not perturbed"
        self._perturbed = False
        self._seed = None

    def unperturb(self) -> None:
        assert self._perturbed, "Not perturbed"
        for leaf, group in zip(
            self._leaf_modules,
            self._generate_noise(self._sigma),
            strict=True,
        ):
            for p, n in zip(leaf.parameters(recurse=False), group, strict=True):
                p.data.sub_(n)
        self.accept()
