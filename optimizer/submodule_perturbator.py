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

    def _device(self) -> torch.device:
        return next(self._module.parameters()).device

    def _rng(self, seed: int) -> torch.Generator:
        g = torch.Generator(device=str(self._device()))
        g.manual_seed(int(seed))
        return g

    def _select_mask(self, *, g: torch.Generator) -> torch.Tensor:
        device = self._device()
        n = len(self._leaf_modules)
        mask = torch.rand((n,), device=device, generator=g) < self._prob
        if not bool(mask.any().item()):
            idx = int(torch.randint(n, (1,), device=device, generator=g).item())
            mask[idx] = True
        return mask

    def _apply(self, *, seed: int, sigma: float, chunk_size: int = 2**16) -> None:
        g = self._rng(seed)
        device = self._device()
        mask = self._select_mask(g=g)

        for i, leaf in enumerate(self._leaf_modules):
            selected = bool(mask[i].item())
            if not selected:
                continue
            for p in leaf.parameters(recurse=False):
                assert p.device == device, "SubmodulePerturbator requires all params on one device"
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
