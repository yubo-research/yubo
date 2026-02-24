import numpy as np
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
        self._k: int | None = None
        if 0 < num_dim_target < 1:
            self._prob = num_dim_target
        else:
            k = int(num_dim_target)
            if k >= dim:
                self._prob = 1.0
            else:
                self._k = max(k, 1)
                self._prob = float(self._k / dim)
        self._perturbed = False
        self._seed: int | None = None
        self._param_numel = [p.numel() for p in module.parameters()]
        self._dim = int(sum(self._param_numel))
        self._param_offsets: list[int] = []
        off = 0
        for n in self._param_numel:
            self._param_offsets.append(off)
            off += int(n)

    def _device(self) -> torch.device:
        return next(self._module.parameters()).device

    def _rng(self, seed: int) -> torch.Generator:
        g = torch.Generator(device=str(self._device()))
        g.manual_seed(int(seed))
        return g

    def _apply(self, *, seed: int, sigma: float, chunk_size: int = 2**16) -> None:
        device = self._device()
        if self._k is not None:
            self._apply_k_sparse(device=device, seed=seed, sigma=sigma)
        else:
            self._apply_prob_sparse(device=device, seed=seed, sigma=sigma, chunk_size=chunk_size)

    def _apply_k_sparse(self, *, device: torch.device, seed: int, sigma: float) -> None:
        idx_np, vals_np = self.sample_global_nz(seed=seed, sigma=sigma)
        idx_np = np.asarray(idx_np, dtype=np.int64)
        vals_np = np.asarray(vals_np)
        assert idx_np.shape == vals_np.shape
        for p, start in zip(self._module.parameters(), self._param_offsets, strict=True):
            assert p.device == device, "SparseGaussianPerturbator requires all params on one device"
            end = start + p.numel()
            lo = int(np.searchsorted(idx_np, start, side="left"))
            hi = int(np.searchsorted(idx_np, end, side="left"))
            if lo >= hi:
                continue
            local = idx_np[lo:hi] - start
            v = vals_np[lo:hi]
            local_t = torch.from_numpy(local).to(device=device, dtype=torch.int64)
            v_t = torch.from_numpy(v).to(device=device, dtype=p.data.dtype)
            p.data.view(-1).index_add_(0, local_t, v_t)

    def _apply_prob_sparse(self, *, device: torch.device, seed: int, sigma: float, chunk_size: int) -> None:
        g = self._rng(seed)
        num_perturbed = torch.zeros((), device=device, dtype=torch.int64)
        for p in self._module.parameters():
            assert p.device == device, "SparseGaussianPerturbator requires all params on one device"
            flat = p.data.view(-1)
            n = flat.numel()
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                u = torch.rand((end - start,), device=device, generator=g)
                mask = u < self._prob
                noise = torch.randn((end - start,), device=device, dtype=flat.dtype, generator=g)
                noise.mul_(sigma)
                noise.mul_(mask)
                flat[start:end].add_(noise)
                num_perturbed = num_perturbed + mask.sum()

        if int(num_perturbed.item()) == 0:
            g2 = self._rng(seed + 2**31)
            idx = int(torch.randint(self._dim, (1,), device=device, generator=g2).item())
            val = float(torch.randn((1,), device=device, generator=g2).item()) * float(sigma)
            offset = 0
            for p in self._module.parameters():
                numel = p.numel()
                if offset <= idx < offset + numel:
                    p.data.view(-1)[idx - offset].add_(val)
                    break
                offset += numel

    def sample_global_nz(self, *, seed: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (global_indices, values) for the k-sparse perturbation.

        Only valid when constructed with num_dim_target >= 1 and < dim.
        Deterministic given seed.
        """
        assert self._k is not None, "sample_global_nz only supported for k-sparse mode (num_dim_target >= 1)"
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(self._dim, size=int(self._k), replace=False)
        idx.sort()
        vals = rng.standard_normal(size=idx.shape).astype(np.float32) * float(sigma)
        return idx.astype(np.int64), vals

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
