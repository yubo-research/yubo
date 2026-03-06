from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from torch import nn

from sampling.gather_proj_t import GatherProjSpec, project_module
from sampling.sparse_jl_t import (
    block_sparse_jl_noise_from_seed_wr,
    block_sparse_jl_transform_module_to_cpu_wr,
)


@dataclass
class ENNSeedSelectConfig:
    d: int = 100
    s: int = 4
    jl_seed: int = 123
    k: int = 25
    fit_interval: int = 50
    warmup_real_obs: int = 200
    num_candidates: int = 2
    select_interval: int = 1
    embedder: str = "sparse_jl"
    gather_t: int = 64
    noise_prob: float | None = None
    noise_chunk_size: int = 2**16


class ENNMuPlusSeedSelector:
    """ENN-based candidate seed selector trained on mu_plus.

    Features: z_plus = T(x) + T(noise(seed, sigma)), computed without O(D) allocation.
    Target: y = mu_plus (real positive-phase evaluation).
    Acquisition: UCB = mu + se (project convention).
    """

    def __init__(self, *, module: nn.Module, perturbator: object, cfg: ENNSeedSelectConfig):
        self._module = module
        self._perturbator = perturbator
        self._cfg = cfg

        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._num_new_since_fit = 0

        self._enn_model: object | None = None
        self._enn_params: object | None = None

        self._y_mean = 0.0
        self._y_std = 1.0

        self._pending_x: np.ndarray | None = None
        self._num_choose_calls = 0
        self._gather_spec: GatherProjSpec | None = None

        if str(self._cfg.embedder) == "gather":
            num_dim = int(sum(p.numel() for p in self._module.parameters()))
            self._gather_spec = GatherProjSpec.make(
                dim_ambient=num_dim,
                d=int(self._cfg.d),
                t=int(self._cfg.gather_t),
                seed=int(self._cfg.jl_seed),
            )

    def _maybe_fit(self) -> None:
        if len(self._x) < max(2, int(self._cfg.warmup_real_obs)):
            return
        if self._enn_params is not None and self._num_new_since_fit < int(self._cfg.fit_interval):
            return

        x = np.asarray(self._x, dtype=np.float64)
        y = np.asarray(self._y, dtype=np.float64)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) if float(y.std()) > 0 else 1.0
        y_std = (y - self._y_mean) / self._y_std

        self._enn_model = EpistemicNearestNeighbors(
            x,
            y_std[:, None],
            None,
            scale_x=False,
            index_driver=ENNIndexDriver.FLAT,
        )
        rng = np.random.default_rng(0)
        self._enn_params = enn_fit(
            self._enn_model,
            k=int(self._cfg.k),
            num_fit_candidates=200,
            num_fit_samples=200,
            rng=rng,
        )

        self._num_new_since_fit = 0

    def _embed_base(self) -> tuple[torch.Tensor, int]:
        z_base = block_sparse_jl_transform_module_to_cpu_wr(
            self._module,
            d=int(self._cfg.d),
            s=int(self._cfg.s),
            seed=int(self._cfg.jl_seed),
            chunk_size=int(self._cfg.noise_chunk_size),
        )
        num_dim = int(sum(p.numel() for p in self._module.parameters()))
        return z_base, num_dim

    def _embed_dz(self, *, seed: int, sigma: float, num_dim: int) -> torch.Tensor:
        return block_sparse_jl_noise_from_seed_wr(
            num_dim_ambient=int(num_dim),
            d=int(self._cfg.d),
            s=int(self._cfg.s),
            jl_seed=int(self._cfg.jl_seed),
            noise_seed=int(seed),
            sigma=float(sigma),
            prob=self._cfg.noise_prob,
            chunk_size=int(self._cfg.noise_chunk_size),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def _embed_z_plus_np(self, *, z_base: torch.Tensor, seed: int, sigma: float, num_dim: int) -> np.ndarray:
        dz = self._embed_dz(seed=int(seed), sigma=float(sigma), num_dim=int(num_dim))
        return (z_base + dz).numpy()

    def _embed_z_plus_np_gather(self, *, seed: int, sigma: float) -> np.ndarray:
        assert self._gather_spec is not None

        with torch.no_grad():
            self._perturbator.perturb(int(seed), float(sigma))
            try:
                z = project_module(self._module, spec=self._gather_spec)
            finally:
                self._perturbator.unperturb()
        z_cpu = z.float().cpu()
        return z_cpu.numpy().astype(np.float64, copy=False)

    def _posterior_std(self, x_cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._enn_model is not None and self._enn_params is not None
        post = self._enn_model.posterior(x_cand, params=self._enn_params, flags=PosteriorFlags(observation_noise=False))
        mu_std = np.asarray(post.mu).reshape(-1)
        se_std = np.asarray(post.se).reshape(-1)
        return mu_std, se_std

    def choose_seed_ucb(self, *, base_seed: int, sigma: float) -> tuple[int, float | None]:
        self._num_choose_calls += 1
        if int(self._cfg.select_interval) > 1 and (self._num_choose_calls % int(self._cfg.select_interval) != 0):
            self._pending_x = None
            return int(base_seed), None

        m = int(self._cfg.num_candidates)
        if str(self._cfg.embedder) == "gather":
            x_base = self._embed_z_plus_np_gather(seed=int(base_seed), sigma=float(sigma))
            z_base = None
            num_dim = 0
        else:
            z_base, num_dim = self._embed_base()
            x_base = self._embed_z_plus_np(z_base=z_base, seed=int(base_seed), sigma=float(sigma), num_dim=int(num_dim))

        if m <= 1:
            self._pending_x = x_base
            return int(base_seed), None

        self._maybe_fit()
        if self._enn_params is None:
            self._pending_x = x_base
            return int(base_seed), None

        seeds = np.arange(int(base_seed), int(base_seed) + m, dtype=np.int64)
        if str(self._cfg.embedder) == "gather":
            x_cand = np.asarray(
                [self._embed_z_plus_np_gather(seed=int(s), sigma=float(sigma)) for s in seeds.tolist()],
                dtype=np.float64,
            )
        else:
            assert z_base is not None
            x_cand = np.asarray(
                [self._embed_z_plus_np(z_base=z_base, seed=int(s), sigma=float(sigma), num_dim=int(num_dim)) for s in seeds.tolist()],
                dtype=np.float64,
            )
        mu_std, se_std = self._posterior_std(x_cand)

        mu = self._y_mean + self._y_std * mu_std
        se = abs(self._y_std) * se_std
        ucb = mu + se
        j = int(np.argmax(ucb))

        self._pending_x = np.asarray(x_cand[j], dtype=np.float64)
        return int(seeds[j]), float(ucb[j])

    def tell_mu_plus(self, *, mu_plus: float) -> None:
        if self._pending_x is None:
            return
        self._x.append(np.asarray(self._pending_x, dtype=np.float64))
        self._y.append(float(mu_plus))
        self._num_new_since_fit += 1
        self._pending_x = None
