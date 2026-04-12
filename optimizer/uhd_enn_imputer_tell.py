"""Observation bookkeeping for :class:`ENNMinusImputer`."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

from sampling.gather_proj_t import GatherProjSpec, project_module
from sampling.sparse_jl_t import _block_sparse_hash_scatter_from_nz_t

from .uhd_enn_config import ENNImputerConfig, _compute_z


class ENNMinusImputerTellMixin:
    _module: nn.Module
    _cfg: ENNImputerConfig
    _noise_nz_fn: Callable[[int, float], tuple[np.ndarray, np.ndarray]]
    _z_base: torch.Tensor
    _delta_z: torch.Tensor | None
    _delta_x: np.ndarray | None
    _x: list[np.ndarray]
    _y: list[float]
    _num_new_since_fit: int
    _num_real_evals: int
    _last_mu_plus: float | None
    _gather_spec: GatherProjSpec | None

    def begin_pair(self, *, seed: int, sigma: float) -> None:
        """Precompute delta_z = T(noise) for this antithetic pair."""
        if self._cfg.embedder == "gather":
            return
        idx_np, vals_np = self._noise_nz_fn(int(seed), float(sigma))
        idx_t = torch.from_numpy(np.asarray(idx_np, dtype=np.int64))
        vals_t = torch.from_numpy(np.asarray(vals_np))
        self._delta_z = _block_sparse_hash_scatter_from_nz_t(
            nz_indices=idx_t,
            nz_values=vals_t,
            d=self._cfg.d,
            s=self._cfg.s,
            seed=self._cfg.jl_seed,
            dtype=self._z_base.dtype,
            device=self._z_base.device,
        )
        # For target="delta", we train/predict on direction embeddings directly.
        self._delta_x = self._delta_z.double().numpy()

    def _tell_real_gather(self, *, mu: float, phase: str) -> None:
        assert self._gather_spec is not None
        if self._cfg.target == "delta":
            raise RuntimeError("target='delta' not supported with embedder='gather'")
        x_t = project_module(self._module, spec=self._gather_spec).float().cpu()
        x = x_t.numpy().astype(np.float64, copy=False)
        if phase == "plus":
            self._last_mu_plus = float(mu)
        if self._cfg.target == "mu_plus" and phase != "plus":
            self._num_real_evals += 1
            return
        self._x.append(x)
        self._y.append(float(mu))
        self._num_new_since_fit += 1
        self._num_real_evals += 1

    def tell_real(self, *, mu: float, phase: str) -> None:
        """Add a real observation at z_plus or z_minus (current pair)."""
        if self._cfg.embedder == "gather":
            self._tell_real_gather(mu=mu, phase=phase)
            return
        if phase == "plus":
            self._last_mu_plus = float(mu)
            if self._cfg.target == "mu_plus":
                assert self._delta_x is not None
                self._x.append(self._delta_x)
                self._y.append(float(mu))
                self._num_new_since_fit += 1
            elif self._cfg.target != "delta":
                self._x.append(_compute_z(self._z_base, self._delta_z, +1.0))
                self._y.append(float(mu))
                self._num_new_since_fit += 1
            self._num_real_evals += 1
            return
        if phase != "minus":
            raise ValueError(f"Unknown phase: {phase!r}")
        if self._cfg.target == "mu_plus":
            self._num_real_evals += 1
            return
        if self._cfg.target == "delta":
            if self._last_mu_plus is None:
                raise RuntimeError("delta target requires a preceding real mu_plus in the same pair")
            assert self._delta_x is not None
            self._x.append(self._delta_x)
            self._y.append(float(self._last_mu_plus) - float(mu))
        else:
            self._x.append(_compute_z(self._z_base, self._delta_z, -1.0))
            self._y.append(float(mu))
        self._num_new_since_fit += 1
        self._num_real_evals += 1
