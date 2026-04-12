"""Deep Networks for Global Optimization (DNGO) surrogate (Snoek et al., ICML 2015).

Trains a feedforward net to minimize MSE, then treats the last hidden layer as a fixed
basis \\phi(x) and fits Bayesian linear regression on top:

    y \\mid w \\sim \\mathcal{N}(\\Phi w, \\beta^{-1} I), \\quad
    w \\sim \\mathcal{N}(0, \\alpha^{-1} I).

Hyperparameters \\alpha (weight prior precision) and \\beta (noise precision) are tuned
by maximizing the marginal likelihood (Bishop PRML eq. 3.86 pattern).

Architecture and preprocessing follow the reference PyTorch sketch in
``Projects/DNGO`` (tanh MLP, basis = penultimate layer).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .dngo_model import (
    DNGOConfig,
    _dngo_optimize_blr_head,
    _dngo_train_mlp_epochs,
    _dngo_train_val_split,
    _DNGONet,
    _marginal_log_likelihood,
    _neg_mll,
    _normalize_x,
    _normalize_y,
)

__all__ = [
    "DNGOConfig",
    "DNGOSurrogate",
    "_DNGONet",
    "_marginal_log_likelihood",
    "_neg_mll",
]


class DNGOSurrogate:
    """DNGO: trained MLP basis + Bayesian linear regression head."""

    def __init__(self, config: DNGOConfig | None = None) -> None:
        self.cfg = config or DNGOConfig()
        self._net: _DNGONet | None = None
        self._mx: np.ndarray | None = None
        self._sx: np.ndarray | None = None
        self._my: float | None = None
        self._sy: float | None = None
        self._m_w: np.ndarray | None = None
        self._chol_a: np.ndarray | None = None
        self._beta: float | None = None
        self._device = torch.device("cpu")

    def fit(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Fit on training inputs ``x`` (N, d) and targets ``y`` (N,) or (N, 1)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"x and y length mismatch: {x.shape[0]} vs {y.shape[0]}")

        x_n, self._mx, self._sx = _normalize_x(x)
        y_n, self._my, self._sy = _normalize_y(y)

        torch.manual_seed(self.cfg.seed)
        dim_in = x.shape[1]
        self._net = _DNGONet(
            dim_in,
            self.cfg.hidden_width,
            self.cfg.feature_dim,
            self.cfg.num_middle_layers,
        ).to(self._device)

        xt = torch.from_numpy(x_n).float().to(self._device)
        yt = torch.from_numpy(y_n).float().to(self._device)
        opt = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        loss_fn = nn.MSELoss()
        n = x.shape[0]
        xt_tr, yt_tr, xt_val, yt_val, use_val = _dngo_train_val_split(xt, yt, n, self.cfg)
        best_val_mse, best_state = _dngo_train_mlp_epochs(
            self._net,
            opt,
            loss_fn,
            xt_tr,
            yt_tr,
            xt_val,
            yt_val,
            use_val,
            self.cfg.num_epochs,
        )

        self._net.eval()
        if best_state is not None:
            self._net.load_state_dict(best_state)

        with torch.no_grad():
            phi = self._net.basis(xt).cpu().numpy().astype(np.float64)
        # Match the trained readout ``output_layer(phi) = w^T phi + b`` by augmenting with a
        # constant column so BLR can represent the same affine family (Snoek et al. replace
        # only the last linear map).
        phi = np.concatenate([phi, np.ones((phi.shape[0], 1), dtype=np.float64)], axis=1)
        mll, parts, res = _dngo_optimize_blr_head(phi, y_n, self.cfg)
        if not parts:
            raise RuntimeError("DNGO marginal likelihood optimization failed (singular A).")
        self._m_w = parts["m_w"]
        self._chol_a = parts["chol_a"]
        self._beta = float(parts["beta"])

        out: dict[str, Any] = {
            "marginal_log_likelihood": mll,
            "log_alpha": float(res.x[0]),
            "log_beta": float(res.x[1]),
            "optimize_success": bool(res.success),
            "optimize_message": str(res.message),
        }
        if use_val and best_state is not None:
            out["early_stopping_best_val_mse_normalized_y"] = best_val_mse
        return out

    def predict(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and variance in the *original* y scale.

        Variance follows the BLR head on ``[\\phi(x), 1]``. The mean is either the trained MLP
        readout (default, :attr:`DNGOConfig.use_mse_readout_for_mean`) or the BLR posterior mean
        on features; the two need not match because MSE and marginal likelihood optimize
        different linear heads on the same basis.
        """
        if self._net is None or self._m_w is None or self._chol_a is None or self._beta is None:
            raise RuntimeError("Call fit() before predict().")
        assert self._mx is not None and self._sx is not None
        assert self._my is not None and self._sy is not None

        x_test = np.asarray(x_test, dtype=np.float64)
        x_n = (x_test - self._mx) / self._sx
        xt = torch.from_numpy(x_n).float().to(self._device)
        with torch.no_grad():
            phi_raw = self._net.basis(xt).cpu().numpy().astype(np.float64)
            if self.cfg.use_mse_readout_for_mean:
                mean_n = self._net(xt).cpu().numpy().astype(np.float64).reshape(-1)
            else:
                phi_tmp = np.concatenate([phi_raw, np.ones((phi_raw.shape[0], 1), dtype=np.float64)], axis=1)
                mean_n = (phi_tmp @ self._m_w).reshape(-1)

        phi_star = np.concatenate([phi_raw, np.ones((phi_raw.shape[0], 1), dtype=np.float64)], axis=1)
        v = np.linalg.solve(self._chol_a, phi_star.T)
        pred_var_n = (1.0 / self._beta) + np.sum(v**2, axis=0)

        mean = mean_n * self._sy + self._my
        var = pred_var_n * (self._sy**2)
        return mean, var
