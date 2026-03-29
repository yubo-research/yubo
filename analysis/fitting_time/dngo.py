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

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

__all__ = ["DNGOConfig", "DNGOSurrogate"]


@dataclass(frozen=True)
class DNGOConfig:
    hidden_width: int = 50
    feature_dim: int = 50
    num_middle_layers: int = 2
    num_epochs: int = 1000
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    log_hyp_bounds: tuple[float, float] = (-5.0, 10.0)
    seed: int = 0
    val_fraction: float = 0.15
    """Hold out this fraction of training rows for MSE early stopping (min val MSE checkpoint)."""
    min_obs_for_val_split: int = 36
    """Below this many rows, train on all data (no split)."""
    use_mse_readout_for_mean: bool = True
    """If True (default), point predictions use the MSE-trained last layer; BLR still supplies
    ``\\alpha, \\beta`` and predictive **variance**. Marginal-likelihood BLR means can disagree
    with SGD weights and generalize worse on some surfaces (e.g. ``f:`` benchmarks) while the
    trained readout tracks the MSE objective. Set False for the strict ``\\phi^\\top w`` mean."""


class _DNGONet(nn.Module):
    """MLP: input -> H (tanh) -> (H tanh)^L -> D features -> 1 output."""

    def __init__(
        self,
        dim_in: int,
        hidden_width: int,
        feature_dim: int,
        num_middle_layers: int,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(dim_in, hidden_width)
        self.middle = nn.ModuleList(nn.Linear(hidden_width, hidden_width) for _ in range(num_middle_layers))
        self.last_hidden = nn.Linear(hidden_width, feature_dim)
        self.output_layer = nn.Linear(feature_dim, 1)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.input_layer(x))
        for layer in self.middle:
            h = torch.tanh(layer(h))
        return self.last_hidden(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.basis(x))


def _normalize_x(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx = x.mean(axis=0)
    sx = x.std(axis=0, ddof=1)
    sx = np.where(sx < 1e-12, 1.0, sx)
    return (x - mx) / sx, mx, sx


def _normalize_y(y: np.ndarray) -> tuple[np.ndarray, float, float]:
    y1 = y.reshape(-1)
    my = float(y1.mean())
    sy = float(y1.std(ddof=1))
    if sy < 1e-12:
        sy = 1.0
    return ((y1 - my) / sy).reshape(-1, 1), my, sy


def _marginal_log_likelihood(
    log_alpha: float,
    log_beta: float,
    phi: np.ndarray,
    y: np.ndarray,
) -> tuple[float, dict[str, np.ndarray]]:
    """Log marginal likelihood and posterior moments (Bishop PRML-style BLR on basis phi)."""
    alpha = float(np.exp(log_alpha))
    beta = float(np.exp(log_beta))
    n, k = phi.shape
    phit_y = phi.T @ y
    a_mat = alpha * np.eye(k, dtype=np.float64) + beta * (phi.T @ phi)
    try:
        chol = np.linalg.cholesky(a_mat)
    except np.linalg.LinAlgError:
        return -1e25, {}

    t = np.linalg.solve(chol, phit_y)
    t = np.linalg.solve(chol.T, t)
    m_w = beta * t

    resid = y - phi @ m_w
    e_w = 0.5 * beta * float(resid.T @ resid) + 0.5 * alpha * float(m_w.T @ m_w)
    log_det_a = 2.0 * float(np.sum(np.log(np.diag(chol))))
    mll = 0.5 * k * np.log(alpha) + 0.5 * n * np.log(beta) - e_w - 0.5 * log_det_a - 0.5 * n * np.log(2.0 * np.pi)
    return float(mll), {"m_w": m_w, "chol_a": chol, "alpha": alpha, "beta": beta}


def _neg_mll(theta: np.ndarray, phi: np.ndarray, y: np.ndarray) -> float:
    mll, _ = _marginal_log_likelihood(float(theta[0]), float(theta[1]), phi, y)
    return -mll


def _clone_dngo_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _dngo_train_val_split(
    xt: torch.Tensor, yt: torch.Tensor, n: int, cfg: DNGOConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, bool]:
    use_val = cfg.val_fraction > 0.0 and n >= cfg.min_obs_for_val_split and int(round(n * cfg.val_fraction)) >= 1 and n - int(round(n * cfg.val_fraction)) >= 8
    if not use_val:
        return xt, yt, None, None, False
    rng_split = np.random.default_rng(int(cfg.seed) + 901)
    perm = rng_split.permutation(n)
    n_val = max(1, int(round(n * cfg.val_fraction)))
    n_tr = n - n_val
    tr_idx = perm[:n_tr]
    val_idx = perm[n_tr:]
    return xt[tr_idx], yt[tr_idx], xt[val_idx], yt[val_idx], True


def _dngo_train_mlp_epochs(
    net: _DNGONet,
    opt: torch.optim.Optimizer,
    loss_fn: nn.MSELoss,
    xt_tr: torch.Tensor,
    yt_tr: torch.Tensor,
    xt_val: torch.Tensor | None,
    yt_val: torch.Tensor | None,
    use_val: bool,
    num_epochs: int,
) -> tuple[float, dict[str, torch.Tensor] | None]:
    best_val_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    net.train()
    for _ in range(num_epochs):
        opt.zero_grad()
        pred = net(xt_tr)
        loss = loss_fn(pred, yt_tr)
        loss.backward()
        opt.step()
        if use_val and xt_val is not None and yt_val is not None:
            net.eval()
            with torch.no_grad():
                pv = net(xt_val)
                v_mse = float(loss_fn(pv, yt_val).item())
            net.train()
            if v_mse < best_val_mse:
                best_val_mse = v_mse
                best_state = _clone_dngo_state_dict(net)
    return best_val_mse, best_state


def _dngo_optimize_blr_head(phi: np.ndarray, y_n: np.ndarray, cfg: DNGOConfig) -> tuple[float, dict[str, np.ndarray], Any]:
    yn = y_n.reshape(-1)
    rng = np.random.default_rng(cfg.seed)
    theta0 = rng.uniform(cfg.log_hyp_bounds[0], cfg.log_hyp_bounds[1], size=2)
    res = minimize(
        _neg_mll,
        theta0,
        args=(phi, yn),
        method="L-BFGS-B",
        bounds=(cfg.log_hyp_bounds, cfg.log_hyp_bounds),
    )
    mll, parts = _marginal_log_likelihood(float(res.x[0]), float(res.x[1]), phi, yn)
    return mll, parts, res


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
