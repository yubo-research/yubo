"""DNGO MLP architecture and training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


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
