"""ENN fit, UCB seed choice, and imputation for :class:`ENNMinusImputer`."""

from __future__ import annotations

import numpy as np
import torch
from enn.enn.enn_params import PosteriorFlags

from sampling.gather_proj_t import project_module
from sampling.sparse_jl_t import _block_sparse_hash_scatter_from_nz_t

from .uhd_enn_config import _compute_z
from .uhd_enn_fit_helpers import fit_enn_regressor_on_points


class ENNMinusImputerPredictMixin:
    def _maybe_fit(self) -> None:
        if len(self._x) < max(2, self._cfg.warmup_real_obs):
            return
        if self._enn_params is not None and self._num_new_since_fit < self._cfg.fit_interval:
            return

        y_mean, y_std, model, params = fit_enn_regressor_on_points(self._x, self._y, k=int(self._cfg.k))
        self._y_mean = y_mean
        self._y_std = y_std
        self._enn_model = model
        self._enn_params = params
        self._num_new_since_fit = 0

    def choose_seed_ucb(self, *, base_seed: int, sigma: float) -> tuple[int, float | None]:
        """Pick a seed among candidates {base_seed..base_seed+num_candidates-1} via UCB.

        Returns (chosen_seed, ucb_of_chosen_or_None). If the ENN isn't ready, falls back
        to base_seed and returns (base_seed, None).
        """
        m = int(self._cfg.num_candidates)
        if m <= 1:
            return int(base_seed), None
        if self._cfg.target not in {"delta", "mu_plus"}:
            return int(base_seed), None

        self._maybe_fit()
        if self._enn_params is None:
            return int(base_seed), None

        seeds = np.arange(int(base_seed), int(base_seed) + m, dtype=np.int64)
        xs: list[np.ndarray] = []
        for s in seeds.tolist():
            idx_np, vals_np = self._noise_nz_fn(int(s), float(sigma))
            idx_t = torch.from_numpy(np.asarray(idx_np, dtype=np.int64))
            vals_t = torch.from_numpy(np.asarray(vals_np))
            delta_z = _block_sparse_hash_scatter_from_nz_t(
                nz_indices=idx_t,
                nz_values=vals_t,
                d=self._cfg.d,
                s=self._cfg.s,
                seed=self._cfg.jl_seed,
                dtype=self._z_base.dtype,
                device=self._z_base.device,
            )
            xs.append(delta_z.double().numpy())
        x_cand = np.asarray(xs, dtype=np.float64)

        post = self._enn_model.posterior(
            x_cand,
            params=self._enn_params,
            flags=PosteriorFlags(observation_noise=False),
        )
        mu_std = np.asarray(post.mu).reshape(-1)
        se_std = np.asarray(post.se).reshape(-1)

        mu = self._y_mean + self._y_std * mu_std
        se = abs(self._y_std) * se_std
        ucb = mu + se
        j = int(np.argmax(ucb))
        return int(seeds[j]), float(ucb[j])

    def should_impute_negative(self) -> bool:
        """Return True if we should impute mu_minus on this negative phase."""
        if self._cfg.target == "mu_plus":
            return False
        self._num_negative_phases += 1
        if len(self._x) < self._cfg.warmup_real_obs:
            return False
        if self._cfg.refresh_interval > 0 and (self._num_negative_phases % self._cfg.refresh_interval == 0):
            return False
        self._maybe_fit()
        if self._enn_params is None:
            return False
        if self._enn_model is None:
            return False
        if self._num_calib < int(self._cfg.min_calib_points):
            return False
        if self._abs_err_ema is None:
            return False
        return float(self._abs_err_ema) <= float(self._cfg.max_abs_err_ema)

    def _predict_y_current(self) -> tuple[float, float]:
        """Predict (y_hat, y_se) in the target space at z_minus for the current pair.

        target "mu_minus": y is mu_minus.
        target "delta": y is Δμ = mu_plus - mu_minus.
        target "mu_plus": y is mu_plus (seed selection only; not used for negative imputation).
        """
        assert self._enn_model is not None and self._enn_params is not None
        if self._cfg.embedder == "gather":
            assert self._gather_spec is not None
            z_t = project_module(self._module, spec=self._gather_spec).float().cpu()
            z = z_t.numpy().astype(np.float64, copy=False)[None, :]
        elif self._cfg.target in {"delta", "mu_plus"}:
            assert self._delta_x is not None
            z = self._delta_x[None, :]
        else:
            z = _compute_z(self._z_base, self._delta_z, -1.0)[None, :]
        post = self._enn_model.posterior(z, params=self._enn_params, flags=PosteriorFlags(observation_noise=False))
        mu_std = float(np.asarray(post.mu).reshape(-1)[0])
        se_std = float(np.asarray(post.se).reshape(-1)[0])
        y_hat = self._y_mean + self._y_std * mu_std
        y_se = abs(self._y_std) * se_std
        return float(y_hat), float(y_se)

    def predict_current(self) -> tuple[float, float]:
        """Predict (mu_minus, se) at z_minus for the current pair.

        If cfg.target == "delta", we predict Δμ and convert:
          mu_minus = mu_plus_real - Δμ_hat
        """
        y_hat, y_se = self._predict_y_current()
        if self._cfg.target == "mu_plus":
            return float(y_hat), float(y_se)
        if self._cfg.target == "delta":
            if self._last_mu_plus is None:
                raise RuntimeError("delta target requires a preceding real mu_plus in the same pair")
            mu_minus = float(self._last_mu_plus) - float(y_hat)
            return float(mu_minus), float(y_se)
        return float(y_hat), float(y_se)

    def try_impute_current(self) -> tuple[bool, float, float]:
        """Return (imputed?, mu, se) at current params.

        If ENN is not confident enough, returns (False, nan, nan).
        """
        if self._cfg.target == "mu_plus":
            return False, float("nan"), float("nan")
        if not self.should_impute_negative():
            return False, float("nan"), float("nan")
        mu, se = self.predict_current()
        if se > float(self._cfg.se_threshold):
            return False, float("nan"), float("nan")
        self._num_imputed += 1
        return True, mu, se

    def update_base_after_step(self, *, step_scale: float, sigma: float) -> None:
        """Update z_base using the accepted update x <- x + step_scale * eps.

        We have delta_z = T(noise) = sigma * T(eps), so T(step_scale * eps) =
        (step_scale / sigma) * delta_z.
        """
        if self._cfg.embedder == "gather":
            return
        if self._delta_z is None:
            return
        if sigma == 0:
            return
        self._z_base = self._z_base + (float(step_scale) / float(sigma)) * self._delta_z
