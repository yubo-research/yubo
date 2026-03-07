from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_fit import enn_fit
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from torch import nn

from sampling.gather_proj_t import GatherProjSpec, project_module
from sampling.sparse_jl_t import _block_sparse_hash_scatter_from_nz_t


@dataclass
class ENNImputerConfig:
    d: int = 100
    s: int = 4
    jl_seed: int = 123
    k: int = 25
    fit_interval: int = 50  # refit after this many new real observations
    warmup_real_obs: int = 200  # don't impute until we have this many real points
    refresh_interval: int = 50  # force a real negative eval every N negative phases
    se_threshold: float = 0.25  # if predicted se > threshold, do real eval instead
    # What to model with ENN:
    # - "mu_minus": predict mu(x-) directly (original prototype).
    # - "delta": predict Δμ = mu(x+) - mu(x-) and convert back to mu(x-).
    # - "mu_plus": predict mu(x+) from direction embeddings (for seed selection only).
    target: str = "mu_minus"  # "mu_minus" | "delta" | "mu_plus"
    # If >1 (and target="delta"), we can do seed filtering by scoring multiple
    # candidate direction embeddings and choosing max UCB.
    num_candidates: int = 1
    select_interval: int = 1
    # Embedding backend:
    # - "direction": use SparseJL direction embeddings from k-sparse noise (fast, but requires nz noise).
    # - "gather": sparse-by-row gather projection of current module params (no O(D) buffers).
    embedder: str = "direction"  # "direction" | "gather"
    gather_t: int = 64
    # Calibration gate: only impute if recent prediction error is low.
    err_ema_beta: float = 0.95
    max_abs_err_ema: float = 0.25
    min_calib_points: int = 10


def _compute_z(z_base: torch.Tensor, delta_z: torch.Tensor, sign: float) -> np.ndarray:
    return (z_base + sign * delta_z).double().numpy()


class ENNMinusImputer:
    """Impute mu_minus in UHD's negative phase using ENN on SparseJL embeddings.

    This is a prototype to test "paired-minus imputation" (docs/uhd_enn_2.md #1):
    - always evaluate real x+ (positive phase),
    - predict mu(x-) for most negative phases once warm,
    - periodically refresh with real x- to recalibrate.
    """

    def __init__(
        self,
        *,
        module: nn.Module,
        cfg: ENNImputerConfig,
        noise_nz_fn: Callable[[int, float], tuple[np.ndarray, np.ndarray]],
    ):
        self._module = module
        self._cfg = cfg
        self._noise_nz_fn = noise_nz_fn

        # We do NOT embed the initial parameters. ENN operates on distances in embedding
        # space, and distances are translation-invariant. We maintain a consistent
        # coordinate system by starting at z_base=0 and updating via delta embeddings.
        self._z_base = torch.zeros((self._cfg.d,), dtype=torch.float32, device=torch.device("cpu"))
        self._delta_z = None
        self._delta_x: np.ndarray | None = None

        self._x: list[np.ndarray] = []
        self._y: list[float] = []
        self._num_new_since_fit = 0

        self._enn_model: object | None = None
        self._enn_params: object | None = None

        self._num_negative_phases = 0
        self._num_real_evals = 0
        self._num_imputed = 0

        # Standardization stats (computed at fit time)
        self._y_mean = 0.0
        self._y_std = 1.0

        # Calibration stats (real mu_minus vs predicted)
        self._abs_err_ema: float | None = None
        self._num_calib = 0

        # Cache last real mu_plus for the current antithetic pair (needed for delta target).
        self._last_mu_plus: float | None = None

        self._gather_spec: GatherProjSpec | None = None
        if self._cfg.embedder == "gather":
            dim_ambient = int(sum(p.numel() for p in self._module.parameters()))
            self._gather_spec = GatherProjSpec.make(
                dim_ambient=dim_ambient,
                d=int(self._cfg.d),
                t=int(self._cfg.gather_t),
                seed=int(self._cfg.jl_seed),
            )

    @property
    def num_real_evals(self) -> int:
        return self._num_real_evals

    @property
    def num_imputed(self) -> int:
        return self._num_imputed

    @property
    def abs_err_ema(self) -> float | None:
        return self._abs_err_ema

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

    def calibrate_minus(self, *, mu_minus_real: float) -> None:
        """Update prediction error stats using a real mu_minus at current pair."""
        if self._enn_params is None:
            return
        if self._cfg.target == "mu_plus":
            return
        if self._cfg.target == "delta" and self._last_mu_plus is None:
            return
        try:
            err = _calibration_error(self, mu_minus_real)
        except Exception:
            return
        if self._abs_err_ema is None:
            self._abs_err_ema = float(err)
        else:
            b = float(self._cfg.err_ema_beta)
            self._abs_err_ema = b * float(self._abs_err_ema) + (1.0 - b) * float(err)
        self._num_calib += 1

    def _maybe_fit(self) -> None:
        if len(self._x) < max(2, self._cfg.warmup_real_obs):
            return
        if self._enn_params is not None and self._num_new_since_fit < self._cfg.fit_interval:
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

        post = self._enn_model.posterior(x_cand, params=self._enn_params, flags=PosteriorFlags(observation_noise=False))
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


def _calibration_error(imputer: ENNMinusImputer, mu_minus_real: float) -> float:
    if imputer._cfg.target == "delta":
        delta_hat, _delta_se = imputer._predict_y_current()
        delta_real = float(imputer._last_mu_plus) - float(mu_minus_real)
        return abs(float(delta_real) - float(delta_hat))
    mu_hat, _se_hat = imputer.predict_current()
    return abs(float(mu_minus_real) - float(mu_hat))
