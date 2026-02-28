from __future__ import annotations

import numpy as np
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder

from .gaussian_perturbator import GaussianPerturbator
from .uhd_bszo import UHDBSZO
from .uhd_simple_be import _embed_module, _maybe_fit_enn, _predict_enn


class UHDBSZOBE:
    """BSZO with ENN-based perturbation seed selection.

    Before each gradient step, generates N candidate perturbation seed
    sets, embeds the module under each candidate's first perturbation
    direction, and selects the candidate with highest gradient-magnitude
    UCB.

    The ENN is trained on (embedding_under_perturbation, directional_derivative)
    pairs from all previous gradient steps.
    """

    def __init__(
        self,
        perturbator: GaussianPerturbator,
        dim: int,
        module: nn.Module,
        embedder: BehavioralEmbedder,
        *,
        epsilon: float = 1e-4,
        k: int = 2,
        sigma_p_sq: float = 1.0,
        sigma_e_sq: float = 1.0,
        alpha: float = 0.1,
        lr: float = 0.001,
        num_candidates: int = 10,
        warmup: int = 20,
        fit_interval: int = 10,
        enn_k: int = 25,
    ):
        from .lr_scheduler import ConstantLR

        self._bszo = UHDBSZO(
            perturbator,
            dim,
            lr_scheduler=ConstantLR(lr),
            epsilon=epsilon,
            k=k,
            sigma_p_sq=sigma_p_sq,
            sigma_e_sq=sigma_e_sq,
            alpha=alpha,
        )
        self._module = module
        self._embedder = embedder
        self._num_candidates = num_candidates
        self._warmup = warmup
        self._fit_interval = fit_interval
        self._enn_k = enn_k

        self._next_perturb_base = 0
        self._current_embed: np.ndarray | None = None

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._enn_model: object | None = None
        self._enn_params: object | None = None
        self._posterior_flags: object | None = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._num_new_since_fit = 0

    # ------------------------------------------------------------------
    # Delegated properties
    # ------------------------------------------------------------------

    @property
    def eval_seed(self) -> int:
        return self._bszo.eval_seed

    @property
    def sigma(self) -> float:
        return self._bszo.epsilon

    @property
    def y_best(self) -> float | None:
        return self._bszo.y_best

    @property
    def mu_avg(self) -> float:
        return self._bszo.mu_avg

    @property
    def se_avg(self) -> float:
        return self._bszo.se_avg

    @property
    def phase(self) -> int:
        return self._bszo.phase

    @property
    def k(self) -> int:
        return self._bszo.k

    # ------------------------------------------------------------------
    # Ask / Tell
    # ------------------------------------------------------------------

    def ask(self) -> None:
        if self._bszo.phase == 0:
            if self._enn_params is not None and len(self._zs) >= self._warmup:
                self._select_seeds()
            else:
                self._bszo.set_perturb_base(self._next_perturb_base)
                self._next_perturb_base += self._bszo.k

        self._bszo.ask()

        if self._bszo.phase >= 1:
            self._current_embed = _embed_module(self._module, self._embedder)

    def tell(self, mu: float, se: float) -> None:
        phase_before = self._bszo.phase

        if phase_before >= 1 and self._current_embed is not None:
            y_i = (mu - self._bszo.baseline_mu) / self._bszo.epsilon
            self._zs.append(self._current_embed)
            self._ys.append(y_i)
            self._num_new_since_fit += 1

        self._bszo.tell(mu, se)

        if phase_before >= 1 and self._bszo.phase == 0:
            self._maybe_fit()

    # ------------------------------------------------------------------
    # ENN seed selection
    # ------------------------------------------------------------------

    def _select_seeds(self) -> None:
        base = self._next_perturb_base
        k = self._bszo.k
        n_cand = self._num_candidates
        perturbator = self._bszo.perturbator
        epsilon = self._bszo.epsilon

        was_training = self._module.training
        self._module.eval()

        embeds = []
        for j in range(n_cand):
            seed = base + j * k
            perturbator.perturb(seed, epsilon)
            embeds.append(self._embedder.embed(self._module).cpu().numpy().astype(np.float64))
            perturbator.unperturb()

        if was_training:
            self._module.train()

        x_cand = np.array(embeds, dtype=np.float64)
        mu_pred, se_pred = _predict_enn(self._enn_model, self._enn_params, self._posterior_flags, x_cand)

        mu_real = self._y_mean + self._y_std * mu_pred
        se_real = abs(self._y_std) * se_pred
        ucb = np.abs(mu_real) + se_real
        best = int(np.argmax(ucb))

        self._bszo.set_perturb_base(base + best * k)
        self._next_perturb_base = base + n_cand * k

    def _maybe_fit(self) -> None:
        _maybe_fit_enn(self)
