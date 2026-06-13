from __future__ import annotations

import numpy as np
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder

from .gaussian_perturbator import GaussianPerturbator
from .uhd_be_enn import acquisition_from_incremental, be_enn_selection_ready
from .uhd_bszo import UHDBSZO
from .uhd_simple_be import _embed_module, _make_be_enn, _tell_be_enn


class UHDBSZOBE:
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
        num_fit_candidates: int = 1,
        num_fit_samples: int = 10,
        enn_index_driver: str = "flat",
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
        self._acquisition = "ucb"

        self._next_perturb_base = 0
        self._current_embed: np.ndarray | None = None

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._be_enn = _make_be_enn(
            enn_k=enn_k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            fit_interval=fit_interval,
            index_driver=enn_index_driver,
        )
        self._enn_model: object | None = None
        self._enn_params: object | None = None

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

    def ask(self) -> None:
        if self._bszo.phase == 0:
            if be_enn_selection_ready(
                obs_count=len(self._zs),
                warmup=self._warmup,
                enn_k=self._enn_k,
                has_params=self._enn_params is not None,
            ):
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
            _tell_be_enn(self, self._current_embed, y_i)

        self._bszo.tell(mu, se)

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
        scores = acquisition_from_incremental(self._be_enn, x_cand, acquisition=self._acquisition)
        best = int(np.argmax(np.abs(scores)))

        self._bszo.set_perturb_base(base + best * k)
        self._next_perturb_base = base + n_cand * k
