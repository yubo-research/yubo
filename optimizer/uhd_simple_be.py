from __future__ import annotations

import numpy as np
from torch import nn

from embedding.behavioral_embedder import BehavioralEmbedder

from .gaussian_perturbator import GaussianPerturbator
from .uhd_be_enn import IncrementalBEEnn
from .uhd_mezo_be_ask_shared import run_mezo_be_ask
from .uhd_simple_base import UHDSimpleBase


def _embed_module(module: nn.Module, embedder: BehavioralEmbedder) -> np.ndarray:
    was_training = module.training
    module.eval()
    z = embedder.embed(module)
    if was_training:
        module.train()
    return z.cpu().numpy().astype(np.float64)


def _predict_enn(enn_model, enn_params, x_cand: np.ndarray):
    from enn.enn.enn_params import PosteriorFlags

    post = enn_model.posterior(x_cand, params=enn_params, flags=PosteriorFlags(observation_noise=False))
    return np.asarray(post.mu).reshape(-1), np.asarray(post.se).reshape(-1)


def _make_be_enn(
    *,
    enn_k: int,
    num_fit_candidates: int = 1,
    num_fit_samples: int = 10,
    index_driver: str = "flat",
) -> IncrementalBEEnn:
    return IncrementalBEEnn(
        k=int(enn_k),
        num_fit_candidates=num_fit_candidates,
        num_fit_samples=num_fit_samples,
        index_driver=index_driver,
        rng=np.random.default_rng(0),
    )


def _tell_be_enn(obj, z: np.ndarray, y: float) -> None:
    obj._be_enn.add_obs(z, y)
    obj._enn_model = obj._be_enn.model
    obj._enn_params = obj._be_enn.params


class UHDSimpleBE(UHDSimpleBase):
    def __init__(
        self,
        perturbator: GaussianPerturbator,
        sigma_0: float,
        dim: int,
        module: nn.Module,
        embedder: BehavioralEmbedder,
        *,
        num_candidates: int = 10,
        warmup: int = 20,
        fit_interval: int = 10,
        enn_k: int = 25,
        num_fit_candidates: int = 1,
        num_fit_samples: int = 10,
        enn_index_driver: str = "flat",
        sigma_range: tuple[float, float] | None = None,
        adapt_sigma: bool = True,
    ):
        super().__init__(perturbator, sigma_0, dim, sigma_range=sigma_range, adapt_sigma=adapt_sigma)
        self._module = module
        self._embedder = embedder
        self._num_candidates = num_candidates
        self._warmup = warmup
        self._fit_interval = fit_interval  # ignored; kept for API compat
        self._enn_k = enn_k

        self._next_seed = 0

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._be_enn = _make_be_enn(
            enn_k=enn_k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            index_driver=enn_index_driver,
        )
        self._enn_model: object | None = None
        self._enn_params: object | None = None

    def ask(self) -> None:
        if self._enn_params is not None and len(self._zs) >= self._warmup:
            self._eval_seed, self._z_current = self._select_seed()
            # Resume the sequential seed stream after the chosen candidate (do not skip the whole batch).
            self._next_seed = self._eval_seed + 1
        else:
            self._eval_seed = self._next_seed
            self._next_seed += 1
            self._perturbator.perturb(self._eval_seed, self._adapter.sigma)
            self._z_current = _embed_module(self._module, self._embedder)

    def tell(self, mu: float, se: float) -> None:
        self._mu_prev = mu
        self._se_prev = se

        self._zs.append(self._z_current)
        self._ys.append(mu)
        _tell_be_enn(self, self._z_current, mu)

        self._accept_or_reject(mu)

    def _select_seed(self) -> tuple[int, np.ndarray]:
        base = self._next_seed
        was_training = self._module.training
        self._module.eval()

        sigmas = self._sample_sigmas(base, self._num_candidates)
        zs = []
        for i in range(self._num_candidates):
            self._perturbator.perturb(base + i, float(sigmas[i]))
            zs.append(self._embedder.embed(self._module).cpu().numpy().astype(np.float64))
            self._perturbator.unperturb()

        if was_training:
            self._module.train()

        x_cand = np.array(zs, dtype=np.float64)
        mu_pred, se_pred = _predict_enn(self._enn_model, self._enn_params, x_cand)
        ucb = mu_pred + se_pred
        best = int(np.argmax(ucb))

        self._perturbator.perturb(base + best, float(sigmas[best]))
        return base + best, zs[best]


class UHDMeZOBE:
    def __init__(
        self,
        perturbator: GaussianPerturbator,
        dim: int,
        module: nn.Module,
        embedder: BehavioralEmbedder,
        *,
        sigma: float = 0.001,
        lr: float = 0.001,
        beta: float = 0.9,
        num_candidates: int = 10,
        warmup: int = 20,
        fit_interval: int = 10,
        enn_k: int = 25,
        num_fit_candidates: int = 1,
        num_fit_samples: int = 10,
        enn_index_driver: str = "flat",
    ):
        from .lr_scheduler import ConstantLR
        from .uhd_mezo import UHDMeZO

        self._mezo = UHDMeZO(
            perturbator,
            dim,
            lr_scheduler=ConstantLR(lr),
            sigma=sigma,
            beta=beta,
        )
        self._module = module
        self._embedder = embedder
        self._num_candidates = num_candidates
        self._warmup = warmup
        self._fit_interval = fit_interval
        self._enn_k = enn_k

        self._selected = False
        self._z_plus: np.ndarray | None = None
        self._z_minus: np.ndarray | None = None

        self._zs: list[np.ndarray] = []
        self._ys: list[float] = []
        self._be_enn = _make_be_enn(
            enn_k=enn_k,
            num_fit_candidates=num_fit_candidates,
            num_fit_samples=num_fit_samples,
            index_driver=enn_index_driver,
        )
        self._enn_model: object | None = None
        self._enn_params: object | None = None

    @property
    def eval_seed(self) -> int:
        return self._mezo.eval_seed

    @property
    def sigma(self) -> float:
        return self._mezo.sigma

    @property
    def y_best(self) -> float | None:
        return self._mezo.y_best

    @property
    def mu_avg(self) -> float:
        return self._mezo.mu_avg

    @property
    def se_avg(self) -> float:
        return self._mezo.se_avg

    @property
    def positive_phase(self) -> bool:
        return self._mezo.positive_phase

    def skip_negative(self) -> None:
        self._mezo.skip_negative()

    def ask(self) -> None:
        run_mezo_be_ask(self, embed_unselected=lambda: _embed_module(self._module, self._embedder))

    def tell(self, mu: float, se: float) -> None:
        is_positive = self._mezo.positive_phase
        z = self._z_plus if is_positive else self._z_minus
        self._zs.append(z)
        self._ys.append(mu)
        self._mezo.tell(mu, se)
        if not is_positive:
            _tell_be_enn(self, z, mu)

    def _select_seed(self) -> tuple[int, np.ndarray, np.ndarray]:
        base = self._mezo.eval_seed
        sigma = self._mezo.sigma
        perturbator = self._mezo.perturbator
        was_training = self._module.training
        self._module.eval()

        z_plus_list = []
        z_minus_list = []
        for i in range(self._num_candidates):
            perturbator.perturb(base + i, sigma)
            z_plus_list.append(self._embedder.embed(self._module).cpu().numpy().astype(np.float64))
            perturbator.unperturb()

            perturbator.perturb(base + i, -sigma)
            z_minus_list.append(self._embedder.embed(self._module).cpu().numpy().astype(np.float64))
            perturbator.unperturb()

        if was_training:
            self._module.train()

        z_plus_arr = np.array(z_plus_list, dtype=np.float64)
        z_minus_arr = np.array(z_minus_list, dtype=np.float64)
        mu_plus, se_plus = _predict_enn(self._enn_model, self._enn_params, z_plus_arr)
        mu_minus, se_minus = _predict_enn(self._enn_model, self._enn_params, z_minus_arr)

        two_sigma = 2.0 * sigma
        g = (mu_plus - mu_minus) / two_sigma
        seg = np.sqrt(se_plus**2 + se_minus**2) / two_sigma

        ucb = np.abs(g) + seg
        best = int(np.argmax(ucb))

        return base + best, z_plus_list[best], z_minus_list[best]
