from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from torch import Tensor

from model.enn_t import EpistemicNearestNeighborsT
from model.enn_weighter_t import ENNWeighterT


class EpistemicNearestNeighborsBoTorchT:
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        k: int = 3,
        var_scale: float = 1.0,
        num_hnsw: Optional[int] = None,
    ) -> None:
        assert train_X.ndim == 2
        if train_Y.ndim == 1:
            train_Y = train_Y.unsqueeze(-1)
        if train_Yvar.ndim == 1:
            train_Yvar = train_Yvar.unsqueeze(-1)
        assert train_Y.shape == train_Yvar.shape
        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self._num_outputs = int(train_Y.shape[-1])
        if num_hnsw is None:
            self._enn = EpistemicNearestNeighborsT(k=k)
        else:
            self._enn = EpistemicNearestNeighborsT(k=k, small_world_M=32, num_hnsw=num_hnsw)
        base_X = train_X.reshape(-1, train_X.shape[-1])
        base_Y = train_Y.reshape(-1, self._num_outputs)
        base_Yvar = train_Yvar.reshape(-1, self._num_outputs)
        self._enn.add(base_X, base_Y, base_Yvar)
        self._train_Yvar = base_Yvar
        self._k: int = int(k)
        self._var_scale: float = float(var_scale)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def tuned_hyperparams(self) -> dict[str, float]:
        return {"k": float(self._k), "var_scale": float(self._var_scale)}

    def set_k(self, k: int) -> None:
        assert isinstance(k, int)
        max_k = max(1, len(self._enn))
        k_clamped = max(1, min(k, max_k))
        self._k = k_clamped
        self._enn.k = k_clamped

    def set_var_scale(self, var_scale: float) -> None:
        assert var_scale > 0
        self._var_scale = float(var_scale)
        self._enn.set_var_scale(self._var_scale)

    def forward(
        self,
        X: Tensor,
        exclude_nearest: bool = False,
        observation_noise: bool = False,
    ) -> MultivariateNormal:
        X2 = X.view(-1, X.shape[-1])
        self._enn.set_var_scale(self._var_scale)
        max_k = max(1, len(self._enn))
        k_int = max(1, min(self._k, max_k))
        mvn_enn = self._enn.posterior(
            X2,
            k=k_int,
            exclude_nearest=exclude_nearest,
            observation_noise=observation_noise,
        )
        mu2 = mvn_enn.mu
        se2 = mvn_enn.se
        mu_flat = mu2.squeeze(-1)
        se_flat = se2.squeeze(-1)
        if not torch.isfinite(mu_flat).all():
            mu_flat = torch.zeros_like(mu_flat)
        if not torch.isfinite(se_flat).all() or (se_flat <= 0).any():
            se_flat = torch.ones_like(se_flat)
        var = se_flat.pow(2)
        covar = DiagLinearOperator(var)
        return MultivariateNormal(mu_flat, covar)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        exclude_nearest: bool = False,
    ) -> Posterior:
        if output_indices is not None:
            assert output_indices == list(range(self._num_outputs))
        mvn = self.forward(X, exclude_nearest=exclude_nearest, observation_noise=observation_noise)
        posterior = GPyTorchPosterior(distribution=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior


class EpistemicNearestNeighborsWeighterBoTorchT:
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        weighting: str,
        k: int = 3,
    ) -> None:
        assert train_X.ndim == 2
        if train_Y.ndim > 1:
            train_Y = train_Y.squeeze(-1)
        train_Y = train_Y[..., None]
        self.train_inputs = (train_X,)
        self.train_targets = train_Y
        self._num_outputs = int(train_Y.shape[-1])
        self._enn = ENNWeighterT(weighting=weighting, k=k)
        self._enn.add(train_X, train_Y)
        self._k: int = int(k)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def forward(
        self,
        X: Tensor,
        exclude_nearest: bool = False,
    ) -> MultivariateNormal:
        X2 = X.view(-1, X.shape[-1])
        mvn_enn = self._enn.posterior(X2, k=self._k, exclude_nearest=exclude_nearest)
        mu2 = mvn_enn.mu
        se2 = mvn_enn.se
        mu_flat = mu2.squeeze(-1)
        se_flat = se2.squeeze(-1)
        if not torch.isfinite(mu_flat).all():
            mu_flat = torch.zeros_like(mu_flat)
        if not torch.isfinite(se_flat).all() or (se_flat <= 0).any():
            se_flat = torch.ones_like(se_flat)
        var = se_flat.pow(2)
        covar = DiagLinearOperator(var)
        return MultivariateNormal(mu_flat, covar)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[list[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[PosteriorTransform] = None,
        exclude_nearest: bool = False,
    ) -> Posterior:
        if observation_noise:
            pass
        if output_indices is not None:
            assert output_indices == list(range(self._num_outputs))
        mvn = self.forward(X, exclude_nearest=exclude_nearest)
        posterior = GPyTorchPosterior(distribution=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior
