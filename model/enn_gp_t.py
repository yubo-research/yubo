from __future__ import annotations

from typing import Optional

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.posteriors.posterior import Posterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from torch import Tensor

from model.enn_t import ENNNormalT, EpistemicNearestNeighborsT


class EpistemicNearestNeighborsGP(BatchedMultiOutputGPyTorchModel, ExactGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
    ) -> None:
        self._validate_tensor_args(X=train_X, Y=train_Y, Yvar=train_Yvar)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        likelihood = GaussianLikelihood()
        num_train = int(train_X.shape[-2])
        num_dim = int(train_X.shape[-1])
        num_outputs = int(train_Y.shape[-1])
        if num_train == 0:
            dummy_X = torch.zeros((1, num_dim), dtype=train_X.dtype, device=train_X.device)
            if num_outputs == 1:
                dummy_targets = torch.zeros(1, dtype=train_Y.dtype, device=train_Y.device)
            else:
                dummy_targets = torch.zeros((1, num_outputs), dtype=train_Y.dtype, device=train_Y.device)
            ExactGP.__init__(self, dummy_X, dummy_targets, likelihood)
        else:
            targets = train_Y
            if targets.size(-1) == 1:
                targets = targets.squeeze(-1)
            ExactGP.__init__(self, train_X, targets, likelihood)

        self._num_outputs = train_Y.shape[-1]
        max_k = int(train_X.shape[-2])
        if max_k == 0:
            k_0 = 1
        else:
            k_0 = min(10, max(1, max_k))
        self._k: int = k_0
        self._var_scale: float = 1.0
        self._enn = EpistemicNearestNeighborsT(k=k_0)
        base_X = train_X.reshape(-1, train_X.shape[-1])
        base_Y = train_Y.reshape(-1, self._num_outputs)
        base_Yvar = train_Yvar.reshape(-1, self._num_outputs)
        self._enn.add(base_X, base_Y, base_Yvar)
        self._train_Yvar = base_Yvar

    def train(self, mode: bool = True) -> "EpistemicNearestNeighborsGP":
        super().train(mode)
        return self

    def eval(self, mode: bool = True) -> "EpistemicNearestNeighborsGP":
        if mode:
            super().eval()
        else:
            super().train()
        return self

    def forward(self, X: Tensor, exclude_nearest: bool = False, observation_noise: bool = False) -> MultivariateNormal:
        X2 = X.view(-1, X.shape[-1])
        self._enn.set_var_scale(self._var_scale)
        max_k = max(1, len(self._enn))
        k_int = max(1, min(self._k, max_k))
        mvn_enn: ENNNormalT = self._enn.posterior(X2, k=k_int, exclude_nearest=exclude_nearest, observation_noise=observation_noise)
        mu2 = mvn_enn.mu
        se2 = mvn_enn.se
        mu_flat = mu2.squeeze(-1)
        se_flat = se2.squeeze(-1)

        if not torch.isfinite(mu_flat).all():
            mu_flat = torch.zeros_like(mu_flat)
        if not torch.isfinite(se_flat).all() or (se_flat <= 0).any():
            se_flat = torch.ones_like(se_flat)

        covar = torch.diag_embed(se_flat.pow(2))
        return MultivariateNormal(mu_flat, covar)

    def posterior(
        self,
        X: Tensor,
        posterior_transform: Optional[PosteriorTransform] = None,
        exclude_nearest: bool = False,
        observation_noise: bool = False,
    ) -> Posterior:
        self.eval()
        X = self.transform_inputs(X)
        mvn = self.forward(X, exclude_nearest=exclude_nearest, observation_noise=observation_noise)
        posterior = GPyTorchPosterior(distribution=mvn)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior

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
