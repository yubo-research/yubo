import logging
import time

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.model import Model
from botorch.posteriors.torch import TorchPosterior
from botorch.sampling.base import MCSampler
from torch.distributions import Normal

import acq.fit_gp as fit_gp
from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate
from optimizer.sobol_designer import SobolDesigner

_logger = logging.getLogger(__name__)


class DNGOModel(Model):
    """BoTorch Model wrapper for DNGOSurrogate."""

    def __init__(
        self,
        dngo_surrogate: DNGOSurrogate,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> None:
        super().__init__()
        self._dngo = dngo_surrogate
        self._train_X = train_X
        self._train_Y = train_Y

    @property
    def num_outputs(self) -> int:
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        return torch.Size([])

    @property
    def _input_batch_shape(self) -> torch.Size:
        return torch.Size([])

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform=None,
    ) -> TorchPosterior:
        X_np = X.detach().cpu().numpy()
        original_shape = X_np.shape[:-1]
        X_flat = X_np.reshape(-1, X_np.shape[-1])

        mean_np, var_np = self._dngo.predict(X_flat)

        mean_tensor = torch.tensor(mean_np, dtype=X.dtype, device=X.device)
        std_tensor = torch.tensor(var_np, dtype=X.dtype, device=X.device).sqrt()

        mean_tensor = mean_tensor.view(*original_shape, 1)
        std_tensor = std_tensor.view(*original_shape, 1)

        dist = Normal(mean_tensor, std_tensor)
        return TorchPosterior(distribution=dist)


class IIDNormalSampler(MCSampler):
    """Simple MC sampler for TorchPosterior with Normal distribution."""

    def __init__(self, sample_shape: torch.Size) -> None:
        super().__init__(sample_shape=sample_shape)

    def forward(self, posterior: TorchPosterior) -> torch.Tensor:
        dist = posterior.distribution
        return dist.rsample(self.sample_shape)


class DNGODesigner:
    """Bayesian Optimization designer using DNGO surrogate."""

    def __init__(
        self,
        policy,
        *,
        num_candidates: int = 1000,
        num_mc_samples: int = 128,
        init_sobol: int = 1,
    ) -> None:
        self._policy = policy
        self._num_candidates = num_candidates
        self._num_mc_samples = num_mc_samples
        self._init_sobol = init_sobol

    def __call__(self, data, num_arms, *, telemetry=None):
        dtype = torch.double
        device = torch.device("cpu")

        if len(data) < self._init_sobol:
            sobol = SobolDesigner(self._policy.clone())
            return sobol(data, num_arms, telemetry=telemetry)

        t0 = time.perf_counter()

        Y, X = fit_gp.extract_X_Y(data, dtype, device)

        X_np = X.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy().flatten()

        dngo = DNGOSurrogate(DNGOConfig())
        try:
            dngo.fit(X_np, Y_np)
        except RuntimeError as e:
            _logger.warning("DNGO fitting failed (%s), falling back to Sobol", e)
            sobol = SobolDesigner(self._policy.clone())
            return sobol(data, num_arms, telemetry=telemetry)

        dt_fit = time.perf_counter() - t0
        if telemetry is not None:
            telemetry.set_dt_fit(dt_fit)

        t0 = time.perf_counter()

        model = DNGOModel(dngo, X, Y)

        sampler = IIDNormalSampler(sample_shape=torch.Size([self._num_mc_samples]))

        acq_fn = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X,
            sampler=sampler,
            cache_root=False,
        )

        num_dim = X.shape[-1]
        X_cand = torch.rand(self._num_candidates, num_arms, num_dim, dtype=dtype, device=device)

        with torch.no_grad():
            acq_values = acq_fn(X_cand)

        best_idx = acq_values.argmax()
        X_best = X_cand[best_idx]

        dt_select = time.perf_counter() - t0
        if telemetry is not None:
            telemetry.set_dt_select(dt_select)

        return fit_gp.mk_policies(self._policy, X_best)
