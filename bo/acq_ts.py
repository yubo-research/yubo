from contextlib import ExitStack

import gpytorch.settings as gpts
import torch
from botorch.generation import MaxPosteriorSampling
from torch.quasirandom import SobolEngine


class AcqTS:
    """Thompson Sampler
    From https://botorch.org/tutorials/thompson_sampling
    """

    def __init__(self, model, n_candidates=10000, sampler="cholesky"):
        assert sampler in ("cholesky", "ciq", "lanczos")

        self.model = model
        self._n_candidates = n_candidates
        self._sampler = sampler

        X_0 = self.model.train_inputs[0].detach()
        self._num_dim = X_0.shape[-1]
        self._dtype = X_0.dtype
        self._device = X_0.device

    def draw(self, num_arms):
        sobol = SobolEngine(self._num_dim, scramble=True)
        X_samples = sobol.draw(self._n_candidates).to(dtype=self._dtype, device=self._device)

        with ExitStack() as es:
            if self._sampler == "cholesky":
                es.enter_context(gpts.max_cholesky_size(float("inf")))
            elif self._sampler == "ciq":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(True))
                es.enter_context(gpts.minres_tolerance(2e-3))  # Controls accuracy and runtime
                es.enter_context(gpts.num_contour_quadrature(15))
            elif self._sampler == "lanczos":
                es.enter_context(gpts.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True))
                es.enter_context(gpts.max_lanczos_quadrature_iterations(10))
                es.enter_context(gpts.max_cholesky_size(0))
                es.enter_context(gpts.ciq_samples(False))

        with torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self.model, replacement=False)
            X_cand = thompson_sampling(X_samples, num_samples=num_arms)

        return X_cand
