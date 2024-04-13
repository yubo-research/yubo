import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_normal_samples
from scipy.optimize import minimize

from acq.acq_util import find_max


class _FunJac:
    def __init__(self, an, include_jacobian=True):
        self._an = an
        self._X = None
        self._include_jacobian = include_jacobian

    def fun(self, x):
        self._x = x
        x = torch.tensor(x, requires_grad=self._include_jacobian)
        loss = self._an.evalutate(x)
        if self._include_jacobian:
            loss.backward()
            self._grad = x.grad.detach().numpy()

        loss = loss.detach().numpy()
        return loss

    def jac(self, x):
        assert np.all(x == self._x)
        return self._grad


def appx_normal(model, num_X_samples, num_Y_samples=128, num_tries=30, use_gradients=True, seed=None):
    mu = find_max(model)
    an = AppxNormal(model, mu, num_X_samples, num_Y_samples, use_soft_max=use_gradients)

    fun_jac = _FunJac(an, include_jacobian=use_gradients)
    rng = np.random.default_rng(seed)

    f_min = 1e99
    x_best = None
    for _ in range(num_tries):
        sigma_0 = torch.tensor(rng.uniform(size=(an.num_dim,)), dtype=an.dtype)
        res = minimize(
            x0=sigma_0,
            method="L-BFGS-B" if use_gradients else "Powell",
            fun=fun_jac.fun,
            jac=fun_jac.jac if use_gradients else None,
            bounds=[torch.tensor((0.0, 2.0), dtype=an.dtype)] * an.num_dim,
        )
        if res.success:
            if res.fun < f_min:
                f_min = res.fun
                x_best = res.x
        else:
            print("Failed")
    sigma = torch.tensor(x_best)
    print("MAX_IW:", an.importance_weights(sigma).max())
    return mu, sigma


class AppxNormal:
    def __init__(
        self,
        model: SingleTaskGP,
        mu: torch.tensor,
        num_X_samples,
        num_Y_samples,
        use_soft_max=True,
    ):
        self._model = model
        self._mu = mu
        X = model.train_inputs[0]
        self._use_soft_max = use_soft_max
        assert len(X.shape) == 2, (X.shape, "I only handle single-output models")
        self.num_dim = X.shape[1]
        self.device = X.device
        self.dtype = X.dtype

        self._X_base_samples = draw_sobol_normal_samples(
            self.num_dim,
            num_X_samples,
            device=self.device,
        )
        self._p_x = torch.exp(-(self._X_base_samples**2).sum(dim=1) / 2)
        self._p_x = self._p_x / self._p_x.sum()
        assert self._p_x.shape == (num_X_samples,), (self._p_x.shape, num_X_samples)

        self._sampler_y = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

    def importance_weights(self, sigma):
        p_star = self._mk_p_star(self._resample_x(self._mu, sigma))
        assert p_star.shape == self._p_x.shape, (p_star.shape, self._p_x.shape)
        return p_star / self._p_x

    def evalutate(self, sigma):
        importance_weights = self.importance_weights(sigma)
        # We don't mind if importance_weight <1. That
        #  just means there is enough support in p_x
        #  for p_star.
        # return importance_weights.amax()
        importance_weights[importance_weights < 1] = 1
        return ((importance_weights - 1) ** 2).sum()

    def _resample_x(self, mu, sigma):
        return mu + sigma * self._X_base_samples

    def _mk_p_star(self, X):
        mvn = self._model.posterior(X, observation_noise=False)
        # calls rsample_from_base_samples(), enabling SAA
        Y = self._sampler_y(mvn).squeeze(-1)

        if self._use_soft_max:
            # Differentiable
            z = torch.exp(20 * Y)
            z = z / z.sum(dim=1, keepdim=True)
        else:
            # Not differentiable
            z = torch.zeros(size=Y.shape)
            i = Y.max(dim=1).indices
            j = torch.arange(128)
            z[j, i] = 1

        p_star = z.mean(dim=0)
        p_star = p_star / p_star.sum()
        return p_star
