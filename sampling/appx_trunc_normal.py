import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_normal_samples
from scipy.optimize import minimize

from acq.acq_util import find_max
from sampling.mv_truncated_normal import MVTruncatedNormal


def appx_trunc_normal(
    model,
    num_X_samples,
    *,
    num_Y_samples=256,
    num_tries=10,
    use_gradients=True,
    min_sigma=1e-9,
    max_sigma=3.0,
):
    mu = find_max(model)
    an = _AppxTruncNormal(model, mu, num_X_samples, num_Y_samples, use_soft_max=use_gradients)
    an_no_grad = _AppxTruncNormal(model, mu, num_X_samples, num_Y_samples, use_soft_max=False)

    fun_jac = _FunJac(an, include_jacobian=use_gradients)

    f_min = 1e99
    x_best = None
    for _ in range(num_tries):
        # TODO: parallelize into num_threads threads?
        sigma_0 = 0.1 / np.sqrt(an.num_dim)
        res = minimize(
            x0=sigma_0,
            method="L-BFGS-B" if use_gradients else "Powell",
            fun=fun_jac.fun,
            jac=fun_jac.jac if use_gradients else None,
            bounds=[(min_sigma, max_sigma)],
        )
        if res.success:
            if use_gradients:
                sigma = res.x
                f_check = an_no_grad.evaluate(torch.tensor(sigma))
            else:
                f_check = res.fun
            if f_check < f_min:
                f_min = f_check
                x_best = res.x
        else:
            pass

    sigma = float(x_best)
    # print("F_MIN:", f_min)
    # print("MU:", mu)
    # print("SIGMA:", sigma)

    return AppxTruncNormal(model, mu, sigma, num_Y_samples=num_Y_samples)


class AppxTruncNormal:
    def __init__(
        self,
        model: SingleTaskGP,
        mu: torch.Tensor,
        sigma: float,
        num_Y_samples: int,
    ):
        self._model = model
        self.mu = mu.flatten()
        self.sigma = sigma
        self._sigma = sigma * _shape(model)
        self._num_dim = len(self.mu)
        self._num_Y_samples = num_Y_samples

        self.device = self.mu.device

    def sample(self, num_X_samples):
        return self.mu + self._sigma * draw_sobol_normal_samples(
            self._num_dim,
            num_X_samples,
            device=self.device,
        )

    def p_normal(self, X):
        Z = (X - self.mu) / self._sigma
        p_normal = torch.exp(-(Z**2).sum(dim=1) / 2)
        return p_normal / p_normal.sum()

    def p_star(self, X, num_Y_samples):
        mvn = self._model.posterior(X, observation_noise=False)
        Y = mvn.sample(torch.Size([num_Y_samples])).squeeze(-1)

        return _calc_pstar(Y, beta_soft_max=None)

    def calc_importance_weights(self, X):
        p_star = self.p_star(X, num_Y_samples=self._num_Y_samples)
        p_normal = self.p_normal(X)
        assert torch.all(p_normal > 0), p_normal
        w = p_star / p_normal
        assert not torch.any(torch.isnan(w)), w
        return w / w.mean()


class _AppxTruncNormal:
    def __init__(
        self,
        model: SingleTaskGP,
        mu,
        num_X_samples,
        num_Y_samples,
        use_soft_max,
    ):
        self._model = model
        self._mu = mu
        self._shape = _shape(model)
        X = model.train_inputs[0]
        self._use_soft_max = use_soft_max
        assert len(X.shape) == 2, (X.shape, "I only handle single-output models")
        self.num_dim = X.shape[1]
        self._beta_soft_max = 1
        self.device = X.device
        self.dtype = X.dtype
        self._num_X_samples = num_X_samples
        self._sampler_y = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

    def _sample_trunc_normal(self, sigma):
        # TODO: Try sobol samples in TruncatedNormal
        tn = MVTruncatedNormal(self._mu, sigma)
        X = tn.rsample(torch.Size([self._num_X_samples]))
        up = tn.unnormed_prob(X)
        p = up / up.sum()
        return X, p

    def evaluate(self, sigma):
        X, p_normal = self._sample_trunc_normal(sigma * self._shape)
        assert torch.all(p_normal > 0), p_normal
        p_star = self._mk_p_star(X)

        # i = torch.where(p_star > 0)[0]
        # kl_i = p_star[i] * torch.log(p_star[i] / p_normal[i])
        # One more thing: We're sampling from the p_normal distribution,
        # (not from a uniform distribution)
        #  so we need an importance weight of 1/p_normal.
        # kl = torch.sum(kl_i / p_normal[i])

        # w = p_star / p_normal
        # w = w / w.sum()
        # return -torch.abs(torch.sum(w * torch.log(w)))
        # return ((w - 1) ** 2).sum()
        # print("E:", p_star - p_normal)
        return ((p_star - p_normal) ** 2).mean()

    def _mk_p_star(self, X):
        mvn = self._model.posterior(X, observation_noise=False)
        # calls rsample_from_base_samples(), enabling SAA
        Y = self._sampler_y(mvn).squeeze(-1)
        return _calc_pstar(Y, self._beta_soft_max if self._use_soft_max else None)


def _shape(model):
    # Use fitted covar for shape, a la TuRBO.
    shape = model.covar_module.base_kernel.lengthscale.detach()
    num_dim = len(shape)
    shape = shape / shape.mean()
    shape = shape / (torch.prod(shape) ** (1 / num_dim))
    return shape


def _calc_pstar(Y, beta_soft_max):
    if beta_soft_max:
        # Differentiable
        z = torch.exp(beta_soft_max * Y)
        z = z / z.sum(dim=1, keepdim=True)
        assert not torch.any(torch.isnan(z)), z
    else:
        # Not differentiable
        z = torch.zeros(size=Y.shape)
        i = Y.max(dim=1).indices
        j = torch.arange(z.shape[0])
        z[j, i] = 1

    p_star = z.mean(dim=0)
    assert not torch.any(torch.isnan(p_star)), p_star
    norm = p_star.sum()
    assert torch.all(norm > 0)
    return p_star / norm


class _FunJac:
    def __init__(self, an, include_jacobian=True):
        self._an = an
        self._X = None
        self._include_jacobian = include_jacobian

    def fun(self, x):
        assert not np.any(np.isnan(x)), x
        self._x = x
        x = torch.tensor(x, requires_grad=self._include_jacobian)
        loss = self._an.evaluate(x)
        if self._include_jacobian:
            loss.backward()
            self._grad = x.grad.detach().numpy()

        loss = loss.detach().numpy()
        return loss

    def jac(self, x):
        assert np.all(x == self._x)
        return self._grad
