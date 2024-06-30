import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples

# Sample uniformly from a box of side s.
# But s isn't a constant, it's s ~ Stagger = exp(log_s_min + log_s_max*Sobol[0,1]) ~ loguniform(x)
# Naive way: x ~ Sobol in [0,1]^2, then TS y(x) & take the best. But the x's are too sparse to find the maximizer.
#
# TODO
#   x_c ~ TS(control points)
#   s ~ Stagger
#   x ~ Sobol(box side s that includes x_c at the same relative position as the original box)
#   pi(x) = 1 / s^d

#   x_s ~ Sobol([0,1]^2)
#   x = x_c + s*(x_s - x_c)
# x is always in bounds.
#
# This is a StaggerSobol proposal. Importance-sample x from it using
#  n = int(1/pi(x) + .5). Then TS y(x). You should get a more precise Thompson
# sample than the naive way does b/c you have more points near the maximizer.
#
# Include X_control in the proposal w/weight 1/num_samples


class _StaggerSobolISSampler:
    def __init__(self, pi, X):
        assert len(pi) == len(X), len(pi) == len(X)
        self._X = X
        self._pi = pi

    def ask(self, num_samples):
        """
        Return points that are appropriate for use in a Thompson Sample.
        The points are from a stagger about the control point(s), but
         their frequency is from a uniform distribution over the bounding box (feasible region).
        """
        i = np.arange(len(self._X))
        ips = 1 / self._pi
        ips = ips / ips.sum()
        j = np.random.choice(i, size=(num_samples,), replace=True, p=ips)
        return self._X[j, :]

    def tell(self, p_target, w_max=np.inf):
        assert len(p_target) == len(self._X), len(p_target) == len(self._X)
        p_target = p_target / p_target.sum()
        w = p_target / self._pi
        w = torch.min(torch.tensor(w_max), w)
        self._w = w / w.sum()

    def importance_sample(self, num_samples):
        i = np.arange(len(self._X))
        i = np.random.choice(i, size=(num_samples,), p=self._w)
        return self._X[i]


class StaggerSobol:
    def __init__(self, X_control):
        self._X_control = torch.atleast_2d(X_control)
        self._num_dim = self._X_control.shape[-1]
        self.device = X_control.device
        self.dtype = X_control.dtype

    def get_sampler(self, num_proposal_points, s_min=1e-6, s_max=1):
        num_sobol = num_proposal_points - 1
        X = draw_sobol_samples(
            bounds=torch.tensor([[0.0] * self._num_dim, [1.0] * self._num_dim], device=self.device, dtype=self.dtype),
            n=num_sobol,
            q=1,
        ).squeeze(-2)

        l_s_min = torch.log(torch.tensor(s_min))
        l_s_max = torch.log(torch.tensor(s_max))
        s = torch.exp(l_s_min + (l_s_max - l_s_min) * torch.rand(size=(num_sobol, 1)))
        assert s.max() <= 1, s
        assert s.min() > 0, s

        X_stagger = self._X_control + s * (X - self._X_control)
        pi_stagger = (1 / s**2).flatten()
        pi_stagger = pi_stagger / pi_stagger.sum()

        X_proposal = torch.cat((self._X_control, X_stagger), dim=0)
        pi_proposal = torch.cat((torch.tensor([1.0 / num_proposal_points]), pi_stagger))
        pi_proposal = pi_proposal / pi_proposal.sum()

        return _StaggerSobolISSampler(pi_proposal, X_proposal)
