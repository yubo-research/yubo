import torch
from botorch.utils.sampling import draw_sobol_samples

"""


Initialize num_ts points, x, to a guess at x_max.
Draw from a proposal distribution defined this way:
  - Pick a distance, s ~ LogUniform.
  - Pick a batch of num_ts points, x_proposal, randomly (or Sobol), in the (convex) feasible region (ex., bounding box).
  - Set x_proposal = x + s*(x_proposal-x)  <== Contraction mapping
  - Take a joint sample over the points x and x_proposal.
  - Move each x[i] to x_proposal[i] if x_proposal[i] > x[i].
  - Repeat until done.

  The contraction mapping results in a point in the feasible region b/c the region is convex. This should be faster
   to compute than the combination of hnr and truncated normal (used in PStarSampler). Also, hopefully, it'll converge
   in fewer iterations. Also, there's no need to find or adapt the scale of the normal distribution.

  The proposal points -- the random ones, before contraction -- could all be chosen up front as a large batch of
   num_iterations * num_ts to take advantage of parallel computation. This could be particularly advantageous
   when sampling from a general polytope (eg., when there are inequality constraints).

"""


class StaggerThompsonSampler:
    def __init__(self, model, X_control, num_samples, no_stagger=False):
        assert len(X_control) == 1, "NYI: multiple control points"
        self._model = model
        self._num_samples = num_samples
        self._no_stagger = no_stagger
        self._X_samples = torch.tile(X_control, dims=(self._num_samples, 1))
        self._num_dim = self._X_samples.shape[-1]
        self.device = self._X_samples.device
        self.dtype = self._X_samples.dtype
        self._bounds = torch.tensor([[0.0] * self._num_dim, [1.0] * self._num_dim], device=self.device, dtype=self.dtype)

    def samples(self):
        return self._X_samples

    def refine(self, num_refinements, s_min=1e-6, s_max=1):
        for _ in range(num_refinements):
            self._refine(s_min=s_min, s_max=s_max)

    def improve(self, num_improvements, s_min=1e-6, s_max=1):
        num_improved = 0
        while num_improved < num_improvements:
            num_improved += self._refine(s_min=s_min, s_max=s_max)

    def _refine(self, s_min=1e-6, s_max=1):
        u = torch.rand(size=(self._num_samples, 1))
        if self._no_stagger:
            s = s_min + (s_max - s_min) * u
        else:
            l_s_min = torch.log(torch.tensor(s_min)).to(self._X_samples)
            l_s_max = torch.log(torch.tensor(s_max)).to(self._X_samples)
            s = torch.exp(l_s_min + (l_s_max - l_s_min) * u)
        assert s.max() <= 1, s
        assert s.min() > 0, s

        X_unif = (
            draw_sobol_samples(
                bounds=self._bounds,
                n=self._num_samples,
                q=1,
            )
            .squeeze(-2)
            .to(self._X_samples)
        )
        assert self._X_samples.shape == (self._num_samples, self._num_dim), self._X_samples.shape

        X_perturbed = self._X_samples + s * (X_unif - self._X_samples)
        X_both = torch.cat([self._X_samples, X_perturbed], dim=0)
        assert X_both.shape == (2 * self._num_samples, self._num_dim), (X_both.shape, self._num_samples, self._num_dim)

        mvn = self._model.posterior(X_both)
        Y = mvn.sample(torch.Size([1])).squeeze()
        assert Y.shape == (2 * self._num_samples,), Y.shape
        Y_ts = Y[: self._num_samples]
        Y_perturbed = Y[self._num_samples :]

        i_improved = Y_perturbed > Y_ts

        self._X_samples[i_improved] = X_perturbed[i_improved]
        return i_improved.sum().item()
