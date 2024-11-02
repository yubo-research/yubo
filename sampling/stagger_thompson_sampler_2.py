import torch
from botorch.generation import MaxPosteriorSampling
from botorch.utils.sampling import draw_sobol_samples

"""


Initialize num_cand points, x, to some guess at x_max.
  - Generate a vector of num_cand distances, s ~ LogUniform.
  - Pick a batch of num_cand points, x_pivot, uniformly randomly (or Sobol), in the (convex) feasible region (ex., bounding box).
  - Set x_cand = x + s*(x_pivot - x)  <== Contraction mapping
  - Take a joint sample over the points x_cand.
  - 

  The contraction mapping results in a point in the feasible region b/c the region is convex. This should be faster
   to compute than the combination of hnr and truncated normal (used in PStarSampler).

"""


class StaggerThompsonSampler2:
    def __init__(self, model, X_control, num_candidates):
        assert len(X_control) == 1, "NYI: multiple control points"
        self._model = model
        self._num_candidates = num_candidates
        self._X_control = torch.tile(X_control, dims=(self._num_candidates, 1))
        self._num_dim = self._X_control.shape[-1]
        self.device = self._X_control.device
        self.dtype = self._X_control.dtype
        self._bounds = torch.tensor([[0.0] * self._num_dim, [1.0] * self._num_dim], device=self.device, dtype=self.dtype)

    def sample(self, num_arms, s_min=1e-6, s_max=1):
        l_s_min = torch.log(torch.tensor(s_min)).to(self._X_control)
        l_s_max = torch.log(torch.tensor(s_max)).to(self._X_control)
        s = torch.exp(l_s_min + (l_s_max - l_s_min) * torch.rand(size=(self._num_candidates, 1)))
        assert s.max() <= 1, s
        assert s.min() > 0, s
        s[0] = 0

        X_pivot = (
            draw_sobol_samples(
                bounds=self._bounds,
                n=self._num_candidates,
                q=1,
            )
            .squeeze(-2)
            .to(self._X_control)
        )
        assert self._X_control.shape == (self._num_candidates, self._num_dim), self._X_control.shape
        assert X_pivot.shape == self._X_control.shape

        X_cand = self._X_control + s * (X_pivot - self._X_control)

        with torch.no_grad():
            thompson_sampling = MaxPosteriorSampling(model=self._model, replacement=False)
            X_arms = thompson_sampling(X_cand, num_samples=num_arms)
        return X_arms
