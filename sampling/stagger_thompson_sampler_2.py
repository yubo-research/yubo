import torch
from botorch.utils.sampling import draw_sobol_normal_samples

from sampling.ray_boundary import ray_boundary

# Maybe perturbing toward a target is better than perturbing in a random direction
#  b/c targets can pull you toward any spot in the space, whereas a random
#  direction can get you stuck in a corner.
# This is discussed in https://proceedings.mlr.press/v238/rashidi24a/rashidi24a.pdf


class StaggerThompsonSampler2:
    def __init__(self, model, X_control, num_samples, no_stagger=False):
        assert len(X_control) == 1, "NYI: multiple control points"
        self._model = model
        self._num_samples = num_samples
        self._no_stagger = no_stagger
        self._X_samples = torch.tile(X_control, dims=(self._num_samples, 1))
        self._num_dim = self._X_samples.shape[-1]
        self.device = self._X_samples.device
        self.dtype = self._X_samples.dtype

    def samples(self):
        return self._X_samples

    def refine(self, num_refinements=1, s_min=1e-6, s_max=1):
        for _ in range(num_refinements):
            self._refine(s_min=s_min, s_max=s_max)

    def improve(self, num_acc_rej, s_min=1e-6, s_max=1):
        assert num_acc_rej == 0, num_acc_rej

    def _stagger(self, s_min, s_max):
        u = torch.rand(size=(self._num_samples, 1))
        if self._no_stagger:
            s = s_min + (s_max - s_min) * u
        else:
            l_s_min = torch.log(torch.tensor(s_min)).to(self._X_samples)
            l_s_max = torch.log(torch.tensor(s_max)).to(self._X_samples)
            s = torch.exp(l_s_min + (l_s_max - l_s_min) * u)
        assert s.max() <= 1, s
        assert s.min() > 0, s
        return s

    def _select_targets(self):
        X_dir = draw_sobol_normal_samples(
            n=self._num_samples,
            d=self._num_dim,
        ).to(self._X_samples)
        X_dir = X_dir / X_dir.norm(dim=-1, keepdim=True)
        assert X_dir.shape == (self._num_samples, self._num_dim), (X_dir.shape, self._num_samples, self._num_dim)
        assert torch.abs(torch.linalg.norm(X_dir, dim=-1).min() - 1) < 1e-6

        X_target = ray_boundary(self._X_samples, X_dir)
        assert X_target.min() >= 0, X_target.min()
        assert X_target.max() <= 1, X_target.max()
        return X_target

    def _thompson_sample(self, X_target, s):
        X_perturbed = self._X_samples + s * (X_target - self._X_samples)
        X_both = torch.cat([self._X_samples, X_perturbed], dim=0)
        assert X_both.shape == (2 * self._num_samples, self._num_dim), (X_both.shape, self._num_samples, self._num_dim)

        mvn = self._model.posterior(X_both)
        Y = mvn.sample(torch.Size([1])).squeeze()
        assert Y.shape == (2 * self._num_samples,), Y.shape
        Y_ts = Y[: self._num_samples]
        Y_perturbed = Y[self._num_samples :]
        i_improved = Y_perturbed > Y_ts
        self._X_samples[i_improved] = X_perturbed[i_improved]
        assert self._X_samples.shape == (self._num_samples, self._num_dim), self._X_samples.shape
        assert self._X_samples.min() >= 0, self._X_samples.min()
        assert self._X_samples.max() <= 1, self._X_samples.max()

    def _refine(self, s_min=1e-6, s_max=1):
        X_target = self._select_targets()
        s = self._stagger(s_min, s_max)
        self._thompson_sample(X_target, s)
