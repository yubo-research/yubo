import torch

from third_party.torch_truncnorm.TruncatedNormal import TruncatedNormal

_ygtiw = "You got the indexing wrong"


class StaggerDistribution:
    def __init__(self, X_0, num_samples_per_dimension, seed=None):
        self.device = X_0.device
        self.dtype = X_0.dtype
        self._X_0 = X_0.flatten()
        self._num_dim = len(X_0)
        self._num_samples_per_dimension = num_samples_per_dimension
        self._num_samples = num_samples_per_dimension * self._num_dim
        assert seed is None, "NYI"
        self._rng = None
        # rng = torch.Generator(device=device).manual_seed(seed)

    def _sigma(self, l_min_sigma, l_max_sigma):
        u = l_min_sigma + (l_max_sigma - l_min_sigma) * torch.rand(
            size=torch.Size([self._num_samples_per_dimension, self._num_dim]),
            device=self.device,
            dtype=self.dtype,
            generator=self._rng,
        )
        return torch.exp(u)

    def _index_one_dim_at_a_time(self):
        i = torch.arange(self._num_samples)
        # all 0, then all 1, then ...
        j = torch.repeat_interleave(torch.arange(self._num_dim), repeats=(self._num_samples_per_dimension))
        assert torch.all(j[: self._num_samples_per_dimension] == 0), _ygtiw

        return i, j

    def _pi_and_X_1d(self, l_min_sigma, l_max_sigma):
        sigma = self._sigma(l_min_sigma, l_max_sigma)
        i, j = self._index_one_dim_at_a_time()

        loc = torch.zeros(size=(self._num_samples,), dtype=self.dtype, device=self.device)
        loc[i] = self._X_0[j]

        scale = torch.zeros(size=(self._num_samples,), dtype=self.dtype, device=self.device)
        scale[i] = sigma.T.flatten()
        assert scale[1] == sigma[1, 0], _ygtiw
        assert self._num_dim == 1 or scale[self._num_samples_per_dimension] == sigma[0, 1], _ygtiw

        tn = TruncatedNormal(
            loc=loc,
            scale=scale,
            a=torch.zeros_like(loc),
            b=torch.ones_like(loc),
            # TODO: rng
        )

        pi_and_x = tn.p_and_sample((1,))
        pi = pi_and_x.pi.flatten()
        X_perturbed_dimension = pi_and_x.X.flatten()
        assert X_perturbed_dimension.dtype == self.dtype, (X_perturbed_dimension.dtype, self.dtype)
        assert pi.dtype == self.dtype, (pi.dtype, self.dtype)

        X_perturbed_dimension = X_perturbed_dimension.squeeze()
        pi = pi.squeeze()
        assert X_perturbed_dimension.shape == (self._num_samples,), X_perturbed_dimension.shape
        assert pi.shape == (self._num_samples,)

        X = torch.tile(self._X_0, dims=(self._num_samples, 1))
        assert X.shape == (self._num_samples, self._num_dim), _ygtiw

        X[i, j] = X_perturbed_dimension

        return pi, X

    def propose(self, sigma_min, sigma_max):
        l_min_sigma = torch.log(sigma_min)
        l_max_sigma = torch.log(sigma_max)

        pi, X = self._pi_and_X_1d(l_min_sigma, l_max_sigma)

        assert not torch.any((X.flatten() < 0) | (X.flatten() > 1)), (X.min(), X.max())

        pi = StaggerDistribution.norm_by_dim(pi, self._num_dim, self._num_samples_per_dimension)
        # pi = torch.maximum(torch.tensor(1e-5 / len(pi)), pi)

        assert torch.all(torch.isfinite(pi)), (sigma_min, sigma_max, pi)
        assert torch.all(pi > 0), (sigma_min, sigma_max, pi)
        return pi, X

    @staticmethod
    def norm_by_dim(p, num_dim, num_samples_per_dimension):
        # Treat each dimension as a separate problem.
        # We're only solving them in parallel b/c we
        #  think we can be more efficient with
        #  parallel sampling(ex., w/C++, threads, GPU).

        p_r = p.reshape(num_dim, num_samples_per_dimension)
        assert num_dim == 1 or p_r[1, 0] == p[num_samples_per_dimension]
        norm = p_r.sum(axis=1, keepdim=True)
        p_r = p_r / norm
        i = torch.where(norm == 0)[0]
        p_r[i, :] = 1 / p_r.shape[1]
        p = p_r.reshape(num_dim * num_samples_per_dimension)
        assert num_dim == 1 or p_r[1, 0] == p[num_samples_per_dimension]
        return p
