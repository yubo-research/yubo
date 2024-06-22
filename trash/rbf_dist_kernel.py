import gpytorch
from gpytorch.kernels.kernel import Distance
from gpytorch.kernels.rbf_kernel import RBFKernel


class RBFDistKernelFactory:
    def __init__(self, covar_dist):
        self._covar_dist = covar_dist

    def __call__(self, **kwargs):
        kwargs["covar_dist"] = self._covar_dist
        return _RBFDistKernel(**kwargs)


class _RBFDistKernel(RBFKernel):
    def __init__(self, **kwargs):
        if "covar_dist" in kwargs:
            self._covar_dist = kwargs["covar_dist"]
            del kwargs["covar_dist"]
        super().__init__(**kwargs)
        self._covar_dist_orig = self.covar_dist
        self.covar_dist = self._covar_dist_stub

    def _covar_dist_stub(self, x_1, x_2, **kwargs):
        assert kwargs["square_dist"], kwargs
        assert not kwargs["diag"], kwargs
        assert not kwargs["last_dim_is_batch"], kwargs

        if self._covar_dist is None:
            return self._covar_dist_orig(x_1, x_2, **kwargs)
        else:
            return self._covar_dist(x_1, x_2, **kwargs)
