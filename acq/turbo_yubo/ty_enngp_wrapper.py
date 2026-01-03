class ENNGPWrapper:
    def __init__(self, model):
        self._model = model

    def posterior(self, X):
        posterior = self._model.posterior(X)
        mvn = posterior.distribution

        class _PosteriorWrapper:
            def __init__(self, mvn_dist):
                self._mvn = mvn_dist

            @property
            def mu(self):
                mu = self._mvn.mean
                if mu.dim() == 1:
                    mu = mu.unsqueeze(-1)
                return mu

            @property
            def se(self):
                se = self._mvn.variance.sqrt()
                if se.dim() == 1:
                    se = se.unsqueeze(-1)
                return se

            def sample(self, sample_shape):
                return self._mvn.sample(sample_shape)

        return _PosteriorWrapper(mvn)

    def __getattr__(self, name):
        return getattr(self._model, name)
