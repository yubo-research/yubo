import gpytorch
import torch

# GPyTorch does numerical calculations that
#  require many distance calculations at non-observed, non-predicting points.
# So we can't use it.


class _GPModel(gpytorch.models.ExactGP):
    def __init__(self, kernel, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel(ard_num_dims=None, lengthscale_prior=None)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    def __init__(self, kernel, train_x, train_y):
        self._train_model(kernel, train_x, train_y)

    def __call__(self, x):
        x = torch.atleast_2d(torch.as_tensor(x))
        with torch.no_grad():
            # TODO: optionally return just the epistemic uncertainty
            return self._likelihood(self._model(x))

    def _train_model(self, kernel, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = _GPModel(kernel, train_x, train_y, likelihood)
        model.train()
        likelihood.train()
        self._train(model, likelihood, train_x, train_y, num_iterations=1000)

        model.eval()
        likelihood.eval()
        self._model = model
        self._likelihood = likelihood

    def _train(self, model, likelihood, train_x, train_y, num_iterations):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        for i in range(num_iterations):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
