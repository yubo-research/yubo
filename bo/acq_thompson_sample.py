from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor


class AcqThompsonSample(AnalyticAcquisitionFunction):
    def __init__(self, model: Model, **kwargs) -> None:
        super().__init__(model=self._thompson_sample_model(model), **kwargs)

    def _thompson_sample_model(self, model):
        x = model.train_inputs[0].detach()
        if len(x) == 0:
            return model
        y = model.posterior(x).rsample().squeeze(0).detach()

        # Should we standardize again?
        gp_ts = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(gp_ts.likelihood, gp_ts)
        fit_gpytorch_mll(mll)
        return gp_ts

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        assert X.shape[-2] == 1, ("NYI q > 1", X.shape)

        self.to(device=X.device)

        posterior = self.model.posterior(X=X, posterior_transform=self.posterior_transform)
        Y = posterior.mean.squeeze()
        return Y
