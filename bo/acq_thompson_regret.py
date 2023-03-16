from botorch.acquisition.monte_carlo import (
    qSimpleRegret,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from gpytorch.mlls import ExactMarginalLogLikelihood


class AcqThompsonRegret(qSimpleRegret):
    def __init__(self, model: Model, **kwargs) -> None:
        super().__init__(model=self._thompson_sample_model(model), **kwargs)

    def _thompson_sample_model(self, model):
        x = model.train_inputs[0].detach()
        if len(x) == 0:
            return model
        # y = model.posterior(x).mean.detach()
        y = model.posterior(x).rsample().squeeze(0).detach()

        # Should we standardize again?
        gp_ts = SingleTaskGP(x, y)
        mll = ExactMarginalLogLikelihood(gp_ts.likelihood, gp_ts)
        fit_gpytorch_mll(mll)
        return gp_ts
