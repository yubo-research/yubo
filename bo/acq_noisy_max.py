import torch

from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform

from torch import Tensor
from torch.quasirandom import SobolEngine
from IPython.core.debugger import set_trace

class AcqNoisyMax(AnalyticAcquisitionFunction):
    def __init__(self, model: Model, num_X_samples=None, q=1, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        if num_X_samples is None:
            self.noisy_models = [self._get_noisy_model() for _ in range(q)]
        else:
            # num_X_samples = 1 + len(model.train_inputs)
            self.noisy_models = [
                self._get_noisy_model_2(num_X_samples)
                for _ in range(q)
            ]

    def _get_noisy_model(self):
        X = self.model.train_inputs[0].detach()
        if len(X) == 0:
            return self.model
        # rsample: one random sample, w/gradient
        # sample: one random sample, w/o gradient; calls rsample()
        # get_posterior_samples: repeated "frozen" random sampling, suitable for
        #  optimization
        Y = self.model.posterior(X, observation_noise=True).sample().squeeze(0).detach()
        model_ts = SingleTaskGP(X, Y, self.model.likelihood)
        model_ts.initialize(**dict(self.model.named_parameters()))
        model_ts.eval()
        return model_ts

    def _get_noisy_model_2(self, num_X_samples):
        X_0 = self.model.train_inputs[0].detach()
        num_dim = X_0.shape[-1]
        dtype = X_0.dtype
        
        sobol_engine = SobolEngine(num_dim, scramble=True)
        X = torch.cat( (X_0, sobol_engine.draw(num_X_samples, dtype=dtype)), axis=0 )

        # rsample: one random sample, w/gradient
        # sample: one random sample, w/o gradient; calls rsample()
        # get_posterior_samples: repeated "frozen" random sampling, suitable for
        #  optimization
        Y = self.model.posterior(X, observation_noise=True).sample().squeeze(0).detach()
        model_ts = SingleTaskGP(X, Y, self.model.likelihood)
        model_ts.initialize(**dict(self.model.named_parameters()))
        model_ts.eval()

        return model_ts

    
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        assert X.shape[-2] == 1, ("NYI q > 1", X.shape)

        self.to(device=X.device)

        return self.noisy_models[0].posterior(X=X, posterior_transform=self.posterior_transform).mean.squeeze()
