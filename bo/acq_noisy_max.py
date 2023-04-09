from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from torch import Tensor


class AcqNoisyMax(AnalyticAcquisitionFunction):
    def __init__(self, model: Model, q=1, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.noisy_models = [self._get_noisy_model() for _ in range(q)]
        
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
