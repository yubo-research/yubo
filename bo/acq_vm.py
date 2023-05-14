import torch

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.acquisition import PosteriorMean
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from botorch.sampling.normal import SobolQMCNormalSampler

from torch import Tensor
from torch.quasirandom import SobolEngine
from IPython.core.debugger import set_trace

class AcqVM(MCAcquisitionFunction):
    """ Soft entropy """
    def __init__(self, model: Model, num_X_samples, num_Y_samples, **kwargs) -> None:
        super().__init__(model=model, **kwargs)
        self.sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_Y_samples]))

        X_0 = self.model.train_inputs[0].detach()
        num_dim = X_0.shape[-1]
        dtype = X_0.dtype

        sobol_engine = SobolEngine(num_dim, scramble=True)
        self.X_samples = sobol_engine.draw(num_X_samples, dtype=dtype)
    
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.
        """
        self.to(device=X.device)

        mvn_0 = self.model(X)
        mu_0 = mvn_0.mean
        sg_0 = mvn_0.stddev
        vr_0 = mvn_0.variance

        mvn = self.model(self.X_samples)
        mu = mvn.mean
        sg = mvn.stddev
        vr = mvn.variance
        
        mvn = self.model.posterior(X)
        model_f = self.model.condition_on_observations(X=X, Y=mvn.mean)

        mvn_0_f = self.model(X)
        mu_0_f = mvn_0_f.mean
        sg_0_f = mvn_0_f.stddev
        vr_0_f = mvn_0_f.variance

        mvn_f = model_f.posterior(self.X_samples, observation_noise=True)
        mu_f = mvn_f.mean.squeeze(-1)
        sg_f = mvn_f.stddev
        vr_f = mvn_f.variance.squeeze(-1)
            

        p_max_0 = torch.exp(mu_0 + vr_0/2)
        # p_max = torch.exp(mu + vr/2)

        p_max_0_f = torch.exp(mu_0_f + vr_0_f/2)
        p_max_f = torch.exp(mu_f + vr_f/2)
        
        H_0 = -(p_max_0*torch.log(p_max_0)).mean(dim=-1)
        # H = -(p_max*torch.log(p_max)).mean(dim=-1)
        
        H_0_f = -(p_max_0_f*torch.log(p_max_0_f)).mean(dim=-1)
        H_f = -(p_max_f*torch.log(p_max_f)).mean(dim=-1)

        # H_0?
        return -H_0_f - H_f
        
