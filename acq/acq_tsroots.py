import numpy as np
import torch
from tsroots.optim import TSRoots


# See https://github.com/UQUH/TSRoots/tree/main
class AcqTSRoots:
    def __init__(self, model):
        # TODO: avoid fitting the model
        self._X = model.train_inputs[0].detach().numpy()
        num_obs = self._X.shape[0]
        assert num_obs > 0, num_obs
        self._num_dim = self._X.shape[-1]
        self._Y = model.train_targets.detach().numpy()

    def draw(self, num_arms):
        tsr = TSRoots(
            self._X,
            self._Y,
            np.zeros(shape=(self._num_dim,)),
            np.ones(shape=(self._num_dim,)),
        )

        return torch.stack([torch.as_tensor(tsr.xnew_TSroots()[0]) for _ in range(num_arms)])
