import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import farthest_neighbor, random_direction


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1, num_samples=256):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._num_samples = num_samples
        self._num_dim = self._X_train.shape[-1]

        if len(self._X_train) > 0:
            self._enn = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=1)
            # self._enn_ts = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=k)
        else:
            self._enn = None
            self._enn_ts = None

    def get_max(self):
        assert len(self._X_train) > 0
        if False:  # TODO: study noisy observations self._enn_ts:
            Y = self._enn_ts(self._X_train).sample()
        else:
            Y = self._Y_train

        i = np.random.choice(np.where(Y == Y.max())[0])
        return self._X_train[[i], :]

    def draw(self, num_arms):
        assert num_arms == 1, num_arms

        if len(self._X_train) == 0:
            # TODO: Sobol or LHD ?
            return 0.5 + np.zeros(shape=(1, self._num_dim))

        # TODO: study noisy observations;  move X_0 inside loop
        # X_0 = np.tile(self._get_max(), reps=(self._num_samples, 1))

        x_0 = self.get_max()
        assert len(x_0.shape) == 2, x_0.shape
        u = random_direction(self._num_dim)
        return farthest_neighbor(self._enn, x_0, u)
