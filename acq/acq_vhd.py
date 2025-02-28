import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.scale_free_sampler import scale_free_sampler


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1, num_samples=256):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._num_samples = num_samples

        if len(self._X_train) > 0:
            self._enn = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=1)
            # self._enn_ts = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=k)
        else:
            self._enn = None
            self._enn_ts = None

    def _get_max(self):
        assert len(self._X_train) > 0
        if False:  # tODO: study noisy observations self._enn_ts:
            Y = self._enn_ts(self._X_train).sample()
        else:
            Y = self._Y_train

        i = np.random.choice(np.where(Y == Y.max())[0])
        return self._X_train[i, :]

    def draw(self, num_arms):
        if len(self._X_train) == 0:
            # TODO: Sobol or LHD
            return np.random.uniform(size=(num_arms, self._X_train.shape[-1]))

        # TODO: study noisy observations;  move X_0 inside loop
        X_0 = np.tile(self._get_max(), reps=(self._num_samples, 1))

        X_a = []
        for _ in range(num_arms):
            # TODO: farthest_neighbor(), N times
            # TODO: Choose N that maximizes distance; L2? L1? L1/2? log(1+|dx|)?
            # X_cand = None
            # X_a.append(X_cand[i])
            assert False, "NYI"

        return np.array(X_a)
