import numpy as np
import torch

from model.enn import EpistemicNearestNeighbors
from sampling.scale_free_sampler import scale_free_sampler


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1, num_samples=256):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._num_samples = num_samples
        self._b_raasp = False

        if k > 0 and len(self._X_train) > 0:
            self._enn = EpistemicNearestNeighbors(self._X_train, self._Y_train, k=k)
        else:
            self._enn = None

    def _get_max(self):
        assert len(self._X_train) > 0
        if self._enn:
            Y = self._enn(self._X_train).sample()
        else:
            Y = self._Y_train
        i = np.random.choice(np.where(Y == Y.max())[0])
        return self._X_train[i, :]

    def draw(self, num_arms):
        if len(self._X_train) == 0:
            return np.random.uniform(size=(num_arms, self._X_train.shape[-1]))

        # TODO: self._num_samples ts-maxes for variety, then move X_0 inside loop
        X_0 = np.tile(self._get_max(), reps=(self._num_samples, 1))

        # TODO: MTV for batches
        X_a = []
        for _ in range(num_arms):
            X_cand = scale_free_sampler(X_0, b_raasp=self._b_raasp)
            if self._enn:
                Y_cand = self._enn(X_cand).sample()
                i = np.random.choice(np.where(Y_cand == Y_cand.max())[0])
            else:
                i = np.random.randint(low=0, high=len(X_cand), size=(1,))
            X_a.append(X_cand[i])

        return np.array(X_a)
