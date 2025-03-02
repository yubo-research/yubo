import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import farthest_neighbor, random_directions
from sampling.lhd import latin_hypercube_design


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1, num_candidates_per_arm=10):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._num_candidates_per_arm = num_candidates_per_arm
        self._num_dim = self._X_train.shape[-1]
        self._seed = np.random.randint(0, 9999)

        if len(self._X_train) > 0:
            self._enn = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=1)
            if k > 0:
                self._enn_ts = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=k)
            else:
                self._enn_ts = None
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

    def _thompson_sample(self, x, num_arms):
        num_dim = x.shape[-1]
        y = self._enn_ts.posterior(x).sample()
        y = np.reshape(y, newshape=(num_arms, self._num_candidates_per_arm))
        x = np.reshape(x, newshape=(num_arms, self._num_candidates_per_arm, num_dim))
        i = np.where(y == y.max(axis=1, keepdims=True))
        return x[i]

    def draw(self, num_arms):
        if len(self._X_train) == 0:
            x_a = 0.5 + np.zeros(shape=(num_arms, self._num_dim))
            if num_arms == 1:
                return x_a
            else:
                # TODO: random, sobol
                return np.append(x_a, latin_hypercube_design(num_arms - 1, self._num_dim, seed=self._seed), axis=0)
                # return np.append(x_a, np.random.uniform(size=(num_arms - 1, self._num_dim)), axis=0)

        # TODO: study noisy observations;  move X_0 inside loop
        num_candidates = self._num_candidates_per_arm * num_arms
        x_0 = np.tile(self.get_max(), reps=(num_candidates, 1))
        assert x_0.shape == (num_candidates, self._num_dim), x_0.shape

        u = random_directions(num_candidates, self._num_dim)
        x_cand = farthest_neighbor(self._enn, x_0, u)
        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())
        if num_candidates > num_arms:
            x_a = self._thompson_sample(x_cand, num_arms)
        else:
            x_a = x_cand
        return x_a
