import numpy as np
import torch

from model.enn import EpistemicNearestNeighbors
from sampling.knn_tools import farthest_neighbor, random_directions
from sampling.lhd import latin_hypercube_design


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, *, k: int = 1, num_candidates_per_arm=100, lhd=False):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._num_candidates_per_arm = num_candidates_per_arm
        self._lhd = lhd
        self._num_dim = self._X_train.shape[-1]
        self._seed = np.random.randint(0, 9999)

        if len(self._X_train) > 0:
            self._enn_1 = EpistemicNearestNeighbors(self._X_train, self._Y_train, k=1)
            if k > 0:
                self._enn_ts = EpistemicNearestNeighbors(self._X_train, self._Y_train, k=k)
            else:
                self._enn_ts = None
        else:
            self._enn_1 = None
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
        if self._enn_ts is None:
            i = np.random.choice(np.arange(len(x)), num_arms)
            return x[i]

        y = self._enn_ts.posterior(x).sample(2 * num_arms)
        i_arms = np.array([]).astype(np.int64)
        i_retry = 0
        while len(i_arms) < num_arms and i_retry < 10:
            y[i_arms] = -100
            i = np.where(y == y.max(axis=0, keepdims=True))[0]
            i_arms = np.unique(
                np.concatenate(
                    (i_arms, i),
                )
            )
            i_retry += 1

        assert len(i_arms) >= num_arms, (i_retry, len(i_arms), x.shape, num_arms)

        i_arms = i_arms[:num_arms].astype(np.int64)
        return x[i_arms]

    def draw(self, num_arms):
        num_candidates = self._num_candidates_per_arm * num_arms

        if len(self._X_train) == 0:
            x_c = 0.5 + np.zeros(shape=(1, self._num_dim))
            if self._lhd:
                xs = latin_hypercube_design(num_arms - len(x_c), self._num_dim, seed=self._seed + len(self._X_train) + 1)
            else:
                xs = np.random.uniform(size=(num_arms - len(x_c), self._num_dim))
            x_a = np.append(x_c, xs, axis=0)
        else:
            num_near = num_candidates // 2
            num_far = num_candidates - num_near

            x_0 = np.tile(self.get_max(), reps=(num_near, 1))
            assert x_0.shape == (num_near, self._num_dim), x_0.shape

            u = random_directions(num_near, self._num_dim)
            # TODO: try uniform in cell
            x_fn = farthest_neighbor(self._enn_1, x_0, u)
            alpha = np.random.uniform(size=x_0.shape)
            x_near = (1 - alpha) * x_0 + alpha * x_fn

            assert x_near.min() >= 0.0 and x_near.max() <= 1.0, (x_near.min(), x_near.max())

            x_far = np.random.uniform(size=(num_far, self._num_dim))

            x_cand = np.append(x_near, x_far, axis=0)
            x_a = self._thompson_sample(x_cand, num_arms)

        assert len(x_a) == num_arms, (len(x_a), num_arms)
        return x_a
