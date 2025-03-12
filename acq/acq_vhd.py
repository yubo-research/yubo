import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import farthest_neighbor, random_directions, target_directions


class AcqVHD:
    # TODO: Yvar
    def __init__(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        *,
        k: int = 1,
        num_candidates_per_arm=100,
        two_level=False,
        target_directions=False,
        max_cell=False,
        num_refinements=1,
    ):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._num_candidates_per_arm = num_candidates_per_arm
        self._num_dim = self._X_train.shape[-1]
        self._seed = np.random.randint(0, 9999)
        self._two_level = two_level
        self._target_directions = target_directions
        self._max_cell = max_cell
        self._k = k
        self._num_refinements = num_refinements

        if len(self._X_train) > 0:
            self._enn_ts = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=max(1, k))
        else:
            self._enn_ts = None

    def _ts_pick_cell(self):
        assert len(self._X_train) > 0
        if False:  # TODO: study noisy observations self._enn_ts:
            y = self._enn_ts(self._X_train).sample()
        else:
            y = self._Y_train

        # We Thompson sample over the Voronoi cells.
        # Our model of the function values in a cell is N(y, se**2)
        # where y is the (single) measured value in the cell, and
        #  se**2 is the (homoscedastic) variance, estimated by
        #  var(y)/N over all measured y's.
        se = y.std() / np.sqrt(len(y))
        if self._max_cell:
            se = 0 * se
        y = y + se * np.random.normal(size=y.shape)

        i = np.random.choice(np.where(y == y.max())[0])

        return self._X_train[[i], :]

    def _ts_in_cell(self, x, num_arms):
        if self._enn_ts is None or self._k == 0:
            i = np.random.choice(np.arange(len(x)), num_arms)
            return x[i]

        y = self._enn_ts.posterior(x).sample(num_arms)

        i = np.where(y == y.max(axis=0, keepdims=True))[0]
        i_arms = np.unique(i)
        i_arms = i_arms[:num_arms].astype(np.int64)
        return x[i_arms]

    def _ts_2(self, x_0, x_1):
        assert len(x_0) == len(x_1), (len(x_0), len(x_1))
        x_a = x_0.copy()

        if self._enn_ts is None or self._k == 0:
            i = np.ones(shape=(len(x_0))).astype(bool)
        else:
            x = np.concatenate((x_0, x_1), axis=0)
            y = self._enn_ts.posterior(x).sample(1).squeeze()
            n = len(x_0)
            y_0 = y[:n]
            y_1 = y[n:]
            i = y_1 > y_0

        x_a[i] = x_1[i]
        return x_a

    def draw(self, num_arms):
        if len(self._X_train) == 0:
            x_a = np.random.uniform(size=(num_arms, self._num_dim))
        else:
            if self._two_level:
                x_a = self._draw_two_level(num_arms)
            else:
                x_a = self._draw_near_far(num_arms)

        assert x_a.shape == (num_arms, self._num_dim), x_a.shape

        return x_a

    def _draw_two_level(self, num_arms):
        x_0 = np.vstack([self._ts_pick_cell() for _ in range(num_arms)])
        x_0 = np.tile(x_0, reps=(self._num_candidates_per_arm, 1))

        x_cand = x_0
        for _ in range(self._num_refinements):
            if self._target_directions:
                u = target_directions(x_cand)
            else:
                u = random_directions(len(x_cand), self._num_dim)
            x_fn = farthest_neighbor(self._enn_ts, x_cand, u, boundary_is_neighbor=False)
            # We want to uniformly sample over the Voronoi cell, but this is
            #  easier. Maybe we'll come up with something better.
            alpha = np.random.uniform(size=x_0.shape)
            x_cand_next = (1 - alpha) * x_cand + alpha * x_fn
            assert x_cand_next.min() >= 0.0 and x_cand_next.max() <= 1.0, (x_cand_next.min(), x_cand_next.max())
            if self._num_candidates_per_arm == 1:
                x_cand = self._ts_2(x_cand, x_cand_next)
            else:
                x_cand = x_cand_next

        x_a = x_cand
        if self._num_candidates_per_arm > 1:
            x_a = self._ts_in_cell(x_cand, num_arms)
        x_a = np.append(x_a, np.random.uniform(size=(num_arms - len(x_a), self._num_dim)), axis=0)
        return x_a
