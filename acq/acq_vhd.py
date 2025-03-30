from dataclasses import dataclass

import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import farthest_neighbor, random_directions, target_directions


@dataclass
class VHDConfig:
    k: int
    two_level: bool = True
    num_candidates_per_arm: int = 100
    direction_type: str = "random"
    max_cell: bool = False
    bidirectional: bool = False
    ucb: bool = False
    se_scale: float = 1


class AcqVHD:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, config: VHDConfig):
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._config = config
        self._num_dim = self._X_train.shape[-1]
        self._seed = np.random.randint(0, 9999)

        if len(self._X_train) > 0:
            self._enn_ts = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=max(1, self._config.k))
            self._enn_ts.calibrate(self._config.se_scale**2)
        else:
            self._enn_ts = None

    def _ts_pick_cells(self, num_arms):
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
        n = len(y)
        se = y.std() / np.sqrt(n)
        if self._config.max_cell:
            se = 0 * se
        y = y + se * np.random.normal(size=(n, num_arms))

        i = np.where(y == y.max(axis=0, keepdims=True))[0]
        # i = np.random.choice(np.where(y == y.max())[0])

        return self._X_train[i, :]

    def _ucb(self, x, num_arms):
        assert num_arms == 1, num_arms
        # if self._enn_ts is None or self._config.k == 0:
        #     i = np.random.choice(np.arange(len(x)), num_arms)
        #     return x[i]

        mvn = self._enn_ts.posterior(x)
        ucb = mvn.mu + mvn.se

        i = np.where(ucb == ucb.max(axis=0, keepdims=True))[0]
        i_arms = np.unique(i)
        i_arms = i_arms[:num_arms].astype(np.int64)
        return x[i_arms]

    def _ts_in_cell(self, x, num_arms):
        # TODO: joint sampling
        if self._enn_ts is None or self._config.k == 0:
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

        if self._enn_ts is None or self._config.k == 0:
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

    def _sample_in_cell(self, x_cand):
        if self._config.direction_type == "target":
            u = target_directions(x_cand)
        elif self._config.direction_type == "random":
            u = random_directions(len(x_cand), self._num_dim)

        x_a = farthest_neighbor(self._enn_ts, x_cand, u, boundary_is_neighbor=False)
        if self._config.bidirectional:
            x_b = farthest_neighbor(self._enn_ts, x_cand, -u, boundary_is_neighbor=False)
        else:
            x_b = x_cand

        # We want to uniformly sample over the Voronoi cell, but this is
        #  easier. Maybe we'll come up with something better.

        self._num_alpha = 1
        alpha = np.random.uniform(size=(self._num_alpha * x_cand.shape[0], 1))
        # x_cand_next = (1 - alpha) * x_a + alpha * x_b
        x_cand_next = (1 - alpha) * np.tile(x_a, reps=(self._num_alpha, 1)) + alpha * np.tile(x_b, reps=(self._num_alpha, 1))
        return x_cand_next

    def _acquire(self, x_cand, num_arms):
        if self._config.ucb:
            return self._ucb(x_cand, num_arms)
        else:
            return self._ts_in_cell(x_cand, num_arms)

    def _draw_two_level(self, num_arms):
        x_0 = self._ts_pick_cells(num_arms)
        x_0 = np.tile(x_0, reps=(self._config.num_candidates_per_arm, 1))

        x_cand = x_0
        x_cand = self._sample_in_cell(x_0)

        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())

        x_a = x_cand
        if self._config.num_candidates_per_arm > 1:
            x_a = self._acquire(x_cand, num_arms)
        x_a = np.append(x_a, np.random.uniform(size=(num_arms - len(x_a), self._num_dim)), axis=0)
        return x_a

    def _draw_near_far(self, num_arms):
        num_candidates = self._config.num_candidates_per_arm * num_arms

        if len(self._X_train) == 0:
            x_c = 0.5 + np.zeros(shape=(1, self._num_dim))
            xs = np.random.uniform(size=(num_arms - len(x_c), self._num_dim))
            x_a = np.append(x_c, xs, axis=0)
        else:
            num_near = num_candidates // 2
            num_far = num_candidates - num_near

            x_0 = self._ts_pick_cells(num_near)
            assert x_0.shape == (num_near, self._num_dim), (num_near, self._num_dim, x_0.shape)

            x_near = self._sample_in_cell(x_0)

            assert x_near.min() >= 0.0 and x_near.max() <= 1.0, (x_near.min(), x_near.max())

            x_far = np.random.uniform(size=(num_far, self._num_dim))

            x_cand = np.append(x_near, x_far, axis=0)
            x_a = self._acquire(x_cand, num_arms)

        assert len(x_a) == num_arms, (len(x_a), num_arms)
        return x_a

    def draw(self, num_arms):
        if len(self._X_train) == 0:
            x_a = np.random.uniform(size=(num_arms, self._num_dim))
        else:
            if self._config.two_level:
                x_a = self._draw_two_level(num_arms)
            else:
                x_a = self._draw_near_far(num_arms)

        assert x_a.shape == (num_arms, self._num_dim), x_a.shape

        return x_a
