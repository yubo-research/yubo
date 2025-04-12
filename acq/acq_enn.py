from dataclasses import dataclass

import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import farthest_neighbor, random_directions
from sampling.sampling_util import greedy_maximin

"""
Don't calibrate (fit).
TODO
- TR: Voronoi cell for distance (se); avoids hyperparameter
    - Any rejection sampling (on delta_mu < k) in the HnR phase takes as much time as oversampling candidates would.
    - Do this: Find the boundary, then take several alpha samples along the line (to reduce the number of time-consuming bisection searches).
    - Then reject all points with mu(x) < mean(mu(all boundary points)).
- TR: max(0, se(x)**2 - se(x_max)**2) < k  and  max(0, mu(x_max) - mu(x)) < k
- Take samples in TR, find Pareto front (maximizing se, maximizing mu), sample arm from that
  - Batching: Sample all arms from Pareto front, maximizing min dist between arms

- Ref CRBO, Pareto BO


Tests
- Performance should increase to an asymptote w/increasing num_candidates_per_arm.
- Performance should increase to an asymptote w/increasing k (ideally, but maybe the model only works locally. That's ok, too).
"""


@dataclass
class ENNConfig:
    k: int
    max_cell: bool = False
    num_boundary: int = 100
    num_interior: int = 0
    maximin: bool = False
    acq: str = "pareto"

    se_scale: float = 1
    num_over_sample_per_arm: int = 1

    constrain_by_mu: bool = False
    num_alpha: int = 1
    p_boundary_is_neighbor: float = 0.0

    def __post_init__(self):
        assert self.num_over_sample_per_arm > 0
        assert self.num_boundary + self.num_interior > 0


class AcqENN:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, config: ENNConfig):
        assert config.k > 0, config
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._config = config
        self._num_dim = self._X_train.shape[-1]

        if len(self._X_train) > 0:
            self._enn = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=self._config.k)
            self._enn.calibrate(self._config.se_scale**2)
        else:
            self._enn = None

    def _ts_pick_cells(self, num_arms):
        assert len(self._X_train) > 0
        if False:  # TODO: study noisy observations self._enn_ts:
            y = self._enn(self._X_train).sample()
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

        return self._X_train[i, :]

    def _sample_boundary(self, x_0):
        u = random_directions(len(x_0), self._num_dim)
        return farthest_neighbor(self._enn, x_0, u, p_boundary_is_neighbor=self._config.p_boundary_is_neighbor)

    def _sample_in_cell(self, x_cand):
        u = random_directions(len(x_cand), self._num_dim)
        x_far = farthest_neighbor(self._enn, x_cand, u, p_boundary_is_neighbor=self._config.p_boundary_is_neighbor)

        # We want to uniformly sample over the Voronoi cell, but this is
        #  easier. Maybe we'll come up with something better.

        alpha = np.random.uniform(size=(self._config.num_alpha * x_cand.shape[0], 1))
        x_cand = np.tile(x_cand, reps=(self._config.num_alpha, 1))
        x_far = np.tile(x_far, reps=(self._config.num_alpha, 1))

        x_cand = alpha * x_cand + (1 - alpha) * x_far

        if self._config.constrain_by_mu:
            mvn_0 = self._enn.posterior(x_far)
            mu_0 = mvn_0.mu.mean()
            mvn = self._enn.posterior(x_cand)
            i = np.where(mvn.mu >= mu_0)[0]
            x_cand = x_cand[i, :]

        return x_cand

    def _ucb(self, x_cand, num_arms):
        assert num_arms == 1, num_arms
        mvn = self._enn.posterior(x_cand)
        ucb = mvn.mu + mvn.se

        i = np.where(ucb == ucb.max(axis=0, keepdims=True))[0]
        i_arms = np.unique(i)
        i_arms = i_arms[:num_arms].astype(np.int64)
        return x_cand[i_arms]

    def _pareto_front(self, x_cand, num_arms):
        mvn = self._enn.posterior(x_cand)

        i = np.argsort(-mvn.mu, axis=0).flatten()
        x_cand = x_cand[i]
        se = mvn.se[i]

        num_keep = self._config.num_over_sample_per_arm * num_arms
        i_all = list(range(len(se)))
        i_keep = []
        while len(i_keep) < num_keep:
            se_max = -1e99
            for i in i_all:
                if se[i] >= se_max:
                    i_keep.append(i)
                    se_max = se[i]
            i_all = sorted(set(i_all) - set(i_keep))

        i_keep = np.array(i_keep)
        x_front = x_cand[i_keep]

        assert len(x_front) >= num_arms, (len(x_front), num_arms)

        if len(x_front) > num_arms:
            if num_arms > 1 and self._config.maximin:
                i = greedy_maximin(x_front, num_arms)
            else:
                i = np.random.choice(np.arange(len(x_front)), size=num_arms, replace=False)
            return x_front[i]
        else:
            return x_front

    def _pareto_cheb_noisy(self, x_cand, num_arms):
        mvn = self._enn.posterior(x_cand)
        y = np.concatenate([mvn.mu, mvn.se], axis=1)
        norm = y.max(axis=0, keepdims=True) - y.min(axis=0, keepdims=True)
        y = y - y.min(axis=0, keepdims=True)
        i = np.where(norm == 0)[0]
        norm[i] = 1
        y[i] = 0.5
        y = y / norm

        w = np.random.uniform(size=y.shape)

        w = w / w.sum(axis=1, keepdims=True)
        y = y * w
        y = y.min(axis=1)

        i = np.argsort(-y, axis=0).flatten()
        x_cand = x_cand[i]
        return x_cand[:num_arms]

    def _pareto_cheb(self, x_cand, num_arms):
        mvn = self._enn.posterior(x_cand)
        y = np.concatenate([mvn.mu, mvn.se], axis=1)
        norm = y.max(axis=0, keepdims=True) - y.min(axis=0, keepdims=True)
        y = y - y.min(axis=0, keepdims=True)
        i = np.where(norm == 0)[0]
        norm[i] = 1
        y[i] = 0.5
        y = y / norm

        num_cands_per_arm = x_cand.shape[0] // num_arms
        x_cand = np.reshape(x_cand, (num_arms, num_cands_per_arm, x_cand.shape[1]))
        y = np.reshape(y, (num_arms, num_cands_per_arm, y.shape[1]))

        w = np.random.uniform(size=y.shape)
        w = w / w.sum(axis=-1, keepdims=True)
        y = y * w
        y = y.min(axis=-1)

        i = np.argmax(y, axis=1)
        return np.diagonal(x_cand[:, i, :], axis1=0, axis2=1).T

    def _pareto_front_cheb(self, x_cand, num_arms):
        mvn = self._enn.posterior(x_cand)

        i = np.argsort(-mvn.mu, axis=0).flatten()
        x_cand = x_cand[i]
        se = mvn.se[i]

        i_front = []
        se_max = -1e99
        for i in range(len(se)):
            if se[i] >= se_max:
                i_front.append(i)
                se_max = se[i]

        i_front = np.array(i_front)
        assert len(i_front) > 0, (len(i_front), len(x_cand))
        i_arm = np.random.choice(i_front, size=1, replace=False)
        x_arms = [x_cand[i_arm]]

        if num_arms > 1:
            x_cand = np.delete(x_cand, i_arm, axis=0)
            x_arms.extend(self._pareto_cheb_noisy(x_cand, num_arms - 1))

        return np.vstack(x_arms)

    def _thompson_sample(self, x_cand, num_arms):
        # was 2*num_arms
        y = self._enn.posterior(x_cand).sample(num_arms)
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

        assert len(i_arms) >= num_arms, (i_retry, len(i_arms), x_cand.shape, num_arms)

        i_arms = i_arms[:num_arms].astype(np.int64)
        return x_cand[i_arms]

    def _uniform(self, x_cand, num_arms):
        if self._config.maximin:
            i = greedy_maximin(x_cand, num_arms)
        else:
            i = np.random.choice(np.arange(len(x_cand)), size=num_arms, replace=False)
        return x_cand[i]

    def _draw_two_level(self, num_arms):
        x_0 = self._ts_pick_cells((self._config.num_boundary + self._config.num_interior) * num_arms)
        x_cand = np.empty(shape=(0, self._num_dim))
        if self._config.num_boundary > 0:
            x_cand = np.concatenate([x_cand, self._sample_boundary(x_0[: self._config.num_boundary])], axis=0)
        if self._config.num_interior > 0:
            x_cand = np.concatenate([x_cand, self._sample_in_cell(x_0[self._config.num_boundary :])], axis=0)

        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())

        if self._config.num_boundary + self._config.num_interior == 1:
            return x_cand

        if self._config.acq == "ucb":
            return self._ucb(x_cand, num_arms)
        elif self._config.acq == "pareto":
            return self._pareto_front(x_cand, num_arms)
        elif self._config.acq == "pareto_cheb":
            return self._pareto_cheb(x_cand, num_arms)
        elif self._config.acq == "pareto_cheb_noisy":
            return self._pareto_cheb_noisy(x_cand, num_arms)
        elif self._config.acq == "pareto_front_cheb":
            return self._pareto_front_cheb(x_cand, num_arms)
        elif self._config.acq == "ts":
            return self._thompson_sample(x_cand, num_arms)
        elif self._config.acq == "uniform":
            return self._uniform(x_cand, num_arms)
        else:
            assert False, self._config.acq

    def draw(self, num_arms):
        if len(self._X_train) == 0:
            # x_a = 0.5 + np.zeros(shape=(1, self._num_dim))
            # x_a = np.append(x_a, np.random.uniform(size=(num_arms - 1, self._num_dim)), axis=0)
            x_a = np.random.uniform(size=(num_arms, self._num_dim))
        else:
            x_a = self._draw_two_level(num_arms)

        assert x_a.shape == (num_arms, self._num_dim), (num_arms, self._num_dim, x_a.shape)

        return x_a
