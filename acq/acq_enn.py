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
    num_candidates_per_arm: int = 100
    max_cell: bool = False
    boundary: bool = False
    maximin: bool = False
    acq: str = "pareto"

    se_scale: float = 1
    num_over_sample_per_arm: int = 1

    constrain_by_mu: bool = False
    num_alpha: int = 1

    def __post_init__(self):
        assert self.num_over_sample_per_arm > 0


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
        # i = np.random.choice(np.where(y == y.max())[0])

        return self._X_train[i, :]

    def _sample_boundary(self, x_0):
        u = random_directions(len(x_0), self._num_dim)
        return farthest_neighbor(self._enn, x_0, u, boundary_is_neighbor=False)

    def _sample_in_cell(self, x_cand):
        u = random_directions(len(x_cand), self._num_dim)
        x_far = farthest_neighbor(self._enn, x_cand, u, boundary_is_neighbor=False)

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

    # def _idx_pareto(self, x_cand: np.ndarray, mu, se, fn_min_dist, k_top_k=None):
    #     if k_top_k is not None and len(x_cand) >= k_top_k:
    #         i_mu = set(top_k(mu.flatten(), k_top_k))
    #         i_se = set(top_k(se.flatten(), k_top_k))
    #         i_orig = np.array(sorted(i_mu.intersection(i_se)))
    #         x_cand = x_cand[i_orig, :]
    #         del i_mu, i_se
    #     else:
    #         i_orig = np.arange(len(x_cand))

    #     num_samples = x_cand.shape[0]
    #     dominated = np.zeros(num_samples, dtype=bool)

    #     md_cache = np.nan * np.ones(shape=(num_samples,))

    #     def _get_md(i):
    #         if np.isnan(md_cache[i]):
    #             md_cache[i] = fn_min_dist(i)
    #         return md_cache[i]

    #     for i in range(num_samples):
    #         if dominated[i]:
    #             continue
    #         mu_i = mu[i]
    #         se_i = se[i]
    #         md_i = _get_md(i)

    #         for j in range(num_samples):
    #             if j == i:
    #                 continue
    #             mu_j = mu[j]
    #             se_j = se[j]
    #             if mu_j < mu_i or se_j < se_i:
    #                 continue
    #             md_j = _get_md(j)
    #             if md_j < md_i:
    #                 continue
    #             if md_j > md_i or mu_j > mu_i or se_j > se_i:
    #                 dominated[i] = True
    #                 break

    #     return i_orig[np.where(~dominated)[0]]

    # def _acq_pareto(self, x_cand, num_arms):
    #     mvn = self._enn.posterior(x_cand)

    #     def _fn_min_dist(i):
    #         return np.linalg.norm(x_cand[i, :] - np.concatenate([x_cand[:i, :], x_cand[i + 1 :, :]], axis=0), axis=1).min()

    #     t_0 = time.time()
    #     i = self._idx_pareto(x_cand, mvn.mu, mvn.se, _fn_min_dist, k_top_k=None)
    #     t_f = time.time()
    #     print("P:", len(i), len(x_cand), t_f - t_0)

    #     if len(i) < num_arms:
    #         print("BORK:", num_arms - len(i))
    #         i = set(i)
    #         i.update(np.random.choice(list(set(np.arange(len(x_cand))) - i), size=num_arms - len(i), replace=False).tolist())
    #         i = list(i)
    #     elif len(i) >= num_arms:
    #         i = np.random.choice(i, size=num_arms, replace=False)
    #     return x_cand[i]

    # def _acq_approx_pareto(self, x_cand, num_arms):
    #     # slow
    #     mvn = self._enn.posterior(x_cand)
    #     mu_min = mvn.mu.min()
    #     mu = (mvn.mu - mu_min) / (mvn.mu.max() - mu_min)
    #     se_min = mvn.se.min()
    #     se = (mvn.se - se_min) / (mvn.se.max() - se_min)

    #     n = len(x_cand)
    #     w = np.random.uniform(size=(n, 2))
    #     w = w / w.sum(axis=1, keepdims=True)
    #     phi = mu * w[:, 0] + se * w[:, 1]
    #     i = np.argsort(-phi)
    #     x_cand = x_cand[i[self._config.num_sub_cand_per_arm], :]
    #     i = greedy_maximin(x_cand, num_arms)
    #     return x_cand[i]

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
        x_0 = self._ts_pick_cells(self._config.num_candidates_per_arm * num_arms)
        if self._config.boundary:
            x_cand = self._sample_boundary(x_0)
        else:
            x_cand = self._sample_in_cell(x_0)

        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())

        if self._config.num_candidates_per_arm == 1:
            return x_cand

        if self._config.acq == "ucb":
            return self._ucb(x_cand, num_arms)
        elif self._config.acq == "pareto":
            return self._pareto_front(x_cand, num_arms)
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
