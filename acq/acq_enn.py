from dataclasses import dataclass

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import random_directions  #  single_coordinate_perturbation
from sampling.ray_boundary import ray_boundary_np

"""
- Remove least-likely points from observation set, i.e. Pareto(-mu_as_if_missing, -sigma_as_if_missing), i.e., "If the
-   observation were removed, would we want to put it back?"
- Select P pivot observations randomly w/replacement from Pareto(mu, abs(mu - mu_hat_as_if_missing))
- For each pivot point, sample a candidate along a line segment starting at the pivot point and ending at a boundary.
- Select A points w/o replacement from Pareto(mu, sigma) from the P candidates.
"""


@dataclass
class ENNConfig:
    k: int
    num_interior: int = 0
    stagger: bool = False
    acq: str = "pareto"
    small_world_M: int = None

    num_over_sample_per_arm: int = 1

    region_type: str = "sobol"

    def __post_init__(self):
        assert self.num_over_sample_per_arm > 0


class AcqENN:
    # TODO: Yvar

    def __init__(self, num_dim, config: ENNConfig):
        assert config.k > 0, config
        self._num_dim = num_dim

        self._x_train = np.empty(shape=(0, num_dim))
        self._y_train = np.empty(shape=(0, 1))

        self._config = config
        self._enn = None

    def add(self, x, y):
        if len(x) == 0:
            return
        x = np.asarray(x)
        y = np.asarray(y)
        if self._enn is None:
            self._enn = EpsitemicNearestNeighbors(x, y, k=self._config.k, small_world_M=self._config.small_world_M)
        else:
            self._enn.add(x, y)
        self._x_train = np.append(self._x_train, x, axis=0)
        self._y_train = np.append(self._y_train, y, axis=0)

    def keep_top_n(self, num_keep):
        if len(self._x_train) <= num_keep:
            return self._x_train, self._y_train
        i = self._i_pareto_fronts_strict(self._x_train, num_keep, exclude_nearest=True)
        return self._x_train[i], self._y_train[i]

    def _sample_segments(self, x_0, x_far):
        assert x_0.shape == x_far.shape, (x_0.shape, x_far.shape)

        alpha = np.random.uniform(size=(len(x_0), 1))

        if self._config.stagger:
            l_s_min = np.log(1e-5)
            l_s_max = np.log(1)
            alpha = np.exp(l_s_min + (l_s_max - l_s_min) * alpha)

        x_cand = (1 - alpha) * x_0 + alpha * x_far

        return x_cand

    def _select_pivots(self, num_pivot):
        # mvn = self._enn.posterior(self._x_train, exclude_nearest=False)
        y = self._y_train
        mvn_as_if_missing = self._enn.posterior(self._x_train, exclude_nearest=True)
        discrep = np.abs(y - mvn_as_if_missing.mu)

        i = np.argsort(-y, axis=0).flatten()
        x_cand = self._x_train[i]
        discrep = discrep[i]

        i_front = []
        discrep_max = -1e99
        for i in range(len(discrep)):
            if discrep[i] >= discrep_max:
                i_front.append(i)
                discrep_max = discrep[i]

        i_front = np.array(i_front)
        x_front = x_cand[i_front]

        i = np.random.choice(np.arange(len(x_front)), size=num_pivot, replace=True)
        return x_front[i]

    def _i_pareto_fronts_strict(self, x_cand, num_arms, exclude_nearest=False):
        # Full fronts, then random selection from *last* front.
        assert self._config.num_over_sample_per_arm == 1, self._config.num_over_sample_per_arm
        mvn = self._enn.posterior(x_cand, exclude_nearest=exclude_nearest)

        i = np.argsort(-mvn.mu, axis=0).flatten()
        x_cand = x_cand[i]
        se = mvn.se[i]

        i_all = list(range(len(se)))
        i_keep = []

        num_tries = 0
        while len(i_keep) < num_arms:
            num_tries += 1
            assert num_tries < 100, num_tries
            se_max = -1e99
            i_front = []
            for i in i_all:
                if se[i] >= se_max:
                    i_front.append(i)
                    se_max = se[i]
            if len(i_keep) + len(i_front) <= num_arms:
                i_keep.extend(i_front)
            else:
                i_keep.extend(np.random.choice(i_front, size=num_arms - len(i_keep), replace=False))
            i_all = sorted(set(i_all) - set(i_front))

        return np.array(i_keep)

    def _pareto_fronts_strict(self, x_cand, num_arms, exclude_nearest=False):
        i_keep = self._i_pareto_fronts_strict(x_cand, num_arms, exclude_nearest)
        x_arms = x_cand[i_keep]

        print("P:", i_keep)
        assert len(x_arms) == num_arms, (len(x_arms), num_arms)
        return x_arms

    def _uniform(self, x_cand, num_arms):
        i = np.random.choice(np.arange(len(x_cand)), size=num_arms, replace=False)
        return x_cand[i]

    def _candidates(self, num_arms):
        if self._config.region_type == "sobol":
            x_cand = (
                draw_sobol_samples(
                    bounds=torch.tensor([[0.0, 1.0]] * self._num_dim).T,
                    n=self._config.num_interior * num_arms,
                    q=1,
                )
                .detach()
                .numpy()
            ).squeeze(1)
        elif self._config.region_type == "pivots":
            x_0 = self._select_pivots(self._config.num_interior * num_arms)
            x_far = ray_boundary_np(x_0, random_directions(len(x_0), self._num_dim))
            x_cand = self._sample_segments(x_0, x_far)
        elif self._config.region_type == "rand_train":
            x_0 = self._x_train[np.random.choice(np.arange(len(self._x_train)), size=num_arms, replace=True)]
            x_far = ray_boundary_np(x_0, random_directions(len(x_0), self._num_dim))
            x_cand = self._sample_segments(x_0, x_far)
        elif self._config.region_type == "convex":
            num_cand = self._config.num_interior * num_arms
            mx = self._x_train.mean(axis=0, keepdims=True)

            x_0 = self._select_pivots(num_cand)
            w = np.random.dirichlet(size=num_cand, alpha=np.ones(len(x_0)))
            w = w / w.sum(axis=1, keepdims=True)
            m = (x_0.T @ w.T).T
            x_cand = m  #  + (m - mx) * np.sqrt(len(x_0))

            u = x_cand - mx
            assert not np.isnan(u.sum())
            norm = np.linalg.norm(u, axis=1, keepdims=True)
            i = np.where(norm < 1e-9)[0]
            norm[i] = 1
            u = u / norm
            u[i] = 0
            assert not np.isnan(u.sum())
            x_boundary = ray_boundary_np(mx, 0.1 * u)
            dist_cand = np.linalg.norm(x_cand - mx, axis=1)
            dist_boundary = np.linalg.norm(x_boundary - mx, axis=1)
            i_clip = dist_cand > dist_boundary
            x_cand[i_clip] = x_boundary[i_clip]

        else:
            assert False, self._config.region_type

        x_cand = np.unique(x_cand, axis=0)

        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())
        assert len(np.unique(x_cand, axis=0)) == x_cand.shape[0], (len(np.unique(x_cand, axis=0)), x_cand.shape[0])

        return x_cand

    def _draw_two_level(self, num_arms):
        x_cand = self._candidates(num_arms)

        if len(x_cand) == num_arms:
            return x_cand

        if self._config.acq == "pareto_strict":
            return self._pareto_fronts_strict(x_cand, num_arms)
        elif self._config.acq == "uniform":
            return self._uniform(x_cand, num_arms)
        else:
            assert False, self._config.acq

    def draw(self, num_arms):
        if len(self._x_train) == 0:
            # x_a = 0.5 + np.zeros(shape=(1, self._num_dim))
            # x_a = np.append(x_a, np.random.uniform(size=(num_arms - 1, self._num_dim)), axis=0)
            x_a = np.random.uniform(size=(num_arms, self._num_dim))
        else:
            x_a = self._draw_two_level(num_arms)

        assert x_a.shape == (num_arms, self._num_dim), (num_arms, self._num_dim, x_a.shape)
        assert len(np.unique(x_a, axis=0)) == num_arms, (len(np.unique(x_a, axis=0)), num_arms)

        return x_a
