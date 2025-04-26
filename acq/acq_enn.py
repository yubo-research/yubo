from dataclasses import dataclass

import numpy as np
import torch

from model.enn import EpsitemicNearestNeighbors
from sampling.knn_tools import confidence_region_fast, far_as_you_can_go, farthest_neighbor, farthest_neighbor_fast, random_directions
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
    ts_enn: bool = False
    num_boundary: int = 100
    num_interior: int = 0
    num_quick: int = 0
    keep_bdy: bool = False
    maximin: bool = False
    stagger: bool = False
    weight_by_length: bool = False
    acq: str = "pareto"
    bug_fix: bool = False

    se_scale: float = 1
    num_over_sample_per_arm: int = 1
    se_2: int = None

    region_type: str = "fn"
    se_max: float = 0.1

    p_boundary_is_neighbor: float = 0.0

    def __post_init__(self):
        assert self.num_over_sample_per_arm > 0
        assert self.num_quick > 0 or self.num_boundary + self.num_interior > 0
        assert self.num_quick == 0 or self.num_boundary + self.num_interior == 0, (self.num_quick, self.num_boundary, self.num_interior)


class AcqENN:
    # TODO: Yvar
    def __init__(self, X_train: torch.Tensor, Y_train: torch.Tensor, config: ENNConfig):
        assert config.k > 0, config
        self._X_train = np.asarray(X_train)
        self._Y_train = np.asarray(Y_train)

        self._config = config
        self._num_dim = self._X_train.shape[-1]

        if len(self._X_train) > 0:
            self._enn = EpsitemicNearestNeighbors(self._X_train, self._Y_train, k=self._config.k, bug_fix=self._config.bug_fix)
            self._enn.calibrate(self._config.se_scale**2)
        else:
            self._enn = None

    def _var_calib(self):
        if len(self._X_train) < 2:
            return 1
        _, dists = self._enn.about_neighbors(self._X_train, k=self._config.k)
        dist_i = dists.mean(axis=1, keepdims=True)
        return 1 / dist_i.mean()

    def _ts_pick_cells(self, num_samples):
        assert len(self._X_train) > 0
        y = self._Y_train

        n = len(y)
        se = y.std() / np.sqrt(n)

        if self._config.max_cell:
            se = 0 * se
        else:
            # We Thompson sample over the Voronoi cells.
            # Our model of the function values in a cell is N(y, se**2)
            # where y is the (single) measured value in the cell, and
            #  se**2 is the (homoscedastic) variance, estimated by
            #  var(y)/N over all measured y's.

            if self._config.ts_enn:
                if n > 1:
                    _, dists = self._enn.about_neighbors(self._X_train, k=self._config.k)
                    dist_i = dists.mean(axis=1, keepdims=True)
                    dist_norm = dist_i.mean()
                    w = dist_i / dist_norm
                    w = w / w.sum(axis=0, keepdims=True)
                    se = se * np.sqrt(w)

        y = y + se * np.random.normal(size=(n, num_samples))
        i = np.where(y == y.max(axis=0, keepdims=True))[0]

        return self._X_train[i, :]

    def _sample_boundary(self, x_0):
        u = random_directions(len(x_0), self._num_dim)
        if self._config.region_type == "fn_fast":
            return farthest_neighbor_fast(self._enn, x_0, u, p_boundary_is_neighbor=self._config.p_boundary_is_neighbor)
        elif self._config.region_type == "fn":
            return farthest_neighbor(self._enn, x_0, u, p_boundary_is_neighbor=self._config.p_boundary_is_neighbor)
        elif self._config.region_type == "cr":
            return confidence_region_fast(self._enn, x_0, u, se_max=self._config.se_max, num_steps=100)
        elif self._config.region_type in ["far", "farp"]:
            return far_as_you_can_go(x_0, u)
        else:
            assert False, self._config.region_type

    def _xform_dm2(self, u):
        a = 1e-9
        b = 1
        d = self._num_dim
        if d == 1:
            return a * (b / a) ** u
        else:
            return (a ** (d - 1) + u * (b ** (d - 1) - a ** (d - 1))) ** (1 / (d - 1))

    def _sample_in_cell(self, x_0, x_far):
        # We want to uniformly sample over the Voronoi cell, but this is
        #  easier. Maybe we'll come up with something better.

        alpha = np.random.uniform(size=(len(x_0), 1))

        if self._config.weight_by_length:
            dists = np.linalg.norm(x_0 - x_far, axis=1, keepdims=True)
            dists = dists / dists.sum(axis=0, keepdims=True)
            i = np.random.choice(np.arange(len(dists)), size=len(dists), replace=True, p=dists.flatten())
            x_0 = x_0[i]
            x_far = x_far[i]

        if self._config.stagger:
            l_s_min = np.log(1e-4)
            l_s_max = np.log(1)
            alpha = np.exp(l_s_min + (l_s_max - l_s_min) * alpha)

        if self._config.region_type == "farp":
            assert not self._config.stagger
            alpha = 1 - self._xform_dm2(alpha)

        x_cand = alpha * x_0 + (1 - alpha) * x_far

        if self._config.keep_bdy:
            x_cand = np.concatenate([x_cand, x_far], axis=0)

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

    def _pareto_fronts_strict(self, x_cand, num_arms):
        assert self._config.num_over_sample_per_arm == 1, self._config.num_over_sample_per_arm
        mvn = self._enn.posterior(x_cand)

        i = np.argsort(-mvn.mu, axis=0).flatten()
        x_cand = x_cand[i]
        se = mvn.se[i]

        i_all = list(range(len(se)))
        i_keep = []

        while len(i_keep) < num_arms:
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

        i_keep = np.array(i_keep)
        x_arms = x_cand[i_keep]

        assert len(x_arms) == num_arms, (len(x_arms), num_arms)
        return x_arms

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
        if self._config.se_2 is not None:
            y = np.concatenate([y, self._config.se_2 * mvn.se_2], axis=1)

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
        self._enn.calibrate(self._var_calib())
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

    def _candidates(self, num_arms):
        if self._config.num_boundary > 0:
            num_boundary = int(self._config.num_boundary * 1.2)
        else:
            num_boundary = 0
        if self._config.num_interior > 0:
            num_interior = int(self._config.num_interior * 1.2)
        else:
            num_interior = 0

        x_0 = self._ts_pick_cells((num_boundary + num_interior) * num_arms)

        x_cand = np.empty(shape=(0, self._num_dim))
        i_cut = num_boundary * num_arms

        x_bdy = self._sample_boundary(x_0)

        if num_boundary > 0:
            x_cand = np.concatenate([x_cand, x_bdy[:i_cut]], axis=0)
        if num_interior > 0:
            x_cand = np.concatenate([x_cand, self._sample_in_cell(x_0[i_cut:], x_bdy[i_cut:])], axis=0)

        x_cand = np.unique(x_cand, axis=0)
        num_desired = num_arms * (self._config.num_boundary + self._config.num_interior)
        if len(x_cand) < num_desired:
            x_cand = np.concatenate(
                [x_cand, np.random.uniform(size=(num_desired - len(x_cand), self._num_dim))],
                axis=0,
            )
        else:
            x_cand = x_cand[:num_desired]

        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())
        assert len(np.unique(x_cand, axis=0)) == x_cand.shape[0], (len(np.unique(x_cand, axis=0)), x_cand.shape[0])

        return x_cand

    def _candidates_quick(self, num_arms):
        x_0 = self._ts_pick_cells(self._config.num_quick * num_arms)
        assert x_0.min() >= 0.0 and x_0.max() <= 1.0, (x_0.min(), x_0.max())

        _, dists = self._enn.about_neighbors(x_0, k=self._config.k)
        dists = dists.mean(axis=1, keepdims=True)
        x_target = np.random.uniform(size=(x_0.shape[0], self._num_dim))
        u = x_target - x_0
        u = u / np.linalg.norm(u, axis=1, keepdims=True)
        # u = random_directions(len(x_0), self._num_dim)
        x_target = np.minimum(1.0, np.maximum(0.0, x_0 + dists * u))

        alpha = np.random.uniform(size=(x_0.shape[0], 1))
        # alpha = 10 ** (-6 * alpha)

        x_cand = x_0 + alpha * (x_target - x_0)

        x_cand = np.unique(x_cand, axis=0)
        num_desired = num_arms * self._config.num_quick
        if len(x_cand) < num_desired:
            x_cand = np.concatenate(
                [x_cand, np.random.uniform(size=(num_desired - len(x_cand), self._num_dim))],
                axis=0,
            )

        assert len(x_cand) == len(np.unique(x_cand, axis=0)), (len(x_cand), len(np.unique(x_cand, axis=0)))
        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max(), dists.max())
        return x_cand

    def _draw_two_level(self, num_arms):
        if self._config.num_quick > 0:
            x_cand = self._candidates_quick(num_arms)
        else:
            x_cand = self._candidates(num_arms)

        if len(x_cand) == num_arms:
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
        elif self._config.acq == "pareto_strict":
            return self._pareto_fronts_strict(x_cand, num_arms)
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
        assert len(np.unique(x_a, axis=0)) == num_arms, (len(np.unique(x_a, axis=0)), num_arms)

        return x_a
