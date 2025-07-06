from dataclasses import dataclass

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from nds import ndomsort

from model.enn import EpistemicNearestNeighbors
from sampling.knn_tools import random_directions  #  single_coordinate_perturbation
from sampling.ray_boundary import ray_boundary_np
from sampling.sampling_util import raasp_np, raasp_np_choice, raasp_np_p, sobol_perturb_np

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
    num_candidates_per_arm: int = 0
    stagger: bool = False
    acq: str = "pareto"
    small_world_M: int = None

    num_over_sample_per_arm: int = 1

    region_type: str = "sobol"
    tr_type: str = "mean"
    raasp_type: str = None

    k_novelty: int = None

    def __post_init__(self):
        assert self.num_over_sample_per_arm > 0


class AcqENN:
    # TODO: Yvar

    def __init__(self, num_dim, config: ENNConfig):
        assert config.k > 0, config
        self._num_dim = num_dim

        self._x_train = np.empty(shape=(0, num_dim))
        self._y_train = np.empty(shape=(0, 1))
        self._d_train = None

        self._config = config
        self._enn = None
        self._enn_d = None

    def add(self, x, y, d=None):
        if len(x) == 0:
            return
        x = np.asarray(x)
        y = np.asarray(y)
        if self._enn is None:
            # Metric surrogate
            self._enn = EpistemicNearestNeighbors(k=self._config.k, small_world_M=self._config.small_world_M)
            self._enn.add(x, y)
        else:
            self._enn.add(x, y)
        self._x_train = np.append(self._x_train, x, axis=0)
        self._y_train = np.append(self._y_train, y, axis=0)

        if self._config.k_novelty is not None:
            d = np.asarray(d)
            if self._enn_d is None:
                # Descriptor surrogate (predicts d)
                self._enn_d = EpistemicNearestNeighbors(k=self._config.k, small_world_M=self._config.small_world_M)
                self._enn_d.add(x, d)
                self._d_train = np.empty(shape=(0, d.shape[-1]))
            else:
                self._enn_d.add(x, d)
            self._d_train = np.append(self._d_train, d, axis=0)

    def keep_top_n(self, num_keep):
        if len(self._x_train) <= num_keep:
            return self._x_train, self._y_train

        # Use a greedy algorithm to remove points one-by-one
        #  otherwise you could remove two points that are close to each other.
        while len(self._x_train) > num_keep:
            i = self._i_pareto_fronts_discrepancy(len(self._x_train) - 1)
            self._x_train = self._x_train[i]
            self._y_train = self._y_train[i]
        return self._x_train, self._y_train

    def _sample_segments(self, x_0, x_far):
        assert x_0.shape == x_far.shape, (x_0.shape, x_far.shape)

        alpha = np.random.uniform(size=(len(x_0), 1))

        if self._config.stagger:
            l_s_min = np.log(1e-4)
            l_s_max = np.log(1.0)
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

    def _i_pareto_fronts_discrepancy(self, num_keep):
        # Full fronts, then random selection from *last* front.
        y = self._y_train
        mvn = self._enn.posterior(self._x_train, exclude_nearest=True)
        discrep = np.abs(y - mvn.mu)

        i = np.argsort(-y, axis=0).flatten()
        discrep = discrep[i]

        i_all = list(range(len(discrep)))
        i_keep = []

        num_tries = 0
        while len(i_keep) < num_keep:
            num_tries += 1
            assert num_tries < 100, num_tries
            discrep_max = -1e99
            i_front = []
            for i in i_all:
                if discrep[i] >= discrep_max:
                    i_front.append(i)
                    discrep_max = discrep[i]
            if len(i_keep) + len(i_front) <= num_keep:
                i_keep.extend(i_front)
            else:
                i_keep.extend(np.random.choice(i_front, size=num_keep - len(i_keep), replace=False))
            i_all = sorted(set(i_all) - set(i_front))

        return np.array(i_keep)

    def _i_pareto_fronts_strict(self, x_cand, num_arms, exclude_nearest=False):
        # Full fronts, then random selection from *last* front.
        assert self._config.num_over_sample_per_arm == 1, self._config.num_over_sample_per_arm
        mvn = self._enn.posterior(x_cand, exclude_nearest=exclude_nearest)

        i = np.argsort(-mvn.mu, axis=0).flatten()
        x_cand = x_cand[i]
        se = mvn.se[i]
        i_rev = list(range(len(se)))
        i_rev = np.array(i_rev)[i]

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

        return i_rev[np.array(i_keep)]

    def _pareto_fronts_strict(self, x_cand, num_arms, exclude_nearest=False):
        i_keep = self._i_pareto_fronts_strict(x_cand, num_arms, exclude_nearest)
        x_arms = x_cand[i_keep]

        assert len(x_arms) == num_arms, (len(x_arms), num_arms)
        return x_arms

    def _uniform(self, x_cand, num_arms):
        i = np.random.choice(np.arange(len(x_cand)), size=num_arms, replace=False)
        return x_cand[i]

    # def _dns(self, x_cand, mu_se):
    #     fronts = ndomsort.non_domin_sort(mu_se)
    #     mvn_d = self._enn_d.posterior(x_cand, exclude_nearest=False)
    #     # TODO: Consider se
    #     mu_d = mvn_d.mu
    #     zero = np.zeros(shape=(1, 1))

    #     enn_dn = EpistemicNearestNeighbors(
    #         mu_d[fronts[0]],
    #         zero,
    #         k=self._config.k_novelty,
    #     )
    #     dns = []
    #     for i in fronts[0]:
    #         idx, dists = enn_dn.about_neighbors(mu_d[i])
    #         idx = idx.flatten()
    #         dists = dists.flatten()
    #         dns.append(np.mean(dists))
    #     enn_dn.add(mu_d[i], zero)

    def _dominated_novelty_selection(self, x_cand, num_arms):
        assert False, "NYI"
        # if len(self._x_train) == 0:
        #     return self._uniform(x_cand, num_arms)

        # mvn = self._enn.posterior(x_cand, exclude_nearest=False)
        # mu_se = np.concatenate([mvn.mu, mvn.se], axis=1)
        # fronts = ndomsort.non_domin_sort(mu_se)
        # mvn_d = self._enn_d.posterior(x_cand, exclude_nearest=False)
        # # TODO: Consider se
        # mu_d = mvn_d.mu

        # zero = np.zeros(shape=(1, 1))
        # enn_dn = EpistemicNearestNeighbors(
        #     mu_d[fronts[0]],
        #     zero,
        #     k=self._config.k_novelty,
        # )

        # dns = np.array(dns)
        # assert len(dns) == len(x_cand), (len(dns), len(x_cand))

        # # TODO: Pareto(mu, se, dns)
        # assert False, "TODO"
        # i_selected = np.argsort(-dns)[:num_arms]
        # return x_cand[i_selected]

    def _draw_sobol(self, num_cand, bounds=None):
        if bounds is None:
            bounds = torch.tensor([[0.0, 1.0]] * self._num_dim).T
        else:
            bounds = torch.as_tensor(bounds, dtype=torch.float32)
        return (
            draw_sobol_samples(
                bounds=bounds,
                n=num_cand,
                q=1,
            )
            .detach()
            .numpy()
        ).squeeze(1)

    def _biased_raasp(self, x_center, num_cand, lb=0.0, ub=1.0):
        x_cand_0 = raasp_np(x_center, lb, ub, num_cand, num_pert=1)
        num_dim_raasp = max(1, min(20, int(0.2 * self._num_dim + 0.5)))
        x_cand_0 = self._pareto_fronts_strict(x_cand_0, num_dim_raasp)
        delta = np.abs(x_cand_0 - x_center)
        i_dim_perturbed = np.where(delta.sum(axis=0) > 1e-6)[0]
        assert len(i_dim_perturbed) > 0, i_dim_perturbed
        mask = np.zeros(shape=(num_cand, self._num_dim), dtype=bool)
        mask[:, i_dim_perturbed] = True
        x_cand_1 = sobol_perturb_np(x_center, lb, ub, num_cand, mask)
        return np.concatenate([x_cand_0, x_cand_1], axis=0)

    def _raasp(self, x_center, num_cand, lb=0.0, ub=1.0):
        if self._config.raasp_type == "raasp":
            return raasp_np(x_center, lb, ub, num_cand)
        elif self._config.raasp_type == "raasp_choice":
            return raasp_np_choice(x_center, lb, ub, num_cand)
        elif self._config.raasp_type == "raasp_p":
            return raasp_np_p(x_center, lb, ub, num_cand)
        elif self._config.raasp_type == "biased_raasp":
            return self._biased_raasp(x_center, num_cand, lb, ub)
        elif self._config.raasp_type == "raasp_1":
            return raasp_np(x_center, lb, ub, num_cand, num_pert=1)
        else:
            return None

    def _raasp_or_sobol(self, x_center, num_cand, lb=0.0, ub=1.0):
        raasp = self._raasp(x_center, num_cand, lb, ub)
        if raasp is not None:
            return raasp
        else:
            return self._draw_sobol(num_cand)

    def _trust_region(self, num_cand):
        if len(self._x_train) < 3:
            return self._draw_sobol(num_cand)
        x_center = self._x_train[[np.argmax(self._y_train)]]

        mvn = self._enn.posterior(self._x_train, exclude_nearest=True, k=2)
        loocv = mvn.mu - self._y_train  # / mvn.se
        if loocv.std() < 1e-9:
            return self._raasp_or_sobol(x_center, num_cand)
        idx, dists = self._enn.about_neighbors(x_center, k=len(self._x_train))
        idx = idx.flatten()
        dists = dists.flatten()
        d = np.abs(loocv)
        if self._config.tr_type == "mean":
            d = d[idx] / d.mean()
        elif self._config.tr_type == "0":
            d = d[idx] / d[0]
        elif self._config.tr_type == "median":
            d = d[idx] / np.median(d)
        else:
            assert False, self._config.tr_type

        if d[0] > 1.0 + 1e-9:
            tr = dists[1] / 2
        else:
            d[0] = 0.0
            i = np.where(d > 1.0 + 1e-9)[0]
            if len(i) == 0:
                tr = dists[1] / 2
            else:
                i = i.min()
                if i == len(idx):
                    return self._draw_sobol(num_cand)
                tr = (dists[i] + dists[i - 1]) / 2

        bounds = np.array([x_center[0] - tr, x_center[0] + tr])
        bounds = np.maximum(0.0, bounds)
        bounds = np.minimum(1.0, bounds)
        return self._raasp_or_sobol(x_center, num_cand, bounds[0], bounds[1])

    def _candidates(self, num_arms):
        num_cand = self._config.num_candidates_per_arm * num_arms
        x_cands = []
        for region_type in self._config.region_type.split("+"):
            if region_type == "tr":
                x_cand = self._trust_region(num_cand)
            elif region_type == "sobol":
                return self._draw_sobol(num_cand)
            elif region_type == "best":
                x_0 = self._x_train[[np.argmax(self._y_train)]]
                raasp = self._raasp(x_0, num_cand)
                if raasp is not None:
                    return raasp
                else:
                    x_0 = x_0[np.random.choice(np.arange(len(x_0)), size=num_cand, replace=True)]
                    x_far = ray_boundary_np(x_0, random_directions(len(x_0), self._num_dim))
                    return self._sample_segments(x_0, x_far)
            elif region_type == "pivots":
                x_0 = self._select_pivots(num_cand)
                x_far = ray_boundary_np(x_0, random_directions(len(x_0), self._num_dim))
                x_cand = self._sample_segments(x_0, x_far)
            elif region_type == "rand_train":
                x_0 = self._x_train[np.random.choice(np.arange(len(self._x_train)), size=num_arms, replace=True)]
                x_0 = x_0[np.random.choice(np.arange(len(x_0)), size=num_cand, replace=True)]
                x_far = ray_boundary_np(x_0, random_directions(len(x_0), self._num_dim))
                x_cand = self._sample_segments(x_0, x_far)
            elif region_type == "convex":
                # slow
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
                assert False, region_type

            x_cands.append(x_cand)

        x_cand = np.concatenate(x_cands, axis=0)
        x_cand = np.unique(np.concatenate(x_cands, axis=0), axis=0)

        assert x_cand.min() >= 0.0 and x_cand.max() <= 1.0, (x_cand.min(), x_cand.max())
        assert len(np.unique(x_cand, axis=0)) == x_cand.shape[0], (len(np.unique(x_cand, axis=0)), x_cand.shape[0])

        return x_cand

    def _draw_two_level(self, num_arms):
        x_cand = self._candidates(num_arms)

        if len(x_cand) == num_arms:
            return x_cand

        if self._config.acq == "dominated_novelty":
            return self._dominated_novelty_selection(x_cand, num_arms)
        elif self._config.acq == "pareto_strict":
            return self._pareto_fronts_strict(x_cand, num_arms)
        elif self._config.acq == "uniform":
            return self._uniform(x_cand, num_arms)
        else:
            assert False, self._config.acq

    def draw(self, num_arms):
        # print("T:", len(self._x_train), len(self._y_train))
        if len(self._x_train) == 0:
            # x_a = 0.5 + np.zeros(shape=(1, self._num_dim))
            # x_a = np.append(x_a, np.random.uniform(size=(num_arms - 1, self._num_dim)), axis=0)
            x_a = np.random.uniform(size=(num_arms, self._num_dim))
        else:
            x_a = self._draw_two_level(num_arms)

        assert x_a.shape == (num_arms, self._num_dim), (num_arms, self._num_dim, x_a.shape)
        assert len(np.unique(x_a, axis=0)) == num_arms, (len(np.unique(x_a, axis=0)), num_arms)

        return x_a
