from dataclasses import dataclass

import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from nds import ndomsort

from model.edn import EpistemicNovelty
from model.enn import EpistemicNearestNeighbors
from sampling.knn_tools import random_directions
from sampling.ray_boundary import ray_boundary_np
from sampling.sampling_util import raasp_np, raasp_np_1d, raasp_np_choice, raasp_np_p, sobol_perturb_np

"""
- Pick starting point(s), x_0
- num_candidates: Sample each candidate randomly along one randomly-chosen dimension, full extent (global reach)
- Run Pareto(mu, sigma) over the candidates. Alternatively, run Pareto(mu, ~N(0,sigma^2)) over the candidates.
- Estimate the stddev (in x, diagonal only) of the Pareto front points.
- Sample num_candidates from N(x_0, stddev^2), run Pareto(mu, *) again.
- Sample arm  from the Pareto front.
"""


@dataclass
class ENNConfig:
    k: int
    num_candidates_per_arm: int = 0
    stagger: bool = False
    acq: str = "pareto"
    small_world_M: int = None

    num_over_sample_per_arm: int = 1

    candidate_generator: str = "sobol"
    tr_type: str = "mean"
    raasp_type: str = None
    thompson: bool = False
    met_3: str = None
    met_4: str = None

    k_novelty: int = None

    def __post_init__(self):
        assert self.num_over_sample_per_arm > 0


class AcqENN:
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
                # Behavior/descriptor surrogate
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

    def _i_pareto_front_selection(self, num_select, *metrics):
        combined_data = np.concatenate(metrics, axis=1)
        idx_front = np.array(ndomsort.non_domin_sort(-combined_data, only_front_indices=True))

        i_keep = []
        for n_front in range(1 + max(idx_front)):
            front_indices = np.where(idx_front == n_front)[0]
            if num_select is None:
                i_keep.extend(front_indices)
                break

            if len(i_keep) + len(front_indices) <= num_select:
                i_keep.extend(front_indices)
            else:
                remaining = num_select - len(i_keep)
                i_keep.extend(np.random.choice(front_indices, size=remaining, replace=False))
                break

        return np.array(i_keep)

    def _i_pareto_fronts_discrepancy(self, num_keep):
        y = self._y_train
        mvn = self._enn.posterior(self._x_train, exclude_nearest=True)
        discrep = np.abs(y - mvn.mu)

        return self._i_pareto_front_selection(num_keep, y, discrep)

    def _i_pareto_fronts_strict(self, x_cand, num_arms, *, x_center=None):
        assert self._config.num_over_sample_per_arm == 1, self._config.num_over_sample_per_arm
        mvn = self._enn.posterior(x_cand)

        if self._config.thompson:
            met_2 = mvn.se * np.random.normal(size=mvn.se.shape)
        else:
            met_2 = mvn.se

        mets = [mvn.mu, met_2]
        if self._config.met_3 is not None:
            if self._config.met_3 == "L2":
                dist = -np.linalg.norm(x_cand - x_center, axis=1)[:, None]
            elif self._config.met_3 == "L1":
                dist = -np.abs(x_cand - x_center).sum(axis=1)[:, None]
            elif self._config.met_3 == "L0":
                dist = -np.zeros_like(x_cand - x_center).sum(axis=1)[:, None]
            elif self._config.met_3 == "tau":
                dist = -self._tau(self._x_train, x_cand)
            else:
                assert False, self._config.met_3
            mets.append(dist)

        if self._config.met_4 is not None:
            if self._config.met_4 == "qd":
                diversity, diversity_se = self._novelty(x_cand, num_arms)
                if diversity is not None:
                    mets.append(diversity[:, None])
                    mets.append(diversity_se[:, None])
            else:
                assert False, self._config.met_4
        return self._i_pareto_front_selection(num_arms, *mets)

    def _tau(self, x_obs, x):
        assert x.ndim == 2, x.ndim
        diff = np.abs(x_obs[:, None, :] - x[None, :, :])
        return np.sum(np.min(diff, axis=0), axis=1, keepdims=True)

    def _pareto_fronts_strict(self, x_cand, num_arms):
        i_keep = self._i_pareto_fronts_strict(x_cand, num_arms)
        x_arms = x_cand[i_keep]

        assert num_arms is None or len(x_arms) == num_arms, (len(x_arms), num_arms)
        return x_arms

    def _pareto_fronts_dist(self, x_center, x_cand, num_arms):
        i_keep = self._i_pareto_fronts_strict(x_cand, num_arms, x_center=x_center)
        x_arms = x_cand[i_keep]

        assert num_arms is None or len(x_arms) == num_arms, (len(x_arms), num_arms)
        return x_arms

    def _novelty(self, x_cand, num_arms):
        en = EpistemicNovelty(self._config.k_novelty)
        d_se = np.zeros_like(self._d_train)
        en.add(self._d_train, d_se)

        mvn_d = self._enn_d.posterior(x_cand)
        nv, nv_se = en.novelty(mvn_d.mu, mvn_d.se)

        return nv, nv_se

    def _novelty_search(self, x_cand, num_arms):
        nv, nv_se = self._novelty(x_cand, num_arms)
        if nv is None:
            print("Novelty is None, returning uniform")
            return self._uniform(x_cand, num_arms)
        i_arms = self._i_pareto_front_selection(num_arms, nv[:, None], nv_se[:, None])
        return x_cand[i_arms]

    def _quality_diversity(self, x_cand, num_arms):
        mvn = self._enn.posterior(x_cand)
        quality = mvn.mu
        quality_se = mvn.se
        diversity, diversity_se = self._novelty(x_cand, num_arms)
        if diversity is None:
            print("Diversity is None, returning uniform")
            return self._uniform(x_cand, num_arms)

        i_arms = self._i_pareto_front_selection(num_arms, quality, quality_se, diversity[:, None], diversity_se[:, None])
        return x_cand[i_arms]

    def _uniform(self, x_cand, num_arms):
        i = np.random.choice(np.arange(len(x_cand)), size=num_arms, replace=False)
        return x_cand[i]

    def _edn(self, x_cand, mu_se):
        idx_front = np.array(ndomsort.non_domin_sort(-mu_se, only_front_indices=True))

        mvn_b = self._enn_d.posterior(x_cand, exclude_nearest=False)

        max_front = 1 + max(idx_front)
        edn = EpistemicNovelty(self._config.k_novelty)
        dns = np.zeros(shape=(len(x_cand), 1))
        dns_se = np.zeros(shape=(len(x_cand), 1))
        for n_front in range(max_front):
            front_indices = np.where(idx_front == n_front)[0]
            edn.add(mvn_b.mu[front_indices], mvn_b.se[front_indices])
            for i in front_indices:
                if n_front == 0 and len(front_indices) == 1:
                    dns[i] = np.inf
                    dns_se[i] = 0
                else:
                    dns[i], dns_se[i] = edn.dominated_novelty_of_last_addition()

        return dns, dns_se

    def _dominated_novelty_selection(self, x_cand, num_arms):
        if len(self._x_train) == 0:
            return self._uniform(x_cand, num_arms)

        mvn = self._enn.posterior(x_cand, exclude_nearest=False)
        mu_se = np.concatenate([mvn.mu, mvn.se], axis=1)
        dns, dns_se = self._edn(x_cand, mu_se)
        assert len(dns) == len(x_cand), (len(dns), len(x_cand))
        assert len(dns_se) == len(x_cand), (len(dns_se), len(x_cand))

        mu_se_dns = np.concatenate(
            [
                mvn.mu,
                mvn.se,
                dns,
                # dns_se,
            ],
            axis=1,
        )
        idx_front = np.array(ndomsort.non_domin_sort(-mu_se_dns, only_front_indices=True))

        i_keep = []
        for n_front in range(1 + max(idx_front)):
            front_indices = np.where(idx_front == n_front)[0]
            if len(i_keep) + len(front_indices) <= num_arms:
                i_keep.extend(front_indices)
            else:
                remaining = num_arms - len(i_keep)
                i_keep.extend(np.random.choice(front_indices, size=remaining, replace=False))
                break

        i_keep = sorted(i_keep)
        return x_cand[i_keep]

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
            return raasp_np_p(x_center, lb, ub, num_cand, stagger=self._config.stagger)
        elif self._config.raasp_type == "biased_raasp":
            return self._biased_raasp(x_center, num_cand, lb, ub)
        elif self._config.raasp_type == "two-stage":
            return self._two_stage_raasp(x_center, lb, ub, num_cand)
        else:
            return None

    def _two_stage_raasp(self, x_center, lb, ub, num_cand):
        assert len(x_center) == 1, len(x_center)
        lb = lb * np.ones(shape=(1, x_center.shape[-1]))
        ub = ub * np.ones(shape=(1, x_center.shape[-1]))

        x_cand = raasp_np_p(x_center, lb, ub, num_cand)

        for _ in range(1):
            x_cand = self._pareto_fronts_strict(x_cand, num_arms=None)
            # N.B.: This part only works with a single x_center.
            delta = x_cand - x_center
            lb = x_center + delta.min(axis=0)
            ub = x_center + delta.max(axis=0)

            print("XSTD:", (np.abs(delta).max(axis=0) > 0).sum(), np.abs(delta.max()))
            i_dim_allowed = np.where(np.abs(delta).max(axis=0) > 0)[0]
            # x_cand = np.random.uniform(lb, ub, size=(num_cand, self._num_dim))
            x_cand = raasp_np_p(x_center, lb, ub, num_cand, i_dim_allowed=i_dim_allowed)
        print("XC:", x_cand.shape, np.unique(x_cand, axis=0).shape)
        return x_cand

    def _raasp_or_sobol(self, x_center, num_cand, lb=0.0, ub=1.0):
        raasp = self._raasp(x_center, num_cand, lb, ub)
        if raasp is not None:
            print("RAASP")
            return raasp
        else:
            print("SOBOL")
            return self._draw_sobol(num_cand)

    def _trust_region(self, num_cand):
        if len(self._x_train) < 2:
            return None, self._draw_sobol(num_cand)
        x_center = self._x_train[[np.argmax(self._y_train)]]

        idx, dists = self._enn.about_neighbors(x_center)
        tr = dists.flatten()[-1]

        lb = np.maximum(0.0, x_center[0] - tr)
        ub = np.minimum(1.0, x_center[0] + tr)
        return raasp_np_p(x_center, lb, ub, num_cand)

    def _candidates(self, num_arms):
        num_cand = self._config.num_candidates_per_arm * num_arms
        x_0 = None
        x_cands = []
        for region_type in self._config.candidate_generator.split("+"):
            if region_type == "tr":
                x_cand = self._trust_region(num_cand)
            elif region_type == "sobol":
                x_cand = self._draw_sobol(num_cand)
            elif region_type == "best":
                x_0 = self._x_train[[np.argmax(self._y_train)]]
                raasp = self._raasp(x_0, num_cand)
                if raasp is not None:
                    x_cand = raasp
                else:
                    x_0 = x_0[np.random.choice(np.arange(len(x_0)), size=num_cand, replace=True)]
                    x_far = ray_boundary_np(x_0, random_directions(len(x_0), self._num_dim))
                    x_cand = self._sample_segments(x_0, x_far)
            elif region_type == "best_seg":
                x_0 = self._x_train[[np.argmax(self._y_train)]]
                x_far = ray_boundary_np(x_0, random_directions(len(x_0), self._num_dim))
                x_cand = self._sample_segments(x_0, x_far)
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

        return x_0, x_cand

    def _draw_two_level(self, num_arms):
        x_center, x_cand = self._candidates(num_arms)

        if len(x_cand) == num_arms:
            return x_cand

        x_cand = np.unique(x_cand, axis=0)

        if self._config.acq == "dominated_novelty":
            return self._dominated_novelty_selection(x_cand, num_arms)
        elif self._config.acq == "pareto_strict":
            return self._pareto_fronts_strict(x_cand, num_arms)
        elif self._config.acq == "pareto_dist":
            return self._pareto_fronts_dist(x_center, x_cand, num_arms)
        elif self._config.acq == "novelty_search":
            return self._novelty_search(x_cand, num_arms)
        elif self._config.acq == "quality_diversity":
            return self._quality_diversity(x_cand, num_arms)
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
