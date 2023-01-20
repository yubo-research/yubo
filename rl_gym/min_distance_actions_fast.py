from typing import List

import numpy as np

from rl_gym.datum import Datum


class MinDistanceActionsFast:
    def __init__(self, data: List[Datum], k_slim=10, ttype="eu"):
        self._states = []
        self._actions = []
        self._idx = []
        self._n = []
        i = 0
        for d in data:
            s, a = self._slim(k_slim, d)
            self._states.append(s)
            self._actions.append(a)
            n = s.shape[1]
            i += n
            self._idx.append(i - 1)
            self._n.append(n)
        self._states = np.concatenate(self._states, axis=1)
        self._actions = np.concatenate(self._actions, axis=1)
        self._idx = np.array(self._idx)
        self._n = np.array(self._n)
        if ttype == "eu":
            self._dists = self._eu_dists
        elif ttype == "corr":
            self._dists = self._corr_dists
        elif ttype == "cov":
            self._dists = self._cov_dists
        else:
            assert False

    def _slim(self, k_slim, d):
        if True:
            return d.trajectory.states, d.trajectory.actions
        else:
            states = d.trajectory.states
            n = states.shape[1]
            k = min(n, max(100, n // k_slim))
            i = np.random.choice(list(range(n)), size=(k,), replace=False)
            return states[:, i], d.trajectory.actions[:, i]

    def __call__(self, policy):
        return self.distances(policy).min()

    def distances(self, policy):
        return self._dists(policy)

    def _eu_dists(self, policy):
        dist_2_cs = np.cumsum((self._actions - policy(self._states)) ** 2)
        dist_2 = np.diff(dist_2_cs[self._idx], prepend=0)
        return np.sqrt(dist_2 / self._n)

    def _means(self, x):
        return np.diff(np.cumsum(x)[self._idx], prepend=0) / self._n

    def _corr_dists(self, policy):
        a = policy(self._states)
        xy = self._means(self._actions * a)
        x2 = self._means(self._actions**2)
        y2 = self._means(a**2)

        rho = xy / np.sqrt(1e-9 + x2 * y2)
        return 0.5 * (1 - rho)

    def _cov_dists(self, policy):
        a = policy(self._states)
        x = self._means(self._actions)
        y = self._means(a)
        xy = self._means(self._actions * a)

        # Don't normalize b/c
        #  we want the scale of the actions to matter
        cov = xy - x * y
        return -cov
