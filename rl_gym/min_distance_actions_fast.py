import numpy as np


class MinDistanceActionsFast:
    def __init__(self, data, k_slim=10):
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

    def _slim(self, k_slim, d):
        return d.trajectory.states, d.trajectory.actions
        states = d.trajectory.states
        n = states.shape[1]
        k = min(n, max(100, n // k_slim))
        i = np.random.choice(list(range(n)), size=(k,), replace=False)
        return states[:, i], d.trajectory.actions[:, i]

    def __call__(self, policy):
        dist_2_cs = np.cumsum((self._actions - policy(self._states)) ** 2)
        dist_2 = np.diff(dist_2_cs[self._idx], prepend=0)
        return np.sqrt(dist_2 / self._n).min()
