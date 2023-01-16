import numpy as np

from rl_gym.distance import distance_parameters


class MinDistanceParameters:
    def __init__(self, data):
        self._data = data

    def __call__(self, policy):
        return self.distances(policy).min()

    def distances(self, policy):
        dists = np.array([distance_parameters(datum, policy) for datum in self._data])
        return dists
