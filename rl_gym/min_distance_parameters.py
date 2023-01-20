import numpy as np

from rl_gym.distance import distance_parameters, distance_parameters_toroidal


class MinDistanceParameters:
    def __init__(self, data, toroidal):
        self._data = data
        if toroidal:
            self._distance_fn = distance_parameters_toroidal
        else:
            self._distance_fn = distance_parameters

    def __call__(self, policy):
        return self.distances(policy).min()

    def distances(self, policy):
        return np.array([self._distance_fn(datum, policy) for datum in self._data])
