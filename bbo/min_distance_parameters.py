import numpy as np

from bbo.distance import distance_parameters


class MinDistanceParameters:
    def __init__(self, data):
        self._data = data

    def __call__(self, policy):
        dists = np.array([distance_parameters(datum, policy) for datum in self._data])
        return dists.min()
