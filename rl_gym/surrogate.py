from typing import List

import numpy as np

from bbo.gp import GP
from rl_gym.datum import Datum


class Surrogate:
    """Model and acquisition function.
    Provides acqfn interface (__call__())
    """

    def __init__(self, data: List[Datum], behavioral_distance):
        self._behavioral_distance = behavioral_distance

        self._build_model(data)

    def _build_model(self, data):
        n = len(data)
        distance_matrix_train = np.zeros(shape=(n, n))
        y_train = []
        for i, d in enumerate(data):
            y_train.append(d.trajectory.rreturn)
            distances = self._behavioral_distance.distances(d.policy)
            assert len(distances) == n, (len(distances), n)
            for j, dist in enumerate(distances):
                if i == j:
                    distance_matrix_train[i, j] = 0
                else:
                    assert dist >= 0, dist
                    distance_matrix_train[i, j] = dist
        distance_matrix_train = (distance_matrix_train + distance_matrix_train.T) / 2
        self._gp = GP(distance_matrix_train, np.array(y_train)[:, None])

    def __call__(self, policy):
        distances = np.atleast_2d(self._behavioral_distance.distances(policy)).T
        y, y_var = self._gp(distances)
        y_std = np.sqrt(y_var)
        return y_std
