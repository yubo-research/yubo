import numpy as np
from scipy.stats import qmc


class NegDiscrepancy:
    def __init__(self, data):
        self._data = data

    def __call__(self, policy):
        x = np.array([policy.get_params()] + [datum.policy.get_params() for datum in self._data])
        x = (1 + x) / 2  # unit hypercube
        return -qmc.discrepancy(x)
