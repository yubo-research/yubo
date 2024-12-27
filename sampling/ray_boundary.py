import numpy as np


def ray_boundary(x, u):
    t_min = (0 - x) / u  # When r_i(t) = 0
    t_max = (1 - x) / u  # When r_i(t) = 1

    t_candidates = np.where(u > 0, t_max, np.where(u < 0, t_min, np.inf))
    t_min_positive = np.min(t_candidates[t_candidates > 0])

    return x + t_min_positive * u
