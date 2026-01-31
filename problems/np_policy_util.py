import numpy as np


def set_params_pm1(policy, x):
    x = np.asarray(x, dtype=np.float64)
    assert x.shape == (policy._num_p,), x.shape
    assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
    policy._x_orig = x.copy()
    policy._set_derived(policy._x_orig)
