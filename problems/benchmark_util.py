import numpy as np


def mk_2d(x):
    x = np.asarray(x)
    if len(x) == 1:
        y = x
    else:
        no2 = len(x) // 2
        y = x[:no2].mean()
        x = x[no2:].mean()
    return np.array([x, y]).flatten()


def mk_4d(x):
    x = mk_2d(x)
    return np.concatenate((x, x))
