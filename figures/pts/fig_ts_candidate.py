import numpy as np


def prob_inner(p_inner, num_samples, num_tries):
    num_inner = int(p_inner * num_samples + 0.5)
    num_outer = num_samples - num_inner

    data = []
    for _ in range(num_tries):
        inner_max = np.random.normal(size=(num_inner,)).max()
        outer_max = np.random.normal(size=(num_outer,)).max()
        data.append(inner_max > outer_max)
    return np.array(data).mean()
