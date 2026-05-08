import numpy as np


def greedy_maximin(x, num_subsamples):
    # Credit to ChatGPT

    num_samples = x.shape[0]
    assert num_subsamples < num_samples, (num_samples, num_subsamples)

    selected_indices = []

    first_idx = np.random.choice(num_samples)
    selected_indices.append(first_idx)

    diff = x - x[first_idx]
    min_dists = np.linalg.norm(diff, axis=1)
    min_dists[first_idx] = 0

    for _ in range(1, num_subsamples):
        next_idx = np.argmax(min_dists)
        selected_indices.append(next_idx)

        dists = np.linalg.norm(x - x[next_idx], axis=1)
        min_dists = np.minimum(min_dists, dists)

    return selected_indices


def top_k(x, k):
    assert len(x.shape) == 1, x.shape

    if k >= len(x):
        return np.argsort(x)[::-1]

    return np.argpartition(x, -k)[-k:]


def intersect_with_box(x_inside, x_outside):
    t_min, t_max = 0.0, 1.0
    d = x_outside - x_inside
    for i in range(len(x_inside)):
        di = d[i]
        if abs(di) < 1e-15:
            if not (0.0 <= float(x_inside[i]) <= 1.0):
                return None
            continue
        t0 = (0.0 - x_inside[i]) / di
        t1 = (1.0 - x_inside[i]) / di
        t_lo, t_hi = min(t0, t1), max(t0, t1)
        t_min = max(t_min, t_lo)
        t_max = min(t_max, t_hi)

    if t_min > t_max:
        return None
    return x_inside + t_min * d
