def _setup():
    import numpy as np

    np.random.seed(17)

    num_chains = 8
    num_dim = 300
    X = np.random.uniform(size=(num_chains, num_dim))
    u = np.random.normal(size=(num_chains, num_dim))
    u = u / np.linalg.norm(u, axis=1, keepdims=True)

    return X, u


def test_find_bounds():
    import numpy as np

    from sampling.hnr import find_bounds

    X, u = _setup()

    llambda = find_bounds(X, u, eps_bound=1e-6)

    bound = X + llambda[:, None] * u
    ub = np.abs(bound - 1).min()
    lb = np.abs(bound).min()

    assert np.all((ub < 1e-6) | (lb < 1e-6))


def test_perturb_normal():
    import numpy as np

    from sampling.hnr import find_bounds, perturb_normal

    X, u = _setup()
    eps_bound = 1e-6
    llambda_minus = find_bounds(X, -u, eps_bound)
    llambda_plus = find_bounds(X, u, eps_bound)

    X_1 = perturb_normal(X, u, 1e-3, llambda_minus, llambda_plus)
    assert np.all(X_1 > -eps_bound) and np.all(X_1 < 1 + eps_bound)


def test_perturb_uniform():
    import numpy as np

    from sampling.hnr import find_bounds, perturb_uniform

    X, u = _setup()
    eps_bound = 1e-6
    llambda_minus = find_bounds(X, -u, eps_bound)
    llambda_plus = find_bounds(X, u, eps_bound)

    X_1 = perturb_uniform(X, u, llambda_minus, llambda_plus)
    assert np.all(X_1 > -eps_bound) and np.all(X_1 < 1 + eps_bound)
