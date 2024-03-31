import numpy as np
from scipy.stats import truncnorm


def perturb_uniform(X, u, llambda_minus, llambda_plus):
    """
    Make a 1D perturbation from X along u
    Perturbation ~ U(-llambda_minus, llambda_plus)

    llambda_*: (num_chains,)
    """

    num_chains = X.shape[0]
    num_dim = X.shape[1]
    delta = np.random.uniform(size=(num_chains, num_dim))
    X_min = X - llambda_minus[:, None] * u
    X_max = X + llambda_plus[:, None] * u
    return X_min + delta * (X_max - X_min)


def perturb_normal(X, u, eps, llambda_minus, llambda_plus):
    """
    Make a 1D perturbation from X along u
    Perturbation ~ N(0, eps^2), but truncated to [-llambda_minus, llambda_plus]

    X: num_chains x num_dim, starting point
    u: num_chains x num_dim, unit-length direction vectors
    eps: scale of perturbation
    llambda_minus, llambda_plus: bounds (along u)
    """

    num_chains = X.shape[0]
    rv = truncnorm(-llambda_minus / eps, llambda_plus / eps, scale=eps)
    X_1 = X + np.array(rv.rvs(num_chains))[:, None] * u
    assert np.all((X_1.min(axis=1) >= 0) & (X_1.max(axis=1) <= 1)), "Perturbation failed"
    return X_1


def find_bounds(
    X: np.array,
    u: np.array,
    eps_bound: float,
):
    """
    Bisection search along u until bound is found within eps_bound
    Assumes a bounding box [0,1]^num_dim

    X: num_chains x num_dim, starting point
    u: num_chains x num_dim, unit-length direction vectors
    """
    num_chains = X.shape[0]
    l_low = np.zeros(shape=(num_chains, 1))
    l_high = np.ones(shape=(num_chains, 1))

    lb, ub = 0, 1

    def _accept(X):
        return (X.min(axis=1) >= lb) & (X.max(axis=1) <= ub)

    while (l_high - l_low).max() > eps_bound:
        l_mid = (l_low + l_high) / 2
        X_mid = X + l_mid * u
        a = _accept(X_mid)
        l_low[a] = l_mid[a]
        l_high[~a] = l_mid[~a]

    # TODO: maybe return l_mid
    # l_mid = (l_low + l_high) / 2
    llambda = l_low.flatten()
    return llambda
