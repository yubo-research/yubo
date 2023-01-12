import numpy as np


def distance_parameters(datum_0, policy):
    return np.sqrt(((datum_0.policy.get_params() - policy.get_params()) ** 2).mean())


def distance_actions(datum_0, policy):
    actions = policy(datum_0.trajectory.states)
    return np.sqrt(((datum_0.trajectory.actions - actions) ** 2).mean())


def distance_actions_corr(datum_0, policy):
    actions = policy(datum_0.trajectory.states)
    return 0.5 * (1 - np.corrcoef(datum_0.trajectory.actions, actions))


def min_weighted_dist(distance_fn, data, policy):
    weights = np.array([d.trajectory.rreturn for d in data])
    weights = 1 + (weights - weights.min()) / (1e-9 + weights.max() - weights.min())
    dists = np.array([distance_fn(datum, policy) for datum in data])
    return dists.min()
