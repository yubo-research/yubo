from scipy.stats import qmc


class SobolDesigner:
    def __init__(self, policy):
        self._policy = policy
        max_points = 1024  # TODO: worry later
        self._xs = qmc.Sobol(policy.num_params()).random(max_points)

    def __call__(self, data):
        assert len(self._xs) > 0, "max_points exceeded"

        x = self._xs[0, :]
        self._xs = self._xs[1:, :]
        policy = self._policy.clone()
        policy.set_params(2 * x - 1)
        return policy
