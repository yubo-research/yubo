import common.all_bounds as all_bounds
from sampling.lhd import latin_hypercube_design


class LHDDesigner:
    def __init__(self, policy, max_points=2**12):
        self._policy = policy
        self._seed = policy.problem_seed + 99
        self._i_seed = 0
        self._policies = []

    def _get_policy(self):
        if len(self._policies) == 0:
            self._policies = latin_hypercube_design(
                num_samples=2 * self._policy.num_params(),
                num_dim=self._policy.num_params(),
            )

        x = self._policies[0, :]
        self._policies = self._policies[1:, :]
        policy = self._policy.clone()
        policy.set_params(all_bounds.p_low + all_bounds.p_width * x)
        return policy

    def __call__(self, _, num_arms):
        policies = []
        for _ in range(num_arms):
            policies.append(self._get_policy())
        return policies
