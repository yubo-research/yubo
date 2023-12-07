import cma

import common.all_bounds as all_bounds


class CMADesigner:
    def __init__(self, policy):
        self._policy = policy
        self._n_told = 0
        self._es = None

    def init_center(self):
        return False

    def __call__(self, data, num_arms):
        if self._es is None:
            assert self._policy.num_params() > 1, "CMA needs num_params > 1"
            self._es = cma.CMAEvolutionStrategy(
                [0] * self._policy.num_params(),
                0.2,
                inopts={
                    "bounds": [
                        all_bounds.p_low,
                        all_bounds.p_high,
                    ],
                    "popsize": num_arms,
                },
            )

        assert num_arms == self._es.popsize, f"CMADesigner wants num_arms == {self._es.popsize} every time."

        n = len(data) - self._n_told
        if n > 0:
            todo = data[-n:]
            x = [d.policy.get_params() for d in todo]
            y = [d.trajectory.rreturn for d in todo]
            self._es.tell(x, y)
            self._n_told += len(todo)

        policies = []
        for x in self._es.ask(num_arms):
            policy = self._policy.clone()
            policy.set_params(x)
            policies.append((None, policy))
        return policies
