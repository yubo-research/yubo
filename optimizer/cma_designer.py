import time

import cma
import numpy as np

import common.all_bounds as all_bounds
from optimizer.designer_asserts import assert_scalar_rreturn


class CMAESDesigner:
    def __init__(self, policy):
        self._policy = policy
        self._n_told = 0
        seed = policy.problem_seed + 98765
        self._rng = np.random.default_rng(seed)
        self._es = None

    def __call__(self, data, num_arms, *, telemetry=None):
        assert num_arms > 1, "CMAESDesigner does not support num_arms < 2"
        if self._es is None:
            assert self._policy.num_params() > 1, "CMA needs num_params > 1"
            x_0 = all_bounds.p_low + all_bounds.p_width * self._rng.uniform(
                size=(self._policy.num_params(),)
            )
            sigma_0 = 0.2
            self._es = cma.CMAEvolutionStrategy(
                x_0,
                sigma_0,
                inopts={
                    "bounds": [
                        all_bounds.p_low,
                        all_bounds.p_high,
                    ],
                    "popsize": num_arms,
                },
            )

        assert num_arms == self._es.popsize, (
            f"CMAESDesigner wants num_arms == {self._es.popsize} every time."
        )

        n = len(data) - self._n_told
        if n > 0:
            todo = data[-n:]
            assert_scalar_rreturn(todo)
            x = [d.policy.get_params() for d in todo]
            y = [-d.trajectory.rreturn for d in todo]
            self._es.tell(x, y)
            self._n_told += len(todo)

        if telemetry is not None:
            telemetry.set_dt_fit(0.0)
        t0 = time.perf_counter()
        policies = []
        for x in self._es.ask(num_arms):
            policy = self._policy.clone()
            policy.set_params(x)
            policies.append(policy)
        dt_select = time.perf_counter() - t0
        if telemetry is not None:
            telemetry.set_dt_select(dt_select)
        return policies
