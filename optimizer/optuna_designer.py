import numpy as np
import optuna

import common.all_bounds as all_bounds


class OptunaDesigner:
    def __init__(self, policy):
        self._policy = policy
        seed = policy.problem_seed + 3141
        sampler = optuna.samplers.TPESampler(seed=seed)
        self._study = optuna.create_study(sampler=sampler, direction="maximize")

        self._trials = []

    def __call__(self, data, num_arms):
        num_todo = len(self._trials)
        todo = data[-num_todo:]
        for d, trial in zip(todo, self._trials):
            y = d.trajectory.rreturn
            self._study.tell(trial, y)

        self._trials = []
        policies = []
        for _ in range(num_arms):
            trial = self._study.ask()
            x = np.array(
                [
                    trial.suggest_float(
                        f"x_{i}",
                        all_bounds.p_low,
                        all_bounds.p_high,
                    )
                    for i in range(self._policy.num_params())
                ]
            )
            policy = self._policy.clone()
            policy.set_params(x)
            policies.append(policy)
            self._trials.append(trial)

        return policies
