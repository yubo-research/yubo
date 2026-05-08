import logging
import time

import numpy as np
from ConfigSpace import ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue

import common.all_bounds as all_bounds
from optimizer.designer_asserts import assert_scalar_rreturn


class SMACDesigner:
    """Bayesian optimization designer using SMAC with RF surrogate and EI acquisition."""

    def __init__(self, policy):
        self._policy = policy
        self._num_params = policy.num_params()
        seed = policy.problem_seed + 2718  # offset to differentiate from other designers

        # Build ConfigurationSpace with D uniform-float hyperparameters
        cs = ConfigurationSpace(seed=seed)
        for i in range(self._num_params):
            cs.add(Float(f"x_{i}", bounds=(all_bounds.p_low, all_bounds.p_high)))

        # Create Scenario - n_trials is large since we control the loop externally
        scenario = Scenario(
            configspace=cs,
            n_trials=100000,
            seed=seed,
            deterministic=True,
        )

        # Suppress SMAC's verbose logging
        logging.getLogger("smac").setLevel(logging.WARNING)

        # Dummy target function (never called in ask/tell mode)
        def _dummy_target(config, seed=0):
            return 0.0

        # Instantiate facade with HyperparameterOptimizationFacade (RF-based)
        self._smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=_dummy_target,
            overwrite=True,
        )

        # Store pending trial infos for tell() on next call
        self._pending_trial_infos = []

    def __call__(self, data, num_arms, *, telemetry=None):
        # Tell previous observations
        num_pending = len(self._pending_trial_infos)
        if num_pending > 0 and data is not None:
            todo = data[-num_pending:]
            assert_scalar_rreturn(todo)
            for d, trial_info in zip(todo, self._pending_trial_infos):
                rreturn = d.trajectory.rreturn
                # SMAC minimizes cost; negate rreturn to maximize
                self._smac.tell(trial_info, TrialValue(cost=-rreturn, time=1.0))

        self._pending_trial_infos = []
        policies = []

        if telemetry is not None:
            telemetry.set_dt_fit(0.0)

        t0 = time.perf_counter()

        # Ask num_arms new configs
        for _ in range(num_arms):
            trial_info = self._smac.ask()
            config = trial_info.config

            # Convert config dict to numpy param vector
            x = np.array([config[f"x_{i}"] for i in range(self._num_params)])

            # Clone policy and set params
            policy = self._policy.clone()
            policy.set_params(x)
            policies.append(policy)

            self._pending_trial_infos.append(trial_info)

        dt_select = time.perf_counter() - t0
        if telemetry is not None:
            telemetry.set_dt_select(dt_select)

        return policies
