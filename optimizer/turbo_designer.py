import threading

import numpy as np

import common.all_bounds as all_bounds
from optimizer.ask_tell_inverter import AskTellInverter, ATIStopped, ATITimeoutError
from turbo_m_ref.turbo_1 import Turbo1
from turbo_m_ref.turbo_m import TurboM


class TuRBODesigner:
    def __init__(self, policy, num_trust_regions=1, num_init=None):
        self._policy = policy
        self._num_trust_regions = num_trust_regions
        self._num_init = num_init
        self._turbo = None
        self._num_arms = None

    def init_center(self):
        return False

    def _run_opt(self):
        if self._num_init is not None:
            num_init = self._num_arms * int(self._num_init / self._num_arms + 0.5)

        lb = np.array([all_bounds.x_low] * self._policy.num_params())
        ub = np.array([all_bounds.x_high] * self._policy.num_params())
        if self._num_trust_regions == 1:
            opt = Turbo1(
                f=self._ati,
                lb=lb,
                ub=ub,
                n_init=num_init,
                batch_size=self._num_arms,
                max_evals=999999,
                verbose=False,
            )
        else:
            opt = TurboM(
                f=self._ati,
                lb=lb,
                ub=ub,
                n_init=num_init,
                n_trust_regions=self._num_trust_regions,
                batch_size=self._num_arms,
                max_evals=999999,
                verbose=False,
            )
        try:
            return opt.optimize()
        except (ATITimeoutError, ATIStopped):
            pass

    def _start(self):
        self._ati = AskTellInverter(timeout_seconds=10)
        self._thread = threading.Thread(target=self._run_opt, args=())
        self._thread.start()

    def stop(self):
        self._ati.stop()
        self._thread.join()

    def __call__(self, data, num_arms):
        if self._num_arms is None:
            self._num_arms = num_arms
            self._start()

        if len(data) > 0:
            y = [d.trajectory.rreturn for d in data[-self._num_arms :]]
            # x = data[-1].policy.get_params()
            self._ati.tell(y)

        policies = []
        for p in self._ati.ask():
            policy = self._policy.clone()
            policy.set_params(p)
            policies.append(policy)
        assert len(policies) == num_arms, (len(policies), num_arms)
        return policies
