import threading

import numpy as np
import torch

import common.all_bounds as all_bounds
from optimizer.ask_tell_inverter import AskTellInverter, ATIStopped, ATITimeoutError
from turbo_m_ref.turbo_1 import Turbo1
from turbo_m_ref.turbo_m import TurboM


class TuRBODesigner:
    """
    David Eriksson and Matthias Poloczek. Scalable constrained bayesian optimization. In
        Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International
        Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine
        Learning Research, pages 730–738. PMLR, 13–15 Apr 2021. URL https://proceedings.mlr.press/v130/eriksson21a.html.
    """

    def __init__(self, policy, num_trust_regions=1, num_init=None, *, surrogate_type="original", ard=True):
        self._policy = policy
        self._num_trust_regions = num_trust_regions
        self._num_init = num_init
        self._turbo = None
        self._ard = ard
        self._num_arms = None
        self._surrogate_type = surrogate_type
        self._default_device = torch.empty(size=(1,)).device

    def _run_opt(self):
        if self._num_init is not None:
            num_init = max(self._num_arms, self._num_init)
            num_init = self._num_arms * int(num_init / self._num_arms + 0.5)
            assert num_init > 0, (num_init, self._num_init, self._num_arms)
        else:
            num_init = self._num_arms

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
                device=self._default_device,
                surrogate_type=self._surrogate_type,
                use_ard=self._ard,
            )
        else:
            assert self._surrogate_type == "original", "NYI: surrogate_type for TurboM"
            opt = TurboM(
                f=self._ati,
                lb=lb,
                ub=ub,
                n_init=num_init,
                n_trust_regions=self._num_trust_regions,
                batch_size=self._num_arms,
                max_evals=999999,
                verbose=False,
                device=self._default_device,
                use_ard=self._ard,
            )
        try:
            return opt.optimize()
        except (ATITimeoutError, ATIStopped):
            pass

    def _start(self):
        self._ati = AskTellInverter(timeout_seconds=30 * 60)
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
            y = [-d.trajectory.rreturn for d in data[-self._num_arms :]]
            # x = data[-1].policy.get_params()
            self._ati.tell(y)

        policies = []
        for p in self._ati.ask():
            policy = self._policy.clone()
            policy.set_params(p)
            policies.append(policy)
        assert len(policies) == num_arms, (len(policies), num_arms)
        return policies
