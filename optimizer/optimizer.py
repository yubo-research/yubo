from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)

from bo.acq_idopt import AcqIDOpt
from bo.acq_iei import AcqIEI
from bo.acq_iopt import AcqIOpt
from bo.acq_iucb import AcqIUCB
from bo.acq_min_dist import AcqMinDist
from bo.acq_var import AcqVar

from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .datum import Datum
from .random_designer import RandomDesigner
from .sobol_designer import SobolDesigner
from .trajectories import collect_trajectory


class Optimizer:
    def __init__(self, env_conf, policy, cb_trace=None):
        self._env_conf = env_conf
        self._cb_trace = cb_trace
        self._data = []
        self._datum_best = None
        self._designers = {
            "random": RandomDesigner(policy),
            "sobol": SobolDesigner(policy),
            "minimax": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=False)),
            "minimax-toroidal": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=True)),
            "variance": BTDesigner(policy, AcqVar),
            "iopt": BTDesigner(policy, AcqIOpt, {"explore_only": True}),
            "iopt_ei": BTDesigner(policy, AcqIOpt, {"use_sqrt": True}),
            "ioptv_ei": BTDesigner(policy, AcqIOpt, {"use_sqrt": False}),
            "idopt": BTDesigner(policy, AcqIDOpt, acq_kwargs={"X_max": None, "bounds": None}),
            "ei": BTDesigner(policy, qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}),
            "iei": BTDesigner(policy, AcqIEI, acq_kwargs={"Y_max": None, "bounds": None}),
            "ucb": BTDesigner(policy, qUpperConfidenceBound, acq_kwargs={"beta": 1}),
            "iucb": BTDesigner(policy, AcqIUCB, acq_kwargs={"bounds": None}),
            "ax": AxDesigner(policy),
        }

    def _collect_trajectory(self, policy):
        return collect_trajectory(self._env_conf, policy, seed=self._env_conf.seed)

    def _iterate(self, designer):
        policy = designer(self._data)
        traj = self._collect_trajectory(policy)
        return Datum(policy, traj)

    def collect_trace(self, ttype, num_iterations):
        assert ttype in self._designers, f"Unknown optimizer type {ttype}"

        designer = self._designers[ttype]
        trace = []
        for i_iter in range(num_iterations):
            datum = self._iterate(designer)
            self._data.append(datum)

            if self._datum_best is None or datum.trajectory.rreturn > self._datum_best.trajectory.rreturn:
                self._datum_best = datum

            if i_iter % 1 == 0:
                print(
                    f"ITER: i_iter = {i_iter} ret = {datum.trajectory.rreturn:.2f} ret_best = {self._datum_best.trajectory.rreturn:.2f} n_data = {len(self._data)}"
                )
            trace.append(self._datum_best.trajectory.rreturn)
            if self._cb_trace:
                self._cb_trace(self._datum_best)

        return trace
