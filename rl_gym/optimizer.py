from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import ExpectedImprovement

from bo.acq_idopt import AcqIDOpt
from bo.acq_min_dist import AcqMinDist
from bo.acq_var import AcqVar
from rl_gym.ax_designer import AxDesigner
from rl_gym.bt_designer import BTDesigner
from rl_gym.datum import Datum
from rl_gym.random_designer import RandomDesigner
from rl_gym.sobol_designer import SobolDesigner
from rl_gym.trajectories import collect_trajectory


class Optimizer:
    def __init__(self, env_conf, policy):
        self._env_conf = env_conf
        self._data = []  # Datum(policy, self._collect_trajectory(policy))]
        self._datum_best = None  # self._data[0]
        self._designers = {
            "variance": BTDesigner(policy, AcqVar),
            "minimax": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=False)),
            "minimax-toroidal": BTDesigner(policy, lambda m: AcqMinDist(m, toroidal=True)),
            "iopt": BTDesigner(policy, AcqIDOpt, acq_kwargs={"bounds": None}),
            "idopt": BTDesigner(policy, AcqIDOpt, acq_kwargs={"X_max": None, "bounds": None}),
            "ucb": BTDesigner(policy, UpperConfidenceBound, acq_kwargs={"beta": 1}),
            "ei": BTDesigner(policy, ExpectedImprovement, acq_kwargs={"best_f": None}),
            "random": RandomDesigner(policy),
            "sobol": SobolDesigner(policy),
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

        return trace
