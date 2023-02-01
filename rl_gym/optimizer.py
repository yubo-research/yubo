from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.analytic import LogExpectedImprovement

from bo.acq_bt import AcqBT
from bo.acq_idopt import AcqIDOpt
from bo.acq_min_dist import AcqMinDist
from bo.acq_var import AcqVar
from rl_gym.datum import Datum
from rl_gym.policy_designer_bt import PolicyDesignerBT
from rl_gym.sobol_designer import SobolDesigner
from rl_gym.trajectories import collect_trajectory


class Optimizer:
    def __init__(self, env_conf, policy):
        self._env_conf = env_conf

        self._datum_best = Datum(policy, self._collect_trajectory(policy))
        self._data = []
        self._sobol = None  # SobolDesigner is stateful

    def _collect_trajectory(self, policy):
        return collect_trajectory(self._env_conf, policy, seed=self._env_conf.seed)

    def _iterate(self, acq_fn, data_opt):
        policy = PolicyDesignerBT(acq_fn, data_opt).design()
        traj = self._collect_trajectory(policy)
        return Datum(policy, traj)

    def collect_trace(self, ttype, num_iterations, num_init):
        assert ttype in [
            "random",
            "sobol",
            "idopt-ex",
            "variance",
            "maximin",
            "maximin-toroidal",
            "rs",
            "ucb",
            "ei",
            "idopt",
        ], f"Unknown optimizer type {ttype}"

        trace = []
        max_trajs = 99999

        for _ in range(num_iterations):
            i_iter = len(trace)
            self._data = self._data[-max_trajs:]
            data_opt = self._data + [self._datum_best]
            if ttype == "idopt-ex":
                acq_fn = AcqBT(AcqIDOpt, data_opt, acq_kwargs={"bounds": None})
            elif ttype == "variance":
                acq_fn = AcqBT(AcqVar, data_opt)
            elif ttype == "maximin":
                acq_fn = AcqBT(lambda m: AcqMinDist(m, toroidal=False), data_opt)
            elif ttype == "maximin-toroidal":
                acq_fn = AcqBT(lambda m: AcqMinDist(m, toroidal=True), data_opt)
            elif ttype == "idopt":
                acq_fn = AcqBT(AcqIDOpt, data_opt, acq_kwargs={"X_max": None, "bounds": None})
            elif ttype == "ucb":
                acq_fn = AcqBT(UpperConfidenceBound, data_opt, acq_kwargs={"beta": 1})
            elif ttype == "ei":
                acq_fn = AcqBT(LogExpectedImprovement, data_opt, acq_kwargs={"best_f": None})
            elif ttype == "sobol":
                if self._sobol is None:
                    print("CREATED_SOBOL")
                    self._sobol = SobolDesigner(self._datum_best.policy.num_params())
                acq_fn = self._sobol
            elif ttype in ["random", "rs"]:
                acq_fn = ttype
            else:
                assert False

            datum = self._iterate(acq_fn, data_opt)
            if datum.trajectory.rreturn > self._datum_best.trajectory.rreturn:
                self._data.append(self._datum_best)
                self._datum_best = datum
                # print (f"BEST: ret_best = {self._datum_best.trajectory.rreturn:.2f}")
            else:
                self._data.append(datum)

            if i_iter % 1 == 0:
                print(
                    f"ITER: i_iter = {i_iter} ret = {datum.trajectory.rreturn:.2f} ret_best = {self._datum_best.trajectory.rreturn:.2f} n_data = {len(self._data)}"
                )

            trace.append(self._datum_best.trajectory.rreturn)
        return trace
