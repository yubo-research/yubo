from rl_gym.acq_bt import AcqBT
from rl_gym.acq_iopt import AcqIOpt
from rl_gym.acq_min_dist import AcqMinDist
from rl_gym.acq_var import AcqVar
from rl_gym.datum import Datum
from rl_gym.policy_designer_bt import PolicyDesignerBT
from rl_gym.trajectories import collect_trajectory


class Optimizer:
    def __init__(self, env_conf, policy):
        self._env_conf = env_conf

        traj = self._collect_trajectory(policy)
        self._datum_best = Datum(policy, traj)
        self._data = []

    def _collect_trajectory(self, policy):
        return collect_trajectory(self._env_conf, policy, seed=self._env_conf.seed)

    def _iterate(self, acq_fn, data_opt):
        pd = PolicyDesignerBT(acq_fn, data_opt)

        times = pd.design()
        self._times_trace.append(times.mean() / 1e3)
        policy = pd.get_policy()
        traj = self._collect_trajectory(policy)
        return Datum(policy, traj)

    def collect_trace(self, ttype, num_iterations, num_init):
        assert ttype in [
            "random",
            "sobol",
            "iopt",
            "variance",
            "maximin",
            "maximin-toroidal",
            "dumb",
        ]

        trace = []
        self._times_trace = []
        max_trajs = 99999

        for _ in range(num_iterations):
            i_iter = len(trace)
            self._data = self._data[-max_trajs:]
            data_opt = self._data + [self._datum_best]
            if ttype == "iopt":
                acq_fn = AcqBT(AcqIOpt, data_opt)
            elif ttype == "variance":
                acq_fn = AcqBT(AcqVar, data_opt)
            elif ttype == "maximin":
                acq_fn = AcqBT(lambda m: AcqMinDist(m, toroidal=False), data_opt)
            elif ttype == "maximin-toroidal":
                acq_fn = AcqBT(lambda m: AcqMinDist(m, toroidal=True), data_opt)
            elif ttype in ["random", "sobol", "dumb"]:
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
