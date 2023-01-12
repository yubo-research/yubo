from rl_gym.datum import Datum
from rl_gym.distance import distance_actions, distance_actions_corr, distance_parameters
from rl_gym.min_distance_actions_fast import MinDistanceActionsFast
from rl_gym.min_distance_parameters import MinDistanceParameters
from rl_gym.policy_designer import PolicyDesigner
from rl_gym.trajectories import collect_trajectory


class Optimizer:
    def __init__(self, env_conf, policy):
        self._env_conf = env_conf

        traj = self._collect_trajectory(policy)
        self._datum_best = Datum(policy, traj)
        self._data = []

    def _collect_trajectory(self, policy):
        traj = collect_trajectory(self._env_conf, policy, seed=self._env_conf.seed)
        # w = np.array(policy.get_params()).flatten()
        # print ("CT:", traj.rreturn, w.mean(), w.std())
        return traj

    def _iterate(self, trust_distance_fn, acq_fn, data_opt, delta_tr, dumb):
        pd = PolicyDesigner(trust_distance_fn, acq_fn, data_opt, delta_tr)

        if dumb:
            pd.design_dumb(0.1)
        else:
            times = pd.design(self._env_conf.num_opt_0)
            self._times_trace.append(times.mean() / 1e3)
            print(f"MD: md = {pd.min_dist():.4f} tr = {pd.trust():.4f}")

        policy = pd.get_policy()
        traj = self._collect_trajectory(policy)
        return Datum(policy, traj)

    def collect_trace(self, ttype, num_iterations, num_init):
        assert ttype in ["actions_corr", "actions", "params", "dumb"]

        trace = []
        self._times_trace = []
        max_trajs = 99999

        if ttype == "actions_corr":
            trust_distance_fn = distance_actions_corr
        if ttype == "actions":
            trust_distance_fn = distance_actions
        else:
            trust_distance_fn = distance_parameters

        for _ in range(num_iterations):
            i_iter = len(trace)

            # p_explore = max(.1, 5 / (1. + i_iter))
            if i_iter < num_init:
                delta_tr = 1e9
            else:
                delta_tr = 0.1
            self._data = self._data[-max_trajs:]
            data_opt = self._data + [self._datum_best]
            if ttype == "actions_corr":
                acq_fn = MinDistanceActionsFast(data_opt, ttype="corr")
            elif ttype == "actions":
                acq_fn = MinDistanceActionsFast(data_opt)
            else:
                acq_fn = MinDistanceParameters(data_opt)
            datum = self._iterate(trust_distance_fn, acq_fn, data_opt, delta_tr, dumb=ttype == "dumb")
            if datum.trajectory.rreturn > self._datum_best.trajectory.rreturn:
                self._data.append(self._datum_best)
                self._datum_best = datum
                # print (f"BEST: ret_best = {self._datum_best.trajectory.rreturn:.2f}")
            else:
                self._data.append(datum)

            if i_iter % 1 == 0:
                print(
                    f"ITER: i_iter = {i_iter} ret = {datum.trajectory.rreturn:.2f} ret_best = {self._datum_best.trajectory.rreturn:.2f} delta_tr = {delta_tr:.2f} n_data = {len(self._data)}"
                )

            trace.append(self._datum_best.trajectory.rreturn)
            if self._datum_best.trajectory.rreturn > self._env_conf.solved:
                print(f"SOLVED! ret_best = {self._datum_best.trajectory.rreturn:.1f}")
                break
        return trace
