import cma
import numpy as np

from bbo.linear_policy import LinearPolicy
from bbo.trajectories import collect_trajectory


def optimize(env_conf, num_iterations):
    policy = LinearPolicy(env_conf)

    es = cma.CMAEvolutionStrategy([0] * policy.num_params(), sigma0=0.1)

    phi_best = -1e99
    f_eval = 0
    for i_iter in range(num_iterations):
        phis = []
        ws = es.ask()
        for w in ws:
            policy.set_params(w)
            phis.append(collect_trajectory(env_conf, policy, seed=env_conf.seed).rreturn)
        phis = np.array(phis)

        es.tell(ws, -phis)
        phi_best = max(phi_best, phis.max())
        f_eval += len(ws)
        print(f"EVAL: i_iter = {i_iter} f_eval = {f_eval} phi_best = {phi_best:.2f}")


if __name__ == "__main__":
    from bbo.env_conf import EnvConf

    num_iterations = 10000

    seed = None
    # env_conf = EnvConf("MountainCarContinuous-v0", seed=seed, max_steps=1000, solved=9999, show_frames=100, num_opt_0=1000, k_action=10)
    # env_conf = EnvConf("LunarLander-v2", seed=seed, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30, num_opt_0=3000)
    env_conf = EnvConf("Ant-v4", seed=seed, max_steps=1000, solved=999, show_frames=30, num_opt_0=3000)

    optimize(env_conf, num_iterations)
