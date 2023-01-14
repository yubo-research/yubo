import cma
import numpy as np

from rl_gym.linear_policy import LinearPolicy
from rl_gym.trajectories import collect_trajectory


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
            phi = collect_trajectory(env_conf, policy, seed=env_conf.seed).rreturn
            phis.append(phi)
            w = np.array(w)
            # print("POP:", phi, w.mean(), w.std())
        phis = np.array(phis)

        es.tell(ws, -phis)
        phi_best = max(phi_best, phis.max())
        f_eval += len(ws)
        print(f"EVAL: i_iter = {i_iter} f_eval = {f_eval} phi_best = {phi_best:.2f}")


if __name__ == "__main__":
    from rl_gym.env_conf import get_env_conf

    env_tag = "lunar"
    num_iterations = 10000

    seed = 19
    env_conf = get_env_conf(env_tag, seed)

    optimize(env_conf, num_iterations)
