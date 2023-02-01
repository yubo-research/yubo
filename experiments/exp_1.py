import numpy as np

from rl_gym.env_conf import default_policy
from rl_gym.optimizer import Optimizer


def sample(env_conf, ttype, tag, num_iterations):
    policy = default_policy(env_conf)
    opt = Optimizer(env_conf, policy)
    for i_iter, rreturn in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations)):
        print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {rreturn:.3f}")


if __name__ == "__main__":
    import sys
    import time

    from rl_gym.env_conf import get_env_conf

    env_tag = sys.argv[1]
    ttype = sys.argv[2]

    num_iterations = 100
    for i_sample in range(100):
        t0 = time.time()
        seed = 1234 + 17 + i_sample
        np.random.seed(seed)  # cma in PolicyDesigner needs this
        env_conf = get_env_conf(env_tag, seed)
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_iterations)
        print(f"TIME_SAMPLE: {time.time() - t0:.2f}")
