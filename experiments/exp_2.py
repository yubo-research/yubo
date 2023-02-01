from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy


def sample(env_conf, ttype, tag, num_iterations):
    policy = default_policy(env_conf)

    i_iter = 0

    def cb_trace(datum):
        nonlocal i_iter
        print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {datum.trajectory.rreturn:.3f}")
        i_iter += 1

    opt = Optimizer(env_conf, policy, cb_trace)
    opt.collect_trace(ttype=ttype, num_iterations=num_iterations)


if __name__ == "__main__":
    import sys
    import time

    from problems.env_conf import get_env_conf

    env_tag = sys.argv[1]
    ttype = sys.argv[2]

    num_iterations = 1000
    for i_sample in range(10):
        t0 = time.time()
        seed = 1234 + 17 + i_sample
        env_conf = get_env_conf(env_tag, seed)
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_iterations)
        print(f"TIME_SAMPLE: {time.time() - t0:.2f}")
