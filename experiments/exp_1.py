import numpy as np


def sample(env_conf, ttype, tag, num_iterations):
    policy = LinearPolicy(env_conf)
    opt = Optimizer(env_conf, policy)
    for i_iter, rreturn in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations, num_init=num_iterations)):
        print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {rreturn:.3f}")


if __name__ == "__main__":
    import sys

    from bbo.env_conf import EnvConf
    from bbo.linear_policy import LinearPolicy
    from bbo.optimizer import Optimizer

    env_tag = sys.argv[1]
    ttype = sys.argv[2]
    num_iterations = 30

    import time
    t0 = time.time()
    for i_sample in range(1):
        seed = 17 + i_sample
        np.random.seed(seed)  # cma in PolicyDesigner needs this
        if env_tag == "mcc":
            env_conf = EnvConf("MountainCarContinuous-v0", seed=seed, max_steps=1000, solved=9999, show_frames=100, num_opt_0=100)
        elif env_tag == "lunar":
            env_conf = EnvConf("LunarLander-v2", seed=seed, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30, num_opt_0=100)
        elif env_tag == "ant":
            env_conf = EnvConf("Ant-v4", seed=seed, max_steps=1000, solved=999, show_frames=30, num_opt_0=100)
        else:
            assert False, env_tag
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=30)
    print (time.time() - t0)
    
