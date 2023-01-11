def sample(env_conf, ttype, tag):
    policy = LinearPolicy(env_conf)
    opt = Optimizer(env_conf, policy)
    for i_iter, rreturn in enumerate(opt.collect_trace(ttype=ttype, num_iterations=10)):
        print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {rreturn:.3f}")


if __name__ == "__main__":
    import sys

    from bbo.env_conf import EnvConf
    from bbo.linear_policy import LinearPolicy
    from bbo.optimizer import Optimizer

    ttype = sys.argv[1]

    seed = None
    # env_conf = EnvConf("MountainCarContinuous-v0", seed=seed, max_steps=1000, solved=9999, show_frames=100)
    env_conf = EnvConf('LunarLander-v2', seed=seed, max_steps=500, kwargs={'continuous':True}, solved=999, show_frames=30)

    for i_sample in range(100):
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}")
