def sample(env_conf, ttype, tag, num_iterations):
    policy = LinearPolicy(env_conf)
    opt = Optimizer(env_conf, policy, env_conf.num_opt_0)
    for i_iter, rreturn in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations, num_init=num_iterations)):
        print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} return = {rreturn:.3f}")


if __name__ == "__main__":
    import sys

    from bbo.env_conf import EnvConf
    from bbo.linear_policy import LinearPolicy
    from bbo.optimizer import Optimizer

    ttype = sys.argv[1]
    num_iterations = 30
    
    seed = None
    env_conf = EnvConf("MountainCarContinuous-v0", seed=seed, max_steps=1000, solved=9999, show_frames=100, num_opt_0=1000)
    env_conf = EnvConf("LunarLander-v2", seed=seed, max_steps=500, kwargs={"continuous": True}, solved=999, show_frames=30, num_opt_0=3000)
    env_conf = EnvConf("Ant-v4", seed=seed, max_steps=1000, solved=999, show_frames=30, num_opt_0=3000)
    

    for i_sample in range(100):
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=30)

        
