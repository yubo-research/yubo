from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy


def sample(env_conf, ttype, tag, num_iterations, num_arms, num_denoise):
    policy = default_policy(env_conf)
    opt = Optimizer(env_conf, policy, num_arms=num_arms)
    for i_iter, te in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations, num_denoise=num_denoise)):
        print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}")


if __name__ == "__main__":
    import sys
    import time

    from common.seed_all import seed_all
    from problems.env_conf import get_env_conf

    assert len(sys.argv) in [6, 7, 8], f"{sys.argv[0]} env_tag ttype num_arms num_replications num_rounds [num_denoise [noise]]"
    env_tag = sys.argv[1]
    ttype = sys.argv[2]
    num_arms = int(sys.argv[3])
    num_replications = int(sys.argv[4])  # was 30
    num_rounds = int(sys.argv[5])  # was 3
    if len(sys.argv) > 6:
        num_denoise = int(sys.argv[6])  # was None
    else:
        num_denoise = None

    if len(sys.argv) > 7:
        noise = float(sys.argv[7])  # was None
    else:
        noise = None

    print(f"EXPERIMENT: env_tag = {env_tag} ttype = {ttype} num_arms = {num_arms} num_replications = {num_replications} num_rounds = {num_rounds}")
    for i_sample in range(num_replications):
        t0 = time.time()
        seed_all(17 + i_sample)
        seed = 18 + i_sample
        env_conf = get_env_conf(env_tag, seed, noise=noise)
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_rounds, num_arms=num_arms, num_denoise=num_denoise)
        print(f"TIME_REPLICATE: {time.time() - t0:.2f}")
