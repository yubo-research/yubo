if __name__ == "__main__":
    import sys
    import time

    from common.seed_all import seed_all
    from experiments.exp_1 import sample
    from problems.env_conf import get_env_conf

    assert len(sys.argv) in [6, 7], f"{sys.argv[0]} env_tag ttype num_arms num_replications num_rounds [noise]"
    env_tag = sys.argv[1]
    ttype = sys.argv[2]
    num_arms = int(sys.argv[3])
    num_replications = int(sys.argv[4])  # was 30
    num_rounds = int(sys.argv[5])  # was 3
    if len(sys.argv) > 6:
        noise = float(sys.argv[6])  # was 0
    else:
        noise = None

    print(f"EXPERIMENT: env_tag = {env_tag} ttype = {ttype} num_arms = {num_arms} num_replications = {num_replications} num_rounds = {num_rounds}")
    for i_sample in range(num_replications):
        t0 = time.time()
        seed_all(17 + i_sample)
        seed = 18 + i_sample
        env_conf = get_env_conf(env_tag, seed, noise=noise)
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_rounds, num_arms=num_arms)
        print(f"TIME_REPLICATE: {time.time() - t0:.2f}")
