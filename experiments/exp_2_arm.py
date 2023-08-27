if __name__ == "__main__":
    import sys
    import time
    import warnings
    warnings.simplefilter("ignore")
    from common.seed_all import seed_all
    from experiments.exp_1_arm import sample
    from problems.env_conf import get_env_conf
    from tqdm import trange
    
    env_tag = sys.argv[1]
    ttype = sys.argv[2]
    num_arms = int(sys.argv[3])
    num_samples = int(sys.argv[4])  # was 30
    iterations = int(sys.argv[5])
    foldername = sys.argv[6]

    seed_all(17)
    num_iterations = iterations  # TEST 3
    for i_sample in trange(num_samples):
        t0 = time.time()
        seed = 13547 + i_sample
        env_conf = get_env_conf(env_tag, seed)
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_iterations, num_arms=num_arms, folder=foldername)
        # print(f"TIME_SAMPLE: {time.time() - t0:.2f}")
