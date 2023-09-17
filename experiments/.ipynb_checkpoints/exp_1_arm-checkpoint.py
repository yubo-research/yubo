from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy
import os
import warnings
warnings.simplefilter("ignore")
def sample(env_conf, ttype, tag, num_iterations, num_arms,folder):
    policy = default_policy(env_conf)
    opt = Optimizer(env_conf, policy, num_arms=num_arms)
    name = env_conf.env_name[2:].replace("-", "_")
    for i_iter, te in enumerate(opt.collect_trace(ttype=ttype, num_iterations=num_iterations)):
        # print(f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}")
        dir_path = f"results/{folder}/{name}/{num_iterations}iter/"
        filename = f"{ttype}_{num_arms}arm"
        path = os.path.join(dir_path, filename)
        # print(path)
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "a") as f:
            f.write(
                f"TRACE: name = {env_conf.env_name} ttype = {ttype} {tag} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}\n"
            )


if __name__ == "__main__":
    import sys
    import time
    from tqdm import trange
    from common.seed_all import seed_all
    from problems.env_conf import get_env_conf

    env_tag = sys.argv[1]
    ttype = sys.argv[2]
    num_arms = int(sys.argv[3])
    num_samples = int(sys.argv[4])  # was 30
    num_iterations = int(sys.argv[5])
    foldername = sys.argv[6]
    seed_all(17)
    num_iterations = 100
    for i_sample in trange(30):
        t0 = time.time()
        seed = 17 + i_sample
        env_conf = get_env_conf(env_tag, seed)
        sample(env_conf, ttype, tag=f"i_sample = {i_sample}", num_iterations=num_iterations, num_arms=num_arms)
        # print(f"TIME_SAMPLE: {time.time() - t0:.2f}")
        with open(f"results/{foldername}/{name}/{num_iterations}iter/{ttype}_{num_arms}arm", "a") as f:
            f.write(f"TIME_SAMPLE: {time.time() - t0:.2f}\n")
    f.close()
