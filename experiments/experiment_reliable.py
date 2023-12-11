import os

from analysis.data_io import data_is_done, data_writer
from optimizer.optimizer import Optimizer, arm_best_acqf, arm_best_obs
from problems.env_conf import default_policy


def sample(out_fn, env_conf, opt_name, num_rounds, num_arms, num_denoise):
    policy = default_policy(env_conf)
    if num_denoise is not None:
        arm_selector = arm_best_acqf
        print("BEST_ACQF")
    else:
        arm_selector = arm_best_obs
        print("BEST_OBS")
    opt = Optimizer(env_conf, policy, num_arms=num_arms, arm_selector=arm_selector)

    def _write(f, line):
        print(line)
        f.write(line + "\n")

    with data_writer(out_fn) as f:
        for i_iter, te in enumerate(opt.collect_trace(ttype=opt_name, num_iterations=num_rounds, num_denoise=num_denoise)):
            _write(
                f, f"TRACE: name = {env_conf.env_name} opt_name = {opt_name} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}"
            )
        f.write("DONE\n")


def parse_kv(argv):
    d = {}
    for a in argv:
        k, v = a.split("=")
        d[k] = v
    return d


if __name__ == "__main__":
    import sys
    import time

    from common.seed_all import seed_all
    from problems.env_conf import get_env_conf

    d_args = parse_kv(sys.argv[1:])
    reqd_keys = ["exp_dir", "env_tag", "opt_name", "num_arms", "num_reps", "num_rounds"]
    for k in reqd_keys:
        assert k in d_args, f"Missing {k} in {list(d_args.keys())}. Required: {reqd_keys}"

    out_dir = f"{d_args['exp_dir']}/{d_args['env_tag']}/{d_args['opt_name']}"
    # TODO: subdirs for all params? Maybe just a key?
    os.makedirs(out_dir, exist_ok=True)
    print(f"PARAMS: {d_args}")
    for i_rep in range(int(d_args["num_reps"])):
        out_fn = f"{out_dir}/{i_rep:05d}"
        if data_is_done(out_fn):
            print(f"Skipping i_rep = {i_rep}. Already done.")
            continue
        else:
            t0 = time.time()
            seed_all(17 + i_rep)
            problem_seed = 18 + i_rep
            env_conf = get_env_conf(d_args["env_tag"], problem_seed=problem_seed, noise_level=d_args.get("noise", None), noise_seed_0=10 * problem_seed)
            num_denoise = d_args.get("num_denoise", None)
            if num_denoise is not None:
                num_denoise = int(num_denoise)
            sample(
                out_fn,
                env_conf,
                d_args["opt_name"],
                num_rounds=int(d_args["num_rounds"]),
                num_arms=int(d_args["num_arms"]),
                num_denoise=num_denoise,
            )
            print(f"TIME_REPLICATE: {time.time() - t0:.2f}")
