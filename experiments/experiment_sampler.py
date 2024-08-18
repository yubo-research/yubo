import os
import time

import torch

from analysis.data_io import data_is_done, data_writer
from common.collector import Collector
from common.seed_all import seed_all
from experiments.experiment_util import ensure_parent

# from optimizer.arm_best_est import ArmBestEst
from optimizer.arm_best_obs import ArmBestObs
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


def sample_1(env_conf, opt_name, num_rounds, num_arms, num_denoise):
    seed_all(env_conf.problem_seed + 27)

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    default_device = torch.empty(size=(1,)).device
    print("DEFAULT_DEVICE:", default_device)

    policy = default_policy(env_conf)

    # arm_selector = ArmBestEst()
    arm_selector = ArmBestObs()

    collector_log = Collector()
    opt = Optimizer(
        collector_log,
        env_conf=env_conf,
        policy=policy,
        num_arms=num_arms,
        num_denoise=num_denoise,
        arm_selector=arm_selector,
    )

    collector_trace = Collector()
    for i_iter, te in enumerate(opt.collect_trace(designer_name=opt_name, num_iterations=num_rounds)):
        collector_trace(
            f"TRACE: name = {env_conf.env_name} opt_name = {opt_name} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}"
        )
    collector_trace("DONE")

    return collector_log, collector_trace


def post_process(collector_log, collector_trace, trace_fn):
    ensure_parent(trace_fn)

    def _w(f, line):
        f.write(line + "\n")
        print(line)

    with data_writer(trace_fn) as f:
        for line in collector_log:
            _w(f, line)
        for line in collector_trace:
            _w(f, line)


def extract_trace_fns(all_args):
    trace_fns = []
    for d_args in all_args:
        trace_fns.append(d_args.pop("trace_fn"))
    return trace_fns


def scan_local(all_args):
    t_0 = time.time()
    trace_fns = extract_trace_fns(all_args)
    for trace_fn, d_args in zip(trace_fns, all_args):
        collector_log, collector_trace = sample_1(**d_args)
        post_process(collector_log, collector_trace, trace_fn)
    t_f = time.time()
    print(f"TIME_LOCAL: {t_f - t_0:.2f}")


def mk_replicates(d_args):
    assert "noise" not in d_args, "NYI"

    out_dir = (
        f"{d_args['exp_dir']}/env={d_args['env_tag']}--opt_name={d_args['opt_name']}--num_arms={d_args['num_arms']}"
        f"--num_rounds={d_args['num_rounds']}--num_reps={d_args['num_reps']}--num_denoise={d_args.get('num_denoise',None)}"
    )

    os.makedirs(out_dir, exist_ok=True)
    print(f"PARAMS: {d_args}")
    all_d_args = []
    for i_rep in range(int(d_args["num_reps"])):
        trace_fn = f"{out_dir}/{i_rep:05d}"
        if data_is_done(trace_fn):
            print(f"Skipping i_rep = {i_rep}. Already done.")
            continue
        else:
            problem_seed = 18 + i_rep
            env_conf = get_env_conf(d_args["env_tag"], problem_seed=problem_seed, noise_level=d_args.get("noise", None), noise_seed_0=10 * problem_seed)
            num_denoise = d_args.get("num_denoise", None)
            if num_denoise == "None":
                num_denoise = None
            if num_denoise is not None:
                num_denoise = int(num_denoise)
            all_d_args.append(
                dict(
                    trace_fn=trace_fn,
                    env_conf=env_conf,
                    opt_name=d_args["opt_name"],
                    num_rounds=int(d_args["num_rounds"]),
                    num_arms=int(d_args["num_arms"]),
                    num_denoise=num_denoise,
                )
            )
    return all_d_args


def sampler(d_args, distributor_fn):
    all_d_args = mk_replicates(d_args)
    distributor_fn(all_d_args)


def _prep_args_1(results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, noise=None, num_denoise=None):
    # TODO: noise subdir?
    assert noise is None, "NYI"

    exp_dir = f"{results_dir}/{exp_dir}"

    if noise is None:
        noise = ""
    else:
        noise = f"--noise={noise}"
        assert False, ("NYI", noise)

    # python experiments/experiment_reliable.py num_rounds=30 num_arms=5 env_tag=tlunar opt_name=gibbon num_reps=1 exp_dir=y_test num_denoise=100
    # return f"python experiments/experiment.py env_tag={problem} opt_name={opt} num_arms={num_arms} num_reps={num_replications} num_rounds={num_rounds} {num_obs} {num_denoise} {noise} exp_dir={exp_dir} > {logs_dir}/{opt} 2>&1"
    # return f"modal run experiments/experiment.py --env-tag={problem} --opt-name={opt} --num-arms={num_arms} --num-reps={num_replications} --num-rounds={num_rounds} {num_obs} {num_denoise} {noise} --exp-dir={exp_dir}"
    return dict(
        exp_dir=exp_dir,
        env_tag=problem,
        opt_name=opt,
        num_arms=num_arms,
        num_reps=num_replications,
        num_rounds=num_rounds,
        num_denoise=num_denoise,
    )


def prep_d_args(results_dir, exp_dir, funcs, dims, num_arms, num_replications, opts, noises, num_rounds=3, func_category="f", num_denoise=None):
    d_argss = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                for noise in noises:
                    problem = f"{func_category}:{func}-{dim}d"
                    d_argss.append(_prep_args_1(results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, noise, num_denoise=num_denoise))
    return d_argss
