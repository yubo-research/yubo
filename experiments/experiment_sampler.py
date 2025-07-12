import os
import time

import torch

from analysis.data_io import data_is_done, data_writer
from common.collector import Collector
from common.seed_all import seed_all
from experiments.experiment_util import ensure_parent
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


def sample_1(env_conf, opt_name, num_rounds, num_arms, num_denoise, max_proposal_seconds, b_trace=True):
    # print("PROBLEM_SEED:", env_conf.problem_seed)

    seed_all(env_conf.problem_seed + 27)

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    default_device = torch.empty(size=(1,)).device
    print("DEFAULT_DEVICE:", default_device)

    policy = default_policy(env_conf)

    collector_log = Collector()
    opt = Optimizer(
        collector_log,
        env_conf=env_conf,
        policy=policy,
        num_arms=num_arms,
        num_denoise_measurement=num_denoise,
    )

    collector_trace = Collector()
    for i_iter, te in enumerate(opt.collect_trace(designer_name=opt_name, max_iterations=num_rounds, max_proposal_seconds=max_proposal_seconds)):
        if b_trace:
            collector_trace(
                f"TRACE: name = {env_conf.env_name} opt_name = {opt_name} i_iter = {i_iter} dt_prop = {te.dt_prop:.3e} dt_eval = {te.dt_eval:.3e} return = {te.rreturn:.3e}"
            )

        pass
    collector_trace("DONE")

    return collector_log, collector_trace


def post_process_stdout(
    collector_log,
    collector_trace,
):
    for line in collector_log:
        print(line)
    for line in collector_trace:
        print(line)


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


def true_false(string_bool):
    string_bool = str(string_bool).lower()
    if string_bool in ["false", "f"]:
        return False
    if string_bool in ["true", "t"]:
        return True
    assert False, string_bool


def mk_replicates(d_args):
    assert "noise" not in d_args, "NYI"

    out_dir = (
        f"{d_args['exp_dir']}/env={d_args['env_tag']}--opt_name={d_args['opt_name']}--num_arms={d_args['num_arms']}"
        f"--num_rounds={d_args['num_rounds']}--max_proposal_seconds={d_args['max_proposal_seconds']}--num_reps={d_args['num_reps']}--num_denoise={d_args.get('num_denoise', None)}"
    )

    os.makedirs(out_dir, exist_ok=True)
    print(f"PARAMS: {d_args}")
    all_d_args = []
    for i_rep in range(int(d_args["num_reps"])):
        trace_fn = f"{out_dir}/{i_rep:05d}"
        if data_is_done(trace_fn):
            print(f"Skipping trace_fn = {trace_fn}. Already done.")
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
                    max_proposal_seconds=int(d_args["max_proposal_seconds"]),
                    num_arms=int(d_args["num_arms"]),
                    num_denoise=num_denoise,
                    b_trace=true_false(d_args.get("b_trace", True)),
                )
            )
    return all_d_args


def sampler(d_args, distributor_fn):
    all_d_args = mk_replicates(d_args)
    distributor_fn(all_d_args)


def prep_args_1(results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, max_proposal_seconds, noise=None, num_denoise=None):
    # TODO: noise subdir?
    assert noise is None, "NYI"

    exp_dir = f"{results_dir}/{exp_dir}"

    if noise is None:
        noise = ""
    else:
        noise = f"--noise={noise}"
        assert False, ("NYI", noise)

    return dict(
        exp_dir=exp_dir,
        env_tag=problem,
        opt_name=opt,
        num_arms=num_arms,
        num_reps=num_replications,
        num_rounds=num_rounds,
        num_denoise=num_denoise,
        max_proposal_seconds=max_proposal_seconds,
    )


def prep_d_args(
    results_dir, exp_dir, funcs, dims, num_arms, num_replications, opts, noises, num_rounds, max_proposal_seconds, func_category="f", num_denoise=None
):
    d_argss = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                for noise in noises:
                    problem = f"{func_category}:{func}-{dim}d"
                    d_argss.append(
                        prep_args_1(
                            results_dir, exp_dir, problem, opt, num_arms, num_replications, num_rounds, max_proposal_seconds, noise, num_denoise=num_denoise
                        )
                    )
    return d_argss
