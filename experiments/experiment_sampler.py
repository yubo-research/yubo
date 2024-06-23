import os
import time

from analysis.data_io import data_is_done, data_writer
from common.collector import Collector
from common.seed_all import seed_all
from experiments.experiment_modal import modal_app, modal_image
from optimizer.arm_best_est import ArmBestEst
from optimizer.arm_best_obs import ArmBestObs
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


@modal_app.function(image=modal_image)
def _sample_1_modal(d_args):
    return _sample_1(**d_args)


def _sample_1(env_conf, opt_name, num_rounds, num_arms, num_obs, num_denoise):
    seed_all(env_conf.problem_seed + 27)
    policy = default_policy(env_conf)

    if num_denoise is not None and num_denoise > 0:
        arm_selector = ArmBestEst()
    else:
        arm_selector = ArmBestObs()

    collector_log = Collector()
    opt = Optimizer(
        collector_log,
        env_conf=env_conf,
        policy=policy,
        num_arms=num_arms,
        num_denoise=num_denoise,
        num_obs=num_obs,
        arm_selector=arm_selector,
    )

    collector_trace = Collector()
    for i_iter, te in enumerate(opt.collect_trace(designer_name=opt_name, num_iterations=num_rounds)):
        collector_trace(
            f"TRACE: name = {env_conf.env_name} opt_name = {opt_name} i_iter = {i_iter} dt = {te.time_iteration_seconds:.3e} return = {te.rreturn:.3e}"
        )
    collector_trace("DONE")

    return collector_log, collector_trace


def _post_process(collector_log, collector_trace, trace_fn):
    def _w(f, l):
        f.write(l + "\n")
        print(l)

    with data_writer(trace_fn) as f:
        for line in collector_log:
            _w(f, line)
        for line in collector_trace:
            _w(f, line)


def _extract_trace_fns(all_args):
    trace_fns = []
    for d_args in all_args:
        trace_fns.append(d_args.pop("trace_fn"))
    return trace_fns


def _scan_local(all_args):
    t_0 = time.time()
    trace_fns = _extract_trace_fns(all_args)
    for trace_fn, d_args in zip(trace_fns, all_args):
        collector_log, collector_trace = _sample_1(**d_args)
        _post_process(collector_log, collector_trace, trace_fn)
    t_f = time.time()
    print(f"TIME_LOCAL: {t_f - t_0:.2f}")


def _dist_modal(all_args):
    t_0 = time.time()
    trace_fns = _extract_trace_fns(all_args)
    for trace_fn, (collector_log, collector_trace) in zip(trace_fns, _sample_1_modal.map(all_args)):
        _post_process(collector_log, collector_trace, trace_fn)
    t_f = time.time()
    print(f"TIME_MODAL: {t_f - t_0:.2f}")


def parse_kv(argv):
    d = {}
    for a in argv:
        k, v = a.split("=")
        d[k] = v
    return d


def sampler(d_args, b_modal=False):
    out_dir = f"{d_args['exp_dir']}/{d_args['env_tag']}/{d_args['opt_name']}"
    # TODO: subdirs for all params? Maybe just a key?
    os.makedirs(out_dir, exist_ok=True)
    print(f"PARAMS: {d_args}")
    all_args = []
    for i_rep in range(int(d_args["num_reps"])):
        trace_fn = f"{out_dir}/{i_rep:05d}"
        if data_is_done(trace_fn):
            print(f"Skipping i_rep = {i_rep}. Already done.")
            continue
        else:
            problem_seed = 18 + i_rep
            env_conf = get_env_conf(d_args["env_tag"], problem_seed=problem_seed, noise_level=d_args.get("noise", None), noise_seed_0=10 * problem_seed)
            num_denoise = d_args.get("num_denoise", None)
            if num_denoise is not None:
                num_denoise = int(num_denoise)
            all_args.append(
                dict(
                    trace_fn=trace_fn,
                    env_conf=env_conf,
                    opt_name=d_args["opt_name"],
                    num_rounds=int(d_args["num_rounds"]),
                    num_arms=int(d_args["num_arms"]),
                    num_obs=int(d_args.get("num_obs", 1)),
                    num_denoise=num_denoise,
                )
            )

    if b_modal:
        _dist_modal(all_args)
    else:
        _scan_local(all_args)
