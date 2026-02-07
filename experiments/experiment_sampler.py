import hashlib
import os
import time
from dataclasses import asdict, dataclass
from typing import NamedTuple, Optional

import torch

from analysis.data_io import (
    TraceRecord,
    data_is_done,
    data_writer,
    write_config,
    write_trace_jsonl,
)
from common.collector import Collector
from common.seed_all import seed_all
from experiments.experiment_util import ensure_parent
from optimizer.optimizer import Optimizer
from problems.env_conf import default_policy, get_env_conf


class _SampleResult(NamedTuple):
    collector_log: Collector
    collector_trace: Collector
    trace_records: list[TraceRecord]


@dataclass
class ExperimentConfig:
    exp_dir: str
    env_tag: str
    opt_name: str
    num_arms: int
    num_rounds: int
    num_reps: int
    num_denoise: Optional[int] = None
    num_denoise_passive: Optional[int] = None
    max_proposal_seconds: Optional[float] = None
    max_total_seconds: Optional[float] = None
    b_trace: bool = True

    def to_dir_name(self) -> str:
        config_str = (
            f"env={self.env_tag}--opt_name={self.opt_name}"
            f"--num_arms={self.num_arms}--num_rounds={self.num_rounds}"
            f"--num_reps={self.num_reps}--num_denoise={self.num_denoise}"
            f"--max_proposal_seconds={self.max_proposal_seconds}"
        )
        short_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{self.exp_dir}/{short_hash}"

    def to_dir_name_legacy(self) -> str:
        return (
            f"{self.exp_dir}/env={self.env_tag}--opt_name={self.opt_name}"
            f"--num_arms={self.num_arms}--num_rounds={self.num_rounds}"
            f"--num_reps={self.num_reps}--num_denoise={self.num_denoise}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        max_prop = d.get("max_proposal_seconds")
        if max_prop in (None, "None"):
            max_prop = None
        else:
            max_prop = float(max_prop)
        max_total = d.get("max_total_seconds")
        if max_total in (None, "None"):
            max_total = None
        else:
            max_total = float(max_total)

        return cls(
            exp_dir=d["exp_dir"],
            env_tag=d["env_tag"],
            opt_name=d["opt_name"],
            num_arms=int(d["num_arms"]),
            num_rounds=int(d["num_rounds"]),
            num_reps=int(d["num_reps"]),
            num_denoise=None if d.get("num_denoise") in (None, "None") else int(d["num_denoise"]),
            num_denoise_passive=None if d.get("num_denoise_passive") in (None, "None") else int(d["num_denoise_passive"]),
            max_proposal_seconds=max_prop,
            max_total_seconds=max_total,
            b_trace=true_false(d.get("b_trace", True)),
        )


@dataclass
class RunConfig:
    env_conf: object
    opt_name: str
    num_rounds: int
    num_arms: int
    num_denoise: Optional[int]
    num_denoise_passive: Optional[int]
    max_proposal_seconds: Optional[float]
    b_trace: bool
    trace_fn: str
    deadline: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def sample_1(run_config: RunConfig):
    import numpy as np

    env_conf = run_config.env_conf
    opt_name = run_config.opt_name
    num_rounds = run_config.num_rounds
    num_arms = run_config.num_arms
    num_denoise = run_config.num_denoise
    num_denoise_passive = run_config.num_denoise_passive
    max_proposal_seconds = run_config.max_proposal_seconds
    b_trace = run_config.b_trace
    deadline = run_config.deadline

    if max_proposal_seconds is None:
        max_proposal_seconds = np.inf

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
        num_denoise_passive=num_denoise_passive,
    )

    trace_records = []
    collector_trace = Collector()
    for i_iter, te in enumerate(
        opt.collect_trace(
            designer_name=opt_name,
            max_iterations=num_rounds,
            max_proposal_seconds=max_proposal_seconds,
            deadline=deadline,
        )
    ):
        record = TraceRecord(
            i_iter=i_iter,
            dt_prop=te.dt_prop,
            dt_eval=te.dt_eval,
            rreturn=te.rreturn,
            env_name=env_conf.env_name,
            opt_name=opt_name,
        )
        trace_records.append(record)
        if b_trace:
            collector_trace(
                f"TRACE: name = {env_conf.env_name} opt_name = {opt_name} i_iter = {i_iter} dt_prop = {te.dt_prop:.3e} dt_eval = {te.dt_eval:.3e} return = {te.rreturn:.3e}"
            )
    collector_trace("DONE")

    return _SampleResult(
        collector_log=collector_log,
        collector_trace=collector_trace,
        trace_records=trace_records,
    )


def post_process_stdout(
    collector_log,
    collector_trace,
    trace_records=None,
):
    for line in collector_log:
        print(line)
    for line in collector_trace:
        print(line)


def post_process(collector_log, collector_trace, trace_fn, trace_records=None):
    ensure_parent(trace_fn)

    if trace_records is not None:
        jsonl_fn = trace_fn + ".jsonl"
        write_trace_jsonl(jsonl_fn, trace_records)

    def _w(f, line):
        f.write(line + "\n")

    with data_writer(trace_fn) as f:
        for line in collector_log:
            _w(f, line)
        for line in collector_trace:
            _w(f, line)


def extract_trace_fns(run_configs: list[RunConfig]) -> list[str]:
    return [rc.trace_fn for rc in run_configs]


def scan_local(run_configs: list[RunConfig], max_total_seconds: Optional[float] = None):
    t_0 = time.time()
    deadline = None if max_total_seconds is None else t_0 + max_total_seconds
    for run_config in run_configs:
        if deadline is not None and time.time() >= deadline:
            print(f"TIME_LIMIT: Stopping after {time.time() - t_0:.2f}s (max_total_seconds={max_total_seconds})")
            break
        run_config.deadline = deadline
        collector_log, collector_trace, trace_records = sample_1(run_config)
        post_process(collector_log, collector_trace, run_config.trace_fn, trace_records)
    t_f = time.time()
    print(f"TIME_LOCAL: {t_f - t_0:.2f}")


def true_false(string_bool):
    string_bool = str(string_bool).lower()
    if string_bool in ["false", "f"]:
        return False
    if string_bool in ["true", "t"]:
        return True
    assert False, string_bool


def mk_replicates(config: ExperimentConfig) -> list[RunConfig]:
    out_dir = config.to_dir_name()

    os.makedirs(out_dir, exist_ok=True)
    write_config(out_dir, config.to_dict())
    print(f"PARAMS: {config}")
    run_configs = []
    for i_rep in range(config.num_reps):
        trace_fn = f"{out_dir}/traces/{i_rep:05d}"
        jsonl_fn = trace_fn + ".jsonl"
        if data_is_done(trace_fn) or data_is_done(jsonl_fn):
            print(f"Skipping trace_fn = {trace_fn}. Already done.")
            continue
        else:
            problem_seed = 18 + i_rep
            env_conf = get_env_conf(
                config.env_tag,
                problem_seed=problem_seed,
                noise_level=None,
                noise_seed_0=10 * problem_seed,
            )
            run_configs.append(
                RunConfig(
                    trace_fn=trace_fn,
                    env_conf=env_conf,
                    opt_name=config.opt_name,
                    num_rounds=config.num_rounds,
                    num_arms=config.num_arms,
                    num_denoise=config.num_denoise,
                    num_denoise_passive=config.num_denoise_passive,
                    max_proposal_seconds=config.max_proposal_seconds,
                    b_trace=config.b_trace,
                )
            )
    return run_configs


def sampler(config: ExperimentConfig, distributor_fn):
    run_configs = mk_replicates(config)
    if distributor_fn == scan_local:
        distributor_fn(run_configs, max_total_seconds=config.max_total_seconds)
    else:
        distributor_fn(run_configs)


def prep_args_1(
    results_dir,
    exp_dir,
    problem,
    opt,
    num_arms,
    num_replications,
    num_rounds,
    noise=None,
    num_denoise=None,
    num_denoise_passive=None,
) -> ExperimentConfig:
    assert noise is None, "NYI"

    full_exp_dir = f"{results_dir}/{exp_dir}"

    return ExperimentConfig(
        exp_dir=full_exp_dir,
        env_tag=problem,
        opt_name=opt,
        num_arms=num_arms,
        num_reps=num_replications,
        num_rounds=num_rounds,
        num_denoise=num_denoise,
        num_denoise_passive=num_denoise_passive,
    )


def prep_d_args(
    results_dir,
    exp_dir,
    funcs,
    dims,
    num_arms,
    num_replications,
    opts,
    noises,
    num_rounds=3,
    func_category="f",
    num_denoise=None,
    num_denoise_passive=None,
) -> list[ExperimentConfig]:
    configs = []
    for dim in dims:
        for func in funcs:
            for opt in opts:
                for noise in noises:
                    problem = f"{func_category}:{func}-{dim}d"
                    configs.append(
                        prep_args_1(
                            results_dir,
                            exp_dir,
                            problem,
                            opt,
                            num_arms,
                            num_replications,
                            num_rounds,
                            noise,
                            num_denoise=num_denoise,
                            num_denoise_passive=num_denoise_passive,
                        )
                    )
    return configs
