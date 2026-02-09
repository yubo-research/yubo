import hashlib
import inspect
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
from optimizer.checkpointing import load_optimizer_checkpoint
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
    rollout_workers: Optional[int] = None
    run_workers: Optional[int] = None
    checkpoint_every: Optional[int] = None
    resume: bool = False
    b_trace: bool = True
    video_enable: bool = False
    video_dir: Optional[str] = None
    video_prefix: str = "bo"
    video_num_episodes: int = 10
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: Optional[int] = None

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
        return cls(
            exp_dir=d["exp_dir"],
            env_tag=d["env_tag"],
            opt_name=d["opt_name"],
            num_arms=int(d["num_arms"]),
            num_rounds=int(d["num_rounds"]),
            num_reps=int(d["num_reps"]),
            num_denoise=_parse_optional_int(d.get("num_denoise")),
            num_denoise_passive=_parse_optional_int(d.get("num_denoise_passive")),
            max_proposal_seconds=_parse_optional_float(d.get("max_proposal_seconds")),
            max_total_seconds=_parse_optional_float(d.get("max_total_seconds")),
            rollout_workers=_parse_optional_int(d.get("rollout_workers")),
            run_workers=_parse_optional_int(d.get("run_workers")),
            checkpoint_every=_parse_optional_int(d.get("checkpoint_every")),
            resume=true_false(d.get("resume", False)),
            b_trace=true_false(d.get("b_trace", True)),
            video_enable=true_false(d.get("video_enable", False)),
            video_dir=d.get("video_dir"),
            video_prefix=d.get("video_prefix", "bo"),
            video_num_episodes=int(d.get("video_num_episodes", 10)),
            video_num_video_episodes=int(d.get("video_num_video_episodes", 3)),
            video_episode_selection=str(d.get("video_episode_selection", "best")),
            video_seed_base=_parse_optional_int(d.get("video_seed_base")),
        )


@dataclass
class RunConfig:
    env_conf: Optional[object]
    env_tag: str
    problem_seed: int
    noise_seed_0: int
    opt_name: str
    num_rounds: int
    num_arms: int
    num_denoise: Optional[int]
    num_denoise_passive: Optional[int]
    max_proposal_seconds: Optional[float]
    rollout_workers: Optional[int]
    b_trace: bool
    trace_fn: str
    deadline: Optional[float] = None
    checkpoint_every: Optional[int] = None
    checkpoint_path: Optional[str] = None
    resume: bool = False
    video_enable: bool = False
    video_dir: Optional[str] = None
    video_prefix: str = "bo"
    video_num_episodes: int = 10
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


def sample_1(run_config: RunConfig):
    env_conf = _resolve_env_conf(run_config)
    max_proposal_seconds = _resolve_max_proposal_seconds(run_config.max_proposal_seconds)

    seed_all(env_conf.problem_seed + 27)
    _set_default_device()

    policy = default_policy(env_conf)
    collector_log = Collector()
    opt = _build_optimizer(run_config, env_conf, policy, collector_log)
    collector_trace, trace_records = _collect_trace(run_config, env_conf, opt, max_proposal_seconds)
    _maybe_render_videos(run_config, env_conf, opt)
    return _SampleResult(collector_log=collector_log, collector_trace=collector_trace, trace_records=trace_records)


def _resolve_env_conf(run_config: RunConfig):
    if run_config.env_conf is not None:
        return run_config.env_conf
    return get_env_conf(
        run_config.env_tag,
        problem_seed=run_config.problem_seed,
        noise_level=None,
        noise_seed_0=run_config.noise_seed_0,
    )


def _resolve_max_proposal_seconds(value: Optional[float]) -> float:
    import numpy as np

    if value is None:
        return float(np.inf)
    return float(value)


def _set_default_device() -> None:
    # Keep runtime visibility without mutating process-global torch defaults.
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEFAULT_DEVICE:", default_device)


def _build_optimizer(run_config: RunConfig, env_conf, policy, collector_log: Collector) -> Optimizer:
    return Optimizer(
        collector_log,
        env_conf=env_conf,
        env_tag=run_config.env_tag,
        policy=policy,
        num_arms=run_config.num_arms,
        num_denoise_measurement=run_config.num_denoise,
        num_denoise_passive=run_config.num_denoise_passive,
        rollout_workers=run_config.rollout_workers,
    )


def _load_resume_state(run_config: RunConfig):
    if not run_config.resume:
        return None
    checkpoint_path = run_config.checkpoint_path
    if not checkpoint_path:
        return None
    if not os.path.exists(checkpoint_path):
        return None
    return load_optimizer_checkpoint(checkpoint_path)


def _collect_trace(
    run_config: RunConfig,
    env_conf,
    opt: Optimizer,
    max_proposal_seconds: float,
) -> tuple[Collector, list[TraceRecord]]:
    collector_trace = Collector()
    trace_records: list[TraceRecord] = []
    resume_state = _load_resume_state(run_config)
    for i_iter, te in enumerate(
        opt.collect_trace(
            designer_name=run_config.opt_name,
            max_iterations=run_config.num_rounds,
            max_proposal_seconds=max_proposal_seconds,
            deadline=run_config.deadline,
            resume_state=resume_state,
            checkpoint_every=run_config.checkpoint_every,
            checkpoint_path=run_config.checkpoint_path,
        )
    ):
        trace_records.append(
            TraceRecord(
                i_iter=i_iter,
                dt_prop=te.dt_prop,
                dt_eval=te.dt_eval,
                rreturn=te.rreturn,
                env_name=env_conf.env_name,
                opt_name=run_config.opt_name,
            )
        )
        if run_config.b_trace:
            collector_trace(
                f"TRACE: name = {env_conf.env_name} opt_name = {run_config.opt_name} i_iter = {i_iter} dt_prop = {te.dt_prop:.3e} dt_eval = {te.dt_eval:.3e} return = {te.rreturn:.3e}"
            )
    collector_trace("DONE")
    return collector_trace, trace_records


def _resolve_video_dir(run_config: RunConfig):
    from pathlib import Path

    trace_path = Path(run_config.trace_fn).resolve()
    run_root = trace_path.parent.parent
    if run_config.video_dir is None:
        video_dir = run_root / "videos" / trace_path.name
    else:
        video_dir = Path(run_config.video_dir).expanduser()
        if not video_dir.is_absolute():
            video_dir = (Path.cwd() / video_dir).resolve()
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir, trace_path


def _maybe_render_videos(run_config: RunConfig, env_conf, opt: Optimizer) -> None:
    if not run_config.video_enable:
        return
    if opt.best_policy is None:
        print("VIDEO: best_policy is None; skipping")
        return
    if env_conf.gym_conf is None:
        print(f"VIDEO: env={env_conf.env_name} is not gym; skipping")
        return

    from experiments.video_render import render_policy_videos

    video_dir, trace_path = _resolve_video_dir(run_config)
    seed_base = int(run_config.video_seed_base) if run_config.video_seed_base is not None else int(env_conf.problem_seed)
    prefix = f"{run_config.video_prefix}_{trace_path.name}"
    print(
        f"VIDEO: dir={video_dir} episodes={run_config.video_num_episodes} "
        f"videos={run_config.video_num_video_episodes} select={run_config.video_episode_selection}",
        flush=True,
    )
    render_policy_videos(
        env_conf,
        opt.best_policy.clone(),
        video_dir=video_dir,
        video_prefix=prefix,
        num_episodes=int(run_config.video_num_episodes),
        num_video_episodes=int(run_config.video_num_video_episodes),
        episode_selection=str(run_config.video_episode_selection),
        seed_base=int(seed_base),
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


def _run_and_post(run_config: RunConfig):
    collector_log, collector_trace, trace_records = sample_1(run_config)
    post_process(collector_log, collector_trace, run_config.trace_fn, trace_records)
    return run_config.trace_fn


def scan_parallel(
    run_configs: list[RunConfig],
    max_total_seconds: Optional[float] = None,
    *,
    max_workers: int,
):
    t_0 = time.time()
    deadline = None if max_total_seconds is None else t_0 + max_total_seconds
    if deadline is not None and time.time() >= deadline:
        print(f"TIME_LIMIT: Stopping after {time.time() - t_0:.2f}s (max_total_seconds={max_total_seconds})")
        return
    for run_config in run_configs:
        run_config.deadline = deadline

    import traceback
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_and_post, rc): rc for rc in run_configs}
        for fut in as_completed(futures):
            rc = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"RUN_FAILED: {rc.trace_fn} err={exc}")
                traceback.print_exc()

    t_f = time.time()
    print(f"TIME_PARALLEL: {t_f - t_0:.2f} workers={max_workers}")


def true_false(string_bool):
    string_bool = str(string_bool).lower()
    if string_bool in ["false", "f"]:
        return False
    if string_bool in ["true", "t"]:
        return True
    assert False, string_bool


def _parse_optional_int(value) -> Optional[int]:
    if value in (None, "None"):
        return None
    return int(value)


def _parse_optional_float(value) -> Optional[float]:
    if value in (None, "None"):
        return None
    return float(value)


def _make_run_config(config: ExperimentConfig, out_dir: str, rep_idx: int) -> RunConfig:
    problem_seed = 18 + rep_idx
    trace_fn = f"{out_dir}/traces/{rep_idx:05d}"
    return RunConfig(
        trace_fn=trace_fn,
        env_conf=None,
        env_tag=config.env_tag,
        problem_seed=problem_seed,
        noise_seed_0=10 * problem_seed,
        opt_name=config.opt_name,
        num_rounds=config.num_rounds,
        num_arms=config.num_arms,
        num_denoise=config.num_denoise,
        num_denoise_passive=config.num_denoise_passive,
        max_proposal_seconds=config.max_proposal_seconds,
        rollout_workers=config.rollout_workers,
        b_trace=config.b_trace,
        checkpoint_every=config.checkpoint_every,
        checkpoint_path=f"{trace_fn}.ckpt.npz",
        resume=bool(config.resume),
        video_enable=bool(config.video_enable),
        video_dir=config.video_dir,
        video_prefix=str(config.video_prefix),
        video_num_episodes=int(config.video_num_episodes),
        video_num_video_episodes=int(config.video_num_video_episodes),
        video_episode_selection=str(config.video_episode_selection),
        video_seed_base=config.video_seed_base,
    )


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
            run_configs.append(_make_run_config(config, out_dir, i_rep))
    return run_configs


def sampler(config: ExperimentConfig, distributor_fn):
    run_configs = mk_replicates(config)
    try:
        signature = inspect.signature(distributor_fn)
    except (TypeError, ValueError):
        signature = None
    supports_timeout = False
    if signature is not None:
        supports_timeout = "max_total_seconds" in signature.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    if supports_timeout:
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
