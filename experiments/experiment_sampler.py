import hashlib
import importlib
import multiprocessing as mp
import os
import time
from contextlib import contextmanager
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
from common.experiment_seeds import (
    global_seed_for_run,
    noise_seed_0_from_problem_seed,
    problem_seed_from_rep_index,
)
from common.seed_all import seed_all
from experiments.bo_console import BOConsoleCollector, print_bo_footer
from experiments.experiment_util import ensure_parent


class _SampleResult(NamedTuple):
    collector_log: Collector
    collector_trace: Collector
    trace_records: list[TraceRecord]


def _load_attr(module_parts: tuple[str, ...], attr_name: str):
    module_name = ".".join(module_parts)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


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
    video_enable: bool = False
    scale: Optional[str] = None  # "auto" | "low" | "medium" | "high" | "huge" for dim-based scaling
    video_num_episodes: int = 8
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: Optional[int] = None
    video_prefix: str = "bo"
    runtime_device: str = "auto"
    local_workers: int = 1

    def to_dir_name(self) -> str:
        config_str = (
            f"env={self.env_tag}--opt_name={self.opt_name}"
            f"--num_arms={self.num_arms}--num_rounds={self.num_rounds}"
            f"--num_reps={self.num_reps}--num_denoise={self.num_denoise}"
            f"--max_proposal_seconds={self.max_proposal_seconds}"
            f"--video_enable={self.video_enable}"
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
        runtime_device = str(d.get("runtime_device", "auto")).strip().lower()
        if runtime_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"runtime_device must be one of: auto, cpu, cuda (got: {runtime_device})")
        local_workers = int(d.get("local_workers", 1))
        if local_workers < 1:
            raise ValueError(f"local_workers must be >= 1 (got: {local_workers})")

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
            video_enable=true_false(d.get("video_enable", False)),
            video_num_video_episodes=3,
            video_episode_selection="best",
            video_seed_base=None,
            video_prefix="bo",
            runtime_device=runtime_device,
            local_workers=local_workers,
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
    bo_console: bool = True
    deadline: Optional[float] = None
    video_enable: bool = False
    video_num_episodes: int = 8
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: Optional[int] = None
    video_prefix: str = "bo"
    runtime_device: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


@contextmanager
def _temporary_default_device(runtime_device: str):
    if str(runtime_device).strip().lower() != "cuda" or not torch.cuda.is_available():
        yield
        return
    prev_default_device = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(prev_default_device)


def _collect_trace_records(opt, opt_name, num_rounds, max_proposal_seconds, deadline, env_conf, b_trace):
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
        trace_records.append(
            TraceRecord(
                i_iter=i_iter,
                dt_prop=te.dt_prop,
                dt_eval=te.dt_eval,
                rreturn=te.rreturn,
                env_name=env_conf.env_name,
                opt_name=opt_name,
            )
        )
        if b_trace:
            collector_trace(
                f"TRACE: env={env_conf.env_name} opt_name={opt_name} iter={i_iter} "
                f"proposal_dt={te.dt_prop:.3e}s eval_dt={te.dt_eval:.3e}s return={te.rreturn:.3e}"
            )
    collector_trace("DONE")
    return trace_records, collector_trace


def _render_sample_video(
    opt,
    run_config,
    env_conf,
    video_prefix,
    video_num_episodes,
    video_num_video_episodes,
    video_episode_selection,
    video_seed_base,
):
    from pathlib import Path

    render_policy_videos_bo = _load_attr(("common", "video"), "render_policy_videos_bo")

    video_dir = Path(run_config.trace_fn).parent / "videos"
    seed_base = int(video_seed_base) if video_seed_base is not None else int(env_conf.problem_seed)
    render_policy_videos_bo(
        env_conf,
        opt.best_policy.clone(),
        video_dir=video_dir,
        video_prefix=str(video_prefix),
        num_episodes=int(video_num_episodes),
        num_video_episodes=int(video_num_video_episodes),
        episode_selection=str(video_episode_selection),
        seed_base=int(seed_base),
    )


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
    video_enable = run_config.video_enable
    video_num_episodes = run_config.video_num_episodes
    video_num_video_episodes = run_config.video_num_video_episodes
    video_episode_selection = run_config.video_episode_selection
    video_seed_base = run_config.video_seed_base
    video_prefix = run_config.video_prefix
    bo_console = getattr(run_config, "bo_console", True)

    if max_proposal_seconds is None:
        max_proposal_seconds = np.inf

    seed_all(global_seed_for_run(env_conf.problem_seed))

    with _temporary_default_device(getattr(run_config, "runtime_device", "auto")):
        default_policy = _load_attr(("problems", "env_conf"), "default_policy")
        policy = default_policy(env_conf)

        if bo_console:
            collector_log = BOConsoleCollector(
                env_tag=env_conf.env_name,
                opt_name=opt_name,
                num_rounds=num_rounds,
                num_arms=num_arms,
            )
        else:
            collector_log = Collector()
        Optimizer = _load_attr(("optimizer", "optimizer"), "Optimizer")
        opt = Optimizer(
            collector_log,
            env_conf=env_conf,
            policy=policy,
            num_arms=num_arms,
            num_denoise_measurement=num_denoise,
            num_denoise_passive=num_denoise_passive,
        )

        trace_records, collector_trace = _collect_trace_records(opt, opt_name, num_rounds, max_proposal_seconds, deadline, env_conf, b_trace)

        if bo_console:
            t0 = getattr(opt, "_t_0", None)
            if t0 is not None and isinstance(t0, (int, float)):
                total_time = time.time() - float(t0)
            else:
                total_time = 0.0
            best = getattr(opt, "r_best_est", None)
            best_val = float(best) if best is not None and isinstance(best, (int, float)) else 0.0
            print_bo_footer(best_val, max(0.0, total_time))

        if video_enable and video_num_video_episodes > 0 and opt.best_policy is not None:
            _render_sample_video(
                opt,
                run_config,
                env_conf,
                video_prefix,
                video_num_episodes,
                video_num_video_episodes,
                video_episode_selection,
                video_seed_base,
            )

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


def _normalize_runtime_device(device: str) -> str:
    mode = str(device).strip().lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"runtime_device must be one of: auto, cpu, cuda (got: {device})")
    return mode


def _is_rollout_heavy_env(env_tag: str) -> bool:
    tag = str(env_tag)
    return tag.startswith(("atari:", "ALE/", "dm_control/", "dm:"))


def _resolve_runtime_device(*, requested: str, env_tag: str, local_workers: int) -> str:
    mode = _normalize_runtime_device(requested)
    if mode == "cpu":
        return "cpu"
    if mode == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("RUNTIME_WARN: runtime_device=cuda requested but CUDA unavailable; falling back to cpu")
        return "cpu"
    if local_workers > 1:
        return "cpu"
    if _is_rollout_heavy_env(env_tag):
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _limit_worker_threads():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _sample_and_post_process(run_config: RunConfig):
    _limit_worker_threads()
    collector_log, collector_trace, trace_records = sample_1(run_config)
    post_process(collector_log, collector_trace, run_config.trace_fn, trace_records)
    return run_config.trace_fn


def _scan_local_parallel(run_configs: list[RunConfig], *, max_workers: int) -> None:
    ctx = mp.get_context()
    pool = ctx.Pool(processes=int(max_workers), initializer=_limit_worker_threads)
    try:
        for _ in pool.imap_unordered(_sample_and_post_process, run_configs):
            pass
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        raise
    except Exception:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()


def scan_local(
    run_configs: list[RunConfig],
    max_total_seconds: Optional[float] = None,
    *,
    local_workers: int = 1,
    env_tag: Optional[str] = None,
):
    if not run_configs:
        print("TIME_LOCAL: 0.00")
        return
    workers = max(1, int(local_workers))
    max_workers = min(workers, len(run_configs))
    runtime_device = _resolve_runtime_device(
        requested=getattr(run_configs[0], "runtime_device", "auto"),
        env_tag=env_tag if env_tag is not None else str(run_configs[0].env_conf.env_name),
        local_workers=max_workers,
    )
    for run_config in run_configs:
        run_config.runtime_device = runtime_device
    print(f"RUNTIME: device={runtime_device} workers={max_workers}")

    t_0 = time.time()
    if max_workers > 1 and max_total_seconds is None:
        _scan_local_parallel(run_configs, max_workers=max_workers)
        t_f = time.time()
        print(f"TIME_LOCAL: {t_f - t_0:.2f}")
        return

    deadline = None if max_total_seconds is None else t_0 + max_total_seconds
    for run_config in run_configs:
        if deadline is not None and time.time() >= deadline:
            print(f"TIME_LIMIT: Stopping after {time.time() - t_0:.2f}s (max_total_seconds={max_total_seconds})")
            break
        run_config.deadline = deadline
        _sample_and_post_process(run_config)
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
    get_env_conf = _load_attr(("problems", "env_conf"), "get_env_conf")

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
            problem_seed = problem_seed_from_rep_index(i_rep)
            env_conf = get_env_conf(
                config.env_tag,
                problem_seed=problem_seed,
                noise_level=None,
                noise_seed_0=noise_seed_0_from_problem_seed(problem_seed),
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
                    video_enable=config.video_enable,
                    video_num_episodes=config.video_num_episodes,
                    video_num_video_episodes=config.video_num_video_episodes,
                    video_episode_selection=config.video_episode_selection,
                    video_seed_base=config.video_seed_base,
                    video_prefix=config.video_prefix,
                    runtime_device=config.runtime_device,
                )
            )
    return run_configs


def sampler(config: ExperimentConfig, distributor_fn):
    run_configs = mk_replicates(config)
    if distributor_fn == scan_local:
        distributor_fn(
            run_configs,
            max_total_seconds=config.max_total_seconds,
            local_workers=config.local_workers,
            env_tag=config.env_tag,
        )
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
