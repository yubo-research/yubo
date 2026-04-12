import os
import time
from typing import Optional

from analysis.data_io import write_summary_json, write_trace_jsonl
from experiments import experiment_sampler_shim as shim
from experiments.experiment_sampler_types import RunConfig


def post_process_stdout(
    collector_log,
    collector_trace,
    trace_records=None,
):
    for line in collector_log:
        print(line)
    for line in collector_trace:
        print(line)


def post_process(
    collector_log,
    collector_trace,
    trace_fn,
    trace_records=None,
    wall_seconds: Optional[float] = None,
    stop_reason: Optional[str] = None,
):
    shim.ensure_parent(trace_fn)

    if trace_records is not None:
        jsonl_fn = trace_fn + ".jsonl"
        write_trace_jsonl(jsonl_fn, trace_records)

    if wall_seconds is not None:
        write_summary_json(trace_fn, wall_seconds, stop_reason)

    def _w(f, line):
        f.write(line + "\n")

    with shim.data_writer(trace_fn) as f:
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
    torch = shim.torch_module()
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
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    try:
        shim.torch_module().set_num_threads(1)
    except Exception:
        pass


def _sample_and_post_process(run_config: RunConfig):
    _limit_worker_threads()
    t_start = time.perf_counter()
    result = shim.sample_1(run_config)
    wall_seconds = time.perf_counter() - t_start
    shim.post_process(
        result.collector_log,
        result.collector_trace,
        run_config.trace_fn,
        result.trace_records,
        wall_seconds=wall_seconds,
        stop_reason=result.stop_reason,
    )
    return run_config.trace_fn


def _scan_local_parallel(run_configs: list[RunConfig], *, max_workers: int) -> None:
    ctx = shim.mp_module().get_context()
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
