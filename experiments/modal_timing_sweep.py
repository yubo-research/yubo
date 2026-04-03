"""Modal timing sweep for BO experiments.

This script runs timing experiments on Modal with:
- 5-hour cumulative proposal-time cap (``max_proposal_seconds`` / ``Optimizer._cum_dt_proposing``)
- Modal timeout 24h as a hard safety ceiling (wall clock)
- Wall-time recording in {trace_fn}-summary.json with stop_reason
"""

import time
from typing import Optional

import modal

from analysis.data_io import data_is_done
from experiments.experiment_sampler import (
    TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS,
    ExperimentConfig,
    mk_replicates,
    post_process,
    prep_args_1,
    sample_1,
)
from experiments.modal_image import mk_image

modal_image = mk_image()

_APP_NAME = "yubo_timing_sweep"
_TIMEOUT_HOURS = 24
_MAX_CONTAINERS = 500

app = modal.App(name=_APP_NAME)


def _results_dict():
    return modal.Dict.from_name("timing_sweep_dict", create_if_missing=True)


def _submitted_dict():
    return modal.Dict.from_name("timing_sweep_submitted_dict", create_if_missing=True)


@app.function(
    image=modal_image,
    max_containers=_MAX_CONTAINERS,
    timeout=_TIMEOUT_HOURS * 60 * 60,
)
def timing_sweep_worker(job):
    res_dict = _results_dict()

    key, run_config = job

    run_config.deadline = None

    print(f"JOB: key = {key} run_config = {run_config}")
    t_start = time.perf_counter()
    result = sample_1(run_config)
    wall_seconds = time.perf_counter() - t_start
    print(f"JOB_DONE: key = {key} wall_seconds = {wall_seconds:.2f} stop_reason = {result.stop_reason}")
    res_dict[key] = (
        run_config.trace_fn,
        result.collector_log,
        result.collector_trace,
        result.trace_records,
        wall_seconds,
        result.stop_reason,
    )


@app.function(
    image=modal_image,
    max_containers=_MAX_CONTAINERS // 10,
    timeout=_TIMEOUT_HOURS * 60 * 60,
)
def timing_sweep_resubmitter(batch_of_configs):
    submitted_dict = _submitted_dict()
    todo = []
    for key, run_config, force in batch_of_configs:
        if (not force) and (key in submitted_dict):
            continue
        submitted_dict[key] = True
        todo.append((key, run_config))

    print("TODO:", len(todo))

    worker = modal.Function.from_name(_APP_NAME, "timing_sweep_worker")
    worker.spawn_map(todo)


def _job_key(batch_tag, path):
    return f"{batch_tag}-{path}"


def _gen_jobs(batch_tag: str, configs: list[ExperimentConfig]):
    for config in configs:
        for run_config in mk_replicates(config):
            if not data_is_done(run_config.trace_fn):
                key = _job_key(batch_tag, run_config.trace_fn)
                yield key, run_config


def submit_configs(batch_tag: str, configs: list[ExperimentConfig], force: bool = False):
    n = 0
    batch = []
    max_batch = 500

    def _flush():
        nonlocal batch
        func = modal.Function.from_name(_APP_NAME, "timing_sweep_resubmitter")
        func.spawn(batch)
        batch = []

    for key, run_config in _gen_jobs(batch_tag, configs):
        print(f"JOB: {key} {run_config}")
        batch.append((key, run_config, force))
        if len(batch) == max_batch:
            _flush()
        n += 1

    _flush()
    print(f"TOTAL: {n}")


@app.function(image=modal_image, max_containers=1, timeout=_TIMEOUT_HOURS * 60 * 60)
def timing_sweep_deleter(collected_keys):
    res_dict = _results_dict()
    for key in collected_keys:
        print("DEL:", key)
        try:
            del res_dict[key]
        except KeyError:
            pass


def collect():
    res_dict = _results_dict()
    print("DICT_SIZE:", res_dict.len())

    collected_keys = set()
    print("ITEMS")
    for key, value in res_dict.items():
        if key.endswith("key_max"):
            print("SKIP", key)
            continue
        print("GETITEM", key)
        t_0 = time.time()
        wall_seconds = None
        stop_reason = None
        if len(value) == 6:
            (trace_fn, collector_log, collector_trace, trace_records, wall_seconds, stop_reason) = value
        elif len(value) == 4:
            (trace_fn, collector_log, collector_trace, trace_records) = value
        else:
            (trace_fn, collector_log, collector_trace) = value
            trace_records = None
        t_f = time.time()
        print(f"GOTITEM: {key} wall_seconds={wall_seconds} stop_reason={stop_reason} fetch_time={t_f - t_0:.1f}")
        if not data_is_done(trace_fn):
            post_process(
                collector_log,
                collector_trace,
                trace_fn,
                trace_records,
                wall_seconds=wall_seconds,
                stop_reason=stop_reason,
            )
        collected_keys.add(key)

    print(f"results_available before del: {res_dict.len()}")
    func = modal.Function.from_name(_APP_NAME, "timing_sweep_deleter")
    func.spawn(collected_keys)

    print(f"num_collected = {len(collected_keys)}")


def status():
    res_dict = _results_dict()
    submitted_dict = _submitted_dict()
    print(f"results_available = {res_dict.len()}")
    print(f"submitted = {submitted_dict.len()}")


def clean_up():
    for name in ["timing_sweep_dict", "timing_sweep_submitted_dict"]:
        try:
            modal.Dict.delete(name)
            print(f"CLEANUP: deleted dict name={name}")
        except Exception as e:
            print(f"CLEANUP: dict delete failed name={name} err={e!r}")


def prep_timing_sweep(
    results_dir: str,
    sweep_tuples: list[tuple],
    exp_dir: str = "exp_timing_sweep",
) -> list[ExperimentConfig]:
    """Build ExperimentConfigs from a list of tuples.

    Each tuple should be:
        (opt_name, env_tag, num_arms, num_rounds, num_denoise, num_denoise_passive)

    All configs use num_reps=1, no wall-clock deadline, and max_proposal_seconds=5*3600
    (cumulative ``dt_prop`` budget).
    """
    configs = []
    for t in sweep_tuples:
        opt_name, env_tag, num_arms, num_rounds, num_denoise, num_denoise_passive = t
        config = prep_args_1(
            results_dir,
            exp_dir=exp_dir,
            problem=env_tag,
            opt=opt_name,
            num_arms=num_arms,
            num_replications=1,
            num_rounds=num_rounds,
            noise=None,
            num_denoise=num_denoise,
            num_denoise_passive=num_denoise_passive,
        )
        config.max_total_seconds = None
        config.max_proposal_seconds = float(TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS)
        configs.append(config)
    return configs


def run_timing_sweep(
    batch_tag: str,
    sweep_tuples: list[tuple],
    results_dir: str = "results",
    exp_dir: str = "exp_timing_sweep",
    force: bool = False,
):
    """Submit a timing sweep to Modal.

    Args:
        batch_tag: Unique identifier for this batch of jobs.
        sweep_tuples: List of (opt_name, env_tag, num_arms, num_rounds, num_denoise, num_denoise_passive).
        results_dir: Base results directory.
        exp_dir: Experiment subdirectory name.
        force: If True, resubmit even if already submitted.
    """
    configs = prep_timing_sweep(results_dir, sweep_tuples, exp_dir=exp_dir)
    submit_configs(batch_tag, configs, force=force)


@app.local_entrypoint()
def main(cmd: str, batch_tag: Optional[str] = None):
    if cmd == "status":
        status()
    elif cmd == "collect":
        collect()
    elif cmd == "clean_up":
        clean_up()
    else:
        raise ValueError(f"Unknown command: {cmd}")
