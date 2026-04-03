import sys
import time

import modal

from analysis.data_io import data_is_done
from experiments.batches import prep_d_argss
from experiments.experiment_sampler import mk_replicates, post_process, sample_1
from experiments.modal_image import mk_image

modal_image = mk_image()

_TIMEOUT_HOURS = 24  # was: 5
_MAX_CONTAINERS = 1000


def _extract_tag_from_argv() -> str:
    for i, arg in enumerate(sys.argv):
        if "modal_batches.py" in arg and i + 1 < len(sys.argv):
            candidate = sys.argv[i + 1]
            if not candidate.startswith("-"):
                return candidate
    return "default"


_tag = _extract_tag_from_argv()


def _get_app_name(tag: str) -> str:
    return f"yubo_{tag}"


_app_name = _get_app_name(_tag)
app = modal.App(name=_app_name)


def _results_dict(tag: str):
    return modal.Dict.from_name(f"batches_dict_{tag}", create_if_missing=True)


def _submitted_dict(tag: str):
    return modal.Dict.from_name(f"submitted_dict_{tag}", create_if_missing=True)


@app.function(
    image=modal_image,
    max_containers=_MAX_CONTAINERS,
    timeout=_TIMEOUT_HOURS * 60 * 60,
)
def modal_batches_worker(job):
    tag, key, run_config = job
    res_dict = _results_dict(tag)

    print(f"JOB: key = {key} run_config = {run_config}")
    t_start = time.perf_counter()
    result = sample_1(run_config)
    wall_seconds = time.perf_counter() - t_start
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
def modal_batches_resubmitter(batch_of_configs, tag: str):
    submitted_dict = _submitted_dict(tag)
    todo = []
    for key, run_config, force in batch_of_configs:
        if (not force) and (key in submitted_dict):
            continue
        submitted_dict[key] = True
        todo.append((tag, key, run_config))

    print("TODO:", len(todo))

    worker = modal.Function.from_name(_get_app_name(tag), "modal_batches_worker")
    worker.spawn_map(todo)


def batches_submitter(tag: str, batch_tag: str, force: bool = False):
    n = 0
    batch = []
    MAX_BATCH = 1000

    def _flush():
        nonlocal batch
        func = modal.Function.from_name(_get_app_name(tag), "modal_batches_resubmitter")
        func.spawn(batch, tag)
        batch = []

    for key, run_config in _gen_jobs(batch_tag):
        print(f"JOB: {key} {run_config}")
        batch.append((key, run_config, force))
        if len(batch) == MAX_BATCH:
            _flush()
        n += 1

    _flush()
    print("TOTAL:", n)


def _gen_jobs(batch_tag):
    configs = prep_d_argss(batch_tag)
    for config in configs:
        for run_config in mk_replicates(config):
            if not data_is_done(run_config.trace_fn):
                key = _job_key(batch_tag, run_config.trace_fn)
                yield key, run_config


def _job_key(job_name, path):
    return f"{job_name}-{path}"


@app.function(image=modal_image, max_containers=1, timeout=_TIMEOUT_HOURS * 60 * 60)
def modal_batch_deleter(collected_keys, tag: str):
    res_dict = _results_dict(tag)
    for key in collected_keys:
        print("DEL:", key)
        try:
            del res_dict[key]
        except KeyError:
            pass


def _collect(tag: str):
    res_dict = _results_dict(tag)
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
        print(f"GOTITEM: {key} {t_f - t_0:.1f}")
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
    func = modal.Function.from_name(_get_app_name(tag), "modal_batch_deleter")
    func.spawn(collected_keys, tag)

    print(f"num_collected = {len(collected_keys)}")


def status(tag: str):
    res_dict = _results_dict(tag)
    submitted_dict = _submitted_dict(tag)
    print(f"results_available = {res_dict.len()}")
    print(f"submitted = {submitted_dict.len()}")


def clean_up(tag: str):
    for name in [f"batches_dict_{tag}", f"submitted_dict_{tag}"]:
        try:
            modal.Dict.delete(name)
            print(f"CLEANUP: deleted dict name={name}")
        except Exception as e:
            print(f"CLEANUP: dict delete failed name={name} err={e!r}")


@app.local_entrypoint()
def batches(tag: str, cmd: str, batch_tag: str = None, num: int = None):
    if cmd == "work":
        modal_function = modal.Function.lookup(_get_app_name(tag), "modal_batches_worker")
        for i in range(num):
            print("WORK:", i)
            modal_function.spawn()
    elif cmd == "submit-missing":
        batches_submitter(tag, batch_tag)
    elif cmd == "submit-missing-force":
        batches_submitter(tag, batch_tag, force=True)
    elif cmd == "status":
        status(tag)
    elif cmd == "collect":
        assert batch_tag is None
        _collect(tag)
    elif cmd == "clean_up":
        assert batch_tag is None
        clean_up(tag)
    else:
        assert False, cmd
