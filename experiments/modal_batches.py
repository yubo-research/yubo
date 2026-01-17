import time

import modal

from analysis.data_io import data_is_done
from experiments.batches import prep_d_argss
from experiments.experiment_sampler import mk_replicates, post_process, sample_1
from experiments.modal_image import mk_image

modal_image = mk_image()

_APP_NAME = "yubo_batches"
_TIMEOUT_HOURS = 24  # was: 5
_MAX_CONTAINERS = 1000

_app_name = "yubo"
app = modal.App(name=_app_name)


# TODO: Make the resubmitter and deleter a single function to save a slot. Or maybe that's not how functions work.


def _results_dict():
    return modal.Dict.from_name("batches_dict", create_if_missing=True)


def _submitted_dict():
    return modal.Dict.from_name("submitted_dict", create_if_missing=True)


@app.function(image=modal_image, concurrency_limit=_MAX_CONTAINERS, timeout=_TIMEOUT_HOURS * 60 * 60)  # , gpu="H100")
def modal_batches_worker(job):
    res_dict = _results_dict()

    key, run_config = job

    print(f"JOB: key = {key} run_config = {run_config}")
    collector_log, collector_trace, trace_records = sample_1(run_config)
    res_dict[key] = (run_config.trace_fn, collector_log, collector_trace, trace_records)


@app.function(image=modal_image, concurrency_limit=_MAX_CONTAINERS // 10, timeout=_TIMEOUT_HOURS * 60 * 60)
def modal_batches_resubmitter(batch_of_configs):
    submitted_dict = _submitted_dict()
    todo = []
    for key, run_config in batch_of_configs:
        if key in submitted_dict:
            continue
        submitted_dict[key] = True
        todo.append((key, run_config))

    print("TODO:", len(todo))

    worker = modal.Function.from_name(_app_name, "modal_batches_worker")
    worker.spawn_map(todo)


def batches_submitter(batch_tag: str):
    n = 0
    batch = []
    MAX_BATCH = 1000

    def _flush():
        nonlocal batch
        func = modal.Function.from_name(_app_name, "modal_batches_resubmitter")
        func.spawn(batch)
        batch = []

    for key, run_config in _gen_jobs(batch_tag):
        print(f"JOB: {key} {run_config}")
        # process_job = modal.Function.from_name(_app_name, "modal_batches_worker")
        # process_job.spawn((key, d_args))
        batch.append((key, run_config))
        if len(batch) == MAX_BATCH:
            _flush()
        n += 1

    _flush()
    print("TOTAL:", n)


def _gen_jobs(batch_tag):
    configs = prep_d_argss(batch_tag)
    i_job = 0
    for config in configs:
        for run_config in mk_replicates(config):
            if not data_is_done(run_config.trace_fn):
                key = _job_key(batch_tag, i_job)
                yield key, run_config
                i_job += 1


def _job_key(job_name, i_job):
    return f"{job_name}-{i_job}"


@app.function(image=modal_image, concurrency_limit=1, timeout=_TIMEOUT_HOURS * 60 * 60)
def modal_batch_deleter(collected_keys):
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
        if len(value) == 4:
            (trace_fn, collector_log, collector_trace, trace_records) = value
        else:
            (trace_fn, collector_log, collector_trace) = value
            trace_records = None
        t_f = time.time()
        print(f"GOTITEM: {key} {t_f - t_0:.1f}")
        if not data_is_done(trace_fn):
            post_process(collector_log, collector_trace, trace_fn, trace_records)
        collected_keys.add(key)

    print(f"results_available before del: {res_dict.len()}")
    func = modal.Function.from_name(_app_name, "modal_batch_deleter")
    func.spawn(collected_keys)

    print(f"num_collected = {len(collected_keys)}")


def status():
    res_dict = _results_dict()
    submitted_dict = _submitted_dict()
    print(f"results_available = {res_dict.len()}")
    print(f"submitted = {submitted_dict.len()}")


def clean_up():
    for name in ["batches_dict", "submitted_dict"]:
        try:
            modal.Dict.delete(name)
            print(f"CLEANUP: deleted dict name={name}")
        except Exception as e:
            print(f"CLEANUP: dict delete failed name={name} err={e!r}")


@app.local_entrypoint()
def batches(cmd: str, batch_tag: str = None, num: int = None):
    if cmd == "work":
        modal_function = modal.Function.lookup("yubo", "modal_batches_worker")
        for i in range(num):
            print("WORK:", i)
            modal_function.spawn()
    # elif cmd == "submit-all":
    #     submitter = modal.Function.lookup("yubo", "modal_batches_submitter")
    #     submitter.spawn(batch_tag)
    elif cmd == "submit-missing":
        batches_submitter(batch_tag)
    elif cmd == "status":
        status()
    elif cmd == "collect":
        assert batch_tag is None
        collect()
    elif cmd == "clean_up":
        assert batch_tag is None
        clean_up()
    else:
        assert False, cmd
