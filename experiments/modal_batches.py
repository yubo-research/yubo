import queue
import time

import modal
from experiment_sampler import post_process, sample_1

from analysis.data_io import data_is_done
from experiments.batches import prep_d_argss
from experiments.experiment_sampler import mk_replicates
from experiments.modal_image import mk_image

modal_image = mk_image()

_APP_NAME = "yubo_batches"
_TIMEOUT_HOURS = 5

app = modal.App(name="yubo")


def _queue():
    return modal.Queue.from_name("batches_queue", create_if_missing=True)


def _dict():
    return modal.Dict.from_name("batches_dict", create_if_missing=True)


@app.function(image=modal_image, concurrency_limit=100, timeout=_TIMEOUT_HOURS * 60 * 60)  # , gpu="H100")
def modal_batches_worker():
    job_queue = _queue()
    res_dict = _dict()

    while True:
        try:
            key, d_args = job_queue.get(block=True, timeout=60)
        except queue.Empty:
            break

        print(f"JOB: key = {key} d_args = {d_args}")
        trace_fn = d_args.pop("trace_fn")
        collector_log, collector_trace = sample_1(**d_args)
        res_dict[key] = (trace_fn, collector_log, collector_trace)


@app.function(image=modal_image, concurrency_limit=1, timeout=_TIMEOUT_HOURS * 60 * 60)
def modal_batches_submitter(job_name: str):
    batches_submitter(job_name)


def batches_submitter(job_name: str, count_only=False):
    if not count_only:
        job_queue = _queue()
    num_submitted = 0
    for key, d_args in _gen_jobs(job_name):
        print(f"JOB: {key} {d_args}")
        if not count_only:
            job_queue.put((key, d_args))
        num_submitted += 1
    print("TOTAL:", num_submitted)


def _gen_jobs(job_name):
    d_argss = prep_d_argss()
    i_job = 0
    for d_args_batch in d_argss:
        for d_args in mk_replicates(d_args_batch):
            if not data_is_done(d_args["trace_fn"]):
                key = _job_key(job_name, i_job)
                yield key, d_args
                i_job += 1


def _job_key(job_name, i_job):
    return f"{job_name}-{i_job}"


def collect():
    job_queue = _queue()
    res_dict = _dict()
    while True:
        num_collected = 0
        for key, value in res_dict.items():
            if key.endswith("key_max"):
                continue

            (trace_fn, collector_log, collector_trace) = res_dict[key]
            print(f"JOB: {key}")
            post_process(collector_log, collector_trace, trace_fn)
            del res_dict[key]
            num_collected += 1
        print("How many jobs are running? Idk.")
        print(f"jobs_remaining = {job_queue.len()}")
        if num_collected == 0:
            time.sleep(30)
        else:
            time.sleep(3)


def status():
    job_queue = _queue()
    res_dict = _dict()
    print(f"jobs_remaining = {job_queue.len()}")
    print(f"results_available = {res_dict.len()}")


@app.local_entrypoint()
def batches(cmd: str, job_name: str = None, num: int = None):
    if cmd == "work":
        modal_function = modal.Function.lookup("yubo", "modal_batches_worker")
        for i in range(num):
            print("WORK:", i)
            modal_function.spawn()
    elif cmd == "submit-all":
        submitter = modal.Function.lookup("yubo", "modal_batches_submitter")
        submitter.spawn(job_name)
    elif cmd == "submit-missing":
        batches_submitter(job_name)
    elif cmd == "count-missing":
        batches_submitter(job_name, count_only=True)
    elif cmd == "status":
        status()
    elif cmd == "collect":
        assert job_name is None
        collect()
    else:
        assert False, cmd
