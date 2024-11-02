import queue

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
    job_queue = _queue()
    res_dict = _dict()
    for key, d_args in _gen_jobs(job_name):
        print(f"JOB: {key}")
        job_queue.put((key, d_args))
        res_dict[f"{job_name}-key_max"] = key


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
    res_dict = _dict()
    # key_max = res_dict[f"{job_name}-key_max"]
    # i_max = int(key_max.split("-")[-1]) + 1
    for key, value in res_dict.items():
        if key.endswith("key_max"):
            continue

        (trace_fn, collector_log, collector_trace) = res_dict[key]
        print(f"JOB: {key}")
        post_process(collector_log, collector_trace, trace_fn)
        del res_dict[key]


def status():
    job_queue = _queue()
    res_dict = _dict()
    print(f"jobs_remaining = {job_queue.len()}")
    print(f"results_available = {res_dict.len() - 1}")


@app.local_entrypoint()
def batches(cmd: str, job_name: str = None, num: int = None):
    if cmd == "dry-run":
        num_jobs = 0
        for key, _ in _gen_jobs(job_name):
            print("JOB_DRY_RUN:", key)
            num_jobs += 1
        print(f"NUM_JOB: {num_jobs}")
    elif cmd == "work":
        modal_function = modal.Function.lookup("yubo", "modal_batches_worker")
        for i in range(num):
            print("WORK:", i)
            modal_function.spawn()
    elif cmd == "submit":
        submitter = modal.Function.lookup("yubo", "modal_batches_submitter")
        submitter.spawn(job_name)
    elif cmd == "status":
        status()
    elif cmd == "collect":
        assert job_name is None
        collect()
    else:
        assert False, cmd
