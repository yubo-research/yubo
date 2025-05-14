import time

import modal
from experiment_sampler import post_process, sample_1

from analysis.data_io import data_is_done
from experiments.batches import prep_d_argss
from experiments.experiment_sampler import mk_replicates
from experiments.modal_image import mk_image

modal_image = mk_image()

_APP_NAME = "yubo_batches"
_TIMEOUT_HOURS = 24  # was: 5

_app_name = "yubo"
app = modal.App(name=_app_name)


def _dict():
    return modal.Dict.from_name("batches_dict", create_if_missing=True)


@app.function(image=modal_image, concurrency_limit=100, timeout=_TIMEOUT_HOURS * 60 * 60)  # , gpu="H100")
def modal_batches_worker(job):
    res_dict = _dict()

    key, d_args = job

    print(f"JOB: key = {key} d_args = {d_args}")
    trace_fn = d_args.pop("trace_fn")
    collector_log, collector_trace = sample_1(**d_args)
    res_dict[key] = (trace_fn, collector_log, collector_trace)


# @app.function(image=modal_image, concurrency_limit=1, timeout=_TIMEOUT_HOURS * 60 * 60)
# def modal_batches_submitter(job_name: str):
#     batches_submitter(job_name)


@app.function(image=modal_image, concurrency_limit=1, timeout=_TIMEOUT_HOURS * 60 * 60)
def modal_batches_resubmitter(batch_of_args):
    for key, d_args in batch_of_args:
        process_job = modal.Function.from_name(_app_name, "modal_batches_worker")
        process_job.spawn((key, d_args))


def batches_submitter(batch_tag: str):
    n = 0
    batch = []
    MAX_BATCH = 100

    def _flush():
        nonlocal batch
        func = modal.Function.from_name(_app_name, "modal_batches_resubmitter")
        func.spawn(batch)
        batch = []

    for key, d_args in _gen_jobs(batch_tag):
        print(f"JOB: {key} {d_args}")
        # process_job = modal.Function.from_name(_app_name, "modal_batches_worker")
        # process_job.spawn((key, d_args))
        batch.append((key, d_args))
        if len(batch) == MAX_BATCH:
            _flush()
        n += 1

    _flush()
    print("TOTAL:", n)


def _gen_jobs(batch_tag):
    d_argss = prep_d_argss(batch_tag)
    i_job = 0
    for d_args_batch in d_argss:
        for d_args in mk_replicates(d_args_batch):
            if not data_is_done(d_args["trace_fn"]):
                key = _job_key(batch_tag, i_job)
                yield key, d_args
                i_job += 1


def _job_key(job_name, i_job):
    return f"{job_name}-{i_job}"


@app.function(image=modal_image, concurrency_limit=1, timeout=_TIMEOUT_HOURS * 60 * 60)
def modal_batch_deleter(collected_keys):
    res_dict = _dict()
    for key in collected_keys:
        print("DEL:", key)
        try:
            del res_dict[key]
        except KeyError:
            pass


def collect():
    res_dict = _dict()
    print("DICT_SIZE:", res_dict.len())

    if True:
        collected_keys = set()
        print("ITEMS")
        for key, value in res_dict.items():
            if key.endswith("key_max"):
                print("SKIP", key)
                continue
            print("GETITEM", key)
            t_0 = time.time()
            (trace_fn, collector_log, collector_trace) = value
            t_f = time.time()
            print(f"GOTITEM: {key} {t_f - t_0:.1f}")
            if not data_is_done(trace_fn):
                post_process(collector_log, collector_trace, trace_fn)
            collected_keys.add(key)

        print(f"results_available before del: {res_dict.len()}")
        # for key in collected_keys:
        #     print("DEL:", key)
        #     del res_dict[key]
        func = modal.Function.from_name(_app_name, "modal_batch_deleter")
        func.spawn(collected_keys)
        print(f"num_collected = {len(collected_keys)}")


def status():
    res_dict = _dict()
    print(f"results_available = {res_dict.len()}")


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
    else:
        assert False, cmd
