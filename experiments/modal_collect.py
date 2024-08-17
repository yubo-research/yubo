import os

import modal
from experiment_sampler import post_process

from experiments.modal_app import app


def get_job_result(call_id):
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=5)
    except TimeoutError:
        result = {"result": "pending"}
    return result


@app.local_entrypoint()
def main(job_fn):
    with open(job_fn) as f:
        num_ids = 0
        for _ in f:
            num_ids += 1

    with open(job_fn) as f:
        num_skipped = 0
        i_call_id = -1
        num = 0
        for call_id in f:
            num += 1
            i_call_id += 1
            call_id = call_id.strip()
            print(f"CALL_ID: {i_call_id} / {num_ids} {call_id}")
            function_call = modal.functions.FunctionCall.from_id(call_id)
            try:
                trace_fn, collector_log, collector_trace = function_call.get(timeout=5)
            except (TimeoutError, modal.exception.FunctionTimeoutError) as e:
                print(f"SKIPPING: Timed out {repr(e)}")
                num_skipped += 1
                continue

            if os.path.exists(trace_fn):
                print("SKIPPING: Exists", i_call_id, call_id, trace_fn)
            else:
                print(f"CALL_ID: {i_call_id} / {num_ids} {call_id} {trace_fn}")
                post_process(collector_log, collector_trace, trace_fn)
    print(f"STATS: num = {num} num_skipped = {num_skipped}")
