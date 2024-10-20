import os

import modal
from experiment_sampler import post_process

from experiments.dist_modal import collect
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
    def _cb(result):
        trace_fn, collector_log, collector_trace = result
        if os.path.exists(trace_fn):
            print("SKIPPING: Exists", trace_fn)
        else:
            post_process(collector_log, collector_trace, trace_fn)

    collect(job_fn, _cb)
