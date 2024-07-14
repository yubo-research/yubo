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
        for call_id in f:
            call_id = call_id.strip()
            print("CALL_ID:", call_id)
            function_call = modal.functions.FunctionCall.from_id(call_id)
            trace_fn, collector_log, collector_trace = function_call.get(timeout=5)
            print("CALL_ID:", call_id, trace_fn)
            post_process(collector_log, collector_trace, trace_fn)
