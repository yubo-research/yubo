# my_job_queue.py
import modal

app = modal.App("my-job-queue")


@app.function()
def process_job(data):
    # Perform the job processing here
    return {"result": data}


def submit_job(data):
    # Since the `process_job` function is deployed, need to first look it up
    process_job = modal.Function.lookup("my-job-queue", "process_job")
    call = process_job.spawn(data)
    return call.object_id


def get_job_result(call_id):
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=5)
    except TimeoutError:
        result = {"result": "pending"}
    return result


@app.local_entrypoint()
def main():
    data = "my-data"

    # Submit the job to Modal
    call_id = submit_job(data)
    print(get_job_result(call_id))
