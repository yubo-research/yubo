# my_job_queue.py
import os
import time

import modal
import torch


def mk_image():
    reqs = """
    numpy==1.26.0
    scipy==1.13.1
    torch==2.1.0
    botorch==0.9.2
    gpytorch==1.11
    gymnasium==0.29.1
    cma==3.3.0
    mujoco==2.3.7
    gymnasium[box2d]
    gymnasium[mujoco]
    """.split(
        "\n"
    )
    sreqs = []
    for req in reqs:
        req = req.strip()
        if len(req) == 0:
            continue
        print("REQ:", req)
        sreqs.append(req)
    # print("SREQS:", sreqs)
    return modal.Image.debian_slim(python_version="3.11.5").apt_install("swig").pip_install(sreqs)


modal_image = mk_image()

app = modal.App(name="my-job-queue")


def time_rand(device):
    t_0 = time.time()
    x = torch.rand(size=(100000, 1000), device=device)
    t_f = time.time()

    print()
    print("DEVICE:", x.device)
    print("X_LAST:", x.flatten()[-1].item())
    print("TIME:", t_f - t_0)


@app.function(image=modal_image, concurrency_limit=2, timeout=30, gpu="T4")
def process_job():
    os.system("nvidia-smi")
    print(torch.__version__)
    print("HAS_CUDA:", torch.cuda.is_available())
    time_rand("cpu")
    time_rand("cpu")
    time_rand("cpu")
    time_rand("cuda:0")
    time_rand("cuda:0")
    time_rand("cuda:0")


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
    process_job.remote()


# @app.local_entrypoint()
# def main():
#     data = "my-data"

#     if cmd == "submit":
#         call_id = submit_job(data)
#         print("CALL_ID:", call_id)
#     else:
#         print("RESULT:", get_job_result(cmd))
