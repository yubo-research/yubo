import queue
import time

import modal

from experiments.modal_image import mk_image

modal_image = mk_image()

app = modal.App(name="my-job")


@app.function(image=modal_image, concurrency_limit=2, timeout=3000)  # , gpu="T4")
def process_job(cmd):
    my_queue = modal.Queue.from_name("my-persisted-queue-b", create_if_missing=True)
    if cmd == "submit":
        for i in range(10000):
            print("PUT:", i)
            my_queue.put(f"key_{i}")
        return

    my_dict = modal.Dict.from_name("my-persisted-dict", create_if_missing=True)

    while True:
        try:
            key = my_queue.get(block=True, timeout=10)
        except queue.Empty:
            break
        print("KEY:", key)
        my_dict[key] = (key, "b", time.time())


def start(cmd):
    process_job = modal.Function.lookup("my-job", "process_job")
    process_job.spawn(cmd)


def get_job_result():
    my_dict = modal.Dict.from_name("my-persisted-dict", create_if_missing=True)
    for i in range(10000):
        key = f"key_{i}"
        print(key, my_dict[key])


@app.local_entrypoint()
def main(cmd):
    if cmd == "start":
        start("processor")
    elif cmd == "submit":
        start("submitter")
    elif cmd == "get":
        get_job_result()
    else:
        assert False, cmd
