import time

import modal
from grpclib import GRPCError


def collect(job_fn, cb_job):
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
                result = function_call.get(timeout=5)
            except (TimeoutError, modal.exception.FunctionTimeoutError) as e:
                print(f"SKIPPING: Timed out {repr(e)}")
                num_skipped += 1
                continue

            cb_job(result)

    print(f"STATS: num = {num} num_skipped = {num_skipped}")


class DistModal:
    def __init__(self, app_name, function_name, job_fn):
        self._app_name = app_name
        self._function_name = function_name
        self._job_fn = job_fn

    def __call__(self, all_args):
        t_0 = time.time()
        modal_function = modal.Function.lookup(self._app_name, self._function_name)
        dt_sleep = 1.0
        with open(self._job_fn, "w") as f:
            for i_args, d_args in enumerate(all_args):
                while True:
                    try:
                        call = modal_function.spawn(d_args)
                    except GRPCError as e:
                        print(f"WARN: spawn() error. Retrying. {repr(e)}")
                        time.sleep(dt_sleep)
                        dt_sleep = min(30, 2 * dt_sleep)
                    else:
                        dt_sleep = 1.0
                        break
                    time.sleep(dt_sleep)
                # else:
                # raise RuntimeError("Could not submit jobs to Modal")

                print(f"SUBMIT: {call.object_id} {i_args} / {len(all_args)} {d_args}")
                f.write(f"{call.object_id}\n")

        t_f = time.time()
        print(f"TIME_MODAL: {t_f - t_0:.2f}")
