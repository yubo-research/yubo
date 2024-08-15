import time

import modal
from grpclib import GRPCError


class DistModal:
    def __init__(self, job_fn):
        self._job_fn = job_fn

    def __call__(self, all_args):
        t_0 = time.time()
        sample_1_modal = modal.Function.lookup("yubo", "sample_1_modal")
        dt_sleep = 1.0
        with open(self._job_fn, "w") as f:
            for i_args, d_args in enumerate(all_args):
                while True:
                    try:
                        call = sample_1_modal.spawn(d_args)
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
