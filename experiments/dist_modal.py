import time

import modal


class DistModal:
    def __init__(self, job_fn):
        self._job_fn = job_fn

    def __call__(self, all_args):
        t_0 = time.time()
        sample_1_modal = modal.Function.lookup("yubo", "sample_1_modal")
        with open(self._job_fn, "w") as f:
            for i_args, d_args in enumerate(all_args):
                call = sample_1_modal.spawn(d_args)
                print(f"SUBMIT: {call.object_id} {i_args} / {len(all_args)}")
                f.write(f"{call.object_id}\n")

        t_f = time.time()
        print(f"TIME_MODAL: {t_f - t_0:.2f}")
