import os


def run_in_batches(items, max_parallel, run_batch_fn, *, b_dry_run=False, num_threads=None):
    if num_threads is not None:
        for k in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
            os.environ[k] = str(int(num_threads))

    items = list(items)
    while len(items) > 0:
        todo = items[:max_parallel]
        items = items[max_parallel:]
        run_batch_fn(todo, b_dry_run)
