from experiments.batch_util import run_in_batches


def test_run_in_batches():
    results = []

    def run_fn(batch, dry_run):
        results.extend(batch)

    run_in_batches([1, 2, 3, 4, 5], max_parallel=2, run_batch_fn=run_fn)
    assert results == [1, 2, 3, 4, 5]
