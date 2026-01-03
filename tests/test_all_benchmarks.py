import pytest

from problems.benchmark_functions import all_benchmarks


@pytest.mark.parametrize("bm_name, num_dim", [(bm, num_dim) for bm in all_benchmarks() for num_dim in [1, 2, 3, 10]])
def test_benchmark(bm_name, num_dim):
    import os
    import shutil

    from experiments.experiment_sampler import ExperimentConfig, sampler, scan_local

    path = "_test/bm"

    print("BM:", bm_name, num_dim)

    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    config = ExperimentConfig(
        exp_dir=path,
        env_tag=f"f:{bm_name}-{num_dim}d",
        opt_name="random",
        num_arms=2,
        num_rounds=3,
        num_reps=1,
    )

    sampler(config, distributor_fn=scan_local)
