import time

from experiments.experiment_sampler import ExperimentConfig, mk_replicates, post_process, sample_1
from experiments.modal_batches_impl import app
from experiments.modal_image import mk_image


@app.function(image=mk_image(), max_containers=1, timeout=60 * 60)
def modal_sample_1(run_config):
    result = sample_1(run_config)
    return result.collector_log, result.collector_trace, result.trace_records, result.stop_reason


@app.local_entrypoint()
def run_job(exp_dir, env_tag, opt_name, num_arms: int, num_rounds: int, num_reps: int):
    assert num_reps == 1, "One at a time"
    config = ExperimentConfig(
        exp_dir=exp_dir,
        env_tag=env_tag,
        opt_name=opt_name,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
    )

    run_configs = mk_replicates(config)
    if not run_configs:
        print("No pending runs")
        return
    run_config = run_configs[0]
    trace_fn = run_config.trace_fn
    print(run_config)

    t_0 = time.time()
    collector_log, collector_trace, trace_records, stop_reason = modal_sample_1.remote(run_config)
    post_process(collector_log, collector_trace, trace_fn, trace_records, stop_reason=stop_reason)
    print("TIME:", time.time() - t_0)
