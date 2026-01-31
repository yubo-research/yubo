import time

from experiment_sampler import mk_replicates, post_process, sample_1

from experiments.modal_batches import app
from experiments.modal_image import mk_image


@app.function(image=mk_image(), max_containers=1, timeout=60 * 60)  # , gpu="A100")
def modal_sample_1(d_args):
    collector_log, collector_trace = sample_1(**d_args)
    return collector_log, collector_trace


@app.local_entrypoint()
def run_job(exp_dir, env_tag, opt_name, num_arms: int, num_rounds: int, num_reps: int):
    assert num_reps == 1, "One at a time"
    d_args = dict(
        exp_dir=exp_dir,
        env_tag=env_tag,
        opt_name=opt_name,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
    )

    d_args = mk_replicates(d_args)[0]
    trace_fn = d_args.pop("trace_fn")
    print(d_args)

    t_0 = time.time()
    post_process(*modal_sample_1.remote(d_args), trace_fn)
    print("TIME:", time.time() - t_0)
