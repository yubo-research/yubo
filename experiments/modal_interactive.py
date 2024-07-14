import time

from experiment_sampler import sampler

from experiments.experiment_sampler import post_process
from experiments.modal_app import app, sample_1_modal


def dist_modal(all_args):
    t_0 = time.time()
    for trace_fn, collector_log, collector_trace in sample_1_modal.map(all_args):
        post_process(collector_log, collector_trace, trace_fn)
    t_f = time.time()
    print(f"TIME_MODAL: {t_f - t_0:.2f}")


@app.local_entrypoint()
def run_job(exp_dir, env_tag, opt_name, num_arms: int, num_rounds: int, num_reps: int):
    d_args = dict(
        exp_dir=exp_dir,
        env_tag=env_tag,
        opt_name=opt_name,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
    )
    sampler(d_args, distributor_fn=dist_modal)
