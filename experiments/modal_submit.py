import os

from experiment_sampler import sampler

from experiments.dist_modal import DistModal
from experiments.modal_app import app


@app.local_entrypoint()
def submit(job_fn, exp_dir, env_tag, opt_name, num_arms: int, num_rounds: int, num_reps: int):
    assert not os.path.exists(job_fn), f"{job_fn} already exists."
    d_args = dict(
        exp_dir=exp_dir,
        env_tag=env_tag,
        opt_name=opt_name,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
    )
    sampler(d_args, distributor_fn=DistModal(job_fn))
