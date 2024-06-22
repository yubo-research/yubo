#!/usr/bin/env python


from experiment_sampler import parse_kv, sampler

from experiments.experiment_modal import modal_app as app


@app.local_entrypoint()
def main_modal(exp_dir, env_tag, opt_name, num_arms: int, num_rounds: int, num_reps: int):
    d_args = dict(
        exp_dir=exp_dir,
        env_tag=env_tag,
        opt_name=opt_name,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
    )
    sampler(d_args, b_modal=True)


if __name__ == "__main__":
    import sys

    d_args = parse_kv(sys.argv[1:])
    reqd_keys = ["exp_dir", "env_tag", "opt_name", "num_arms", "num_rounds", "num_reps"]
    for k in reqd_keys:
        assert k in d_args, f"Missing {k} in {list(d_args.keys())}. Required: {reqd_keys}"

    sampler(d_args, b_modal=False)
