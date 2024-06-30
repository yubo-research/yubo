import time

import modal
from experiment_sampler import extract_trace_fns, post_process, sample_1, sampler


def mk_image():
    reqs = """
    numpy==1.26.0
    scipy==1.13.1
    torch==2.1.0
    botorch==0.9.2
    gpytorch==1.11
    gymnasium==0.29.1
    cma==3.3.0
    mujoco==2.3.7
    gymnasium[box2d]
    gymnasium[mujoco]
    """.split(
        "\n"
    )
    sreqs = []
    for req in reqs:
        req = req.strip()
        if len(req) == 0:
            continue
        print("REQ:", req)
        sreqs.append(req)
    # print("SREQS:", sreqs)
    return modal.Image.debian_slim(python_version="3.11.5").apt_install("swig").pip_install(sreqs)


modal_image = mk_image()

app = modal.App(name="yubo")


@app.function(image=modal_image, concurrency_limit=100)
def _sample_1_modal(d_args):
    return sample_1(**d_args)


def dist_modal(all_args):
    t_0 = time.time()
    trace_fns = extract_trace_fns(all_args)
    for trace_fn, (collector_log, collector_trace) in zip(trace_fns, _sample_1_modal.map(all_args)):
        post_process(collector_log, collector_trace, trace_fn)
    t_f = time.time()
    print(f"TIME_MODAL: {t_f - t_0:.2f}")


@app.local_entrypoint()
def main(exp_dir, env_tag, opt_name, num_arms: int, num_rounds: int, num_reps: int):
    d_args = dict(
        exp_dir=exp_dir,
        env_tag=env_tag,
        opt_name=opt_name,
        num_arms=num_arms,
        num_rounds=num_rounds,
        num_reps=num_reps,
    )
    sampler(d_args, distributor_fn=dist_modal)
