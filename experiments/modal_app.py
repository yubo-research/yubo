import modal
from experiment_sampler import sample_1


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
    """.split("\n")
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


@app.function(image=modal_image, concurrency_limit=100, timeout=30 * 60, gpu="any")
def sample_1_modal(d_args):
    trace_fn = d_args.pop("trace_fn")
    collector_log, collector_trace = sample_1(**d_args)
    return trace_fn, collector_log, collector_trace
