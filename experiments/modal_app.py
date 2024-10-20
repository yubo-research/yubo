import modal
from experiment_sampler import sample_1

from experiments.modal_image import mk_image

modal_image = mk_image()

app = modal.App(name="yubo")


@app.function(image=modal_image, concurrency_limit=100, timeout=90 * 60)  # , gpu="H100")
def sample_1_modal(d_args):
    trace_fn = d_args.pop("trace_fn")
    collector_log, collector_trace = sample_1(**d_args)
    return trace_fn, collector_log, collector_trace
