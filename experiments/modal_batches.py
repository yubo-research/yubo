from experiments.modal_batches_impl import (
    _collect as batches_collector,
)
from experiments.modal_batches_impl import (
    _get_app_name,
    _job_key,
    _results_dict,
    _submitted_dict,
    app,
    batches_submitter,
    clean_up,
    modal,
    modal_batch_deleter,
    modal_batches_resubmitter,
    modal_batches_worker,
    status,
    stop,
)
from experiments.modal_batches_impl import (
    batches as _modal_batches_entrypoint,
)


def batches(cmd: str, batch_tag=None, num: int | None = None, tag: str = "default"):
    if cmd == "work":
        modal_function = modal.Function.lookup(_get_app_name(tag), "modal_batches_worker")
        for _ in range(int(num or 0)):
            modal_function.spawn()
    elif cmd == "submit-missing":
        batches_submitter(tag, batch_tag)
    elif cmd == "status":
        status()
    elif cmd == "collect":
        collect()
    elif cmd == "clean_up":
        clean_up()
    elif cmd == "stop":
        stop(tag)
    else:
        return _modal_batches_entrypoint(tag, cmd, batch_tag=batch_tag, num=num)


__all__ = [
    "_get_app_name",
    "_job_key",
    "_results_dict",
    "_submitted_dict",
    "app",
    "batches",
    "batches_collector",
    "batches_submitter",
    "clean_up",
    "collect",
    "modal_batch_deleter",
    "modal_batches_resubmitter",
    "modal_batches_worker",
    "modal",
    "status",
    "stop",
]

collect = batches_collector
