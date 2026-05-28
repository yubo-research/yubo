"""Async Modal batch workflow for ENN add-timing and fit-timing experiments."""

from __future__ import annotations

import os

import modal

from experiments import modal_enn_incremental_batch_worker as _batch_worker
from experiments import modal_enn_incremental_batches_common as common
from experiments.enn_batch_job_params import normalize_index_driver
from experiments.modal_dict_utils import delete_keys_from_dicts
from experiments.modal_image import mk_image

_TAG = os.environ.get("MODAL_TAG", "add_method-default")
_modal_image = mk_image(_TAG)

_get_app_name = common.get_app_name
_experiment_type_from_tag = common.experiment_type_from_tag
_results_dict = common.results_dict
_submitted_dict = common.submitted_dict
_iter_index_drivers = common.iter_index_drivers
_job_key = common.job_key
result_json_dest = common.result_json_dest
result_to_payload = common.result_to_payload
_pending_jobs = common.pending_jobs
_iter_incremental_jobs = common.iter_incremental_jobs
_iter_fit_jobs = common.iter_fit_jobs
_iter_fit_ind_jobs = common.iter_fit_ind_jobs
_iter_query_jobs = common.iter_query_jobs
_iter_full_opt_jobs = common.iter_full_opt_jobs
_submit_missing = common.submit_missing
_collect = common.collect
status = common.status
clean_up = common.clean_up
stop = common.stop

app = modal.App(name=_get_app_name(_TAG))


@app.function(
    image=_modal_image,
    max_containers=100,
    timeout=12 * 60 * 60,
    memory=4 * 1024,
    cpu=1.0,
)
def enn_incremental_batch_worker(job):
    _batch_worker.dispatch_enn_incremental_batch_worker(
        job,
        experiment_type_from_tag=_experiment_type_from_tag,
        job_key=_job_key,
        result_to_payload=result_to_payload,
        results_dict=_results_dict,
    )


@app.function(
    image=_modal_image,
    max_containers=100,
    timeout=5 * 60 * 60,
    memory=4 * 1024,
    cpu=1.0,
)
def enn_full_optimization_batch_worker(job):
    _batch_worker.dispatch_enn_incremental_batch_worker(
        job,
        experiment_type_from_tag=_experiment_type_from_tag,
        job_key=_job_key,
        result_to_payload=result_to_payload,
        results_dict=_results_dict,
    )


@app.function(image=_modal_image, max_containers=10, timeout=60 * 60)
def enn_incremental_batch_submitter(batch_of_jobs, tag: str, force: bool = False):
    submitted = _submitted_dict(tag)
    todo = []
    for key, job in batch_of_jobs:
        if (not force) and (key in submitted):
            continue
        submitted[key] = True
        todo.append((tag, *job))
    print(f"TODO: {len(todo)}")
    if todo:
        worker_fn = "enn_full_optimization_batch_worker" if _experiment_type_from_tag(tag) == "full_optimization" else "enn_incremental_batch_worker"
        modal.Function.from_name(_get_app_name(tag), worker_fn).spawn_map(todo)


@app.function(image=_modal_image, max_containers=1, timeout=60 * 60)
def enn_incremental_batch_deleter(keys, tag: str):
    delete_keys_from_dicts(keys, _results_dict(tag), _submitted_dict(tag))


@app.local_entrypoint()
def batches(
    tag: str,
    cmd: str,
    output_dir: str = "results/enn_incremental",
    index_driver: str = "all",
    num_reps: int = 10,
    d: int = 10,
    problem_seed: int = 17,
):
    common.run_command(
        tag,
        cmd,
        output_dir=output_dir,
        index_driver=index_driver,
        num_reps=num_reps,
        d=d,
        problem_seed=problem_seed,
    )


__all__ = [
    "_collect",
    "_experiment_type_from_tag",
    "_get_app_name",
    "_iter_fit_ind_jobs",
    "_iter_fit_jobs",
    "_iter_incremental_jobs",
    "_iter_query_jobs",
    "_iter_full_opt_jobs",
    "enn_full_optimization_batch_worker",
    "_job_key",
    "normalize_index_driver",
    "_results_dict",
    "_submitted_dict",
    "_submit_missing",
    "_pending_jobs",
    "app",
    "batches",
    "clean_up",
    "enn_incremental_batch_deleter",
    "enn_incremental_batch_submitter",
    "enn_incremental_batch_worker",
    "result_json_dest",
    "result_to_payload",
    "status",
    "stop",
]
