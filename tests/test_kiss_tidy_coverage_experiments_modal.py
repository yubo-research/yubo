from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_full_opt import EnnFullOptTimingResult
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from experiments.enn_batch_job_params import (
    EnnBatchSharedParams,
    enn_batch_rep_meta_matches,
    enn_batch_shared_params,
    normalize_index_driver,
    validate_enn_batch_scalars,
)
from experiments.modal_dict_utils import delete_keys_from_dicts
from experiments.modal_enn_fit_batches import fit_job_key, iter_fit_jobs
from experiments.modal_enn_fit_ind_batches import fit_ind_job_key, iter_fit_ind_jobs
from experiments.modal_enn_fit_ind_batches_json import fit_ind_result_json_complete
from experiments.modal_enn_full_opt_batches import (
    env_tag_slug,
    full_opt_job_key,
    full_opt_result_to_payload,
    iter_full_opt_jobs,
)
from experiments.modal_enn_full_opt_batches_json import full_opt_meta_matches
from experiments.modal_enn_incremental_batch_worker import dispatch_enn_incremental_batch_worker
from experiments.modal_enn_incremental_batches_common import (
    collect,
    dict_names_for_tag,
    experiment_type_from_tag,
    get_app_name,
    iter_incremental_jobs,
    iter_index_drivers,
    job_key,
    pending_jobs,
    results_dict,
    run_command,
    status,
    submit_missing,
    submitted_dict,
    write_json,
)
from experiments.modal_enn_incremental_batches_common import (
    iter_fit_ind_jobs as common_iter_fit_ind_jobs,
)
from experiments.modal_enn_incremental_batches_common import (
    iter_fit_jobs as common_iter_fit_jobs,
)
from experiments.modal_enn_incremental_batches_common import (
    iter_full_opt_jobs as common_iter_full_opt_jobs,
)
from experiments.modal_enn_incremental_batches_common import (
    iter_query_jobs as common_iter_query_jobs,
)
from experiments.modal_enn_incremental_batches_impl import (
    batches as enn_impl_batches,
)
from experiments.modal_enn_incremental_batches_impl import (
    enn_incremental_batch_deleter,
)
from experiments.modal_enn_incremental_batches_json import add_meta_matches, result_json_complete
from experiments.modal_enn_query_batches import (
    iter_query_jobs,
    query_job_key,
    query_result_json_complete,
)
from experiments.modal_enn_series_batches import iter_replicate_series_jobs
from experiments.modal_result_collect import (
    ModalResultParts,
    gen_jobs_from_configs,
    iter_modal_results_for_collect,
    unpack_modal_result_value,
)


def test_kiss_tidy_experiments_modal_batch_and_common(monkeypatch, tmp_path):
    assert normalize_index_driver("flat") is EnnIncrementalIndexDriver.FLAT
    d_i, nr = validate_enn_batch_scalars(num_reps=1, d=2)
    assert d_i == 2 and nr == 1
    shared = enn_batch_shared_params(num_reps=1, d=2, problem_seed=0)
    assert isinstance(shared, EnnBatchSharedParams)
    meta = {
        "D": 2,
        "function_name": "sphere",
        "problem_seed": 0,
        "data_seed": 0,
        "rep_index": 0,
        "num_reps": 1,
        "index_driver": "flat",
    }
    from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed

    meta["data_seed"] = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=0, rep_index=0)
    assert enn_batch_rep_meta_matches(
        meta,
        d=2,
        function_name="sphere",
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    assert add_meta_matches(
        meta,
        d=2,
        function_name="sphere",
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
    )

    d = {"a": 1, "b": 2}
    delete_keys_from_dicts(["a"], d)
    assert "a" not in d and "b" in d

    assert get_app_name("tag") == "yubo-enn-incremental-tag"
    assert dict_names_for_tag("t") == ("enn_incremental_results_t", "enn_incremental_submitted_t")
    assert experiment_type_from_tag("add_method-x") == "add_method"
    assert env_tag_slug("f:sphere-2d") == "f_sphere-2d"

    from modal_timing_sweep_test_support import FakeResultsDict

    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Dict.from_name",
        lambda name, create_if_missing=False: FakeResultsDict(),
    )
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Function.from_name",
        lambda *a, **k: SimpleNamespace(spawn=lambda *sa, **sk: None, spawn_map=lambda *sa, **sk: None),
    )
    results_dict("t")
    submitted_dict("t")
    assert list(iter_index_drivers("flat")) == [EnnIncrementalIndexDriver.FLAT]

    k = job_key(d=2, function_name="sphere", problem_seed=0, rep_index=0, num_reps=1, index_driver="flat")
    assert "sphere" in k
    dest = tmp_path / "out.json"
    write_json(dest, {"ok": True})
    assert dest.read_text().startswith("{")

    parts = unpack_modal_result_value(("t", "l", "c", None))
    assert isinstance(parts, ModalResultParts)

    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere",),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.enn_batch_checkpoint_ns",
        lambda: (1,),
    )
    jobs = list(pending_jobs("add_method", tmp_path, "flat", 1, 2, 0))
    assert jobs
    assert list(iter_incremental_jobs(tmp_path, "flat", 1, 2, 0))

    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common._fit_batches.iter_fit_jobs",
        lambda *a, **k: iter([]),
    )
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common._fit_ind_batches.iter_fit_ind_jobs",
        lambda *a, **k: iter([]),
    )
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common._query_batches.iter_query_jobs",
        lambda *a, **k: iter([]),
    )
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common._full_opt_batches.iter_full_opt_jobs",
        lambda *a, **k: iter([]),
    )
    assert list(common_iter_fit_jobs(tmp_path, "flat", 1, 2, 0)) == []
    assert list(common_iter_fit_ind_jobs(tmp_path, "flat", 1, 2, 0)) == []
    assert list(common_iter_query_jobs(tmp_path, "flat", 1, 2, 0)) == []
    assert list(common_iter_full_opt_jobs(tmp_path, "flat", 1, 2, 0)) == []

    submit_missing("add_method-x", tmp_path, "flat", 1, 2, 0)
    collect("add_method-x", tmp_path)
    status("add_method-x")
    run_command("add_method-x", "status", output_dir=str(tmp_path))


def test_kiss_tidy_experiments_modal_job_keys_and_json(monkeypatch, tmp_path):
    from modal_timing_sweep_test_support import FakeResultsDict

    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Dict.from_name",
        lambda *a, **k: FakeResultsDict(),
    )
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_common.modal.Function.from_name",
        lambda *a, **k: SimpleNamespace(spawn=lambda *sa, **sk: None, spawn_map=lambda *sa, **sk: None),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere",),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.enn_batch_checkpoint_ns",
        lambda: (1,),
    )
    meta = {
        "D": 2,
        "function_name": "sphere",
        "problem_seed": 0,
        "data_seed": 0,
        "rep_index": 0,
        "num_reps": 1,
        "index_driver": "flat",
    }
    from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed

    meta["data_seed"] = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=0, rep_index=0)
    fk = fit_job_key(
        d=2,
        function_name="sphere",
        n=1,
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    assert "enn_fit" in fk

    def _drv_fn(s):
        return (EnnIncrementalIndexDriver.FLAT,)

    fit_jobs = list(
        iter_fit_jobs(
            tmp_path,
            "flat",
            1,
            2,
            0,
            iter_index_drivers=_drv_fn,
            normalize_function_name=normalize_benchmark_function_name,
        )
    )
    assert fit_jobs

    fik = fit_ind_job_key(
        d=2,
        function_name="sphere",
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    assert "enn_fit_ind" in fik
    assert list(
        iter_fit_ind_jobs(
            tmp_path,
            "flat",
            1,
            2,
            0,
            iter_index_drivers=_drv_fn,
            normalize_function_name=normalize_benchmark_function_name,
        )
    )

    qk = query_job_key(
        d=2,
        function_name="sphere",
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    assert "enn_query" in qk
    assert list(
        iter_query_jobs(
            tmp_path,
            "flat",
            1,
            2,
            0,
            iter_index_drivers=_drv_fn,
            normalize_function_name=normalize_benchmark_function_name,
        )
    )

    fok = full_opt_job_key(
        env_tag="f:sphere-2d",
        problem_seed=18,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
    )
    assert "enn_full_opt" in fok
    assert list(
        iter_full_opt_jobs(
            tmp_path,
            "flat",
            1,
            iter_index_drivers=lambda s: (EnnIncrementalIndexDriver.FLAT,),
            env_tags=("f:sphere-2d",),
        )
    )

    result = EnnFullOptTimingResult(
        n=(1,),
        proposal_elapsed_seconds=(0.1,),
        env_tag="f:sphere-2d",
        opt_name="turbo-enn-fit-ucb",
        index_driver=EnnIncrementalIndexDriver.FLAT,
        problem_seed=18,
        rep_index=0,
        num_rounds=1,
        stop_reason="completed",
    )
    payload = full_opt_result_to_payload(result, num_reps=1)
    assert payload["_meta"]["env_tag"] == "f:sphere-2d"

    inc_dest = tmp_path / "inc.json"
    inc_dest.write_text(
        json.dumps(
            {
                "N": [1],
                "add_seconds": [0.1],
                "log_likelihood": [0.0],
                "_meta": meta,
            }
        )
    )
    assert result_json_complete(
        inc_dest,
        (1,),
        d=2,
        function_name="sphere",
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
    )

    fit_ind_dest = tmp_path / "fit_ind.json"
    fit_ind_dest.write_text(
        json.dumps(
            {
                "N": [1],
                "fit_seconds": [0.1],
                "log_likelihood": [0.0],
                "_meta": meta,
            }
        )
    )
    assert fit_ind_result_json_complete(
        fit_ind_dest,
        (1,),
        d=2,
        function_name="sphere",
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
    )

    assert full_opt_meta_matches(
        payload["_meta"],
        env_tag="f:sphere-2d",
        problem_seed=18,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        opt_name="turbo-enn-fit-ucb",
    )

    q_dest = tmp_path / "query.json"
    q_dest.write_text(
        json.dumps(
            {
                "N": [1],
                "query_seconds": [0.1],
                "query_seconds_per_point": [0.05],
                "_meta": meta,
            }
        )
    )
    query_result_json_complete(
        q_dest,
        (1,),
        d=2,
        function_name="sphere",
        problem_seed=0,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )

    series_jobs = list(
        iter_replicate_series_jobs(
            tmp_path,
            "flat",
            1,
            2,
            0,
            iter_index_drivers=lambda s: (EnnIncrementalIndexDriver.FLAT,),
            normalize_function_name=normalize_benchmark_function_name,
            result_json_dest=lambda *a, **k: tmp_path / "x.json",
            result_json_complete=lambda *a, **k: False,
            job_key=lambda **kw: "jk",
        )
    )
    assert series_jobs

    with pytest.raises(ValueError, match="expected job with tag"):
        dispatch_enn_incremental_batch_worker(
            (),
            experiment_type_from_tag=experiment_type_from_tag,
            job_key=job_key,
            result_to_payload=lambda *a, **k: {},
            results_dict=lambda tag: results_dict(tag),
        )

    enn_incremental_batch_deleter.get_raw_f()([], "add_method-t1")
    monkeypatch.setattr(
        "experiments.modal_enn_incremental_batches_impl.common.run_command",
        lambda *a, **k: None,
    )
    enn_impl_batches("add_method-t1", "status")

    class _Res:
        def len(self):
            return 0

        def items(self):
            return []

    assert list(gen_jobs_from_configs("bt", [], lambda c: [], lambda t: True, lambda bt, t: "k")) == []

    collected = iter_modal_results_for_collect(
        _Res(),
        post_process=lambda *a, **k: None,
        data_is_done=lambda t: True,
        gotitem_log=lambda *a, **k: None,
    )
    assert collected == set()
