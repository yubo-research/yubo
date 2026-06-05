from __future__ import annotations

import json
from pathlib import Path

from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.fitting_time_enn_incremental import (
    EnnIncrementalIndexDriver,
    EnnIncrementalTimingResult,
)


def test_enn_incremental_payload_and_dest(tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    result = EnnIncrementalTimingResult(
        n=(1, 3),
        add_seconds=(0.1, 0.2),
        log_likelihood=(-1.0, -0.5),
        target="sphere",
        d=2,
        problem_seed=17,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )

    payload = impl.result_to_payload(result)
    assert payload == {
        "N": [1, 3],
        "add_seconds": [0.1, 0.2],
        "log_likelihood": [-1.0, -0.5],
        "_meta": {
            "D": 2,
            "function_name": "sphere",
            "problem_seed": 17,
            "index_driver": "flat",
        },
    }
    assert impl.result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=3,
        num_reps=10,
        index_driver="flat",
    ).name == ("enn_incremental_D2_sphere_pseed17_nrep10_rep3_flat.json")


def test_client_reexports_match_impl_payload_and_dest(tmp_path: Path):
    import experiments.modal_enn_incremental_batches_client as client
    import experiments.modal_enn_incremental_batches_impl as impl

    result = EnnIncrementalTimingResult(
        n=(1, 3),
        add_seconds=(0.1, 0.2),
        log_likelihood=(-1.0, -0.5),
        target="sphere",
        d=2,
        problem_seed=17,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    assert client.result_to_payload(result) == impl.result_to_payload(result)
    assert client.result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=3,
        num_reps=10,
        index_driver="flat",
    ) == impl.result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=3,
        num_reps=10,
        index_driver="flat",
    )


def test_pending_jobs_uses_shared_benchmark_functions(monkeypatch, tmp_path: Path):
    import dataclasses

    import experiments.enn_batch_job_params as batch_params
    import experiments.modal_enn_incremental_batches_common as common
    import experiments.modal_enn_incremental_batches_impl as impl

    original_shared = batch_params.enn_batch_shared_params

    def shared_sphere_only(**kwargs):
        shared = original_shared(**kwargs)
        return dataclasses.replace(shared, benchmark_functions=("sphere",))

    monkeypatch.setattr(common, "enn_batch_shared_params", shared_sphere_only)
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere", "ackley"),
    )

    jobs = list(impl._pending_jobs("add_method", tmp_path, "flat", 1, 2, 17))
    function_names = {job[1][1] for job in jobs}

    assert function_names == {"sphere"}


def test_iter_incremental_jobs_deduplicates_n_grid_and_skips_existing(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere", "ackley"),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.enn_batch_checkpoint_ns",
        lambda: (1, 3),
    )
    existing = impl.result_json_dest(
        tmp_path,
        d=2,
        function_name="ackley",
        problem_seed=17,
        rep_index=0,
        num_reps=2,
        index_driver="flat",
    )
    existing.parent.mkdir(parents=True, exist_ok=True)

    def _complete_payload(*, function_name: str, rep_index: int, num_reps: int) -> dict:
        data_seed = synthetic_benchmark_data_seed(
            function_name=function_name,
            problem_seed=17,
            rep_index=rep_index,
        )
        return {
            "N": [1, 3],
            "add_seconds": [0.01, 0.02],
            "log_likelihood": [-2.0, -1.0],
            "_meta": {
                "D": 2,
                "function_name": function_name,
                "problem_seed": 17,
                "data_seed": data_seed,
                "rep_index": rep_index,
                "num_reps": num_reps,
                "index_driver": "flat",
            },
        }

    existing.write_text(json.dumps(_complete_payload(function_name="ackley", rep_index=0, num_reps=2)))
    existing_1 = impl.result_json_dest(
        tmp_path,
        d=2,
        function_name="ackley",
        problem_seed=17,
        rep_index=1,
        num_reps=2,
        index_driver="flat",
    )
    existing_1.write_text(json.dumps(_complete_payload(function_name="ackley", rep_index=1, num_reps=2)))

    jobs = list(impl._iter_incremental_jobs(tmp_path, "flat", 2, 2, 17))

    assert jobs == [
        (
            "enn_incremental_D2_sphere_pseed17_nrep2_rep0_flat",
            (2, "sphere", 17, 0, 2, "flat"),
        ),
        (
            "enn_incremental_D2_sphere_pseed17_nrep2_rep1_flat",
            (2, "sphere", 17, 1, 2, "flat"),
        ),
    ]


def test_iter_incremental_jobs_resubmits_incomplete_existing_json(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere",),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.enn_batch_checkpoint_ns",
        lambda: (1, 3, 10),
    )
    existing = impl.result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
    )
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps({"N": [1, 3], "add_seconds": [0.01, 0.02], "log_likelihood": [-2.0, -1.0]}))

    jobs = list(impl._iter_incremental_jobs(tmp_path, "flat", 1, 2, 17))

    assert jobs == [
        (
            "enn_incremental_D2_sphere_pseed17_nrep1_rep0_flat",
            (2, "sphere", 17, 0, 1, "flat"),
        )
    ]


def test_enn_incremental_worker_writes_result(monkeypatch):
    import experiments.modal_enn_incremental_batches_impl as impl

    store = {}
    captured = {}

    def fake_benchmark(*, D, function_name, problem_seed, index_driver):
        captured["args"] = (D, function_name, problem_seed, index_driver)
        return EnnIncrementalTimingResult(
            n=(1,),
            add_seconds=(0.01,),
            log_likelihood=(-2.0,),
            target=function_name,
            d=D,
            problem_seed=problem_seed,
            index_driver=index_driver,
        )

    import experiments.modal_enn_incremental_batch_worker as worker_mod

    monkeypatch.setattr(worker_mod, "benchmark_enn_incremental_add_timing", fake_benchmark)
    monkeypatch.setattr(impl, "_results_dict", lambda _tag: store)

    impl.enn_incremental_batch_worker.info.raw_f(("add_method-tag-x", 2, "sphere", 17, 3, 10, "hnsw"))

    data_seed = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=3)
    assert captured["args"] == (2, "sphere", data_seed, EnnIncrementalIndexDriver.HNSW)
    payload, d, fn, problem_seed, rep_index, num_reps, driver = store["enn_incremental_D2_sphere_pseed17_nrep10_rep3_hnsw"]
    assert (d, fn, problem_seed, rep_index, num_reps, driver) == (
        2,
        "sphere",
        17,
        3,
        10,
        "hnsw",
    )
    assert payload["_meta"]["index_driver"] == "hnsw"
    assert payload["_meta"]["data_seed"] == data_seed
