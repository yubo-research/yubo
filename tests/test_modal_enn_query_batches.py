from __future__ import annotations

from pathlib import Path

from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from analysis.fitting_time.fitting_time_enn_query import EnnQueryTimingResult


def test_query_payload_and_dest(tmp_path: Path):
    import experiments.modal_enn_query_batches as query_batches

    result = EnnQueryTimingResult(
        n=(1, 3),
        query_seconds=(0.1, 0.2),
        query_seconds_per_point=(0.001, 0.002),
        num_query_points=100,
        target="sphere",
        d=2,
        problem_seed=17,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    payload = query_batches.query_result_to_payload(
        result,
        problem_seed=17,
        data_seed=99,
        rep_index=3,
        num_reps=10,
    )
    assert payload["N"] == [1, 3]
    assert payload["query_seconds"] == [0.1, 0.2]
    assert payload["query_seconds_per_point"] == [0.001, 0.002]
    assert payload["_meta"]["num_query_points"] == 100
    dest = query_batches.query_result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=3,
        num_reps=10,
        index_driver="flat",
        normalize_function_name=lambda x: x,
    )
    assert dest.name == "enn_query_D2_sphere_pseed17_nrep10_rep3_flat.json"


def test_query_worker_writes_result(monkeypatch):
    import experiments.modal_enn_incremental_batches_impl as impl

    store = {}
    captured = {}

    def fake_benchmark(*, D, function_name, problem_seed, index_driver):
        captured["args"] = (D, function_name, problem_seed, index_driver)
        return EnnQueryTimingResult(
            n=(1,),
            query_seconds=(0.1,),
            query_seconds_per_point=(0.001,),
            num_query_points=100,
            target=function_name,
            d=D,
            problem_seed=problem_seed,
            index_driver=index_driver,
        )

    import experiments.modal_enn_incremental_batch_worker as worker_mod

    monkeypatch.setattr(worker_mod, "benchmark_enn_query_timing", fake_benchmark)
    monkeypatch.setattr(impl, "_results_dict", lambda _tag: store)

    impl.enn_incremental_batch_worker.info.raw_f(("query-tag-x", 2, "sphere", 17, 3, 10, "hnsw"))

    data_seed = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=3)
    assert captured["args"] == (2, "sphere", data_seed, EnnIncrementalIndexDriver.HNSW)
    key = "enn_query_D2_sphere_pseed17_nrep10_rep3_hnsw"
    payload, d, fn, problem_seed, rep_index, num_reps, driver = store[key]
    assert payload["query_seconds_per_point"] == [0.001]
    assert payload["_meta"]["num_query_points"] == 100
    assert (d, fn, problem_seed, rep_index, num_reps, driver) == (
        2,
        "sphere",
        17,
        3,
        10,
        "hnsw",
    )
