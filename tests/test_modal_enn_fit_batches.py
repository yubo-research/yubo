from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.modal_enn_fit_batches as fit_batches
from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from analysis.fitting_time.fitting_time_enn_fit import EnnFitTimingResult
from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver


def test_fit_payload_and_dest(tmp_path: Path):
    result = EnnFitTimingResult(
        n=3,
        fit_seconds=0.2,
        target="sphere",
        d=2,
        problem_seed=17,
        data_seed=99,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    payload = fit_batches.fit_result_to_payload(
        result,
        problem_seed=17,
        data_seed=99,
        rep_index=3,
        num_reps=10,
    )
    assert payload == {
        "N": 3,
        "fit_seconds": 0.2,
        "_meta": {
            "D": 2,
            "function_name": "sphere",
            "problem_seed": 17,
            "data_seed": 99,
            "rep_index": 3,
            "num_reps": 10,
            "index_driver": "flat",
        },
    }
    dest = fit_batches.fit_result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        n=3,
        problem_seed=17,
        rep_index=2,
        num_reps=5,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    assert dest.name == "enn_fit_D2_sphere_N3_pseed17_nrep5_rep2_flat.json"


def test_iter_fit_jobs_skips_complete(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere", "ackley"),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.enn_batch_checkpoint_ns",
        lambda: (3,),
    )
    ack0 = fit_batches.fit_result_json_dest(
        tmp_path,
        d=2,
        function_name="ackley",
        n=3,
        problem_seed=17,
        rep_index=0,
        num_reps=2,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    ack0.parent.mkdir(parents=True, exist_ok=True)
    ack0_seed = synthetic_benchmark_data_seed(function_name="ackley", problem_seed=17, rep_index=0)
    ack0.write_text(
        json.dumps(
            {
                "N": 3,
                "fit_seconds": 0.03,
                "_meta": {
                    "D": 2,
                    "function_name": "ackley",
                    "problem_seed": 17,
                    "data_seed": ack0_seed,
                    "rep_index": 0,
                    "num_reps": 2,
                    "index_driver": "flat",
                },
            }
        )
    )
    ack1 = fit_batches.fit_result_json_dest(
        tmp_path,
        d=2,
        function_name="ackley",
        n=3,
        problem_seed=17,
        rep_index=1,
        num_reps=2,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    ack1_seed = synthetic_benchmark_data_seed(function_name="ackley", problem_seed=17, rep_index=1)
    ack1.write_text(
        json.dumps(
            {
                "N": 3,
                "fit_seconds": 0.04,
                "_meta": {
                    "D": 2,
                    "function_name": "ackley",
                    "problem_seed": 17,
                    "data_seed": ack1_seed,
                    "rep_index": 1,
                    "num_reps": 2,
                    "index_driver": "flat",
                },
            }
        )
    )

    jobs = list(impl._iter_fit_jobs(tmp_path, "flat", 2, 2, 17))

    assert jobs == [
        (
            "enn_fit_D2_sphere_N3_pseed17_nrep2_rep0_flat",
            (2, "sphere", 3, 17, 0, 2, "flat"),
        ),
        (
            "enn_fit_D2_sphere_N3_pseed17_nrep2_rep1_flat",
            (2, "sphere", 3, 17, 1, 2, "flat"),
        ),
    ]


def test_iter_fit_jobs_index_driver_all_yields_flat_and_hnsw(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere",),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.enn_batch_checkpoint_ns",
        lambda: (3,),
    )
    jobs = list(impl._iter_fit_jobs(tmp_path, "all", 1, 2, 17))
    drivers = {job[1][6] for job in jobs}
    assert drivers == {"flat", "hnsw"}
    assert len(jobs) == 2


def test_iter_fit_jobs_resubmits_when_json_incomplete(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    monkeypatch.setattr(
        "experiments.enn_batch_job_params.ENN_BATCH_BENCHMARK_FUNCTIONS",
        ("sphere",),
    )
    monkeypatch.setattr(
        "experiments.enn_batch_job_params.enn_batch_checkpoint_ns",
        lambda: (10,),
    )
    existing = fit_batches.fit_result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        n=10,
        problem_seed=17,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text(json.dumps({"N": 9, "fit_seconds": 0.01}))

    jobs = list(impl._iter_fit_jobs(tmp_path, "flat", 1, 2, 17))

    assert jobs == [
        (
            "enn_fit_D2_sphere_N10_pseed17_nrep1_rep0_flat",
            (2, "sphere", 10, 17, 0, 1, "flat"),
        ),
    ]


def test_fit_worker_writes_result_via_incremental_modal_fn(monkeypatch):
    import experiments.modal_enn_incremental_batches_impl as impl

    store = {}
    captured = {}

    def fake_bm(*args, **kwargs):
        captured["kwargs"] = kwargs
        return EnnFitTimingResult(
            n=int(kwargs["n"]),
            fit_seconds=0.5,
            target=kwargs["function_name"],
            d=int(kwargs["D"]),
            problem_seed=int(kwargs["problem_seed"]),
            data_seed=int(kwargs["data_seed"]),
            index_driver=kwargs["index_driver"],
        )

    monkeypatch.setattr(impl, "benchmark_enn_fit_timing", fake_bm)
    monkeypatch.setattr(impl, "_results_dict", lambda _tag: store)

    impl.enn_incremental_batch_worker.info.raw_f(("fit_method-tag-fit", 2, "sphere", 3, 17, 1, 10, "hnsw"))

    ds = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=1)
    assert captured["kwargs"] == {
        "D": 2,
        "function_name": "sphere",
        "data_seed": ds,
        "problem_seed": 17,
        "n": 3,
        "index_driver": EnnIncrementalIndexDriver.HNSW,
    }

    payload, d, fn, n, problem_seed, rep_index, num_reps, drv = store["enn_fit_D2_sphere_N3_pseed17_nrep10_rep1_hnsw"]
    assert (d, fn, int(n), problem_seed, rep_index, num_reps, drv) == (
        2,
        "sphere",
        3,
        17,
        1,
        10,
        "hnsw",
    )
    assert payload["_meta"]["index_driver"] == "hnsw"
    assert payload["_meta"]["data_seed"] == ds
    assert payload["N"] == 3


def test_pending_jobs_rejects_unknown_kind(tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    with pytest.raises(ValueError, match="unknown job kind"):
        list(
            impl._pending_jobs(
                "bogus",
                tmp_path,
                "flat",
                1,
                2,
                17,
            )
        )


def test_worker_rejects_add_method_job_with_fit_method_tag():
    import experiments.modal_enn_incremental_batches_impl as impl

    with pytest.raises(ValueError, match="add_method job expected 7"):
        impl.enn_incremental_batch_worker.info.raw_f(("add_method-t", 2, "sphere", 17, 0, 10, "flat", "hnsw"))


def test_worker_rejects_fit_method_job_with_wrong_len():
    import experiments.modal_enn_incremental_batches_impl as impl

    with pytest.raises(ValueError, match="fit_method job expected 8"):
        impl.enn_incremental_batch_worker.info.raw_f(("fit_method-t", 2, "sphere", 3, 17, 1, 10))


def test_enn_fit_collect_dict_payload_writes(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    payload = {
        "N": 3,
        "fit_seconds": 0.42,
        "_meta": {
            "D": 2,
            "function_name": "sphere",
            "problem_seed": 17,
            "data_seed": 99,
            "rep_index": 2,
            "num_reps": 10,
            "index_driver": "flat",
        },
    }
    results = {"enn_fit_D2_sphere_N3_pseed17_nrep10_rep2_flat": payload}
    deleted = []

    class _Func:
        def spawn(self, keys, tag):
            deleted.append((list(keys), tag))

    monkeypatch.setattr(impl, "_results_dict", lambda _tag: results)
    monkeypatch.setattr(
        impl.modal,
        "Function",
        SimpleNamespace(from_name=lambda *_args, **_kwargs: _Func()),
    )

    impl._collect("fit_method-tag-x", tmp_path)

    dest = tmp_path / "enn_fit_D2_sphere_N3_pseed17_nrep10_rep2_flat.json"
    assert dest.exists()
    assert json.loads(dest.read_text()) == payload
    assert deleted == [(["enn_fit_D2_sphere_N3_pseed17_nrep10_rep2_flat"], "fit_method-tag-x")]


def test_fit_worker_writes_result_real_benchmark(monkeypatch):
    import experiments.modal_enn_incremental_batches_impl as impl

    store = {}
    monkeypatch.setattr(impl, "_results_dict", lambda _tag: store)

    impl.enn_incremental_batch_worker.info.raw_f(("fit_method-real", 2, "sphere", 1, 17, 0, 1, "flat"))

    key = "enn_fit_D2_sphere_N1_pseed17_nrep1_rep0_flat"
    assert key in store
    payload, d, fn, n, problem_seed, rep_index, num_reps, drv = store[key]
    assert (d, fn, int(n), problem_seed, rep_index, num_reps, drv) == (
        2,
        "sphere",
        1,
        17,
        0,
        1,
        "flat",
    )
    assert payload["N"] == 1
    assert payload["fit_seconds"] > 0.0
    assert payload["_meta"]["data_seed"] == synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=0)


def test_fit_result_json_complete_requires_meta(tmp_path: Path):
    dest = fit_batches.fit_result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        n=3,
        problem_seed=17,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"N": 3, "fit_seconds": 0.01}))

    assert not fit_batches.fit_result_json_complete(
        dest,
        3,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=0,
        num_reps=1,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )


def test_enn_fit_collect_writes_and_deletes(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    results = {
        "enn_fit_D2_sphere_N3_pseed17_nrep10_rep2_flat": (
            {
                "N": 3,
                "fit_seconds": 0.42,
                "_meta": {
                    "D": 2,
                    "function_name": "sphere",
                    "problem_seed": 17,
                    "rep_index": 2,
                    "num_reps": 10,
                    "index_driver": "flat",
                },
            },
            2,
            "sphere",
            3,
            17,
            2,
            10,
            "flat",
        ),
    }
    deleted = []

    class _Func:
        def spawn(self, keys, tag):
            deleted.append((list(keys), tag))

    monkeypatch.setattr(impl, "_results_dict", lambda _tag: results)
    monkeypatch.setattr(
        impl.modal,
        "Function",
        SimpleNamespace(from_name=lambda *_args, **_kwargs: _Func()),
    )

    impl._collect("fit_method-tag-x", tmp_path)

    dest = tmp_path / "enn_fit_D2_sphere_N3_pseed17_nrep10_rep2_flat.json"
    assert dest.exists()
    assert deleted == [(["enn_fit_D2_sphere_N3_pseed17_nrep10_rep2_flat"], "fit_method-tag-x")]
