from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_iter_incremental_jobs_deduplicates_n_grid_and_skips_existing(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    monkeypatch.setattr(impl, "_BENCHMARK_FUNCTIONS", ("sphere", "ackley"))
    monkeypatch.setattr(impl, "enn_incremental_checkpoint_ns", lambda: (1, 3))
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
        return {
            "N": [1, 3],
            "add_seconds": [0.01, 0.02],
            "log_likelihood": [-2.0, -1.0],
            "_meta": {
                "D": 2,
                "function_name": function_name,
                "problem_seed": 17,
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

    monkeypatch.setattr(impl, "_BENCHMARK_FUNCTIONS", ("sphere",))
    monkeypatch.setattr(impl, "enn_incremental_checkpoint_ns", lambda: (1, 3, 10))
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

    monkeypatch.setattr(impl, "benchmark_enn_incremental_add_timing", fake_benchmark)
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


def test_enn_incremental_collect_writes_and_deletes(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl

    results = {
        "enn_incremental_D2_sphere_pseed17_nrep10_rep3_flat": (
            {
                "N": [1],
                "add_seconds": [0.01],
                "log_likelihood": [-2.0],
                "_meta": {
                    "D": 2,
                    "function_name": "sphere",
                    "problem_seed": 17,
                    "rep_index": 3,
                    "num_reps": 10,
                    "index_driver": "flat",
                },
            },
            2,
            "sphere",
            17,
            3,
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

    impl._collect("add_method-tag-x", tmp_path)

    dest = tmp_path / "enn_incremental_D2_sphere_pseed17_nrep10_rep3_flat.json"
    assert dest.exists()
    assert deleted == [(["enn_incremental_D2_sphere_pseed17_nrep10_rep3_flat"], "add_method-tag-x")]


def test_add_collect_should_overwrite_stale_complete_incremental_json(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_impl as impl
    from analysis.fitting_time.fitting_time_enn_incremental import (
        enn_incremental_checkpoint_ns,
    )

    chk = enn_incremental_checkpoint_ns()
    dest = impl.result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=3,
        num_reps=10,
        index_driver="flat",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "N": list(chk),
                "add_seconds": [0.01] * len(chk),
                "log_likelihood": [-9.0] * len(chk),
                "_meta": {
                    "D": 2,
                    "function_name": "sphere",
                    "problem_seed": 17,
                    "rep_index": 3,
                    "num_reps": 10,
                    "index_driver": "flat",
                },
            }
        )
    )
    fresh_rp = {
        "N": list(chk),
        "add_seconds": [0.99] * len(chk),
        "log_likelihood": [-1.0] * len(chk),
        "_meta": {
            "D": 2,
            "function_name": "sphere",
            "problem_seed": 17,
            "rep_index": 3,
            "num_reps": 10,
            "index_driver": "flat",
        },
    }
    results = {
        "enn_incremental_D2_sphere_pseed17_nrep10_rep3_flat": (
            fresh_rp,
            2,
            "sphere",
            17,
            3,
            10,
            "flat",
        ),
    }

    monkeypatch.setattr(impl, "_results_dict", lambda _tag: results)
    monkeypatch.setattr(
        impl.modal,
        "Function",
        SimpleNamespace(from_name=lambda *_a, **_k: SimpleNamespace(spawn=lambda *a, **k: None)),
    )

    impl._collect("add_method-tag-x", tmp_path)

    on_disk = json.loads(dest.read_text())
    assert on_disk["add_seconds"][0] == 0.99
