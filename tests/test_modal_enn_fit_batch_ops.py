from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.modal_enn_fit_batches as fit_batches
from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name


def _run_submitter(monkeypatch, *, force: bool) -> list[tuple[str, ...]]:
    import experiments.modal_enn_incremental_batches_impl as impl

    submitted: dict[str, bool] = {"stuck_key": True}
    spawned: list[tuple[str, ...]] = []

    worker = SimpleNamespace(spawn_map=spawned.extend)

    monkeypatch.setattr(impl, "_submitted_dict", lambda _tag: submitted)
    monkeypatch.setattr(
        impl.modal,
        "Function",
        SimpleNamespace(from_name=lambda *_a, **_k: worker),
    )
    impl.enn_incremental_batch_submitter.info.raw_f(
        [("stuck_key", (2, "sphere", 3, 17, 0, 1, "flat"))],
        "fit_method-tag-x",
        force,
    )
    return spawned


def test_submitter_force_retries_submitted_key(monkeypatch):
    assert len(_run_submitter(monkeypatch, force=True)) == 1


def test_submitter_skips_submitted_without_force(monkeypatch):
    assert _run_submitter(monkeypatch, force=False) == []


def test_collect_overwrites_stale_complete_fit_json(monkeypatch, tmp_path: Path):
    import experiments.modal_enn_incremental_batches_common as common
    import experiments.modal_enn_incremental_batches_impl as impl

    dest = fit_batches.fit_result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        n=3,
        problem_seed=17,
        rep_index=2,
        num_reps=10,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "N": 3,
                "fit_seconds": 0.01,
                "log_likelihood": -1.0,
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
        )
    )
    fresh_payload = {
        "N": 3,
        "fit_seconds": 0.99,
        "log_likelihood": -0.9,
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
    results = {
        "enn_fit_D2_sphere_N3_pseed17_nrep10_rep2_flat": (
            fresh_payload,
            2,
            "sphere",
            3,
            17,
            2,
            10,
            "flat",
        ),
    }

    monkeypatch.setattr(common, "results_dict", lambda _tag: results)
    monkeypatch.setattr(
        common.modal,
        "Function",
        SimpleNamespace(from_name=lambda *_args, **_kwargs: SimpleNamespace(spawn=lambda *a, **k: None)),
    )

    impl._collect("fit_method-tag-x", tmp_path)

    on_disk = json.loads(dest.read_text())
    assert on_disk["fit_seconds"] == 0.99


def test_submit_missing_reported_count_matches_spawned_jobs(monkeypatch, tmp_path: Path, capsys):
    import experiments.modal_enn_incremental_batches_common as common
    import experiments.modal_enn_incremental_batches_impl as impl

    submitted: dict[str, bool] = {"enn_fit_D2_sphere_N3_pseed17_nrep10_rep0_flat": True}
    spawned_todos: list[tuple] = []

    def _spawn(batch, tag, force=False):
        spawned_todos.append((list(batch), tag, force))

    submitter = SimpleNamespace(spawn=_spawn)

    monkeypatch.setattr(common, "submitted_dict", lambda _tag: submitted)
    monkeypatch.setattr(
        common.modal,
        "Function",
        SimpleNamespace(
            from_name=lambda _app, name, **_k: submitter if name == "enn_incremental_batch_submitter" else SimpleNamespace(spawn_map=lambda *_a, **_k: None)
        ),
    )
    monkeypatch.setattr(
        common,
        "iter_fit_jobs",
        lambda *_a, **_k: [
            (
                "enn_fit_D2_sphere_N3_pseed17_nrep10_rep0_flat",
                (2, "sphere", 3, 17, 0, 10, "flat"),
            ),
            (
                "enn_fit_D2_sphere_N3_pseed17_nrep10_rep1_flat",
                (2, "sphere", 3, 17, 1, 10, "flat"),
            ),
        ],
    )

    impl._submit_missing("fit_method-t", tmp_path, "flat", 10, 2, 17)

    out = capsys.readouterr().out
    assert len(spawned_todos) == 1
    assert len(spawned_todos[0][0]) == 1
    assert "submitted 1 ENN batch jobs" in out


def _fit_json_dest_with_meta(tmp_path: Path, meta: dict) -> Path:
    dest = fit_batches.fit_result_json_dest(
        tmp_path,
        d=2,
        function_name="sphere",
        n=3,
        problem_seed=17,
        rep_index=0,
        num_reps=10,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"N": 3, "fit_seconds": 0.01, "log_likelihood": -1.0, "_meta": meta}))
    return dest


def _assert_fit_json_incomplete(dest: Path) -> None:
    assert not fit_batches.fit_result_json_complete(
        dest,
        3,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=0,
        num_reps=10,
        index_driver="flat",
        normalize_function_name=normalize_benchmark_function_name,
    )


@pytest.mark.parametrize(
    "meta_patch",
    [
        {"rep_index": 99},
        {"data_seed_offset": 1},
    ],
)
def test_fit_result_json_complete_rejects_bad_meta(tmp_path: Path, meta_patch: dict):
    expected_data_seed = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=0)
    meta = {
        "D": 2,
        "function_name": "sphere",
        "problem_seed": 17,
        "data_seed": expected_data_seed + int(meta_patch.get("data_seed_offset", 0)),
        "rep_index": int(meta_patch.get("rep_index", 0)),
        "num_reps": 10,
        "index_driver": "flat",
    }
    dest = _fit_json_dest_with_meta(tmp_path, meta)
    _assert_fit_json_incomplete(dest)
