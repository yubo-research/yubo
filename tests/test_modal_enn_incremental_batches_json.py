from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.modal_enn_incremental_batches_json as add_json
from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed


def _write_add_json(dest: Path, chk: tuple[int, ...], meta: dict) -> None:
    dest.write_text(
        json.dumps(
            {
                "N": list(chk),
                "add_seconds": [0.01] * len(chk),
                "log_likelihood": [-1.0] * len(chk),
                "_meta": meta,
            }
        )
    )


def _assert_add_json_incomplete(dest: Path, chk: tuple[int, ...]) -> None:
    assert not add_json.result_json_complete(
        dest,
        chk,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=0,
        num_reps=10,
        index_driver="flat",
    )


@pytest.mark.parametrize(
    "meta_patch",
    [
        {"problem_seed": 999},
        {"data_seed_offset": 1},
    ],
)
def test_add_result_json_complete_rejects_bad_meta(tmp_path: Path, meta_patch: dict):
    from analysis.fitting_time.fitting_time_enn_incremental import (
        enn_incremental_checkpoint_ns,
    )

    chk = enn_incremental_checkpoint_ns()
    expected_data_seed = synthetic_benchmark_data_seed(function_name="sphere", problem_seed=17, rep_index=0)
    dest = tmp_path / "enn_incremental_D2_sphere_pseed17_nrep10_rep0_flat.json"
    meta = {
        "D": 2,
        "function_name": "sphere",
        "problem_seed": int(meta_patch.get("problem_seed", 17)),
        "data_seed": expected_data_seed + int(meta_patch.get("data_seed_offset", 0)),
        "rep_index": 0,
        "num_reps": 10,
        "index_driver": "flat",
    }
    _write_add_json(dest, chk, meta)
    _assert_add_json_incomplete(dest, chk)
